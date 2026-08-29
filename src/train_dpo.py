#!/usr/bin/env python
"""MP-DPO: minimal-pair Direct Preference Optimization post-training (lever 9).

The user's premise (correct, per the 2026-07-02 audit): Phase-1 saturated at chck_70M
(fast BLiMP 70.13); the last ~30M words of MLE bought nothing. This phase spends a small
slice of that leftover lineage budget on a *different* signal, starting FROM the best
checkpoint instead of competing with MLE from scratch (the failure mode of lever 1).

Why DPO and not literal GRPO: we have no non-hackable closed-system reward for rollouts,
generation-quality objectives mismatch our likelihood-scored eval (BLiMP/EWoK/COMPS/entity
are LL comparisons), and BabyLM-2025 evidence (PPO "mixed to adversarial") is negative for
zero-shot suites. DPO is the RLHF objective in closed form (Rafailov et al. 2023) applied
to preferences we can construct compliantly: real corpus sentences (preferred) vs their
length-preserving corruptions (dispreferred) — reusing lever-1's generic corruption
machinery (contrastive.py; NEVER BLiMP-specific paradigms). The KL anchor to the frozen
reference (chck_70M itself) plus a small MLE-retain mix are the two stability guards the
lever-1 collapse taught us to require.

Safety rails:
  - policy initialized from --init; reference = frozen copy of --init (never modified)
  - reads existing checkpoints READ-ONLY; writes only to <out_dir>/<run_name>/
  - checkpoints every --ckpt_every steps -> canary eval via scripts/dpo_canary.sh
    (abort rule: wh_vs_that_with_gap or distractor_agreement drop >10 abs vs chck_70M)
  - budget: logs cumulative words-seen (pairs count real+corrupted); default run ~1M words,
    minuscule vs the ~30M-word lineage headroom (7 MLE epochs + this phase << 10-epoch cap)

  PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_dpo.py \
      --init checkpoints/hub/baseline_chck_70M --run_name mpo_repro
"""
import argparse
import datetime
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contrastive as contr                              # noqa: E402
from dataset import StreamDataset, build_example        # noqa: E402
from model_config import save_hf_checkpoint             # noqa: E402
from gpt_bert.configuration_gpt_bert import ModelConfig  # noqa: E402
from gpt_bert.modeling_gpt_bert import GPTBERTForMaskedLM  # noqa: E402

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_init():
    """The saturation checkpoint, from whichever workflow produced it.

    A user who trained the baseline has it under checkpoints/<run_name>/chck_70M; a user
    who ran src/fetch_checkpoints.py has it under checkpoints/hub/baseline_chck_70M.
    Prefer a locally trained one, fall back to a fetched one, and if neither exists return
    the local path so the error message names the expected location.
    """
    local = os.path.join(REPO, "checkpoints", "gptbert_small_v1", "chck_70M")
    fetched = os.path.join(REPO, "checkpoints", "hub", "baseline_chck_70M")
    return local if os.path.isdir(local) else (fetched if os.path.isdir(fetched) else local)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hms(s):
    return str(datetime.timedelta(seconds=int(max(0, s))))


def load_model(ckpt_dir, device):
    cfg = ModelConfig(os.path.join(ckpt_dir, "config.json"))
    m = GPTBERTForMaskedLM(cfg)
    state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not [k for k in missing if "position_indices" not in k], f"missing: {missing}"
    return m.to(device)


def seq_logliks(model, batch, lengths, device):
    """Per-sequence causal LLs (sum over content tokens), same math as contrastive_loss."""
    inp = batch.to(device)
    M, L = inp.shape
    logits = model(input_ids=inp, attention_mask=None).logits
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tok_lp = torch.gather(logp, -1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)
    pos = torch.arange(L - 1, device=device).unsqueeze(0)
    lens = torch.tensor(lengths, device=device, dtype=torch.long).unsqueeze(1)
    valid = (pos + 1) < lens
    return (tok_lp * valid).sum(1)                        # [M]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=default_init())
    p.add_argument("--ref", default=None, help="reference checkpoint (default: same as --init)")
    p.add_argument("--data_dir", default=f"{REPO}/data/processed")
    p.add_argument("--c_data", default=f"{REPO}/data/processed/contrastive_sentences.pt")
    p.add_argument("--tokenizer", default=f"{REPO}/artifacts/tokenizer")
    p.add_argument("--out_dir", default=f"{REPO}/checkpoints")
    p.add_argument("--run_name", default="gptbert_small_dpo_v1")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--pairs", type=int, default=16, help="preference pairs per step")
    p.add_argument("--beta", type=float, default=0.1, help="DPO inverse-temperature (KL strength)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mle_mix", type=float, default=0.5,
                   help="probability of adding a retain-MLE micro-batch each step")
    p.add_argument("--mle_weight", type=float, default=1.0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--mle_batch", type=int, default=16)
    p.add_argument("--p_masked", type=float, default=15.0 / 16.0)
    p.add_argument("--mask_p", type=float, default=0.15, help="fixed (post-anneal) mask rate")
    p.add_argument("--c_func_k", type=int, default=120)
    p.add_argument("--hard_neg", action="store_true",
                   help="bias corruptions toward HARD in-distribution negatives "
                        "(swaps + function-word subs, down-weight random-replace)")
    p.add_argument("--hard_neg_weights", default="0.35,0.35,0.25,0.05",
                   help="4 weights over (adj-swap, distant-swap, func-sub, random-replace)")
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    if args.smoke:
        args.steps, args.ckpt_every = 20, 10

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "train.log"), "a", buffering=1)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    import json
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab_size, n_special = meta["vocab_size"], meta["n_special_tokens"]
    mask_id = meta["mask_id"]
    words_per_token = 10_000_000 / (meta["train_tokens"] + meta["dev_tokens"])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    policy = load_model(args.init, device)
    ref = load_model(args.ref or args.init, device).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    log(f"=== MP-DPO '{args.run_name}' | device={device} ===")
    log(f"init={args.init}")
    log(f"ref ={args.ref or args.init} (frozen)")
    log(f"steps={args.steps} pairs/step={args.pairs} beta={args.beta} lr={args.lr} "
        f"mle_mix={args.mle_mix}@{args.mle_weight} | ckpt every {args.ckpt_every}")

    sent_ds = contr.SentenceDataset(args.c_data)
    sent_loader = DataLoader(sent_ds, batch_size=args.pairs, shuffle=True, drop_last=True,
                             num_workers=0, generator=torch.Generator().manual_seed(args.seed + 1))

    def _inf(loader):
        while True:
            for b in loader:
                yield b
    sent_iter = _inf(sent_loader)
    flist = contr.function_word_ids(os.path.join(args.data_dir, "train_tokens.pt"),
                                    n_special, args.c_func_k)
    fset = set(flist)
    import random as _random
    crng = _random.Random(args.seed)
    kind_weights = None
    if args.hard_neg:
        kind_weights = [float(x) for x in args.hard_neg_weights.split(",")]
        assert len(kind_weights) == 4, "hard_neg_weights must be 4 comma-sep numbers"
        log(f"HARD NEGATIVES on | corruption weights (adj,dist,func,rand) = {kind_weights}")

    stream_ds = StreamDataset(os.path.join(args.data_dir, "train_tokens.pt"), args.seq_len)
    stream_loader = DataLoader(stream_ds, batch_size=args.mle_batch, shuffle=True, drop_last=True,
                               num_workers=0, generator=torch.Generator().manual_seed(args.seed + 2))
    stream_iter = _inf(stream_loader)
    rng = torch.Generator().manual_seed(args.seed)

    decay, no_decay = [], []
    for n, prm in policy.named_parameters():
        (no_decay if (prm.ndim < 2 or n.endswith("bias")) else decay).append(prm)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(args.steps * args.warmup_ratio)), args.steps)

    t0 = time.time()
    words_seen, acc_sum, acc_n = 0.0, 0.0, 0
    policy.train()
    for step in range(1, args.steps + 1):
        sb, lb = next(sent_iter)
        batch, lengths = contr.build_batch(sb, lb, 1, flist, fset, n_special, vocab_size, crng, kind_weights)
        policy.transformer.is_causal = True
        ref.transformer.is_causal = True
        ll_pol = seq_logliks(policy, batch, lengths, device).view(-1, 2)     # [G, (real, neg)]
        with torch.no_grad():
            ll_ref = seq_logliks(ref, batch, lengths, device).view(-1, 2)
        margin = (ll_pol[:, 0] - ll_ref[:, 0]) - (ll_pol[:, 1] - ll_ref[:, 1])
        dpo = -F.logsigmoid(args.beta * margin).mean()
        loss = dpo
        mle_val = float("nan")
        if torch.rand(1, generator=rng).item() < args.mle_mix:
            block = next(stream_iter)
            mode = "masked" if torch.rand(1, generator=rng).item() < args.p_masked else "causal"
            inp, tgt = build_example(block, mode, args.mask_p, mask_id, vocab_size, n_special,
                                     generator=rng)
            policy.transformer.is_causal = (mode == "causal")
            lg = policy(input_ids=inp.to(device), attention_mask=None).logits
            mle = F.cross_entropy(lg.reshape(-1, vocab_size), tgt.to(device).reshape(-1),
                                  ignore_index=-100)
            loss = loss + args.mle_weight * mle
            mle_val = mle.item()
            words_seen += inp.numel() * words_per_token

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        words_seen += sum(lengths) * words_per_token       # real + corrupted candidates
        acc_sum += (margin > 0).float().mean().item()
        acc_n += 1

        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            log(f"[{hms(el)}] step {step:>5}/{args.steps} | dpo {dpo.item():.4f} "
                f"| pref-acc {acc_sum/max(1,acc_n):.3f} | mle {mle_val:6.3f} "
                f"| margin {margin.mean().item():+.2f} | lr {sched.get_last_lr()[0]:.2e} "
                f"| {words_seen/1e6:.3f}M words | ETA {hms(el/step*(args.steps-step))}")
            acc_sum, acc_n = 0.0, 0

        if step % args.ckpt_every == 0:
            name = f"chck_dpo_{step}"
            save_hf_checkpoint(policy, os.path.join(run_dir, name), tokenizer=tokenizer)
            log(f"  >> saved {name}")

    save_hf_checkpoint(policy, os.path.join(run_dir, "final"), tokenizer=tokenizer)
    log(f"DONE: {args.steps} steps | {words_seen/1e6:.3f}M words seen (phase) | {hms(time.time()-t0)} "
        f"| saved {run_dir}/final")
    logf.close()


if __name__ == "__main__":
    main()
