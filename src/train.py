#!/usr/bin/env python
"""Train a GPT-BERT hybrid (masked + causal) LM on the BabyLM Strict-Small corpus on MPS.

Design (see dataset.py / model_config.py):
  - ONE model; toggle `transformer.is_causal` per batch to switch objective.
  - Hybrid mix: each micro-batch masked w.p. `p_masked` (default 15/16) else causal.
  - Shifted (next-position) targets for both -> single consistent LM head.
  - AdamW + cosine warmup; <=10 epochs (BabyLM 2026 cap).
  - Word-milestone checkpoints (chck_1M..10M, 20M..100M) as self-contained HF models.

Live progress is written to  <out_dir>/<run_name>/train.log  (tail -f to watch):
  progress bar, elapsed, step, epoch, loss, lr, words-seen, tok/s, ETA.

  python src/train.py --config small --max_epochs 10 --run_name gptbert_small_v1
  # watch:  tail -f checkpoints/gptbert_small_v1/train.log
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import StreamDataset, build_example          # noqa: E402
from model_config import build_config, save_hf_checkpoint, count_params, GPTBERTForMaskedLM  # noqa: E402
import contrastive as contr                                # noqa: E402  (Phase-2 contrastive objective)
import distill as distill_mod                               # noqa: E402  (Phase-2 self-distillation)

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hms(seconds):
    return str(datetime.timedelta(seconds=int(max(0, seconds))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="small", choices=["small", "base"])
    p.add_argument("--data_dir", default=f"{REPO}/data/processed")
    p.add_argument("--tokenizer", default=f"{REPO}/artifacts/tokenizer")
    p.add_argument("--out_dir", default=f"{REPO}/checkpoints")
    p.add_argument("--run_name", default="gptbert_small_v1")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=10)            # BabyLM cap
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--p_masked", type=float, default=15.0 / 16.0)   # hybrid ratio
    p.add_argument("--mask_p_start", type=float, default=0.30)
    p.add_argument("--mask_p_end", type=float, default=0.15)
    p.add_argument("--span_masking", action="store_true", help="SpanBERT-style span masking (recipe v2)")
    p.add_argument("--muon", action="store_true", help="Muon optimizer for 2D hidden matrices (AdamW for the rest)")
    p.add_argument("--muon_lr", type=float, default=0.02, help="Muon learning rate")
    p.add_argument("--adaptive_mask", action="store_true", help="adaptive per-token-id masking (Edman & Fraser 2025)")
    p.add_argument("--adaptive_period", type=int, default=200, help="optimizer steps between adaptive mask-prob updates")
    p.add_argument("--adaptive_lambda", type=float, default=0.2, help="EMA weight on previous mask probs")
    p.add_argument("--slm", action="store_true",
                   help="Selective Language Modeling (token-level RHO-LOSS / Rho-1): CE only on the "
                        "top --slm_keep fraction of tokens by excess loss (student - reference)")
    p.add_argument("--slm_ref", default=f"{REPO}/data/processed/ref_nll_final.pt",
                   help="per-position reference NLLs from precompute_ref_losses.py (same seq_len!)")
    p.add_argument("--slm_keep", type=float, default=0.6, help="fraction of loss tokens kept per micro-batch")
    p.add_argument("--distill", action="store_true", help="self-distillation from an ensemble of teachers")
    p.add_argument("--distill_teachers", default=None, help="comma-sep teacher checkpoint dirs (default: the 3 phase models)")
    p.add_argument("--distill_alpha", type=float, default=0.5, help="weight on hard-label LM loss (1-alpha on KD)")
    p.add_argument("--distill_temp", type=float, default=2.0, help="distillation temperature")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    # Phase-2 contrastive grammaticality objective (default off -> identical to Phase-1)
    p.add_argument("--contrastive", action="store_true", help="add contrastive grammaticality loss")
    p.add_argument("--c_data", default=f"{REPO}/data/processed/contrastive_sentences.pt")
    p.add_argument("--cw", type=float, default=0.5, help="contrastive loss weight (lambda)")
    p.add_argument("--c_bs", type=int, default=16, help="sentences per contrastive step")
    p.add_argument("--n_neg", type=int, default=2, help="negatives per real sentence")
    p.add_argument("--c_temp", type=float, default=1.0, help="softmax temperature over candidate LLs")
    p.add_argument("--c_func_k", type=int, default=120, help="top-k frequent tokens treated as function words")
    p.add_argument("--c_norm", action="store_true", help="length-normalize sequence LL (default: total)")
    # Resume model weights from a checkpoint (e.g. after an OOM kill). NOTE: optimizer/scheduler
    # state was not checkpointed -> fresh AdamW + LR schedule fast-forwarded to the resumed step.
    p.add_argument("--resume_from", default=None, help="checkpoint dir to load model weights from")
    p.add_argument("--resume_words", type=float, default=0.0, help="words (millions) already seen at --resume_from")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    total_tokens = meta["train_tokens"] + meta["dev_tokens"]
    words_per_token = 10_000_000 / total_tokens          # corpus is 10M words
    vocab_size = meta["vocab_size"]
    mask_id, n_special = meta["mask_id"], meta["n_special_tokens"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_ds = StreamDataset(os.path.join(args.data_dir, "train_tokens.pt"), args.seq_len,
                             return_idx=args.slm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, generator=torch.Generator().manual_seed(args.seed))

    cfg = build_config(args.config, vocab_size=vocab_size, max_position_embeddings=max(512, args.seq_len))
    model = GPTBERTForMaskedLM(cfg).to(device)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    max_steps = steps_per_epoch * args.max_epochs
    if args.smoke:
        max_steps = 30
    warmup_steps = max(1, int(max_steps * args.warmup_ratio))

    if args.muon:
        from muon import Muon
        muon_params, adam_decay, adam_nodecay = [], [], []
        for n, prm in model.named_parameters():
            if prm.ndim == 2 and "embedding" not in n:      # hidden matrices -> Muon
                muon_params.append(prm)
            elif prm.ndim < 2 or n.endswith("bias"):
                adam_nodecay.append(prm)
            else:
                adam_decay.append(prm)                      # embeddings (word/relative)
        adam_opt = torch.optim.AdamW(
            [{"params": adam_decay, "weight_decay": args.weight_decay},
             {"params": adam_nodecay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.98), eps=1e-8,
        )
        optimizers = [Muon(muon_params, lr=args.muon_lr, momentum=0.95, nesterov=True), adam_opt]
        log_opt = f"MUON on {len(muon_params)} matrices (lr={args.muon_lr}) + AdamW on {len(adam_decay)+len(adam_nodecay)} tensors (lr={args.lr})"
    else:
        decay, no_decay = [], []
        for n, prm in model.named_parameters():
            (no_decay if (prm.ndim < 2 or n.endswith("bias")) else decay).append(prm)
        optimizers = [torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.98), eps=1e-8,
        )]
        log_opt = f"AdamW (lr={args.lr})"
    schedulers = [get_cosine_schedule_with_warmup(o, warmup_steps, max_steps) for o in optimizers]

    milestones = [i * 1_000_000 for i in range(1, 11)] + [i * 1_000_000 for i in range(20, 101, 10)]
    next_ms = 0
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")
    logf = open(log_path, "a", buffering=1)  # line-buffered -> live tail -f

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    log(f"=== run '{args.run_name}' | device={device} | config={args.config} "
        f"({count_params(model)/1e6:.1f}M params) ===")
    log(f"seq_len={args.seq_len} batch={args.batch_size} grad_accum={args.grad_accum} "
        f"(eff batch {args.batch_size*args.grad_accum}) | lr={args.lr} | epochs={args.max_epochs}")
    log(f"blocks/epoch={len(train_ds):,} | steps/epoch={steps_per_epoch:,} | max_steps={max_steps:,} "
        f"| ~{10*total_tokens/1e6:.0f}M tokens over {args.max_epochs} epochs")
    log(f"live log: {log_path}")
    log(f"optimizer: {log_opt}")

    # --- Phase-2 contrastive grammaticality objective ---
    contrastive_on = args.contrastive
    last_closs, c_steps, c_tokens = float("nan"), 0, 0
    if contrastive_on:
        sent_ds = contr.SentenceDataset(args.c_data)
        sent_loader = DataLoader(sent_ds, batch_size=args.c_bs, shuffle=True, drop_last=True,
                                 num_workers=0, generator=torch.Generator().manual_seed(args.seed + 1))

        def _infinite(loader):
            while True:
                for b in loader:
                    yield b
        sent_iter = _infinite(sent_loader)
        flist = contr.function_word_ids(os.path.join(args.data_dir, "train_tokens.pt"), n_special, args.c_func_k)
        fset = set(flist)
        crng = random.Random(args.seed)
        log(f"CONTRASTIVE ON | {len(sent_ds):,} sentences | cw={args.cw} n_neg={args.n_neg} "
            f"c_bs={args.c_bs} temp={args.c_temp} norm={args.c_norm} | {len(flist)} function-word ids")

    # --- adaptive per-token masking (Edman & Fraser 2025, "Mask and You Shall Receive") ---
    adaptive_on = args.adaptive_mask
    adapt_w = None
    if adaptive_on:
        corpus_freq = torch.bincount(train_ds.tokens.to(torch.long), minlength=vocab_size).float()
        corpus_freq[:n_special] = 0.0
        adapt_w = torch.full((vocab_size,), args.mask_p_start, dtype=torch.float32)   # CPU per-token mask prob
        adapt_w[:n_special] = 0.0
        correct_counts = torch.zeros(vocab_size, device=device)
        total_counts = torch.zeros(vocab_size, device=device)
        log(f"ADAPTIVE MASKING ON | period={args.adaptive_period} lambda={args.adaptive_lambda} "
            f"| mask rate {args.mask_p_start}->{args.mask_p_end} (renormalized per-token)")

    # --- self-distillation from an ensemble of same-budget teachers (BabyLlama-2) ---
    distill_on = args.distill
    teachers, last_kd = None, float("nan")
    if distill_on:
        if args.distill_teachers:
            tpaths = args.distill_teachers.split(",")
        else:
            tpaths = [os.path.join(args.out_dir, "gptbert_small_v1", "final"),
                      os.path.join(args.out_dir, "gptbert_small_span_v2", "final"),
                      os.path.join(args.out_dir, "gptbert_small_muon_v1", "final")]
        teachers = distill_mod.load_teachers(tpaths, device)
        log(f"SELF-DISTILL ON | {len(teachers)} teachers | alpha={args.distill_alpha} temp={args.distill_temp}")
        for tp in tpaths:
            log(f"  teacher: {tp}")

    # --- Selective Language Modeling (token-level RHO-LOSS / Rho-1) ---
    slm_on = args.slm
    slm_ref = None
    slm_all_loss = None            # last micro-batch mean loss over ALL valid tokens (for logging)
    if slm_on:
        slm_ref = torch.load(args.slm_ref)                       # float16 CPU, ~32MB
        assert slm_ref.numel() == train_ds.tokens.numel(), \
            f"ref NLL length {slm_ref.numel()} != stream {train_ds.tokens.numel()} (re-run precompute?)"
        slm_pos_off = torch.arange(1, args.seq_len + 1).unsqueeze(0)     # target stream offsets within a block
        log(f"SLM ON | ref={args.slm_ref} | keep top {args.slm_keep:.0%} of tokens by excess loss "
            f"(mean scored ref NLL {slm_ref[slm_ref > 0].float().mean():.3f})")

    rng = torch.Generator().manual_seed(args.seed)
    global_step, tokens_done = 0, 0
    t0 = time.time()
    for o in optimizers:
        o.zero_grad(set_to_none=True)
    model.train()
    done = False

    if args.resume_from:
        state = torch.load(os.path.join(args.resume_from, "pytorch_model.bin"), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        tokens_done = int(args.resume_words * 1e6 / words_per_token)
        global_step = round(tokens_done / (args.batch_size * args.seq_len * args.grad_accum))
        while next_ms < len(milestones) and milestones[next_ms] <= args.resume_words * 1e6:
            next_ms += 1
        for _ in range(global_step):
            for s in schedulers:
                s.step()
        log(f"RESUMED from {args.resume_from} | weights loaded (missing={len(missing)}, unexpected={len(unexpected)}) "
            f"| start step {global_step}/{max_steps} | {args.resume_words:.0f}M words seen "
            f"| next milestone idx {next_ms} | lr now {schedulers[0].get_last_lr()[0]:.2e}")
        if missing or unexpected:
            log(f"  WARN resume key mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")

    for epoch in range(args.max_epochs):
        if done:
            break
        for micro, block in enumerate(train_loader):
            if slm_on:
                block, blk_idx = block
            mode = "masked" if torch.rand(1, generator=rng).item() < args.p_masked else "causal"
            mask_p = args.mask_p_start + (args.mask_p_end - args.mask_p_start) * min(1.0, global_step / max_steps)
            input_ids, target_ids = build_example(block, mode, mask_p, mask_id, vocab_size, n_special,
                                                   generator=rng, span=args.span_masking, token_probs=adapt_w)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            model.transformer.is_causal = (mode == "causal")
            logits = model(input_ids=input_ids, attention_mask=None).logits
            if slm_on:
                tok_loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1),
                                           ignore_index=-100, reduction="none")
                pos = blk_idx.unsqueeze(1) * args.seq_len + slm_pos_off        # B x L target stream positions
                ref = slm_ref[pos].to(device=device, dtype=tok_loss.dtype).reshape(-1)
                valid = target_ids.reshape(-1) != -100
                excess = (tok_loss - ref)[valid]
                k = max(1, int(args.slm_keep * excess.numel()))
                kept = tok_loss[valid][torch.topk(excess, k).indices]
                loss = kept.mean()
                slm_all_loss = tok_loss[valid].mean()          # lazy tensor; .item() only at log time
            else:
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1), ignore_index=-100)
            if distill_on:
                ens = distill_mod.ensemble_logits(teachers, input_ids, model.transformer.is_causal)
                kd = distill_mod.kd_loss(logits, ens, target_ids, args.distill_temp)
                last_kd = kd.item()
                loss = args.distill_alpha * loss + (1.0 - args.distill_alpha) * kd
            (loss / args.grad_accum).backward()
            tokens_done += input_ids.numel()

            if adaptive_on and mode == "masked":
                with torch.no_grad():
                    m = target_ids != -100
                    if m.any():
                        tids = target_ids[m]
                        correct_counts.index_add_(0, tids, (logits.argmax(-1)[m] == tids).float())
                        total_counts.index_add_(0, tids, torch.ones_like(tids, dtype=torch.float32))

            if (micro + 1) % args.grad_accum == 0:
                if contrastive_on:
                    sb, lb = next(sent_iter)
                    cbatch, clens = contr.build_batch(sb, lb, args.n_neg, flist, fset, n_special, vocab_size, crng)
                    model.transformer.is_causal = True
                    closs = contr.contrastive_loss(model, cbatch, clens, args.n_neg, device,
                                                   temp=args.c_temp, normalize=args.c_norm)
                    (args.cw * closs).backward()
                    last_closs = closs.item()
                    c_steps += 1
                    c_tokens += int(lb.sum())
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                for o in optimizers:
                    o.step()
                for s in schedulers:
                    s.step()
                for o in optimizers:
                    o.zero_grad(set_to_none=True)
                global_step += 1

                if adaptive_on and global_step % args.adaptive_period == 0:
                    with torch.no_grad():
                        score = (correct_counts.cpu() + 0.5) / (total_counts.cpu() + 1.0)   # smoothed accuracy
                        w_tilde = mask_p * (1.0 - score)                                     # mask hard tokens more
                        adapt_w.mul_(args.adaptive_lambda).add_(w_tilde, alpha=1.0 - args.adaptive_lambda)
                        adapt_w[:n_special] = 0.0
                        fw_mean = (corpus_freq * adapt_w).sum() / corpus_freq.sum()          # freq-weighted mean rate
                        if fw_mean > 1e-8:
                            adapt_w.mul_(mask_p / fw_mean)                                    # renormalize to target rate
                        adapt_w.clamp_(0.0, 0.95)
                        correct_counts.zero_(); total_counts.zero_()
                        log(f"  adaptive-mask @{global_step}: w[min={adapt_w.min():.3f} max={adapt_w.max():.3f}] target {mask_p:.3f}")

                if global_step % args.log_every == 0 or global_step == 1:
                    elapsed = time.time() - t0
                    frac = global_step / max_steps
                    rate = tokens_done / max(1e-9, elapsed)
                    eta = (max_steps - global_step) * (elapsed / max(1, global_step))
                    words = tokens_done * words_per_token
                    filled = int(24 * frac)
                    bar = "#" * filled + "-" * (24 - filled)
                    closs_str = f" | closs {last_closs:6.3f}" if contrastive_on else ""
                    kd_str = f" | kd {last_kd:6.3f}" if distill_on else ""
                    slm_str = f" | all {slm_all_loss.item():6.3f}" if slm_on and slm_all_loss is not None else ""
                    log(f"[{hms(elapsed)}] [{bar}] {100*frac:5.1f}% | step {global_step:>6}/{max_steps} "
                        f"| ep {epoch} | {mode:6s} | loss {loss.item():6.3f}{slm_str}{closs_str}{kd_str} | lr {schedulers[0].get_last_lr()[0]:.2e} "
                        f"| {words/1e6:6.2f}M words | {rate:5.0f} tok/s | ETA {hms(eta)}")

                words_seen = tokens_done * words_per_token
                while next_ms < len(milestones) and words_seen >= milestones[next_ms]:
                    name = f"chck_{milestones[next_ms]//1_000_000}M"
                    save_hf_checkpoint(model, os.path.join(run_dir, name), tokenizer=tokenizer)
                    log(f"  >> saved checkpoint {name} at {words_seen/1e6:.2f}M words")
                    next_ms += 1

                if global_step >= max_steps:
                    done = True
                    break

    save_hf_checkpoint(model, os.path.join(run_dir, "final"), tokenizer=tokenizer)
    if contrastive_on:
        log(f"contrastive: {c_steps} steps | ~{c_tokens*words_per_token/1e6:.2f}M auxiliary word-touches "
            f"(corpus-derived sentences; << 1 extra epoch of exposure)")
    log(f"DONE: {global_step} steps | {tokens_done*words_per_token/1e6:.2f}M words seen "
        f"| {hms(time.time()-t0)} elapsed | saved {run_dir}/final")
    logf.close()


if __name__ == "__main__":
    main()
