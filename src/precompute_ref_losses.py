#!/usr/bin/env python
"""Precompute per-token reference NLLs for Selective Language Modeling (SLM / token-level
RHO-LOSS, cf. Rho-1 Lin et al. 2024; Mindermann et al. 2022).

A frozen *reference* model (our own within-budget Phase-1 final — closed system, no external
teacher) scores every training-stream token once, offline:

    ref_nll[p] = -log P_ref(token_p | tokens in the same seq_len block before p)   (causal mode)

The training loop (`train.py --slm`) then computes per-token *excess* loss
(student_loss - ref_nll) and keeps only the top `--slm_keep` fraction of tokens in the CE —
focusing gradient on tokens that are learnable (reference finds them predictable) but not yet
learnt (student still misses them), and dropping unlearnable noise (both find them hard).

Blocks are scored with the SAME chunking as StreamDataset (seq_len+1 blocks, block i covers
stream positions [i*seq_len, i*seq_len+seq_len]), so ref_nll[i*seq_len + t + 1] aligns exactly
with target_ids[:, t] of training block i, for both causal and masked (shifted-target) modes.

Output: a float16 1-D tensor, same length as train_tokens.pt (positions never used as targets
keep NLL=0; they are never read). Progress is tailable in <out>.log.

  python src/precompute_ref_losses.py \
      --ref checkpoints/gptbert_small_v1/final \
      --out data/processed/ref_nll_final.pt
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import StreamDataset  # noqa: E402
from gpt_bert.configuration_gpt_bert import ModelConfig  # noqa: E402
from gpt_bert.modeling_gpt_bert import GPTBERTForCausalLM  # noqa: E402

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=f"{REPO}/checkpoints/gptbert_small_v1/final",
                   help="reference checkpoint dir (own within-budget model)")
    p.add_argument("--data_dir", default=f"{REPO}/data/processed")
    p.add_argument("--out", default=f"{REPO}/data/processed/ref_nll_final.pt")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit_blocks", type=int, default=0, help="score only the first N blocks (smoke)")
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()

    device = get_device()
    logf = open(args.out + ".log", "a", buffering=1)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    cfg = ModelConfig(os.path.join(args.ref, "config.json"))
    model = GPTBERTForCausalLM(cfg)
    state = torch.load(os.path.join(args.ref, "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not [k for k in missing if "position_indices" not in k], f"missing keys: {missing}"
    model.to(device).eval()

    ds = StreamDataset(os.path.join(args.data_dir, "train_tokens.pt"), args.seq_len)
    n_blocks = len(ds) if not args.limit_blocks else min(args.limit_blocks, len(ds))
    ref_nll = torch.zeros(ds.tokens.numel(), dtype=torch.float16)

    log(f"=== ref-NLL precompute | ref={args.ref} | device={device} ===")
    log(f"blocks={n_blocks:,} (seq_len={args.seq_len}, batch={args.batch_size}) "
        f"| stream={ds.tokens.numel():,} tokens | out={args.out}")

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_blocks, args.batch_size):
            idx = list(range(start, min(start + args.batch_size, n_blocks)))
            block = torch.stack([ds[i] for i in idx])                       # B x (L+1)
            inp = block[:, :-1].to(device)
            tgt = block[:, 1:].to(device)
            logits = model(input_ids=inp, attention_mask=None).logits       # B x L x V
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  tgt.reshape(-1), reduction="none").view(tgt.shape)
            nll = nll.to("cpu", dtype=torch.float16)
            for j, i in enumerate(idx):
                s = i * args.seq_len
                ref_nll[s + 1 : s + 1 + args.seq_len] = nll[j]
            done = start + len(idx)
            if (start // args.batch_size) % args.log_every == 0 or done >= n_blocks:
                el = time.time() - t0
                eta = el / max(1, done) * (n_blocks - done)
                frac = done / n_blocks
                bar = "#" * int(24 * frac) + "-" * (24 - int(24 * frac))
                log(f"[{bar}] {100*frac:5.1f}% | block {done:,}/{n_blocks:,} "
                    f"| mean NLL {nll.float().mean():.3f} | {el:6.0f}s | ETA {eta:6.0f}s")

    torch.save(ref_nll, args.out)
    scored = ref_nll[ref_nll > 0]
    log(f"DONE in {time.time()-t0:.0f}s | saved {args.out} "
        f"| scored {scored.numel():,} positions | mean {scored.float().mean():.4f} "
        f"| p10/p50/p90 = {[round(float(torch.quantile(scored.float()[:2_000_000], q)), 3) for q in (0.1, 0.5, 0.9)]}")

    # per-source diagnostic (which corpora carry the noisiest tokens)
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    for src, info in meta.get("sources", {}).items():
        a, b = info["train_span"]
        seg = ref_nll[a:b]
        seg = seg[seg > 0]
        if seg.numel():
            log(f"  source {src:15s}: mean ref NLL {seg.float().mean():.3f} over {seg.numel():,} tokens")
    logf.close()


if __name__ == "__main__":
    main()
