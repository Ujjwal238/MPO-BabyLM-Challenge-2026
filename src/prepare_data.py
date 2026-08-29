#!/usr/bin/env python
"""Tokenize the BabyLM 2026 Strict-Small corpus into packed token streams for training.

v1 packing: every document (line) is prefixed with <s>, all documents are concatenated
into one int16 token stream, and the dataset chunks it into fixed-length segments
(GPT-2 style — no padding waste, which matters because CHILDES/OpenSubtitles lines are short).
A small dev split is held out per-source so validation loss reflects all domains.

Outputs (to data/processed/):
  train_tokens.pt   int16 1-D tensor
  dev_tokens.pt     int16 1-D tensor
  meta.json         token counts, per-source spans, special-token ids
"""
import argparse
import glob
import json
import os

import torch
from transformers import AutoTokenizer

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=f"{REPO}/data/strict_small")
    ap.add_argument("--tokenizer", default=f"{REPO}/artifacts/tokenizer")
    ap.add_argument("--out_dir", default=f"{REPO}/data/processed")
    ap.add_argument("--dev_fraction", type=float, default=0.01)
    ap.add_argument("--batch_lines", type=int, default=20000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bos_id = tok.convert_tokens_to_ids("<s>")
    assert bos_id is not None and bos_id >= 0
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.train.txt")))
    assert files, f"no *.train.txt in {args.data_dir}"

    train_parts, dev_parts, meta_sources = [], [], {}
    train_cursor = 0
    for fp in files:
        name = os.path.basename(fp).replace(".train.txt", "")
        ids = []
        buf = []

        def flush(buf):
            if not buf:
                return
            for enc in tok(buf, add_special_tokens=False)["input_ids"]:
                ids.append(bos_id)
                ids.extend(enc)

        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                buf.append(s)
                if len(buf) >= args.batch_lines:
                    flush(buf)
                    buf = []
            flush(buf)

        t = torch.tensor(ids, dtype=torch.int16)  # vocab 16384 < 32767, fits int16
        n_dev = int(len(t) * args.dev_fraction)
        train_t, dev_t = t[:-n_dev], t[-n_dev:]
        train_parts.append(train_t)
        dev_parts.append(dev_t)
        meta_sources[name] = {"tokens": len(t), "train": len(train_t),
                              "dev": len(dev_t),
                              "train_span": [train_cursor, train_cursor + len(train_t)]}
        train_cursor += len(train_t)
        print(f"  {name:28s} {len(t):>11,} tokens  (dev {len(dev_t):,})")

    train_tokens = torch.cat(train_parts)
    dev_tokens = torch.cat(dev_parts)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(train_tokens, os.path.join(args.out_dir, "train_tokens.pt"))
    torch.save(dev_tokens, os.path.join(args.out_dir, "dev_tokens.pt"))
    meta = {
        "train_tokens": int(train_tokens.numel()),
        "dev_tokens": int(dev_tokens.numel()),
        "vocab_size": tok.vocab_size,
        "bos_id": bos_id,
        "mask_id": tok.convert_tokens_to_ids("<mask>"),
        "pad_id": tok.convert_tokens_to_ids("<pad>"),
        "n_special_tokens": 5,
        "sources": meta_sources,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTRAIN tokens: {train_tokens.numel():,}  | DEV tokens: {dev_tokens.numel():,}")
    print(f"saved to {args.out_dir}")
    print("epochs->tokens:  1 epoch =", f"{train_tokens.numel():,}", " | 10 epochs =", f"{10*train_tokens.numel():,}")


if __name__ == "__main__":
    main()
