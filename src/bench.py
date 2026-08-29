#!/usr/bin/env python
"""Throughput benchmark: efficient (target-only) loss path x precision x batch size on MPS.

The efficient path runs the 16k-vocab projection only on positions that have a target
(huge for masked batches). Reports tokens/sec so we can pick the fastest stable config.
"""
import json
import os
import sys
import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import StreamDataset, build_example
from model_config import build_config, GPTBERTForMaskedLM

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
meta = json.load(open(f"{REPO}/data/processed/meta.json"))
V, MASK, NSP = meta["vocab_size"], meta["mask_id"], meta["n_special_tokens"]
dev = torch.device("mps")
ds = StreamDataset(f"{REPO}/data/processed/train_tokens.pt", 128)
model = GPTBERTForMaskedLM(build_config("small", vocab_size=V)).to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)


def run(precision, batch, seq=128, steps=22, warmup=6):
    dl = iter(DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True))
    if precision == "fp32":
        ctx = nullcontext()
    else:
        ctx = torch.autocast("mps", dtype=(torch.float16 if precision == "fp16" else torch.bfloat16))
    t0, toks, last_loss = None, 0, None
    for i in range(steps):
        try:
            block = next(dl)
        except StopIteration:
            dl = iter(DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)); block = next(dl)
        inp, tgt = build_example(block, "masked", 0.2, MASK, V, NSP)
        inp, tgt = inp.to(dev), tgt.to(dev)
        opt.zero_grad(set_to_none=True)
        with ctx:
            seq_out = model.transformer.get_contextualized_embeddings(inp, None)[0]
            flat = tgt.reshape(-1); keep = flat != -100
            hid = seq_out.reshape(-1, seq_out.size(-1))[keep]
            logits = model.classifier(hid)
            loss = torch.nn.functional.cross_entropy(logits.float(), flat[keep])
        loss.backward(); opt.step()
        last_loss = loss.item()
        if i == warmup:
            torch.mps.synchronize(); t0 = time.time(); toks = 0
        if i >= warmup:
            toks += inp.numel()
    torch.mps.synchronize()
    return toks / (time.time() - t0), last_loss


print(f"{'config':18s} {'tok/s':>9s} {'loss':>7s}   (efficient target-only loss path)")
for prec in ["fp32", "fp16", "bf16"]:
    for batch in [32, 64, 128]:
        try:
            tps, loss = run(prec, batch)
            print(f"{prec+' b='+str(batch):18s} {tps:9.0f} {loss:7.2f}")
        except Exception as e:
            print(f"{prec+' b='+str(batch):18s}   ERROR  {type(e).__name__}: {str(e)[:70]}")
