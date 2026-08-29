#!/usr/bin/env python
"""Build sentence-level units for the Phase-2 contrastive grammaticality objective.

Reads the raw Strict-Small corpus (line-oriented), produces clean sentence-sized units
(speaker tags stripped, wiki markup dropped, long prose sentence-split), tokenizes each
with a <s> prefix (matching training), filters by length, dedups, and saves a padded
int16 tensor + lengths to data/processed/contrastive_sentences.pt.

These units are perturbed into negatives at train time (see contrastive.py). They are
derived from the SAME 10M-word corpus -> no new data: the contrastive term is an auxiliary
objective over corpus-derived units (BERT/ELECTRA-style multi-objective), and only a small
fraction is sampled during the 10 epochs (<< 1 extra epoch of corpus exposure).

  python src/prepare_contrastive_data.py
"""
import argparse
import glob
import os
import re

import torch
from transformers import AutoTokenizer

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPEAKER = re.compile(r'^\s*[\*%@][A-Za-z0-9_]{1,8}:\s*')   # CHILDES *CHI:/%mor:/@ tiers
SWB = re.compile(r'^\s*[A-B][0-9]?:\s*')                   # switchboard A:/B:
DASH = re.compile(r'^\s*-\s*')                             # subtitle dialogue dash
WIKI_HEAD = re.compile(r'^\s*=+.*=+\s*$')                  # = = = Title = = =
MULTISPACE = re.compile(r'\s+')


def clean_line(line, source):
    s = line.rstrip('\n')
    if source == 'childes':
        ls = s.lstrip()
        if ls.startswith('%') or ls.startswith('@'):
            return None                       # annotation tier, not text
        s = SPEAKER.sub('', s)
    elif source in ('switchboard', 'bnc_spoken'):
        s = SPEAKER.sub('', s)
        s = SWB.sub('', s)
    elif source == 'open_subtitles':
        s = DASH.sub('', s)
    elif source == 'simple_wiki':
        if WIKI_HEAD.match(s):
            return None
    s = MULTISPACE.sub(' ', s).strip()
    return s or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=f"{REPO}/data/strict_small")
    ap.add_argument("--tokenizer", default=f"{REPO}/artifacts/tokenizer")
    ap.add_argument("--out", default=f"{REPO}/data/processed/contrastive_sentences.pt")
    ap.add_argument("--min_len", type=int, default=4)     # content tokens (excl <s>)
    ap.add_argument("--max_len", type=int, default=48)    # content tokens (excl <s>)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bos_id = tok.convert_tokens_to_ids("<s>")
    pad_id = tok.convert_tokens_to_ids("<pad>")
    from nltk.tokenize import sent_tokenize

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.train.txt")))
    assert files, f"no *.train.txt in {args.data_dir}"

    all_cands = []
    for fp in files:
        source = os.path.basename(fp).replace(".train.txt", "")
        cands = []
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = clean_line(line, source)
                if not s:
                    continue
                if source in ("simple_wiki", "gutenberg"):
                    cands.extend(sent_tokenize(s))   # split prose into sentences
                else:
                    cands.append(s)                  # spoken: one utterance per line
        print(f"  {source:16s} {len(cands):>10,} candidate strings", flush=True)
        all_cands.extend(cands)

    print(f"tokenizing {len(all_cands):,} candidates...", flush=True)
    units, seen = [], set()
    B = 20000
    for k in range(0, len(all_cands), B):
        for cids in tok(all_cands[k:k + B], add_special_tokens=False)["input_ids"]:
            if not (args.min_len <= len(cids) <= args.max_len):
                continue
            key = tuple(cids)
            if key in seen:
                continue
            seen.add(key)
            units.append(cids)

    L = args.max_len + 1                              # +1 for <s>
    N = len(units)
    arr = torch.full((N, L), pad_id, dtype=torch.int16)
    lengths = torch.zeros(N, dtype=torch.int16)
    for i, cids in enumerate(units):
        row = [bos_id] + cids
        arr[i, :len(row)] = torch.tensor(row, dtype=torch.int16)
        lengths[i] = len(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"sentences": arr, "lengths": lengths, "bos_id": bos_id,
                "pad_id": pad_id, "max_len": L}, args.out)
    print(f"\nsaved {N:,} unique sentence units -> {args.out}  (padded shape {tuple(arr.shape)})")
    print(f"mean content length: {(lengths.float().mean()-1):.1f} tokens")


if __name__ == "__main__":
    main()
