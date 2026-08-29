#!/usr/bin/env python
"""Phase-2 contrastive grammaticality objective: corruptions, scoring, loss.

For each real sentence we build minimally-corrupted negatives via GENERIC linguistic
perturbations (never BLiMP-specific paradigms): adjacent/distant token swaps, function-word
substitution, and random-token replacement -- all LENGTH-PRESERVING, so within a group every
candidate has the same length and total log-likelihoods are directly comparable (BLiMP-style).
The model must assign higher causal log-likelihood to the real sentence than to its
corruptions (softmax-over-candidates cross-entropy, real = positive). This mirrors how
BLiMP/EWoK/COMPS/entity_tracking score the model: a minimal-pair likelihood comparison.

`is_causal` must be set True on the model by the caller before scoring.
"""
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    """Right-padded, <s>-prefixed sentence units from prepare_contrastive_data.py."""

    def __init__(self, path):
        d = torch.load(path)
        self.sent = d["sentences"]        # [N, L] int16
        self.len = d["lengths"]           # [N] int16 (incl. <s>)
        self.pad_id = int(d["pad_id"])
        self.bos_id = int(d["bos_id"])
        self.L = int(d["max_len"])

    def __len__(self):
        return self.sent.shape[0]

    def __getitem__(self, i):
        return self.sent[i].to(torch.long), int(self.len[i])


def function_word_ids(train_tokens_path, n_special, top_k=120):
    """Top-k most frequent (non-special) token ids -- in English these are overwhelmingly
    function words (determiners, prepositions, auxiliaries, pronouns), the right targets for
    agreement / closed-class substitution errors."""
    t = torch.load(train_tokens_path).to(torch.long)
    counts = torch.bincount(t)
    counts[:n_special] = 0
    k = min(top_k, int((counts > 0).sum()))
    return torch.topk(counts, k).indices.tolist()


def corrupt_row(row, length, flist, fset, n_special, vocab_size, crng, kind_weights=None):
    """One length-preserving edit on a single right-padded row. Index 0 (<s>) and the
    padding region [length:] are never touched.

    kind_weights (optional): 4-tuple of relative probabilities over
    (adjacent-swap, distant-swap, function-word-sub, random-replace). Default None = uniform.
    Passing e.g. (0.35, 0.35, 0.25, 0.05) biases toward HARD in-distribution negatives
    (swaps + closed-class subs) and away from the trivially-detectable random-replace,
    yielding more useful preference gradient (DPO --hard_neg)."""
    out = row.clone()
    c0, c1 = 1, length                       # content region [1, length)
    n = c1 - c0
    if n < 2:                                # too short to swap -> substitute
        out[c0] = crng.choice(flist)
        return out
    kind = crng.randrange(4) if kind_weights is None else crng.choices(range(4), weights=kind_weights)[0]
    if kind == 0:                            # swap adjacent
        i = crng.randrange(c0, c1 - 1)
        a, b = int(out[i]), int(out[i + 1])
        out[i], out[i + 1] = b, a
    elif kind == 1:                          # swap distant
        i = crng.randrange(c0, c1)
        j = crng.randrange(c0, c1)
        a, b = int(out[i]), int(out[j])
        out[i], out[j] = b, a
    elif kind == 2:                          # function-word substitution
        cand = [p for p in range(c0, c1) if int(row[p]) in fset]
        if cand:
            p = crng.choice(cand)
            new = crng.choice(flist)
            while new == int(row[p]):
                new = crng.choice(flist)
            out[p] = new
        else:                                # fall back to random replacement
            out[crng.randrange(c0, c1)] = crng.randrange(n_special, vocab_size)
    else:                                    # random-token replacement
        i = crng.randrange(c0, c1)
        new = crng.randrange(n_special, vocab_size)
        while new == int(out[i]):
            new = crng.randrange(n_special, vocab_size)
        out[i] = new
    return out


def build_batch(sent_batch, len_batch, n_neg, flist, fset, n_special, vocab_size, crng, kind_weights=None):
    """Expand a [G, L] batch of real sentences into [G*(1+n_neg), L] of (real, neg, ...).
    All candidates in a group share the real sentence's length (length-preserving edits).
    kind_weights (optional) is forwarded to corrupt_row (see there); default None = uniform."""
    rows, lengths = [], []
    for gi in range(sent_batch.shape[0]):
        row, ln = sent_batch[gi], len_batch[gi]
        rows.append(row.clone())
        lengths.append(ln)
        for _ in range(n_neg):
            rows.append(corrupt_row(row, ln, flist, fset, n_special, vocab_size, crng, kind_weights))
            lengths.append(ln)
    return torch.stack(rows, 0), lengths


def contrastive_loss(model, batch, lengths, n_neg, device, temp=1.0, normalize=False):
    """Causal-LL softmax-over-candidates CE. Caller must have set model.transformer.is_causal=True.
    batch: [M, L] long (M = G*(1+n_neg)); lengths: list[int] (real-token count incl. <s>)."""
    inp = batch.to(device)
    M, L = inp.shape
    logits = model(input_ids=inp, attention_mask=None).logits         # [M, L, V]
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)           # predict positions 1..L-1
    tok_lp = torch.gather(logp, -1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)   # [M, L-1]

    pos = torch.arange(L - 1, device=device).unsqueeze(0)            # predicted token = original idx pos+1
    lens = torch.tensor(lengths, device=device, dtype=torch.long).unsqueeze(1)
    valid = (pos + 1) < lens                                         # content tokens only
    tok_lp = tok_lp * valid
    seq_ll = tok_lp.sum(1)
    if normalize:
        seq_ll = seq_ll / valid.sum(1).clamp(min=1)
    seq_ll = seq_ll.view(-1, 1 + n_neg)                              # [G, 1+n_neg]
    target = torch.zeros(seq_ll.size(0), dtype=torch.long, device=device)  # real = candidate 0
    return F.cross_entropy(seq_ll / temp, target)
