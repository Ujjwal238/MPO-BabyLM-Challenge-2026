#!/usr/bin/env python
"""Dataset + masking for GPT-BERT-style hybrid training on a packed token stream.

The corpus is a single 1-D token stream (documents joined with <s>). We chunk it into
(seq_len+1)-token blocks (GPT-2 style packing, no padding). The training loop turns each
block into either a CAUSAL or a MASKED example using the *shifted* (next-position) framing
that GPT-BERT uses for both objectives, so the single LM head stays consistent:

  causal:  input = block[:, :-1],  target = block[:, 1:]                 (predict next token)
  masked:  input = masked(block)[:, :-1],
           target = (block where masked else -100)[:, 1:]                (predict next, MLM-style)

v1 uses token-level masking (annealed rate, 80/10/10). Span masking + count-balancing
(from the LTG recipe) are a Phase-2 refinement.
"""
import math

import torch
from torch.utils.data import Dataset


class StreamDataset(Dataset):
    """Chunks a 1-D token stream into (seq_len+1)-token blocks.

    With return_idx=True each item is (block, block_index) so the training loop can map
    positions back to the stream (block i covers stream positions [i*seq_len, i*seq_len+seq_len]);
    used by the SLM objective to look up precomputed per-position reference NLLs.
    """

    def __init__(self, tokens_path: str, seq_len: int, return_idx: bool = False):
        self.tokens = torch.load(tokens_path)  # int16 1-D
        self.seq_len = seq_len
        self.n = max(0, (self.tokens.numel() - 1) // seq_len)
        self.return_idx = return_idx

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        s = i * self.seq_len
        block = self.tokens[s : s + self.seq_len + 1].to(torch.long)
        return (block, i) if self.return_idx else block


def _build_span_sel(maskable, mask_p, p_geo, max_span, generator):
    """Span-based mask selection (SpanBERT / LTG GPT-BERT recipe): mask contiguous spans
    (geometric lengths, mean ~1/p_geo, capped at max_span) until ~mask_p of the maskable
    tokens are covered. Predicting spans forces more syntactic competence than isolated tokens."""
    B, T = maskable.shape
    sel = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        target = int(mask_p * int(maskable[b].sum()))
        cnt, guard = 0, 0
        while cnt < target and guard < 4 * T:
            guard += 1
            u = torch.rand(1, generator=generator).item()
            span_len = min(max_span, 1 + int(math.log(1.0 - u) / math.log(1.0 - p_geo)))
            start = int(torch.randint(0, T, (1,), generator=generator).item())
            for i in range(start, min(T, start + span_len)):
                if maskable[b, i] and not sel[b, i]:
                    sel[b, i] = True
                    cnt += 1
    return sel


def mask_tokens(block, mask_p, mask_id, vocab_size, n_special, random_p=0.1, keep_p=0.1,
                generator=None, span=False, p_geo=0.3, max_span=10, token_probs=None):
    """Token-level (or span-level) MLM masking on a B x T long batch (operates on CPU tensors).

    Returns (masked_input, target_full): target_full holds original ids at masked
    positions and -100 elsewhere. 80% -> <mask>, 10% -> random token, 10% -> keep.
    Special tokens (id < n_special) are never masked. With span=True, masked positions are
    chosen as contiguous spans. With token_probs (a [vocab] CPU tensor), each position is
    masked with its token-id's adaptive probability (Edman & Fraser 2025 "Mask and You Shall
    Receive") instead of the uniform mask_p.
    """
    B, T = block.shape
    maskable = block >= n_special
    if span:
        sel = _build_span_sel(maskable, mask_p, p_geo, max_span, generator)
    elif token_probs is not None:
        sel = (torch.rand(B, T, generator=generator) < token_probs[block]) & maskable
    else:
        sel = (torch.rand(B, T, generator=generator) < mask_p) & maskable
    target_full = torch.where(sel, block, torch.full_like(block, -100))

    masked = block.clone()
    r = torch.rand(B, T, generator=generator)
    mask_thr = 1.0 - random_p - keep_p          # default 0.8
    rand_thr = 1.0 - keep_p                      # default 0.9
    to_mask = sel & (r < mask_thr)
    to_rand = sel & (r >= mask_thr) & (r < rand_thr)
    masked[to_mask] = mask_id
    n_rand = int(to_rand.sum())
    if n_rand > 0:
        masked[to_rand] = torch.randint(n_special, vocab_size, (n_rand,), generator=generator, dtype=masked.dtype)
    # remaining ~10% are kept unchanged
    return masked, target_full


def build_example(block, mode, mask_p, mask_id, vocab_size, n_special, generator=None, span=False, token_probs=None):
    """Turn a B x (L+1) block into (input_ids, target_ids), both B x L, for the given mode."""
    if mode == "causal":
        return block[:, :-1].contiguous(), block[:, 1:].contiguous()
    masked, target_full = mask_tokens(block, mask_p, mask_id, vocab_size, n_special,
                                      generator=generator, span=span, token_probs=token_probs)
    return masked[:, :-1].contiguous(), target_full[:, 1:].contiguous()
