#!/usr/bin/env python
"""Train a morpheme-aware tokenizer (Morfessor -> HF Unigram) on the Strict-Small corpus.

Replicates Bölücü & Can 2025 ("A Morpheme-Aware Child-Inspired Language Model"):
unsupervised Morfessor Baseline trained ONLY on the 10M-word corpus (strict-legal),
vocab 8192, same special-token layout as our BPE tokenizer (<unk>,<s>,</s>,<pad>,<mask> = 0..4).

Key engineering idea: Morfessor Baseline IS a unigram lexicon model (MDL-flavoured), so we
export its lexicon + usage counts into a native `tokenizers` **Unigram** model whose Viterbi
segmentation closely reproduces Morfessor's, while staying a standard fast tokenizer:
AutoTokenizer-loadable, offset-mapping capable (the eval pipeline requires fast tokenizers),
OOV-safe via single-character fallback pieces.

Pipeline:
  1. word counts from *.train.txt (Metaspace-style word marker ▁ added later)
  2. Morfessor Baseline train (types weighted by counts via dampening; default settings)
  3. segment every word type -> morph sequence; count surface pieces:
       first morph of word -> "▁"+morph ; later morphs -> morph
  4. vocab = specials + all single-char fallback pieces + top surface pieces (to --vocab_size)
  5. build tokenizers.models.Unigram with log-prob scores; Metaspace pre-tokenizer/decoder
  6. save PreTrainedTokenizerFast + validation battery (round-trip, OOV, tokens/word vs BPE)

  python src/train_morph_tokenizer.py --vocab_size 8192 \
      --out_dir artifacts/tokenizer_morph_8k
"""
import argparse
import collections
import glob
import math
import os
import re

import morfessor
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, AutoTokenizer

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]  # order fixes ids 0..4
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)  # words | single punctuation marks


def word_counts(files):
    counts = collections.Counter()
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                counts.update(WORD_RE.findall(line.strip()))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=f"{REPO}/data/strict_small")
    ap.add_argument("--out_dir", default=f"{REPO}/artifacts/tokenizer_morph_8k")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--dampening", default="log", choices=["none", "log", "ones"],
                    help="Morfessor count dampening (log = their common default for text)")
    ap.add_argument("--bpe_tokenizer", default=f"{REPO}/artifacts/tokenizer",
                    help="existing BPE tokenizer for the tokens/word comparison")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.train.txt")))
    assert files, f"no *.train.txt in {args.data_dir}"
    print("corpus files:", [os.path.basename(f) for f in files])

    # --- 1. word counts ---
    counts = word_counts(files)
    total_tokens = sum(counts.values())
    print(f"word types: {len(counts):,} | word tokens: {total_tokens:,}")

    # --- 2. Morfessor Baseline ---
    io = morfessor.MorfessorIO()
    model = morfessor.BaselineModel()
    damp = {"none": None,
            "log": lambda x: max(1, int(round(math.log(x + 1, 2)))),
            "ones": lambda x: 1}[args.dampening]
    model.load_data(((c, w) for w, c in counts.items()), count_modifier=damp)
    print("training Morfessor Baseline (recursive MDL search)...")
    model.train_batch()
    print(f"morfessor lexicon size (constructions): {len(list(model.get_constructions())):,}")

    # --- 3. segment word types; count surface pieces (Metaspace convention) ---
    seg_cache = {}
    piece_counts = collections.Counter()
    for w, c in counts.items():
        try:
            segs = model.viterbi_segment(w)[0]
        except Exception:
            segs = [w]
        seg_cache[w] = segs
        piece_counts["▁" + segs[0]] += c
        for m in segs[1:]:
            piece_counts[m] += c
        # single punctuation marks also appear word-attached ("mat.") -> count the bare piece too
        if len(w) == 1 and not w.isalnum():
            piece_counts[w] += c

    # --- 4. vocab: specials + char fallbacks + top pieces ---
    chars = set()
    for w in counts:
        chars.update(w)
    fallback = {"▁"} | {ch for ch in chars} | {"▁" + ch for ch in chars}
    n_reserved = len(SPECIALS) + len(fallback)
    top_budget = args.vocab_size - n_reserved
    assert top_budget > 0
    top_pieces = [p for p, _ in piece_counts.most_common() if p not in fallback][:top_budget]
    print(f"vocab: {len(SPECIALS)} specials + {len(fallback)} char-fallback + {len(top_pieces)} morph pieces "
          f"= {n_reserved + len(top_pieces)} (target {args.vocab_size})")

    # Unigram scores = log relative frequency (floor for fallback pieces)
    total_pc = sum(piece_counts.values())
    def score(p):
        c = piece_counts.get(p, 0)
        return math.log(max(c, 0.5) / total_pc)
    vocab = [(s, 0.0) for s in SPECIALS]
    vocab += sorted(((p, score(p)) for p in fallback), key=lambda x: -x[1])
    vocab += [(p, score(p)) for p in top_pieces]

    tok = Tokenizer(models.Unigram(vocab, unk_id=0, byte_fallback=False))
    tok.normalizer = normalizers.NFC()
    # Metaspace marks word starts; Punctuation("isolated") then splits attached punctuation
    # ("mat." -> "▁mat" + ".") so tokenization matches the WORD_RE counting convention.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"),
        pre_tokenizers.Punctuation(behavior="isolated"),
    ])
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    os.makedirs(args.out_dir, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", mask_token="<mask>",
        model_max_length=512, clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(args.out_dir)
    io.write_binary_model_file(os.path.join(args.out_dir, "morfessor_model.bin"), model)
    print("saved tokenizer to", args.out_dir)

    # ---------------- validation battery ----------------
    print("\n=== VALIDATION ===")
    reloaded = AutoTokenizer.from_pretrained(args.out_dir)
    for t in SPECIALS:
        print(f"  {t:7s} -> id {reloaded.convert_tokens_to_ids(t)}")
    assert reloaded.is_fast, "must be a fast tokenizer (eval needs offset mappings)"

    # (a) morphology sanity
    for w in ["undoing", "reconsideration", "cats", "walked", "unhappiness", "playing"]:
        print(f"  morfessor {w:18s} -> {seg_cache.get(w, model.viterbi_segment(w)[0])} "
              f"| tokenizer -> {reloaded.tokenize(' ' + w)}")

    # (b) round-trip fidelity on corpus sample + OOV words
    import random
    random.seed(0)
    sample_lines = []
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        sample_lines += random.sample(lines, min(300, len(lines)))
    ok = 0
    for s in sample_lines:
        dec = reloaded.decode(reloaded.encode(s, add_special_tokens=False))
        if dec.strip() == re.sub(r"\s+", " ", s).strip():
            ok += 1
    print(f"  round-trip exact: {ok}/{len(sample_lines)} ({100*ok/len(sample_lines):.1f}%)")

    for w in ["zygomorphic", "blicket", "Kardashian", "email", "iPhone"]:  # OOV-ish
        print(f"  OOV '{w}' -> {reloaded.tokenize(' ' + w)}")

    # (c) tokens/word ratio vs BPE (fertility)
    bpe = AutoTokenizer.from_pretrained(args.bpe_tokenizer)
    n_words = sum(len(s.split()) for s in sample_lines)
    n_morph = sum(len(reloaded.encode(s, add_special_tokens=False)) for s in sample_lines)
    n_bpe = sum(len(bpe.encode(s, add_special_tokens=False)) for s in sample_lines)
    print(f"  fertility (tokens/word): morph {n_morph/n_words:.3f} vs BPE {n_bpe/n_words:.3f} "
          f"(ratio {n_morph/max(1,n_bpe):.3f}; >1.35 = seq-128 context concern)")
    unk_id = reloaded.convert_tokens_to_ids("<unk>")
    n_unk = sum(1 for s in sample_lines for i in reloaded.encode(s, add_special_tokens=False) if i == unk_id)
    print(f"  <unk> rate on sample: {100*n_unk/max(1,n_morph):.4f}%")


if __name__ == "__main__":
    main()
