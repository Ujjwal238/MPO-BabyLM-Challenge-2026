#!/usr/bin/env python
"""Train a SentencePiece-style Unigram tokenizer (vocab 8192) on the Strict-Small corpus.

Track-B implementation of the morpheme-aware tokenizer lever (NEXT_APPROACH_PLAN.md):
Unigram LM subwords are strongly morph-aligned (Bostrom & Durrett 2020) and deliver the
mechanism claimed by Bölücü & Can 2025 (stable stem/affix token identities, small vocab)
with standard fast-tokenizer machinery: exact round-trip, offset mappings, ~0 <unk>.

Validation battery mirrors train_morph_tokenizer.py and adds a Morfessor-agreement
diagnostic (%% of word types where Unigram picks the same boundaries as the trained
Morfessor model, if its binary is present).

  python src/train_uni_tokenizer.py --vocab_size 8192 \
      --out_dir artifacts/tokenizer_uni_8k
"""
import argparse
import glob
import os
import random
import re

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]  # order fixes ids 0..4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=f"{REPO}/data/strict_small")
    ap.add_argument("--out_dir", default=f"{REPO}/artifacts/tokenizer_uni_8k")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--bpe_tokenizer", default=f"{REPO}/artifacts/tokenizer")
    ap.add_argument("--morfessor_model", default=f"{REPO}/artifacts/tokenizer_morph_8k/morfessor_model.bin")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.train.txt")))
    assert files, f"no *.train.txt in {args.data_dir}"
    print("corpus files:", [os.path.basename(f) for f in files])

    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = trainers.UnigramTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        unk_token="<unk>",
        show_progress=True,
    )
    tok.train(files, trainer)

    os.makedirs(args.out_dir, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", mask_token="<mask>",
        model_max_length=512, clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(args.out_dir)
    print("saved tokenizer to", args.out_dir)

    # ---------------- validation battery ----------------
    print("\n=== VALIDATION ===")
    reloaded = AutoTokenizer.from_pretrained(args.out_dir)
    assert reloaded.is_fast
    for t in SPECIALS:
        print(f"  {t:7s} -> id {reloaded.convert_tokens_to_ids(t)}")

    for w in ["undoing", "reconsideration", "cats", "walked", "unhappiness", "playing"]:
        print(f"  {w:18s} -> {reloaded.tokenize(' ' + w)}")

    random.seed(0)
    sample_lines = []
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        sample_lines += random.sample(lines, min(300, len(lines)))
    ok = 0
    bad_example = None
    for s in sample_lines:
        dec = reloaded.decode(reloaded.encode(s, add_special_tokens=False))
        if dec.strip() == re.sub(r"\s+", " ", s).strip():
            ok += 1
        elif bad_example is None:
            bad_example = (s, dec)
    print(f"  round-trip exact: {ok}/{len(sample_lines)} ({100*ok/len(sample_lines):.1f}%)")
    if bad_example:
        print(f"    first mismatch: {bad_example[0][:70]!r} -> {bad_example[1][:70]!r}")

    for w in ["zygomorphic", "blicket", "Kardashian", "email", "iPhone"]:
        print(f"  OOV '{w}' -> {reloaded.tokenize(' ' + w)}")

    bpe = AutoTokenizer.from_pretrained(args.bpe_tokenizer)
    n_words = sum(len(s.split()) for s in sample_lines)
    n_uni = sum(len(reloaded.encode(s, add_special_tokens=False)) for s in sample_lines)
    n_bpe = sum(len(bpe.encode(s, add_special_tokens=False)) for s in sample_lines)
    print(f"  fertility (tokens/word): uni {n_uni/n_words:.3f} vs BPE {n_bpe/n_words:.3f} "
          f"(ratio {n_uni/max(1,n_bpe):.3f}; gate 1.35)")
    unk_id = reloaded.convert_tokens_to_ids("<unk>")
    n_unk = sum(1 for s in sample_lines for i in reloaded.encode(s, add_special_tokens=False) if i == unk_id)
    print(f"  <unk> rate: {100*n_unk/max(1,n_uni):.4f}%")

    # Morfessor-agreement diagnostic (mechanism check, not a gate)
    if os.path.exists(args.morfessor_model):
        import morfessor
        mm = morfessor.MorfessorIO().read_binary_model_file(args.morfessor_model)
        words = [w for s in sample_lines for w in re.findall(r"[A-Za-z]{4,}", s)]
        words = random.sample(words, min(2000, len(words)))
        agree = 0
        for w in words:
            try:
                m_seg = tuple(mm.viterbi_segment(w)[0])
            except Exception:
                continue
            u_seg = tuple(t.lstrip("▁") for t in reloaded.tokenize(" " + w))
            agree += (m_seg == u_seg)
        print(f"  Morfessor boundary agreement on {len(words)} word tokens: {100*agree/max(1,len(words)):.1f}%")


if __name__ == "__main__":
    main()
