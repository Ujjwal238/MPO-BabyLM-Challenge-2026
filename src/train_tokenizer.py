#!/usr/bin/env python
"""Train a byte-level BPE tokenizer on the BabyLM 2026 Strict-Small corpus.

Matches the official GPT-BERT baseline tokenizer interface:
  - ByteLevel BPE, vocab_size 16,384
  - special tokens <unk>,<s>,</s>,<pad>,<mask> mapped to ids 0..4
  - model_max_length 512, clean_up_tokenization_spaces False
...but the vocabulary is learned from OUR 10M-word corpus.

Usage:
  python src/train_tokenizer.py [--vocab_size 16384] [--data_dir ...] [--out_dir ...]
"""
import argparse
import glob
import os

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

REPO = os.environ.get("MPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = f"{REPO}/data/strict_small"
DEFAULT_OUT = f"{REPO}/artifacts/tokenizer"
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]  # order fixes ids 0..4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--vocab_size", type=int, default=16384)
    ap.add_argument("--min_frequency", type=int, default=2)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.train.txt")))
    assert files, f"no *.train.txt found in {args.data_dir}"
    print("training on:", [os.path.basename(f) for f in files])

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train(files, trainer)

    os.makedirs(args.out_dir, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=512,
        clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(args.out_dir)
    print("saved tokenizer to", args.out_dir)

    # verification
    print("vocab_size:", fast.vocab_size, "| len(tokenizer):", len(fast))
    for t in SPECIALS:
        print(f"  {t:7s} -> id {fast.convert_tokens_to_ids(t)}")
    sample = 'The cat sat on the mat. *CHI: more milk please!'
    ids = fast.encode(sample)
    print("sample:", sample)
    print("n_tokens:", len(ids), "| ids[:20]:", ids[:20])
    print("decoded:", repr(fast.decode(ids)))


if __name__ == "__main__":
    main()
