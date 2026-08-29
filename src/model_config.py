#!/usr/bin/env python
"""GPT-BERT model configs for the BabyLM Strict-Small run + a self-contained HF saver.

- `small` (~33M): fast-iteration config (LTG's proven small shape, our 16k vocab).
- `base`  (119M): matches the official 2026 GPT-BERT Strict-Small baseline (for the final model).

Every saved checkpoint includes the modeling/config .py files + `auto_map`, so it loads via
AutoModelFor{CausalLM,MaskedLM}.from_pretrained(..., trust_remote_code=True) — exactly what the
BabyLM eval pipeline uses.
"""
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gpt_bert.configuration_gpt_bert import ModelConfig  # noqa: E402
from gpt_bert.modeling_gpt_bert import (  # noqa: E402, F401
    GPTBERT,
    GPTBERTForCausalLM,
    GPTBERTForMaskedLM,
)

AUTO_MAP = {
    "AutoConfig": "configuration_gpt_bert.ModelConfig",
    "AutoModel": "modeling_gpt_bert.GPTBERT",
    "AutoModelForCausalLM": "modeling_gpt_bert.GPTBERTForCausalLM",
    "AutoModelForMaskedLM": "modeling_gpt_bert.GPTBERTForMaskedLM",
}

CONFIGS = {
    "small": dict(hidden_size=384, num_layers=12, num_attention_heads=6, intermediate_size=1280),
    "base": dict(hidden_size=768, num_layers=12, num_attention_heads=12, intermediate_size=2560),
}


def build_config(name="small", vocab_size=16384, max_position_embeddings=512) -> ModelConfig:
    spec = dict(CONFIGS[name])
    return ModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        position_bucket_size=32,
        layer_norm_eps=1e-5,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        architectures=["GPTBERTForCausalLM"],
        auto_map=AUTO_MAP,
        **spec,
    )


def save_hf_checkpoint(model, out_dir, tokenizer=None):
    """Write a self-contained trust_remote_code checkpoint (weights + modeling .py + tokenizer)."""
    os.makedirs(out_dir, exist_ok=True)
    # .bin (not safetensors) to match the baseline and avoid tied-weight serialization issues
    model.save_pretrained(out_dir, safe_serialization=False)
    for fn in ("modeling_gpt_bert.py", "configuration_gpt_bert.py"):
        shutil.copy(os.path.join(_HERE, "gpt_bert", fn), os.path.join(out_dir, fn))
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name in ("small", "base"):
        cfg = build_config(name)
        m = GPTBERTForMaskedLM(cfg)
        n = count_params(m)
        emb = cfg.vocab_size * cfg.hidden_size
        print(f"{name:6s}: {n/1e6:6.1f}M params | hidden={cfg.hidden_size} layers={cfg.num_layers} "
              f"heads={cfg.num_attention_heads} ffn={cfg.intermediate_size} "
              f"(embeddings {emb/1e6:.1f}M, non-embed {(n-emb)/1e6:.1f}M)")
