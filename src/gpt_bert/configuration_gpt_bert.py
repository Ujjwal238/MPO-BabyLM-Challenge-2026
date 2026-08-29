"""GPT-BERT configuration, vendored from the BabyLM 2026 official baseline.

Upstream authorship as in modeling_gpt_bert.py (Charpentier and Samuel). One change is
ours and is marked inline: the original __init__ hard-coded the base configuration when
no config_file was passed, so a custom model size was silently reset on a
save_pretrained -> from_pretrained round-trip. The fix honors kwargs, and the custom
to_dict / to_json overrides that caused the same loss were removed.
"""
from __future__ import annotations

import json
import pathlib
import copy

from typing import Any
from transformers.configuration_utils import PretrainedConfig


class ModelConfig(PretrainedConfig):

    def __init__(self: ModelConfig, config_file: pathlib.Path | str | None = None, **kwargs):
        """ """
        super().__init__(**kwargs)

        if config_file is not None:
            if isinstance(config_file, str):
                config_file = pathlib.Path(config_file)
            config: dict[str, Any] = json.load(config_file.open("r"))
            for key, value in config.items():
                setattr(self, key, value)

        # Honor kwargs / values restored by super().__init__ so that custom model
        # sizes survive a save_pretrained -> from_pretrained round-trip. (The original
        # code unconditionally hard-coded the base config here, silently resetting any
        # custom size on reload -- which would break the eval pipeline's from_pretrained.)
        self.attention_probs_dropout_prob = getattr(self, "attention_probs_dropout_prob", 0.1)
        self.hidden_dropout_prob = getattr(self, "hidden_dropout_prob", 0.1)
        self.hidden_size = getattr(self, "hidden_size", 768)
        self.intermediate_size = getattr(self, "intermediate_size", 2560)
        self.max_position_embeddings = getattr(self, "max_position_embeddings", getattr(self, "max_sequence_length", 512))
        self.position_bucket_size = getattr(self, "position_bucket_size", 32)
        self.num_attention_heads = getattr(self, "num_attention_heads", 12)
        n_layers = getattr(self, "num_layers", getattr(self, "num_hidden_layers", 12))
        self.num_layers = n_layers
        self.num_hidden_layers = n_layers
        self.vocab_size = getattr(self, "vocab_size", 16384)
        self.layer_norm_eps = getattr(self, "layer_norm_eps", 1e-5)

    # Note: to_dict / to_json_string / to_json_file / __repr__ are intentionally NOT
    # overridden -- we inherit PretrainedConfig's implementations, which serialize
    # torch_dtype correctly and support use_diff. Our __init__ defaults equal the base
    # config, so a diff-based save still round-trips exactly.
