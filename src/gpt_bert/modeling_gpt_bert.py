"""GPT-BERT modeling code, vendored from the BabyLM 2026 official baseline.

Not our work. The architecture and this implementation are by Lucas Georges Gabriel
Charpentier and David Samuel, "GPT or BERT: why not both?" (BabyLM 2024, winning
Strict-Small entry), distributed with the BabyLM baseline checkpoints. It is included
here unmodified so that the released checkpoints load without a network fetch and so
that a reader can see exactly which backbone the paper trained.

Upstream: https://huggingface.co/babylm/babylm-2026-baseline-gpt-bert-strict-small
Paper:    https://aclanthology.org/2024.conll-babylm.24/
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from .configuration_gpt_bert import ModelConfig

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput
)

from typing import Optional, Union


# From https://github.com/epfml/DenseFormer
class InPlaceSetSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_idx, x_val):
        full_tensor[x_idx] = x_val
        ctx.x_idx = x_idx
        ret = torch.Tensor().to(full_tensor.device)
        ret.set_(full_tensor[:x_idx + 1])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.x_idx == 0:
            return None, None, None, grad_out[ctx.x_idx]
        else:
            return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx]


def apply_inplace_set(x_acc, x_idx, x_val):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
    return full_tensor, new_slice


class DWAModules(torch.nn.Module):
    def __init__(self, hidden_size, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(i + 2)) for i in range(n_blocks)])
        self.accumulator = None
        self._init_weights()

    def _init_weights(self):
        for module in self.alphas:
            module.data.zero_()
            module.data[-1] = 1.0

    def init_accumulator(self, x):
        self.accumulator = (torch.zeros((self.n_blocks + 1, *x.shape), device=x.device, dtype=x.dtype), None)
        self.accumulator = apply_inplace_set(self.accumulator, 0, x)

    def forward(self, x, block_idx):
        assert self.accumulator is not None, "`init_accumulator(x)` needs to be called first"
        self.accumulator = apply_inplace_set(
            self.accumulator,
            block_idx + 1,
            x
        )
        x = torch.tensordot(self.alphas[block_idx], self.accumulator[1], dims=1)
        return x


class Layer(nn.Module):

    def __init__(self: "Layer", config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

        self.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + layer_idx)))
        self.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + layer_idx)))

    def forward(self: "Layer", x: torch.Tensor, attention_mask: torch.Tensor, relative_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention: torch.Tensor
        attention_probs: torch.Tensor
        attention, attention_probs = self.attention(x, attention_mask, relative_embedding)
        x += attention
        x += self.mlp(x)

        return x, attention_probs


class MaskClassifier(nn.Module):

    def __init__(self: "MaskClassifier", config: ModelConfig, subword_embedding: nn.Parameter):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self: "MaskClassifier", hidden_size: int, embedding: nn.Parameter):
        std: float = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self: "MaskClassifier", x: torch.Tensor, masked_lm_labels: torch.Tensor | None = None) -> torch.Tensor:
        if masked_lm_labels is not None:
            x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        x = self.nonlinearity(x)

        return x


class GeGLU(nn.Module):
    def forward(self: "GeGLU", x: torch.Tensor) -> torch.Tensor:
        gate: torch.Tensor
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class FeedForward(nn.Module):
    def __init__(self: "FeedForward", config: ModelConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self: "FeedForward", hidden_size: int) -> None:
        std: float = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self: "FeedForward", x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self: "MaskedSoftmax", x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        self.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self: "MaskedSoftmax", grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        output: torch.Tensor
        output, = self.saved_tensors
        inputGrad: torch.Tensor = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class Attention(nn.Module):
    def __init__(self: "Attention", config: ModelConfig) -> None:
        super().__init__()

        self.config: ModelConfig = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_size: int = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        self.in_proj_vg = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)

        position_indices: torch.Tensor = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=False)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale: float = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self: "Attention", relative_pos: torch.Tensor, bucket_size: int, max_position: int) -> torch.Tensor:
        sign: torch.Tensor = torch.sign(relative_pos)
        mid: int = bucket_size // 2
        abs_pos: torch.Tensor = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos: torch.Tensor = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)).int() + mid
        bucket_pos: torch.Tensor = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self: "Attention") -> None:
        std: float = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_vg.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_vg.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def _create_position_tensors(
        self: "Attention",
        relative_embedding: torch.Tensor,
        query_len: int,
        key_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.in_proj_qk(self.dropout(relative_embedding))

        indices = self.position_indices[:query_len, :key_len]
        if indices.numel() > 0:
            indices = indices.clamp(0, pos.size(0) - 1)

        pos = F.embedding(indices, pos)
        query_pos, key_pos = pos.chunk(2, dim=-1)
        query_pos = query_pos.view(query_len, key_len, self.num_heads, self.head_size)
        key_pos = key_pos.view(query_len, key_len, self.num_heads, self.head_size)

        return query_pos, key_pos

    def attention_operation(self: "Attention", query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor, query_pos: torch.Tensor, key_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_len: int
        batch_size: int
        key_len, batch_size, _ = key.size()
        query_len: int
        query_len, _, _ = query.size()

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_probs: torch.Tensor = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)
        attention_probs = attention_probs.view(batch_size, self.num_heads, query_len, key_len)
        attention_probs.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        attention_probs.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))

        attention_probs = MaskedSoftmax.apply(attention_probs, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        attention_output: torch.Tensor = torch.bmm(attention_probs.flatten(0, 1), value)
        attention_output = attention_output.transpose(0, 1).reshape(query_len, batch_size, self.hidden_size)

        return attention_output, attention_probs

    def forward(self: "Attention", hidden_states: torch.Tensor, attention_mask: torch.Tensor, relative_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_len: int
        batch_size: int
        key_len, batch_size, _ = hidden_states.size()
        query_len: int = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) \
                - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.config.position_bucket_size, 512)
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.register_buffer("position_indices", position_indices.to(hidden_states.device), persistent=True)

        hidden_states = self.pre_layer_norm(hidden_states)
        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)
        value, gate = self.in_proj_vg(hidden_states).chunk(2, dim=2)
        gate = F.gelu(gate)

        query_pos, key_pos = self._create_position_tensors(relative_embedding, query_len, key_len)

        attention_output, attention_probs = self.attention_operation(query, key, value, attention_mask, query_pos, key_pos)
        attention_output = attention_output * gate
        attention_output = self.post_layer_norm(attention_output)
        attention_output = self.out_proj(attention_output)
        attention_output = self.dropout(attention_output)

        return attention_output, attention_probs


class Embedding(nn.Module):
    def __init__(self: "Embedding", config: ModelConfig):
        super().__init__()
        self.hidden_size: int = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self: "Embedding"):
        std: float = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self: "Embedding", input_ids: torch.Tensor):
        word_embedding: torch.Tensor = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings: torch.Tensor = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings


class GPTBERTPreTrainedModel(PreTrainedModel):
    config_class = ModelConfig
    supports_gradient_checkpointing = False
    base_model_prefix = "transformer"

    def _set_gradient_checkpointing(self, module, value=False):
        raise NotImplementedError("Gradient checkpointing is not supported by this model")

    def _init_weights(self, module):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))

        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2*std, b=2*std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPTBERT(GPTBERTPreTrainedModel):

    def __init__(self, config: ModelConfig, is_causal: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embedding = Embedding(config)
        self.attention_layers = nn.ModuleList([Attention(config) for _ in range(config.num_layers)])
        self.mlp_layers = nn.ModuleList([FeedForward(config) for _ in range(config.num_layers)])
        self.dwa_modules = DWAModules(config.hidden_size, config.num_layers * 2)

        for i, layer in enumerate(self.mlp_layers):
            layer.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.is_causal = is_causal

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def get_contextualized_embeddings(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = input_ids.new_zeros((batch_size, seq_length), dtype=torch.bool).unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = ~attention_mask.bool()

        if len(attention_mask.size()) == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.size()) == 3:
            attention_mask = attention_mask.unsqueeze(1)

        if self.is_causal:
            attention_mask = attention_mask | input_ids.new_ones((seq_length, seq_length), dtype=torch.bool).triu(1).unsqueeze(0).unsqueeze(0)

        static_embeddings, relative_embeddings = self.embedding(input_ids.t())
        contextualized_embeddings = [static_embeddings]
        attention_probs = []
        self.dwa_modules.init_accumulator(static_embeddings)
        for i, (attention_layer, mlp_layer) in enumerate(zip(self.attention_layers, self.mlp_layers)):
            attention, layer_attention_probs = attention_layer(contextualized_embeddings[-1], attention_mask, relative_embeddings)
            layer_embeddings = contextualized_embeddings[-1] + attention
            layer_embeddings = self.dwa_modules(layer_embeddings, block_idx=i * 2)
            layer_embeddings = layer_embeddings + mlp_layer(layer_embeddings)
            layer_embeddings = self.dwa_modules(layer_embeddings, block_idx=i * 2 + 1)
            contextualized_embeddings.append(layer_embeddings)
            attention_probs.append(layer_attention_probs)
        contextualized_embeddings = [emb.transpose(0, 1) for emb in contextualized_embeddings]
        last_layer = contextualized_embeddings[-1]
        return last_layer, contextualized_embeddings, attention_probs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[tuple[torch.Tensor], BaseModelOutput]:
        return_dict = True if return_dict is None else return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)

        if not return_dict:
            return (
                sequence_output,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class GPTBERTForCausalLM(GPTBERTPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["head"]

    @property
    def all_tied_weights_keys(self):
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        return {k: True for k in keys}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.transformer = GPTBERT(config, is_causal=True, **kwargs)
        self.vocab_size = config.vocab_size
        self.classifier = MaskClassifier(config, self.transformer.embedding.word_embedding.weight)
        self.hidden_size = config.hidden_size

    def get_output_embeddings(self):
        return self.classifier.nonlinearity[-1].weight

    def set_output_embeddings(self, new_embeddings):
        self.classifier.nonlinearity[-1].weight = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.transformer.embedding.word_embedding = value

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def can_generate(self):
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[tuple, CausalLMOutput]:
        return_dict = True if return_dict is None else return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.transformer.get_contextualized_embeddings(
            input_ids, attention_mask
        )
        subword_prediction = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            gold_labels = labels.flatten()
            gold_labels = gold_labels[gold_labels != -100]
            loss = F.cross_entropy(subword_prediction, gold_labels)

        if not return_dict:
            output = (
                subword_prediction,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=subword_prediction,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        num_logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class GPTBERTForMaskedLM(GPTBERTPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["head"]

    @property
    def all_tied_weights_keys(self):
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        return {k: True for k in keys}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.transformer = GPTBERT(config, is_causal=False, **kwargs)
        self.vocab_size = config.vocab_size
        self.classifier = MaskClassifier(config, self.transformer.embedding.word_embedding.weight)
        self.hidden_size = config.hidden_size

    def get_output_embeddings(self):
        return self.classifier.nonlinearity[-1].weight

    def set_output_embeddings(self, new_embeddings):
        self.classifier.nonlinearity[-1].weight = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.transformer.embedding.word_embedding = value

    def set_encoder(self, encoder):
        self.transformer = encoder

    def get_encoder(self):
        return self.transformer

    def can_generate(self):
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[tuple, CausalLMOutput]:
        return_dict = True if return_dict is None else return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.transformer.get_contextualized_embeddings(
            input_ids, attention_mask
        )
        subword_prediction = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            gold_labels = labels.flatten()
            gold_labels = gold_labels[gold_labels != -100]
            loss = F.cross_entropy(subword_prediction, gold_labels)

        if not return_dict:
            output = (
                subword_prediction,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=subword_prediction,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )