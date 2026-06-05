#!/usr/bin/env python3
"""
misotts_mlx.py  --  MLX (Apple Silicon) port of MisoTTS 8B.

MisoTTS is Sesame-CSM scaled up, and CSM already has a clean MLX implementation
(senstella/csm-mlx). This module adapts that template:

  * the transformer blocks are `mlx_lm`'s LlamaModel with their default attention
    swapped for an interleaved-pair Llama3ScaledRoPE attention (ported verbatim
    from senstella/csm-mlx) -- this matches torchtune's RoPE convention exactly,
    so the torchtune checkpoint loads with NO q/k weight permutation;
  * the heads/embeddings (text_embeddings, audio_embeddings, projection,
    codebook0_head, audio_head) mirror MisoTTS's torch `Model` 1:1;
  * `load_misotts_mlx()` reads the torch `model.safetensors` and renames the
    torchtune keys (attn.output_proj, mlp.w1/w2/w3, sa_norm/mlp_norm.scale) to
    mlx_lm keys (self_attn.o_proj, mlp.{gate,up,down}_proj, input_layernorm /
    post_attention_layernorm.weight).

Requires: mlx, mlx-lm.  (No transformers / torch needed for the model itself.)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.llama import LlamaModel, ModelArgs

# ----------------------------------------------------------------------------
# MisoTTS config (from torch models.py: llama3_2_8B backbone, llama3_2_300M decoder)
# ----------------------------------------------------------------------------
N_TEXT_VOCAB = 128_256
N_AUDIO_VOCAB = 2_051
N_AUDIO_CODEBOOKS = 32

_ROPE_SCALING = {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def _backbone_args() -> ModelArgs:
    return ModelArgs(
        model_type="llama",
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14_336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,  # 4096 / 32
        vocab_size=N_TEXT_VOCAB,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        rope_scaling=_ROPE_SCALING,
    )


def _decoder_args() -> ModelArgs:
    return ModelArgs(
        model_type="llama",
        hidden_size=1536,
        num_hidden_layers=8,
        intermediate_size=6_912,
        num_attention_heads=24,
        num_key_value_heads=6,
        head_dim=64,  # 1536 / 24
        vocab_size=N_TEXT_VOCAB,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        rope_scaling=_ROPE_SCALING,
    )


# ----------------------------------------------------------------------------
# RoPE + Attention  (ported verbatim from senstella/csm-mlx -- torchtune convention)
# ----------------------------------------------------------------------------
class Llama3ScaledRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10_000.0,
        scale_factor: float = 8.0,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        freqs = 1.0 / (
            self.base
            ** (mx.arange(0, self.dim, 2)[: (self.dim // 2)].astype(mx.float32) / self.dim)
        )
        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self._theta = theta
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = mx.arange(max_seq_len, dtype=self._theta.dtype)
        idx_theta = mx.einsum("i, j -> ij", seq_idx, self._theta).astype(mx.float32)
        cache = mx.stack([mx.cos(idx_theta), mx.sin(idx_theta)], axis=-1)
        self._cache = cache

    def apply_scaling(
        self,
        freqs: mx.array,
        scale_factor: float,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return mx.array(new_freqs, dtype=freqs.dtype)

    def __call__(self, x: mx.array, *, offset: int) -> mx.array:
        if not self.is_cache_built:
            raise RuntimeError("RoPE cache is not built. Please call rope_init() first.")
        seq_len = x.shape[1]
        rope_cache = self._cache[None, offset : offset + seq_len]
        xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.reshape(-1, xshaped.shape[1], 1, xshaped.shape[3], 2)
        x_out = mx.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads or n_heads
        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = Llama3ScaledRoPE(
            self.head_dim,
            base=args.rope_theta,
            scale_factor=args.rope_scaling.get("factor", 1.0),
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        b, s_x, _ = x.shape
        q = self.q_proj(x).reshape(b, s_x, -1, self.head_dim)
        k = self.k_proj(x).reshape(b, s_x, -1, self.head_dim)
        v = self.v_proj(x).reshape(b, s_x, -1, self.head_dim)

        q = self.rope(q, offset=cache.offset if cache else 0)
        k = self.rope(k, offset=cache.offset if cache else 0)

        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        if cache:
            k, v = cache.update_and_fetch(k, v)

        if self.n_heads != self.n_kv_heads:
            q_per_kv = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, q_per_kv, axis=1)
            v = mx.repeat(v, q_per_kv, axis=1)

        output = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        output = output.swapaxes(1, 2).reshape(b, s_x, -1)
        return self.o_proj(output)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class MisoTTSMLX(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_text_vocab = N_TEXT_VOCAB
        self.n_audio_vocab = N_AUDIO_VOCAB
        self.n_audio_codebooks = N_AUDIO_CODEBOOKS

        bb = _backbone_args()
        dec = _decoder_args()
        self.backbone = LlamaModel(bb)
        self.decoder = LlamaModel(dec)

        backbone_dim = bb.hidden_size
        decoder_dim = dec.hidden_size

        self.text_embeddings = nn.Embedding(N_TEXT_VOCAB, backbone_dim)
        self.audio_embeddings = nn.Embedding(N_AUDIO_VOCAB * N_AUDIO_CODEBOOKS, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, N_AUDIO_VOCAB, bias=False)
        self.audio_head = mx.zeros((N_AUDIO_CODEBOOKS - 1, decoder_dim, N_AUDIO_VOCAB))

        # The Llama stacks are used as pure transformers over precomputed embeddings.
        self.backbone.embed_tokens = nn.Identity()
        self.decoder.embed_tokens = nn.Identity()

        # Swap in the torchtune-convention attention.
        for layer in self.backbone.layers:
            layer.self_attn = Attention(bb)
        for layer in self.decoder.layers:
            layer.self_attn = Attention(dec)

    def embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        return self.audio_embeddings(tokens + codebook * self.n_audio_vocab)

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        text_embeds = mx.expand_dims(self.text_embeddings(tokens[:, :, -1]), axis=-2)
        audio_tokens = tokens[:, :, :-1] + (self.n_audio_vocab * mx.arange(self.n_audio_codebooks))
        audio_embeds = self.audio_embeddings(audio_tokens.flatten()).reshape(
            (*tokens.shape[:2], self.n_audio_codebooks, -1)
        )
        return mx.concat([audio_embeds, text_embeds], axis=-2)


# ----------------------------------------------------------------------------
# Weight loading: torchtune safetensors -> MLX
# ----------------------------------------------------------------------------
_RENAMES = [
    (".attn.q_proj.", ".self_attn.q_proj."),
    (".attn.k_proj.", ".self_attn.k_proj."),
    (".attn.v_proj.", ".self_attn.v_proj."),
    (".attn.output_proj.", ".self_attn.o_proj."),
    (".mlp.w1.", ".mlp.gate_proj."),
    (".mlp.w3.", ".mlp.up_proj."),
    (".mlp.w2.", ".mlp.down_proj."),
    (".sa_norm.scale", ".input_layernorm.weight"),
    (".mlp_norm.scale", ".post_attention_layernorm.weight"),
]


def remap_key(k: str) -> str:
    for a, b in _RENAMES:
        if a in k:
            k = k.replace(a, b)
    # final norms: backbone.norm.scale / decoder.norm.scale -> ...norm.weight
    if k.endswith(".norm.scale"):
        k = k[: -len(".norm.scale")] + ".norm.weight"
    return k


def _mixed_predicate(group_size: int):
    """backbone Linears -> 4-bit (the bandwidth hog); decoder + heads -> 8-bit;
    embeddings left full precision (quantizing the lookup tables injects per-token
    noise that flips argmax and wrecks the c1/decoder path)."""

    def pred(path: str, module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if "embeddings" in path:
            return False
        if path.startswith("backbone"):
            return {"group_size": group_size, "bits": 4}
        return {"group_size": group_size, "bits": 8}

    return pred


def load_misotts_mlx(
    safetensors_path: str,
    dtype=mx.float32,
    bits: Optional[int] = None,
    group_size: int = 64,
    quant: Optional[str] = None,
) -> MisoTTSMLX:
    """Load the torch checkpoint into the MLX model.

    quant=None & bits=None -> full precision (`dtype`).
    bits in {8, 4}         -> uniform quantization of all Linear/Embedding layers.
    quant="mixed"          -> backbone 4-bit, decoder/heads 8-bit, embeddings fp
                              (best quality/speed trade-off for this RVQ model).
    audio_head (a raw array) is never quantized. Quantization slashes the per-token
    weight-memory traffic that dominates this bandwidth-bound autoregressive loop.
    """
    from safetensors import safe_open

    model = MisoTTSMLX()
    mapped: Dict[str, mx.array] = {}
    with safe_open(safetensors_path, framework="numpy") as f:
        for k in f.keys():
            mapped[remap_key(k)] = mx.array(f.get_tensor(k)).astype(dtype)

    model.load_weights(list(mapped.items()), strict=True)

    if quant == "mixed":
        nn.quantize(model, group_size=group_size, bits=4, class_predicate=_mixed_predicate(group_size))
    elif bits is not None:
        nn.quantize(model, group_size=group_size, bits=bits)

    model.eval()
    mx.eval(model.parameters())
    return model


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python misotts_mlx.py <path-to-model.safetensors>")
        raise SystemExit(1)
    m = load_misotts_mlx(path)
    from mlx.utils import tree_flatten

    n = len(tree_flatten(m.parameters()))
    print(f"[misotts_mlx] loaded OK -- {n} parameter tensors")
