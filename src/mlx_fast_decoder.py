#!/usr/bin/env python3
"""
mlx_fast_decoder.py  --  mx.compile-friendly depth decoder for MisoTTS-MLX.

The depth decoder runs 31 sequential single-token passes per frame; profiling showed
it's ~79% of frame time, dominated by GPU kernel-launch overhead (8 layers x ~7 ops x
31 steps ~= 1700 tiny launches). This module replaces the per-step mlx_lm LlamaModel
call (whose growing KV cache has a variable shape -> uncompilable) with a hand-written
step that uses a FIXED-size (depth=32) cache and writes at a TRACED position, so
mx.compile traces it once and reuses it for every step -> the 8 layers fuse into far
fewer kernels.

It reuses the loaded decoder's (possibly quantized) submodules for all projections /
MLP / norms, so weight precision/bandwidth are unchanged -- only launch overhead drops.

build_compiled_decoder(model) -> decode_frame(last_h, c0_sample, sampler) -> (1, 32) codes.
"""
from typing import Callable

import mlx.core as mx


def build_compiled_decoder(model, max_depth: int = 32, compile_step: bool = True):
    dec = model.decoder
    layers = dec.layers
    n_layers = len(layers)
    a0 = layers[0].self_attn
    n_h, n_kv, hd, scale = a0.n_heads, a0.n_kv_heads, a0.head_dim, a0.scale
    rep = n_h // n_kv
    rope_cache = a0.rope._cache            # (max_seq, hd//2, 2), shared across layers
    proj = model.projection
    norm = dec.norm
    idx = mx.arange(max_depth)
    NEG = mx.array(-1e9, dtype=mx.float32)
    ZERO = mx.array(0.0, dtype=mx.float32)

    def apply_rope_at(x, pos):
        # x: (1, 1, n, hd); pos: traced scalar int. Gather the single rope row.
        rc = rope_cache[pos].reshape(1, 1, 1, hd // 2, 2)         # (.,hd//2,2)
        xs = x.astype(mx.float32).reshape(1, 1, x.shape[2], hd // 2, 2)
        o0 = xs[..., 0] * rc[..., 0] - xs[..., 1] * rc[..., 1]
        o1 = xs[..., 1] * rc[..., 0] + xs[..., 0] * rc[..., 1]
        return mx.stack([o0, o1], -1).reshape(1, 1, x.shape[2], hd).astype(x.dtype)

    def step(x4096, k_bufs, v_bufs, pos):
        h = proj(x4096)                                          # (1,1,1536)
        write = (idx == pos).reshape(1, 1, max_depth, 1)         # one-hot depth slot
        amask = mx.where(idx <= pos, ZERO, NEG).reshape(1, 1, 1, max_depth)
        nk, nv = [], []
        for li in range(n_layers):
            L = layers[li]
            at = L.self_attn
            normed = L.input_layernorm(h)
            q = apply_rope_at(at.q_proj(normed).reshape(1, 1, n_h, hd), pos)
            k = apply_rope_at(at.k_proj(normed).reshape(1, 1, n_kv, hd), pos)
            v = at.v_proj(normed).reshape(1, 1, n_kv, hd)
            q = q.swapaxes(1, 2)                                 # (1,n_h,1,hd)
            k = k.swapaxes(1, 2)                                 # (1,n_kv,1,hd)
            v = v.swapaxes(1, 2)                                 # (1,n_kv,1,hd)
            # write k,v into the fixed buffer at depth `pos` (one-hot, fixed shape):
            # k*write keeps k only in the pos slot; buf*(1-write) clears that slot.
            Kb = k_bufs[li] * (1 - write) + k * write            # (1,n_kv,max,hd)
            Vb = v_bufs[li] * (1 - write) + v * write
            Kr = mx.repeat(Kb, rep, axis=1)                      # (1,n_h,max,hd)
            Vr = mx.repeat(Vb, rep, axis=1)
            scores = (q @ Kr.swapaxes(-1, -2)) * scale + amask   # (1,n_h,1,max)
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            o = (w @ Vr).swapaxes(1, 2).reshape(1, 1, n_h * hd)
            h = h + at.o_proj(o)
            h = h + L.mlp(L.post_attention_layernorm(h))
            nk.append(Kb)
            nv.append(Vb)
        return norm(h), nk, nv

    step_fn = mx.compile(step) if compile_step else step

    def decode_frame(last_h, c0_sample, sampler: Callable[[mx.array], mx.array]) -> mx.array:
        k_bufs = [mx.zeros((1, n_kv, max_depth, hd)) for _ in range(n_layers)]
        v_bufs = [mx.zeros((1, n_kv, max_depth, hd)) for _ in range(n_layers)]
        # pos 0: backbone hidden state (no output consumed)
        _, k_bufs, v_bufs = step_fn(mx.expand_dims(last_h, 1), k_bufs, v_bufs, mx.array(0, dtype=mx.int32))
        codes = [c0_sample]
        cur = model.embed_audio(0, c0_sample)                    # (1,1,4096)
        for i in range(1, model.n_audio_codebooks):
            hidden, k_bufs, v_bufs = step_fn(cur, k_bufs, v_bufs, mx.array(i, dtype=mx.int32))
            ci_logits = mx.matmul(hidden[:, -1, :], model.audio_head[i - 1])
            ci = sampler(ci_logits).reshape(1, 1).astype(mx.int32)
            codes.append(ci)
            cur = model.embed_audio(i, ci)
        return mx.concat(codes, axis=1)

    return decode_frame


# ----------------------------------------------------------------------------
# Self-test: compiled decoder must reproduce the naive mlx_lm decoder codes (argmax).
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    import os
    import time

    from mlx_lm.models.cache import make_prompt_cache
    from misotts_mlx import load_misotts_mlx

    mp = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--MisoLabs--MisoTTS/snapshots/*/model.safetensors"))[0]
    quant = os.environ.get("MISO_QUANT", "mixed")
    model = load_misotts_mlx(mp, quant=(quant if quant != "none" else None))
    n_cb = model.n_audio_codebooks
    argmax = lambda l: mx.argmax(l, axis=-1)

    # a deterministic backbone hidden + c0
    mx.random.seed(0)
    last_h = mx.random.normal((1, 4096))
    c0 = mx.array([[42]], dtype=mx.int32)

    # naive decoder (reference)
    def naive(last_h, c0):
        dec_in = mx.concat([mx.expand_dims(last_h, 1), model.embed_audio(0, c0)], axis=1)
        dc = make_prompt_cache(model.decoder)
        codes = [c0]
        for i in range(1, n_cb):
            dh = model.decoder(model.projection(dec_in), cache=dc)
            ci = mx.argmax(mx.matmul(dh[:, -1, :], model.audio_head[i - 1]), axis=-1).reshape(1, 1).astype(mx.int32)
            codes.append(ci)
            dec_in = model.embed_audio(i, ci)
        return mx.concat(codes, axis=1)

    ref = naive(last_h, c0); mx.eval(ref)
    decode_frame = build_compiled_decoder(model, compile_step=True)
    fast = decode_frame(last_h, c0, argmax); mx.eval(fast)

    match = bool(mx.all(ref == fast).item())
    print(f"[selftest] quant={quant}  codes match (naive vs compiled): {match}")
    if not match:
        print("  ref :", ref.tolist())
        print("  fast:", fast.tolist())

    # timing
    for _ in range(3):
        mx.eval(decode_frame(last_h, c0, argmax))
        mx.eval(naive(last_h, c0))
    N = 10
    t = time.time()
    for _ in range(N):
        mx.eval(naive(last_h, c0))
    t_naive = (time.time() - t) / N * 1000
    t = time.time()
    for _ in range(N):
        mx.eval(decode_frame(last_h, c0, argmax))
    t_fast = (time.time() - t) / N * 1000
    print(f"[selftest] decoder/frame: naive={t_naive:.1f}ms  compiled={t_fast:.1f}ms  speedup={t_naive/t_fast:.2f}x")
