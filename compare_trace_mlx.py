#!/usr/bin/env python3
"""
compare_trace_mlx.py  --  deep teacher-forced logit comparison: MLX vs CUDA reference.

For each prompt trace produced by capture_trace.py, this feeds the CUDA-sampled token
sequence through the MLX model (teacher forcing: the model never samples, it consumes the
recorded codes), and compares the MLX codebook-0 / codebook-1 logits against the recorded
CUDA logits AT EVERY FRAME. Because the inputs are identical and logits are deterministic,
a correct port should match within fp tolerance over the entire utterance — including the
long (>=30s, hundreds of frames) prompts that exercise the KV-cache deeply.

Usage:
  python compare_trace_mlx.py --traces ref_trace                 # fp32
  python compare_trace_mlx.py --traces ref_trace --bits 8        # Q8
  python compare_trace_mlx.py --traces ref_trace --quant mixed   # mixed-Q4
"""
import argparse
import glob
import os

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from misotts_mlx import load_misotts_mlx


def find_model():
    env = os.environ.get("MISO_TTS_8B_MODEL")
    if env and os.path.isfile(env):
        return env
    return glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--MisoLabs--MisoTTS/snapshots/*/model.safetensors"))[0]


def cos_rows(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="ref_trace")
    ap.add_argument("--bits", type=int, default=None, choices=[4, 8])
    ap.add_argument("--quant", default=None, choices=["mixed"])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    tag = args.quant or (f"q{args.bits}" if args.bits else "fp32")
    model = load_misotts_mlx(args.model or find_model(), dtype=mx.float32,
                             bits=args.bits, group_size=args.group_size, quant=args.quant)
    n_cb = model.n_audio_codebooks
    dirs = sorted(d for d in glob.glob(os.path.join(args.traces, "*")) if os.path.isdir(d))
    print(f"[trace-cmp] precision={tag}  prompts={len(dirs)}\n")

    agg = {"c0_cos": [], "c1_cos": [], "c0_top1": [], "c1_top1": []}
    print(f"{'prompt':22} {'frames':>6}  {'c0 cos(mean/min)':>20}  {'c0 top1':>8}  "
          f"{'c1 cos(mean/min)':>20}  {'c1 top1':>8}")
    print("-" * 96)

    for d in dirs:
        toks = np.load(os.path.join(d, "input_tokens.npy")).astype(np.int32)   # (1,S,33)
        codes = np.load(os.path.join(d, "frame_codes.npy")).astype(np.int32)    # (T,32)
        ref_c0 = np.load(os.path.join(d, "c0_logits.npy"))                      # (T,2051)
        ref_c1 = np.load(os.path.join(d, "c1_logits.npy"))                      # (T,2051)
        T = codes.shape[0]

        bb_cache = make_prompt_cache(model.backbone)
        # frame 0 input = the text prompt (mask: text column only)
        curr = mx.array(toks)
        S = toks.shape[1]
        mask = np.zeros((1, S, n_cb + 1), np.float32); mask[:, :, -1] = 1.0
        curr_mask = mx.array(mask)
        audio_mask = mx.array(np.concatenate([np.ones((1, 1, n_cb), np.float32),
                                              np.zeros((1, 1, 1), np.float32)], axis=2))

        mlx_c0 = np.empty((T, model.n_audio_vocab), np.float32)
        mlx_c1 = np.empty((T, model.n_audio_vocab), np.float32)
        for t in range(T):
            embeds = model.embed_tokens(curr) * mx.expand_dims(curr_mask, -1)
            last_h = model.backbone(embeds.sum(axis=-2), cache=bb_cache)[:, -1, :]
            c0l = model.codebook0_head(last_h)
            # decoder c1 using the CUDA-recorded c0 sample at this frame
            c0_tok = mx.array([[int(codes[t, 0])]], dtype=mx.int32)
            dec_in = mx.concat([mx.expand_dims(last_h, 1), model.embed_audio(0, c0_tok)], axis=1)
            dh = model.decoder(model.projection(dec_in), cache=make_prompt_cache(model.decoder))
            c1l = mx.matmul(dh[:, -1, :], model.audio_head[0])
            mx.eval(c0l, c1l)
            mlx_c0[t] = np.array(c0l)[0]
            mlx_c1[t] = np.array(c1l)[0]
            # teacher-force next backbone input = the CUDA-sampled codes for this frame
            nxt = np.zeros((1, 1, n_cb + 1), np.int32)
            nxt[0, 0, :n_cb] = codes[t]
            curr = mx.array(nxt)
            curr_mask = audio_mask

        c0_cos = cos_rows(mlx_c0, ref_c0)
        c1_cos = cos_rows(mlx_c1, ref_c1)
        c0_top1 = np.mean(np.argmax(mlx_c0, 1) == np.argmax(ref_c0, 1)) * 100
        c1_top1 = np.mean(np.argmax(mlx_c1, 1) == np.argmax(ref_c1, 1)) * 100
        agg["c0_cos"].append(c0_cos); agg["c1_cos"].append(c1_cos)
        agg["c0_top1"].append(c0_top1); agg["c1_top1"].append(c1_top1)
        name = os.path.basename(d)
        print(f"{name:22} {T:6d}  {c0_cos.mean():9.5f}/{c0_cos.min():8.5f}  {c0_top1:6.1f}%  "
              f"{c1_cos.mean():9.5f}/{c1_cos.min():8.5f}  {c1_top1:6.1f}%")

    print("-" * 96)
    allc0 = np.concatenate(agg["c0_cos"]); allc1 = np.concatenate(agg["c1_cos"])
    print(f"{'AGGREGATE':22} {len(allc0):6d}  {allc0.mean():9.5f}/{allc0.min():8.5f}  "
          f"{np.mean(agg['c0_top1']):6.1f}%  {allc1.mean():9.5f}/{allc1.min():8.5f}  "
          f"{np.mean(agg['c1_top1']):6.1f}%")
    print(f"\nPass guide: cos > 0.999 clean; 0.99-0.999 acceptable (bf16-vs-fp32); "
          f"top1 should be ~100% for fp32/Q8.")


if __name__ == "__main__":
    main()
