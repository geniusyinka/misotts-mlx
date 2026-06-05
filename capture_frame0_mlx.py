#!/usr/bin/env python3
"""
capture_frame0_mlx.py  --  MLX candidate capture for the MisoTTS port.

Reproduces the frame-0 forward pass (mirrors capture_frame0.py's torch path) using
the MLX model in misotts_mlx.py, and writes the same tensors to a candidate dir so
compare_frame0.py can diff it against the CUDA reference.

It FEEDS the reference's saved input_tokens.npy directly, so it needs no tokenizer
(sidesteps the gated meta-llama repo entirely). The reference prompt is text-only,
so the token mask is reconstructed as "text column on, audio columns off".

Usage:
  python capture_frame0_mlx.py --ref reference --out candidate_mlx \
      [--model /path/to/model.safetensors]
  python compare_frame0.py reference candidate_mlx
"""
import argparse
import glob
import os

import numpy as np

import mlx.core as mx

from misotts_mlx import load_misotts_mlx, N_AUDIO_CODEBOOKS


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32), copy=True)


def find_model() -> str:
    env = os.environ.get("MISO_TTS_8B_MODEL")
    if env and os.path.isfile(env):
        return env
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--MisoLabs--MisoTTS/snapshots/*/model.safetensors"
        )
    )
    if not hits:
        raise FileNotFoundError(
            "model.safetensors not found in HF cache; set MISO_TTS_8B_MODEL"
        )
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="reference", help="reference dir (for input_tokens.npy)")
    ap.add_argument("--out", default="candidate_mlx")
    ap.add_argument("--model", default=None)
    ap.add_argument("--bits", type=int, default=None, choices=[4, 8], help="uniform quantize to N bits")
    ap.add_argument("--quant", default=None, choices=["mixed"], help="mixed: backbone Q4, decoder/heads Q8")
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_path = args.model or find_model()
    print(f"[mlx] model={model_path}")

    # Same input tokens the reference consumed: (1, S, n_codebooks+1)
    tokens_np = np.load(os.path.join(args.ref, "input_tokens.npy")).astype(np.int32)
    assert tokens_np.shape[-1] == N_AUDIO_CODEBOOKS + 1, tokens_np.shape
    # Reference prompt is text-only: audio columns must be zero.
    assert np.all(tokens_np[:, :, :-1] == 0), "audio columns not zero -- prompt isn't text-only"
    S = tokens_np.shape[1]

    tokens = mx.array(tokens_np)  # int32
    mask = np.zeros(tokens_np.shape, dtype=np.float32)
    mask[:, :, -1] = 1.0  # text column on
    mask = mx.array(mask)

    model = load_misotts_mlx(model_path, dtype=mx.float32, bits=args.bits,
                             group_size=args.group_size, quant=args.quant)
    print(f"[mlx] model loaded (seq_len={S}, quant={args.quant or args.bits or 'fp32'})")

    # ---- frame-0 forward (mirrors capture_frame0.py) ----
    embeds = model.embed_tokens(tokens)            # (1, S, 33, 4096)
    masked = embeds * mx.expand_dims(mask, -1)
    h_in = masked.sum(axis=-2)                     # (1, S, 4096)

    backbone_hidden = model.backbone(h_in, cache=None)   # (1, S, 4096)
    last_h = backbone_hidden[:, -1, :]                   # (1, 4096)
    c0_logits = model.codebook0_head(last_h)             # (1, 2051)

    # Deterministic c0 (argmax) so the decoder step is reproducible.
    c0_tok = mx.argmax(c0_logits, axis=-1).reshape(1, 1).astype(mx.int32)
    c0_embed = model.embed_audio(0, c0_tok)              # (1, 1, 4096)
    dec_in = mx.concat([mx.expand_dims(last_h, 1), c0_embed], axis=1)  # (1, 2, 4096)
    dec_hidden = model.decoder(model.projection(dec_in), cache=None)   # (1, 2, 1536)
    c1_logits = mx.matmul(dec_hidden[:, -1, :], model.audio_head[0])   # (1, 2051)

    mx.eval(last_h, c0_logits, c1_logits, c0_tok)

    def save(name, arr):
        np.save(os.path.join(args.out, name + ".npy"), arr)

    save("input_tokens", tokens_np)
    save("last_h", _np(last_h))
    save("c0_logits", _np(c0_logits))
    save("c1_logits", _np(c1_logits))
    save("c0_argmax_token", np.array(_np(c0_tok), dtype=np.int32))
    c0_np = _np(c0_logits)
    save("c0_top5", np.argsort(-c0_np, axis=-1)[:, :5])

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"backend=mlx\ndtype=float32\nseq_len={S}\n")

    top5 = np.argsort(-c0_np, axis=-1)[:, :5].tolist()
    print(f"[mlx] frame-0 tensors saved to ./{args.out}/")
    print(f"[mlx] c0 argmax token = {int(np.argmax(c0_np))}  top-5 = {top5}")


if __name__ == "__main__":
    main()
