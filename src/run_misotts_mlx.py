#!/usr/bin/env python3
"""
run_misotts_mlx.py  --  end-to-end MisoTTS generation on Apple Silicon via MLX.

Pipeline (all on the Metal GPU except text tokenization):
  text --(Llama-3.2 tokenizer, ungated unsloth mirror)--> tokens
  tokens --(MLX MisoTTS: backbone KV-cache loop + per-frame decoder)--> Mimi codes
  codes --(mlx-audio Mimi, kyutai checkpoint)--> 24 kHz waveform

Reports the real-time factor (RTF) = compute-seconds / audio-seconds. The fp32/MPS
torch baseline was ~21x; this is the number the MLX port aims to beat (and that
Phase-3 quantization should push below 1.0).

The SilentCipher watermark is applied by default (CPU/torch, ~0.7s, imperceptible);
pass --no-watermark to disable it for local debugging only. See MisoTTS terms.

Usage:
  python run_misotts_mlx.py --text "Hey! This is running on MLX." --ms 6000
  python run_misotts_mlx.py --bits 8               # lossless, ~2x slower than real-time
  # voice cloning: speak in a reference voice (pass its transcript for best results)
  python run_misotts_mlx.py --ref-audio jane.wav --ref-text "what jane says in jane.wav" \
      --text "This line is spoken in Jane's voice."
"""
import argparse
import glob
import os
import time

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from misotts_mlx import load_misotts_mlx


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
        raise FileNotFoundError("model.safetensors not in HF cache; set MISO_TTS_8B_MODEL")
    return hits[0]


def load_text_tokenizer():
    from transformers import AutoTokenizer
    from tokenizers.processors import TemplateProcessing

    # Ungated mirror of the Llama-3.2 tokenizer (identical vocab to meta-llama).
    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    bos, eos = tok.bos_token, tok.eos_token
    tok._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, tok.bos_token_id), (eos, tok.eos_token_id)],
    )
    return tok


def tokenize_text(tok, text: str, speaker: int):
    # MisoTTS framing: "[<speaker>] <text>" (note the space), BOS/EOS via post-processor.
    ids = tok.encode(f"[{speaker}] {text.lstrip()}")
    S = len(ids)
    frame = np.zeros((S, 33), dtype=np.int32)
    mask = np.zeros((S, 33), dtype=np.float32)
    frame[:, -1] = ids
    mask[:, -1] = 1.0
    return mx.array(frame), mx.array(mask)


def load_ref_audio(path: str) -> mx.array:
    """Load a reference clip as mono 24 kHz (the rate Mimi expects)."""
    import soundfile as sf
    a, sr0 = sf.read(path)
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a.astype(np.float32)
    if sr0 != 24000:
        import torch, torchaudio
        a = torchaudio.functional.resample(torch.from_numpy(a), sr0, 24000).numpy()
    return mx.array(a)


def audio_context_frames(mimi, audio_24k: mx.array):
    """Mimi-encode a reference clip into context frames (audio cols set, + EOS frame).
    Mirrors the torch tokenize_audio so the cloned voice conditions exactly as on CUDA."""
    codes = np.array(mimi.encode(audio_24k[None, None])[0])        # (32, T)
    codes = np.concatenate([codes, np.zeros((32, 1))], axis=1)     # + EOS column
    T1 = codes.shape[1]
    frame = np.zeros((T1, 33), dtype=np.int32)
    mask = np.zeros((T1, 33), dtype=np.float32)
    frame[:, :32] = codes.T
    mask[:, :32] = 1.0
    return mx.array(frame), mx.array(mask)


def sample_topk(logits: mx.array, topk: int, temp: float) -> mx.array:
    """Mirror MisoTTS sample_topk: temperature, top-k mask, multinomial."""
    if temp <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temp
    vals = mx.topk(logits, topk, axis=-1)
    kth = vals.min(axis=-1, keepdims=True)
    neg_inf = mx.array(-float("inf"), dtype=logits.dtype)
    masked = mx.where(logits < kth, neg_inf, logits)
    return mx.random.categorical(masked, axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hey! I can't believe this is running locally on my Mac.")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--ref-audio", default=None,
                    help="reference voice clip to clone (any wav; resampled to 24kHz)")
    ap.add_argument("--ref-text", default="",
                    help="transcript of --ref-audio (recommended for best cloning)")
    ap.add_argument("--ref-speaker", type=int, default=None,
                    help="speaker id for the reference segment (default: --speaker)")
    ap.add_argument("--ms", type=int, default=6000, help="max audio length in ms")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eos-every", type=int, default=8,
                    help="check EOS every N frames (1 = every frame; higher = less sync overhead)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--bits", type=int, default=None, choices=[4, 8], help="uniform quantize to N bits")
    ap.add_argument("--quant", default=None, choices=["mixed"], help="mixed: backbone Q4, decoder/heads Q8")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--compiled-decoder", action="store_true",
                    help="use the mx.compile fixed-cache depth decoder")
    ap.add_argument("--no-watermark", dest="watermark", action="store_false",
                    help="DISABLE the SilentCipher watermark (default: on, per MisoTTS terms)")
    ap.set_defaults(watermark=True)
    ap.add_argument("--out", default="mlx_out.wav")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[args.dtype]
    mx.random.seed(args.seed)
    model_path = args.model or find_model()
    print(f"[mlx] device={mx.default_device()} dtype={args.dtype} model={model_path}")

    t = time.time()
    model = load_misotts_mlx(model_path, dtype=dtype, bits=args.bits,
                             group_size=args.group_size, quant=args.quant)
    print(f"[mlx] model loaded in {time.time() - t:.1f}s (quant={args.quant or args.bits or args.dtype})")

    from mlx_audio.codec.models.mimi.mimi import Mimi

    t = time.time()
    mimi = Mimi.from_pretrained("kyutai/moshiko-pytorch-bf16")
    tok = load_text_tokenizer()
    print(f"[mlx] mimi + tokenizer loaded in {time.time() - t:.1f}s")

    # Build the prompt. With --ref-audio, prepend a voice-clone context:
    #   [ref-text | ref-audio (Mimi codes) | new-text]  -> the model speaks `text` in the
    # reference voice. Without it, just the text (a fresh, random voice).
    text_frame, text_mask = tokenize_text(tok, args.text, args.speaker)
    if args.ref_audio:
        ref_spk = args.ref_speaker if args.ref_speaker is not None else args.speaker
        rtf, rtm = tokenize_text(tok, args.ref_text, ref_spk)
        raf, ram = audio_context_frames(mimi, load_ref_audio(args.ref_audio))
        text_frame = mx.concat([rtf, raf, text_frame], axis=0)
        text_mask = mx.concat([rtm, ram, text_mask], axis=0)
        print(f"[mlx] voice-cloning from {args.ref_audio} "
              f"(ref {raf.shape[0]} frames, speaker {ref_spk})")
    curr_tokens = mx.expand_dims(text_frame, 0).astype(mx.int32)   # (1, S, 33)
    curr_mask = mx.expand_dims(text_mask, 0)                       # (1, S, 33)

    max_frames = int(args.ms / 80)
    n_cb = model.n_audio_codebooks
    backbone_cache = make_prompt_cache(model.backbone)
    samples = []

    decode_frame = None
    if args.compiled_decoder:
        from mlx_fast_decoder import build_compiled_decoder
        decode_frame = build_compiled_decoder(model)

    def topk_sampler(logits):
        return sample_topk(logits, args.topk, args.temp)

    # Next-frame token mask is constant (audio cols on, text col off) -- build once.
    next_mask = mx.expand_dims(mx.concat([mx.ones((1, n_cb)), mx.zeros((1, 1))], axis=1), 1)
    eos_every = max(1, args.eos_every)
    stop = False

    t_gen = time.time()
    for step in range(max_frames):
        embeds = model.embed_tokens(curr_tokens) * mx.expand_dims(curr_mask, -1)
        h = embeds.sum(axis=-2)
        backbone_hidden = model.backbone(h, cache=backbone_cache)
        last_h = backbone_hidden[:, -1, :]

        c0 = sample_topk(model.codebook0_head(last_h), args.topk, args.temp).reshape(1, 1).astype(mx.int32)
        if decode_frame is not None:
            frame = decode_frame(last_h, c0, topk_sampler)        # (1, n_cb)
        else:
            codes = [c0]
            dec_in = mx.concat([mx.expand_dims(last_h, 1), model.embed_audio(0, c0)], axis=1)
            dec_cache = make_prompt_cache(model.decoder)
            for i in range(1, n_cb):
                dh = model.decoder(model.projection(dec_in), cache=dec_cache)
                ci_logits = mx.matmul(dh[:, -1, :], model.audio_head[i - 1])
                ci = sample_topk(ci_logits, args.topk, args.temp).reshape(1, 1).astype(mx.int32)
                codes.append(ci)
                dec_in = model.embed_audio(i, ci)
            frame = mx.concat(codes, axis=1)   # (1, n_cb)
        samples.append(frame)
        curr_tokens = mx.expand_dims(mx.concat([frame, mx.zeros((1, 1), dtype=mx.int32)], axis=1), 1)
        curr_mask = next_mask
        # Pipeline: kick off this frame's compute (and bound the lazy graph) WITHOUT a
        # blocking CPU readback -- the GPU runs ahead while Python builds the next frame.
        mx.async_eval(curr_tokens)

        # EOS check only every N frames -> one sync per N frames instead of per frame.
        if (step + 1) % eos_every == 0:
            recent = samples[-eos_every:]
            mx.eval(recent)
            for j, fr in enumerate(recent):
                if not bool(mx.any(fr).item()):
                    samples = samples[: len(samples) - eos_every + j]
                    stop = True
                    break
            if stop:
                break
    if samples:
        mx.eval(samples)
    gen_secs = time.time() - t_gen

    if not samples:
        raise RuntimeError("no frames generated (immediate EOS)")

    t_dec = time.time()
    codes_all = mx.stack(samples).transpose(1, 2, 0)   # (1, n_cb, T)
    audio = mimi.decode(codes_all)
    mx.eval(audio)
    dec_secs = time.time() - t_dec

    audio_np = np.array(audio).squeeze().astype(np.float32)
    n_frames = len(samples)
    sr = int(mimi.sample_rate)
    audio_secs = audio_np.shape[-1] / sr   # measured on the model output (pre-watermark)

    # Re-attach the SilentCipher watermark (CPU/torch, separable from the MLX model).
    # On by default per MisoTTS terms; identifies the audio as AI-generated.
    wm_status = "DISABLED (--no-watermark)"
    if args.watermark:
        import torch
        from watermarking import MISO_TTS_WATERMARK, load_watermarker, watermark as apply_watermark
        tw = time.time()
        watermarker = load_watermarker(device="cpu")
        wm_audio, sr = apply_watermark(watermarker, torch.from_numpy(audio_np), sr, MISO_TTS_WATERMARK)
        audio_np = wm_audio.detach().cpu().numpy().astype(np.float32)
        wm_status = f"SilentCipher applied in {time.time() - tw:.1f}s (sr={sr})"

    import soundfile as sf
    sf.write(args.out, audio_np, sr)

    total = gen_secs + dec_secs
    rtf = total / audio_secs if audio_secs > 0 else float("nan")
    print(f"[mlx] frames={n_frames}  gen={gen_secs:.1f}s  mimi_decode={dec_secs:.1f}s")
    print(f"[mlx] generated {audio_secs:.2f}s of audio in {total:.1f}s  ->  RTF={rtf:.2f}x "
          f"({'faster than real-time' if rtf < 1 else 'slower than real-time'})")
    print(f"[mlx] per-frame: {gen_secs / n_frames * 1000:.0f} ms  (target frame period = 80 ms)")
    print(f"[mlx] watermark: {wm_status}")
    print(f"[mlx] saved {args.out}")


if __name__ == "__main__":
    main()
