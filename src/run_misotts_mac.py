#!/usr/bin/env python3
"""
run_misotts_mac.py  --  MisoTTS quick-win runner for Apple Silicon (MPS).

Drop this file into the root of a MisoTTS checkout and run it. No edits to the
repo's own files are required.

What it does differently from run_misotts.py
--------------------------------------------
The shipped run_misotts.py deliberately skips MPS "due to float64 limitations."
We work around that with PYTORCH_ENABLE_MPS_FALLBACK=1, which transparently runs
any unsupported / float64 op on the CPU while the big backbone+decoder matmuls
stay on the GPU. We also default to float32 (safest, most op coverage on MPS;
~32 GB for the 8B model, comfortable on a 64 GB machine). Pass --dtype bfloat16
to try the faster/leaner path once float32 is confirmed working.

Usage
-----
  python run_misotts_mac.py
  python run_misotts_mac.py --text "Whatever you want spoken." --ms 6000
  python run_misotts_mac.py --dtype bfloat16            # leaner, try after fp32 works

It prints the real-time factor (RTF): seconds-of-compute / seconds-of-audio.
RTF < 1.0 means faster than real-time. Expect RTF >> 1 here on fp32/MPS — that's
fine; this phase is about CORRECTNESS, not speed. Speed comes with the MLX port.
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio

from generator import DEFAULT_MISO_TTS_REPO_ID, load_miso_8b


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--text", default="Hey! I can't believe this is running locally on my Mac.")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--ms", type=int, default=6000, help="max audio length in ms")
    ap.add_argument("--out", default="mac_out.wav")
    ap.add_argument("--model", default=os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID))
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"[mac] device={device} dtype={dtype}")
    if device == "mps":
        print("[mac] MPS fallback enabled — unsupported/float64 ops will run on CPU.")

    t0 = time.time()
    gen = load_miso_8b(device, model_path_or_repo_id=args.model, dtype=dtype)
    # torchtune's setup_caches() (run inside Generator.__init__, after model.to(device)) allocates
    # the KV-cache / cache_pos buffers on CPU, which crashes the cached attention path with a
    # mps-vs-cpu (or cuda-vs-cpu) device mismatch. Re-assert the device to move those buffers.
    gen._model.to(device)
    print(f"[mac] model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    audio = gen.generate(
        text=args.text,
        speaker=args.speaker,
        context=[],
        max_audio_length_ms=args.ms,
    )
    dt = time.time() - t1

    secs = audio.shape[-1] / gen.sample_rate
    torchaudio.save(args.out, audio.unsqueeze(0).cpu(), gen.sample_rate)
    rtf = dt / secs if secs > 0 else float("nan")
    print(f"[mac] generated {secs:.2f}s of audio in {dt:.1f}s  ->  RTF={rtf:.2f}x "
          f"({'faster than real-time' if rtf < 1 else 'slower than real-time'})")
    print(f"[mac] saved {args.out}")


if __name__ == "__main__":
    main()
