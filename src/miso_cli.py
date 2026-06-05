#!/usr/bin/env python3
"""
miso_cli.py  --  interactive terminal wizard for MisoTTS on Apple Silicon (MLX).

Launch it (see the `miso-mlx` wrapper) and answer a few prompts -- arrow-key selects
for mode / speaker / length, text boxes for the script and filename. The model loads
ONCE, so you can generate clip after clip in one session (~7s each) without reloading.

  miso-mlx                # interactive
  python miso_cli.py --selftest   # non-interactive smoke test (loads + generates once)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from misotts_mlx import load_misotts_mlx
from run_misotts_mlx import (
    audio_context_frames,
    find_model,
    load_ref_audio,
    load_text_tokenizer,
    sample_topk,
    tokenize_text,
)

MODES = {
    "q8": ("Q8 — lossless, ~2x slower than real-time (recommended)", 8, None),
    "fp32": ("fp32 — cleanest, slowest (the fidelity baseline)", None, None),
    "mixed": ("mixed-Q4 — fastest, slight quality tradeoff", None, "mixed"),
}


def _clean_path(p: str) -> str:
    """Forgiving path input: strip whitespace/surrounding quotes, unescape shell-style
    '\\ ' spaces (from pasted/dragged paths), and expand ~."""
    if not p:
        return ""
    p = p.strip().strip('"').strip("'").strip()
    p = p.replace("\\ ", " ")
    return os.path.expanduser(p)


def load(mode: str):
    _, bits, quant = MODES[mode]
    print(f"\n  loading the {mode} model (~60s on first use)…", flush=True)
    t = time.time()
    model = load_misotts_mlx(find_model(), dtype=mx.float32, bits=bits, quant=quant)
    from mlx_audio.codec.models.mimi.mimi import Mimi

    mimi = Mimi.from_pretrained("kyutai/moshiko-pytorch-bf16")
    tok = load_text_tokenizer()
    from watermarking import load_watermarker

    wmk = load_watermarker(device="cpu")
    print(f"  model ready in {time.time() - t:.0f}s.\n", flush=True)
    return model, mimi, tok, wmk


def synth(model, mimi, tok, wmk, *, text, speaker, ms, temp=0.9, topk=50,
          ref_audio=None, ref_text="", ref_speaker=None, watermark=True, out="miso_out.wav"):
    import soundfile as sf

    ncb = model.n_audio_codebooks
    tf, tm = tokenize_text(tok, text, speaker)
    if ref_audio:
        rs = ref_speaker if ref_speaker is not None else speaker
        rtf, rtm = tokenize_text(tok, ref_text or "", rs)
        raf, ram = audio_context_frames(mimi, load_ref_audio(ref_audio))
        tf = mx.concat([rtf, raf, tf], axis=0)
        tm = mx.concat([rtm, ram, tm], axis=0)
    curr = mx.expand_dims(tf, 0).astype(mx.int32)
    cmask = mx.expand_dims(tm, 0)
    next_mask = mx.expand_dims(mx.concat([mx.ones((1, ncb)), mx.zeros((1, 1))], axis=1), 1)
    bb = make_prompt_cache(model.backbone)
    samples = []
    t = time.time()
    for step in range(int(ms / 80)):
        e = model.embed_tokens(curr) * mx.expand_dims(cmask, -1)
        last = model.backbone(e.sum(axis=-2), cache=bb)[:, -1, :]
        c0 = sample_topk(model.codebook0_head(last), topk, temp).reshape(1, 1).astype(mx.int32)
        codes = [c0]
        din = mx.concat([mx.expand_dims(last, 1), model.embed_audio(0, c0)], axis=1)
        dc = make_prompt_cache(model.decoder)
        for i in range(1, ncb):
            dh = model.decoder(model.projection(din), cache=dc)
            ci = sample_topk(mx.matmul(dh[:, -1, :], model.audio_head[i - 1]), topk, temp).reshape(1, 1).astype(mx.int32)
            codes.append(ci)
            din = model.embed_audio(i, ci)
        fr = mx.concat(codes, axis=1)
        samples.append(fr)
        mx.async_eval(fr)
        curr = mx.expand_dims(mx.concat([fr, mx.zeros((1, 1), dtype=mx.int32)], axis=1), 1)
        cmask = next_mask
        if (step + 1) % 8 == 0:
            mx.eval(samples[-8:])
            if not bool(mx.any(mx.concatenate(samples[-8:])).item()):
                samples = samples[:-8]
                break
    if not samples:
        raise RuntimeError("no audio generated")
    mx.eval(samples)
    gen = time.time() - t
    audio = np.array(mimi.decode(mx.stack(samples).transpose(1, 2, 0))).squeeze().astype(np.float32)
    sr = int(mimi.sample_rate)
    secs = audio.shape[-1] / sr
    if watermark:
        import torch
        from watermarking import MISO_TTS_WATERMARK, watermark as apply_wm

        a, sr = apply_wm(wmk, torch.from_numpy(audio), sr, MISO_TTS_WATERMARK)
        audio = a.detach().cpu().numpy().astype(np.float32)
    sf.write(out, audio, sr, subtype="PCM_16")  # PCM_16 = plays everywhere (afplay + browsers)
    return {"out": out, "secs": secs, "gen": gen, "rtf": gen / secs if secs else 0.0, "frames": len(samples)}


# ----------------------------------------------------------------------------
def _selftest():
    m = load("q8")
    s = synth(*m, text="Self test of the interactive CLI.", speaker=0, ms=3000, out="/tmp/miso_cli_selftest.wav")
    print(f"[selftest] OK -> {s}")


def main():
    if "--selftest" in sys.argv:
        return _selftest()

    import questionary
    from questionary import Choice, Style

    style = Style([
        ("qmark", "fg:#d98a4f bold"), ("question", "bold"),
        ("answer", "fg:#f0b072 bold"), ("pointer", "fg:#d98a4f bold"),
        ("highlighted", "fg:#f0b072 bold"), ("selected", "fg:#7cffb2"),
    ])

    print("\n  \033[38;5;179mMISO TTS\033[0m · Apple Silicon (MLX) — interactive synth\n")

    def pick_mode():
        return questionary.select(
            "Quality / speed mode",
            choices=[Choice(MODES[k][0], k) for k in ("q8", "fp32", "mixed")],
            style=style,
        ).ask()

    mode = pick_mode()
    if mode is None:
        return
    model, mimi, tok, wmk = load(mode)

    while True:
        text = questionary.text("What should Miso say?", style=style, multiline=False).ask()
        if not text:
            break
        speaker = int(questionary.select("Speaker", choices=["0", "1", "2", "3"], default="0", style=style).ask())
        secs = questionary.select(
            "Max length", choices=["4 sec", "6 sec", "10 sec", "20 sec", "30 sec"], default="6 sec", style=style
        ).ask()
        ms = int(secs.split()[0]) * 1000

        ref_audio = ref_text = None
        if questionary.confirm("Clone a reference voice?", default=False, style=style).ask():
            rp = _clean_path(questionary.path("Reference .wav (Tab to autocomplete)", style=style).ask() or "")
            if rp and os.path.isfile(rp):
                ref_audio = rp
                ref_text = questionary.text(
                    "What is said in that clip? (the transcript text — optional, improves cloning)",
                    style=style,
                ).ask()
            elif rp:
                print(f"  \033[31m! file not found: {rp}\n    → skipping clone, using a default voice.\033[0m")

        out = questionary.text("Save to", default="miso_out.wav", style=style).ask()

        print("\n  synthesizing…", flush=True)
        try:
            s = synth(model, mimi, tok, wmk, text=text, speaker=speaker, ms=ms,
                      ref_audio=ref_audio, ref_text=ref_text, out=out or "miso_out.wav")
        except Exception as exc:  # noqa: BLE001
            print(f"  \033[31m✗ {exc}\033[0m\n")
            continue
        print(f"  \033[38;5;115m✓ {s['secs']:.1f}s audio in {s['gen']:.1f}s "
              f"(RTF {s['rtf']:.2f}) → {s['out']}\033[0m\n")

        if questionary.confirm("Play it now?", default=True, style=style).ask():
            subprocess.run(["afplay", s["out"]], check=False)

        nxt = questionary.select(
            "Next?", choices=["Generate another (same mode)", "Change mode", "Quit"], style=style
        ).ask()
        if nxt is None or nxt == "Quit":
            break
        if nxt == "Change mode":
            new_mode = pick_mode()
            if new_mode is None:
                break
            if new_mode != mode:
                mode = new_mode
                model, mimi, tok, wmk = load(mode)

    print("\n  done. 🍜\n")


if __name__ == "__main__":
    main()
