#!/usr/bin/env python3
"""
capture_frame0.py  --  device-agnostic, RNG-independent capture for porting MisoTTS.

Run this from inside a checkout of the MisoTTS repo (it imports generator.py / models.py).

Why this script exists
----------------------
generate_frame() samples stochastically (temperature/top-k), so sampled tokens will NOT
match across CUDA / MPS / MLX even with the same seed (RNG differs by backend).
LOGITS, however, are a deterministic function of (weights, inputs). So we capture the
frame-0 logits and hidden state. Any correct port must reproduce these within fp tolerance.

Usage
-----
  # On the rented CUDA box (the gold reference):
  python capture_frame0.py --out reference --wav

  # On the Mac (the candidate, MPS quick-win):
  python capture_frame0.py --out candidate --device mps --dtype float32

Then diff them with compare_frame0.py.
"""
import os
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"
# float64 / unsupported ops transparently fall back to CPU on MPS:
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import numpy as np
import torch

from generator import load_miso_8b
from models import _index_causal_mask


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save(out_dir: str, name: str, t: torch.Tensor) -> None:
    np.save(os.path.join(out_dir, name + ".npy"), t.detach().float().cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reference", help="output directory")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--text", default="Hello from Miso. This is a reference capture for porting.")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("MISO_TTS_8B_MODEL", "MisoLabs/MisoTTS"))
    ap.add_argument("--wav", action="store_true", help="also render reference.wav (full generate, seeded)")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    os.makedirs(args.out, exist_ok=True)
    print(f"[capture] device={device} dtype={dtype} model={args.model}")

    gen = load_miso_8b(device, model_path_or_repo_id=args.model, dtype=dtype)
    model = gen._model
    # torchtune's setup_caches() runs inside Generator.__init__ AFTER the model was moved to
    # `device`, and it allocates the KV-cache / cache_pos buffers on CPU. Re-assert the device
    # so the cached attention path doesn't mix cuda/cpu (or mps/cpu) tensors.
    model.to(device)
    mdtype = next(model.parameters()).dtype

    # ---- Deterministic frame-0 forward (mirrors Model.generate_frame up to logits) ----
    model.reset_caches()
    tokens, mask = gen._tokenize_text_segment(args.text, args.speaker)   # (S, 33)
    curr_tokens = tokens.unsqueeze(0)
    curr_mask = mask.unsqueeze(0)
    S = tokens.size(0)
    curr_pos = torch.arange(0, S, device=device).unsqueeze(0)

    bmask = _index_causal_mask(model.backbone_causal_mask, curr_pos)
    embeds = model._embed_tokens(curr_tokens)
    masked = embeds * curr_mask.unsqueeze(-1)
    h = masked.sum(dim=2)
    h = model.backbone(h, input_pos=curr_pos, mask=bmask).to(dtype=mdtype)

    last_h = h[:, -1, :]
    c0_logits = model.codebook0_head(last_h)

    # Fix the codebook-0 token deterministically (argmax) so the decoder step is reproducible.
    c0_tok = c0_logits.float().argmax(dim=-1, keepdim=True).to(torch.int)
    c0_embed = model._embed_audio(0, c0_tok)
    cur_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
    dpos = torch.arange(0, cur_h.size(1), device=device).unsqueeze(0).repeat(cur_h.size(0), 1)
    model.decoder.reset_caches()
    dmask = _index_causal_mask(model.decoder_causal_mask, dpos)
    dh = model.decoder(model.projection(cur_h), input_pos=dpos, mask=dmask).to(dtype=mdtype)
    c1_logits = torch.mm(dh[:, -1, :], model.audio_head[0])

    save(args.out, "input_tokens", curr_tokens.to(torch.int32))
    save(args.out, "input_pos", curr_pos.to(torch.int32))
    save(args.out, "last_h", last_h)
    save(args.out, "c0_logits", c0_logits)
    save(args.out, "c0_argmax_token", c0_tok.to(torch.int32))
    save(args.out, "c1_logits", c1_logits)
    np.save(os.path.join(args.out, "c0_top5.npy"),
            c0_logits.float().topk(5, dim=-1).indices.cpu().numpy())

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"text={args.text}\nspeaker={args.speaker}\ndevice={device}\n"
                f"dtype={mdtype}\nseq_len={S}\nsample_rate={gen.sample_rate}\n")

    print(f"[capture] frame-0 tensors saved to ./{args.out}/  (seq_len={S})")
    print(f"[capture] c0 argmax token = {int(c0_tok.item())}  "
          f"top-5 = {c0_logits.float().topk(5, dim=-1).indices.cpu().numpy().tolist()}")

    # ---- Optional: full audio render for listening (stochastic; seeded) ----
    if args.wav:
        import torchaudio
        torch.manual_seed(0)
        audio = gen.generate(text=args.text, speaker=args.speaker, context=[],
                             max_audio_length_ms=8000, temperature=0.9, topk=50)
        wav_path = os.path.join(args.out, "reference.wav")
        torchaudio.save(wav_path, audio.unsqueeze(0).cpu(), gen.sample_rate)
        print(f"[capture] wrote {wav_path} ({audio.shape[-1] / gen.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
