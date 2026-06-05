#!/usr/bin/env python3
"""
capture_trace.py  --  deep reference capture for the MisoTTS MLX validation.

Run inside a CUDA MisoTTS checkout. For each of 5 varied prompts it:
  * generates seeded audio with the real model,
  * records, AT EVERY FRAME, the sampled 32 codes plus the codebook-0 and
    codebook-1 logits (deterministic functions of the context),
  * writes a watermarked reference .wav.

The recorded (codes, c0_logits, c1_logits) per frame let `compare_trace_mlx.py`
TEACHER-FORCE the MLX model with the identical token sequence and compare logits
frame-by-frame across the whole utterance (not just frame 0) — exercising the
KV-cache path over hundreds of frames, including the two long (>=30s) prompts.

Usage (on the CUDA box):
  HF_TOKEN=hf_xxx python capture_trace.py --out ref_trace
"""
import os
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import numpy as np
import torch

from generator import Generator, load_miso_8b  # noqa: F401
from models import MISO_TTS_8B_CONFIG, Model, _index_causal_mask, sample_topk
from watermarking import MISO_TTS_WATERMARK, load_watermarker, watermark


def load_lowmem(repo_or_path, device):
    """GPU-direct bf16 loader so CPU RAM never holds the 32GB model (the OOM cause).

    Builds the model ON THE GPU in bf16 (~16GB VRAM, the actual inference dtype, real
    RoPE caches), then streams weights from the safetensors file ONE TENSOR AT A TIME
    straight to the GPU (CPU never holds more than a single tensor). The result is the
    bf16 model used for inference -- the canonical reference precision."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if os.path.isfile(repo_or_path):
        mf = repo_or_path
    elif os.path.isdir(repo_or_path):
        mf = os.path.join(repo_or_path, "model.safetensors")
    else:
        mf = hf_hub_download(repo_id=repo_or_path, filename="model.safetensors")

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = Model(MISO_TTS_8B_CONFIG)        # ~16GB VRAM, bf16, real RoPE
    finally:
        torch.set_default_dtype(prev)
    model.eval()
    sd = model.state_dict()
    with safe_open(mf, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k].copy_(f.get_tensor(k).to(device=device, dtype=torch.bfloat16))
    return model

# 5 varied prompts; (id, speaker, max_ms, text). #3 and #5 are the long (>=30s) ones.
PROMPTS = [
    ("p1_conversational", 0, 12_000,
     "So I was telling you about the weekend, right? Honestly it was kind of a disaster, "
     "but in a funny way. We can laugh about it now."),
    ("p2_expressive", 1, 11_000,
     "Wait, you actually remembered? Oh my gosh — this is, this is the best surprise anyone "
     "has ever given me. I genuinely cannot believe it!"),
    ("p3_narration_long", 0, 40_000,
     "The old lighthouse had stood at the edge of the cape for nearly a century, and in all "
     "that time it had never once gone dark. Every evening, just as the sun dipped below the "
     "water, the keeper would climb the spiral stairs, trim the wick, and light the great lamp. "
     "Sailors miles out at sea would see that steady glow and know exactly how far they were "
     "from home. It was, in its quiet way, the most dependable thing for a hundred miles in "
     "any direction."),
    ("p4_technical", 1, 12_000,
     "Step one: preheat the oven to four hundred degrees. Step two: combine the flour, the "
     "sugar, and a pinch of salt. Step three: fold in the butter until the mixture looks like "
     "coarse sand."),
    ("p5_monologue_long", 0, 40_000,
     "You want to know what I really think? Fine. I think we spend our whole lives waiting for "
     "permission — waiting for someone to tell us it's okay to begin. But nobody is coming to "
     "hand you that permission. There is no perfect moment, no final sign, no gentle voice that "
     "says now, go. There is only this: the work in front of you, and the choice to start it "
     "today instead of tomorrow. So start. Start badly if you have to. Just start."),
]


@torch.inference_mode()
def frame_traced(model, tokens, tokens_mask, input_pos, temperature, topk):
    """Mirror Model.generate_frame, additionally returning c0/c1 logits."""
    dtype = next(model.parameters()).dtype
    mask = _index_causal_mask(model.backbone_causal_mask, input_pos)
    embeds = model._embed_tokens(tokens)
    h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)
    h = model.backbone(h, input_pos=input_pos, mask=mask).to(dtype=dtype)
    last_h = h[:, -1, :]

    c0_logits = model.codebook0_head(last_h)
    c0_sample = sample_topk(c0_logits, topk, temperature)
    c0_embed = model._embed_audio(0, c0_sample)

    curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
    curr_sample = c0_sample.clone()
    curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0)
    model.decoder.reset_caches()

    c1_logits = None
    for i in range(1, model.config.audio_num_codebooks):
        dmask = _index_causal_mask(model.decoder_causal_mask, curr_pos)
        dh = model.decoder(model.projection(curr_h), input_pos=curr_pos, mask=dmask).to(dtype=dtype)
        ci_logits = torch.mm(dh[:, -1, :], model.audio_head[i - 1])
        if i == 1:
            c1_logits = ci_logits
        ci_sample = sample_topk(ci_logits, topk, temperature)
        curr_h = model._embed_audio(i, ci_sample)
        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
        curr_pos = curr_pos[:, -1:] + 1

    return curr_sample, c0_logits, c1_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref_trace")
    ap.add_argument("--model", default=os.environ.get("MISO_TTS_8B_MODEL", "MisoLabs/MisoTTS"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lowmem(args.model, device)
    gen = Generator(model)                  # sets up caches, tokenizer, Mimi, watermarker
    gen._model.to(device)                   # re-assert: torchtune cache buffers -> model device
    model = gen._model
    wmk = load_watermarker(device=device)
    os.makedirs(args.out, exist_ok=True)
    print(f"[trace] device={device} model={args.model}  prompts={len(PROMPTS)}")

    for pid, speaker, max_ms, text in PROMPTS:
        torch.manual_seed(args.seed)
        model.reset_caches()
        ptoks, pmask = gen._tokenize_text_segment(text, speaker)
        curr = ptoks.unsqueeze(0).long().to(device)
        cmask = pmask.unsqueeze(0).bool().to(device)
        pos = torch.arange(0, ptoks.size(0), device=device).unsqueeze(0)

        frames, c0s, c1s = [], [], []
        for _ in range(int(max_ms / 80)):
            samp, c0l, c1l = frame_traced(model, curr, cmask, pos, args.temperature, args.topk)
            if torch.all(samp == 0):
                break
            frames.append(samp.squeeze(0).int().cpu().numpy())
            c0s.append(c0l.squeeze(0).float().cpu().numpy())
            c1s.append(c1l.squeeze(0).float().cpu().numpy())
            curr = torch.cat([samp, torch.zeros(1, 1).long().to(device)], dim=1).unsqueeze(1)
            cmask = torch.cat([torch.ones_like(samp).bool(),
                               torch.zeros(1, 1).bool().to(device)], dim=1).unsqueeze(1)
            pos = pos[:, -1:] + 1

        T = len(frames)
        codes = np.stack(frames)                      # (T, 32)
        c0 = np.stack(c0s)                             # (T, 2051)
        c1 = np.stack(c1s)                             # (T, 2051)

        # render + watermark the audio
        codes_t = torch.from_numpy(codes).to(device).T.unsqueeze(0)        # (1, 32, T)
        audio = gen._audio_tokenizer.decode(codes_t).squeeze(0).squeeze(0)
        audio, wm_sr = watermark(wmk, audio, gen.sample_rate, MISO_TTS_WATERMARK)
        import torchaudio
        d = os.path.join(args.out, pid)
        os.makedirs(d, exist_ok=True)
        torchaudio.save(os.path.join(d, f"{pid}.wav"), audio.unsqueeze(0).cpu(), wm_sr)

        np.save(os.path.join(d, "input_tokens.npy"), curr_input := ptoks.unsqueeze(0).int().cpu().numpy())
        np.save(os.path.join(d, "frame_codes.npy"), codes)
        np.save(os.path.join(d, "c0_logits.npy"), c0)
        np.save(os.path.join(d, "c1_logits.npy"), c1)
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"id={pid}\nspeaker={speaker}\ntext={text}\nframes={T}\n"
                    f"audio_sec={T*0.08:.2f}\nseed={args.seed}\nsample_rate={gen.sample_rate}\n")
        print(f"[trace] {pid}: {T} frames ({T*0.08:.1f}s)  saved -> {d}/")

    print(f"[trace] done. pull back ./{args.out}/")


if __name__ == "__main__":
    main()
