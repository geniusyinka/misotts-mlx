#!/usr/bin/env python3
"""
clone_experiment_cuda.py  --  CUDA side of the cross-backend identity test.

Creates 5 distinct "characters" as voice-clone references (5 seeds), then for each
character voice-clones the reference and speaks ONE shared test script TWICE (two
sampling rolls, temp 0.9). Output:
  refs/<char>_ref.wav        the reference voice (defines the character)
  cuda/<char>_take1.wav      cloned, test script, take 1
  cuda/<char>_take2.wav      cloned, test script, take 2
  meta.json                  texts/seeds

Both backends clone from the SAME refs/, so "Jane" is genuinely the same person on
each. (Seeds can't cross backends; reference audio can.)

Run on a CUDA box:  python clone_experiment_cuda.py
Uses the ungated unsloth Llama-3.2 tokenizer (no HF token needed).
"""
import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ["NO_TORCH_COMPILE"] = "1"

import json
import numpy as np
import torch
import torchaudio

import generator as G
from generator import Generator, Segment
from models import MISO_TTS_8B_CONFIG, Model

CHARS = [
    ("char1", 0, 10, "Hey there — it's really good to finally meet you in person."),
    ("char2", 1, 20, "Honestly? I think this might be the best idea we've had all year."),
    ("char3", 0, 30, "Let me walk you through how the whole thing works, step by step."),
    ("char4", 1, 40, "Oh, come on. You cannot be serious right now, can you?"),
    ("char5", 0, 50, "It was a long, quiet evening, and nobody said a single word for hours."),
]
TEST_SCRIPT = ("I still can't quite believe how all of this turned out — but you know what, "
               "I wouldn't change a single thing.")


def _unsloth_tokenizer():
    from transformers import AutoTokenizer
    from tokenizers.processors import TemplateProcessing
    t = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    bos, eos = t.bos_token, t.eos_token
    t._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, t.bos_token_id), (eos, t.eos_token_id)],
    )
    return t


def load_lowmem(repo, device):
    """GPU-direct bf16 load so a 62GB box doesn't OOM on the 32GB fp32 checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    mf = repo if os.path.isfile(repo) else (
        os.path.join(repo, "model.safetensors") if os.path.isdir(repo)
        else hf_hub_download(repo_id=repo, filename="model.safetensors"))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = Model(MISO_TTS_8B_CONFIG)
    finally:
        torch.set_default_dtype(prev)
    model.eval()
    sd = model.state_dict()
    with safe_open(mf, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k].copy_(f.get_tensor(k).to(device=device, dtype=torch.bfloat16))
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G.load_llama3_tokenizer = _unsloth_tokenizer          # ungated, no HF token
    model = load_lowmem(os.environ.get("MISO_TTS_8B_MODEL", "MisoLabs/MisoTTS"), device)
    gen = Generator(model)
    gen._model.to(device)
    os.makedirs("refs", exist_ok=True)
    os.makedirs("cuda", exist_ok=True)
    sr = gen.sample_rate
    meta = {"test_script": TEST_SCRIPT, "sample_rate": sr, "chars": []}
    print(f"[cuda] device={device}  chars={len(CHARS)}  test_script set")

    for cid, spk, seed, ref_text in CHARS:
        # 1) create the character's reference voice (defines identity)
        torch.manual_seed(seed)
        ref_audio = gen.generate(text=ref_text, speaker=spk, context=[], max_audio_length_ms=9000)
        torchaudio.save(f"refs/{cid}_ref.wav", ref_audio.unsqueeze(0).cpu(), sr)

        ctx = [Segment(speaker=spk, text=ref_text, audio=ref_audio)]
        # 2) clone it and say the shared test script TWICE (two rolls)
        for take, tseed in ((1, seed + 1), (2, seed + 2)):
            torch.manual_seed(tseed)
            a = gen.generate(text=TEST_SCRIPT, speaker=spk, context=ctx, max_audio_length_ms=12000)
            torchaudio.save(f"cuda/{cid}_take{take}.wav", a.unsqueeze(0).cpu(), sr)
            print(f"[cuda] {cid} take{take}: {a.shape[-1]/sr:.1f}s")
        meta["chars"].append({"id": cid, "speaker": spk, "ref_seed": seed,
                              "ref_text": ref_text, "ref_sec": float(ref_audio.shape[-1]/sr)})

    with open("meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[cuda] DONE. pull back refs/, cuda/, meta.json")


if __name__ == "__main__":
    main()
