#!/usr/bin/env python3
"""
clone_experiment_mlx.py  --  MLX side of the cross-backend identity test.

Voice-clones the SAME 5 reference voices the CUDA run produced (refs/<char>_ref.wav +
meta.json) and speaks the same shared test script TWICE per character (temp 0.9),
mirroring clone_experiment_cuda.py. Output: mlx/<char>_take1.wav, mlx/<char>_take2.wav.

Adds audio-context conditioning (voice cloning) to the MLX path: Mimi-encode the
reference, build [ref-text | ref-audio | new-text] context frames (mirrors the torch
tokenize_segment/tokenize_audio), then run the autoregressive loop from that prompt.

Usage:  python clone_experiment_mlx.py --refs refs --meta meta.json --out mlx --dtype float32
"""
import argparse, glob, json, os, time
import numpy as np
import soundfile as sf
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from misotts_mlx import load_misotts_mlx, N_AUDIO_CODEBOOKS as NCB
from run_misotts_mlx import find_model, load_text_tokenizer, sample_topk


def text_frames(tok, text, speaker):
    ids = tok.encode(f"[{speaker}] {text.lstrip()}")
    fr = np.zeros((len(ids), NCB + 1), np.int32)
    m = np.zeros((len(ids), NCB + 1), np.float32)
    fr[:, -1] = ids
    m[:, -1] = 1.0
    return fr, m


def audio_frames(mimi, audio_24k):
    # audio_24k: 1-D float mx.array @ 24kHz -> Mimi codes (NCB, T), + EOS frame (mirror torch)
    codes = np.array(mimi.encode(audio_24k[None, None])[0])          # (NCB, T)
    codes = np.concatenate([codes, np.zeros((NCB, 1))], axis=1)      # + EOS column
    T1 = codes.shape[1]
    fr = np.zeros((T1, NCB + 1), np.int32)
    m = np.zeros((T1, NCB + 1), np.float32)
    fr[:, :NCB] = codes.T
    m[:, :NCB] = 1.0
    return fr, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--meta", default="meta.json")
    ap.add_argument("--out", default="mlx")
    ap.add_argument("--dtype", default="float32", choices=["float32"])
    ap.add_argument("--ms", type=int, default=12000)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    test_script = meta["test_script"]
    chars = meta["chars"]
    os.makedirs(args.out, exist_ok=True)

    model = load_misotts_mlx(find_model(), dtype=mx.float32)
    from mlx_audio.codec.models.mimi.mimi import Mimi
    mimi = Mimi.from_pretrained("kyutai/moshiko-pytorch-bf16")
    tok = load_text_tokenizer()
    import torch
    from watermarking import MISO_TTS_WATERMARK, load_watermarker, watermark as apply_wm
    wmk = load_watermarker(device="cpu")
    print(f"[mlx-clone] ready. {len(chars)} chars, test_script set")

    next_mask = mx.expand_dims(mx.concat([mx.ones((1, NCB)), mx.zeros((1, 1))], axis=1), 1)

    for c in chars:
        cid, spk = c["id"], c["speaker"]
        ref_wav = os.path.join(args.refs, f"{cid}_ref.wav")
        a, sr = sf.read(ref_wav)
        if a.ndim > 1:
            a = a.mean(1)
        ref_audio = mx.array(a.astype(np.float32))

        # context = ref text + ref audio (cloning) ; then the new line
        rtf, rtm = text_frames(tok, c["ref_text"], spk)
        raf, ram = audio_frames(mimi, ref_audio)
        gtf, gtm = text_frames(tok, test_script, spk)
        prompt = np.concatenate([rtf, raf, gtf], axis=0)
        pmask = np.concatenate([rtm, ram, gtm], axis=0)

        for take, seed in ((1, c["ref_seed"] + 1), (2, c["ref_seed"] + 2)):
            mx.random.seed(seed)
            curr = mx.array(prompt)[None]
            cmask = mx.array(pmask)[None]
            bb = make_prompt_cache(model.backbone)
            samples = []
            t = time.time()
            for step in range(int(args.ms / 80)):
                e = model.embed_tokens(curr) * mx.expand_dims(cmask, -1)
                last = model.backbone(e.sum(axis=-2), cache=bb)[:, -1, :]
                c0 = sample_topk(model.codebook0_head(last), args.topk, args.temp).reshape(1, 1).astype(mx.int32)
                codes = [c0]
                din = mx.concat([mx.expand_dims(last, 1), model.embed_audio(0, c0)], axis=1)
                dc = make_prompt_cache(model.decoder)
                for i in range(1, NCB):
                    dh = model.decoder(model.projection(din), cache=dc)
                    ci = sample_topk(mx.matmul(dh[:, -1, :], model.audio_head[i - 1]), args.topk, args.temp).reshape(1, 1).astype(mx.int32)
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
            mx.eval(samples)
            audio = np.array(mimi.decode(mx.stack(samples).transpose(1, 2, 0))).squeeze().astype(np.float32)
            wm_audio, osr = apply_wm(wmk, torch.from_numpy(audio), int(mimi.sample_rate), MISO_TTS_WATERMARK)
            sf.write(os.path.join(args.out, f"{cid}_take{take}.wav"), wm_audio.cpu().numpy().astype(np.float32), osr)
            print(f"[mlx-clone] {cid} take{take}: {len(samples)} frames {len(samples)*0.08:.1f}s  gen {time.time()-t:.1f}s")

    print("MLXCLONE_DONE")


if __name__ == "__main__":
    main()
