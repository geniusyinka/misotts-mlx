---
title: "Porting an 8B Text-to-Speech Model to Apple Silicon (and proving it actually works)"
published: false
description: "MisoTTS 8B shipped CUDA-only. I ported it to run entirely on a Mac via Apple's MLX — ~12.8x faster than the naive path, validated against the original CUDA model down to the logits."
tags: mlx, applesilicon, machinelearning, tts
cover_image: ""
canonical_url: "https://geniusyinka.github.io/MisoTTS/"
---

> **TL;DR** — MisoTTS is an 8-billion-parameter, Sesame-CSM-style text-to-speech model that
> shipped CUDA-only. I ported it to run **entirely on an Apple Silicon Mac** via MLX: ~**12.8×
> faster** than the naive MPS path, with a **lossless ~2× real-time-adjacent** mode — and, crucially,
> **validated against the original CUDA model at the logit level**, not just "it makes sound." Code
> and a deep writeup: [github.com/geniusyinka/MisoTTS](https://github.com/geniusyinka/MisoTTS).

## The gap worth closing

Most of the current generation of speech models ship CUDA-only. MisoTTS is no exception — its
reference code picks `cuda` or `cpu` and explicitly skips Apple's Metal backend ("float64
limitations"). That leaves a genuinely good model unusable on the hardware a huge number of
developers actually have on their desk.

Getting it onto Apple Silicon isn't just a portability checkbox. It's **local and private** (speech
often involves personal text), it has **no cloud bill or cold starts**, and an M4 Pro's **unified
memory** can hold an 8B model the GPU addresses directly. And MLX has matured enough — with
`mlx-lm` and `mlx-audio` — that this is finally tractable.

## What MisoTTS is

Two transformers: a Llama-3.2-style **8B backbone** that predicts the first audio codebook over
time, and a smaller **300M depth decoder** that predicts the remaining 31 codebooks within each
frame. Those become Kyutai **Mimi** codes at 12.5 Hz, decoded to 24 kHz audio. The key realization:
MisoTTS is **Sesame CSM, scaled up** — and CSM already has clean MLX implementations. So the job
became "adapt an existing template and remap the weights," not "write an 8B model from scratch."

## The part most ports skip: validation

Generation is stochastic (temperature, top-k), so the generated *tokens* will never match across
CUDA / MPS / MLX, even with a fixed seed — the RNG differs per backend. If you only check "does it
make plausible speech," you can ship a port with a subtle RoPE or weight-layout bug that *sounds*
fine but is quietly wrong.

**Logits, however, are deterministic.** So I validated on logits, three ways:

1. **Frame-0 gate** — one forward pass vs the CUDA reference. The bug-catcher.
2. **Teacher-forced trace** — feed the *exact* CUDA token sequence through the MLX model and diff
   the logits **at every frame** across 5 prompt styles and **1,117 frames** (including two 30s+
   clips). This exercises the KV-cache over hundreds of steps, which one frame never does.
3. **Voice-identity test** — clone 5 reference voices on both backends and score them with a
   speaker-verification embedding.

The numbers:

| | result |
|---|---|
| fp32 / Q8 logit cosine vs CUDA (1,117 frames) | **0.9999**, ~99% top-1, no drift over 30s |
| cross-backend speaker similarity | **0.882** — equal to a backend's own take-to-take variation (0.886) |

In plain terms: **Q8 is lossless**, and a voice cloned on MLX is the **same person** as on CUDA —
it differs no more than CUDA's own two takes differ from each other.

## The port itself

The classic trap porting Llama-family weights is the **RoPE convention**: HF checkpoints store q/k
permuted because HF uses "split-half" RoPE, while Meta/torchtune use "interleaved-pair." Get it
wrong and logits look plausible but cosine sits at ~0.9.

The CSM-in-MLX work I adapted (`senstella/csm-mlx`) already used an **interleaved-pair
`Llama3ScaledRoPE`** matching torchtune — which MisoTTS also uses. So the checkpoint loaded into
MLX with **no q/k permutation at all**: the weight remap is a pure rename table (367 tensors,
strict load passed first try). That's the whole ballgame.

## Quantization, and a surprise

The intuitive next step — fp16 — was **slower** (RTF 10.6× vs fp32's 4.71×). The autoregressive
loop runs at batch=1, seq=1: tiny, latency-bound shapes where half precision doesn't win and the
RoPE's internal float32 casts add overhead. Lesson: for single-token decode, half precision is not
a free win.

Quantization *is* the lever. But **naive Q4 broke the model** — 4-bit on the small decoder and the
embedding tables collapsed the codebook-1 logits (cosine 0.51). The fix was a **mixed recipe**:
quantize the big backbone to 4-bit, keep the decoder/heads at 8-bit, leave embeddings full
precision.

**The speed ladder (M4 Pro):**

| Config | RTF | Quality |
|---|---|---|
| torch / MPS fp32 | 21.2× | clean |
| MLX fp32 | 4.71× | clean |
| **MLX Q8** | **2.01×** | **lossless** |
| MLX mixed-Q4 | 1.65× | slight tradeoff |

**21.2× → 1.65×: a ~12.8× end-to-end speedup**, with a lossless 2× option.

## An honest ending

Real-time (RTF < 1.0) did **not** pan out. Profiling showed the depth decoder's 31 sequential
passes per frame are ~79% of the cost — ~40% irreducible weight-bandwidth, ~60% GPU kernel-launch
overhead. Async pipelining didn't help (the loop is genuinely sequential). A hand-written,
`mx.compile`-friendly fixed-cache decoder was numerically correct and 1.19× in isolation, but the
gain vanished in the full pipeline. Getting under 1.0× for an 8B backbone + 32-deep RVQ decoder at
batch=1 would need a fully-unrolled single-graph compiled frame, custom Metal kernels, or
architectural changes — all high-effort, uncertain payoff.

So I banked it: an 8B emotive TTS model running **entirely on a Mac**, validated against CUDA at the
logit level across 5 styles and 1,117 frames, **~12.8× faster** than the naive path, lossless at
~2× real-time, with working voice cloning. Not real-time — but a genuinely usable, private, local
speech model on hardware the reference code refused to even try.

The validation harness is the quiet hero: every "it worked" above is a number, not a vibe.

---

*Code, the run guide, and the full technical account:
[github.com/geniusyinka/MisoTTS](https://github.com/geniusyinka/MisoTTS). Built on
[MLX](https://github.com/ml-explore/mlx) + mlx-audio's Mimi codec. Generated audio is watermarked;
don't clone real people's voices.*
