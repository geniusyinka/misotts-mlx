# Running MisoTTS 8B on your Mac (Apple Silicon / MLX)

A practical guide to running the 8B MisoTTS speech model **locally on Apple Silicon** — both
the simple PyTorch/MPS path and the fast MLX path — plus the gotchas that will bite you.

> **TL;DR:** An 8B text-to-speech model runs on an M-series Mac. The MLX build is ~12.8×
> faster than the naive path; **Q8 is lossless at ~2× slower than real-time**, mixed-Q4 is a
> bit faster with a small quality tradeoff. First run downloads a ~33 GB model and takes a
> minute to load; generation of a 5 s clip then takes ~8–10 s.

---

## What you need

- An Apple Silicon Mac (M1 or later). Tested on an **M4 Pro, 64 GB**.
- ~40 GB free disk (the checkpoint is fp32, **32.75 GB**) and enough RAM to hold it
  (32 GB resident at fp32; less when quantized).
- Python 3.10 and [`uv`](https://docs.astral.sh/uv/) (or pip).
- A Hugging Face account/token **with access to `meta-llama/Llama-3.2-1B`** for the PyTorch
  path (see gotchas). The MLX path avoids this.

---

## Setup

```bash
git clone https://github.com/geniusyinka/MisoTTS.git
cd MisoTTS
uv sync --python 3.10            # base (torch) deps
uv pip install -r requirements-mlx.txt   # adds mlx, mlx-lm, mlx-audio for the MLX path
```

The first generation downloads the model (~33 GB) and the Mimi codec into your Hugging Face
cache. Later runs reuse the cache.

---

## Option A — PyTorch on MPS (simple, slow, good for a correctness check)

```bash
python run_misotts_mac.py --text "Hey! This is running on my Mac." --ms 6000
```

This enables `PYTORCH_ENABLE_MPS_FALLBACK=1` (float64/unsupported ops run on CPU, the big
matmuls stay on the GPU) and defaults to fp32. Expect **RTF ≈ 21×** — i.e. slow. This path is
about *correctness*, not speed.

## Option B — MLX (fast, the real path)

```bash
# fp32 (clean), Q8 (lossless, recommended), or mixed-Q4 (fastest)
python run_misotts_mlx.py --text "Hey! This is running on my Mac." --ms 6000 --bits 8
python run_misotts_mlx.py --text "..." --quant mixed      # fastest
python run_misotts_mlx.py --text "..."                    # fp32
```

It prints timing and the real-time factor:

```
[mlx] frames=75  gen=27.7s  mimi_decode=0.6s
[mlx] generated 6.00s of audio in 28.2s  ->  RTF=4.71x
```

**Pick your mode:**

| Mode | Flag | RTF | Quality | Use when |
|---|---|---|---|---|
| fp32 | *(default)* | 4.71× | clean | reference / debugging |
| **Q8** | `--bits 8` | **2.01×** | **lossless** | **default for real use** |
| mixed-Q4 | `--quant mixed` | 1.65× | slight tradeoff | you want it as fast as possible |

> The MLX path uses the **ungated** `unsloth/Llama-3.2-1B` tokenizer and the public Kyutai Mimi
> codec, so it needs **no Hugging Face token**.

### Voice cloning

Pass a reference clip with `--ref-audio` to speak in that voice (give its transcript via
`--ref-text` for best results — without an audio prompt the model picks a fresh random voice
each run):

```bash
python run_misotts_mlx.py --bits 8 \
  --ref-audio jane.wav --ref-text "what jane says in jane.wav" \
  --text "This line is spoken in Jane's voice." --out jane_says.wav
```

The reference is resampled to 24 kHz and Mimi-encoded into the context. A cross-backend test
(speaker-verification embeddings over 5 cloned characters) found a cloned voice is the **same
person** on MLX as on CUDA — cross-backend similarity equals a backend's own take-to-take
variation. Identity is anchored by the reference, so it's stable run to run (pitch/energy still
vary take to take); a bare `--speaker` id is only a turn marker, not a fixed identity.

---

## What to expect (be realistic)

- **One-time model load:** ~60 s per process (loading/quantizing 33 GB). Keep the process warm
  if you're generating repeatedly.
- **Generation is not real-time.** Q8 ≈ 2× slower than real-time: a 5 s clip ≈ 10 s of compute.
  This is an 8B model with a 32-deep autoregressive codebook decoder; that depth loop is the
  bottleneck, and it's inherently sequential.
- **fp16 is *slower* than fp32 here** — don't reach for it. The loop is latency-bound at
  batch=1/seq=1, where half precision doesn't help and adds cast overhead.
- **Watermarking:** the MLX runner applies MisoTTS's SilentCipher watermark **by default**
  (CPU/torch, ~0.7 s, imperceptible — RMS change ≈ 0 dB). Verify any clip with
  `python watermarking.py --audio_path out.wav`. Use `--no-watermark` only for local
  debugging. Don't ship un-watermarked audio, and don't clone real people's voices.

---

## Verifying your port is actually correct

Don't trust "it makes sound." Generation is stochastic, so validate on **logits**, which are
deterministic:

```bash
# capture frame-0 tensors on a reference (CUDA) box, copy the reference/ folder over, then:
python capture_frame0_mlx.py --ref reference --out candidate_mlx          # or --bits 8 / --quant mixed
python compare_frame0.py reference candidate_mlx
```

Pass bar: c0/c1 **top-1 must match**, top-5 overlap ≥ 4/5, cosine > 0.99. A cosine around 0.9
with a top-1 mismatch almost always means a RoPE-convention or weight-layout bug, not noise.

For a **deeper** check (the one used to validate this port), capture a per-frame logit trace on
the CUDA box and teacher-force it through MLX — comparing logits at *every* frame across whole
utterances, which exercises the KV-cache over hundreds of steps:

```bash
# on the CUDA box (records codes + per-frame c0/c1 logits + wav for 5 prompts):
HF_TOKEN=hf_xxx python capture_trace.py --out ref_trace
# copy ref_trace/ to the Mac, then:
python compare_trace_mlx.py --traces ref_trace            # fp32
python compare_trace_mlx.py --traces ref_trace --bits 8   # Q8
```

Reference results over 1,117 frames (5 styles, two ≥30s): fp32 & Q8 hold mean cosine **0.9999**
with ~99% top-1 and **no drift over the long clips**; mixed-Q4 stays >0.99 cosine but ~93% top-1
(a real fidelity trade). `capture_trace.py` loads the model **GPU-direct in bf16** (streams
weights one tensor at a time) so a 62 GB box doesn't OOM on the 32 GB fp32 checkpoint.

---

## Gotchas (the ones that cost real time)

1. **Gated tokenizer (PyTorch path).** `generator.py` loads `meta-llama/Llama-3.2-1B`, a gated
   repo — you'll get a 401 without an authorized `HF_TOKEN`. Export it before running. (The MLX
   path uses an ungated mirror and sidesteps this.)

2. **The KV-cache device bug.** torchtune's `setup_caches()` allocates cache buffers on CPU even
   after the model is on GPU/MPS, causing a `device mismatch` in the attention path. Fix: call
   `model.to(device)` again *after* the generator is constructed. (Already handled in
   `run_misotts_mac.py` and `capture_frame0.py`.)

3. **Hugging Face downloads stalling on a VPN.** If `huggingface.co` times out at the TLS layer
   while everything else works, a zero-trust/VPN client (e.g. Tailscale's network extension) is
   likely intercepting it. **Fully quit** the VPN (not just "disconnect") for the one-time 33 GB
   download — mirrors like `hf-mirror.com` won't help (they 308-redirect file fetches straight
   back to huggingface.co). The bump in HF download timeouts also helps:
   `export HF_HUB_DOWNLOAD_TIMEOUT=600`.

4. **Don't quantize everything to Q4.** Naive uniform Q4 breaks the model (the small decoder and
   the embeddings can't take it — codebook-1 logits collapse). Use `--quant mixed` (backbone Q4,
   decoder/heads Q8, embeddings full precision) or stick to `--bits 8`.

5. **Dependency pin clash.** `moshi_mlx` pins an older `mlx`/`huggingface_hub` and will downgrade
   your environment, breaking `mlx-lm`. Use **`mlx-audio`'s** Mimi codec instead (it ships the
   same Kyutai checkpoint loader) and keep `mlx==0.31.2`.

---

## Files

| File | What it does |
|---|---|
| `run_misotts_mlx.py` | End-to-end MLX generation (`--dtype/--bits/--quant/--compiled-decoder`), prints RTF |
| `misotts_mlx.py` | The MLX model: config, RoPE/attention, torchtune→MLX weight loader, quant recipes |
| `mlx_fast_decoder.py` | Experimental `mx.compile` fixed-cache depth decoder |
| `run_misotts_mac.py` | PyTorch/MPS runner (fallback path) |
| `capture_frame0_mlx.py` / `capture_frame0.py` / `compare_frame0.py` | The logit validation harness |
| `docs/MisoTTS_MLX_Port_Plan.md` | The original porting plan |
| `docs/porting-misotts-to-apple-silicon-technical.md` | The deep technical write-up |
