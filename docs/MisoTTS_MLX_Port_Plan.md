# Porting MisoTTS to Apple Silicon — Plan

Target machine: **Mac mini M4 Pro, 64 GB** (273 GB/s memory bandwidth).
Strategy: **MPS quick-win first → MLX port → quantize for real-time.**
Reference hardware: **short cloud-GPU rental** to capture gold-standard outputs.

---

## What MisoTTS actually is (from reading the source)

It's a two-transformer, RVQ design — and crucially, the blog says it's "inspired by the
Sesame CSM architecture." Reading `models.py` confirms it *is* CSM, just scaled up. This is
the single most important fact for the port: **Sesame CSM is already implemented in MLX
(`mlx-audio`), so we adapt that rather than writing a model from scratch.**

| Component        | MisoTTS (`models.py`)                                              | Notes |
|------------------|-------------------------------------------------------------------|-------|
| Backbone         | Llama-3.2-style, 32 layers, dim 4096, 32 heads / 8 KV heads, FFN 14336, rope_base 500k, scale 32, max_seq 2048 | predicts codebook 0 over time |
| Depth decoder    | Llama-3.2-style, 8 layers, dim 1536, 24 heads / 6 KV heads, FFN 6912 | predicts codebooks 1..31 over depth, **caches reset every frame** |
| Text embeddings  | `Embedding(128256, 4096)`                                         | |
| Audio embeddings | `Embedding(2051 * 32, 4096)`                                      | one flat table, indexed `token + codebook*2051` |
| projection       | `Linear(4096 → 1536, bias=False)`                                 | backbone hidden → decoder input |
| codebook0_head   | `Linear(4096 → 2051, bias=False)`                                 | |
| audio_head       | `Parameter(31, 1536, 2051)`                                       | per-codebook output matrices for cb 1..31 |
| Audio tokenizer  | Kyutai **Mimi** (24 kHz, via `moshi.models.loaders`)              | already in `mlx-audio` |
| Watermark        | SilentCipher (`sony/silentcipher`)                                | signal-processing; fine to run on CPU |

Per-frame loop (`Model.generate_frame`): embed tokens → sum over the codebook axis →
backbone forward (1 pass) → `codebook0_head` → sample cb0 → then **31 tiny decoder passes**
over depth, each producing one more codebook via `audio_head[i-1]`. Frame rate is 12.5 Hz
(`max_audio_length_ms / 80`).

### Embedding tables (`llama-8B` flavor) are replaced by `Identity()`

`_prepare_transformer()` swaps `tok_embeddings` and `output` for `nn.Identity`. The Llama
stacks are used as **pure transformer blocks over precomputed embeddings** — we feed in the
summed codebook+text embedding and read the hidden state out. The MLX port must do the same
(don't wire up Llama's own embedding/LM head).

---

## The Apple Silicon blocker, and the fix

`run_misotts.py` selects device with this comment:

> *"Select the best available device, skipping MPS due to float64 limitations."*

Metal/MPS has no float64. Somewhere in the pipeline (most likely the Mimi codec and/or the
SilentCipher watermarker — both do signal processing) a float64 op appears. The fix for the
**quick-win** is `PYTORCH_ENABLE_MPS_FALLBACK=1`: unsupported/float64 ops silently run on CPU
while the heavy 8B matmuls stay on the GPU. That's exactly what `run_misotts_mac.py` does.
In the MLX phase this stops mattering — MLX is fp32/fp16/bf16 on Metal natively and we use the
MLX Mimi.

---

## Validation contract (how we know the port is correct)

Sampling is stochastic, so generated tokens won't match across backends even with a fixed seed.
**Logits are deterministic**, so we validate on those.

`capture_frame0.py` dumps, for one fixed prompt: `last_h`, `c0_logits`, `c1_logits`, and the
exact `input_tokens`. Run it on the reference GPU and on each candidate (MPS, then MLX), then
`compare_frame0.py` reports cosine similarity + top-1/top-5 agreement.

Pass bar:
- top-1 token must match for both c0 and c1;
- top-5 overlap ≥ 4/5;
- cosine > 0.999 = clean; 0.99–0.999 = acceptable bf16-vs-fp32 noise; < 0.99 = bug.

A matching listening test on `reference.wav` vs the port's output is the final human check.

---

## Phases

### Phase 0 — Capture reference (cloud GPU, ~1 hour, ~$1–3)
1. Rent a 24 GB GPU (3090/4090) or any CUDA box. RunPod/Vast/Lambda all fine.
2. `git clone https://github.com/MisoLabsAI/MisoTTS && cd MisoTTS && uv sync --python 3.10`
3. Copy in `capture_frame0.py`. Run: `python capture_frame0.py --out reference --wav`
4. Download the whole `reference/` folder (tiny — a few MB) + `reference.wav`. Tear down the box.

### Phase 1 — MPS quick-win (your Mac, correctness not speed)
1. Same clone + `uv sync` on the Mac.
2. Copy in `run_misotts_mac.py` and `capture_frame0.py`.
3. `python run_misotts_mac.py` → should produce `mac_out.wav`. Expect it to be slow (RTF ≫ 1).
4. `python capture_frame0.py --out candidate --device mps --dtype float32`
5. `python compare_frame0.py reference candidate` → confirm logits match.
   - **This proves the model runs correctly on your Mac.** Speed is Phase 2's job.
   - If MPS throws an unsupported-op error the fallback didn't catch, note the op name —
     we pin that submodule to CPU and move on.

### Phase 2 — MLX port (the real performance work)
1. `pip install mlx mlx-audio`. Study `mlx_audio`'s CSM model as the template.
2. Define the MisoTTS config (table above) — same shapes, just bigger than CSM's defaults.
3. **Weight remap**: load `model.safetensors`, map PyTorch keys → MLX module keys
   (backbone.layers.*, decoder.layers.*, text_embeddings, audio_embeddings, projection,
   codebook0_head, audio_head). Watch for: RoPE convention, q/k/v packing, RMSNorm naming.
4. Reuse `mlx-audio`'s **Mimi** for encode/decode. Keep SilentCipher watermarking on
   CPU/Torch (or numpy) — it's not perf-critical and must stay on for compliance.
5. Validate with `capture_frame0.py --out candidate_mlx` (extend it to call the MLX path) →
   `compare_frame0.py reference candidate_mlx`.

### Phase 3 — Quantize + real-time
1. MLX built-in quantization → Q8 first (safe), then Q4. Re-run the validation contract after
   each; listen for artifacts.
2. Benchmark RTF. Expected on M4 Pro (273 GB/s, bandwidth-bound):
   - fp16 ≈ 1.5× slower than real-time, Q8 ≈ borderline real-time, **Q4 ≈ ~2.5× faster than RT**.
3. Add streaming (yield audio per N frames) so time-to-first-audio is low for agent use.

---

## Risks / unknowns (honest list)
- **Someone may beat us to it.** Given the MOSS-TTS and CSM MLX precedents, an `mlx-community`
  MisoTTS port could appear within days. Worth checking HF before sinking time into Phase 2.
- **Mimi parity in MLX.** The `mlx-audio` Mimi must use the same codebook config
  (`set_num_codebooks(32)`) and 24 kHz as Torch Mimi, or audio decode will be wrong.
- **RoPE / weight-layout mismatches** are the usual source of "logits look plausible but
  cosine ≈ 0.9" bugs. The frame-0 harness catches these immediately.
- **Watermark.** Keep it on (Miso's terms). It's separable from the model, so it won't block
  the port — just runs on CPU.
- **License.** Modified MIT; fine to port and run locally. Don't ship voice-cloning of real
  people; output stays watermarked.

## Files in this drop
- `capture_frame0.py` — device-agnostic reference/candidate capture.
- `compare_frame0.py` — logit diff / pass-fail report.
- `run_misotts_mac.py` — MPS quick-win runner with RTF readout.
- `MisoTTS_MLX_Port_Plan.md` — this document.
