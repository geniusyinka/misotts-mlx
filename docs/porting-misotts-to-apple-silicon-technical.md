# Porting an 8B Text-to-Speech Model to Apple Silicon: A Technical Account

*How MisoTTS 8B went from CUDA-only to running entirely on a Mac mini M4 Pro — the
validation methodology, the MLX port, and an honest look at the real-time wall.*

---

## Why this mattered

MisoTTS 8B is a high-quality, emotive text-to-dialogue model. Like most of the current
generation of speech models, it shipped **CUDA-only**: the reference code selects `cuda`
or falls back to `cpu`, and explicitly skips Apple's Metal backend with the comment
*"skipping MPS due to float64 limitations."*

That leaves a large, capable model unusable on the hardware a huge number of developers
actually have on their desk. Getting it onto Apple Silicon isn't just a portability
exercise:

- **Local & private.** Speech synthesis often involves personal or sensitive text. Running
  on-device means nothing leaves the machine.
- **No cloud bill, no cold starts.** A one-time model download, then it's yours.
- **Unified memory.** An M4 Pro with 64 GB can hold an 8B model (32 GB at fp32) in memory
  that the GPU addresses directly — no host↔device copies.
- **MLX exists now.** Apple's MLX array framework gives native fp32/fp16/bf16 on Metal plus
  a quantization path, and the ecosystem (`mlx-lm`, `mlx-audio`) has matured to the point
  where this is tractable.

This is the story of doing it — and doing it *verifiably*, not just "it produces sound."

---

## What MisoTTS actually is

Reading the source confirms the blog's hint: MisoTTS is **Sesame CSM, scaled up**. It's a
two-transformer RVQ design.

| Component | Spec | Role |
|---|---|---|
| Backbone | Llama-3.2-style, 32 layers, dim 4096, 32 heads / 8 KV, FFN 14336, RoPE base 500k, scale 32 | predicts codebook 0 over time |
| Depth decoder | Llama-3.2-style, 8 layers, dim 1536, 24 heads / 6 KV, FFN 6912 | predicts codebooks 1–31 within each frame |
| Text embeddings | `Embedding(128256, 4096)` | |
| Audio embeddings | `Embedding(2051·32, 4096)` | one flat table, indexed `token + codebook·2051` |
| projection | `Linear(4096 → 1536)` | backbone hidden → decoder input |
| codebook0_head | `Linear(4096 → 2051)` | |
| audio_head | `Parameter(31, 1536, 2051)` | per-codebook output matrices for cb 1–31 |
| Codec | Kyutai **Mimi**, 24 kHz, 32 codebooks | tokens ↔ waveform |

Crucially, the backbone and decoder are torchtune `llama3_2` stacks with their token
embeddings and output heads **replaced by `nn.Identity`** — they're used as pure transformers
over precomputed embeddings. The per-frame loop is: embed the input tokens → sum over the
codebook axis → one backbone pass → `codebook0_head` samples codebook 0 → then **31 tiny
sequential decoder passes** over depth, each producing one more codebook. Frame rate is
12.5 Hz (one frame per 80 ms of audio).

That "CSM, scaled up" fact is the single most important one for the port: **CSM already has
clean MLX implementations** (`Blaizzy/mlx-audio`, `senstella/csm-mlx`). The job becomes
"adapt an existing template to a bigger config and remap the weights," not "write a model
from scratch."

---

## The validation contract (the part most ports skip)

Sampling is stochastic (temperature, top-k), so generated *tokens* will never match across
CUDA / MPS / MLX even with a fixed seed — the RNG differs per backend. If you only check "does
it make plausible speech," you can ship a port with a subtle RoPE or weight-layout bug that
*sounds* fine but is quietly wrong.

**Logits, however, are a deterministic function of (weights, inputs).** So the port is
validated on logits, not audio:

1. `capture_frame0.py` runs one fixed prompt through the model and dumps the frame-0
   `last_h` (backbone hidden), `c0_logits`, `c1_logits`, and the exact input tokens.
2. `compare_frame0.py` diffs a candidate capture against a gold reference: cosine similarity
   plus top-1 / top-5 agreement.

**Pass bar:** c0 and c1 top-1 must match; top-5 overlap ≥ 4/5; cosine > 0.999 is clean,
0.99–0.999 is acceptable bf16-vs-fp32 noise, < 0.99 is a real bug (usually RoPE or weight
layout). This harness caught every issue and confirmed every success below.

---

## Phase 0 — Capture the gold reference (cloud GPU)

We rented a single RunPod RTX 3090 (community cloud, $0.22/hr), cloned MisoTTS, and ran
`capture_frame0.py --wav` to produce the deterministic **CUDA/bf16** reference tensors plus a
`reference.wav`. Total cost: **$0.14**, pod destroyed immediately after. This is the ground
truth every Mac/MLX candidate is measured against. (The model checkpoint, it turns out, is
fp32 — 32.75 GB — even though inference runs in bf16.)

A small but real lesson: the Llama-3.2 *tokenizer* repo is gated on Hugging Face, so any run
needs an authenticated token even though the MisoTTS weights themselves are public.

---

## Phase 1 — MPS quick-win (prove correctness on the Mac)

Before MLX, the fastest way to prove the model runs *correctly* on Apple Silicon is plain
PyTorch with `PYTORCH_ENABLE_MPS_FALLBACK=1`: unsupported / float64 ops (in the Mimi codec
and SilentCipher watermarker) silently run on CPU while the big matmuls stay on the GPU.

One real bug surfaced here, and it's instructive: **torchtune's `setup_caches()` allocates the
KV-cache buffers on CPU**, but it runs *inside* the generator's constructor *after* the model
was moved to the GPU — so the cached-attention path crashes with a `cuda:0` vs `cpu` (later
`mps` vs `cpu`) device mismatch. The fix is a one-liner: re-assert `model.to(device)` after
construction so the freshly-created cache buffers follow the model. (We hit the same bug on
CUDA in the reference capture — it's not Apple-specific.)

Result: MPS/fp32 reproduced the CUDA reference logits cleanly (cosines 0.9999, both top-1s
match), and produced a 5.36 s clip — at **RTF 21.2×** (21× slower than real-time). Correctness:
proven. Speed: that's the next phase's problem.

---

## Phase 2 — The MLX port

We adapted `senstella/csm-mlx`, whose `CSM` class is a near-exact structural twin of MisoTTS's
`Model`: same `backbone`/`decoder` (built from `mlx_lm`'s `LlamaModel`), same
`text_embeddings`/`audio_embeddings`/`projection`/`codebook0_head`/`audio_head`, same
`embed_tokens`/`embed_audio` logic, same `Identity()` embedding patch.

### The RoPE convention is the whole ballgame

The classic trap porting Llama-family weights is the **RoPE convention**. HF-format Llama
checkpoints store q/k projections *permuted* because HF applies RoPE in the "split-half"
(GPT-NeoX) convention, while Meta's original / torchtune use the "interleaved-pair" (GPT-J)
convention. Get this wrong and you get the textbook symptom: logits look plausible but cosine
sits around 0.9 and audio is subtly garbled.

`senstella/csm-mlx` already solved this for CSM by **replacing `mlx_lm`'s default attention
with a hand-written `Llama3ScaledRoPE`** that matches torchtune's interleaved convention
exactly (same `low_freq=1, high_freq=4, old_context=8192`, scale factor from config). Since
MisoTTS uses the *identical* torchtune `llama3_2`, this means our checkpoint loads into MLX
**with no q/k permutation at all** — the weight remap is a pure rename table.

### The weight remap

The torch checkpoint uses torchtune names; `mlx_lm`'s `LlamaModel` uses HF names. The mapping:

| torchtune (torch) | mlx_lm (MLX) |
|---|---|
| `…attn.q_proj` / `k_proj` / `v_proj` | `…self_attn.q_proj` / `k_proj` / `v_proj` |
| `…attn.output_proj` | `…self_attn.o_proj` |
| `…mlp.w1` / `w3` / `w2` | `…mlp.gate_proj` / `up_proj` / `down_proj` |
| `…sa_norm.scale` / `mlp_norm.scale` | `…input_layernorm` / `post_attention_layernorm.weight` |
| `backbone.norm.scale` | `backbone.norm.weight` |
| embeddings / projection / heads | (unchanged) |

No transposes (both frameworks store `Linear` weights as `(out, in)`), no permutations. The
checkpoint has 367 tensors and they map 1:1 — `model.load_weights(..., strict=True)` passed
on the first try, which is itself strong evidence the config and mapping are right.

### Result

Feeding the saved reference tokens directly into the MLX model (sidestepping the gated
tokenizer entirely) and comparing frame-0 logits:

```
last_h    cosine 0.999883
c0_logits cosine 0.999784   top-1 184 = 184   top-5 5/5
c1_logits cosine 0.999967   top-1 1880 = 1880 top-5 4/5
```

Identical, to the digit, to the MPS/fp32 candidate. The MLX port introduced **zero** numeric
drift versus PyTorch on the same hardware. The model was correct.

End-to-end MLX generation (autoregressive frame loop with a backbone KV-cache + a fresh
decoder cache per frame, then `mlx-audio`'s Mimi for decode — the same Kyutai checkpoint the
torch path uses) ran at **RTF 4.71×** in fp32. Same math, same precision as Phase 1, but
**4.5× faster** — MLX's Metal kernels and a real KV-cache loop versus torch's MPS-with-CPU-
fallback.

---

## Phase 3 — Quantization, and a surprise

### fp16 is a trap here

The intuitive next step — fp16 — was **slower**, not faster: RTF 10.6× (845 ms/frame) versus
fp32's 4.71× (369 ms/frame). The autoregressive loop runs at batch=1, seq=1: tiny, latency-
bound shapes where fp16 doesn't win and the RoPE's internal float32 round-trips add per-layer
cast overhead. Lesson: for single-token autoregressive decode, **don't assume half precision
is a free win.**

### Quantization, done carefully

MLX's `nn.quantize` is the right lever — the backbone's 32 GB of weights streamed per token is
the real cost, and quantized matmul kernels are bandwidth-bound and batch-1 friendly.

- **Q8 (everything):** essentially lossless. Logit cosines 0.9999, both top-1s match. **RTF
  2.01×** — a clean 2.3× over MLX fp32.
- **Naive Q4 (everything):** **broken.** Quantizing the *embeddings* and the small *decoder*
  to 4-bit is too lossy — `c1_logits` cosine collapsed to **0.51** and the c0 top-1 flipped.
- **Mixed-Q4:** the fix. A `class_predicate` quantizes the big **backbone to 4-bit**, keeps the
  **decoder and heads at 8-bit**, and leaves the **embeddings full precision** (quantizing
  lookup tables injects per-token noise). This rescued c1 (0.51 → 0.9989) and restored the c0
  top-1. **RTF 1.65×**, with `last_h` cosine 0.99 (borderline — greedy top-1 holds, audio is
  intelligible, but it's not the clean match Q8 is).

### The speed ladder

| Config | RTF | ms/frame | Quality |
|---|---|---|---|
| torch / MPS fp32 (start) | 21.2× | — | clean |
| MLX fp32 | 4.71× | 369 | clean |
| MLX fp16 | 10.6× | 845 | — (slower) |
| **MLX Q8** | **2.01×** | 158 | **lossless** |
| **MLX mixed-Q4** | **1.65×** | 128 | borderline |

**21.2× → 1.65×: a ~12.8× end-to-end speedup**, with a lossless 2× option.

---

## Deeper validation: 5 styles, 1,117 frames, teacher-forced

The frame-0 harness validates a single forward pass. To trust the port for *real* use we went
much further: a **teacher-forced, full-sequence logit comparison** across **5 varied prompts**
(conversational, expressive, a 31s narration, a technical read, and a 33s monologue — chosen to
span tone, speaker, and length), totaling **1,117 frames**.

The method: on the CUDA box, generate each clip while recording, *at every frame*, the sampled
32 codes plus the `c0`/`c1` logits. On the Mac, **feed the exact CUDA token sequence through the
MLX model** — the model never samples, it consumes the recorded codes — and compare the MLX
logits against the CUDA logits frame by frame. Because the inputs are identical and logits are
deterministic, a correct port must match across the *entire* utterance. Crucially, this exercises
the **KV-cache path over hundreds of frames**, which frame-0 never touched.

| Precision | c0 cosine (mean / min) | c0 top-1 | c1 cosine (mean) | c1 top-1 |
|---|---|---|---|---|
| **fp32** | **0.99987 / 0.99893** | **98.6%** | 0.99995 | 96.1% |
| **Q8** | **0.99986 / 0.99881** | **99.0%** | 0.99991 | 95.0% |
| mixed-Q4 | 0.99721 / 0.98602 | 93.3% | 0.99863 | 86.2% |

Two findings worth stating:

- **No drift over long contexts.** The 389- and 418-frame clips are as clean as the 81-frame
  ones — cosine holds at 0.9999 to the last frame. The cache is correct over hundreds of steps.
- **fp32 and Q8 are effectively exact; mixed-Q4 is a measurable trade.** The ~1% of top-1 misses
  for fp32/Q8 are near-tied candidates flipping under bf16(CUDA)-vs-fp32(MLX) rounding — cosine
  0.9999 confirms the distributions are identical. mixed-Q4 stays in the acceptable >0.99 cosine
  band but its top-1 falls to ~93%/86%, i.e. roughly one sampled token in ten diverges. **Q8 is
  the lossless default even over 30s; mixed-Q4 is for when you need the speed.**

(A practical aside from capturing the references: loading the fp32 32 GB checkpoint the stock way
peaks at ~64 GB CPU RAM — 32 GB random-init model plus a 32 GB state_dict copy — which OOM-kills
a 62 GB cloud box, and RunPod blocks adding swap. The fix is to build the model directly on the
GPU in bf16 and stream weights one tensor at a time, so CPU RAM never holds the full model.)

---

## Same character on both backends: a voice-identity test

Logits prove the math. But a separate, human-meaningful question remained: if you have a
*character* — a voice with a recognizable vibe — does it survive the port? Is MLX-Jane the same
person as CUDA-Jane?

There's a subtlety that shapes the whole experiment: **the speaker tag `[0]`/`[1]` is not a
stored identity** — it's a turn-marker, and without an audio prompt the model invents a fresh
voice per seed. And **seeds don't transfer across backends** (`torch.manual_seed` and
`mx.random.seed` are different generators). So you can't pin "the same character" on both sides
with a number. The only portable anchor is a **reference voice clip** — i.e. voice cloning. (This
test therefore also exercises, and validates, voice cloning on the MLX side: Mimi-encode the
reference, assemble `[ref-text | ref-audio | new-text]` context, generate.)

The design: generate **5 distinct reference voices** on CUDA, then on *each* backend voice-clone
every reference and speak one shared line **twice** (temp 0.9). Score all clips with a pretrained
speaker-verification embedding (Resemblyzer) — cosine similarity is the "same person?" meter —
with three baselines: a backend's own take-to-take variation (the natural floor), the
cross-backend difference (the question), and different-character pairs (the "different person"
calibration).

| Speaker-similarity (mean over 5 characters) | cosine |
|---|---|
| within-CUDA (take 1 vs take 2) | 0.886 |
| within-MLX (take 1 vs take 2) | 0.889 |
| **cross-backend (CUDA vs MLX, same character)** | **0.882** |
| between-character (different people) | 0.776 |

**The cross-backend similarity (0.882) is statistically identical to a backend's own
take-to-take variation (0.886).** In plain terms: MLX-Jane differs from CUDA-Jane *no more than
CUDA-Jane's two takes differ from each other.* The port adds **no measurable identity drift**
beyond the model's natural wobble. (The different-character floor sits high at 0.776 because all
voices come from one model and share acoustic DNA — so the decisive signal is *cross ≈ within*,
not the absolute gap.) Reference adherence was equal on both sides (ref→CUDA 0.820, ref→MLX
0.854). And exactly as the architecture predicts: **identity holds, performance breathes** — the
same character's pitch and tempo vary take-to-take on *both* backends (the sampling), in the same
ranges; the *who* is locked, the *how* is free.

The honest framing: this proves **same identity, no extra cross-backend drift** — the strongest
claim *independent* generation allows (a bit-identical take is impossible by sampling, even twice
on CUDA without a frozen seed; that case is covered by the teacher-forced code-replay above).
Anchor a character with a reference, and it's the same person on MLX as on CUDA.

---

## The real-time wall (an honest ending)

Crossing the magic 1.0× (real-time) did **not** pan out, and the reason is worth documenting.

Profiling one mixed-Q4 frame: **backbone 22.7 ms + decoder (31 passes) 83.4 ms.** The decoder
is 79% of the frame. Of that, ~40% is irreducible **bandwidth** — an autoregressive 32-deep
RVQ decoder must read its weights 31× per frame — and ~60% is **GPU kernel-launch overhead**
(~1,700 tiny launches per frame).

We tried the obvious levers:

- **Async pipelining + dropping the per-frame EOS sync:** no change (1.65 → 1.76×). The loop is
  genuinely sequential and compute-bound, not Python/sync-bound.
- **A compiled, fixed-cache decoder:** to use `mx.compile` (which needs static shapes) on a
  growing KV cache, we wrote a fixed-size (depth-32) decoder that writes at a *traced* position
  via masking and gathers RoPE at that position, so it compiles **once** and reuses across all
  31 steps. It is **numerically correct** (codes match the naive decoder exactly) and 1.19×
  faster in isolation — but that gain **vanished in the full pipeline** (RTF 1.67×). Fusing the
  8 layers within a step traded launch overhead for extra compute (fixed-width attention over
  all 32 slots every step), and the weight bandwidth is unmoved.

So we banked it. Getting under 1.0× for an 8B backbone plus a 32-deep RVQ decoder at batch=1 on
an M4 Pro would need genuinely deeper work — a fully-unrolled single-graph compiled frame with
in-graph sampling, custom Metal kernels, or architectural changes (fewer codebooks, speculative
decoding). All high-effort, uncertain payoff.

## Where it landed

An 8B emotive TTS model running **entirely on a Mac**, validated against the CUDA reference at
the logit level — not on one frame, but across **5 styles and 1,117 frames** including two 30s+
clips — **~12.8× faster** than the naive MPS path, with a **lossless 2× real-time-adjacent**
mode (Q8) and a faster borderline-quality mode (mixed-Q4). Not real-time — but a genuinely
usable, private, local speech model on hardware that the reference code refused to even try.

The validation harness is the quiet hero: every "it worked" above is a number, not a vibe.
