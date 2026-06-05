# MisoTTS on Apple Silicon (MLX)

Run [MisoTTS](https://github.com/MisoLabsAI/MisoTTS) — Miso Labs' 8B, CSM-style
text-to-dialogue model — **locally on an Apple Silicon Mac**, ported to Apple's
[MLX](https://github.com/ml-explore/mlx) framework. No CUDA, no cloud.

> Unofficial port. MisoTTS and its weights are by Miso Labs (Kamino Learning, Inc.);
> see [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE). Not affiliated with or endorsed by Miso Labs.

---

## Why this exists

MisoTTS ships CUDA-only. This repo reimplements the inference stack on MLX so the
8B backbone + 300M depth decoder + Kyutai Mimi codec run on the Mac's unified-memory
GPU, and adds the tooling and validation to prove the port is faithful.

## Is it actually the same model? Yes — verified.

The MLX port was checked against the original CUDA stack **component by component**
(teacher-forced on saved CUDA reference traces; Mimi compared against `moshi` on CPU):

| Stage | MLX vs CUDA |
|---|---|
| Sampling (`sample_topk`) | identical (÷temp → top-k mask → sample softmax) |
| Backbone + codebook-0 logits | KL ≈ **0.0008**, entropy 0.940/0.940 — distributions match |
| Mimi **encode** (voice anchor) | codebook-0 **100%** match, 98.8% overall |
| Mimi **decode** (codes → audio) | waveform correlation **1.0000** |

The Apple-Silicon path loses nothing numerically. (Scripts: `compare_trace_mlx.py`,
`compare_frame0.py`, `clone_experiment_*.py`, `score_identity.py`.)

## Speed (M4 Pro mini, 64 GB) — real-time factor (compute-sec ÷ audio-sec, lower = faster)

| Path | RTF | vs native CPU path |
|---|---|---|
| MPS fp32 (naive torch on Mac) | 21.2× | 1× |
| **MLX fp32** | **4.71×** | ~4.5× faster |
| MLX Q8 | 2.01× | ~10.5× faster |
| MLX mixed-Q4 | 1.65× | ~12.8× faster |

Real-time (<1.0×) isn't reached — the depth decoder runs 31 sequential passes per
frame (~79% of the time, launch/bandwidth bound).

## Quality note: use fp32

Q8 matches the logits but is **not** perceptually lossless — in free-running generation
it audibly degrades vs fp32. **Use fp32 for quality; Q8/mixed-Q4 for speed demos.**

## Install

Requires an Apple Silicon Mac and the MisoTTS weights (auto-downloaded from
[`MisoLabs/MisoTTS`](https://huggingface.co/MisoLabs/MisoTTS) on first run; ~32 GB).

```bash
uv venv && source .venv/bin/activate     # or python3 -m venv .venv
uv pip install -r requirements-mlx.txt
```

The text tokenizer mirrors Llama-3.2; if the gated `meta-llama` repo blocks you, the
code falls back to the ungated `unsloth/Llama-3.2-1B` mirror.

## Usage

**One-shot CLI:**
```bash
python run_misotts_mlx.py --text "Hello from Apple Silicon." --dtype fp32 --ms 6000
# voice cloning:
python run_misotts_mlx.py --text "..." --ref-audio voice.wav --ref-text "what the clip says"
# speed modes: --bits 8  (Q8)   |   --quant mixed  (mixed-Q4)
```

**Interactive wizard** (loads the model once, generate clip after clip):
```bash
./miso-mlx
```

## Working with the model (hard-won tips)

MisoTTS is a high-variance conversational model. For natural, consistent output:

- **Avoid letter-dotted / trailing tokens.** `M.L.X.`, `...`, and `--` mid-phrase
  derail generation. Spell acronyms spaced — `M L X` — and keep punctuation plain.
- **Condition with context.** A single short line off a 5 s reference drifts (voice/pitch
  can wander mid-utterance — true on CUDA too). Prime with a same-voice turn first.
- **Generate a few, keep the best.** Variance is real; a single-voice consistency check
  (3-window speaker-embedding) makes a good auto-filter.

## Repository layout

- `misotts_mlx.py` — the MLX model + weight remap from the torchtune checkpoint
- `run_misotts_mlx.py` — end-to-end runner (dtype/quant/cloning flags)
- `miso_cli.py` / `miso-mlx` — interactive synthesis wizard
- `capture_*.py`, `compare_*.py`, `clone_experiment_*.py`, `score_identity.py` — validation
- `generator.py`, `models.py`, `moshi_compat.py`, `watermarking.py`, `run_misotts.py` —
  vendored upstream (CUDA reference / shared deps)
- `docs/` — port write-up, run guide, and the porting plan

## Credits & license

Port and tooling © the repository author. MisoTTS © Miso Labs (Kamino Learning, Inc.).
Both released under the **Modified MIT License** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
All generated audio is watermarked (silentcipher) as AI-generated, per upstream.
