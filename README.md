# MisoTTS on Apple Silicon (MLX)

Run [MisoTTS](https://github.com/MisoLabsAI/MisoTTS), Miso Labs' 8B CSM-style
text-to-dialogue model, locally on an Apple Silicon Mac. It's ported to Apple's
[MLX](https://github.com/ml-explore/mlx) framework, so it runs on the Mac GPU with
no CUDA and no cloud.

> Unofficial port. MisoTTS and its weights belong to Miso Labs (Kamino Learning, Inc.).
> See [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE). Not affiliated with or endorsed by Miso Labs.

## Why this exists

MisoTTS ships CUDA only. This repo reimplements the inference stack on MLX so the 8B
backbone, the 300M depth decoder, and the Kyutai Mimi codec all run on the Mac's
unified-memory GPU. It also ships the tooling and the validation used to confirm the
port matches the original.

## Is it the same model? Verified yes.

The MLX port was checked against the CUDA stack one component at a time (teacher-forced
on saved CUDA reference traces, with Mimi compared against `moshi` on CPU):

| Stage | MLX vs CUDA |
|---|---|
| Sampling (`sample_topk`) | identical (divide by temp, top-k mask, sample from softmax) |
| Backbone + codebook-0 logits | KL about 0.0008, entropy 0.940 vs 0.940 |
| Mimi encode (the voice anchor) | codebook-0 100% match, 98.8% overall |
| Mimi decode (codes to audio) | waveform correlation 1.0000 |

So the Apple Silicon path loses nothing numerically. The validation scripts live in
`src/` (`compare_trace_mlx.py`, `compare_frame0.py`, `clone_experiment_*.py`, `score_identity.py`).

## Speed (M4 Pro mini, 64 GB)

Real-time factor is compute seconds divided by audio seconds, so lower is faster.

| Path | RTF |
|---|---|
| MPS fp32 (naive torch on Mac) | 21.2x |
| MLX fp32 | 4.71x |
| MLX Q8 | 2.01x |
| MLX mixed-Q4 | 1.65x |

Real-time (under 1.0x) is not reached. The depth decoder runs 31 sequential passes per
frame, which is roughly 79% of the time and is launch- and bandwidth-bound.

## Quality note: prefer fp32

Q8 matches the logits but it is not perceptually lossless. In free-running generation it
audibly degrades against fp32. Use fp32 for quality, and Q8 or mixed-Q4 when you want
speed for a quick demo.

## Install

You need an Apple Silicon Mac and the MisoTTS weights (downloaded from
[`MisoLabs/MisoTTS`](https://huggingface.co/MisoLabs/MisoTTS) on first run, about 32 GB).

```bash
uv venv && source .venv/bin/activate     # or: python3 -m venv .venv
uv pip install -r requirements-mlx.txt
```

The text tokenizer mirrors Llama-3.2. If the gated `meta-llama` repo blocks you, the
code falls back to the ungated `unsloth/Llama-3.2-1B` mirror.

## Usage

One-shot CLI:

```bash
python src/run_misotts_mlx.py --text "Hello from Apple Silicon." --dtype fp32 --ms 6000
# voice cloning:
python src/run_misotts_mlx.py --text "..." --ref-audio voice.wav --ref-text "what the clip says"
# speed modes: --bits 8 (Q8), or --quant mixed (mixed-Q4)
```

Interactive wizard (loads the model once, then generate clip after clip):

```bash
./miso-mlx
```

## Working with the model

MisoTTS is a high-variance conversational model. A few things that help a lot:

* Avoid letter-dotted or trailing tokens. `M.L.X.`, `...`, and a mid-phrase double dash
  can derail generation. Spell acronyms spaced (`M L X`) and keep punctuation plain.
* Give it context. A single short line off a 5-second reference tends to drift, and the
  voice or pitch can wander mid-sentence (this happens on CUDA too). Priming with a
  same-voice turn first holds it steady.
* Generate a few and keep the best. Variance is real. A single-voice consistency check
  (3-window speaker embedding) makes a good automatic filter.

## Layout

* `src/` holds the code: the MLX model (`misotts_mlx.py`), the runner
  (`run_misotts_mlx.py`), the wizard (`miso_cli.py`), the validation scripts, and the
  vendored upstream CUDA files (`generator.py`, `models.py`, `moshi_compat.py`,
  `watermarking.py`, `run_misotts.py`).
* `miso-mlx` is the wizard launcher.

## Credits and license

Port and tooling by the repository author. MisoTTS by Miso Labs (Kamino Learning, Inc.).
Both are released under the Modified MIT License. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). All generated audio carries a silentcipher watermark marking it as
AI-generated, per upstream.
