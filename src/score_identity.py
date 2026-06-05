#!/usr/bin/env python3
"""
score_identity.py  --  objective cross-backend identity + prosody scoring.

Embeds every clip with a pretrained speaker-verification encoder (Resemblyzer) and
reports, PER CHARACTER, speaker-similarity cosines for:
  within-CUDA   cuda take1 vs take2        (CUDA's own take-to-take variation)
  within-MLX    mlx  take1 vs take2        (MLX's own take-to-take variation)
  cross         cuda vs mlx (4 pairs avg)  (the question: does ported-Jane = Jane?)
  ref->CUDA / ref->MLX  how well each backend matches the reference voice
plus a BETWEEN-character baseline (different people) to calibrate the scale.

Verdict logic: if cross ~= within-backend, and all of those >> between-character,
then MLX is the SAME identity as CUDA, not vaguely close.

Also reports prosody (mean F0, F0 std, duration) per clip for tone/tempo.

Usage:  python score_identity.py --refs refs --cuda cuda --mlx mlx
"""
import argparse, glob, itertools, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--cuda", default="cuda")
    ap.add_argument("--mlx", default="mlx")
    args = ap.parse_args()

    from resemblyzer import VoiceEncoder, preprocess_wav
    import librosa

    enc = VoiceEncoder(verbose=False)

    def emb(path):
        return enc.embed_utterance(preprocess_wav(path))

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def prosody(path):
        y, sr = librosa.load(path, sr=16000)
        f0, vflag, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
        f0v = f0[~np.isnan(f0)]
        return (float(np.mean(f0v)) if len(f0v) else 0.0,
                float(np.std(f0v)) if len(f0v) else 0.0,
                len(y) / sr)

    chars = sorted({os.path.basename(p).split("_ref")[0]
                    for p in glob.glob(os.path.join(args.refs, "*_ref.wav"))})
    print(f"[score] characters: {chars}\n")

    E, P = {}, {}
    def add(key, path):
        if os.path.exists(path):
            E[key] = emb(path); P[key] = prosody(path)

    for c in chars:
        add(f"{c}/ref", os.path.join(args.refs, f"{c}_ref.wav"))
        for tk in (1, 2):
            add(f"{c}/cuda{tk}", os.path.join(args.cuda, f"{c}_take{tk}.wav"))
            add(f"{c}/mlx{tk}", os.path.join(args.mlx, f"{c}_take{tk}.wav"))

    print(f"{'char':8} {'within-CUDA':>11} {'within-MLX':>11} {'CROSS c-vs-m':>13} "
          f"{'ref->CUDA':>10} {'ref->MLX':>9}")
    print("-" * 70)
    within_c, within_m, cross_all, refc, refm = [], [], [], [], []
    for c in chars:
        def g(k): return E.get(f"{c}/{k}")
        wc = cos(g("cuda1"), g("cuda2")) if g("cuda1") is not None and g("cuda2") is not None else float("nan")
        wm = cos(g("mlx1"), g("mlx2")) if g("mlx1") is not None and g("mlx2") is not None else float("nan")
        cross = [cos(g(f"cuda{i}"), g(f"mlx{j}")) for i in (1, 2) for j in (1, 2)
                 if g(f"cuda{i}") is not None and g(f"mlx{j}") is not None]
        rc = [cos(g("ref"), g(f"cuda{i}")) for i in (1, 2) if g("ref") is not None and g(f"cuda{i}") is not None]
        rm = [cos(g("ref"), g(f"mlx{i}")) for i in (1, 2) if g("ref") is not None and g(f"mlx{i}") is not None]
        cm = np.mean(cross) if cross else float("nan")
        within_c.append(wc); within_m.append(wm); cross_all += cross; refc += rc; refm += rm
        print(f"{c:8} {wc:11.3f} {wm:11.3f} {cm:13.3f} "
              f"{(np.mean(rc) if rc else float('nan')):10.3f} {(np.mean(rm) if rm else float('nan')):9.3f}")

    # between-character baseline: cos between different characters' cuda take1
    bet = []
    cu1 = {c: E.get(f"{c}/cuda1") for c in chars if E.get(f"{c}/cuda1") is not None}
    for a, b in itertools.combinations(cu1, 2):
        bet.append(cos(cu1[a], cu1[b]))

    print("-" * 70)
    print(f"{'MEAN':8} {np.nanmean(within_c):11.3f} {np.nanmean(within_m):11.3f} "
          f"{np.mean(cross_all):13.3f} {np.mean(refc):10.3f} {np.mean(refm):9.3f}")
    print(f"\nbetween-character baseline (different people): mean {np.mean(bet):.3f} "
          f"(min {np.min(bet):.3f}, max {np.max(bet):.3f})")
    print("\n--- prosody (mean F0 Hz / F0 std / dur s) ---")
    for c in chars:
        for k in ("cuda1", "cuda2", "mlx1", "mlx2"):
            if f"{c}/{k}" in P:
                f0, sd, d = P[f"{c}/{k}"]
                print(f"  {c}/{k:6} F0={f0:6.1f}  std={sd:5.1f}  dur={d:4.1f}s")

    cross_m = np.mean(cross_all); within = np.nanmean(within_c + within_m); betw = np.mean(bet)
    print("\n=== VERDICT ===")
    print(f"cross-backend speaker-sim  = {cross_m:.3f}")
    print(f"within-backend (same take) = {within:.3f}")
    print(f"between-character (diff)    = {betw:.3f}")
    gap = cross_m - betw
    if cross_m >= within - 0.05 and gap > 0.15:
        print("-> MLX clips are the SAME identity as CUDA (cross ~= within, far from between-character).")
    elif cross_m > betw + 0.1:
        print("-> MLX clips are clearly the same character but with more drift than within-backend.")
    else:
        print("-> WARNING: cross-backend similarity not clearly above the different-person baseline.")


if __name__ == "__main__":
    main()
