#!/usr/bin/env python3
"""
compare_frame0.py  --  validation contract for the port.

Compares a reference capture (from the CUDA box) against a candidate capture
(from the MPS quick-win or the MLX port).

  python compare_frame0.py reference candidate

PASS criteria (rules of thumb; logits are compared after casting to float64):
  - c0 / c1 top-1 token MATCH                              -> required
  - c0 / c1 top-5 overlap >= 4 / 5                          -> required
  - cosine similarity of logits  > 0.999                    -> strong pass
  - cosine 0.99-0.999                                       -> acceptable (bf16-vs-fp32 noise)
  - cosine < 0.99 or top-1 mismatch                         -> bug in the port

Note: reference is bf16 by default and the MPS candidate is fp32, so an exact
max|Δ| match is NOT expected. Direction (cosine) and argmax agreement are what matter.
"""
import argparse
import os

import numpy as np


def load(d: str, n: str) -> np.ndarray:
    return np.load(os.path.join(d, n + ".npy"))


def stats(a: np.ndarray, b: np.ndarray, name: str) -> None:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        print(f"{name:>10}: SHAPE MISMATCH {a.shape} vs {b.shape}  <-- inputs differ, fix first")
        return
    max_d = float(np.max(np.abs(a - b)))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cos = float(a @ b / denom)
    flag = "OK " if cos > 0.99 else "!! "
    print(f"{flag}{name:>10}: max|Δ|={max_d:.4e}  cosine={cos:.6f}")


def topk_agree(a: np.ndarray, b: np.ndarray, k: int, name: str) -> None:
    a = a.ravel()
    b = b.ravel()
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    overlap = len(ta & tb)
    top1 = int(np.argmax(a)) == int(np.argmax(b))
    flag = "OK " if (top1 and overlap >= max(1, k - 1)) else "!! "
    print(f"{flag}{name:>10}: top-{k} overlap {overlap}/{k}  top1_match={top1} "
          f"(ref={int(np.argmax(a))}, cand={int(np.argmax(b))})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="reference dir (CUDA capture)")
    ap.add_argument("cand", help="candidate dir (MPS/MLX capture)")
    args = ap.parse_args()

    # Sanity: did both consume identical input tokens?
    try:
        ri = load(args.ref, "input_tokens")
        ci = load(args.cand, "input_tokens")
        if ri.shape == ci.shape and np.array_equal(ri, ci):
            print(f"input_tokens: identical (seq_len={ri.shape[1]})  ✓")
        else:
            print("input_tokens: DIFFER — you used different --text/--speaker. Re-capture with the same args.")
    except FileNotFoundError:
        print("input_tokens: not found (older capture?)")

    print("-" * 60)
    for n in ["last_h", "c0_logits", "c1_logits"]:
        stats(load(args.ref, n), load(args.cand, n), n)
    print("-" * 60)
    topk_agree(load(args.ref, "c0_logits"), load(args.cand, "c0_logits"), 5, "c0_logits")
    topk_agree(load(args.ref, "c1_logits"), load(args.cand, "c1_logits"), 5, "c1_logits")


if __name__ == "__main__":
    main()
