"""Compare per-window probabilities written by the image (``--probs``) with reference probabilities.

    python scripts/image_parity.py image_probs.npy reference_probs.npy [--tol 1e-4]

Exit code 0 when the arrays have the same length and the maximum absolute difference is below the tolerance.
"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("reference")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    a, b = np.load(args.image), np.load(args.reference)
    out = {"image": args.image, "reference": args.reference, "n_image": int(len(a)), "n_reference": int(len(b))}
    ok = len(a) == len(b)
    if ok:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        out.update(max_abs_diff=float(diff.max()) if len(diff) else 0.0,
                   mean_abs_diff=float(diff.mean()) if len(diff) else 0.0)
        ok = out["max_abs_diff"] < args.tol
    out.update(tol=args.tol, passed=bool(ok))
    print(json.dumps(out))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
