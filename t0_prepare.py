#!/usr/bin/env python3
"""
TIER 0 -- data integrity. Run this before anything else.

T1  polarity probe        : is protein bright in one stack and dark in the other?
T2  resample synth        : 168px @ 1.0 A/px  ->  128px @ 2.0625 A/px
T3  side-by-side QC plot  : can YOU tell these are the same molecule?

Writes corrected synth stacks to --out and a QC png.
"""
import argparse, glob, os
import numpy as np
import mrcfile
from scipy.ndimage import zoom
from scipy.stats import skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        d = np.array(m.data, dtype=np.float32)
    return d[None] if d.ndim == 2 else d


def polarity(X, name):
    """Protein sits in the middle; background at the corners. Compare them."""
    L = X.shape[-1]; c = L // 2; q = max(2, L // 6)
    core = float(X[:, c - q:c + q, c - q:c + q].mean())
    edge = float(np.concatenate([X[:, :q].ravel(), X[:, -q:].ravel()]).mean())
    bright = core > edge
    print(f"  {name:6s} skew {skew(X.ravel()):+6.2f}   center {core:+8.4f}   "
          f"edge {edge:+8.4f}   -> protein is {'BRIGHT' if bright else 'DARK'}")
    return bright


def fit_box(im, box):
    """center crop or zero-pad to box x box"""
    L = im.shape[-1]
    if L > box:
        o = (L - box) // 2
        return im[o:o + box, o:o + box]
    p = box - L
    a, b = p // 2, p - p // 2
    return np.pad(im, ((a, b), (a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--out", required=True, help="dir for resampled synth")
    ap.add_argument("--real-px", type=float, default=2.0625)
    ap.add_argument("--synth-px", type=float, default=1.0)
    ap.add_argument("--box", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="only process N files (0=all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.synth_dir, "*.mrc*")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no mrc files in {args.synth_dir}")

    R = load(args.real)
    S0 = load(files[0])

    # ---------------- T1 : polarity ----------------
    print("\n[T1] polarity probe (raw)")
    r_bright = polarity(R, "real")
    s_bright = polarity(S0, "synth")
    flip = (r_bright != s_bright)
    print(f"\n  ==> polarity {'MISMATCH -- synth will be negated' if flip else 'agrees'}")

    # ---------------- T2 : resample ----------------
    f = args.synth_px / args.real_px
    print(f"\n[T2] resampling {len(files)} files  zoom={f:.4f}  "
          f"({S0.shape[-1]} -> {int(round(S0.shape[-1]*f))} px, pad/crop to {args.box})")

    for k, p in enumerate(files):
        S = load(p)
        if flip:
            S = -S
        Z = zoom(S, (1, f, f), order=3)
        out = np.stack([fit_box(z, args.box) for z in Z]).astype(np.float32)
        with mrcfile.new(os.path.join(args.out, os.path.basename(p)), overwrite=True) as m:
            m.set_data(out)
            m.voxel_size = args.real_px
        if k % 50 == 0:
            print(f"    [{k:4d}/{len(files)}] -> {out.shape}")
    print(f"  ==> wrote {len(files)} files to {args.out}")

    # ---------------- T3 : eyeball ----------------
    Sc = load(os.path.join(args.out, os.path.basename(files[0])))
    print("\n[T3] polarity re-check (corrected)")
    polarity(R, "real")
    polarity(Sc, "synth")

    n = 8
    fig, ax = plt.subplots(2, n, figsize=(2 * n, 4.6))
    for i in range(n):
        ax[0, i].imshow(R[i % len(R)], cmap="gray"); ax[0, i].axis("off")
        ax[1, i].imshow(Sc[i % len(Sc)], cmap="gray"); ax[1, i].axis("off")
    ax[0, 0].set_ylabel("real"); ax[1, 0].set_ylabel("synth")
    fig.suptitle("T3: real (top) vs corrected synth (bottom) -- same molecule?")
    plt.tight_layout()
    qc = os.path.join(args.out, "..", "T3_qc.png")
    plt.savefig(qc, dpi=110)
    print(f"  ==> QC figure: {os.path.abspath(qc)}")
    print("\n  LOOK AT IT. If you cannot tell these are the same molecule,")
    print("  no algorithm will either, and the problem is upstream.\n")


if __name__ == "__main__":
    main()