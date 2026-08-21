#!/usr/bin/env python3
"""
LightGlue rotation estimation: synthetic templates vs real class averages.

This is a FALSIFICATION TEST, not a production method. The single question it
answers is:

    Does SuperPoint find any keypoints at all on cryo-EM class averages?

If it does not, LightGlue cannot work here, and you have saved yourself a week.
The script therefore reports keypoint counts and match counts as loudly as it
reports angles.

Scale note
----------
Your templates and real averages share a pixel size, so the true scale is 1.0.
We fit a RIGID transform (rotation + translation, scale locked to 1) and
separately report the scale a similarity fit WOULD have produced, purely as a
diagnostic. If that diagnostic comes back far from 1.0, something is wrong with
the data or preprocessing -- do not "fix" it by allowing free scale. That is the
bug that produced NCC=0.57 at scale=4.87 in the old log-polar pipeline.

Usage
-----
    python lightglue_align.py \
        --real  /net/kihara/home/ffarheen/scratch/spherical_2d_matching/cryosparc_P6_J7_020_class_averages.mrc \
        --synth /net/kihara/home/ffarheen/scratch/spherical_2d_matching/all_390_Cryosparced/template_mrcs_links \
        --out   lg_results \
        --upsample 512 \
        --viz 20
"""

import argparse
import glob
import os

import cv2
import mrcfile
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

DEV =  "cpu"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_stack(path):
    """Load an MRC (or MRCS) file as an (N, H, W) float32 array."""
    with mrcfile.open(path, permissive=True) as m:
        d = np.array(m.data, dtype=np.float32)
    if d.ndim == 2:
        d = d[None]
    return d


def load_synth(path):
    """
    Load synthetic templates. `path` may be a single .mrc/.mrcs file or a
    directory of them (e.g. your template_mrcs_links folder).

    Returns (stack, names) where names[i] identifies the source of stack[i].
    """
    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "*.mrc"))
            + glob.glob(os.path.join(path, "*.mrcs"))
        )
        if not files:
            raise SystemExit(f"no .mrc/.mrcs found in {path}")
        stacks, names = [], []
        for f in files:
            s = load_stack(f)
            stacks.append(s)
            base = os.path.basename(f)
            names += [f"{base}:{i}" for i in range(s.shape[0])]
        return np.concatenate(stacks, axis=0), names

    s = load_stack(path)
    base = os.path.basename(path)
    return s, [f"{base}:{i}" for i in range(s.shape[0])]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def prep(img, size, mask_frac=0.9, invert=False):
    """
    Class average -> tensor SuperPoint will accept.

    Steps: optional contrast inversion, soft circular mask, robust percentile
    normalization to [0, 1], upsample (SuperPoint's NMS radius and 8x stride
    starve on small boxes), return (1, 1, S, S) tensor.
    """
    img = img.astype(np.float32)
    if invert:
        img = -img

    L = img.shape[-1]
    yy, xx = np.mgrid[-1:1:L * 1j, -1:1:L * 1j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    inner = mask_frac * 0.9
    m = np.clip((mask_frac - r) / (mask_frac - inner), 0, 1)
    m = (0.5 - 0.5 * np.cos(np.pi * m)).astype(np.float32)

    v = img[m > 1e-3]
    lo, hi = np.percentile(v, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1) * m

    if size and size != L:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    img = np.ascontiguousarray(img, dtype=np.float32)
    return torch.from_numpy(img)[None, None].to(DEV), img


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def rigid_from_matches(p0, p1):
    """
    Kabsch / Umeyama on 2D correspondences.

    Returns (angle_deg, scale_diagnostic, rmse). Scale is REPORTED but the
    angle is computed from the rotation component alone -- we never apply it.
    """
    if len(p0) < 3:
        return None

    c0, c1 = p0.mean(0), p1.mean(0)
    q0, q1 = p0 - c0, p1 - c1

    H = q1.T @ q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt

    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    # scale the similarity fit WOULD have used -- diagnostic only
    denom = (q1 ** 2).sum()
    scale = (S[0] + d * S[1]) / denom if denom > 1e-9 else np.nan

    resid = q0 - (q1 @ R.T)
    rmse = float(np.sqrt((resid ** 2).sum(1).mean()))
    return float(angle % 360.0), float(scale), rmse


def robust_rigid(p0, p1, thresh=3.0):
    """
    RANSAC a partial-affine (rotation+scale+translation), keep the inliers,
    then refit rigid-only on those inliers. Gives an inlier count, which is the
    honest confidence signal here.
    """
    if len(p0) < 3:
        return None
    M, inl = cv2.estimateAffinePartial2D(
        p1.astype(np.float32), p0.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=thresh,
        maxIters=5000, confidence=0.999,
    )
    if M is None or inl is None or int(inl.sum()) < 3:
        return None
    inl = inl.ravel().astype(bool)
    fit = rigid_from_matches(p0[inl], p1[inl])
    if fit is None:
        return None
    angle, scale, rmse = fit
    return angle, scale, rmse, int(inl.sum())


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="real class averages .mrc")
    ap.add_argument("--synth", required=True, help="synthetic .mrc/.mrcs or a directory")
    ap.add_argument("--out", default="lg_results")
    ap.add_argument("--upsample", type=int, default=512,
                    help="resize boxes to this; SuperPoint starves on small images")
    ap.add_argument("--max-kpts", type=int, default=2048)
    ap.add_argument("--det-thresh", type=float, default=0.0005,
                    help="LOWER this if you get no keypoints (default 5e-4)")
    ap.add_argument("--invert-synth", action="store_true",
                    help="flip synthetic contrast polarity")
    ap.add_argument("--viz", type=int, default=0, help="save N match figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    extractor = SuperPoint(
        max_num_keypoints=args.max_kpts,
        detection_threshold=args.det_thresh,
    ).eval().to(DEV)
    matcher = LightGlue(features="superpoint").eval().to(DEV)

    real = load_stack(args.real)
    synth, snames = load_synth(args.synth)
    print(f"real:      {real.shape}")
    print(f"synthetic: {synth.shape}  from {args.synth}")
    print(f"device:    {DEV}")

    # ---- pre-extract features once per image (not per pair) ----
    def feats(stack, invert):
        out, imgs, counts = [], [], []
        for i in range(stack.shape[0]):
            t, vis = prep(stack[i], args.upsample, invert=invert)
            with torch.no_grad():
                f = extractor.extract(t)
            out.append(f)
            imgs.append(vis)
            counts.append(f["keypoints"].shape[1])
        return out, imgs, np.array(counts)

    print("\nextracting SuperPoint keypoints...")
    f_real, im_real, n_real_kp = feats(real, invert=False)
    f_syn, im_syn, n_syn_kp = feats(synth, invert=args.invert_synth)

    # ---- THE ACTUAL TEST -------------------------------------------------
    print(f"\nkeypoints/image  real: median {np.median(n_real_kp):.0f}  "
          f"min {n_real_kp.min()}  max {n_real_kp.max()}")
    print(f"keypoints/image  synth: median {np.median(n_syn_kp):.0f}  "
          f"min {n_syn_kp.min()}  max {n_syn_kp.max()}")

    if np.median(n_real_kp) < 10 or np.median(n_syn_kp) < 10:
        print("\n*** SuperPoint finds essentially no keypoints on this data. ***")
        print("    This is the expected failure: class averages are smooth and")
        print("    textureless. Lowering --det-thresh will manufacture keypoints")
        print("    from noise, which is worse than none. LightGlue is the wrong")
        print("    tool. Stop here.")
        # continue anyway so you can see the numbers

    # ---- match every (real, synth) pair ----
    rows = []
    n_viz = 0
    print(f"\nmatching {len(f_real)} x {len(f_syn)} pairs...")
    for i in range(len(f_real)):
        for j in range(len(f_syn)):
            with torch.no_grad():
                out = matcher({"image0": f_real[i], "image1": f_syn[j]})
            f0, f1, o = rbd(f_real[i]), rbd(f_syn[j]), rbd(out)
            m = o["matches"]
            k0 = f0["keypoints"][m[:, 0]].cpu().numpy()
            k1 = f1["keypoints"][m[:, 1]].cpu().numpy()

            res = robust_rigid(k0, k1) if len(m) >= 3 else None
            if res is None:
                rows.append((i, snames[j], len(m), np.nan, np.nan, np.nan, 0))
            else:
                angle, scale, rmse, ninl = res
                rows.append((i, snames[j], len(m), angle, scale, rmse, ninl))

                if args.viz and n_viz < args.viz and ninl >= 8:
                    save_viz(im_real[i], im_syn[j], k0, k1, angle, scale, ninl,
                             os.path.join(args.out, f"match_r{i:02d}_s{j:03d}.png"))
                    n_viz += 1

    csv = os.path.join(args.out, "lightglue_alignments.csv")
    with open(csv, "w") as f:
        f.write("real_idx,synth,n_matches,angle_deg,scale_diagnostic,rmse,n_inliers\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.3f},{r[4]:.4f},{r[5]:.3f},{r[6]}\n")

    # ---- verdict ----
    nm = np.array([r[2] for r in rows])
    ni = np.array([r[6] for r in rows])
    sc = np.array([r[4] for r in rows])
    ok = ni >= 8

    print(f"\nwrote {csv}")
    print(f"matches/pair:  median {np.median(nm):.0f}")
    print(f"pairs with >=8 RANSAC inliers: {ok.sum()} / {len(rows)}")
    if ok.sum():
        s = sc[ok & np.isfinite(sc)]
        print(f"scale diagnostic on those: median {np.median(s):.3f} "
              f"[{np.percentile(s,5):.2f}, {np.percentile(s,95):.2f}]")
        print("  -> should be ~1.0. If it is not, your two stacks are not on a")
        print("     common pixel size, and THAT is the bug, not the method.")
    else:
        print("  -> no pair produced a usable fit. LightGlue is falsified on this data.")


def save_viz(im0, im1, k0, k1, angle, scale, ninl, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im0, cmap="gray"); ax[0].scatter(k0[:, 0], k0[:, 1], s=4, c="lime")
    ax[1].imshow(im1, cmap="gray"); ax[1].scatter(k1[:, 0], k1[:, 1], s=4, c="lime")
    ax[0].set_title("real"); ax[1].set_title("synthetic")
    fig.suptitle(f"angle={angle:.1f}deg  scale={scale:.3f}  inliers={ninl}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
    