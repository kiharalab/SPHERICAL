#!/usr/bin/env python3
"""
Compare the pixel-density distribution of ONE real class-average image
against ONE synthetic class-average image, restricted to the UNION of
"bright" (foreground) regions in both images.

Pipeline:
  1. Load real + synthetic images (with pixel size from header).
  2. Clip negative values in the REAL image to 0.
  3. Resample synthetic image so its pixel size AND box size match the real image.
  4. Align the synthetic image to the real image using a log-polar registration
     for ROTATION ONLY (no scale correction, since box/pixel size already match).
  5. Threshold each image (Otsu, or mean+k*std) to find its own bright region,
     then take the UNION of the two binary masks -> this is the region used
     for comparison.
  6. Compute normalized cross-correlation (NCC) between real and aligned
     synthetic, restricted to pixels inside the mask. (NCC printed to console.)
  7. Plot: Real | Real+its own mask outline | Synthetic | Synthetic+its own
     mask outline | Real+Synthetic blend + union mask outline | Density histogram (masked).

Example
-------
python single_image_compare.py \
    --real /path/to/real/classes.mrcs      --real-idx 0 \
    --syn  /path/to/synthetic/ds_042/avg.mrcs --syn-idx 10 \
    --out  one_vs_one.png
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
from scipy.ndimage import zoom
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, warp, AffineTransform
from skimage.filters import threshold_otsu


# ── I/O helpers ──────────────────────────────────────────────────────────

def resolve_file(path):
    """Return a .mrcs path. If `path` is a dir, use the first .mrcs found."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for ext in ("*.mrcs", "*.mrc"):
            hits = sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"No .mrcs/.mrc file at: {path}")


def load_image(path, idx):
    """Load a single 2D image (index `idx`) from a .mrcs stack, plus its pixel size (Å/px)."""
    fpath = resolve_file(path)
    with mrcfile.open(fpath, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        px = float(mrc.voxel_size.x)
        if px <= 0:
            print(f"  [warn] {fpath} has no valid pixel size in header (got {px}); "
                  f"treating as 1.0 Å/px")
            px = 1.0

    if data.ndim == 2:                      # already a single image
        if idx not in (0, -1):
            print(f"  [note] {fpath} holds a single image; using it (ignored idx={idx})")
        img = data
    else:
        n = data.shape[0]
        if idx >= n or idx < -n:
            raise IndexError(f"{fpath} has {n} images; index {idx} out of range")
        img = data[idx]
    return img, fpath, px


# ── pixel-size / box-size matching ──────────────────────────────────────

def match_pixel_and_box_size(real_img, real_px, syn_img, syn_px):
    """
    Resample syn_img so its pixel size matches real_px, then center-crop/pad
    it so its box size (shape) matches real_img's shape.
    """
    scale = syn_px / real_px
    if not np.isclose(scale, 1.0, atol=1e-4):
        print(f"  [info] rescaling synthetic image: syn_px={syn_px:.4g} A/px, "
              f"real_px={real_px:.4g} A/px, zoom factor={scale:.4g}")
        syn_img = zoom(syn_img, zoom=scale, order=3)
    else:
        print("  [info] pixel sizes already match; skipping rescale")

    target_shape = real_img.shape
    syn_img = _center_crop_or_pad(syn_img, target_shape)

    return syn_img


def _center_crop_or_pad(img, target_shape):
    """Center-crop (if too big) and/or zero-pad (if too small) `img` to `target_shape`."""
    out = img
    crop_slices = []
    for cur, tgt in zip(out.shape, target_shape):
        if cur > tgt:
            start = (cur - tgt) // 2
            crop_slices.append(slice(start, start + tgt))
        else:
            crop_slices.append(slice(0, cur))
    out = out[tuple(crop_slices)]

    pad_widths = []
    for cur, tgt in zip(out.shape, target_shape):
        if cur < tgt:
            total = tgt - cur
            before = total // 2
            after = total - before
            pad_widths.append((before, after))
        else:
            pad_widths.append((0, 0))
    if any(p != (0, 0) for p in pad_widths):
        out = np.pad(out, pad_widths, mode="constant", constant_values=0.0)

    assert out.shape == target_shape, f"shape mismatch after crop/pad: {out.shape} vs {target_shape}"
    return out


# ── log-polar rotation alignment ────────────────────────────────────────

def correct(img, angle_deg, scale_factor, output_shape):
    """Apply a rotation (+ optional scale) affine warp about image center."""
    h, w = output_shape
    cy, cx = h / 2.0, w / 2.0
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    M = np.array([
        [ cos_a * scale_factor,  sin_a * scale_factor, cx * (1 - cos_a * scale_factor) - cy * sin_a * scale_factor],
        [-sin_a * scale_factor,  cos_a * scale_factor, cy * (1 - cos_a * scale_factor) + cx * sin_a * scale_factor],
        [0, 0, 1]
    ])
    tform = AffineTransform(matrix=M)
    return warp(img, tform, output_shape=output_shape, order=3, preserve_range=True)


def align_rotation_only(image, query, radius=None):
    """
    Log-polar registration for ROTATION ONLY (no scale correction) — box/pixel
    size are assumed already matched between `image` and `query`.
    Returns (rotated_query, angle_deg).
    """
    if radius is None:
        radius = min(image.shape) // 2

    image_polar = warp_polar(image, radius=radius, scaling="log")
    query_polar = warp_polar(query, radius=radius, scaling="log")
    shifts, _, _ = phase_cross_correlation(
        image_polar, query_polar, upsample_factor=20, normalization=None
    )
    shiftr = shifts[0]  # rotation component only; ignore shiftc (scale)
    rotated = correct(query, shiftr, scale_factor=1.0, output_shape=image.shape)
    return rotated, shiftr


# ── bright-region masking ───────────────────────────────────────────────

def bright_mask(img, method="otsu", std_factor=1.0):
    """Return a boolean mask of the 'bright' foreground pixels in img."""
    if method == "otsu":
        thresh = threshold_otsu(img)
    elif method == "mean_std":
        thresh = img.mean() + std_factor * img.std()
    else:
        raise ValueError(f"Unknown mask method: {method}")
    return img > thresh, thresh


def union_bright_mask(real_img, syn_img, method="otsu", std_factor=1.0):
    """Union of bright-region masks from real and synthetic images."""
    mask_real, real_thresh = bright_mask(real_img, method, std_factor)
    mask_syn, syn_thresh = bright_mask(syn_img, method, std_factor)
    mask = mask_real | mask_syn
    return mask, mask_real, mask_syn, real_thresh, syn_thresh


# ── NCC ──────────────────────────────────────────────────────────────────

def ncc_masked(a, b, mask):
    """Zero-mean normalized cross-correlation between a and b, restricted to mask."""
    a_vals = a[mask].astype(np.float64)
    b_vals = b[mask].astype(np.float64)
    a_vals = a_vals - a_vals.mean()
    b_vals = b_vals - b_vals.mean()
    denom = np.sqrt((a_vals ** 2).sum()) * np.sqrt((b_vals ** 2).sum())
    if denom == 0:
        return np.nan
    return (a_vals * b_vals).sum() / denom


# ── main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True, help="Real .mrcs file (or dir)")
    p.add_argument("--syn", required=True, help="Synthetic .mrcs file (or dir)")
    p.add_argument("--real-idx", type=int, default=0, help="Image index in real stack")
    p.add_argument("--syn-idx", type=int, default=10, help="Image index in synthetic stack")
    p.add_argument("--bins", type=int, default=60, help="Histogram bins")
    p.add_argument("--radius", type=int, default=None,
                   help="Log-polar radius (default: min(box_size)//2)")
    p.add_argument("--mask-method", choices=["otsu", "mean_std"], default="otsu",
                   help="Thresholding method to find bright regions (default: otsu)")
    p.add_argument("--std-factor", type=float, default=1.0,
                   help="Only used if --mask-method=mean_std: thresh = mean + factor*std")
    p.add_argument("--out", default="one_vs_one.png", help="Output PNG path")
    args = p.parse_args()

    real_img, real_path, real_px = load_image(args.real, args.real_idx)
    syn_img, syn_path, syn_px = load_image(args.syn, args.syn_idx)

    print(f"  real: {os.path.basename(real_path)}  shape={real_img.shape}  px={real_px:.4g} A/px")
    print(f"  syn : {os.path.basename(syn_path)}  shape={syn_img.shape}  px={syn_px:.4g} A/px")

    # --- 1. clip negative values in the REAL image to 0 ---
    n_neg = int(np.sum(real_img < 0))
    if n_neg > 0:
        frac = n_neg / real_img.size
        print(f"  [info] clipping {n_neg} negative pixels ({frac:.2%}) in real image to 0")
    real_img = np.clip(real_img, 0, None)

    # --- 2. match synthetic pixel size + box size to real ---
    syn_img = match_pixel_and_box_size(real_img, real_px, syn_img, syn_px)
    print(f"  syn (after matching): shape={syn_img.shape}  px={real_px:.4g} A/px (now matches real)")

    # --- 3. log-polar rotation-only alignment ---
    syn_aligned, angle = align_rotation_only(real_img, syn_img, radius=args.radius)
    print(f"  [info] rotation-only alignment: angle={angle:.2f} deg")

    # --- 4. union bright-region mask ---
    mask, mask_real, mask_syn, real_thresh, syn_thresh = union_bright_mask(
        real_img, syn_aligned, method=args.mask_method, std_factor=args.std_factor)
    coverage = mask.mean()
    real_coverage = mask_real.mean()
    syn_coverage = mask_syn.mean()
    print(f"  [info] mask method={args.mask_method}  real_thresh={real_thresh:.4g}  "
          f"syn_thresh={syn_thresh:.4g}")
    print(f"  [info] real mask coverage: {real_coverage:.2%}  |  "
          f"syn mask coverage: {syn_coverage:.2%}  |  union coverage: {coverage:.2%} "
          f"({mask.sum()} / {mask.size})")
    if mask.sum() < 10:
        print("  [warn] mask has very few pixels; NCC below may be unstable")

    # --- 5. NCC restricted to mask ---
    ncc_val = ncc_masked(real_img, syn_aligned, mask)
    print(f"  [info] NCC(real, aligned synthetic), masked to bright union = {ncc_val:.4f}")

    # --- 6. distributions, restricted to mask ---
    real_vals = real_img[mask]
    real_vals = real_vals[np.isfinite(real_vals)]
    syn_vals = syn_aligned[mask]
    syn_vals = syn_vals[np.isfinite(syn_vals)]

    lo = min(real_vals.min(), syn_vals.min())
    hi = max(real_vals.max(), syn_vals.max())
    edges = np.linspace(lo, hi, args.bins + 1)

    # 6 panels: Real | Real+own outline | Synthetic | Synthetic+own outline |
    #           Real+Synthetic blend + union mask outline | Density histogram
    fig, (axr, axro, axs, axso, axm, axh) = plt.subplots(1, 6, figsize=(27, 4.5))

    axr.imshow(real_img, cmap="gray")
    axr.set_title(f"Real image, neg->0")
    axr.axis("off")

    # real image with ONLY its own otsu mask outline
    axro.imshow(real_img, cmap="gray")
    axro.contour(mask_real, levels=[0.5], colors="dodgerblue", linewidths=1.4)
    axro.set_title(f"Real + own mask")
    axro.axis("off")

    axs.imshow(syn_aligned, cmap="gray")
    axs.set_title(f"Synthetic image, aligned")
    axs.axis("off")

    # synthetic image with ONLY its own otsu mask outline
    axso.imshow(syn_aligned, cmap="gray")
    axso.contour(mask_syn, levels=[0.5], colors="orangered", linewidths=1.4)
    axso.set_title(f"Synthetic + own mask")
    axso.axis("off")

    # union mask outline panel: background = blend of BOTH real + synthetic
    # (each independently normalized to [0,1] so intensity scales don't dominate
    # the blend), with ONLY the union contour drawn on top.
    real_norm = (real_img - real_img.min()) / (np.ptp(real_img) + 1e-12)
    syn_norm = (syn_aligned - syn_aligned.min()) / (np.ptp(syn_aligned) + 1e-12)
    blend = 0.5 * real_norm + 0.5 * syn_norm
    axm.imshow(blend, cmap="gray")
    axm.contour(mask, levels=[0.5], colors="lime", linewidths=1.6)
    axm.set_title(f"Real+Synthetic+union mask")
    axm.axis("off")

    axh.hist(real_vals, bins=edges, density=False, alpha=0.55,
             color="#2166ac", label=f"Real image, masked")
    axh.hist(syn_vals, bins=edges, density=False, alpha=0.55,
             color="#b2182b", label=f"Synthetic image, masked")
    axh.set_xlabel("Pixel density value")
    axh.set_ylabel("Count of Pixels")
    axh.set_title("Density Distribution (masked)")
    axh.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"Saved figure -> {args.out}")
    print(f"  real (masked): {os.path.basename(real_path)} img {args.real_idx} "
          f"-> mean={real_vals.mean():.4g} std={real_vals.std():.4g}")
    print(f"  syn  (masked): {os.path.basename(syn_path)} img {args.syn_idx} "
          f"-> mean={syn_vals.mean():.4g} std={syn_vals.std():.4g}")
    print(f"  NCC (real vs aligned synthetic, masked) = {ncc_val:.4f}")


if __name__ == "__main__":
    main()