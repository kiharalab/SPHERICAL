#!/usr/bin/env python3
"""
sharpen_mrcs.py
================
Read an .mrcs stack (e.g. 50 cryo-EM particle images) and sharpen / denoise each
image with several methods, then save one comparison figure per image.

Methods
-------
1. Wiener deconvolution      (frequency-domain inverse of a Gaussian PSF)
2. Unsharp masking           (add back the high-frequency detail)
3. High-pass filtering       (FFT Butterworth high-pass)
4. Wavelet denoising         (BayesShrink, soft thresholding)
5. Non-Local Means denoising (extra method: edge-preserving denoise)

For every image in the stack a single figure is written to the output directory
showing the original alongside all five processed versions -> 50 images in ->
50 files out.

Usage
-----
    python sharpen_mrcs.py input.mrcs -o output_dir
    python sharpen_mrcs.py input.mrcs -o output_dir --dpi 150 --cmap gray
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # headless / no display needed
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import unsharp_mask
from skimage.restoration import denoise_wavelet, denoise_nl_means, estimate_sigma


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize(img):
    """Scale image to [0, 1] float for consistent filtering / display."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def gaussian_psf(shape, sigma):
    """Centered normalized Gaussian point-spread function of a given shape."""
    yy, xx = np.indices(shape)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    psf = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)))
    return psf / psf.sum()


# --------------------------------------------------------------------------- #
# Sharpening / denoising methods
# --------------------------------------------------------------------------- #
def wiener_deconvolution(img, sigma=2.0, reg=0.01):
    """
    Wiener deconvolution assuming a Gaussian PSF. Since a real cryo-EM PSF/CTF
    is not supplied here, a Gaussian blur is used as a stand-in model of the
    imaging blur. `reg` is the noise-to-signal regularization term.
    """
    psf = gaussian_psf(img.shape, sigma)
    psf_f = np.fft.fft2(np.fft.ifftshift(psf))
    img_f = np.fft.fft2(img)
    wiener = np.conj(psf_f) / (np.abs(psf_f) ** 2 + reg)
    out = np.real(np.fft.ifft2(img_f * wiener))
    return normalize(out)


def unsharp(img, radius=2.0, amount=1.5):
    """Classic unsharp mask: original + amount * (original - blurred)."""
    return unsharp_mask(img, radius=radius, amount=amount, preserve_range=True)


def highpass(img, cutoff=0.15, order=2):
    """
    FFT Butterworth high-pass filter. `cutoff` is the normalized cutoff
    frequency (0-0.5). Higher cutoff removes more low-frequency content.
    """
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    # Butterworth high-pass transfer function
    with np.errstate(divide="ignore"):
        hp = 1.0 / (1.0 + (cutoff / (radius + 1e-12)) ** (2 * order))
    out = np.real(np.fft.ifft2(np.fft.fft2(img) * hp))
    return normalize(out)


def wavelet(img):
    """Wavelet denoising (BayesShrink, soft thresholding)."""
    return denoise_wavelet(
        img, method="BayesShrink", mode="soft",
        rescale_sigma=True, channel_axis=None,
    )


def nlmeans(img):
    """Non-Local Means denoising (extra method), edge-preserving."""
    sigma_est = float(np.mean(estimate_sigma(img, channel_axis=None)))
    return denoise_nl_means(
        img, h=1.15 * sigma_est, sigma=sigma_est,
        fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None,
    )


METHODS = [
    ("Original",        lambda im: im),
    ("Wiener deconv.",  wiener_deconvolution),
    ("Unsharp mask",    unsharp),
    ("High-pass",       highpass),
    ("Wavelet denoise", wavelet),
    ("Non-Local Means", nlmeans),
]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process_stack(in_path, out_dir, dpi=150, cmap="gray"):
    os.makedirs(out_dir, exist_ok=True)

    with mrcfile.open(in_path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)

    # Ensure we have a stack: shape (N, H, W)
    if data.ndim == 2:
        data = data[None, ...]
    n_images = data.shape[0]
    print(f"Loaded '{in_path}': {n_images} image(s) of size {data.shape[1:]}")

    n_methods = len(METHODS)
    ncols = 3
    nrows = int(np.ceil(n_methods / ncols))

    for i in range(n_images):
        original = normalize(data[i])

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, (name, func) in zip(axes, METHODS):
            result = func(original)
            ax.imshow(result, cmap=cmap)
            ax.set_title(name, fontsize=11)
            ax.axis("off")

        # blank any unused panels
        for ax in axes[n_methods:]:
            ax.axis("off")

        fig.suptitle(f"Image {i:03d} — sharpening comparison", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = os.path.join(out_dir, f"image_{i:03d}_comparison.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 10 == 0 or i == n_images - 1:
            print(f"  processed {i + 1}/{n_images}")

    print(f"Done. {n_images} figure(s) written to '{out_dir}'.")


def main():
    p = argparse.ArgumentParser(description="Sharpen/denoise an .mrcs stack.")
    p.add_argument("input", help="Path to input .mrcs / .mrc file")
    p.add_argument("-o", "--output", default="output", help="Output directory")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    p.add_argument("--cmap", default="gray", help="Matplotlib colormap")
    args = p.parse_args()
    process_stack(args.input, args.output, dpi=args.dpi, cmap=args.cmap)


if __name__ == "__main__":
    main()