# cryo-EM Real vs. Synthetic Rotational Alignment — Script Guide

This folder contains a set of scripts for aligning real cryo-EM class
averages against synthetic template projections, comparing several
alignment methods (classical log-polar / exhaustive NCC search, AIRLab
deep-learning registration, ASPIRE Fourier-Bessel alignment, and LightGlue
feature matching), plus standalone utilities for single-pair QC, batch
log-polar NCC scoring, and image sharpening/denoising comparisons.

## Pipeline order

Run scripts in this order:

1. **`t0_prepare.py`** — mandatory first step. Resamples
   the synthetic stacks to match the real data's pixel size/box size.
2. Any of the analysis/comparison scripts below, pointed at the **resampled**
   synth directory produced by step 1.

Everything downstream assumes real and synth images share pixel size and box
size (128 px in the example commands here) — if `t0_prepare.py` hasn't been
run, the other scripts will skip files with a "box mismatch" message.

The three standalone scripts described in sections 8–10 below
(`compare_distribution_real_zero_ncc_mask.py`, `run_logpolar_ncc.py`,
`sharpen_diff_methods.py`) don't require step 1 to have been run first —
each handles its own pixel-size/box-size matching (or doesn't need it), but
they're most useful once you already have real and synthetic stacks on hand.

## Common dependencies

```bash
pip install numpy scipy scikit-image matplotlib mrcfile opencv-python torch
pip install airlab            # for airlab_all.py, sharpen_exh_airlab.py, run_all.py
pip install aspire             # for aspire_align.py, run_all.py
pip install lightglue          # for lightglue_align.py, run_all.py
pip install tqdm               # for run_logpolar_ncc.py
```
`airlab`, `aspire`, and `lightglue` are only needed if you actually run the
methods that use them (see below — several are commented out by default in
`run_all.py`).

## Common inputs

- `--real` — path to a single `.mrc`/`.mrcs` stack of real class averages.
- `--synth-dir` — directory of resampled synthetic template `.mrc`/`.mrcs`
  files (output of `t0_prepare.py`).
- `--csv` (where used) — a CSV with columns `filename,RMSD_Angstrom`, used to
  bucket template files into low/median/high RMSD bands for example figures.

---

### 1. `t0_prepare.py` — Resampling (run this first)

Checks that polarity agrees between real and synthetic stacks (negates synth if not), resamples synth from its native pixel size to match the real data's pixel size, 
pads/crops to a common box size, and writes a side-by-side QC figure so you can 
visually confirm the two stacks show the same molecule.

```bash
python t0_prepare.py \
  --real       REAL.mrc \
  --synth-dir  /path/to/raw_synth \
  --out        /path/to/synth_resampled \
  --real-px    2.0625 \
  --synth-px   1.0 \
  --box        128
```

Optional: `--limit N` to only process the first N files (useful for a quick
test before committing to the full dataset).

**Outputs:** resampled `.mrc` files in `--out`, plus `T3_qc.png` (one level up
from `--out`) — inspect this image before trusting anything downstream.

---

### 2. `run_all.py` — all methods, on a handful of example files

Picks 5 lowest-, 5 median-, and 5 highest-RMSD template files from the CSV,
finds the best-matching (real, synth) slice pair for each, and runs the
alignment methods on that pair, producing one annotated figure per file.

By default only **log-polar**, **exhaustive**, and **AIRLab-NCC** are active;
AIRLab-NGF, ASPIRE, and LightGlue are present in the code but commented out
in `main()` — uncomment the relevant blocks if you want them included.

```bash
python run_all.py \
  --real       REAL.mrc \
  --synth-dir  /path/to/synth_resampled \
  --csv        ncc_vs_rmsd_results.csv \
  --out        ./out \
  --step       1.0 \
  --n-each     5
```

**Outputs:** one PNG per example file in `--out`, named
`{band}_{rmsd}_{filename}.png`, with one row per method (REAL / SYNTH /
aligned, with angle, mirror, NCC, and scale annotated).

---

### 3. `exhaustive_all.py` — brute-force NCC search over the whole dataset

For every real image, searches **all** synth files/slices/angles/mirrors for
the best masked-NCC match (scale locked at 1.0). Resumable — files with an
existing per-file CSV are skipped and reloaded rather than recomputed.

```bash
python exhaustive_all.py \
  --real       REAL.mrc \
  --synth-dir  /path/to/synth_resampled \
  --out        /path/to/exhaustive_out \
  --step       1.0
```

Optional: `--mask 0.85` (soft circular mask radius fraction), `--limit-files N`
for a quick test run.

**Outputs (in `--out`):**
- `per_file/<file>.mrcs.csv` — best angle/mirror/NCC per real image, for that file
- `global_best.csv` — best match overall for each real image, across all files
- `best_viz/real{NN}_best.png` — visualization of each real image's global best match
- `run.log` — timestamped progress log with ETA

---

### 4. `airlab_all.py` — AIRLab-NCC on the best-matching pair per file

Full brute-force AIRLab over every (real, synth) pair is computationally
infeasible, so this script first does a cheap coarse correlation search per
file to pick one representative (real, synth) pair, then runs AIRLab's
deep-learning registration (NCC loss; NGF is present but disabled) on just
that pair. Resumable and logged.

```bash
python airlab_all.py \
  --real       REAL.mrc \
  --synth-dir  /path/to/synth_resampled \
  --out        /path/to/airlab_out \
  --nstart     8 \
  --iters      120
```

Optional: `--mask 0.85`, `--limit-files N` (testing), `--pngs N` (number of
visualization figures to save, default 10).

**Outputs (in `--out`):**
- `airlab_bestpair.csv` — per-file: picked (ri, si), coarse-search NCC, AIRLab-NCC angle/mirror/final NCC, AIRLab-NGF columns (blank — currently skipped)
- `viz/<file>.png` — figures for the first `--pngs` files
- `run.log`

---

### 5. `sharpen_exh_airlab.py` — does sharpening the REAL image help alignment?

For RMSD-banded example files, tests 5 sharpening filters (unsharp mask,
high-pass, wavelet denoise, Wiener, non-local means) applied to the **real**
image only, then runs **exhaustive** and **AIRLab-NCC** alignment against the
(unmodified) synth for each sharpened version. Produces a grid figure per
file: rows = alignment method, columns = sharpening filter.

```bash
python sharpen_exh_airlab.py \
  --real       REAL.mrc \
  --synth-dir  /path/to/synth_resampled \
  --csv        ncc_vs_rmsd_results.csv \
  --out        ./sharpen_out \
  --n-each     20
```

`--n-each` controls how many files are picked from each of the low/med/high
RMSD bands. Skips (resumable) any file whose output PNG already exists.

**Outputs:** `{band}_{rmsd}_{filename}_sharpen.png` per file in `--out`.

---

### 6. `aspire_align.py` — Fourier-Bessel (ASPIRE) alignment, single pair

Expands both images in a Fourier-Bessel basis, where rotation becomes a phase
multiply on angular-frequency coefficients (no pixel interpolation). Searches
angle x mirror for the best coefficient-space correlation. This script
operates on a **single specified (real, synth) pair**, not the whole dataset.

```bash
python aspire_align.py \
  --real  REAL.mrc \
  --query /path/to/synth_resampled/SOME_FILE.mrcs \
  --ri    0 \
  --si    10 \
  --out   ./aspire_out \
  --step  0.5
```

`--ri`/`--si` select which slice of the real stack and which slice of the
query stack to align. Real and query must already share box size (i.e. query
must come from the resampled synth directory).

**Outputs:** `aspire_r{ri}_s{si}.png` in `--out`, plus console output
including the peak-vs-mean contrast (a flat curve means no recoverable
angle).

---

### 7. `lightglue_align.py` — feature-matching falsification test

Not a production alignment method — this checks whether SuperPoint finds any
usable keypoints on cryo-EM class averages at all, and if so, whether
LightGlue-matched keypoints yield a consistent rigid rotation. Reports
keypoint counts and match/inlier counts as prominently as angles, since a low
keypoint count means the method simply doesn't apply to this data.

```bash
python lightglue_align.py \
  --real      REAL.mrc \
  --synth     /path/to/synth_resampled \
  --out       lg_results \
  --upsample  512 \
  --viz       20
```

`--synth` can be a single `.mrc`/`.mrcs` file or a directory (all files in it
are pooled). Other options: `--max-kpts` (SuperPoint keypoint budget),
`--det-thresh` (lower only if keypoint counts come back near zero — this
manufactures keypoints from noise, so treat results with suspicion),
`--invert-synth` (flip synth contrast polarity), `--viz N` (save N match
visualizations for pairs with >=8 RANSAC inliers).

**Outputs (in `--out`):**
- `lightglue_alignments.csv` — every (real, synth) pair: n_matches, angle, scale diagnostic, rmse, n_inliers
- `match_r{i}_s{j}.png` — up to `--viz` match visualizations
- Console verdict: fraction of pairs with >=8 inliers, and whether the scale diagnostic clusters near 1.0 (it should — if not, the two stacks aren't on a common pixel size)

---

### 8. `compare_distribution_real_zero_ncc_mask.py` — one-vs-one QC figure (pixel distribution + masked NCC)

Standalone, single-pair diagnostic. Given **one** real class average and
**one** synthetic class average, it clips negative real-image pixels to 0,
resamples the synthetic image to match the real image's pixel size and box
size, aligns the synthetic image to the real image with log-polar
registration (rotation only — no scale correction, since box/pixel size are
already matched), thresholds each image independently (Otsu or mean+k·std)
to find its own bright/foreground region, and takes the **union** of the two
masks as the comparison region. It then computes a masked NCC and plots a
6-panel QC figure: real image, real+its own mask outline, aligned synthetic,
synthetic+its own mask outline, a real/synthetic blend with the union mask
outline, and a masked pixel-density histogram comparing the two.

Unlike the pipeline above, this script does **not** require `t0_prepare.py`
to have been run first — it does its own pixel-size/box-size matching
internally for the single pair given.

```bash
python compare_distribution_real_zero_ncc_mask.py \
  --real       /path/to/real/classes.mrcs \
  --real-idx   0 \
  --syn        /path/to/synthetic/ds_042/avg.mrcs \
  --syn-idx    10 \
  --mask-method otsu \
  --out        one_vs_one.png
```

Key options:
- `--real-idx` / `--syn-idx` — which slice to pull from each stack (a bare
  2D `.mrc` is used as-is regardless of index).
- `--mask-method {otsu,mean_std}` — thresholding method for the bright-region
  mask (default `otsu`); `--std-factor` sets `k` when using `mean_std`
  (`thresh = mean + k*std`).
- `--radius` — log-polar radius for the rotation search (default
  `min(box_size)//2`).
- `--bins` — histogram bin count (default 60).

**Outputs:** one PNG (`--out`, default `one_vs_one.png`) with the 6-panel
figure described above. Console output includes the rescale/alignment
parameters used, mask coverage for each image and their union, the masked
NCC, and masked mean/std for both images.

---

### 9. `run_logpolar_ncc.py` — batch log-polar NCC: one reference vs. many query stacks

Parallelized, whole-dataset batch scorer built on the same log-polar
alignment core as the pipeline above (rotation **and** scale correction, via
the log-radius shift). It compares **one** reference `.mrc`/`.mrcs` stack
against **every** query `.mrcs` stack in a directory, one process per query
file, and writes every (ref_image, query_image) NCC score to a per-query
`.txt` file.

Each query file is processed independently in its own worker process (via
`multiprocessing.Pool`), so no locking is needed and the run is resumable —
files with an existing output `.txt` are skipped unless `--overwrite` is
given. Output is written via a temp-file-then-rename pattern so a killed job
never leaves a half-written result file behind. BLAS/OpenMP thread counts are
pinned to 1 per worker (set before numpy is imported) since parallelism here
is at the file level, not inside individual numpy calls.

```bash
python run_logpolar_ncc.py -n 16
python run_logpolar_ncc.py -n 16 --radius 1500 --overwrite
python run_logpolar_ncc.py -n 8 --ref ./myref.mrc --query-dir ./queries --out-dir ./out
```

Key options:
- `-n / --ncpus` — number of worker processes (default: all cores).
- `--ref` — reference `.mrc` stack (all images in it are compared against
  every query image).
- `--query-dir` / `--pattern` — directory of query `.mrcs` stacks and glob
  pattern to select them (default `*.mrcs`).
- `--out-dir` — output directory for per-query `.txt` files.
- `--radius` — log-polar radius (default 1500).
- `--overwrite` — recompute even if a query's output `.txt` already exists.

**Outputs (in `--out-dir`):** one `<query_name>.txt` per query file, with a
row for every `(ref_idx, query_idx, NCC, angle, scale)` combination — e.g.
50 reference images x 50 query images = 2500 rows per file. Console shows a
progress bar (via `tqdm`) and a final summary of how many files succeeded,
were skipped (already done), or errored.

---

### 10. `sharpen_diff_methods.py` — per-image sharpening/denoising comparison figures

Standalone visual comparison tool (not tied to real/synth alignment). Reads
every image in an `.mrcs`/`.mrc` stack and, for each one, applies five
sharpening/denoising methods — Wiener deconvolution (against a stand-in
Gaussian PSF), unsharp masking, FFT Butterworth high-pass filtering, wavelet
denoising (BayesShrink, soft thresholding), and Non-Local Means denoising —
then saves one comparison figure (original + all five results) per image.

```bash
python sharpen_diff_methods.py input.mrcs -o output_dir
python sharpen_diff_methods.py input.mrcs -o output_dir --dpi 150 --cmap gray
```

Key options:
- `input` (positional) — path to the input `.mrcs`/`.mrc` stack.
- `-o / --output` — output directory for the comparison figures (default
  `output`).
- `--dpi` — figure resolution (default 150).
- `--cmap` — matplotlib colormap (default `gray`).

**Outputs (in `-o`):** one `image_{NNN}_comparison.png` per input image,
each a 6-panel (2x3) grid of the original plus all five processed versions.
Console prints progress every 10 images.

---

## Notes

- All scripts use a **soft circular mask** (`soft_mask`) to suppress edge/
  corner background when computing normalized cross-correlation (NCC), and a
  masked z-score `norm()` for comparing images. The exception is
  `compare_distribution_real_zero_ncc_mask.py`, which instead uses a
  data-driven bright-region mask (Otsu or mean+k·std, unioned across both
  images) rather than a fixed circular mask.
- "Mirror" in outputs (`mir`/`mirror`) indicates whether the synth image was
  horizontally flipped before rotation search — cryo-EM projections can be
  chiral-ambiguous, so both handedness are tested everywhere. (Not tested in
  `compare_distribution_real_zero_ncc_mask.py` or `run_logpolar_ncc.py`,
  which only search rotation and, in the latter's case, scale.)
- Scale is intentionally locked to 1.0 in the rigid-registration methods
  (exhaustive, AIRLab, ASPIRE) since real and resampled synth share pixel
  size after `t0_prepare.py`. LightGlue reports a scale *diagnostic* only,
  never applies it — a value far from 1.0 signals a pixel-size/preprocessing
  bug, not something to "fix" by allowing free scale.
  `compare_distribution_real_zero_ncc_mask.py` likewise locks scale (it
  explicitly matches pixel/box size beforehand and only searches rotation),
  while `run_logpolar_ncc.py` is the one script here that *does* recover and
  apply a scale correction from the log-polar shift.
- `run_all.py` contains a verbatim, unmodified copy of an original
  `compare_one.py` log-polar implementation (see the marked block in the
  source) for direct comparison against the newer masked-NCC methods; its
  NCC values use a different (unmasked) convention and aren't directly
  comparable to the other rows in the same figure.
