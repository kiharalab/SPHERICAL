#!/usr/bin/env python3
"""
Batch NCC comparison: ONE reference MRC stack vs EVERY query MRC stack in a
directory, parallelized across N CPUs.

For each query file it writes one .txt (named after the query file) into the
output directory, containing the NCC for every (ref_image, query_image) pair
-> 50 x 50 = 2500 rows per file. No PNGs.

Each query file is an independent task, so workers never write to the same
file and no locking is needed.

Usage:
    python run_logpolar_ncc.py -n 16
    python run_logpolar_ncc.py -n 16 --radius 1500 --overwrite
    python run_logpolar_ncc.py -n 8 --ref ./myref.mrc --query-dir ./queries --out-dir ./out
"""

# ── Limit BLAS/OpenMP threads BEFORE importing numpy ──────────────────
# Each worker process is CPU-bound numpy/scipy work. Without this, every
# process spawns its own BLAS threads -> n_procs * n_threads oversubscription,
# which makes parallel SLOWER than serial. One thread per process is correct
# here because we parallelize at the file level instead.
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import glob
from multiprocessing import Pool

import numpy as np
import mrcfile
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, warp, AffineTransform

# ── CORE FUNCTIONS (unchanged from your validated single-pair script) ──
def correct(img, angle_deg, scale_factor, output_shape):
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

def ncc(a, b):
    min_r = min(a.shape[0], b.shape[0])
    min_c = min(a.shape[1], b.shape[1])
    a = a[:min_r, :min_c]
    b = b[:min_r, :min_c]
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))

def align(image, query, radius):
    image_polar = warp_polar(image, radius=radius, scaling='log')
    query_polar = warp_polar(query,  radius=radius, scaling='log')
    shifts, _, _ = phase_cross_correlation(
        image_polar, query_polar, upsample_factor=20, normalization=None
    )
    shiftr, shiftc = shifts[:2]
    klog        = radius / np.log(radius)
    shift_scale = 1 / (np.exp(shiftc / klog))
    corrected   = correct(query, shiftr, shift_scale, image.shape)
    return corrected, shiftr, shift_scale

def load_stack(path):
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)   # writable copy
    if data.ndim == 2:
        data = data[None, ...]
    return data

# ── WORKER SETUP ───────────────────────────────────────────────────────
# The reference stack and config are the same for every task, so we load the
# reference ONCE per worker process (via the pool initializer) and stash it +
# config in module globals instead of re-reading/re-pickling it 390 times.
_REF = None
_REF_NAME = None
_RADIUS = None
_OUT_DIR = None
_OVERWRITE = None

def init_worker(ref_path, radius, out_dir, overwrite):
    global _REF, _REF_NAME, _RADIUS, _OUT_DIR, _OVERWRITE
    _REF = load_stack(ref_path)
    _REF_NAME = os.path.basename(ref_path)
    _RADIUS = radius
    _OUT_DIR = out_dir
    _OVERWRITE = overwrite

def process_query(query_path):
    """Compare the reference against one query stack; write one .txt."""
    query_name = os.path.splitext(os.path.basename(query_path))[0]
    out_file = os.path.join(_OUT_DIR, f'{query_name}.txt')

    # resumable: skip files already done unless --overwrite
    if (not _OVERWRITE) and os.path.exists(out_file):
        return (query_name, 'skipped', 0)

    tmp_file = out_file + '.tmp'           # write-then-rename = atomic
    try:
        query_stack = load_stack(query_path)
        n_ref   = _REF.shape[0]
        n_query = query_stack.shape[0]

        with open(tmp_file, 'w') as f:
            f.write(f"# reference: {_REF_NAME}\n")
            f.write(f"# query:     {os.path.basename(query_path)}\n")
            f.write(f"{'ref_idx':>7} {'query_idx':>9} {'NCC':>8} {'angle':>8} {'scale':>8}\n")
            f.write('-' * 45 + '\n')
            for i in range(n_ref):
                image = _REF[i]
                for j in range(n_query):
                    corrected, angle, scale = align(image, query_stack[j], _RADIUS)
                    ncc_score = ncc(image, corrected)
                    f.write(f"{i:>7d} {j:>9d} {ncc_score:>8.4f} {angle:>8.2f} {scale:>8.4f}\n")

        os.replace(tmp_file, out_file)     # only appears once fully written
        return (query_name, 'ok', n_ref * n_query)

    except Exception as e:
        if os.path.exists(tmp_file):
            try: os.remove(tmp_file)
            except OSError: pass
        return (query_name, f'ERROR: {type(e).__name__}: {e}', 0)

# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Parallel NCC: one reference vs all query mrcs.')
    ap.add_argument('-n', '--ncpus', type=int, default=os.cpu_count(),
                    help='number of CPU processes (default: all cores)')
    ap.add_argument('--ref', default='./cryosparc_P6_J7_020_class_averages.mrc',
                    help='reference .mrc stack')
    ap.add_argument('--query-dir', default='./all_390_Cryosparced/template_mrcs_links',
                    help='directory of query .mrcs stacks')
    ap.add_argument('--out-dir', default='./ncc_results_new',
                    help='output directory for the per-query .txt files')
    ap.add_argument('--pattern', default='*.mrcs',
                    help="glob for query files (default: '*.mrcs')")
    ap.add_argument('--radius', type=int, default=1500,
                    help='log-polar radius (default: 1500, matches your validated run)')
    ap.add_argument('--overwrite', action='store_true',
                    help='recompute even if an output .txt already exists')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    query_paths = sorted(glob.glob(os.path.join(args.query_dir, args.pattern)))
    if not query_paths:
        raise SystemExit(f'No files matching {args.pattern!r} in {args.query_dir}')

    print(f'Reference : {args.ref}')
    print(f'Query dir : {args.query_dir}  ({len(query_paths)} files matching {args.pattern!r})')
    print(f'Output    : {args.out_dir}')
    print(f'CPUs      : {args.ncpus}   |   radius: {args.radius}   |   overwrite: {args.overwrite}')

    with Pool(processes=args.ncpus,
              initializer=init_worker,
              initargs=(args.ref, args.radius, args.out_dir, args.overwrite)) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_query, query_paths),
            total=len(query_paths), desc='Query files'))

    ok      = sum(1 for _, s, _ in results if s == 'ok')
    skipped = sum(1 for _, s, _ in results if s == 'skipped')
    errors  = [(q, s) for q, s, _ in results if s.startswith('ERROR')]

    print(f'\nDone. ok={ok}  skipped={skipped}  errors={len(errors)}')
    if errors:
        print('Failures:')
        for q, s in errors:
            print(f'  {q}: {s}')

if __name__ == '__main__':
    main()