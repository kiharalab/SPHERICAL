#!/usr/bin/env python3
"""
exhaustive_all.py -- run the corrected exhaustive rotation search over ALL
(real, synth) pairs across every resampled template file, with timing, a live
log, resumability, and best-match visualizations.

For each real image, finds its best-matching synth template across all files
(exhaustive over angle x mirror, scale locked at 1.0, masked NCC). Writes a
per-file CSV, an overall best-match summary, timing, and a visual for each of
the 50 real images' best match.

RESUMABLE: files already done (a completed .csv exists) are skipped.

Usage:
  python exhaustive_all.py \
    --real  REAL.mrc \
    --synth-dir  ~/.../synth_resampled \
    --out   ~/.../exhaustive \
    --step 1.0
"""
import argparse, glob, os, time, csv, sys
import numpy as np, mrcfile
from scipy.ndimage import rotate as ndrotate
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log(msg, logfile):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(logfile, "a") as f:
        f.write(line + "\n")

def load(p):
    with mrcfile.open(p, permissive=True) as m:
        d = np.array(m.data, dtype=np.float32)
    return d[None] if d.ndim == 2 else d

def soft_mask(L, frac=0.85):
    yy, xx = np.mgrid[-1:1:L*1j, -1:1:L*1j]; r = np.sqrt(xx**2+yy**2)
    t = np.clip((frac-r)/(frac-frac*0.9),0,1)
    return (0.5-0.5*np.cos(np.pi*t)).astype(np.float32)

def norm(im, mask):
    x = im*mask; v = x[mask>1e-3]
    return ((x-v.mean())/(v.std()+1e-8)*mask).astype(np.float32)

def rot(im, ang, mir):
    base = im[:, ::-1] if mir else im
    return ndrotate(base, ang, reshape=False, order=3, mode="constant")


def build_bank(S_stack, angles, mask):
    """pre-rotate every synth slice x angle x mirror, L2-normalized rows."""
    bank, meta = [], []
    for t in range(S_stack.shape[0]):
        for mir in (False, True):
            base = S_stack[t][:, ::-1] if mir else S_stack[t]
            for a in angles:
                bank.append(norm(ndrotate(base, a, reshape=False, order=3,
                                          mode="constant"), mask).ravel())
                meta.append((t, a, int(mir)))
    B = np.asarray(bank, np.float32); B -= B.mean(1, keepdims=True)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return B, np.asarray(meta, np.float32)


def process_file(R_norm_flat, S_stack, mask, angles):
    """returns list of (ri, best_si, angle, mirror, ncc, runner) for this file."""
    bank, meta = build_bank(S_stack, angles, mask)
    out = []
    for i, v in enumerate(R_norm_flat):
        sc = bank @ v
        j = int(np.argmax(sc))
        t, a, mir = int(meta[j,0]), float(meta[j,1]), int(meta[j,2])
        other = meta[:,0] != t
        run = float(sc[other].max()) if other.any() else float("nan")
        out.append((i, t, a, mir, float(sc[j]), run))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--mask", type=float, default=0.85)
    ap.add_argument("--limit-files", type=int, default=0, help="only first N files (testing)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    per_dir = os.path.join(args.out, "per_file"); os.makedirs(per_dir, exist_ok=True)
    viz_dir = os.path.join(args.out, "best_viz"); os.makedirs(viz_dir, exist_ok=True)
    logfile = os.path.join(args.out, "run.log")

    R = load(args.real); L = R.shape[-1]; mask = soft_mask(L, args.mask)
    Rn = np.stack([norm(R[i], mask) for i in range(R.shape[0])])
    Rflat = Rn.reshape(len(Rn), -1); Rflat = Rflat - Rflat.mean(1, keepdims=True)
    Rflat /= (np.linalg.norm(Rflat, axis=1, keepdims=True) + 1e-12)

    files = sorted(glob.glob(os.path.join(args.synth_dir, "*.mrcs")))
    if args.limit_files: files = files[:args.limit_files]
    angles = np.arange(0, 360, args.step)

    log(f"START  {len(R)} real x {len(files)} files x {len(angles)} angles x2 mirror", logfile)
    log(f"  bank per file = {50*len(angles)*2} candidates; box {L}px", logfile)

    # global best per real image: (ncc, file, si, angle, mirror)
    global_best = [(-2, None, -1, 0.0, 0)] * len(R)
    t_start = time.time()
    done = 0

    for fi, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        csv_path = os.path.join(per_dir, fname + ".csv")
        if os.path.exists(csv_path):                       # RESUME
            log(f"[{fi+1}/{len(files)}] {fname}  already done, loading", logfile)
            rows = list(csv.reader(open(csv_path)))[1:]
            results = [(int(r[0]),int(r[1]),float(r[2]),int(r[3]),float(r[4]),float(r[5])) for r in rows]
        else:
            t0 = time.time()
            S = load(fpath)
            if S.shape[-1] != L:
                log(f"[{fi+1}/{len(files)}] {fname}  BOX {S.shape[-1]}!={L}, skip (not resampled)", logfile)
                continue
            results = process_file(Rflat, S, mask, angles)
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["ref","synth_si","angle","mirror","ncc","runner"])
                w.writerows(results)
            dt = time.time() - t0
            done += 1
            eta = (time.time()-t_start)/done * (len(files)-fi-1)
            log(f"[{fi+1}/{len(files)}] {fname}  {dt:.1f}s  "
                f"maxNCC {max(r[4] for r in results):.3f}  ETA {eta/60:.1f}min", logfile)

        for (i, si, a, mir, v, run) in results:
            if v > global_best[i][0]:
                global_best[i] = (v, fname, si, a, mir)

    # overall summary
    summ = os.path.join(args.out, "global_best.csv")
    with open(summ, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["real_idx","best_file","synth_si","angle","mirror","ncc"])
        for i,(v,fn,si,a,mir) in enumerate(global_best):
            w.writerow([i, fn, si, f"{a:.2f}", mir, f"{v:.4f}"])
    log(f"wrote {summ}", logfile)

    # visualize each real image's global best match
    log("rendering best-match visuals...", logfile)
    cache = {}
    for i,(v,fn,si,a,mir) in enumerate(global_best):
        if fn is None: continue
        if fn not in cache: cache[fn] = load(os.path.join(args.synth_dir, fn))
        S = cache[fn][si]
        al = norm(rot(S, a, bool(mir)), mask)
        fig, ax = plt.subplots(1,3, figsize=(11,4))
        ax[0].imshow(Rn[i], cmap="gray"); ax[0].set_title(f"REAL [{i}]")
        ax[1].imshow(norm(S,mask), cmap="gray"); ax[1].set_title(f"{fn}\nsynth[{si}]", fontsize=7)
        ax[2].imshow(al, cmap="gray")
        ax[2].set_title(f"aligned {a:.1f}deg mir={mir}\nNCC={v:.3f}")
        for a_ in ax: a_.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"real{i:02d}_best.png"), dpi=95); plt.close(fig)

    total = time.time() - t_start
    nccs = np.array([b[0] for b in global_best])
    mirs = np.array([b[4] for b in global_best])
    log(f"DONE  total {total/60:.1f} min", logfile)
    log(f"  global-best NCC: median {np.median(nccs):.3f}  range [{nccs.min():.3f},{nccs.max():.3f}]", logfile)
    log(f"  mirror among best: {int(mirs.sum())}/{len(mirs)}", logfile)
    log(f"  visuals -> {viz_dir}", logfile)

if __name__ == "__main__":
    main()