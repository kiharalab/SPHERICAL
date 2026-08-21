#!/usr/bin/env python3
"""
airlab_all_bestpair.py -- AIRLab-NCC and AIRLab-NGF on the BEST-MATCHING pair
per template file, across the whole dataset.

Full brute-force AIRLab across all 50x50xN pairs is not runnable (897,500
pairs x ~37s = years of CPU time). Instead: for each of the ~359 resampled
template files, find its best (real, synth) pair CHEAPLY via a fast
exhaustive/coarse correlation search, then run AIRLab (both NCC and NGF) on
just that one pair. This is O(N files), not O(N files x 2500).

Resumable, logged, timed. ~359 files x (~2 AIRLab runs x ~15-40s) is on the
order of a few hours, not weeks.

Usage:
  python airlab_all_bestpair.py --real R.mrc --synth-dir RESAMPLED \
      --out ~/.../airlab_all --nstart 8 --iters 120
"""
import argparse, glob, os, time, csv
import numpy as np, mrcfile, torch
from scipy.ndimage import rotate as ndrotate
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log(msg, logfile):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(logfile, "a") as f: f.write(line + "\n")

def load(p):
    with mrcfile.open(p, permissive=True) as m:
        d = np.array(m.data, dtype=np.float32)
    return d[None] if d.ndim == 2 else d

def soft_mask(L, frac=0.85):
    yy, xx = np.mgrid[-1:1:L*1j, -1:1:L*1j]; r = np.sqrt(xx**2+yy**2)
    t = np.clip((frac-r)/(frac-frac*0.9), 0, 1)
    return (0.5-0.5*np.cos(np.pi*t)).astype(np.float32)

def norm(im, mask):
    x = im*mask; v = x[mask>1e-3]
    return ((x-v.mean())/(v.std()+1e-8)*mask).astype(np.float32)

def norm_pos(im, mask):
    z = norm(im, mask); return (z - z.min() + 1e-3).astype(np.float32)

def ncc(a, b, mask):
    w = mask>1e-3; x=(a-a[w].mean()).ravel(); y=(b-b[w].mean()).ravel()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))

def rot(im, ang, mir):
    base = im[:, ::-1] if mir else im
    return ndrotate(base, ang, reshape=False, order=3, mode="constant")


def pick_best_pair(R_stack, S_stack, mask, coarse=15.0):
    """cheap coarse search -- ~1s per file, picks (ri, si) to hand to AIRLab"""
    angs = np.arange(0, 360, coarse)
    Rn = np.stack([norm(R_stack[i], mask) for i in range(R_stack.shape[0])])
    Rv = Rn.reshape(len(Rn), -1); Rv -= Rv.mean(1, keepdims=True)
    Rv /= (np.linalg.norm(Rv, axis=1, keepdims=True) + 1e-12)
    best = (-2, 0, 0)
    for si in range(S_stack.shape[0]):
        bank = []
        for mir in (False, True):
            for a in angs:
                bank.append(norm(rot(S_stack[si], a, mir), mask).ravel())
        B = np.asarray(bank, np.float32); B -= B.mean(1, keepdims=True)
        B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        sims = Rv @ B.T
        ri = int(sims.max(1).argmax()); v = float(sims[ri].max())
        if v > best[0]: best = (v, ri, si)
    return best[0], best[1], best[2]


def airlab_run(R, S, mask, loss_name, nstart, iters, lr=0.02, eps=1e-2):
    import airlab as al
    def to_al(a):
        return al.Image(torch.from_numpy(np.ascontiguousarray(a))[None,None].float(),
                        a.shape, [1,1], [0,0])
    Rp = norm_pos(R, mask); best = (np.inf, 0.0, False)
    for mir in (False, True):
        Sp = norm_pos(S[:, ::-1].copy() if mir else S, mask)
        for s in np.arange(0, 360, 360//nstart):
            f, mv = to_al(Rp), to_al(Sp)
            tf = al.transformation.pairwise.RigidTransformation(mv, opt_cm=False)
            tf.init_translation(f)
            with torch.no_grad(): tf._phi_z.fill_(float(np.deg2rad(s)))
            L = (al.loss.pairwise.NGF(f, mv, epsilon=eps) if loss_name == "ngf"
                 else al.loss.pairwise.NCC(f, mv))
            reg = al.PairwiseRegistration(verbose=False)
            reg.set_transformation(tf); reg.set_image_loss([L])
            reg.set_optimizer(torch.optim.Adam(tf.parameters(), lr=lr))
            reg.set_number_of_iterations(iters); reg.start()
            with torch.no_grad():
                lv = float(L(tf()).item())
                ang = float(np.rad2deg(tf._phi_z.item())) % 360
            if np.isfinite(lv) and lv < best[0]: best = (lv, ang, mir)
    return best[1], best[2]  # angle, mirror


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nstart", type=int, default=8)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--mask", type=float, default=0.85)
    ap.add_argument("--limit-files", type=int, default=0, help="testing: only first N files")
    ap.add_argument("--pngs", type=int, default=10, help="save first N figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    viz_dir = os.path.join(args.out, "viz"); os.makedirs(viz_dir, exist_ok=True)
    logfile = os.path.join(args.out, "run.log")
    csv_path = os.path.join(args.out, "airlab_bestpair.csv")

    R_stack = load(args.real); L = R_stack.shape[-1]; mask = soft_mask(L, args.mask)
    Rn_all = np.stack([norm(R_stack[i], mask) for i in range(R_stack.shape[0])])

    files = sorted(glob.glob(os.path.join(args.synth_dir, "*.mrcs")))
    if args.limit_files: files = files[:args.limit_files]

    # RESUME: read what's already done
    done_files = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_files.add(row["filename"])
        log(f"resuming: {len(done_files)} files already done", logfile)

    write_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(["filename","ri","si","pick_ncc",
                         "ncc_angle","ncc_mirror","ncc_final",
                         "ngf_angle","ngf_mirror","ngf_final"])

    log(f"START  {len(files)} files  (nstart={args.nstart} iters={args.iters})", logfile)
    t_start = time.time(); n_done = 0; n_viz = 0

    for fi, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        if fname in done_files:
            continue

        t0 = time.time()
        S_stack = load(fpath)
        if S_stack.shape[-1] != L:
            log(f"[{fi+1}/{len(files)}] {fname}  BOX MISMATCH, skip", logfile)
            writer.writerow([fname, -1, -1, "", "", "", "", "", "", ""]); csv_f.flush()
            continue

        pick_ncc, ri, si = pick_best_pair(R_stack, S_stack, mask)
        R, S = R_stack[ri], S_stack[si]

        ncc_ang, ncc_mir = airlab_run(R, S, mask, "ncc", args.nstart, args.iters)
        ncc_final = ncc(Rn_all[ri], norm(rot(S, ncc_ang, ncc_mir), mask), mask)

        # ngf_ang, ngf_mir = airlab_run(R, S, mask, "ngf", args.nstart, args.iters)
        # ngf_final = ncc(Rn_all[ri], norm(rot(S, ngf_ang, ngf_mir), mask), mask)
        ngf_ang, ngf_mir, ngf_final = 0.0, 0, float("nan")   # NGF skipped

        writer.writerow([fname, ri, si, f"{pick_ncc:.4f}",
                         f"{ncc_ang:.2f}", int(ncc_mir), f"{ncc_final:.4f}",
                         f"{ngf_ang:.2f}", int(ngf_mir), f"{ngf_final:.4f}"])
        csv_f.flush()

        dt = time.time() - t0
        n_done += 1
        eta = (time.time()-t_start)/n_done * (len(files)-fi-1)
        log(f"[{fi+1}/{len(files)}] {fname}  {dt:.1f}s  "
            f"ncc {ncc_final:.3f}  ngf {ngf_final:.3f}  ETA {eta/3600:.1f}h", logfile)

        if n_viz < args.pngs:
            fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
            ax[0].imshow(Rn_all[ri], cmap="gray"); ax[0].set_title(f"real[{ri}]")
            ax[1].imshow(norm(rot(S, ncc_ang, ncc_mir), mask), cmap="gray")
            ax[1].set_title(f"airlab-ncc {ncc_ang:.0f}deg  NCC {ncc_final:.3f}")
            ax[2].imshow(norm(rot(S, ngf_ang, ngf_mir), mask), cmap="gray")
            ax[2].set_title(f"airlab-ngf {ngf_ang:.0f}deg  NCC {ngf_final:.3f}")
            for a_ in ax: a_.axis("off")
            fig.suptitle(fname, fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{fname}.png"), dpi=90); plt.close(fig)
            n_viz += 1

    csv_f.close()
    log(f"DONE  {n_done} files processed this run, total elapsed "
        f"{(time.time()-t_start)/3600:.1f}h", logfile)
    log(f"csv -> {csv_path}", logfile)


if __name__ == "__main__":
    main()