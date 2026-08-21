"""run_all.py -- run all methods on 15 example template files chosen by RMSD.

Picks 5 lowest-RMSD, 5 median-RMSD, 5 highest-RMSD template files from the CSV.
For each file, finds the best-matching (real, synth) slice pair via the classical
sweep, then runs log-polar / exhaustive / AIRLab-NCC / AIRLab-NGF / LightGlue on that pair and
writes a figure. RMSD is printed on every figure. 

Currently commented out to only display log-polar / exhaustive / AIRLab-NCC

Usage:
  python run_examples.py \
    --real  REAL.mrc \
    --synth-dir ~/Downloads/mrc_test/realdatapractice/synth_resampled \
    --csv   /net/kihara/.../ncc_vs_rmsd_results.csv \
    --out   ../out
"""

import argparse, csv, glob, os
import numpy as np, mrcfile, torch
from scipy.ndimage import rotate as ndrotate
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, warp, AffineTransform
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aspire.basis import FFBBasis2D
from aspire.image import Image

_ASPIRE_BASIS = {}
def _get_basis(L):
    if L not in _ASPIRE_BASIS:
        _ASPIRE_BASIS[L] = FFBBasis2D((L, L), dtype=np.float32)
    return _ASPIRE_BASIS[L]

def aspire_one(R, S, mask, step=1.0):
    basis = _get_basis(R.shape[-1])
    Rn = norm(R, mask); Sn = norm(S, mask)
    rR = basis.evaluate_t(Image(Rn[None])).asnumpy()[0]
    cS = basis.evaluate_t(Image(Sn[None]))
    def cc(a, b):
        a=a.ravel()-a.ravel().mean(); b=b.ravel()-b.ravel().mean()
        return float(np.real(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    best=(-2,None,None)
    for refl in (False, True):
        for a in np.arange(0,360,step):
            v=cc(rR, cS.rotate(np.deg2rad(a), refl=refl).asnumpy()[0])
            if v>best[0]: best=(v,float(a),refl)
    sc,ang,refl=best
    aligned = cS.rotate(np.deg2rad(ang), refl=refl).evaluate().asnumpy()[0]
    return ang, refl, sc, norm(aligned, mask)

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
def norm_pos(im, mask):
    z = norm(im, mask); return (z - z.min() + 1e-3).astype(np.float32)
def ncc(a, b, mask):
    w = mask>1e-3
    x=(a-a[w].mean()).ravel(); y=(b-b[w].mean()).ravel()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
def rot(im, ang, mir):
    base = im[:, ::-1] if mir else im
    return ndrotate(base, ang, reshape=False, order=3, mode="constant")

def exhaustive_pair(R, S, mask, step=1.0):
    Rn = norm(R, mask); angs = np.arange(0,360,step); best=(-2,None,None)
    for mir in (False, True):
        c = [ncc(Rn, norm(rot(S,a,mir),mask), mask) for a in angs]
        j = int(np.argmax(c))
        if c[j] > best[0]: best = (float(c[j]), float(angs[j]), mir)
    return best

def pick_best_slices(R_stack, S_stack, mask, coarse=15.0):
    angs = np.arange(0, 360, coarse)
    Rn = np.stack([norm(R_stack[i], mask) for i in range(R_stack.shape[0])])
    Rv = Rn.reshape(len(Rn), -1); Rv = Rv - Rv.mean(1, keepdims=True)
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
    return (best[0], best[1], best[2], 0.0, False)

def airlab_one(R, S, mask, loss_name, nstart=8, iters=120, lr=0.02, eps=1e-2):
    import airlab as al
    def to_al(a):
        t = torch.from_numpy(np.ascontiguousarray(a))[None,None].float()
        return al.Image(t, a.shape, [1,1],[0,0])
    Rp = norm_pos(R, mask); best=(np.inf,None,None)
    for mir in (False, True):
        Sp = norm_pos(S[:, ::-1].copy() if mir else S, mask)
        for s in np.arange(0,360,360//nstart):
            f, mv = to_al(Rp), to_al(Sp)
            tf = al.transformation.pairwise.RigidTransformation(mv, opt_cm=False)
            tf.init_translation(f)
            with torch.no_grad(): tf._phi_z.fill_(float(np.deg2rad(s)))
            L = (al.loss.pairwise.NGF(f,mv,epsilon=eps) if loss_name=="ngf"
                 else al.loss.pairwise.NCC(f,mv))
            reg = al.PairwiseRegistration(verbose=False)
            reg.set_transformation(tf); reg.set_image_loss([L])
            reg.set_optimizer(torch.optim.Adam(tf.parameters(), lr=lr))
            reg.set_number_of_iterations(iters); reg.start()
            with torch.no_grad():
                lv=float(L(tf()).item()); ang=float(np.rad2deg(tf._phi_z.item()))%360
            if np.isfinite(lv) and lv<best[0]: best=(lv,ang,mir)
    return best

def lightglue_one(R, S, mask, upsample=512):
    import cv2
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    ex = SuperPoint(max_num_keypoints=2048).eval()
    mt = LightGlue(features="superpoint").eval()
    def prep(im):
        x = norm(im, mask); lo,hi=np.percentile(x[mask>1e-3],[1,99])
        x = np.clip((x-lo)/(hi-lo+1e-8),0,1)
        x = cv2.resize(x,(upsample,upsample),interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32))[None,None]
    with torch.no_grad():
        f0, f1 = ex.extract(prep(R)), ex.extract(prep(S))
        out = mt({"image0":f0,"image1":f1})
    a,b,o = rbd(f0),rbd(f1),rbd(out); m=o["matches"]
    k0=a["keypoints"][m[:,0]].cpu().numpy(); k1=b["keypoints"][m[:,1]].cpu().numpy()
    if len(m)<3: return None,len(m),0
    M,inl = cv2.estimateAffinePartial2D(k1.astype(np.float32),k0.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
    if M is None or inl is None: return None,len(m),0
    ang=float(np.degrees(np.arctan2(M[1,0],M[0,0]))%360)
    sc=float(np.sqrt(M[0,0]**2+M[1,0]**2))
    return (ang,sc),len(m),int(inl.sum())

# =====================================================================
# FARHANAZ'S ORIGINAL compare_one.py -- COPIED VERBATIM, UNMODIFIED.
# Run on the RAW (un-resampled) synth, which is what it was written for.
# Its own ncc() is used too (no mask, no variance normalisation), so the
# number reported for this row is exactly what the original pipeline
# reported. It is NOT directly comparable to the other five rows.
# =====================================================================
ORIG_RADIUS = 1500

def orig_correct(img, angle_deg, scale_factor, output_shape):
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

def orig_ncc(a, b):
    min_r = min(a.shape[0], b.shape[0])
    min_c = min(a.shape[1], b.shape[1])
    a = a[:min_r, :min_c]
    b = b[:min_r, :min_c]
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))

def orig_align(image, query):
    image_polar = warp_polar(image, radius=ORIG_RADIUS, scaling='log')
    query_polar = warp_polar(query,  radius=ORIG_RADIUS, scaling='log')
    shifts, _, _ = phase_cross_correlation(
        image_polar, query_polar, upsample_factor=20, normalization=None
    )
    shiftr, shiftc = shifts[:2]
    klog        = ORIG_RADIUS / np.log(ORIG_RADIUS)
    shift_scale = 1 / (np.exp(shiftc / klog))
    corrected   = orig_correct(query, shiftr, shift_scale, image.shape)
    return corrected, shiftr, shift_scale
# ================= END VERBATIM ORIGINAL =============================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--real",required=True)
    ap.add_argument("--synth-dir",required=True,help="RESAMPLED synth (128px)")
    ap.add_argument("--csv",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--step",type=float,default=1.0); ap.add_argument("--n-each",type=int,default=5)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    all_rows=[]
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            try: all_rows.append((r["filename"].strip(), float(r["RMSD_Angstrom"])))
            except Exception: pass
    rows=[]; skipped=[]
    for fn,v in all_rows:
        if glob.glob(os.path.join(args.synth_dir, fn+"*")): rows.append((fn,v))
        else: skipped.append((fn,v))
    rows.sort(key=lambda x:x[1])
    k=args.n_each; mid=len(rows)//2
    picks=( [("low",*x) for x in rows[:k]] + [("med",*x) for x in rows[mid-k//2:mid-k//2+k]]
          + [("high",*x) for x in rows[-k:]] )
    R_stack=load(args.real); L=R_stack.shape[-1]; mask=soft_mask(L)
    summary=[]
    for band, fname, rmsd in picks:
        matches=glob.glob(os.path.join(args.synth_dir, fname+"*"))
        if not matches: continue
        S_stack=load(matches[0])
        if S_stack.shape[-1]!=L: continue
        v,ri,si,_,_ = pick_best_slices(R_stack, S_stack, mask)
        R,S = R_stack[ri], S_stack[si]
        Rn,Sn = norm(R,mask), norm(S,mask)
        print(f"\n[{band}] {fname} rmsd={rmsd:.2f} pair r{ri}/s{si}")
        # ---------------- run all 6 methods; EVERY method gets a row ----------
        res = {}
        custom_img = {}          # per-file; holds images that are NOT plain rot()
        custom_synth = {}        # per-file; rows that were fed a DIFFERENT synth
        fails = []

        # 1. ORIGINAL log-polar -- Farhanaz's verbatim code, on RESAMPLED synth
        try:
            corrected, ang_o, scale_o = orig_align(R, S)   # HER function, HER convention
            ncc_o = float(orig_ncc(R, corrected))          # HER ncc: no mask
            res["log-polar"] = (float(ang_o % 360), False, ncc_o, float(scale_o))
            custom_img["log-polar"] = corrected            # show what SHE produced
        except Exception as e:
            res["log-polar"] = (0.0, False, float("nan"), 1.0)
            fails.append(f"log-polar: {e}")

        # 2. exhaustive
        b, a, m = exhaustive_pair(R, S, mask, args.step)
        res["exhaustive"] = (a, m, b, 1.0)

        # 3-4. airlab
        # for ln in ("ncc", "ngf"):
        for ln in ("ncc",):
            try:
                lv, a, m = airlab_one(R, S, mask, ln)
                if a is None:
                    raise RuntimeError("all optimizations non-finite (NaN loss)")
                res[f"airlab-{ln}"] = (a, m, ncc(Rn, norm(rot(S, a, m), mask), mask), 1.0)
            except Exception as e:
                res[f"airlab-{ln}"] = (0.0, False, float("nan"), 1.0)
                fails.append(f"airlab-{ln}: {e}")

        # # 5. aspire
        # try:
        #     aa, am, asc, aimg = aspire_one(R, S, mask, args.step)
        #     res["aspire"] = (aa, am, ncc(Rn, norm(rot(S, aa, am), mask), mask), 1.0)
        # except Exception as e:
        #     res["aspire"] = (0.0, False, float("nan"), 1.0)
        #     fails.append(f"aspire: {e}")

        # # 6. lightglue
        # try:
        #     lg, nm, ninl = lightglue_one(R, S, mask)
        #     if lg:
        #         a, sc = lg
        #         res["lightglue"] = (a, False, ncc(Rn, norm(rot(S, a, False), mask), mask), sc)
        #     else:
        #         res["lightglue"] = (0.0, False, ncc(Rn, norm(S, mask), mask), 1.0)
        #         fails.append(f"lightglue: only {nm} matches / {ninl} inliers (shown un-rotated)")
        # except Exception as e:
        #     res["lightglue"] = (0.0, False, float("nan"), 1.0)
        #     fails.append(f"lightglue: {e}")

        for msg in fails:
            print(f"   !! {msg}")
        print("   " + "  ".join(
            f"{kk}:{val[0]:.0f}deg/mir{int(val[1])}/ncc{val[2]:.3f}/s{val[3]:.2f}"
            for kk, val in res.items()))
        print(f"   ({len(res)} methods -> {len(res)} rows)")

        # ---------------- figure: one row per method --------------------------
        nmeth = len(res)
        fig, ax = plt.subplots(nmeth, 3, figsize=(11, 3.4 * nmeth))
        if nmeth == 1: ax = ax[None, :]
        for r_, (name, (a, m, sc, scl)) in enumerate(res.items()):
            al_im = custom_img.get(name)              # BEFORE the list is built
            if al_im is None:
                al_im = norm(rot(S, a, m), mask)
            ttl3 = (f"aligned {a:.0f}deg mir={m}\nNCC={sc:.3f} scale={scl:.2f}"
                    if np.isfinite(sc) else "FAILED")
            syn_im = custom_synth.get(name)
            syn_ttl = f"SYNTH[{si}]"
            if syn_im is None:
                syn_im = Sn
            else:
                syn_ttl = f"SYNTH[{si}] RAW {syn_im.shape[-1]}px"
            for c_, (im, ttl) in enumerate([(Rn, f"REAL[{ri}]"),
                                            (syn_im, syn_ttl),
                                            (al_im, ttl3)]):
                ax[r_, c_].imshow(im, cmap="gray")
                ax[r_, c_].set_title(ttl, fontsize=9)
                ax[r_, c_].axis("off")
                if c_ == 0:
                    ax[r_, c_].text(-0.14, 0.5, name.upper(),
                                    transform=ax[r_, c_].transAxes, rotation=90,
                                    va="center", ha="center",
                                    fontsize=11, fontweight="bold")

        fig.suptitle(f"[{band.upper()} RMSD]  {fname}   RMSD = {rmsd:.2f} A   "
                     f"(best pair r{ri}/s{si})", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0,0,1,0.97])
        out=os.path.join(args.out,f"{band}_{rmsd:06.2f}_{fname}.png")
        plt.savefig(out,dpi=110); plt.close(fig)
        summary.append((band,fname,rmsd,ri,si,res))
        print(f"   -> {out}")
if __name__=="__main__": main()