#!/usr/bin/env python3
"""
aspire_align.py -- Fourier-Bessel rotational alignment (ASPIRE).

Expands each image in a Fourier-Bessel (FFB) basis. In that basis, rotating by
theta is a phase multiply on the angular-frequency coefficients: coefficient of
angular frequency k picks up exp(i*k*theta). No pixel resampling, so no
interpolation blur, and the angle is continuous.

Search: exhaustive over angle (fine grid) x {mirror}, scored by correlation in
coefficient space. Same search shape as the brute-force classical method, but
with an interpolation-free rotation operator.

Usage:
  python aspire_align.py --real R.mrc --query SYNTH_RESAMPLED.mrcs --ri 0 --si 10 \
      --out ./aspire_out
"""
import argparse, os
import numpy as np, mrcfile
from aspire.basis import FFBBasis2D
from aspire.image import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        d = np.array(m.data, dtype=np.float32)
    return d[None] if d.ndim == 2 else d

def soft_mask(L, frac=0.85):
    yy, xx = np.mgrid[-1:1:L*1j, -1:1:L*1j]; r = np.sqrt(xx**2+yy**2)
    t = np.clip((frac-r)/(frac-frac*0.9),0,1)
    return (0.5-0.5*np.cos(np.pi*t)).astype(np.float32)

def prep(im, mask):
    x = im*mask; v = x[mask>1e-3]
    return ((x-v.mean())/(v.std()+1e-8)*mask).astype(np.float32)

def coef_corr(ca, cb):
    """normalized correlation between two coefficient vectors (real inner prod)"""
    a = ca.ravel(); b = cb.ravel()
    a = a - a.mean(); b = b - b.mean()
    return float(np.real(np.vdot(a, b)) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))


def aspire_align(R, S, mask, basis, step=0.5):
    """
    Return best (angle, mirror, score) aligning template S to real R, plus the
    full angle->score curve for each handedness. Rotation is a phase shift on
    the FFB coefficients of S; R's coefficients are computed once.
    """
    Rn = prep(R, mask); Sn = prep(S, mask)
    cR = basis.evaluate_t(Image(Rn[None]))
    cS = basis.evaluate_t(Image(Sn[None]))
    rR = cR.asnumpy()[0]

    angs = np.arange(0, 360, step)
    curves = {}
    best = (-2, None, None)
    for refl in (False, True):
        scores = np.empty(len(angs), np.float32)
        for i, a in enumerate(angs):
            cr = cS.rotate(np.deg2rad(a), refl=refl)     # phase shift, no resample
            scores[i] = coef_corr(rR, cr.asnumpy()[0])
        curves[refl] = scores
        j = int(np.argmax(scores))
        if scores[j] > best[0]:
            best = (float(scores[j]), float(angs[j]), refl)
    return best, angs, curves, cS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--query", required=True, help="RESAMPLED synth .mrcs")
    ap.add_argument("--ri", type=int, default=0)
    ap.add_argument("--si", type=int, default=10)
    ap.add_argument("--out", default="./aspire_out")
    ap.add_argument("--step", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    R = load(args.real)[args.ri]
    S = load(args.query)[args.si]
    if R.shape != S.shape:
        raise SystemExit(f"box mismatch {R.shape} vs {S.shape} -- use RESAMPLED synth")
    L = R.shape[-1]; mask = soft_mask(L)
    basis = FFBBasis2D((L, L), dtype=np.float32)

    (sc, ang, refl), angs, curves, cS = aspire_align(R, S, mask, basis, args.step)
    peak = curves[refl]; contrast = float(peak.max() - peak.mean())
    print(f"[ASPIRE] real[{args.ri}] synth[{args.si}]  angle {ang:.2f}  "
          f"mirror {refl}  score {sc:.4f}")
    print(f"         peak-vs-mean contrast {contrast:.4f} "
          f"(flat => no recoverable angle)")

    # render aligned template from coefficients (interpolation-free)
    aligned = cS.rotate(np.deg2rad(ang), refl=refl).evaluate().asnumpy()[0]

    Rn, Sn = prep(R, mask), prep(S, mask)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for a_,(im,ttl) in zip(ax, [(Rn,f"REAL [{args.ri}]"),(Sn,f"SYNTH [{args.si}]"),
            (prep(aligned,mask), f"aligned (FFB)\n{ang:.1f}deg mir={refl}\nscore={sc:.4f}")]):
        a_.imshow(im, cmap="gray"); a_.set_title(ttl, fontsize=10); a_.axis("off")
    fig.suptitle(f"ASPIRE Fourier-Bessel alignment  (interpolation-free)  "
                 f"contrast={contrast:.3f}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(args.out, f"aspire_r{args.ri}_s{args.si}.png")
    plt.savefig(out, dpi=115); print(f"figure -> {out}")

if __name__=="__main__":
    main()