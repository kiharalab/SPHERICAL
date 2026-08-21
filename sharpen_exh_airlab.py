#!/usr/bin/env python3
"""
sharpen_exh_airlab.py -- for RMSD-banded example files, one figure per file:
a 2-method x 5-sharpening grid, testing whether SHARPENING THE REAL class
average helps EXHAUSTIVE and AIRLAB-NCC alignment.

Rows (alignment): exhaustive, airlab-ncc
Cols (sharpening of REAL): unsharp, high-pass, wavelet, wiener, nl-means
Header row: the 5 sharpened reals + shows the synth being matched.

Pair selection: best-matching (real, synth) pair per file (same coarse search
used by the other RMSD figures). Real is sharpened; synth used as-is.

-n-each: number of files to pick from each RMSD band (low, med, high). 

Usage:
  python sharpen_exh_airlab.py --real R.mrc --synth-dir RESAMPLED \
      --csv ncc_vs_rmsd_results.csv --out ./out --n-each 20
"""
import argparse, csv, glob, os
import numpy as np, mrcfile, torch
from scipy.ndimage import rotate as ndrotate
from skimage.filters import unsharp_mask
from skimage.restoration import denoise_wavelet, denoise_nl_means, estimate_sigma
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- sharpening (from Farhanaz's sharpen_diff_new.py) ----------
def _norm01(img):
    img=img.astype(np.float32); lo,hi=img.min(),img.max()
    return np.zeros_like(img) if hi-lo<1e-12 else (img-lo)/(hi-lo)
def _gpsf(shape,sigma):
    yy,xx=np.indices(shape); cy,cx=(shape[0]-1)/2,(shape[1]-1)/2
    p=np.exp(-(((yy-cy)**2+(xx-cx)**2)/(2*sigma**2))); return p/p.sum()
def sh_wiener(img,sigma=2.0,reg=0.01):
    psf=_gpsf(img.shape,sigma); pf=np.fft.fft2(np.fft.ifftshift(psf))
    w=np.conj(pf)/(np.abs(pf)**2+reg)
    return _norm01(np.real(np.fft.ifft2(np.fft.fft2(img)*w)))
def sh_unsharp(img,radius=2.0,amount=1.5):
    return unsharp_mask(img,radius=radius,amount=amount,preserve_range=True)
def sh_highpass(img,cutoff=30.0,blend=1.0):
    f=img.astype(np.float32); F=np.fft.fftshift(np.fft.fft2(f))
    r,c=f.shape; Y,X=np.ogrid[:r,:c]
    d=np.sqrt((X-c//2)**2+(Y-r//2)**2); m=(d>cutoff).astype(np.float32)
    hp=np.abs(np.fft.ifft2(np.fft.ifftshift(F*m)))
    return np.clip(f+blend*hp,0,1)
def sh_wavelet(img):
    return denoise_wavelet(img,method="BayesShrink",mode="soft",
                           rescale_sigma=True,channel_axis=None)
def sh_nlmeans(img):
    s=float(np.mean(estimate_sigma(img,channel_axis=None)))
    return denoise_nl_means(img,h=1.15*s,sigma=s,fast_mode=True,
                            patch_size=5,patch_distance=6,channel_axis=None)
SHARPEN = [("unsharp",sh_unsharp),("high-pass",sh_highpass),
           ("wavelet",sh_wavelet),("wiener",sh_wiener),("nl-means",sh_nlmeans)]

# ---------------- image utils -----------------------------------------------
def load(p):
    with mrcfile.open(p,permissive=True) as m: d=np.array(m.data,dtype=np.float32)
    return d[None] if d.ndim==2 else d
def soft_mask(L,frac=0.85):
    yy,xx=np.mgrid[-1:1:L*1j,-1:1:L*1j]; r=np.sqrt(xx**2+yy**2)
    t=np.clip((frac-r)/(frac-frac*0.9),0,1); return (0.5-0.5*np.cos(np.pi*t)).astype(np.float32)
def norm(im,mask):
    x=im*mask; v=x[mask>1e-3]; return ((x-v.mean())/(v.std()+1e-8)*mask).astype(np.float32)
def norm_pos(im,mask):
    z=norm(im,mask); return (z-z.min()+1e-3).astype(np.float32)
def ncc(a,b,mask):
    w=mask>1e-3; x=(a-a[w].mean()).ravel(); y=(b-b[w].mean()).ravel()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
def rot(im,ang,mir):
    base=im[:,::-1] if mir else im
    return ndrotate(base,ang,reshape=False,order=3,mode="constant")

# ---------------- the two methods -------------------------------------------
def m_exhaustive(R,S,mask,step=2.0):
    Rn=norm(R,mask); best=(-2,0,False)
    for mir in (False,True):
        for a in np.arange(0,360,step):
            v=ncc(Rn,norm(rot(S,a,mir),mask),mask)
            if v>best[0]: best=(v,float(a),mir)
    return best[1],best[2],best[0]

def m_airlab(R,S,mask,nstart=6,iters=100):
    import airlab as al
    def to_al(a):
        return al.Image(torch.from_numpy(np.ascontiguousarray(a))[None,None].float(),a.shape,[1,1],[0,0])
    Rp=norm_pos(R,mask); best=(np.inf,0,False)
    for mir in (False,True):
        Sp=norm_pos(S[:,::-1].copy() if mir else S,mask)
        for s in np.arange(0,360,360//nstart):
            f,mv=to_al(Rp),to_al(Sp)
            tf=al.transformation.pairwise.RigidTransformation(mv,opt_cm=False); tf.init_translation(f)
            with torch.no_grad(): tf._phi_z.fill_(float(np.deg2rad(s)))
            L=al.loss.pairwise.NCC(f,mv)
            reg=al.PairwiseRegistration(verbose=False); reg.set_transformation(tf)
            reg.set_image_loss([L]); reg.set_optimizer(torch.optim.Adam(tf.parameters(),lr=0.02))
            reg.set_number_of_iterations(iters); reg.start()
            with torch.no_grad():
                lv=float(L(tf()).item()); ang=float(np.rad2deg(tf._phi_z.item()))%360
            if np.isfinite(lv) and lv<best[0]: best=(lv,ang,mir)
    Rn=norm(R,mask)
    return best[1],best[2],ncc(Rn,norm(rot(S,best[1],best[2]),mask),mask)

METHODS=[("exhaustive",m_exhaustive),("airlab-ncc",m_airlab)]

# ---------------- pair picking (same as other RMSD figures) -----------------
def pick_pair(R_stack,S_stack,mask,coarse=15.0):
    angs=np.arange(0,360,coarse)
    Rn=np.stack([norm(R_stack[i],mask) for i in range(R_stack.shape[0])])
    Rv=Rn.reshape(len(Rn),-1); Rv-=Rv.mean(1,keepdims=True); Rv/=(np.linalg.norm(Rv,axis=1,keepdims=True)+1e-12)
    best=(-2,0,0)
    for si in range(S_stack.shape[0]):
        bank=[]
        for mir in (False,True):
            for a in angs: bank.append(norm(rot(S_stack[si],a,mir),mask).ravel())
        B=np.asarray(bank,np.float32); B-=B.mean(1,keepdims=True); B/=(np.linalg.norm(B,axis=1,keepdims=True)+1e-12)
        sims=Rv@B.T; ri=int(sims.max(1).argmax()); v=float(sims[ri].max())
        if v>best[0]: best=(v,ri,si)
    return best[1],best[2]

# ---------------- main ------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--real",required=True); ap.add_argument("--synth-dir",required=True)
    ap.add_argument("--csv",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--n-each",type=int,default=20)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)

    all_rows=[]
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            try: all_rows.append((r["filename"].strip(),float(r["RMSD_Angstrom"])))
            except: pass
    rows=[(fn,v) for fn,v in all_rows if glob.glob(os.path.join(args.synth_dir,fn+"*"))]
    rows.sort(key=lambda x:x[1])
    k=args.n_each; mid=len(rows)//2
    picks=([("low",*x) for x in rows[:k]]+[("med",*x) for x in rows[mid-k//2:mid-k//2+k]]
           +[("high",*x) for x in rows[-k:]])
    print(f"{len(rows)} files available (of {len(all_rows)} in CSV)")
    print(f"RMSD range available: {rows[0][1]:.1f} .. {rows[-1][1]:.1f}")
    print(f"picking {k} per band:")
    for b in ("low","med","high"):
        band=[v for bb,fn,v in picks if bb==b]
        print(f"  {b:4s}: RMSD {min(band):.1f} .. {max(band):.1f}  ({len(band)} files)")

    R_stack=load(args.real); L=R_stack.shape[-1]; mask=soft_mask(L)
    done=0
    for band,fname,rmsd in picks:
        outpng=os.path.join(args.out,f"{band}_{rmsd:06.2f}_{fname}_sharpen.png")
        if os.path.exists(outpng):
            print(f"[skip] {os.path.basename(outpng)} exists"); done+=1; continue
        S_stack=load(glob.glob(os.path.join(args.synth_dir,fname+"*"))[0])
        if S_stack.shape[-1]!=L:
            print(f"[skip] {fname} box mismatch"); continue
        ri,si=pick_pair(R_stack,S_stack,mask)
        real_raw=_norm01(R_stack[ri]); S=S_stack[si]
        sharp_reals={name:fn(real_raw) for name,fn in SHARPEN}

        nrow=1+len(METHODS); ncol=len(SHARPEN)
        fig,ax=plt.subplots(nrow,ncol,figsize=(3.1*ncol,3.1*nrow))
        for c,(sname,_) in enumerate(SHARPEN):
            ax[0,c].imshow(norm(sharp_reals[sname],mask),cmap="gray")
            ax[0,c].set_title(sname,fontsize=11,fontweight="bold"); ax[0,c].axis("off")
        ax[0,0].text(-0.22,0.5,"SHARPENED\nREAL",transform=ax[0,0].transAxes,rotation=90,
                     va="center",ha="center",fontsize=9,fontweight="bold")
        for r_,(mname,mfunc) in enumerate(METHODS,start=1):
            for c,(sname,_) in enumerate(SHARPEN):
                try:
                    ang,mir,v=mfunc(sharp_reals[sname],S,mask)
                    ax[r_,c].imshow(norm(rot(S,ang,mir),mask),cmap="gray")
                    ax[r_,c].set_title(f"{ang:.0f}deg m{int(mir)}\nNCC {v:.2f}",fontsize=8)
                except Exception:
                    ax[r_,c].text(0.5,0.5,"fail",ha="center",va="center",transform=ax[r_,c].transAxes)
                ax[r_,c].axis("off")
            ax[r_,0].text(-0.22,0.5,mname,transform=ax[r_,0].transAxes,rotation=90,
                          va="center",ha="center",fontsize=9,fontweight="bold")
        fig.suptitle(f"[{band.upper()} RMSD]  {fname}  RMSD={rmsd:.1f}A  (pair r{ri}/s{si})\n"
                     f"rows = alignment · cols = sharpening of REAL · cells = aligned synth",
                     fontsize=12,fontweight="bold")
        fig.tight_layout(rect=[0.02,0,1,0.95])
        fig.savefig(outpng,dpi=105,bbox_inches="tight"); plt.close(fig)
        done+=1
        print(f"[{done}/{len(picks)}] -> {os.path.basename(outpng)}")

if __name__=="__main__":
    main()