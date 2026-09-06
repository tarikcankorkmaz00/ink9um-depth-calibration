# -*- coding: utf-8 -*-
"""Step B2 - the phase effect under real blended inference.

Sliding window with Hann blending over a 512x512 region, positions chosen
exactly as infer.py's _sliding_positions_1d chooses them, scored on the
inner 384x384. Only the stride changes. Produces
experiment2/data/blended.json.

This is a MEASUREMENT step: it needs a CUDA GPU, the villa package, the
ink_9um checkpoint and the label and volume chunks. See experiment2/README.md
for the layout of VZ_ROOT. Results go to VZ_WORK (default: <repo>/work).
The committed data in experiment2/data/ is what this produced; nothing in
experiment2/verify/ needs any of it.
"""
# A few identifiers below are in the author's own language, left as they ran:
#   tuval = canvas, yonga = tile, kayma = shift, faz = phase, secim = selection,
#   aday = candidate, kok = origin, cerceve = frame, cift = pair, olcum = measurement,
#   ortak = shared, kosu/kosum = run, sonuc = result, ham = raw, etiket = label.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_os.path.dirname(_HERE))
ROOT = _os.environ.get("VZ_ROOT", _os.path.join(_REPO, "cache"))
OUT = _os.environ.get("VZ_WORK", _os.path.join(_REPO, "work"))
_os.makedirs(OUT, exist_ok=True)
LABELS = _os.path.join(ROOT, "cache", "labels")
VOLUMES = _os.path.join(ROOT, "cache", "volumes")
LABEL_INDEX = _os.path.join(ROOT, "cache", "label_index.json")
SEGMENTS = _os.path.join(ROOT, "segments.json")
CHECKPOINT = _os.path.join(ROOT, "model", "ink_9um",
                           "hybrid_3d2d-seed42", "step-075000.pth")
VILLA_SRC = _os.path.join(ROOT, "villa", "vesuvius", "src")
TRANSFORM = _os.path.join(_os.path.dirname(_HERE), "data", "context.json")


def _manifest():
    """segments.json, tolerating either top-level key name."""
    m = json.load(open(SEGMENTS, encoding="utf-8"))
    return m.get("segments", m.get("segmentler"))

import json, os, sys, math
import numpy as np, torch, numcodecs
from sklearn.metrics import roc_auc_score
sys.path.insert(0, VILLA_SRC)
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from vesuvius.image_proc.intensity.normalization import normalize_robust
IDX = json.load(open(LABEL_INDEX)); CACHE = LABELS
SEGS = {s["key"]: s for s in _manifest()}
VOL = VOLUMES; _BL = numcodecs.Blosc()
ck=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
WIN=int(ck["config"]["patch_size"][0]); cfg=InkConfig.from_mapping(ck["config"])
mm=make_model(cfg); st=ck["model"]
if st and all(str(k).startswith("module.") for k in st): st={k[len("module."):]:v for k,v in st.items()}
mm.load_state_dict(st,strict=False)
model=TargetModel(mm,input_pad_depth_to=cfg.model.input_pad_depth_to).eval().cuda()

def tuval(key,kind):
    s=SEGS[key]; nz,H,W=s["label_shape_zyx"]; zc=s["annotation_center_channel"]
    out=np.zeros((H,W),np.uint8)
    for yx,h in IDX.get(key+"|"+kind,{}).items():
        fp=os.path.join(CACHE,h[:2],h)
        if not(os.path.exists(fp) and os.path.getsize(fp)): continue
        a=np.frombuffer(_BL.decode(open(fp,"rb").read()),np.uint8).reshape(nz,128,128)
        y,x=(int(v) for v in yx.split(",")); y0,x0=y*128,x*128
        sl=(a[zc]>0).astype(np.uint8); h_,w_=min(128,H-y0),min(128,W-x0)
        out[y0:y0+h_,x0:x0+w_]=sl[:h_,:w_]
    return out
def chunk(d,iy,ix,nz):
    fp="%s/c_0_%d_%d"%(d,iy,ix)
    if not(os.path.exists(fp) and os.path.getsize(fp)==nz*128*128): return None
    return np.fromfile(fp,np.uint8).reshape(nz,128,128)

def hann2d(n=128):
    w=np.hanning(n+2)[1:-1]; return np.outer(w,w).astype(np.float32)

def sliding(length,ps,stride):
    if length<=ps: return [0]
    pos=list(range(0,length-ps+1,max(1,stride)))
    if pos[-1]!=length-ps: pos.append(length-ps)
    return pos

def kosu(fam,key,vd,nzv,ty0,tx0,TH,TW,strides):
    s=SEGS[key]; H,W=s["label_shape_zyx"][1:]
    ny,nx=TH//128,TW//128
    blocks={}
    for iy in range(ty0//128,ty0//128+ny):
        for ix in range(tx0//128,tx0//128+nx):
            c=chunk(vd,iy,ix,nzv)
            if c is None: return None
            blocks[(iy,ix)]=c
    RAW=np.zeros((nzv,TH,TW),np.uint8)
    for (iy,ix),c in blocks.items():
        RAW[:,iy*128-ty0:iy*128-ty0+128, ix*128-tx0:ix*128-tx0+128]=c
    if fam=="native9":
        st_=s["annotation_center_channel"]-WIN//2; G=RAW[st_:st_+WIN].astype(np.float32)
    else:
        z0,z1=s["source_z_slice"]; pool=(z1-z0)//s["label_shape_zyx"][0]
        st_=z0+pool*(s["annotation_center_channel"]-WIN//2)
        G=np.rint(RAW[st_:st_+pool*WIN].astype(np.float32).reshape(WIN,pool,TH,TW).mean(axis=1))
    hw=hann2d(); out={}
    for stride in strides:
        acc=np.zeros((TH,TW),np.float32); wsum=np.zeros((TH,TW),np.float32)
        pos_y=sliding(TH,128,stride); pos_x=sliding(TW,128,stride)
        batch=[]; loc=[]
        for y0 in pos_y:
            for x0 in pos_x:
                g=G[:,y0:y0+128,x0:x0+128]
                batch.append(normalize_robust(g)); loc.append((y0,x0))
                if len(batch)==8:
                    xb=torch.from_numpy(np.stack(batch)).unsqueeze(1).cuda()
                    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): lg=model(xb)
                    lg=lg.float().squeeze(1).cpu().numpy()
                    for (yy,xx),m in zip(loc,lg):
                        acc[yy:yy+128,xx:xx+128]+=m*hw; wsum[yy:yy+128,xx:xx+128]+=hw
                    batch=[];loc=[]
        if batch:
            xb=torch.from_numpy(np.stack(batch)).unsqueeze(1).cuda()
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): lg=model(xb)
            lg=lg.float().squeeze(1).cpu().numpy()
            for (yy,xx),m in zip(loc,lg):
                acc[yy:yy+128,xx:xx+128]+=m*hw; wsum[yy:yy+128,xx:xx+128]+=hw
        out[stride]=acc/np.maximum(wsum,1e-6)
    return out

SET=[("native9","native9-scrollprizeorg-21slices/%s",VOL+"/%s",28,["w035","w039","w040","w041","w044"]),
     ("aligned","aligned-scrollprizeorg-21slices/%s",VOL+"/ali_%s",109,
      [("pherc0139-w035","w035"),("pherc0139-w039","w039"),("pherc0139-w040","w040"),
       ("pherc0139-w041","w041"),("pherc0139-w028","w044")])]
STR=[64,48,51,77,80,88,90,38,40]
sonuc={}
for fam,kf,vf,nzv,segs in SET:
    for it in segs:
        if fam=="native9": key=kf%it; vd=vf%it; nm=it
        else: key=kf%it[0]; vd=vf%it[1]; nm=it[0]
        s=SEGS[key]; H,W=s["label_shape_zyx"][1:]
        ink=tuval(key,"ink"); sup=tuval(key,"sup")
        # find a 4x4 chunk (512x512) region: the most densely supervised one
        best=None
        for iy in range(0,H//128-3):
            for ix in range(0,W//128-3):
                y0,x0=iy*128,ix*128
                if y0+512>H or x0+512>W: continue
                m=sup[y0:y0+512,x0:x0+512]>0
                if m.sum()<150000: continue
                i=(ink[y0:y0+512,x0:x0+512]>0)[m]
                if i.sum()<20000 or (~i).sum()<20000: continue
                sc=m.sum()
                if best is None or sc>best[0]: best=(sc,y0,x0)
        if best is None: print(fam,nm,"no suitable 512x512 region"); continue
        _,ty0,tx0=best
        r=kosu(fam,key,vd,nzv,ty0,tx0,512,512,STR)
        if r is None: print(fam,nm,"missing chunk"); continue
        # inner region: 64 px of blend edge is discarded
        m=(sup[ty0+64:ty0+448,tx0+64:tx0+448]>0)
        yl=(ink[ty0+64:ty0+448,tx0+64:tx0+448]>0)[m]
        row={}
        for stnum,P in r.items():
            row[stnum]=float(roc_auc_score(yl,P[64:448,64:448][m]))
        sonuc[f"{fam}|{nm}"]=row
        print("%-9s %-16s"%(fam,nm)+"  ".join("s%d=%.5f"%(k,v) for k,v in sorted(row.items())))
json.dump(sonuc, open(os.path.join(OUT, "blended.json"), "w"), indent=1)
print()
import numpy as np
print("=== NEIGHBOURING STRIDE PAIRS: coverage nearly equal, only the phase differs ===")
for a,b in [(48,51),(80,77),(88,90),(40,38)]:
    d=[sonuc[k][b]-sonuc[k][a] for k in sonuc]
    print("  stride %3d (mod8=%d) -> %3d (mod8=%d): delta %+0.5f  (%d/%d worse)"%(
        a,a%8,b,b%8,np.mean(d),sum(1 for z in d if z<0),len(d)))
print()
print("=== against stride 64 (the --overlap 0.5 default) ===")
for stn in STR:
    d=[sonuc[k][stn]-sonuc[k][64] for k in sonuc]
    print("  stride %3d (mod8=%d): %+0.5f  (%d/%d worse)"%(stn,stn%8,np.mean(d),sum(1 for z in d if z<0),len(d)))
