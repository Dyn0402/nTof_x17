import numpy as np, sys
from scipy.ndimage import median_filter
sys.path.insert(0,'/home/dylan/.claude/jobs/bcad1a92/tmp/ring'); from c7 import deflect
def spatial(G, win):
    # per sample, running median along channels within each 64-ch block? use full plane but do not cross FEU edges
    return G - median_filter(G, size=(1,win), mode='nearest')
for feu in (3,8):
    z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
    N,ns,_=A.shape
    B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
    G=np.stack([deflect(A[i],nz)[0] for i in range(N)])
    for win in (33,65):
        H=np.stack([spatial(G[i],win) for i in range(N)])
        print(f'FEU {feu} window {win}')
        for name,m in (('1-3',(ms>1)&(ms<3)),('3-8',(ms>3)&(ms<8)),('8-12',(ms>8)&(ms<12)),('>30',ms>30)):
            z0=R[m]/nz; z1=H[m]/nz
            print(f'  {name:6s} n_neg5 {np.median((z0.min(1)<-5).sum(1)):3.0f} -> {np.median((z1.min(1)<-5).sum(1)):3.0f}  (p90 {np.percentile((z0.min(1)<-5).sum(1),90):3.0f} -> {np.percentile((z1.min(1)<-5).sum(1),90):3.0f}); n_pos5 {np.median((z0.max(1)>5).sum(1)):3.0f} -> {np.median((z1.max(1)>5).sum(1)):3.0f} (p90 {np.percentile((z0.max(1)>5).sum(1),90):3.0f} -> {np.percentile((z1.max(1)>5).sum(1),90):3.0f})')
        m=ms>30; z0=R[m]/nz
        sig=(z0.max(1)>8)&(z0.min(1)>-3)
        q0=np.clip(R[m],0,None).sum(1)[sig]; q1=np.clip(H[m],0,None).sum(1)[sig]
        # event-level charge: sum over signal channels per event
        Q0=(np.clip(R[m],0,None).sum(1)*sig).sum(1); Q1=(np.clip(H[m],0,None).sum(1)*sig).sum(1); ok=Q0>0
        print(f'   late signal: per-channel pos-sum ratio median {np.median(q1/q0):.3f} p10 {np.percentile(q1/q0,10):.3f}; per-event signal charge ratio median {np.median(Q1[ok]/Q0[ok]):.3f} p5 {np.percentile(Q1[ok]/Q0[ok],5):.3f}')
