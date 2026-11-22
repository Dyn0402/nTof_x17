import numpy as np
def regress(Araw, nblk=8):
    """per channel x = a + beta * c_b(t) with c_b the block median; returns residual"""
    N,ns,nc=Araw.shape
    B=Araw.reshape(N,ns,nblk,64)
    c=np.median(B,axis=3)                              # N,ns,nblk
    cc=c-c.mean(axis=1,keepdims=True)                  # centred
    xc=B-B.mean(axis=1,keepdims=True)
    beta=np.einsum('nsbk,nsb->nbk',xc,cc)/np.maximum(np.einsum('nsb,nsb->nb',cc,cc),1e-6)[...,None]
    res=xc-beta[:,None]*cc[...,None]
    # restore the per-channel mean level relative to the block (the median baseline offset)
    return res.reshape(N,ns,nc), beta
for feu in (3,8):
    z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
    N,ns,_=A.shape
    B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
    G,beta=regress(A)
    for name,m in (('1-3',(ms>1)&(ms<3)),('3-8',(ms>3)&(ms<8)),('8-12',(ms>8)&(ms<12)),('12-20',(ms>12)&(ms<20)),('>30',ms>30)):
        z0=R[m]/nz; z1=G[m]/nz
        print(f'FEU {feu} {name:6s} n_neg5 med {np.median((z0.min(1)<-5).sum(1)):4.0f} -> {np.median((z1.min(1)<-5).sum(1)):4.0f};  n_pos5 {np.median((z0.max(1)>5).sum(1)):4.0f} -> {np.median((z1.max(1)>5).sum(1)):4.0f}; rms {np.sqrt((R[m]**2).mean()):5.1f} -> {np.sqrt((G[m]**2).mean()):5.1f}; beta spread {np.median(np.std(beta[m],axis=2)):.2f}')
    # signal preservation, late events: channels with a positive pulse > 8 sigma and no negative
    m=ms>30; z0=R[m]/nz
    sig=(z0.max(1)>8)&(z0.min(1)>-3)
    q0=np.clip(R[m],0,None).sum(1)[sig]; q1=np.clip(G[m],0,None).sum(1)[sig]
    pk0=R[m].max(1)[sig]; pk1=G[m].max(1)[sig]
    print(f'   late signal channels {sig.sum()}: positive-sum ratio median {np.median(q1/q0):.3f} (p10 {np.percentile(q1/q0,10):.3f}), peak ratio median {np.median(pk1/pk0):.3f} (p10 {np.percentile(pk1/pk0,10):.3f})')
