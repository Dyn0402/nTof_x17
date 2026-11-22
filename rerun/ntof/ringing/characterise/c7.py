import numpy as np
def deflect(W, nz, nb=4, zsig=5.0):
    """W: (ns,512) pedestal-subtracted raw (no CM). Returns CM-and-ringing-subtracted (ns,512), beta (512,)."""
    ns=W.shape[0]
    B=W.reshape(ns,8,64)
    c=np.median(B,axis=2)                              # ns,8
    cc=np.einsum('sb,sb->b',c,c)
    beta=np.einsum('sbk,sb->bk',B,c)/np.maximum(cc,1e-6)[:,None]
    beta=np.where(cc[:,None]>1e-6,beta,1.0)
    r=B-beta[None]*c[...,None]
    z=r/nz.reshape(8,64)[None]
    flag=z.max(axis=0)>zsig                            # positive excursion left after ringing removal: maybe signal
    # grow the flag by 1 channel
    fl=flag.copy(); fl[:,1:]|=flag[:,:-1]; fl[:,:-1]|=flag[:,1:]
    b2=beta.copy()
    for b in range(8):
        idx=np.flatnonzero(fl[b])
        if len(idx)==0: continue
        good=np.flatnonzero(~fl[b])
        if len(good)<8: b2[b,idx]=1.0; continue
        for k in idx:
            nbrs=good[np.abs(good-k)<=nb]
            if len(nbrs)<2: nbrs=good[np.argsort(np.abs(good-k))[:4]]
            b2[b,k]=np.median(beta[b,nbrs])
    r=B-b2[None]*c[...,None]
    return r.reshape(ns,512), b2.ravel()
if __name__=='__main__':
    for feu in (3,4,7,8):
        z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
        N,ns,_=A.shape
        B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
        G=np.stack([deflect(A[i],nz)[0] for i in range(N)])
        for name,m in (('1-3',(ms>1)&(ms<3)),('3-8',(ms>3)&(ms<8)),('8-12',(ms>8)&(ms<12)),('12-20',(ms>12)&(ms<20)),('>30',ms>30)):
            z0=R[m]/nz; z1=G[m]/nz
            print(f'FEU {feu} {name:6s} n_neg5 {np.median((z0.min(1)<-5).sum(1)):3.0f} -> {np.median((z1.min(1)<-5).sum(1)):3.0f}  (p90 {np.percentile((z0.min(1)<-5).sum(1),90):3.0f} -> {np.percentile((z1.min(1)<-5).sum(1),90):3.0f}); n_pos5 {np.median((z0.max(1)>5).sum(1)):3.0f} -> {np.median((z1.max(1)>5).sum(1)):3.0f}; rms {np.sqrt((R[m]**2).mean()):5.1f} -> {np.sqrt((G[m]**2).mean()):5.1f}')
        m=ms>30; z0=R[m]/nz
        sig=(z0.max(1)>8)&(z0.min(1)>-3)
        q0=np.clip(R[m],0,None).sum(1)[sig]; q1=np.clip(G[m],0,None).sum(1)[sig]
        pk0=R[m].max(1)[sig]; pk1=G[m].max(1)[sig]
        print(f'   late signal channels {sig.sum()}: pos-sum ratio median {np.median(q1/q0):.3f} p10 {np.percentile(q1/q0,10):.3f} p90 {np.percentile(q1/q0,90):.3f}; peak ratio median {np.median(pk1/pk0):.3f} p10 {np.percentile(pk1/pk0,10):.3f}')
