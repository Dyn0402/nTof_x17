import numpy as np
def deflect2(W, nz, nb=4, zsig=5.0, deriv=True):
    ns=W.shape[0]; B=W.reshape(ns,8,64)
    c=np.median(B,axis=2)                              # ns,8
    dc=np.gradient(c,axis=0) if deriv else None
    out=np.empty_like(B); coef=np.zeros((8,64,2 if deriv else 1))
    nzb=nz.reshape(8,64)
    for b in range(8):
        X=np.stack([c[:,b]]+([dc[:,b]] if deriv else []),1)       # ns,k
        if np.abs(X).sum()<1e-6: out[:,b]=B[:,b]; continue
        P,_,_,_=np.linalg.lstsq(X,B[:,b],rcond=None)                # k,64
        r=B[:,b]-X@P
        flag=(r/nzb[b]).max(0)>zsig
        fl=flag.copy(); fl[1:]|=flag[:-1]; fl[:-1]|=flag[1:]
        good=np.flatnonzero(~fl)
        P2=P.copy()
        for k in np.flatnonzero(fl):
            if len(good)<8: P2[:,k]=np.r_[1.0,np.zeros(P.shape[0]-1)]; continue
            nbrs=good[np.abs(good-k)<=nb]
            if len(nbrs)<2: nbrs=good[np.argsort(np.abs(good-k))[:4]]
            P2[:,k]=np.median(P[:,nbrs],axis=1)
        out[:,b]=B[:,b]-X@P2; coef[b]=P2.T
    return out.reshape(ns,512), coef
if __name__=='__main__':
    for feu in (3,8):
        z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
        N,ns,_=A.shape
        B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
        G=np.stack([deflect2(A[i],nz)[0] for i in range(N)])
        for name,m in (('1-3',(ms>1)&(ms<3)),('3-8',(ms>3)&(ms<8)),('8-12',(ms>8)&(ms<12)),('>30',ms>30)):
            z0=R[m]/nz; z1=G[m]/nz
            print(f'FEU {feu} {name:6s} n_neg5 {np.median((z0.min(1)<-5).sum(1)):3.0f} -> {np.median((z1.min(1)<-5).sum(1)):3.0f}  (p90 {np.percentile((z0.min(1)<-5).sum(1),90):3.0f} -> {np.percentile((z1.min(1)<-5).sum(1),90):3.0f}); n_pos5 {np.median((z0.max(1)>5).sum(1)):3.0f} -> {np.median((z1.max(1)>5).sum(1)):3.0f} (p90 {np.percentile((z1.max(1)>5).sum(1),90):3.0f})')
        m=ms>30; z0=R[m]/nz
        sig=(z0.max(1)>8)&(z0.min(1)>-3)
        q0=np.clip(R[m],0,None).sum(1)[sig]; q1=np.clip(G[m],0,None).sum(1)[sig]
        print(f'   late signal: pos-sum ratio median {np.median(q1/q0):.3f} p10 {np.percentile(q1/q0,10):.3f} p90 {np.percentile(q1/q0,90):.3f}')
