import numpy as np
for feu in (3,8):
    z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
    N,ns,_=A.shape
    B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
    zz=R/nz; nneg=(zz.min(axis=1)<-5).sum(axis=1)
    early=(ms>3)&(ms<8)&(nneg>30); late=(ms>30)
    print('FEU',feu,'early ringing events',early.sum())
    for name,m in (('early',early),('late',late)):
        X=R[m]                                 # (n, ns, 512)
        # rank-1 fraction per block
        fr=[]
        for e in X[:300]:
            for b in range(8):
                M=e[:,b*64:(b+1)*64]
                s=np.linalg.svd(M-M.mean(0),compute_uv=False); fr.append(s[0]**2/np.sum(s**2))
        # mean residual rms per sample index (time structure in window)
        prof=np.sqrt((X**2).mean(axis=(0,2)))
        # zero crossings per channel in the window, for channels with |z|>5 somewhere
        Z=X/nz; big=np.abs(Z).max(axis=1)>5
        sgn=np.sign(X); zc=(np.diff(sgn,axis=1)!=0).sum(axis=1)
        print(f'  {name}: rank1 frac per block med {np.median(fr):.2f}; rms per sample {np.round(prof,0)}')
        print(f'     zero crossings per big channel: median {np.median(zc[big]) if big.any() else np.nan}  ; frac big channels with both signs >5sig: {np.mean((Z.max(1)>5)&(Z.min(1)<-5)) / max(np.mean(big),1e-9):.2f}')
    # correlation between channels over samples, early events: neighbouring vs far, within block vs across
    X=R[early][:200]
    C=np.zeros((512,512))
    for e in X:
        e=e-e.mean(0); s=e.std(0)+1e-6; C+=(e/s).T@(e/s)/ns
    C/=len(X)
    ch=np.arange(512); blk=ch//64
    off=~np.eye(512,dtype=bool)
    same=(blk[:,None]==blk[None,:])&off; diff=(blk[:,None]!=blk[None,:])
    nb=np.abs(ch[:,None]-ch[None,:])==1
    print(f'  early corr: neighbours {C[nb].mean():.2f}, same block {C[same].mean():.2f}, other blocks {C[diff].mean():.2f}, |corr| other {np.abs(C[diff]).mean():.2f}')
    # correlation vs channel distance within block
    for dd in (1,2,4,8,16,32,63):
        m=(np.abs(ch[:,None]-ch[None,:])==dd)&same
        print(f'    dist {dd}: {C[m].mean():+.2f}',end='')
    print()
    np.save(f'/media/ucla/work/claude-jobs/bcad1a92/ring_corr_feu{feu}.npy',C)
