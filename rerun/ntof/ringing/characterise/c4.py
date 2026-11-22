import numpy as np
for feu in (3,4,7,8):
    z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{feu}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
    N,ns,_=A.shape
    B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None])
    zz=R.reshape(N,ns,512)/nz; nneg=(zz.min(axis=1)<-5).sum(axis=1)
    early=np.flatnonzero((ms>1)&(ms<12)&(nneg>30))[:300]
    U=np.zeros((len(early),8,64)); S=np.zeros((len(early),8))
    for i,e in enumerate(early):
        for b in range(8):
            M=R[e,:,b,:]                     # ns x 64
            u,s,vt=np.linalg.svd(M,full_matrices=False)
            U[i,b]=vt[0]*np.sign(vt[0].sum()+1e-9); S[i,b]=s[0]**2/np.sum(s**2)
    # stability: cos between event u vectors and the mean u (per block)
    mu=U.mean(0); mu/=np.linalg.norm(mu,axis=1,keepdims=True)
    cs=np.abs(np.einsum('ibk,bk->ib',U,mu))
    # how linear is the mean profile: fraction of mean-u variance explained by a line in channel index
    k=np.arange(64); lin=[]
    for b in range(8):
        c=np.polyfit(k,mu[b],1); r=mu[b]-np.polyval(c,k); lin.append(1-r.var()/mu[b].var())
        c2=np.polyfit(k,mu[b],2); r2=mu[b]-np.polyval(c2,k)
    print(f'FEU {feu}: events {len(early)}  rank1 frac med {np.median(S):.2f}  |cos(u_event, u_mean)| median per block {np.round(np.median(cs,axis=0),2)}  line R2 of mean u {np.round(lin,2)}')
    np.save(f'/media/ucla/work/claude-jobs/bcad1a92/ring_u_feu{feu}.npy',mu)
