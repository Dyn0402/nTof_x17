import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/home/dylan/.claude/jobs/bcad1a92/tmp/ring'); from c7 import deflect
z=np.load('/media/ucla/work/claude-jobs/bcad1a92/ring_feu8.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
N,ns,_=A.shape
G=np.stack([deflect(A[i],nz)[0] for i in range(N)])
n1=((G/nz).min(1)<-5).sum(1)
idx=np.flatnonzero((ms>1)&(ms<8)&(n1>40))[:4]
B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
fig,ax=plt.subplots(len(idx),3,figsize=(18,3.2*len(idx)))
for r,i in enumerate(idx):
    ax[r,0].imshow(R[i],aspect='auto',cmap='RdBu_r',vmin=-150,vmax=150); ax[r,0].set_title(f'plain CM ToF {ms[i]:.2f} n_neg {n1[i]}')
    ax[r,1].imshow(G[i],aspect='auto',cmap='RdBu_r',vmin=-150,vmax=150); ax[r,1].set_title('after deflect')
    mn=np.argsort(G[i].min(0))[:6]
    for c in mn: ax[r,2].plot(G[i][:,c],'b-',lw=0.8); ax[r,2].plot(A[i][:,c]-A[i][:,c].mean(),'c:',lw=0.6)
    cb=np.median(B[i],axis=2)
    for b in range(8): ax[r,2].plot(cb[:,b]-cb[:,b].mean(),'k--',lw=0.5)
    ax[r,2].set_title('ch '+','.join(map(str,mn)))
plt.tight_layout(); plt.savefig('/media/ucla/work/ntof_pass1/ana/ringing/c9_residual_C_y.png',dpi=65)
