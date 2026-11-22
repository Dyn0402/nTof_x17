import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
z=np.load('/media/ucla/work/claude-jobs/bcad1a92/ring_feu3.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=np.maximum(z['noise'],1)
N,ns,_=A.shape
B=A.reshape(N,ns,8,64); R=(B-np.median(B,axis=3)[...,None]).reshape(N,ns,512)
zz=R/nz; nneg=(zz.min(axis=1)<-5).sum(axis=1)
idx=list(np.flatnonzero((ms>3)&(ms<8)&(nneg>40))[:3])+list(np.flatnonzero((ms>12)&(ms<20)&(nneg>5))[:1])+list(np.flatnonzero((ms>40)&((zz.max(axis=1)>5).sum(axis=1)>8))[:2])
fig,ax=plt.subplots(len(idx),2,figsize=(14,3.2*len(idx)))
for r,i in enumerate(idx):
    im=ax[r,0].imshow(R[i],aspect='auto',cmap='RdBu_r',vmin=-150,vmax=150); ax[r,0].set_title(f'ev {z["ev"][i]} ToF {ms[i]:.2f} ms nneg {nneg[i]}'); plt.colorbar(im,ax=ax[r,0])
    ax[r,0].set_xlabel('channel'); ax[r,0].set_ylabel('sample')
    for b in range(8): ax[r,0].axvline(b*64,color='k',lw=0.5)
    # waveforms of the 10 most negative and 10 most positive channels
    mn=np.argsort(R[i].min(0))[:6]; mx=np.argsort(-R[i].max(0))[:6]
    for c in mn: ax[r,1].plot(R[i][:,c],'b-',lw=0.8)
    for c in mx: ax[r,1].plot(R[i][:,c],'r-',lw=0.8)
    ax[r,1].plot(np.median(B[i],axis=2)-np.median(B[i],axis=2).mean(0),'k--',lw=0.6)
plt.tight_layout(); plt.savefig('/media/ucla/work/ntof_pass1/ana/ringing/c5_events_A_x.png',dpi=70)
