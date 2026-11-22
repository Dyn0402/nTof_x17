import numpy as np
Z={f:np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{f}.npz') for f in (3,4,7,8)}
ev=Z[3]['ev']; ms=Z[3]['ms']
assert all((Z[f]['ev']==ev).all() for f in Z)
cm={}; beta={}
for f in Z:
    A=Z[f]['A'].astype(np.float32); N,ns,_=A.shape; B=A.reshape(N,ns,8,64)
    c=np.median(B,axis=3); cm[f]=c
    cc=np.einsum('nsb,nsb->nb',c,c); beta[f]=np.einsum('nsbk,nsb->nbk',B,c)/np.maximum(cc,1e-6)[...,None]
def corr(a,b):
    a=a-a.mean(-1,keepdims=True); b=b-b.mean(-1,keepdims=True)
    return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1)+1e-9)
for name,m in (('3-8 ms',(ms>3)&(ms<8)),('>30 ms',ms>30)):
    print(name, 'n', m.sum())
    # coherence of CM waveform across blocks within FEU, across FEUs (same event)
    c3=cm[3][m]; c7=cm[7][m]; c4=cm[4][m]
    print('  block0 vs block5 same FEU3:', np.median(corr(c3[:,:,0],c3[:,:,5])).round(2),
          ' FEU3 vs FEU4 (A x vs A y, same chamber):', np.median(corr(c3[:,:,0],c4[:,:,0])).round(2),
          ' FEU3 vs FEU7 (A vs C):', np.median(corr(c3[:,:,0],c7[:,:,0])).round(2))
    print('  CM amplitude (std over samples) med FEU3 blk0', np.median(c3[:,:,0].std(1)).round(1))
    # phase relative to flash: correlation between different events at similar ToF
    idx=np.flatnonzero(m); r=[]
    for i,j in zip(idx[:-1],idx[1:]): r.append(corr(cm[3][i,:,0],cm[3][j,:,0]))
    print('  consecutive events same FEU block: median corr', np.median(r).round(2), ' |corr|', np.median(np.abs(r)).round(2))
    # beta pattern stability across events (per block), early only meaningful
    b=beta[3][m]; ok=np.abs(cm[3][m]).std(1).min(1)>20
    b=b[ok]
    if len(b)>10:
        bc=[np.median([corr(b[i,k],b[i+1,k]) for i in range(min(len(b)-1,200))]) for k in range(8)]
        print('  beta pattern corr between events, per block', np.round(bc,2), ' beta spread (std over ch) med', np.median(b.std(2)).round(2))
