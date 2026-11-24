import pandas as pd, numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=sorted(pd.read_pickle('acq_occ.pkl'),key=lambda r:r['tag'])
P=json.load(open('posmap.json')); py=np.array(P['B']['y'],float)-199.29
yreg=np.where((py>-200)&(py<-160))[0]
t=pd.to_datetime([r['t'] for r in R]); cos=np.array([r['rclass']=='cosmic' for r in R])
fig,ax=plt.subplots(2,1,figsize=(7.5,5.6),sharex=True)
for c,col in ((213,'C3'),(219,'C1'),(205,'C2'),(209,'C4'),(214,'0.5')):
    o=np.array([100*r['occ_1'][c]/r['n'] for r in R])
    ax[0].plot(t[~cos],o[~cos],'.',ms=3,color=col,label=f'D-x ch {c} (beam)')
    ax[0].plot(t[cos],o[cos],'x',ms=3,color=col)
ax[0].set_ylabel('hit occupancy (%)'); ax[0].set_title('D x FEU 1 connector 4 (x_local 10-25 mm); x = beam-off acquisitions',fontsize=9); ax[0].legend(fontsize=7,ncol=3)
for lab,f,col in (('B-x ch 258',lambda r:r['occ_5'][258],'C0'),('B-x ch 259',lambda r:r['occ_5'][259],'C3'),('B-y ch 13-63 mean',lambda r:r['occ_6'][yreg].mean(),'C2')):
    o=np.array([100*f(r)/r['n'] for r in R])
    ax[1].semilogy(t[~cos],o[~cos],'.',ms=3,color=col,label=lab); ax[1].semilogy(t[cos],o[cos],'x',ms=3,color=col)
ax[1].set_ylabel('hit occupancy (%)'); ax[1].set_title('B box x_local -52..-44 / y_local -200..-160',fontsize=9); ax[1].legend(fontsize=7)
for a in ax:
    for tt in ('2026-08-04 09:12','2026-08-06 14:46'): a.axvline(pd.Timestamp(tt),color='k',lw=0.6,ls=':')
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig('fig_channels.png',dpi=130)
