import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
res=pickle.load(open(f'{OUT}/resample.pkl','rb')); od=pickle.load(open(f'{OUT}/order_depth.pkl','rb')); mix=pickle.load(open(f'{OUT}/mix.pkl','rb'))
keys=['mesh_x','mesh_y','cath_x','cath_y','gtilt_U']
base=five(Fitter(CLASSES,FLOORS).fit())
# floors sensitivity
fs={}
for lab,fn in (('all floors x0.5',lambda f:{k:v*.5 for k,v in f.items()}),('all floors x2',lambda f:{k:v*2 for k,v in f.items()}),
               ('T,v floors x0.5',lambda f:{**f,'T':f['T']*.5,'v':f['v']*.5}),('T,v floors x2',lambda f:{**f,'T':f['T']*2,'v':f['v']*2}),
               ('hp,w0 floors x0.5',lambda f:{**f,'hp':f['hp']*.5,'w0':f['w0']*.5}),('hp,w0 floors x2',lambda f:{**f,'hp':f['hp']*2,'w0':f['w0']*2})):
    fs[lab]=five(Fitter(CLASSES,fn(FLOORS)).fit())
# resampling error on depth: bootstrap std of kuu*1.125 (base model depth = 1.125 k), jackknife same
def rs(arm,k,fac=1.125):
    B=np.array([x[k] for x in res['bs'][arm]]); v=np.array([x[k] for x in res['jk'][arm]['leave']]); n=len(v)
    jk=np.sqrt((n-1)/n*np.sum((v-v.mean())**2))
    return fac*max(np.std(B),jk)
r={'mesh_x':rs('all','h_kuu'),'mesh_y':rs('all','h_kww'),'cath_x':rs('all','c_kuu'),'cath_y':rs('all','c_kww'),'gtilt_U':rs('all','gtilt_U',1.0)}
# model-order spread: half range over variants
def spread(src):
    out={}
    for k in keys:
        v=[d[k][0] for d in src]; out[k]=(min(v),max(v))
    return out
so=spread(list(od.values()))
sa=spread([x[3] for x in mix['all'][:]]+[])
# a-scan from mix['all'] rows (a,chi2,q,f)
sa=spread([row[3] for row in mix['all']])
sf=spread(list(fs.values())+[base])
# per run
pr=pickle.load(open(f'{OUT}/resample.pkl','rb'))['perrun']
L=['| quantity | committed fit | fit error | resampling error (max of jackknife, bootstrap over cells) | model-order range | a = 0.3..0.7 range | floor x0.5..x2 range | total (quadrature of fit-err-free parts: resampling + half-ranges) |','|---|---|---|---|---|---|---|---|']
tot={}
for k in keys:
    hr=lambda s:(s[k][1]-s[k][0])/2
    t=np.sqrt(r[k]**2+hr(so)**2+hr(sa)**2+hr(sf)**2); tot[k]=t
    L.append(f'| {k} | {base[k][0]:+.2f} | {base[k][1]:.2f} | {r[k]:.2f} | {so[k][0]:+.2f}..{so[k][1]:+.2f} | {sa[k][0]:+.2f}..{sa[k][1]:+.2f} | {sf[k][0]:+.2f}..{sf[k][1]:+.2f} | {t:.2f} |')
L+=['','Floor-sensitivity detail:','','| floors | mesh x | mesh y | cathode x | cathode y | gap tilt x |','|---|---|---|---|---|---|']
for lab,f in fs.items(): L.append(f'| {lab} | '+' | '.join(f'{f[k][0]:+.2f}+-{f[k][1]:.2f}' for k in keys)+' |')
open(f'{OUT}/sec9.md','w').write('\n'.join(L)); print('\n'.join(L))
