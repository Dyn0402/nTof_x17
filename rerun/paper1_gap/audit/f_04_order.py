import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
V={'base':([],[]),'cubic h':(CUBIC,[]),'cubic c':([],CUBIC),'cubic h+c':(CUBIC,CUBIC),
   'quartic u4,w4 h':(QUARTIC_SEP,[]),'quartic u4,w4 c':([],QUARTIC_SEP),'quartic u4,w4 h+c':(QUARTIC_SEP,QUARTIC_SEP),
   'radial r4 h':(QUARTIC_RAD,[]),'radial r4 c':([],QUARTIC_RAD),'radial r4 h+c':(QUARTIC_RAD,QUARTIC_RAD),
   'u4 only h':(['u4'],[]),'u4 only c':([],['u4']),'w4 only h':(['w4'],[]),'w4 only c':([],['w4'])}
res={}
f0,R0=retune(CLASSES,None)
print('retuned base floors',{k:round(v,4) for k,v in f0.items()},'committed',{k:round(v,4) for k,v in FLOORS.items()})
Rb=Fitter(CLASSES,FLOORS).fit()
rows=[]
for name,(eh,ec) in V.items():
    R=Fitter(CLASSES,FLOORS,extra_h=eh,extra_c=ec).fit()
    fl,R2=retune(CLASSES,None,extra_h=eh,extra_c=ec)
    q=quants(R); q2=quants(R2)
    ex={n:(R[n],R.err(n)) for n in R.names if n.split('_')[0] in('h','c') and n[2:] in EXTRA}
    rows.append(dict(name=name,chi2=R.chi2,npar=R.npar,dchi2=R.chi2-Rb.chi2,dnpar=R.npar-Rb.npar,q=q,floors=fl,q2=q2,chi2_2=R2.chi2,extra=ex,per=R.per))
pickle.dump(rows,open(f'{OUT}/order.pkl','wb'))
L=['| model | npar | chi2 (committed floors) | dchi2 vs base | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta |','|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    q=r['q']; L.append(f"| {r['name']} | {r['npar']} | {r['chi2']:.1f} | {r['dchi2']:+.1f} ({r['dnpar']} par) | "+' | '.join(f"{q[k][0]:+.2f}+-{q[k][1]:.2f}" for k in ['h_kuu','h_kww','c_kuu','c_kww','gap_kuu','beta'])+' |')
L+=['','Re-tuned floors (chi2/n = 1 per class re-imposed with the extra terms):','','| model | floor hp | w0 | d | G | GY | T | v | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    q=r['q2']; f=r['floors']
    L.append(f"| {r['name']} | "+' | '.join(f"{f[c]:.4f}" for c in CLASSES)+' | '+' | '.join(f"{q[k][0]:+.2f}+-{q[k][1]:.2f}" for k in ['h_kuu','h_kww','c_kuu','c_kww','gap_kuu','beta'])+' |')
L+=['','Fitted higher-order coefficients (committed floors; units mm per (100 mm)^n):','','| model | terms |','|---|---|']
for r in rows:
    if r['extra']: L.append(f"| {r['name']} | "+', '.join(f"{n} {v:+.3f}+-{e:.3f} ({abs(v/e):.1f} sigma)" for n,(v,e) in r['extra'].items())+' |')
L+=['','Per-class chi2 (committed floors), base vs best:','']
open(f'{OUT}/sec3.md','w').write('\n'.join(L)); print('\n'.join(L))
