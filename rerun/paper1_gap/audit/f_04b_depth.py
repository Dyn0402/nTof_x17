import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
exec(open('f_04_order.py').read().split("res={}")[0].split("OUT=")[0].replace("import sys,pickle",""))  if False else None
V={'base':([],[]),'cubic h+c':(CUBIC,CUBIC),'cubic c':([],CUBIC),'cubic h':(CUBIC,[]),
   'u4,w4 on h':(QUARTIC_SEP,[]),'u4,w4 on c':([],QUARTIC_SEP),'u4 on h':(['u4'],[]),'u4 on c':([],['u4']),
   'r4 on h':(QUARTIC_RAD,[]),'r4 on c':([],QUARTIC_RAD),'r4 on h+c':(QUARTIC_RAD,QUARTIC_RAD)}
res={}
L=['| model | dchi2 | mesh x | mesh y | cathode x | cathode y | gap tilt x |','|---|---|---|---|---|---|---|']
Rb=Fitter(CLASSES,FLOORS).fit()
for n,(eh,ec) in V.items():
    R=Fitter(CLASSES,FLOORS,extra_h=eh,extra_c=ec).fit(); f=five(R,eh,ec); res[n]=f
    L.append(f'| {n} | {R.chi2-Rb.chi2:+.1f} | '+' | '.join(f'{f[k][0]:+.2f}+-{f[k][1]:.2f}' for k in ['mesh_x','mesh_y','cath_x','cath_y','gtilt_U'])+' |')
pickle.dump(res,open(f'{OUT}/order_depth.pkl','wb'))
open(f'{OUT}/sec3b.md','w').write('\n'.join(L)); print('\n'.join(L))
