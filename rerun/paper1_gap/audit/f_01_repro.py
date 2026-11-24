import sys; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
import time
t=time.time()
for arm in ['all','tilt','gap','geo','timing','geo_gap','no_hp']:
    cl=ARMS[arm]
    R=Fitter(cl,FLOORS).fit()
    C=COMMITTED[arm]
    md=max(abs(R[n]-C['params'][n]) for n in C['params'] if n in R.i and C['errors'][n]==C['errors'][n] and n not in('m3_kuu','m3_kww') and C['errors'][n]>0)
    me=max(abs(R.err(n)-C['errors'][n]) for n in C['params'] if n in R.i and C['errors'][n]==C['errors'][n] and C['errors'][n]>0 and n not in('m3_kuu','m3_kww'))
    print(arm,'chi2',round(R.chi2,2),round(C['chi2'],2),'maxdiff param',md,'err',me)
print(time.time()-t)
