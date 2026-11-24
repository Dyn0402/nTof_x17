import sys,pickle,copy; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
for r in OBS: print(r,'cells with G',sum(np.isfinite(o['G']) for o in OBS[r]),'GY',sum(np.isfinite(o['GY']) for o in OBS[r]),'dX',sum(np.isfinite(o['dX']) for o in OBS[r]),'dY',sum(np.isfinite(o['dY']) for o in OBS[r]))
print('U range with G (run1/run2):',[ (min(o['U'] for o in OBS[r] if np.isfinite(o['G'])),max(o['U'] for o in OBS[r] if np.isfinite(o['G']))) for r in OBS])
print('W range with GY:',[ (min(o['W'] for o in OBS[r] if np.isfinite(o['GY'])),max(o['W'] for o in OBS[r] if np.isfinite(o['GY']))) for r in OBS])
print('G cells map run2:'); 
for j in range(3,-1,-1): print(' '.join('G' if np.isfinite([o for o in OBS['run2'] if o['i']==i and o['j']==j][0]['G']) else '.' for i in range(4)))
# product obs
def prod(view):
    O=copy.deepcopy(OBS)
    for r in O:
        for o in O[r]:
            T,v=o['T'],o[f'v_{view}']
            o['T']=T*v; o['T_err']=T*v*np.hypot(o['T_err']/T,o[f'v_{view}_err']/v)
    return O
H0=('h_lu','h_lw','h_kuu','h_kww','h_kuw')
ZX=()
def run(label,cls,beta=None,obs=None,cf=None,zero=H0+('g0',),classes_override=None,hfree=False):
    kw=dict(cellfilter=cf,obs=obs)
    kw['zero']=(zero if not hfree else ())+ZX
    if beta is not None: kw['beta']=beta
    R=Fitter(cls,FLOORS,**kw).fit()
    return R
def table(cf,tag):
    out=[]
    def add(lab,R,s=1.0):
        gu=R.lin({'c_kuu':1,'h_kuu':-1}); gw=R.lin({'c_kww':1,'h_kww':-1}); tu=R.lin({'c_lu':1,'h_lu':-1})
        out.append((lab,gu[0]*s,gu[1]*s,gw[0]*s,gw[1]*s,tu[0]*s,tu[1]*s,R.chi2,R.n-R.npar))
        print(lab,'nobs',R.n,'npar',R.npar)
    # gap-only classes: gap g = g0 + c; h zero
    add('G alone',run('',('G',),cf=cf,zero=H0))
    add('GY alone',run('',('GY',),cf=cf,zero=H0))
    for b in (1.0,0.68):
        add(f'T alone (beta={b}) [g0 fixed 28.2]',run('',('T',),beta=b,cf=cf,zero=H0+('g0',)))
        add(f'v_x alone (beta={b})',run('',('v',),beta=b,cf=cf,zero=H0+('g0',)))
    # separate views for v: fake by obs where other view's v is nan
    for view in 'xy':
        O=copy.deepcopy(OBS)
        for r in O:
            for o in O[r]: o['v_'+('y' if view=='x' else 'x')+'_err']=np.nan
        for b in (1.0,):
            add(f'v_{view} only (beta={b})',run('',('v',),beta=b,obs=O,cf=cf,zero=H0+('g0',)))
    for view in 'xy':
        O=prod(view)
        add(f'v_{view}*T product (beta-free)',run('',('T',),beta=0.0,obs=O,cf=cf,zero=H0+('g0',)))
    # T with v_x*T and v_y*T together
    # tilt route (needs h): hp + w0
    add('tilt route: hp + w0 (a=1/2)',run('',('hp','w0'),cf=cf,hfree=True))
    add('hp + d (timing-free tilt)',run('',('hp','d'),cf=cf,hfree=True))
    add('hp + w0 + d',run('',('hp','w0','d'),cf=cf,hfree=True))
    add('G+GY+hp (geo_gap)',run('',('hp','G','GY'),cf=cf,hfree=True))
    add('T+v (beta=1) alone joint, g0 fixed',run('',('T','v'),beta=1.0,cf=cf,zero=H0+('g0',)))
    L=[f'**{tag}**\n','| method | gap_kuu (c-h) | gap_kww | gap tilt U | chi2/ndf |\n|---|---|---|---|---|']
    for lab,a,ea,b,eb,c,ec,chi,nd in out:
        f=lambda x,e: 'not constrained' if e<1e-6 else f'{x:+.2f} +- {e:.2f}'
        L.append(f'| {lab} | {f(a,ea)} | {f(b,eb)} | {f(c,ec)} | {chi:.0f}/{nd} |')
    return L
L=table(None,'All cells (as in the joint fit)')
ZX=('h_kuu','h_kuw','c_kuu','c_kuw')
L+=['']+table(lambda o: np.isfinite(o['G']),'Same cells as G, curvature along U switched off (c_kuu = c_kuw = h_kuu = h_kuw = 0; G has only two columns, U = -0.43 and +0.42): gap TILT along U is the quantity')
open(f'{OUT}/sec6.md','w').write('\n'.join(L)); print('\n'.join(L))
