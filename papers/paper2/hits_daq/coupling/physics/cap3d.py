# Finite-difference capacitance of the MX17 readout unit cell (2x2 pads), own estimate.
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sl, json, sys
EPS0=8.854e-12
dx=20e-6; N=78  # 1.56 mm period, 2 x 2 cells of 0.78
def run(fr4=264e-6, ins=70.5e-6, er_ins=3.4, er_fr4=4.4, l7=True, esl=None):
    # z layers: thickness list and eps per interval, plane index names
    z=[];er=[];pl={}
    def add(t,e,n):
        for _ in range(n): z.append(t/n); er.append(e)
    pl['mesh']=0; add(150e-6,1.0,8)
    pl['esl']=len(z); add(ins,er_ins,5)
    pl['l4']=len(z); add(fr4,er_fr4,6)
    pl['l5']=len(z); add(fr4,er_fr4,6)
    pl['l6']=len(z); add(fr4,er_fr4,6)
    pl['l7']=len(z)
    nz=len(z)+1; z=np.array(z); er=np.array(er)
    idx=lambda i,j,k:(k*N*N+j*N+i)
    cond=np.zeros((N,N,nz),int)  # 0 free; -1 ground; 1..4 posts; 5,6 ESL
    X,Y=np.meshgrid((np.arange(N)+.5)*dx,(np.arange(N)+.5)*dx,indexing='ij')
    cell=0.78e-3
    owners={(0,0):1,(1,0):2,(0,1):3,(1,1):4}  # 1=A X col0, 4=D X col1, 2=B Y row0, 3=C Y row1
    for (ci,cj),o in owners.items():
        cx,cy=(ci+.5)*cell,(cj+.5)*cell
        cond[...,pl['l4']][(abs(X-cx)<=.34e-3)&(abs(Y-cy)<=.34e-3)]=o
        for L in ('l5','l6'):
            cond[...,pl[L]][(X-cx)**2+(Y-cy)**2<=.25e-3**2]=o
    # buses: X combs on L6 along y at +0.39 offset; Y combs on L5 along x at +0.39 offset
    f=lambda u,c:(abs(((u-c+.78e-3)%1.56e-3)-.78e-3)<=.05e-3)
    m=f(X,.5*cell+.39e-3); a=cond[...,pl['l6']]; a[m&(a==0)]=1
    m=f(X,1.5*cell+.39e-3); a[m&(a==0)]=4
    m=f(Y,.5*cell+.39e-3); a=cond[...,pl['l5']]; a[m&(a==0)]=2
    m=f(Y,1.5*cell+.39e-3); a[m&(a==0)]=3
    # ESL strips along y, width .55, two per x-period; phase in mm of first strip centre relative to column-0 centre
    n=4
    if esl is not None:
        n=6
        for s,o in ((0,5),(1,6)):
            c=(.5+s)*cell+esl*1e-3
            cond[...,pl['esl']][f2(X,c)]=o
    cond[...,0]=-1
    if l7: cond[...,pl['l7']]=-1
    rows=[];cols=[];vals=[]
    # z links
    for k in range(nz-1):
        g=EPS0*er[k]*dx*dx/z[k]
        i=np.arange(N*N)+k*N*N; rows.append(i);cols.append(i+N*N);vals.append(np.full(N*N,g))
    # x,y links, plane k: eps averaged, thickness half of adjacent intervals
    for k in range(nz):
        t=0;e=0
        if k>0: t+=z[k-1]/2; e+=er[k-1]*z[k-1]/2
        if k<nz-1: t+=z[k]/2; e+=er[k]*z[k]/2
        g=EPS0*e*dx/dx
        I,J=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
        a=idx(I,J,k).ravel(); b=idx((I+1)%N,J,k).ravel(); c=idx(I,(J+1)%N,k).ravel()
        rows+= [a,a];cols+=[b,c];vals+=[np.full(a.size,g),np.full(a.size,g)]
    r=np.concatenate(rows);c=np.concatenate(cols);v=np.concatenate(vals)
    n_tot=N*N*nz
    W=sp.coo_matrix((v,(r,c)),shape=(n_tot,n_tot)).tocsr();W=W+W.T
    L=sp.diags(np.asarray(W.sum(1)).ravel())-W
    cf=cond.transpose(2,1,0).ravel()
    free=np.where(cf==0)[0]
    Lff=L[free][:,free].tocsr(); Lfd=L[free]
    D=sl.LinearOperator(Lff.shape,matvec=lambda x:x/Lff.diagonal())
    Cm=np.zeros((n,n))
    x0=None
    for o in range(1,n+1):
        phi=np.zeros(n_tot); phi[cf==o]=1.0
        b=-Lfd@phi
        x,info=sl.cg(Lff,b,M=D,rtol=1e-7,maxiter=8000,x0=x0)
        phi[free]=x
        Q=L@phi
        for p in range(1,n+1): Cm[p-1,o-1]=Q[cf==p].sum()
    return Cm
def f2(u,c): return abs(((u-c+.78e-3)%1.56e-3)-.78e-3)<=.275e-3
if __name__=='__main__':
    out={}
    cases={'base_noESL_L7gnd':dict(),
           'noESL_noL7':dict(l7=False),
           'fr4_100um_noESL_L7gnd':dict(fr4=100e-6),
           'ESLfloat_aligned':dict(esl=0.0),
           'ESLfloat_straddle':dict(esl=0.39),
           'ESLaligned_noL7':dict(esl=0.0,l7=False),
           'ESLaligned_er4':dict(esl=0.0,er_ins=4.0,ins=50e-6)}
    for k,v in cases.items():
        C=run(**v)*256*1e12
        out[k]=C.round(3).tolist()
        print(k); print(np.array2string(C,precision=2,suppress_small=True)); sys.stdout.flush()
    json.dump(out,open(sys.argv[1],'w'),indent=1)
