import numpy as np
from scipy.stats import norm
def run(z0,z1,slope,p,D2,s02=0.0,q=0.2,trim=0.0):
    # track x=slope*(z-z0), z in [z0,z1]; sigma^2=D2*z+s02. Strips width p. returns tan-ratio for estimators
    z=np.linspace(z0,z1,20001); x=slope*(z-z0); sg=np.sqrt(D2*z+s02)
    xs=np.arange(-p*5-(0),slope*(z1-z0)+p*5,p)
    C,Q,Y=[],[],[]
    for c in xs:
        w=norm.cdf((c+p/2-x)/sg)-norm.cdf((c-p/2-x)/sg)
        if w.sum()<=0: continue
        C.append(c);Q.append(w.sum());cum=np.cumsum(w)/w.sum();Y.append(((w*z).sum()/w.sum(),z[np.searchsorted(cum,q)]))
    C,Q,Y=np.array(C),np.array(Q),np.array(Y)
    m=Q>0.15*Q.max()
    if trim>0: m&=(C>trim*slope*(z1-z0))&(C<(1-trim)*slope*(z1-z0))
    r=[]
    for k in range(2):
        b=np.polyfit(C[m],Y[m,k],1)[0]; r.append((1/slope)/b)   # tan_reco/tan_true = dz_true/dz_reco
    return r,m.sum()
print("system               s=dx/dz  sig/col  n | allstrips centroid quant20 | mid60 centroid quant20")
D=lambda d: (d/np.sqrt(10))**2  # d in mm/sqrt(cm) -> mm
for name,z0,z1,sl,p,d,s0 in [
 ("MX17 X 0.78 diff",0.5,28,0.3,0.78,0.35,0.0),
 ("MX17 X +s0 .3",0.5,28,0.3,0.78,0.35,0.3**2),
 ("MX17 tan.6",0.5,28,0.6,0.78,0.35,0.0),
 ("MX17 pitch*2",0.5,28,0.3,1.56,0.35,0.0),
 ("MX17 no diff",0.5,28,0.3,0.78,0.0,1e-6),
 ("ALICE row10 tl1,1.0-1.6m",1000,1600,1.0,10,0.22,0),
 ("ALICE row10 tl.2",1000,1600,5.0,10,0.22,0),
 ("ALICE IROC 7.5 shortdr",100,250,1.0,7.5,0.22,0),
 ("STAR outer20 P10",1000,1600,1.0,20,0.23,0),
 ("T2K pad11 HATPC",400,900,1.0,11.2,0.31,0),
 ("ILD pad6 3.5T",1000,1600,1.0,6,0.03,0),
 ("sPHENIX row12.5 ",300,800,1.0,12.5,0.1,0),
]:
    a,n=run(z0,z1,sl,p,D(d),s0); b,_=run(z0,z1,sl,p,D(d),s0,trim=0.2)
    sg=np.sqrt(D(d)*z1+s0)
    print(f"{name:26s}{sl:5.1f} {sg/(sl*(z1-z0)):6.3f} {n:4d} | {a[0]:.3f} {a[1]:.3f} | {b[0]:.3f} {b[1]:.3f}")
