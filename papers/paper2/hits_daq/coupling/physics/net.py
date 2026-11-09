import numpy as np, itertools
dt=0.5e-9; n=2**14; t=np.arange(n)*dt
w=2*np.pi*np.fft.rfftfreq(n,dt); s=1j*w; s[0]=1j*1e-3*w[1]
def current(fe=0.1,Tion=150e-9):
    i=np.zeros(n); i[(t<4e-9)]=fe/4e-9; m=t<Tion; i[m]+=(1-fe)/Tion; return i
I0=np.fft.rfft(current())*dt
def sim(Cis,Cnode,Rin,Cin,tp,N=31):
    Cg=max(Cnode-2*Cis,0); Z=Rin+1/(s*Cin)
    # shaper CR-RC^2 with peaking tp: tau=tp/2
    tau=tp/2; H=(s*tau)/(1+s*tau)**3
    Ys=[]
    for w_i in range(len(s)):
        y=s[w_i]*Cis
        A=np.zeros((N,N),complex)
        for j in range(N):
            A[j,j]=1/Z[w_i]+s[w_i]*Cg+2*y
            if j>0:A[j,j-1]=-y
            if j<N-1:A[j,j+1]=-y
        b=np.zeros(N,complex);b[N//2]=I0[w_i]
        V=np.linalg.solve(A,b)
        Ys.append(V/Z[w_i])
    Ik=np.array(Ys).T
    # CSA integrates current; output ~ charge into feedback; decay 5 us
    Tf=5e-6; out=Ik*H/(s+1/Tf)
    y=np.fft.irfft(out,n=n,axis=1)/dt
    return y[N//2],y[N//2+1],y[N//2+2]
def report(Cis,Cnode,Rin,Cin,tp):
    a,b,c=sim(Cis,Cnode,Rin,Cin,tp)
    ip=a.argmax(); 
    pk=b.max(); neg=b.min(); ipb=np.abs(b).argmax()
    # integral: late-time charge ratio
    return dict(Cis=Cis,Cnode=Cnode,Rin=Rin,Cin=Cin,tp=tp*1e9,kpos=pk/a.max(),kneg=neg/a.max(),ts=(ipb-ip)*dt*1e9,k_at_peak=b[ip]/a.max(),k2=c.max()/a.max())
for p in [(6.8,130e-12,300,1000e-12,120e-9),(6.8,130e-12,300,1000e-12,230e-9),(6.8,130e-12,100,1000e-12,230e-9),(6.8,130e-12,1000,1000e-12,230e-9),(6.8,130e-12,300,200e-12,230e-9),(6.8,130e-12,300,4000e-12,230e-9),(9.5,100e-12,1000,200e-12,230e-9),(4.6,200e-12,100,4000e-12,230e-9)]:
    p=(p[0]*1e-12,)+p[1:]
    r=report(*p); print({k:(round(v*1e12,1) if k in('Cis','Cnode','Cin') else round(v,4)) for k,v in r.items()})
# moire mean of ESL geometric share, projected overlap model
d=np.linspace(0,.39,2001)
def ov(a,b,lo,hi): return np.clip(np.minimum(b,hi)-np.maximum(a,lo),0,None)
m=ov(d-.275,d+.275,-.34,.34); nb=ov(d-.275,d+.275,.44,1.12); r=nb/m
print('frac phase nb>0',(nb>0).mean(),'mean neighbour/main (near side)',r.mean(),'max',r.max(),'mean nb/(all)',(nb/(m+nb)).mean())
print('moire period mm',0.78*0.8/0.02)
