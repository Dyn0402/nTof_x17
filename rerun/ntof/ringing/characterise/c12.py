import numpy as np, sys
sys.path.insert(0,'.')
from rerun.ntof.ringing.deflect import deflect, plain_cm
bins=[1,2,3,5,8,12,16,20,30,80]
for f in (3,4,7,8):
    z=np.load(f'/media/ucla/work/claude-jobs/bcad1a92/ring_feu{f}.npz'); A=z['A'].astype(np.float32); ms=z['ms']; nz=z['noise']
    N,ns,_=A.shape
    ring=np.zeros(N); per=[]
    for i in range(N):
        P=plain_cm(A[i]); G=deflect(A[i],nz); D=P-G       # the part removed beyond the plain median
        ring[i]=np.sqrt(np.mean(np.sort((D**2).mean(0))[-32:]))   # rms over the 32 most affected channels
        if 3<ms[i]<8:
            c=np.median(A[i].reshape(ns,8,64),axis=2).mean(1); c=c-c.mean()
            sp=np.abs(np.fft.rfft(c*np.hanning(ns),n=200))
            per.append(np.fft.rfftfreq(200,d=60e-9)[np.argmax(sp[3:])+3])
    print(f'FEU {f}: ringing amplitude (rms of removed part, 32 most affected channels) median by ToF:',
          {f'{lo}-{hi}':round(float(np.median(ring[(ms>=lo)&(ms<hi)])),1) for lo,hi in zip(bins[:-1],bins[1:])},
          ' dominant CM frequency 3-8 ms: median %.2f MHz'%(np.median(per)/1e6))
