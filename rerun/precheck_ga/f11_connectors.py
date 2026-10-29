import uproot, glob, numpy as np
sd='/eos/experiment/ntof/data/x17/cosmic_bench/june_tests/mx17_det4_day_6-24-26/long_run'
hf=sorted(glob.glob(sd+'/hits_root_rr1_cm0/*_datrun_*_06_hits.root'))[:3]
df=sorted(glob.glob(sd+'/decoded_root_fa80c06/*_datrun_*_06.root'))[:1]
E,C,A,S=[],[],[],[]
for i,f in enumerate(hf):
    a=uproot.open(f)['hits'].arrays(['eventId','channel','amplitude','significance'],library='np')
    E.append(a['eventId']+i*10**7); C.append(a['channel']); A.append(a['amplitude']); S.append(a['significance'])
E,C,A,S=map(np.concatenate,(E,C,A,S)); E=E.astype(np.int64); C=C.astype(np.int64); nev=len(np.unique(E))
ok=(A>0)&(S>0)
noise_ped=np.full(512,np.nan)
for c in range(512):
    m=ok&(C==c)
    if m.sum()>20: noise_ped[c]=np.median(A[m]/S[m])
# true noise from the data run (pedestal + plain CM, as wft.io)
amp=uproot.open(df[0])['nt'].arrays(['amplitude'],library='np',entry_stop=600)['amplitude']
ns=int(np.bincount([len(x)//512 for x in amp]).argmax())
st=np.stack([x.reshape(ns,512).astype(np.float32) for x in amp if len(x)==ns*512])
ped=np.median(st,axis=(0,1)); sub=st-ped
cm=np.median(sub.reshape(len(st),ns,8,64),axis=3); sub-=np.repeat(cm,64,axis=2)
noise_dat=1.4826*np.median(np.abs(sub),axis=(0,1))
print('events',nev)
print('conn  noise_dat  noise_ped  ped/dat  hits/ev  near-thr(sig<8)/ev  isolated/ev')
for k in range(8):
    ch=np.arange(64*k,64*k+64); m=np.isin(C,ch)
    r=np.nanmedian(noise_ped[ch]/noise_dat[ch])
    # isolated: no other hit within +-2 channels in the same event (crude, channel space)
    key=E[m]*1000+C[m]; s=set(key.tolist())
    iso=sum(1 for e,c in zip(E[m],C[m]) if not any((e*1000+c+d) in s for d in (-2,-1,1,2)))
    print(f'{k+1:4d}  {np.nanmedian(noise_dat[ch]):8.2f}  {np.nanmedian(noise_ped[ch]):8.2f}  {r:6.2f}  {m.sum()/nev:7.3f}  {np.sum(m&(S<8))/nev:9.4f}  {iso/nev:9.4f}')
