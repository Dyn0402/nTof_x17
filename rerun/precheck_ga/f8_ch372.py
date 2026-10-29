import uproot, glob, numpy as np
for sub in ('f000','f010'):
    fs=sorted(glob.glob('/eos/experiment/ntof/data/x17/p2_sps_july/runs/run_71/cfg_gain4.5_peaktime50/decoded_root_fa80c06/*_03.root'))[int(sub[1:]):int(sub[1:])+1]
    if not fs: print(sub,'none'); continue
    a=uproot.open(fs[0])['nt'].arrays(['amplitude'],library='np',entry_stop=1500)['amplitude']
    ns=int(np.bincount([len(x)//512 for x in a]).argmax())
    W=np.stack([x.reshape(ns,512).astype(np.float32) for x in a if len(x)==ns*512])
    W-=np.median(W,axis=(0,1))
    # per-event pre-level removed, then rms over samples per channel, median over events
    W-=W[:,:8,:].mean(1,keepdims=True)
    rms=np.median(W.std(1),axis=0)
    swing=np.mean(np.abs(W).max(1)>300,axis=0)      # fraction of events with a >300 ADC excursion
    for c in (370,371,372,373,374,508,509,510,511):
        print(f'{sub} ch{c}: median rms {rms[c]:7.1f}  frac>300 {swing[c]:.3f}')
    y=np.arange(256,512); print(sub,'Y view median rms',np.median(rms[y]),'frac>300 median',np.median(swing[y]), 'files',len(fs),'events',len(W))
