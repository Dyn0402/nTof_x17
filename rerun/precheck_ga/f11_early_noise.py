import uproot, glob, numpy as np
B='/eos/experiment/ntof/data/x17/cosmic_bench/june_tests/'
cases=[('mx17_det4_day_6-24-26/long_run',6,(7,8)),
       ('mx17_det6_det7_overnight_6-26-26/resist_det6_530V_det7_505V_drift_700V',4,(6,7,2)),
       ('mx17_det6_det7_overnight_6-26-26/resist_det6_530V_det7_505V_drift_700V',8,(4,)),
       ('mx17_det6_det7_overnight_6-26-26/long_run',8,(4,5)),
       ('mx17_det3_saturday_scan_6-27-26/hv_scan_resist_525V_drift_1000V',8,(7,8)),
       ('mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V',8,(7,1))]
for rs,feu,conns in cases:
    sd=B+rs
    comb=sorted(glob.glob(sd+'/combined_hits_root_rr1_cm0/*_datrun_*.root'))[0]
    stem=comb.split('/')[-1].split('_feu-combined')[0]
    dec=glob.glob(f'{sd}/decoded_root_fa80c06/{stem}_{feu:02d}.root')[0]
    h=uproot.open(comb)['hits'].arrays(['feu','channel','amplitude','significance'],library='np')
    m=(h['feu']==feu)&(h['significance']>0)
    ch=h['channel'][m].astype(int); q=h['amplitude'][m]/h['significance'][m]
    pn=np.array([np.median(q[ch==c]) if np.sum(ch==c)>=20 else np.nan for c in range(512)])
    a=uproot.open(dec)['nt'].arrays(['amplitude'],library='np',entry_stop=300)['amplitude']
    ns=int(np.bincount([len(x)//512 for x in a]).argmax())
    st=np.stack([x.reshape(ns,512).astype(np.float32) for x in a if len(x)==ns*512])
    sub=st-np.median(st,axis=(0,1)); cm=np.median(sub.reshape(len(st),ns,8,64),axis=3); sub-=np.repeat(cm,64,axis=2)
    n_all=1.4826*np.median(np.abs(sub),axis=(0,1)); n_early=1.4826*np.median(np.abs(sub[:,:3,:]),axis=(0,1))
    for k in conns:
        cs=slice(64*(k-1),64*k)
        print(f'{rs[-40:]:40s} feu{feu} conn{k}: ped/all {np.nanmedian(pn[cs]/n_all[cs]):.2f}  ped/early {np.nanmedian(pn[cs]/n_early[cs]):.2f}  early/all {np.nanmedian(n_early[cs]/n_all[cs]):.2f}', flush=True)
