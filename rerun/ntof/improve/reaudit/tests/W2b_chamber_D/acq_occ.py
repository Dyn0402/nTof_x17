import glob, json, os, numpy as np, pandas as pd
A='/media/ucla/work/ntof_pass1/channel_layer/acq'
rows=[]
for f in sorted(glob.glob(A+'/run_*.npz')):
    j=json.load(open(f[:-4]+'.json'))
    z=np.load(f)
    d=dict(key=j['key'],run=int(j['run']),subrun=j['subrun'],tag=j['tag'],rclass=j['rclass'],t=j['t_local'],n=int(z['n_events']))
    for fe in (1,2,5,6):
        d[f'occ_{fe}']=z[f'occ_{fe}'].astype(np.int32); d[f'n1_{fe}']=z[f'n1_{fe}']; d[f'f1_{fe}']=z[f'f1_{fe}']
        d[f'on300_{fe}']=z[f'on300_{fe}']
        d[f'hot_{fe}']=j['feu'].get(str(fe),{}).get('hot',[]); d[f'flank_dead_{fe}']=j['feu'].get(str(fe),{}).get('flank_dead',[])
        d[f'silent_{fe}']=j['feu'].get(str(fe),{}).get('silent',[])
    rows.append(d)
pd.to_pickle(rows,'acq_occ.pkl'); print(len(rows))
