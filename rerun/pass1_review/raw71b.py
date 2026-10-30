import numpy as np, pyarrow.parquet as pq, pandas as pd
R='/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps/'
pd.set_option('display.width',250)
for k in ['run71_raw__raw700','run70_flat_drift__d700']:
    t=pq.read_table(R+k+'/events.parquet').to_pandas()
    print('==',k)
    for v in 'xy':
        ok=t[v+'_ok'].astype(bool); tan=t[v+'_tan_theta'].abs()
        grp={'notok':~ok,'wild':ok&(tan>3),'mid 0.5-3':ok&(tan>0.5)&(tan<=3),'good<0.5':ok&(tan<=0.5)}
        rows=[]
        for g,s in grp.items():
            if s.sum()==0: continue
            d=t[s]; r=dict(view=v,grp=g,frac=round(s.mean(),3))
            for c in ['q_sum','n_strips','n_seed','n_dropped','n_candidates','n_flagged_strips','n_band_strips','t0','w','q_uend']:
                col=f'{v}_{c}'
                if col in d: r[c]=round(float(np.nanmedian(d[col])),2)
            r['chi2dof']=round(float(np.nanmedian(d[v+'_chi2']/d[v+'_dof'].clip(lower=1))),1)
            r['spark']=round(d.spark.mean(),3); r['n_hits']=float(np.median(d.n_hits))
            rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))
