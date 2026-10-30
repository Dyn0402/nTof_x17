import glob, json, numpy as np, pyarrow.parquet as pq, pandas as pd
R='/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps/'
rows=[]
def s68(a):
    a=a[np.isfinite(a)]; 
    if len(a)<20: return np.nan
    m=np.median(a); return np.percentile(np.abs(a-m),68.27)
for d in sorted(glob.glob(R+'*/')):
    k=d.rstrip('/').split('/')[-1]
    m=json.load(open(d+'events.meta.json')); t=pq.read_table(d+'events.parquet').to_pandas()
    r=dict(key=k,n=len(t),code=m.get('code'),v=m.get('v_drift'),K=m.get('n_depth_bins'),mount=m['registry_row'].get('mount'),raw=m['registry_row'].get('raw'))
    for v in 'xy':
        ok=t[v+'_ok'].astype(bool)&t[v+'_quality_ok'].astype(bool)
        th=np.degrees(np.arctan(t[v+'_tan_theta'][ok].clip(-3,3)))
        r['fok_'+v]=ok.mean(); r['th_med_'+v]=np.median(th); r['th_s68_'+v]=s68(th.values)
        ref=t['ref_'+v] if 'ref_'+v in t else None
        if ref is not None:
            res=(t[v+'_p0']-ref)[ok&np.isfinite(ref)]
            r['res_med_'+v]=np.median(res); r['res_s68_'+v]=s68(res.values)
    r['tan_mount']=np.nanmedian(t['tan_mount']) if 'tan_mount' in t else np.nan
    rows.append(r)
d=pd.DataFrame(rows)
pd.set_option('display.width',300); pd.set_option('display.max_rows',100); pd.set_option('display.max_columns',30)
print(d.round(3).to_string(index=False))
