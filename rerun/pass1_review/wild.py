import glob, numpy as np, pyarrow.parquet as pq
R='/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/'
tot={'rows':0,'ev':0}
rows=[]
for f in sorted(glob.glob(R+'mx17_*/*/mx17_*/wft/events.parquet'))+sorted(glob.glob(R+'sps/*/events.parquet')):
    cols=['x_ok','y_ok','x_tan_theta','y_tan_theta','x_quality_ok','y_quality_ok','x_p0','y_p0']
    t=pq.read_table(f).to_pandas()
    r=[f.replace(R,'')[:90], len(t)]
    for v in 'xy':
        ok=t[v+'_ok'].astype(bool); tan=t[v+'_tan_theta'].abs()
        w=ok&((tan>3)|~np.isfinite(tan))
        wq=w&t[v+'_quality_ok'].astype(bool)
        hb=(t[v+'_bow_h'].abs()>10).sum() if v+'_bow_h' in t else -1
        r+= [int(w.sum()), int(wq.sum()), int(hb), int((ok&(t[v+'_p0'].abs()>500)).sum())]
    rows.append(r)
import pandas as pd
d=pd.DataFrame(rows,columns=['row','n','x_wild','x_wild_q','x_bowh10','x_p0big','y_wild','y_wild_q','y_bowh10','y_p0big'])
d['sps']=d.row.str.startswith('sps')
print(d.groupby('sps')[['n','x_wild','x_wild_q','x_bowh10','x_p0big','y_wild','y_wild_q','y_bowh10','y_p0big']].sum())
print('rows with any wild:', d.groupby('sps').apply(lambda g:((g.x_wild+g.y_wild)>0).sum()))
print(d[(d.x_p0big+d.y_p0big)>0].to_string())
