import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq
C='cache'
cols=['run','subrun','tag','event_id','arm','track_id','t_since_flash_ns','gated','gated_precut','gated_uncensored','censored_window','d_faint',
'x_local','y_local','tanx','tany','tanx_nomap','tany_nomap','p0_x','p0_y','p0_z','d_x','d_y','d_z','d_x_nomap','d_y_nomap','d_z_nomap',
'x_n_strips','y_n_strips','x_q_sum','y_q_sum','chi2dof_x','chi2dof_y','x_cens_frac','y_cens_frac','x_n_flagged','y_n_flagged','wall_D','wall_B','dca_source_mm','v_t','x_t0','y_t0','x_p0','x_w']
out=[]
for f in sorted(glob.glob(C+'/*.parquet')):
    t=pd.read_parquet(f,columns=cols)
    out.append(t)
d=pd.concat(out,ignore_index=True)
for c in ['gated','gated_precut','gated_uncensored','censored_window','d_faint','wall_D','wall_B']:
    d[c]=d[c].astype('boolean').fillna(False).astype(bool)
d['rn']=d.run.str[4:].astype(int)
d.to_parquet('db_all.parquet',index=False); print(len(d), d.arm.value_counts().to_dict())
