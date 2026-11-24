"""Cache of D and B tracks from stage3_r2_final: all tracks of beam-off tables, t>=12 ms tracks of production tables,
plus per-subrun counts of D/B tracks (all times) in region bins."""
import glob, os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
V='/media/ucla/work/ntof_pass1/stage3_r2_final'
OUT='/media/ucla/work/claude-jobs/eb8200b3/W2b_chamber_D/cache'
COLS=['run','subrun','tag','event_id','arm','track_id','t_since_flash_ns','gated','gated_precut','gated_uncensored',
 'censored_window','d_faint','low_charge','t0_railed','x_local','y_local','tanx','tany','tanx_nomap','tany_nomap',
 'tan_raw_x','tan_raw_y','p0_x','p0_y','p0_z','d_x','d_y','d_z','d_x_nomap','d_y_nomap','d_z_nomap','x_n_strips','y_n_strips','x_q_sum','y_q_sum',
 'chi2dof_x','chi2dof_y','x_cens_frac','y_cens_frac','x_n_flagged','y_n_flagged','wall_D','wall_B','dca_source_mm','v_t',
 'x_t0','y_t0','x_w','y_w','x_p0','y_p0','x_q_uend','drift_t_end_ns','x_split_child','event_class','n_cand_x','x_rank']
def one(f):
    k=os.path.basename(f)[11:-8]; o=f'{OUT}/{k}.parquet'
    if os.path.exists(o): return k
    t=pq.read_table(f,columns=COLS,filters=[('arm','in',['B','D'])]).to_pandas()
    tm=t.t_since_flash_ns.to_numpy(float)
    keep=~np.isfinite(tm)|(tm>=12e6)
    t=t[keep]
    for c in t.columns:
        if t[c].dtype==np.float64: t[c]=t[c].astype(np.float32)
    t.to_parquet(o,index=False); return k
if __name__=='__main__':
    fs=sorted(glob.glob(V+'/tracks_run_*.parquet'))
    with ProcessPoolExecutor(3) as ex:
        for i,k in enumerate(ex.map(one,fs)):
            if i%50==0: print(i,k,flush=True)
    print('done')
