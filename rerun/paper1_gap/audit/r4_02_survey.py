import sys; sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import r4_lib as L, numpy as np, glob, os
rels = sorted({os.path.dirname(p)[len(L.EOS)+1:] for p in glob.glob(L.EOS+'/*/*/events.parquet')} | {L.GOLD1,L.GOLD2,L.R625})
import pandas as pd
for rel in rels:
    if 'det4' in rel: continue
    d = L.load_events(rel); 
    tot=len(pd.read_parquet(f'{L.EOS}/{rel}/events.parquet'))
    print(f'{rel.split("/")[-1][:42]:44s} tot {tot:6d} N {len(d):5d} meanq {d.q.mean():7.1f} medq {d.q.median():7.1f} sat {(d.x_n_sat_strips>0).mean():.2f} atx {d.atx.mean():.2f}')
