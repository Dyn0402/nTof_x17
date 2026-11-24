"""r4_03_fits: shape coefficients of ln q_u90 per sub-run, plus control checks (selection, covariates, q_u50, unsat)."""
import sys, glob, os, json, pickle
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import numpy as np, pandas as pd
import r4_lib as L
import gapmap as GM
rels = sorted({os.path.dirname(p)[len(L.EOS)+1:] for p in glob.glob(L.EOS+'/*/*/events.parquet')} - {x for x in []})
rels = [r for r in rels if 'det4' not in r]
res = {}
for rel in rels:
    d = L.load_events(rel)
    if len(d) < 50: continue
    f = L.fit_shape(d, nboot=200)
    res[rel] = dict(mesh=L.volts(rel)[0], drift=L.volts(rel)[1], N=f['N'], mean_q=f['mean_q'], c=f['c'][:6].tolist(),
                    err=f['err'][:6].tolist(), walk=float(f['c'][6]), tanx=float(f['c'][7]), tany=float(f['c'][8]),
                    keep=f['keep'], boot=f['boot'][:, :6].tolist())
    c, e = f['c'], f['err']
    print(f'{rel.split("/")[-1][:40]:42s} N {f["N"]:5d} gu {c[1]:+.3f} gw {c[2]:+.3f} guu {c[3]:+.3f}+-{e[3]:.3f} gww {c[4]:+.3f}+-{e[4]:.3f} guw {c[5]:+.3f}+-{e[5]:.3f} walk {c[6]:+.2f} tan {c[7]:+.2f},{c[8]:+.2f}', flush=True)
json.dump(res, open(f'{L.OUT}/fits_main.json', 'w'))
