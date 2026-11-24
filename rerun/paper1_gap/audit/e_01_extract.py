"""e_01_extract.py -- build the per-sample centroid tables (both runs, both views) -> $WORK/audit/e/cen_<run>_<view>.parquet
usage: e_01_extract.py [tag] [nsig] [r0] [r1]   (tag '' = default; zpred from e/zk_global.json if present)"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import e_lib as L
from concurrent.futures import ProcessPoolExecutor
tag = sys.argv[1] if len(sys.argv) > 1 else ''
nsig = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
r0 = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
r1 = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
zp = L.zpred_default
zf = f'{L.OUT}/zk_global.json'
if os.path.exists(zf) and not tag.startswith('first'):
    Z = json.load(open(zf)); zt, zz = np.array(Z['t']), np.array(Z['z'])
    zp = lambda t: np.clip(np.interp(t, zt, zz), 0, L.GAP)
def job(a):
    run, view = a
    ev, tab, st = L.load_events(run)
    df = L.extract(ev, tab, view, L.dead_mask(run)[view], zpred=zp, nsig=nsig, r0=r0, r1=r1)
    df['run'] = run; df['view'] = view
    df.to_parquet(f'{L.OUT}/cen{tag}_{run}_{view}.parquet')
    return run, view, len(df), st
if __name__ == '__main__':
    with ProcessPoolExecutor(4) as p:
        for r in p.map(job, [(r, v) for r in ('run1', 'run2') for v in 'xy']):
            print(r, flush=True)
