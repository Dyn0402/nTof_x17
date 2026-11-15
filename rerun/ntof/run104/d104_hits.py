#!/usr/bin/env python3
"""d104_hits.py -- model-free check of D's latency step with the rr1_cm0 hits (timing QA only, no geometry).  lxplus.

Per acquisition: for each event and FEU, the earliest time_of_max among hits with significance >= 8 that are not
left-truncated; then the median over events per FEU.  D's trigger latency stepped ~36 ns between run_103 and run_105
(free-t0 fits, plan §6 11-06), A and C by only 5-10 ns.  So the D - A and D - C differences of those medians should step
by ~36 ns between the L1 beam runs (100, 102) and the L2 ones (106, 108); where run_104's sub-runs sit says which
latency each was taken with.  FEUs: D x 1 / y 2, A 3 / 4, C 7 / 8.

    source <LCG_110 view>; python3 d104_hits.py out.csv [cosmic]   # beam runs 100-108 (default) or cosmic 97-111
"""
import glob, sys
import numpy as np, pandas as pd, uproot

RUNS = '/eos/experiment/ntof/data/x17/july_beam/runs'
FEUS = {'Dx': 1, 'Dy': 2, 'Ax': 3, 'Ay': 4, 'Cx': 7, 'Cy': 8}


BEAM = ((100, 8, 1), (102, 4, 1), (104, 99, 2), (106, 6, 1), (108, 4, 1))
COSMIC = ((97, 99, 3), (99, 99, 3), (101, 99, 3), (103, 12, 2), (105, 99, 2), (107, 99, 3), (109, 99, 3), (111, 99, 2))


def files(spec):
    out = []
    for run, nsub, nper in spec:
        subs = sorted(d for d in glob.glob(f'{RUNS}/run_{run}/*') if '/run_config' not in d and not d.endswith('.json'))
        step = max(1, len(subs) // nsub) if nsub < 99 else 1
        for s in subs[::step][:nsub]:
            fs = sorted(glob.glob(f'{s}/combined_hits_root_rr1_cm0/*_feu-combined_hits.root'))
            idx = np.unique(np.linspace(0, len(fs) - 1, nper).round().astype(int)) if fs else []
            out += [(run, s.split('/')[-1], fs[i]) for i in idx]
    return out


def one(run, sub, f):
    t = uproot.open(f)['hits']
    a = t.arrays(['eventId', 'feu', 'time_of_max', 'significance', 'trunc_left'], library='pd')
    a = a[(a.significance >= 8) & (~a.trunc_left)]
    m = a.groupby(['feu', 'eventId']).time_of_max.min().reset_index()
    r = dict(run=run, subrun=sub, tag=f.split('datrun_')[1][:16], n_ev=int(a.eventId.nunique()))
    for k, feu in FEUS.items():
        v = m[m.feu == feu].time_of_max.to_numpy()
        r[k] = float(np.median(v)) if len(v) > 50 else np.nan
        r[f'n_{k}'] = int(len(v))
    for v in 'xy':
        r[f'DmA_{v}'] = r[f'D{v}'] - r[f'A{v}']
        r[f'DmC_{v}'] = r[f'D{v}'] - r[f'C{v}']
    print(r, flush=True)
    return r


rows = [one(*x) for x in files(COSMIC if 'cosmic' in sys.argv[2:] else BEAM)]
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
