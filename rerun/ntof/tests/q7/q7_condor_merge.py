#!/usr/bin/env python3
"""q7_condor_merge.py -- condor ntof_reco.py outputs (one configuration per bundle copy) -> q7_real.py's table layout.

    q7_condor_merge.py --src $Q7_DATA/condor/q7_beam/out --run 145 --subruns stat090_0000 --arms A C --stem r145full
    q7_condor_merge.py --src $Q7_DATA/condor/q7_line/out --run 149 --subruns cosbounce_cos_0000 ... --stem r149full

src layout (make_ntof_jobs.py): <src>/<cfg>/run_<run>/<subrun>/<arm>/run_<run>_<subrun>_<tag>_<arm>.{parquet,candidates.parquet}
Writes $Q7_DATA/real/<stem>_<arm>.{events,candidates}.parquet (one subrun) or <stem>s<i>_<arm>.* (several), with 'cfg'
and 'acq_tag' columns, and reports missing acquisitions.
"""
import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--subruns', nargs='+', required=True)
    ap.add_argument('--arms', nargs='+', default=['A', 'C'])
    ap.add_argument('--stem', required=True)
    a = ap.parse_args()
    out = os.path.join(L.DATA, 'real')
    os.makedirs(out, exist_ok=True)
    cfgs = sorted(os.listdir(a.src))
    for i, sub in enumerate(a.subruns):
        for arm in a.arms:
            E, C, miss = [], [], []
            for cf in cfgs:
                d = os.path.join(a.src, cf, f'run_{a.run}', sub, arm)
                fs = sorted(glob.glob(os.path.join(d, '*.candidates.parquet')))
                if not fs:
                    miss.append(cf)
                for f in fs:
                    c = pd.read_parquet(f)
                    e = pd.read_parquet(f.replace('.candidates.parquet', '.parquet'))
                    C.append(c.assign(cfg=cf))
                    E.append(e.assign(cfg=cf))
            stem = f'{a.stem}_{arm}' if len(a.subruns) == 1 else f'{a.stem}s{i}_{arm}'
            if E:
                pd.concat(E, ignore_index=True).to_parquet(os.path.join(out, f'{stem}.events.parquet'), index=False)
                pd.concat(C, ignore_index=True).to_parquet(os.path.join(out, f'{stem}.candidates.parquet'), index=False)
            n = pd.concat(E).groupby('cfg').size() if E else {}
            print(stem, 'events per cfg:', dict(n) if len(n) else n, 'missing cfgs:', miss)


if __name__ == '__main__':
    main()
