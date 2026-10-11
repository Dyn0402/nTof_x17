#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_steps.py -- MX17_Full_Geant HitTree -> the prompt DriftGas steps of
the chosen arms, for digitise.py.  Runs on lxplus (LCG_106 python, uproot).

Keeps (event, arm) groups with >= 5 steps; time < 1e8 ns (RadioactiveDecay
is on and 28Al decays sit at 1e9-1e13 ns).  edep in eV, positions in the arm's
local (u, v, w) mm, as in the HitTree.

    python3 extract_steps.py <in.root> <out.parquet> [--arms 2 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import uproot

BR = ['eventID', 'trackID', 'armID', 'detType', 'u', 'v', 'w', 'edep', 'ke', 'time']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--arms', type=int, nargs='+', default=[2, 3])
    a = ap.parse_args()
    t = uproot.open(a.inp)['HitTree']
    out = []
    for arr in t.iterate(BR, step_size=3_000_000, library='np'):
        dt = arr['detType'].astype(str)
        m = (dt == 'DriftGas') & (arr['time'] < 1e8) & np.isin(arr['armID'], a.arms) & (arr['edep'] > 0)
        out.append(pd.DataFrame({k: arr[k][m] for k in BR if k != 'detType'}))
    d = pd.concat(out, ignore_index=True)
    n = d.groupby(['eventID', 'armID']).edep.transform('size')
    d = d[n >= 5].copy()
    d['file'] = os.path.basename(a.inp).replace('_t0.root', '')
    for c in ('u', 'v', 'w', 'edep', 'ke', 'time'):
        d[c] = d[c].astype(np.float32)
    d.to_parquet(a.out, index=False)
    print(f'{a.inp}: {len(d)} steps, {d.groupby(["eventID", "armID"]).ngroups} (event, arm)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
