#!/usr/bin/env python3
"""W3: per (file, event, arm, track) the creation vertex (ox, oy, oz) of every track that has a DriftGas step, read
over xrootd from the Geant4 HitTree; plus a small (u, v, w, gx, gy, gz) sample per arm to fit the local->global frame,
and the EventTree capture point.  Read-only on EOS; output in this directory.

    /tmp/w3v/bin/python extract_origin.py <lib dir name> <tag> <job lo> <job hi>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

HERE = Path(__file__).resolve().parent
lib, tag, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
OUT = HERE / f'origin_{tag}'
OUT.mkdir(exist_ok=True)
BASE = 'root://eospublic.cern.ch//eos/experiment/ntof/data/x17/full_sim/' + lib
for j in range(lo, hi):
    out = OUT / f'o{j:03d}.parquet'
    if out.exists():
        continue
    url = f'{BASE}/neutron_bg_job{j:03d}_t0.root'
    try:
        f = uproot.open(url, handler=uproot.source.xrootd.XRootDSource)
    except Exception as e:      # noqa: BLE001
        print('open failed', url, e, flush=True)
        continue
    t = f['HitTree']
    parts, frame = [], []
    nf = {a: 0 for a in range(4)}
    # v2 (from job 24 on): no detType / time / EventTree (the largest reads): every sensitive-volume track,
    # deduplicated; the join on the reduced table's dominant gas track picks the right one (origin is per track)
    for arr in t.iterate(['eventID', 'trackID', 'armID', 'ox', 'oy', 'oz'], step_size='400 MB', library='np'):
        d = pd.DataFrame({k: arr[k] for k in ('eventID', 'trackID', 'armID', 'ox', 'oy', 'oz')})
        parts.append(d.drop_duplicates(['eventID', 'armID', 'trackID']))
    D = pd.concat(parts, ignore_index=True).drop_duplicates(['eventID', 'armID', 'trackID'])
    D['job'] = j
    for c in ('ox', 'oy', 'oz'):
        D[c] = D[c].astype(np.float32)
    D.to_parquet(out, index=False)
    print(j, len(D), flush=True)
