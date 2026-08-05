#!/usr/bin/env python3
"""
Per-event FEU-presence table for the lead-shielding comparison runs.

Same logic as run67_scan/feu_presence.py (see its docstring for the
live_* vs readout_* distinction — live is an OBSERVABLE (post-flash
blindness), readout_* is the honest efficiency DENOMINATOR cut), extended
to span several runs: the table carries a `run` column and attach() joins
on (run, subrun, eventId).

Run: .venv/bin/python ntof_july_analysis/leadshield_compare/feu_presence.py [--force]
Output -> <CACHE_DIR>/_feu_presence.parquet
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
import lib as L  # noqa: E402
from ntof_tracking.reco import io  # noqa: E402

DET_FEUS = {'A': (3, 4), 'B': (5, 6), 'C': (7, 8), 'D': (1, 2)}
OUT_PATH = os.path.join(L.CACHE_DIR, '_feu_presence.parquet')
FEU_COLS = [f'feu{k}' for k in range(1, 9)]


def build(force=False, runs=None):
    """Presence table for every sub-run that has a RECO CACHE (see run67_scan
    version for why; rebuild with --force after processing more sub-runs)."""
    if os.path.exists(OUT_PATH) and not force:
        return pd.read_parquet(OUT_PATH)
    runs = runs or L.RUNS
    frames = []
    subs = [(run, d['name']) for run in runs for d in L.list_subruns(run)
            if os.path.exists(L._cache_paths(run, d['name'])[0])]
    print(f'feu_presence: {len(subs)} cached sub-run(s) to scan', flush=True)
    for i, (run, sub) in enumerate(subs):
        fs = sorted(glob.glob(os.path.join(
            io.BASE_PATH, run, sub, 'combined_hits_root', '*_datrun_*.root')))
        good = []
        for f in fs:
            try:
                with uproot.open(f) as h:
                    if 'hits' in h:
                        good.append(f)
            except Exception:
                continue
        if not good:
            continue
        df = uproot.concatenate([f'{f}:hits' for f in good], ['eventId', 'feu'],
                                library='pd').drop_duplicates(['eventId', 'feu'])
        piv = (df.assign(v=True)
               .pivot_table(index='eventId', columns='feu', values='v',
                            fill_value=False)
               .reindex(columns=range(1, 9), fill_value=False)
               .astype(bool))
        piv.columns = FEU_COLS
        piv = piv.reset_index()
        piv['run'] = run
        piv['subrun'] = sub
        for det, (fx, fy) in DET_FEUS.items():
            piv[f'live_{det}'] = piv[f'feu{fx}'] & piv[f'feu{fy}']

        # FILE-level readout flags: which FEUs were actually decoded for each
        # file-group; an event inherits its group's flags (honest denominator).
        dec_by_fnum = {}
        for f in glob.glob(os.path.join(io.BASE_PATH, run, sub,
                                        'decoded_root', '*_datrun_*.root')):
            m = re.search(r'_(\d{3})_(\d{2})\.root$', os.path.basename(f))
            if m and '_pedestals_' not in os.path.basename(f):
                dec_by_fnum.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        ev_fnum = {}
        for f in good:
            m = re.search(r'_(\d{3})_feu-combined', os.path.basename(f))
            if not m:
                continue
            fn = int(m.group(1))
            try:
                ids = uproot.open(f)['hits'].arrays(['eventId'],
                                                    library='np')['eventId']
            except Exception:
                continue
            for e in np.unique(ids):
                ev_fnum[int(e)] = fn
        piv['file_num'] = piv['eventId'].map(ev_fnum)
        for det, (fx, fy) in DET_FEUS.items():
            piv[f'readout_{det}'] = piv['file_num'].map(
                lambda fn: (fn in dec_by_fnum
                            and fx in dec_by_fnum[fn] and fy in dec_by_fnum[fn])
                if pd.notna(fn) else False).astype(bool)
        frames.append(piv)
        print(f'[{i + 1}/{len(subs)}] {run}/{sub}: {len(piv)} ev, '
              f'all8={piv[FEU_COLS].all(axis=1).mean():.3f}', flush=True)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_parquet(OUT_PATH)
    print('wrote', OUT_PATH, len(out), 'rows')
    return out


def attach(ev):
    """Left-join per-event FEU flags onto an events table (see run67_scan
    feu_presence.attach docstring; join key here includes `run`)."""
    pres = pd.read_parquet(OUT_PATH)
    cols = (['run', 'subrun', 'eventId'] + [f'live_{d}' for d in 'ABCD']
            + [f'readout_{d}' for d in 'ABCD'])
    out = ev.merge(pres[cols], on=['run', 'subrun', 'eventId'], how='left')
    for d in 'ABCD':
        out[f'live_{d}'] = out[f'live_{d}'].fillna(False).astype(bool)
        out[f'readout_{d}'] = out[f'readout_{d}'].fillna(False).astype(bool)
    return out


if __name__ == '__main__':
    build(force='--force' in sys.argv)
