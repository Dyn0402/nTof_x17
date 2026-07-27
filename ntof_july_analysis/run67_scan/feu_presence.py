#!/usr/bin/env python3
"""
Per-event FEU-presence table for run_64 — the fix for its readout dropouts.

run_64 (unlike run_58, which is clean in all 63 sub-runs) suffers FEU dropouts:
in many sub-runs only a subset of the 8 FEUs participates, typically for the
first ~90% of the sub-run with the rest joining late. Example:
sngPS_dr700_r525_018 ran with ONLY FEU 6 until its last decile.

An absent FEU means the detector was never read out, so those events must leave
that detector's efficiency DENOMINATOR — otherwise the dropout is scored as
tracking inefficiency, and because run_64 was taken in a coarse-first interleave
(drift 700/500/300 first, then 600/400, then 200) the surviving clean cells are
exactly the early drift-700/500 high-resist ones. That is what produced the
spurious "optimum at the grid corner" in the first pass.

In RAW full-readout mode a participating FEU always yields ~550 hits/event, so
presence is unambiguous: present <=> the FEU appears in the event at all.

Reads only the (eventId, feu) branches, so it is cheap (no reco) and joins onto
the existing reco cache by (subrun, eventId).

Run: .venv/bin/python ntof_july_analysis/run64_scan/feu_presence.py [--force]
Output -> <CACHE_DIR>/run_64/_feu_presence.parquet
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan_lib as L  # noqa: E402
from ntof_tracking.reco import io  # noqa: E402

# detector -> (x-plane FEU, y-plane FEU), from io.build_channel_lut(run_64 cfg)
DET_FEUS = {'A': (3, 4), 'B': (5, 6), 'C': (7, 8), 'D': (1, 2)}
OUT_PATH = os.path.join(L.CACHE_DIR, L.RUN, '_feu_presence.parquet')
FEU_COLS = [f'feu{k}' for k in range(1, 9)]


def build(force=False):
    """Build the presence table for every sub-run that has a RECO CACHE.

    Restricted to cached sub-runs for two reasons: it avoids opening ~37 GB of
    combined_hits for sub-runs that are not yet reco'd (this box shares I/O with
    a live DAQ), and it keeps the table aligned with what the analysis actually
    loads. NOTE the corollary: as more sub-runs are processed this table becomes
    STALE, and a stale table is not benign — attach() left-joins on
    (subrun, eventId) and fills misses with readout_*=False, which would silently
    drop those events from every efficiency denominator. Always rebuild with
    force=True after processing more sub-runs (run_all.py --force-feu).
    """
    if os.path.exists(OUT_PATH) and not force:
        return pd.read_parquet(OUT_PATH)
    frames = []
    subs = [d for d in L.list_subruns()
            if os.path.exists(L._cache_paths(L.RUN, d['name'])[0])]
    print(f'feu_presence: {len(subs)} cached sub-run(s) to scan', flush=True)
    for i, d in enumerate(subs):
        sub = d['name']
        fs = sorted(glob.glob(os.path.join(
            io.BASE_PATH, L.RUN, sub, 'combined_hits_root', '*_datrun_*.root')))
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
        piv['subrun'] = sub
        for det, (fx, fy) in DET_FEUS.items():
            piv[f'live_{det}'] = piv[f'feu{fx}'] & piv[f'feu{fy}']

        # FILE-level readout flags: which FEUs were actually decoded for each
        # file-group of this sub-run. An event inherits the flags of its group.
        # This is the honest denominator cut (see attach()).
        dec_by_fnum = {}
        for f in glob.glob(os.path.join(io.BASE_PATH, L.RUN, sub,
                                        'decoded_root', '*_datrun_*.root')):
            m = re.search(r'_(\d{3})_(\d{2})\.root$', os.path.basename(f))
            if m and '_pedestals_' not in os.path.basename(f):
                dec_by_fnum.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        # map eventId -> file_num via each combined file's own event range
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
        print(f'[{i + 1}/{len(subs)}] {sub}: {len(piv)} ev, '
              f'all8={piv[FEU_COLS].all(axis=1).mean():.3f}', flush=True)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_parquet(OUT_PATH)
    print('wrote', OUT_PATH, len(out), 'rows')
    return out


def attach(ev):
    """Left-join per-event FEU flags onto an events table.

    Adds TWO different things — do not confuse them:

    ``live_{A..D}``  — this detector produced >=1 hit in this event.
      **NOT a readout flag.** Verified 2026-07-21 on the fully-recovered run_64
      (113/113 groups decoded, all 8 FEUs, equal counts, so every FEU demonstrably
      read out every event): live is ~1.00 in the late window but drops to
      A 0.84 / C 0.78 / D 0.42 in the ~4 ms post-flash window, and those events
      carry a median of 3 raw hits versus 653 for live ones. That is the
      detector being BLIND from the gamma flash — i.e. exactly the post-flash
      inefficiency this analysis exists to measure. Cutting on it removes the
      signal from the denominator and inflates the early-window efficiency.
      Use it as an OBSERVABLE (blind fraction), never as a denominator cut.

    ``readout_{A..D}`` — both of this detector's FEUs were successfully decoded
      for this event's file-group. THIS is the correct denominator cut: it
      removes only events the processing chain never delivered. On recovered
      data it is True everywhere, so it is a no-op; on pre-fix data it excises
      the missing/truncated decodes (see the module docstring).
    """
    pres = pd.read_parquet(OUT_PATH)
    cols = ['subrun', 'eventId'] + [f'live_{d}' for d in 'ABCD']
    have_readout = all(f'readout_{d}' in pres.columns for d in 'ABCD')
    if have_readout:
        cols += [f'readout_{d}' for d in 'ABCD']
    out = ev.merge(pres[cols], on=['subrun', 'eventId'], how='left')
    for d in 'ABCD':
        out[f'live_{d}'] = out[f'live_{d}'].fillna(False).astype(bool)
        if have_readout:
            out[f'readout_{d}'] = out[f'readout_{d}'].fillna(False).astype(bool)
        else:
            # presence table predates the readout flags: fall back to "an event
            # that appears in the table at all was read out".
            out[f'readout_{d}'] = out['eventId'].notna()
    return out


if __name__ == '__main__':
    build(force='--force' in sys.argv)
