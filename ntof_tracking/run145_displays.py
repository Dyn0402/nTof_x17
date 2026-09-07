#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run145_displays.py — event displays + wall-segment grid for run_145 arm A.

Reuses run79_event_display's figure code (sed-copied to
run145_event_display_impl so titles say run_145) with the merged-style table
built from the slim `ntof_hits` file instead of the old V12 matcher: per
event, the earliest in-time WAL / PSS hit in the arm gives (detn, dt).

Usage:
    python -m ntof_tracking.run145_displays pick [--n 12]
    python -m ntof_tracking.run145_displays make --best 1
    python -m ntof_tracking.run145_displays tour        # 4 golden events,
                                                        # one per wall segment
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import uproot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ntof_tracking import run145_event_display_impl as disp
from ntof_tracking.run145_target_imaging import apply_w0_kw

SLIM = os.environ.get(
    'RUN145_SLIM',
    '/media/dylan/data/x17/slim/out_224670/'
    'ntof_hits_run_145_stat090_0000_224670.root')
TRACKS = ('/media/dylan/data/x17/beam_july/analysis/wft/run_145/'
          'stat090_0000/mx17_A/events_prelim.parquet')
OUT = ('/media/dylan/data/x17/beam_july/analysis/wft/run_145/'
       'stat090_0000/mx17_A/displays')
DT_LO, DT_HI = -100.0, 60.0
ARM = 'A'


def merged_table():
    """events_prelim + slim WAL/PSS earliest in-time hit, run79-merge style."""
    df = pd.read_parquet(TRACKS)
    import json
    meta = json.load(open(TRACKS.replace('.parquet', '.meta.json')))
    df, _ = apply_w0_kw(df, ARM, meta['bundle']['v_drift'], meta)
    f = uproot.open(SLIM)
    h = f['hits'].arrays(['eventId', 'det', 'detn', 'dt_ns', 'is_control'],
                         library='np')
    ev = f['events'].arrays(['eventId', 'bunch', 't_dream_ns'], library='np')
    it = (h['is_control'] == 0) & (h['dt_ns'] >= DT_LO) & (h['dt_ns'] <= DT_HI)
    out = {}
    for kind, code in (('wal', 0), ('pss', 4)):
        m = it & (h['det'] == code)
        o = np.argsort(np.abs(h['dt_ns'][m]))          # earliest|nearest first
        eid, dn, dt = (h['eventId'][m][o], h['detn'][m][o], h['dt_ns'][m][o])
        first = pd.DataFrame(dict(eventId=eid, detn=dn, dt=dt)) \
                  .drop_duplicates('eventId', keep='first')
        out[kind] = first.set_index('eventId')
    df['wal_detn'] = df.event_id.map(out['wal']['detn']).astype(float)
    df['wal_dt'] = df.event_id.map(out['wal']['dt']).astype(float)
    df['pss_detn'] = df.event_id.map(out['pss']['detn']).astype(float)
    df['pss_dt'] = df.event_id.map(out['pss']['dt']).astype(float)
    evd = pd.DataFrame(ev).set_index('eventId')
    df['BunchNumber'] = df.event_id.map(evd['bunch']).fillna(-1).astype(int)
    df['t_since_flash_ns'] = df.event_id.map(evd['t_dream_ns']).astype(float)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pick')
    p.add_argument('--n', type=int, default=12)
    m = sub.add_parser('make')
    m.add_argument('--best', type=int, default=0)
    m.add_argument('--event', type=int, default=None)
    sub.add_parser('tour')
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    d = merged_table()
    tr = disp.transforms('run_145')[f'mx17_{ARM}']
    picks = disp.pick_events(d, ARM, n=40, tr=tr)
    picks.to_csv(os.path.join(OUT, 'picks.csv'), index=False)

    if a.cmd == 'pick':
        print(picks.head(a.n).to_string())
        return 0
    if a.cmd == 'make':
        evs = ([a.event] if a.event else
               picks.head(max(a.best, 1))['event_id'].tolist())
        for e in evs:
            print('rendering', e)
            disp.make_event(d, int(e), ARM, OUT, tr=tr)
        return 0
    # tour: best event per wall segment, projections only, composed 2x2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    paths = []
    for seg in range(4):
        sub_p = picks[picks['seg'] == seg]
        if not len(sub_p):
            continue
        e = int(sub_p.iloc[0]['event_id'])
        print(f'segment {seg}: event {e}')
        row = d[d.event_id == e].iloc[0]
        g = disp.event_geometry(row, ARM, tr)
        pth = os.path.join(OUT, f'tour_seg{seg}_evt{e}.png')
        disp.fig_projections(row, g, ARM, e, pth)
        paths.append((seg, e, pth))
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (seg, e, pth) in zip(axs.flat, paths):
        ax.imshow(mpimg.imread(pth))
        ax.set_axis_off()
    for ax in axs.flat[len(paths):]:
        ax.set_axis_off()
    fig.suptitle('run_145 / mx17_A wall-segment tour: one confirmed track per '
                 'fired SiPM segment (0-3)', fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'wall_segment_tour_run145.png'), dpi=110)
    print('wrote', os.path.join(OUT, 'wall_segment_tour_run145.png'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
