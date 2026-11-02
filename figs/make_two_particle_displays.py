#!/usr/bin/env python3
"""make_two_particle_displays.py -- Paper I section 5 (two particles in one chamber): det3 event displays in the A1 style.

    .venv/bin/python papers/paper1/figs/make_two_particle_displays.py \
        [--root <paper1_eff>] [--ana <Analysis_pass1>] [--eos-run-dir <june_tests run dir>] [--out papers/paper1/figs/out]
    ... --sheet merged=4780,14247,... --sheet-out DIR       # contact sheets, to choose the events by eye

Figures (golden row sat_det3, pass 1, rr1 waveforms; both views, strip position x drift time, M3 dashed in both views):
  display_merged_projection  two particle-like segments in one view, one merged column in the other (one-view class
                             'merged in projection' of rerun/effbudget/single/single_track_v2.py: oneview_events);
  display_muon_deposit       a clean muon on M3 plus a separate deposit elsewhere in the plane, the trigger vetoed by the
                             50-strip discharge rule (sparks class 'muon+other', x17rerun/paper1_eff/a1/<rel>/sparks_det3);
                             no pass fit exists for a vetoed trigger, so only M3 is drawn;
  display_two_separate       two separate segments in both views, both fitted by the pass (n_tracks = 2).
Lines: the pass's candidate fits from events.candidates.parquet -- the pass's first track (track_id 0) vermillion, a
second accepted track (track_id 1) amber, a fitted candidate the pass did not keep (track_id -1) amber dotted.  Data access is a1_displays.py's (wft.io.FeuReader, pedestal + CM as the
reconstruction reads them); an event outside the pass table (vetoed) is taken from the one data file that holds its id
(the run stops if more than one does).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'papers'), os.path.join(REPO, 'rerun', 'paper1_figs')]
import a1_displays as a1  # noqa: E402  (data access: posmap, read_waveforms, the registry row)

KEY = 'sat_det3'
#: figure id -> (det3 event, title, which candidate fits to draw: 'all', 'tracks' (track_id >= 0) or 'none')
DISPLAYS = {
    'display_merged_projection': (17907, 'Two particles: separate in X, merged in projection in Y', 'all'),
    'display_muon_deposit': (36603, 'Muon + separate deposit: trigger vetoed as a discharge (> 50 strips)', 'none'),
    'display_two_separate': (1063, 'Two particles, separate in both views', 'tracks'),
}


def load(a):
    row = dict(a1.inputs.row_for(KEY)); row['key'] = KEY
    if a.eos_run_dir:
        row['eos_run_dir'] = a.eos_run_dir
    rel = os.path.join(row['run'], row['subrun'], row['det'])
    R = pd.read_parquet(os.path.join(a.root, 'budget', rel, 'rays_det3.parquet')).drop_duplicates('event_id').set_index('event_id')
    wdir = os.path.join(a.ana, rel, 'wft')
    E = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    E = E[~E.event_id.duplicated(keep=False)].set_index('event_id')
    K = pd.read_parquet(os.path.join(wdir, 'events.candidates.parquet'))
    B = json.load(open(os.path.join(wdir, 'calib_bundle', 'bundle.json')))
    return row, R, E, K, float(B['v_drift']) * 1e-3


def waveforms(row, E, ids):
    """{eid: {'x': W, 'y': W}}; the acquisition from the pass table, else the one data file holding the id."""
    import glob
    from wft import io as wio
    dec = a1.inputs.GENERATIONS['rr1']['decoded_root']
    acq = {e: str(E.loc[e, 'acq_tag']) for e in ids if e in E.index}
    for e in ids:
        if e in acq:
            continue
        fs = sorted(f for f in glob.glob(os.path.join(row['eos_run_dir'], row['subrun'], dec, f'*_{int(row["feu_x"]):02d}.root'))
                    if '_datrun_' in os.path.basename(f))
        tags = [wio.file_tag(f) for f in fs if e in set(wio.FeuReader(f).event_ids.tolist())]
        if len(tags) != 1:
            sys.exit(f'event {e} (not in the pass table) is in {len(tags)} data files: ambiguous')
        acq[e] = tags[0]
    W = a1.read_waveforms(row, 'rr1', {(acq[e], e) for e in ids})
    return {e: W.get((acq[e], e), {}) for e in ids}


def draw(axs, e, w, PM, R, E, K, v_mm, es, cands='all'):
    r = R.loc[e] if e in R.index else None
    f = E.loc[e] if e in E.index else None
    pc = None
    for v, ax in zip('xy', axs):
        P = PM[v].astype(float)
        mesh = float(r[f'mesh_{v}']) if r is not None else np.nan
        tan = float(r[f'tan_{v}']) if r is not None else 0.0
        t0 = float(f[f'{v}_t0']) if f is not None and np.isfinite(f[f'{v}_t0']) else (257.5 if v == 'x' else 302.5)
        k = K[(K.event_id == e) & (K.plane == v)]
        if f is not None:
            k = k[k.acq_tag == f['acq_tag']]
        if cands == 'tracks':
            k = k[k.track_id >= 0]
        elif cands == 'none':
            k = k.iloc[:0]
        # the window: M3 and every fitted candidate, plus the charge (vetoed events)
        xs = [mesh, mesh + 30 * tan] if np.isfinite(mesh) else []
        for c in k.itertuples():
            xs += [c.p0, c.p0 + c.w * 30 / v_mm]
        Wv = w[v]
        amp = Wv.max(1)
        lit = np.isfinite(P) & (amp > 0.25 * np.nanmax(amp))
        if not len(k):
            xs += [np.nanmin(P[lit]), np.nanmax(P[lit])]
        lo, hi = np.nanmin(xs), np.nanmax(xs)
        cen, zoom = 0.5 * (lo + hi), 0.5 * (hi - lo) + 10
        pc, _ = es.cells(ax, Wv, P, cen, zoom)
        if np.isfinite(mesh):
            es.m3_track(ax, mesh, tan, t0, v_mm, label='M3 track')
        for c in k.sort_values('track_id', key=lambda s: s.replace(-1, 99)).itertuples():
            if c.track_id == 0:
                kw, lab = es.FIT_KW, 'pass fit, first track'
            elif c.track_id > 0:
                kw, lab = es.FIT2_KW, 'pass fit, second track'
            else:
                kw, lab = dict(es.FIT2_KW, ls=(0, (1.5, 1.5))), 'fitted candidate, not kept'
            es.track(ax, c.p0, c.w, c.t0, v_mm, kw, label=lab)
        es.frame(ax, cen, zoom, v.upper())
        ax.set_title(f'{v.upper()} view', loc='left', fontsize=8)
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=a1.ROOT); ap.add_argument('--ana', default=a1.ANA)
    ap.add_argument('--eos-run-dir', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--sheet', action='append', default=[], help='NAME=EID,EID,... contact sheet instead of figures')
    ap.add_argument('--sheet-out', default=None)
    ap.add_argument('--only', default='', help='figure ids, comma-separated')
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as ps
    import evd_style as es
    ps.use()
    row, R, E, K, v_mm = load(a)
    PM, _ = a1.posmap(row)
    if a.sheet:
        sheets = {s.split('=')[0]: [int(x) for x in s.split('=')[1].split(',')] for s in a.sheet}
        ids = sorted({e for v in sheets.values() for e in v})
        WF = waveforms(row, E, ids)
        os.makedirs(a.sheet_out, exist_ok=True)
        for name, es_ in sheets.items():
            fig, axs = plt.subplots(len(es_), 2, figsize=(6.1, 2.4 * len(es_)), squeeze=False, constrained_layout=True)
            for i, e in enumerate(es_):
                draw(axs[i], e, WF[e], PM, R, E, K, v_mm, es)
                axs[i, 0].set_ylabel(str(e))
            fig.savefig(os.path.join(a.sheet_out, f'{name}.png'), dpi=90); plt.close(fig)
            print('sheet', name)
        return
    todo = {k: v for k, v in DISPLAYS.items() if not a.only or k in a.only.split(',')}
    WF = waveforms(row, E, [v[0] for v in todo.values()])
    for fid, (e, title, cands) in todo.items():
        fig, axs = plt.subplots(1, 2, figsize=(ps.FULL[0], 2.75), sharey=True, constrained_layout=True)
        pc = draw(axs, e, WF[e], PM, R, E, K, v_mm, es, cands)
        axs[0].set_ylabel('drift time [ns]')
        fig.suptitle(f'{title}  (det3 event {e})', fontsize=9, x=0.01, ha='left')
        cb = fig.colorbar(pc, ax=axs, shrink=0.92, pad=0.01)
        cb.set_label('amplitude / view maximum', fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        hs, ls = {}, {}
        for ax in axs:
            for h, l in zip(*ax.get_legend_handles_labels()):
                hs.setdefault(l, h)
        axs[0].legend(list(hs.values()), list(hs.keys()), fontsize=6.5, loc='lower left', frameon=True,
                      framealpha=0.9, edgecolor='none')
        ps.save(fig, os.path.join(a.out, fid))
        print('wrote', fid, flush=True)


if __name__ == '__main__':
    main()
