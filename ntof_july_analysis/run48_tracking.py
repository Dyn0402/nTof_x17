#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run48_tracking.py

Track reconstruction driver for the run_48 scint-DOUBLES dataset (2026-07-16,
Ar/iso 95/5, drift 800 V, resist A/B/C 460 / D 440, 20 mm Pb filter,
32 smp x 60 ns). Built on the new ntof_tracking.reco package.

Modes:
  event   reconstruct + display one event (per-plane views with fits, and
          the global-geometry extrapolation figure)
  search  sift a whole subrun for track candidates (ranked CSV + top-N
          event displays)

Run:
  .venv/bin/python ntof_july_analysis/run48_tracking.py event 1107
  .venv/bin/python ntof_july_analysis/run48_tracking.py search [--subrun S] [--top 12]

Output -> <ANALYSIS_DIR>/July_HV_Scan/run48_tracking/<subrun>/  (flask tab).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from ntof_tracking.reco import io, noise, segments as segmod, pairing  # noqa: E402
from ntof_tracking.reco import geometry as geo, display, search  # noqa: E402

RUN = 'run_48'
DEFAULT_SUBRUN = 'scintd_dr800_A460_D440_00'
OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'run48_tracking')


def load(subrun):
    print(f'  loading {RUN}/{subrun} ...')
    cfg = io.load_run_config(RUN)
    lut = io.build_channel_lut(cfg)
    hits = io.load_subrun_hits(RUN, subrun, lut)
    if hits is None:
        sys.exit(f'no combined hits for {RUN}/{subrun}')
    drift_hv = io.parse_drift_hv(subrun) or 800.0
    drift = geo.DriftModel.from_drift_hv(drift_hv)
    print(f'  {len(hits)} mapped hits, {hits["eventId"].nunique()} events; '
          f'v_drift={drift.v_um_ns:.1f} um/ns (drift {drift_hv:.0f} V), '
          f't0={drift.t0_ns:.0f} ns')
    return cfg, hits, drift


def do_event(ev, subrun, hits=None, cfg=None, drift=None, out_dir=None):
    if hits is None:
        cfg, hits, drift = load(subrun)
    out_dir = out_dir or os.path.join(OUT_BASE, subrun)
    g = hits[hits['eventId'] == ev]
    if g.empty:
        print(f'  [skip] event {ev}: no hits')
        return
    g = noise.flag_noise(g)
    segs = segmod.segments_for_event(g)
    pairs = pairing.pair_xy_3d(segs, drift)
    trs = geo.detector_transforms(cfg)
    gsegs = [geo.segment_to_global(p, trs[p['det']]) for p in pairs]

    n_band = int(g['in_band'].sum())
    print(f'  event {ev}: {len(g)} hits ({n_band} in bands, '
          f'{int(g["clean"].sum())} clean), {len(segs)} clusters, '
          f'{sum(1 for s in segs if s["cls"] == "track")} track segments, '
          f'{len(pairs)} 3D pairs')
    for s in segs:
        if s['cls'] != 'track':
            continue
        print(f"    {s['det']}/{s['plane']}: n={s['n_strips']} "
              f"pspan={s['pspan_mm']:.1f}mm tspan={s['tspan_ns']:.0f}ns "
              f"r2={s.get('r2', np.nan):.3f} amax={s['a_max']:.0f}")
    for p in gsegs:
        d = p['dir_global']
        print(f"    3D {p['det']}: iou={p['iou']:.2f} tan_theta={p['tan_theta']:.2f} "
              f"dir=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}) "
              f"dca_origin={p['dca_origin_mm']:.0f}mm "
              f"dca_beam={p['dca_beam_axis_mm']:.0f}mm "
              f"vert={p['angle_to_vertical_deg']:.1f}deg")
        sp = geo.split_crossings(p)
        for c in sp['outward']:
            print(f"      path (outward) {c['arm']}/{c['name']}: "
                  f"({c['p_in'][0]:+.0f},{c['p_in'][1]:+.0f},{c['p_in'][2]:+.0f})"
                  f" -> ({c['p_out'][0]:+.0f},{c['p_out'][1]:+.0f},"
                  f"{c['p_out'][2]:+.0f}) mm")
        for c in sp['backward']:
            print(f"      back-line only (not path) {c['arm']}/{c['name']}")

    p1 = display.plot_event_planes(
        g, segs, f'{RUN}/{subrun} evt {ev} — hits + reco '
        f'(grey = coherent-band / isolated noise)', out_dir,
        f'evt_{ev:06d}_planes.png')
    print(f'    -> {p1}')
    if gsegs:
        p2 = display.plot_global_tracks(
            gsegs, f'{RUN}/{subrun} evt {ev} — global extrapolation '
            f'(t0={drift.t0_ns:.0f} ns, v={drift.v_um_ns:.1f} um/ns)',
            out_dir, f'evt_{ev:06d}_global.png')
        print(f'    -> {p2}')
    return gsegs


def do_search(subrun, top):
    cfg, hits, drift = load(subrun)
    hits = noise.flag_noise(hits)
    print('  noise flags done; sifting ...')
    out_dir = os.path.join(OUT_BASE, subrun)
    trs = geo.detector_transforms(cfg)
    cand, tracks = search.sift_events(hits, drift, trs)
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, 'candidates.csv')
    cand.to_csv(csv, index=False)
    tracks.to_csv(os.path.join(out_dir, 'tracks.csv'), index=False)
    n_pos = int((cand['score'] > 0).sum())
    n_3d = int((cand['kind'] == '3d_pair').sum())
    n_burst = int((cand['kind'] == 'burst').sum())
    print(f'  {len(cand)} events: {n_pos} with track evidence '
          f'({n_3d} with a 3D X/Y pair, {len(tracks)} pairs total), '
          f'{n_burst} burst-vetoed -> {csv}')
    print(cand.head(top).to_string(index=False))
    for ev in cand.head(top).loc[cand['score'] > 0, 'eventId']:
        do_event(int(ev), subrun, hits=hits, cfg=cfg, drift=drift,
                 out_dir=out_dir)


def all_subruns():
    run_dir = os.path.join(io.BASE_PATH, RUN)
    return sorted(n for n in os.listdir(run_dir)
                  if os.path.isdir(os.path.join(run_dir, n))
                  and n.startswith('scintd_'))


def do_beamproj(dca_cut=150.0):
    """Beam-axis projection of ALL 3D pairs across every searched subrun:
    where along the beamline (global Y) do the reconstructed tracks point
    back to, and how close do they come to the axis?"""
    import glob
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ntof_tracking.reco import geometry as geo

    files = sorted(glob.glob(os.path.join(OUT_BASE, '*', 'tracks.csv')))
    if not files:
        sys.exit('no tracks.csv found — run search first')
    frames = []
    for f in files:
        t = pd.read_csv(f)
        t['subrun'] = os.path.basename(os.path.dirname(f))
        frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    tr = tr[tr['n_pair_dets'] < 3]          # drop multi-det pile-up events
    print(f'  {len(tr)} 3D pairs from {len(files)} subruns '
          f'({(tr.dca_beam_mm < dca_cut).sum()} with dca_beam < {dca_cut:.0f} mm)')

    sel = tr[tr['dca_beam_mm'] < dca_cut]
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.2))

    ax = axs[0]
    ax.hist(tr['dca_beam_mm'].clip(0, 400), bins=40, color='steelblue',
            alpha=.85)
    ax.axvline(dca_cut, color='crimson', ls='--', lw=1,
               label=f'projection cut {dca_cut:.0f} mm')
    ax.set_xlabel('DCA to beam axis [mm]')
    ax.set_ylabel('3D pairs')
    ax.set_title('impact parameter to the beamline')
    ax.legend(fontsize=8)

    ax = axs[1]
    ax.hist(sel['beam_y_mm'].clip(-600, 600), bins=60, color='steelblue',
            alpha=.85)
    ax.axvspan(geo.HE3_GAS_Y[0], geo.HE3_GAS_Y[-1], color='#99d8f5',
               alpha=.6, label='He-3 gas (target)')
    ax.axvline(0, color='k', lw=.5)
    ax.set_xlabel('Y at beam-axis closest approach [mm]')
    ax.set_ylabel(f'pairs (dca < {dca_cut:.0f} mm)')
    ax.set_title('source profile along the beamline')
    ax.legend(fontsize=8)

    ax = axs[2]
    for det, m in sel.groupby('det'):
        ax.scatter(m['beam_y_mm'].clip(-600, 600), m['dca_beam_mm'],
                   s=12, alpha=.7, label=f'{det} ({len(m)})')
    ax.axvspan(geo.HE3_GAS_Y[0], geo.HE3_GAS_Y[-1], color='#99d8f5', alpha=.4)
    ax.set_xlabel('Y at closest approach [mm]')
    ax.set_ylabel('DCA to beam axis [mm]')
    ax.set_title('per detector')
    ax.legend(fontsize=8)

    ax = axs[3]
    ax.hist(tr['vert_deg'], bins=36, range=(0, 90), color='steelblue',
            alpha=.85, label='all pairs')
    ax.hist(sel['vert_deg'], bins=36, range=(0, 90), color='crimson',
            alpha=.6, label=f'dca < {dca_cut:.0f} mm')
    ax.set_xlabel('track angle to vertical/beam axis [deg]')
    ax.set_ylabel('pairs')
    ax.set_title('inclination')
    ax.legend(fontsize=8)

    med = sel['beam_y_mm'].median()
    p16, p84 = np.percentile(sel['beam_y_mm'], [16, 84])
    s68 = 0.5 * (p84 - p16)
    fig.suptitle(f'{RUN} — beam-axis projection of all reconstructed 3D pairs '
                 f'(t0=450 ns, v=26.5 um/ns) — median Y = {med:.0f} mm, '
                 f'sigma68 = {s68:.0f} mm', fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_BASE, 'beam_projection.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'  median beam_y = {med:.1f} mm, sigma68 = {s68:.1f} mm '
          f'(He-3 gas spans {geo.HE3_GAS_Y[0]:.0f}..{geo.HE3_GAS_Y[-1]:.0f} mm)')
    print(sel.groupby('det')['beam_y_mm'].describe()[['count', 'mean', '50%', 'std']]
          .round(1).to_string())
    print(f'  figure -> {out}')


def do_ensemble(dca_cut=75.0):
    """Pool every searched subrun's 3D pairs, keep those whose line passes
    within dca_cut mm of the TARGET (dca to the origin = He-3 centre; multi-
    det pile-up events dropped), re-reconstruct just those events to recover
    the global segment endpoints, and draw the whole sample together on
    (a) the 3-view global extrapolation figure and (b) a 3D model of the
    active Geant4 geometry."""
    import glob
    files = sorted(glob.glob(os.path.join(OUT_BASE, '*', 'tracks.csv')))
    if not files:
        sys.exit('no tracks.csv found — run search first')
    frames = []
    for f in files:
        t = pd.read_csv(f)
        t['subrun'] = os.path.basename(os.path.dirname(f))
        frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    sel = tr[(tr['n_pair_dets'] < 3) & (tr['dca_origin_mm'] < dca_cut)]
    print(f'  {len(sel)}/{len(tr)} 3D pairs with dca(target) < {dca_cut:.0f} mm '
          f'({sel["eventId"].nunique()} events, '
          f'{sel["subrun"].nunique()} subruns)')

    from ntof_tracking.reco import search
    gsegs_all = []
    for subrun, m in sel.groupby('subrun'):
        cfg, hits, drift = load(subrun)
        hits = hits[hits['eventId'].isin(m['eventId'].unique())]
        hits = noise.flag_noise(hits)
        trs = geo.detector_transforms(cfg)
        for ev, g in hits.groupby('eventId'):
            busy = search.busy_detectors(g)
            segs = [s for s in segmod.segments_for_event(g)
                    if s['det'] not in busy]
            pairs = pairing.pair_xy_3d(segs, drift)
            if len({p['det'] for p in pairs}) >= 3:   # same pile-up veto
                continue
            for p in pairs:
                gs = geo.segment_to_global(p, trs[p['det']])
                if gs['dca_origin_mm'] < dca_cut:
                    gs['subrun'] = subrun
                    gsegs_all.append(gs)
    print(f'  re-reconstructed {len(gsegs_all)} pairs passing the cut')
    if not gsegs_all:
        sys.exit('nothing selected — loosen --dca-cut')

    out_dir = os.path.join(OUT_BASE, f'ensemble_dca{dca_cut:.0f}')
    os.makedirs(out_dir, exist_ok=True)
    rows = [dict(subrun=s['subrun'], eventId=s['eventId'], det=s['det'],
                 dca_origin_mm=s['dca_origin_mm'],
                 dca_beam_mm=s['dca_beam_axis_mm'], beam_y_mm=s['beam_y_mm'],
                 vert_deg=s['angle_to_vertical_deg'], iou=s['iou'])
            for s in gsegs_all]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'ensemble_tracks.csv'),
                              index=False)
    ttl = (f'{RUN} — all {len(gsegs_all)} 3D pairs with DCA(target) < '
           f'{dca_cut:.0f} mm, {len(files)} subruns pooled '
           f'(t0=450 ns, in-plane alignment provisional)')
    p1 = display.plot_global_ensemble(
        gsegs_all, ttl + ' — global extrapolation', out_dir,
        'ensemble_global.png')
    print(f'  -> {p1}')
    p2 = display.plot_global_3d(
        gsegs_all, ttl + ' — 3D active-geometry model (Geant4 volumes)',
        out_dir, 'ensemble_3d.png')
    print(f'  -> {p2}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    ap_e = sub.add_parser('event')
    ap_e.add_argument('event', type=int)
    ap_e.add_argument('--subrun', default=DEFAULT_SUBRUN)
    ap_s = sub.add_parser('search')
    ap_s.add_argument('--subrun', default=DEFAULT_SUBRUN,
                      help="subrun name, comma list, or 'all'")
    ap_s.add_argument('--top', type=int, default=12)
    ap_b = sub.add_parser('beamproj')
    ap_b.add_argument('--dca-cut', type=float, default=150.0)
    ap_n = sub.add_parser('ensemble')
    ap_n.add_argument('--dca-cut', type=float, default=75.0,
                      help='max DCA of the track line to the target '
                           '(origin) [mm]')
    args = ap.parse_args()
    if args.mode == 'event':
        do_event(args.event, args.subrun)
    elif args.mode == 'beamproj':
        do_beamproj(args.dca_cut)
    elif args.mode == 'ensemble':
        do_ensemble(args.dca_cut)
    else:
        subruns = (all_subruns() if args.subrun == 'all'
                   else args.subrun.split(','))
        for s in subruns:
            do_search(s, args.top)
    print('done')
