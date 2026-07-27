#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py — aggregate the Det-A double-track scan, rank candidates, and render
event-display galleries.

Reads the per-subrun caches written by scan.py:
  cache/<run>/<subrun>_ev.parquet   (per-event features)
  cache/<run>/<subrun>_cand.pkl     (full detail for is_double events)

Outputs (under <ANALYSIS>/July_HV_Scan/detA_doubletrack/):
  census.txt              — counts per run / topology / clean-vs-busy
  candidates.csv          — every double-track candidate, ranked, with metrics
  gallery/rank###_*.png   — event displays for the top candidates
"""
import argparse
import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan as SC  # noqa: E402
import dtrack_lib as D  # noqa: E402
from ntof_tracking.reco import io, geometry as geo  # noqa: E402

_TR_CACHE = {}


def plane_separation(lines):
    """In-plane spatial separation [mm] of the two highest-charge lines: how
    far apart the two candidate tracks actually are in this projection.
    Well-separated (>~50 mm) = an unambiguous double; small (~15-25 mm) =
    marginal / possible over-split; large = clean."""
    if len(lines) < 2:
        return np.nan
    a, b = sorted(lines, key=lambda l: -l['q_sum'])[:2]
    return D._max_union_sep(a, b)


def _transform(run):
    """Cached mx17_A local->global DetTransform for a run."""
    if run not in _TR_CACHE:
        _TR_CACHE[run] = geo.detector_transforms(io.load_run_config(run))
    return _TR_CACHE[run]['mx17_A']


def enrich_geometry(c):
    """Add global-frame pointing to a candidate's two highest-charge 3D pairs
    and the inter-track 3D closest approach (the vertex test). All provisional:
    depth uses the uncalibrated DAQ t0 (~450 ns) + bench 95/5 v_drift, so the
    absolute global scale carries systematic error -- these rank & flag, they
    do not measure. Writes fields into c['geo']."""
    r = c['res']
    tr = _transform(c['run'])
    pairs = sorted(r['pairs'], key=lambda p: -(p['q_x'] + p['q_y']))[:2]
    g = [geo.segment_to_global(p, tr) for p in pairs]
    out = dict(n_pair=len(r['pairs']))
    if len(g) >= 2:
        g1, g2 = g[0], g[1]
        out.update(
            dca_beam_1=g1['dca_beam_axis_mm'], dca_beam_2=g2['dca_beam_axis_mm'],
            beam_y_1=g1['beam_y_mm'], beam_y_2=g2['beam_y_mm'],
            vert_deg_1=g1['angle_to_vertical_deg'],
            vert_deg_2=g2['angle_to_vertical_deg'],
            # 3D distance of closest approach between the two track lines:
            # small -> they meet at a point (a vertex); large -> independent
            track_dca_3d=geo.line_line_dist(
                g1['p_lo_global'], g1['dir_global'],
                g2['p_lo_global'], g2['dir_global']),
            # do both point back near the beamline (common upstream origin)?
            both_point_beam=bool(g1['dca_beam_axis_mm'] < 400
                                 and g2['dca_beam_axis_mm'] < 400),
        )
        out['dca_beam_max'] = max(g1['dca_beam_axis_mm'], g2['dca_beam_axis_mm'])
    c['geo'] = out
    return c

OUT = SC.OUT_BASE
GALLERY = os.path.join(OUT, 'gallery')
LINE_COL = ['crimson', 'royalblue', 'seagreen', 'darkorange', 'purple', 'brown']


def load_events(runs):
    frames = []
    for run in runs:
        for p in sorted(glob.glob(os.path.join(SC.CACHE_DIR, run, '*_ev.parquet'))):
            df = pd.read_parquet(p)
            if len(df):
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def refilter(c):
    """Re-apply the CURRENT distinctness test to a cached candidate's stored
    lines, and re-pair. The test only ever got stricter (no-extrapolation
    rule), and dtrack_lib.distinct_lines is greedy over the same charge order,
    so a line the cached pass dropped would also be dropped now -- re-filtering
    the cache is therefore equivalent to re-scanning, without the 2 h re-run.
    Returns True if the event is still a double."""
    r = c['res']
    xl = D.distinct_lines(r['xlines'])
    yl = D.distinct_lines(r['ylines'])
    if len(xl) < 2 or len(yl) < 2:
        return False
    drift = geo.DriftModel.from_drift_hv(c['meta'].get('drift') or 800.0)
    r['xlines'], r['ylines'] = xl, yl
    r['n_xline'], r['n_yline'] = len(xl), len(yl)
    r['pairs'] = D._pair_lines(xl, yl, drift)
    r['n_pair'] = len(r['pairs'])
    return True


def load_candidates(runs):
    cands, dropped = [], 0
    for run in runs:
        for p in sorted(glob.glob(os.path.join(SC.CACHE_DIR, run, '*_cand.pkl'))):
            with open(p, 'rb') as f:
                for c in pickle.load(f):
                    c['run'] = run
                    if refilter(c):
                        cands.append(c)
                    else:
                        dropped += 1
    if dropped:
        print(f'[refilter] dropped {dropped} cached candidates that fail the '
              f'no-extrapolation distinctness test (fragmented single tracks)')
    return cands


def score(meta, res, gg, min_sep):
    """Rank a double-track candidate. Golden = clean, well-separated, X/Y-paired,
    recovered-window, high-quality lines, and pointing back toward the beamline;
    penalise busy / high-multiplicity / unpaired / marginally-separated."""
    s = 0.0
    s += res['n_pair'] * 2.0                        # 3D X/Y confirmation is king
    s += meta['min_r2'] * 3.0                       # line straightness
    s += 2.0 if not meta['busy'] else 0.0           # low occupancy
    # a clean 2+2 (or 2+3) is more trustworthy than a 5+5 shower/pileup
    s -= 0.6 * max(0, res['n_xline'] - 3)
    s -= 0.6 * max(0, res['n_yline'] - 3)
    s += 1.0 if meta['dt_ms'] > 8 else 0.0          # post-flash recovered window
    if gg.get('both_point_beam'):                   # both radiate from beamline
        s += 2.0
    # reward clear spatial separation: a 90 mm gap is an unambiguous double,
    # a 15 mm gap (near the distinctness floor) is marginal / possible split
    s += min(np.nan_to_num(min_sep) / 45.0, 2.5)
    return s


def topology(gg):
    """Physics-motivated tag from the two tracks' 3D geometry (provisional)."""
    if 'track_dca_3d' not in gg:
        return 'unpaired'
    vtx = gg['track_dca_3d'] < 30.0                 # meet within ~1 gap-width
    pt = gg.get('both_point_beam', False)
    if vtx and pt:
        return 'vertex+beam'                        # common origin, beam-pointing
    if vtx:
        return 'vertex'
    if pt:
        return 'separated+beam'                     # two beam-pointing tracks
    return 'separated'


def build_table(cands):
    rows = []
    for c in cands:
        enrich_geometry(c)
        m, r, gg = c['meta'], c['res'], c['geo']
        sep_x = plane_separation(r['xlines'])
        sep_y = plane_separation(r['ylines'])
        min_sep = float(np.nanmin([sep_x, sep_y]))
        c['min_sep'] = min_sep
        rows.append(dict(
            run=c['run'], subrun=m['subrun'], eventId=m['eventId'],
            drift=m['drift'], dt_ms=round(m['dt_ms'], 2), busy=m['busy'],
            n_clean_strips=m['n_clean_strips'],
            n_xline=r['n_xline'], n_yline=r['n_yline'], n_pair=r['n_pair'],
            min_r2=round(m['min_r2'], 3), q_lines=round(m['q_lines'], 0),
            sep_x=round(sep_x, 0), sep_y=round(sep_y, 0),
            min_sep=round(min_sep, 0), topo=topology(gg),
            track_dca_3d=round(gg.get('track_dca_3d', np.nan), 1),
            dca_beam_max=round(gg.get('dca_beam_max', np.nan), 0),
            beam_y_1=round(gg.get('beam_y_1', np.nan), 0),
            beam_y_2=round(gg.get('beam_y_2', np.nan), 0),
            score=round(score(m, r, gg, min_sep), 3),
        ))
    df = pd.DataFrame(rows)
    return df.sort_values('score', ascending=False).reset_index(drop=True)


def draw_candidate(c, rank, path):
    m, r, dump = c['meta'], c['res'], c['dump']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True)
    for ax, pl, lines in zip(axes, ('x', 'y'), (r['xlines'], r['ylines'])):
        d = dump[pl]
        dn = dump[pl + '_noise']
        ax.scatter(dn['time'], dn['pos'], s=7, c='0.82', marker='.',
                   label='flagged noise', zorder=1)
        ax.scatter(d['time'], d['pos'], s=16, c='0.4', marker='o',
                   label='clean (unassigned)', zorder=2)
        for i, ln in enumerate(lines):
            col = LINE_COL[i % len(LINE_COL)]
            ii = ln['idx']
            ax.scatter(d['time'][ii], d['pos'][ii], s=32, facecolors='none',
                       edgecolors=col, linewidths=1.6, zorder=4,
                       label=f"line {i}: n={ln['n_hits']} r2={ln['r2']:.2f} "
                             f"slope={ln['slope_mm_ns']*1000:.0f}um/ns")
            tt = np.array([ln['t0_ns'], ln['t1_ns']])
            ax.plot(tt, ln['slope_mm_ns'] * tt + ln['intercept_mm'],
                    c=col, lw=2, zorder=3)
        ax.set_title(f'Det A  plane {pl}   ({len(lines)} lines)')
        ax.set_xlabel('drift time [ns]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')
    axes[0].set_ylabel('strip position [mm, centred]')
    gg = c.get('geo', {})
    dca = gg.get('track_dca_3d', np.nan)
    fig.suptitle(
        f"#{rank}  {c['run']}/{m['subrun']}  ev{m['eventId']}   "
        f"drift={m['drift']:.0f}V dt={m['dt_ms']:.1f}ms   "
        f"lines {r['n_xline']}x/{r['n_yline']}y  pairs={r['n_pair']}  "
        f"{'BUSY ' if m['busy'] else ''}topo={topology(gg)}"
        + (f"  trackDCA={dca:.0f}mm" if np.isfinite(dca) else ''),
        fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*',
                    default=['run_58', 'run_61', 'run_62', 'run_63'])
    ap.add_argument('--top', type=int, default=40, help='gallery size')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ev = load_events(a.runs)
    cands = load_candidates(a.runs)
    tbl = build_table(cands)

    # ---- census ----
    lines = []
    lines.append(f'Det-A double-track scan census  (runs: {", ".join(a.runs)})')
    lines.append('=' * 60)
    lines.append(f'total reco events (survived prefilter): {len(ev)}')
    if len(ev):
        for run, g in ev.groupby('run'):
            nd = int(g['is_double'].sum())
            lines.append(f'  {run}: {len(g):6d} reco ev, {nd:4d} doubles '
                         f'({100*nd/max(len(g),1):.2f}%)')
    if len(tbl):
        lines.append('')
        lines.append(f'double-track candidates: {len(tbl)}')
        lines.append(f'  clean (not busy): {int((~tbl.busy).sum())}   '
                     f'busy: {int(tbl.busy.sum())}')
        lines.append(f'  with >=2 3D pairs: {int((tbl.n_pair>=2).sum())}')
        lines.append(f'  exactly 2x/2y lines: '
                     f'{int(((tbl.n_xline==2)&(tbl.n_yline==2)).sum())}')
        for tp, g in tbl.groupby('topo'):
            lines.append(f'  topo {tp}: {len(g)}')
        lines.append(f'  well-separated (min_sep>=40mm): '
                     f'{int((tbl.min_sep >= 40).sum())}')
        lines.append('')
        lines.append('GOLDEN sample (not busy, >=2 pairs, well-separated >=40mm):')
        gold = tbl[(~tbl.busy) & (tbl.n_pair >= 2) & (tbl.min_sep >= 40)]
        lines.append(f'  {len(gold)} events')
        for _, g in gold.head(15).iterrows():
            lines.append(f"    {g.run}/{g.subrun} ev{g.eventId}  "
                         f"sep={g.min_sep:.0f}mm dca3d={g.track_dca_3d:.0f}mm "
                         f"topo={g.topo} dt={g.dt_ms:.0f}ms score={g.score:.1f}")
    census = '\n'.join(lines)
    print(census)
    with open(os.path.join(OUT, 'census.txt'), 'w') as f:
        f.write(census + '\n')
    tbl.to_csv(os.path.join(OUT, 'candidates.csv'), index=False)
    print(f'\n-> candidates.csv ({len(tbl)} rows)')

    # ---- gallery for the top-ranked ----
    key = {(c['run'], c['meta']['subrun'], c['meta']['eventId']): c
           for c in cands}
    n = min(a.top, len(tbl))
    for rank in range(n):
        row = tbl.iloc[rank]
        c = key[(row['run'], row['subrun'], row['eventId'])]
        fn = (f"rank{rank:03d}_{row['run']}_{row['subrun']}_ev"
              f"{row['eventId']}.png")
        draw_candidate(c, rank, os.path.join(GALLERY, fn))
    print(f'-> gallery/ ({n} displays)')


if __name__ == '__main__':
    main()
