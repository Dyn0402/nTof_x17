#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_microtpc.py

Apply the June cosmic-bench micro-TPC reconstruction (ntof_tracking) to the
run_30 nTOF scintillator events, detector A (= bench det3, the fleet default).

This is a first-look driver for TRACK_PLAN_02: per plane it finds compact
clusters, runs the bench chain
    anchored_time_fit  ->  hit_features  ->  apply_tan_regression(frozen det3,
    restandardize on the target compact-cluster population)  ->  apply_sign
and keeps the ones that look like real micro-TPC tracks (a resolvable drift-time
gradient with a clean position-vs-time correlation).

We hunt ANY track in this ONE detector, NOT chamber pairs.  "Plane" here is a
READOUT plane of detector A (each Micromegas has two orthogonal strip planes, X
and Y) — NOT a second Micromegas.  A single-plane gradient is already a valid
micro-TPC track segment and is reported/rendered on its own; a gradient in BOTH
the X and Y planes of this same chamber merely upgrades it to a full 3-D
micro-TPC track *within the one detector*.  Two-*Micromegas* linking (needs a
target/vertex) is a separate downstream stage (TRACK_PLAN_04), not done here and
not required for anything below.  Each rendered display is 2-panel (X | Y); for a
single-plane candidate the empty plane's panel is simply hidden.

KEY QUESTION this answers (asked by the user): are the nTOF hits fundamentally
different from the Saclay cosmics?  Most compact nTOF clusters are ISOCHRONOUS
(all strips at one drift time = a point-like deposit + charge sharing), unlike a
cosmic which crosses the 30 mm gap and spans up to T_sat~691 ns.  Only a small
minority carry a genuine micro-TPC gradient — those are reconstructable.

Output -> {ANALYSIS_DIR}July_HV_Scan/run30_microtpc/<block>/
  microtpc_evt_*.png   per-track-event displays (both planes) w/ angle
  summary.png          duration & |corr| distribution: tracks vs point deposits

Run:  .venv/bin/python ntof_july_analysis/run30_microtpc.py [block ...]
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from ntof_tracking import microtpc_lib as mt          # noqa: E402
from ntof_tracking import bench_constants as bc       # noqa: E402
from common.Mx17StripMap import Detector, Mx17StripMap  # noqa: E402
from july_hv_scan import (  # noqa: E402
    BASE_PATH, ANALYSIS_DIR, MAP_CSV_PATH, load_config, _save, _real_files,
)
import uproot  # noqa: E402


def load_hits_tot(run, subrun, feu_ids):
    """Like july_hv_scan.load_hits but keeps time_over_threshold (hit_features)."""
    hits_dir = os.path.join(BASE_PATH, run, subrun, 'combined_hits_root')
    good = []
    for s in _real_files(hits_dir, '.root'):
        try:
            with uproot.open(s) as f:
                if 'hits' in f:
                    good.append(s)
        except Exception:
            continue
    if not good:
        return None
    df = uproot.concatenate(
        [f'{s}:hits' for s in good],
        ['eventId', 'feu', 'channel', 'time', 'amplitude', 'time_over_threshold'],
        library='pd')
    return df[df['feu'].isin(feu_ids)].copy()

RUN = 'run_30'
DET = 'mx17_A'
DET_MX = 3                       # detector A = bench det3
MODEL_PATH = os.path.join(_ROOT, 'ntof_tracking', 'models', 'mx17_3_hits6.json')
DEFAULT_BLOCKS = ['scintOff_A700_00', 'scintOn_A700_00']

THR = 150.0                      # beam hit threshold (ADC)
# compact single-plane cluster (a track candidate, not a flash blob)
CL_NMIN, CL_NMAX = 5, 30
CL_EXTENT_MAX = 30.0             # mm
# micro-TPC track cut: resolvable drift gradient + clean linear correlation
TRK_DUR_MIN = 150.0              # ns  (>> single-strip timing sigma ~39 ns)
TRK_R_MIN = 0.85                 # |pearson(pos,time)|


def _plane_of(cfg):
    """Return dict feu->'x'/'y' for detector A from dream_feus."""
    dc = [d for d in cfg['detectors'] if d['name'] == DET][0]
    feu_axis = {}
    for det_key, (feu, conn) in dc['dream_feus'].items():
        feu_axis[feu] = det_key[0]      # 'x' or 'y'
    return feu_axis


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std()))


def cluster_candidates(df, det, coord):
    """Yield (eventId, pos, time, amp, tot) for the compact cluster of each
    event in this plane."""
    for ev, g in df.groupby('eventId'):
        gg = g[g[coord].notna()]
        if not (CL_NMIN - 1 <= len(gg) <= 60):
            continue
        pos = gg[coord].to_numpy(); t = gg['time'].to_numpy()
        a = gg['amplitude'].to_numpy(); q = gg['time_over_threshold'].to_numpy()
        m = mt.gap_cluster_largest(pos)
        yield ev, pos[m], t[m], a[m], q[m]


def collect(df, det, coord):
    """Per-plane: fit every compact cluster, return list of segment dicts."""
    segs = []
    for ev, pos, t, a, q in cluster_candidates(df, det, coord):
        if not (CL_NMIN <= len(pos) <= CL_NMAX and np.ptp(pos) <= CL_EXTENT_MAX):
            continue
        fit = mt.anchored_time_fit(pos, t, a)
        if fit is None:
            continue
        feat = mt.hit_features(pos, a, t, q)
        r = _pearson(pos, t)
        segs.append(dict(ev=ev, pos=pos, t=t, a=a, q=q, r=r, fit=fit, feat=feat,
                         is_track=(fit['duration_ns'] > TRK_DUR_MIN
                                   and abs(r) > TRK_R_MIN)))
    return segs


def reco_angles(segs, model, plane):
    """Restandardize the frozen det3 model on this plane's candidate population
    and attach tan_reg (signed) to each segment that has features."""
    mplane = model['planes'][plane]
    feats = mplane['feats']
    F = np.array([[s['feat'][f] if s['feat'] else np.nan for f in feats]
                  for s in segs], float)
    tan_abs, ok = mt.apply_tan_regression(mplane, F, restandardize=True)
    wg = mplane.get('wg')
    if wg is not None:
        aas = np.array([s['feat'].get('a_asym_sgn', np.nan) if s['feat'] else np.nan
                        for s in segs], float)
        tas = np.array([s['feat'].get('t_asym_sgn', np.nan) if s['feat'] else np.nan
                        for s in segs], float)
        sign = mt.apply_sign(wg, aas, tas,
                             fallback_sign=np.sign([s['fit']['slope_ns_per_mm']
                                                    for s in segs]))
    else:
        sign = np.sign([s['fit']['slope_ns_per_mm'] for s in segs])
    for s, ta, sg in zip(segs, tan_abs, sign):
        s['tan_reg'] = (sg * ta) if np.isfinite(ta) else np.nan
        s['theta_deg'] = np.degrees(np.arctan(s['tan_reg'])) if np.isfinite(s['tan_reg']) else np.nan


def render_track(ev, sx, sy, block, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, s, lab in [(axes[0], sx, 'X'), (axes[1], sy, 'Y')]:
        if s is None:
            ax.set_visible(False); continue
        pos, t = s['pos'], s['t']
        sc = ax.scatter((t - t.min()), pos, c=s['a'], cmap='viridis', s=45,
                        vmin=150, vmax=3000, zorder=3)
        # anchored fit line: t = t0 + slope*(pos-p0)  ->  invert for pos(t)
        fit = s['fit']; sl = fit['slope_ns_per_mm']
        if np.isfinite(sl) and sl != 0:
            pl = np.array([pos.min(), pos.max()])
            tl = fit['earliest_time_ns'] + sl * (pl - fit['mesh_position_mm'])
            ax.plot(tl - t.min(), pl, 'r--', lw=1.5, zorder=2)
        ax.set_xlabel('drift time − t0 [ns]'); ax.set_ylabel(f'{lab} position [mm]')
        th = s.get('theta_deg', np.nan)
        ax.set_title(f'{lab}: n={fit["n_strips"]}  ext={fit["extent_mm"]:.1f}mm  '
                     f'dur={fit["duration_ns"]:.0f}ns  r={s["r"]:.2f}\n'
                     f'θ_reco={th:.1f}°  (|corr| gradient)', fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label='amp [ADC]')
    fig.suptitle(f'{RUN}/{block} — event {ev} — detector A micro-TPC track',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, f'microtpc_evt_{ev:06d}.png')


def summary(all_segs, block, out_dir):
    dur = np.array([s['fit']['duration_ns'] for s in all_segs], float)
    r = np.array([abs(s['r']) if np.isfinite(s['r']) else np.nan for s in all_segs], float)
    trk = np.array([s['is_track'] for s in all_segs], bool)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(dur[~trk], bins=np.linspace(0, 600, 40), color='gray',
                 alpha=0.7, label='point-like / non-track')
    axes[0].hist(dur[trk], bins=np.linspace(0, 600, 40), color='crimson',
                 alpha=0.8, label='micro-TPC track')
    axes[0].axvline(bc.BENCH_T_SAT_NS, color='k', ls=':', label=f'bench T_sat {bc.BENCH_T_SAT_NS:.0f}ns')
    axes[0].set_xlabel('cluster drift duration [ns]'); axes[0].set_ylabel('compact clusters')
    axes[0].legend(fontsize=8); axes[0].set_title('Drift-time gradient (cosmics fill to T_sat)')
    axes[1].scatter(dur, r, s=10, c=np.where(trk, 'crimson', 'gray'))
    axes[1].axhline(TRK_R_MIN, color='r', ls='--'); axes[1].axvline(TRK_DUR_MIN, color='r', ls='--')
    axes[1].set_xlabel('duration [ns]'); axes[1].set_ylabel('|corr(pos,time)|')
    axes[1].set_title('Track cut (red box = reconstructable)')
    fig.suptitle(f'{RUN}/{block} — detector A: micro-TPC tracks vs point deposits\n'
                 f'{trk.sum()}/{len(all_segs)} compact clusters have a real gradient',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out_dir, 'summary.png')


def process(block, cfg, det, model):
    feu_axis = _plane_of(cfg)
    allf = sorted(det.feu_map.keys())
    df = load_hits_tot(RUN, block, allf)
    if df is None or df.empty:
        print(f'  [skip] {block}'); return
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    xy = np.array([det.map_hit(int(f), int(ch)) or (np.nan, np.nan)
                   for f, ch in zip(df['feu'], df['channel'])])
    df = df.assign(x_mm=xy[:, 0], y_mm=xy[:, 1])
    df = df[df['amplitude'] >= THR]

    segx = collect(df, det, 'x_mm')
    segy = collect(df, det, 'y_mm')
    if segx:
        reco_angles(segx, model, 'x')
    if segy:
        reco_angles(segy, model, 'y')
    all_segs = segx + segy
    ntrk = sum(s['is_track'] for s in all_segs)
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', 'run30_microtpc', block)
    print(f'  {block}: {len(all_segs)} compact clusters (x={len(segx)} y={len(segy)}), '
          f'{ntrk} with a micro-TPC gradient (dur>{TRK_DUR_MIN:.0f}ns & |r|>{TRK_R_MIN})')
    if all_segs:
        summary(all_segs, block, out_dir)

    # events with a track in BOTH planes -> 3D micro-TPC track candidates
    tx = {s['ev']: s for s in segx if s['is_track']}
    ty = {s['ev']: s for s in segy if s['is_track']}
    both = sorted(set(tx) & set(ty))
    only = sorted((set(tx) | set(ty)) - (set(tx) & set(ty)))
    print(f'    both-plane 3D track candidates: {both}')
    print(f'    single-plane track candidates: {only}')
    for ev in both:
        render_track(ev, tx[ev], ty[ev], block, out_dir)
    for ev in only[:12]:
        render_track(ev, tx.get(ev), ty.get(ev), block, out_dir)
    print(f'    -> {out_dir}')


if __name__ == '__main__':
    import re as _re
    args = sys.argv[1:]
    if args and _re.fullmatch(r'run_\d+', args[0]):
        RUN = args[0]                      # allow: run30_microtpc.py run_33 sub1 sub2
        args = args[1:]
    blocks = args or DEFAULT_BLOCKS
    cfg = load_config(BASE_PATH, RUN)
    sm = Mx17StripMap(MAP_CSV_PATH)
    det = Detector(DET, [d for d in cfg['detectors'] if d['name'] == DET][0], sm)
    model = mt.load_model(MODEL_PATH)
    for b in blocks:
        process(b, cfg, det, model)
    print('done')
