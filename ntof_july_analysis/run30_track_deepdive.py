#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/run30_track_deepdive.py

Deep-dive event displays + full QA for the confirmed micro-TPC track candidates
in run_30 detector A (= bench det3).  A genuine 3-D micro-TPC track needs a
resolvable drift-time gradient COINCIDENT in BOTH readout planes (X and Y) of
this ONE Micromegas — that is the bar, and evt 1017 / evt 162 (scintOff_A700_00)
clear it.  This tool zooms in on those events (any event list may be passed) and
produces, per event, in ITS OWN directory:

  tpc_display.png   the primary display.  Drift TIME on the vertical axis,
                    strip POSITION on the horizontal — two panels (X | Y) that
                    SHARE the vertical time axis, so a real coincidence shows the
                    two plane gradients occupying the same time band.  Anchored
                    fit line + reconstructed angle per plane.
  hitmap_2d.png     the (x,y) spatial view — proof it is one localized cluster,
                    not a whole-plane flash; fired X strips vertical, Y horizontal.
  diagnostics.png   amplitude vs position and amplitude vs drift-time per plane.
  QA.txt            the full numeric QA: coincidence, per-plane cluster + anchored
                    fit + hits6 features + reconstructed angle + charge balance +
                    quality flags, and the raw cluster hit table.

Angles are restandardized on the whole block's compact-cluster population (as in
run30_microtpc.py), so the numbers here match that first-look driver.

Output -> {ANALYSIS_DIR}July_HV_Scan/run30_track_deepdive/<block>/evt_<id>/

Run:  .venv/bin/python ntof_july_analysis/run30_track_deepdive.py [run_N] [block] [ev ...]
      (defaults: run_30 scintOff_A700_00 1017 162)
"""
import os
import sys
import re

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
    BASE_PATH, ANALYSIS_DIR, MAP_CSV_PATH, load_config, _save,
)
import run30_microtpc as mtpc                          # noqa: E402
from run30_scint_neutron import event_times, fmt_dt    # noqa: E402

DET = 'mx17_A'
DET_MX = 3                                             # detector A = bench det3
DEFAULT_BLOCK = 'scintOff_A700_00'
DEFAULT_EVENTS = [1017, 162]

# illustrative drift speeds for the depth read-out (Ar/iso 80/20 @ ~267 V/cm is
# NOT in the garfield table set — only the 95/5 bench gas is; so the column depth
# is quoted as a RANGE and explicitly flagged un-calibrated).
V_DRIFT_LO_UM_NS = 20.0
V_DRIFT_HI_UM_NS = 30.0
AMP_CLIP = (150, 3000)


def build_df(run, block, det):
    """Load detector-A hits with time_over_threshold, map to x_mm/y_mm, threshold."""
    allf = sorted(det.feu_map.keys())
    df = mtpc.load_hits_tot(run, block, allf)
    if df is None or df.empty:
        return None
    df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
    xy = np.array([det.map_hit(int(f), int(ch)) or (np.nan, np.nan)
                   for f, ch in zip(df['feu'], df['channel'])])
    df = df.assign(x_mm=xy[:, 0], y_mm=xy[:, 1])
    return df[df['amplitude'] >= mtpc.THR].copy()


def _fit_line_xy(fit, pos):
    """Return (pos_line, time_line) for the anchored fit drawn as time(pos)."""
    sl = fit['slope_ns_per_mm']
    pl = np.array([pos.min(), pos.max()])
    tl = fit['earliest_time_ns'] + sl * (pl - fit['mesh_position_mm'])
    return pl, tl


def render_tpc(ev, sx, sy, t0, block, run, out_dir):
    """Primary display: time on Y (shared), position on X, X | Y panels.
    Each plane gets its OWN amplitude colour scale + a distinct colormap, so a
    low-amplitude plane is not washed out by a high-amplitude one."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.6), sharey=True)
    for ax, s, lab, col, cmap in [(axes[0], sx, 'X', 'FEU 3', 'viridis'),
                                  (axes[1], sy, 'Y', 'FEU 4', 'inferno')]:
        if s is None:
            ax.text(0.5, 0.5, f'no {lab}-plane\ntrack', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xlabel(f'{lab} position [mm]')
            continue
        pos, t = s['pos'], s['t'] - t0
        vmn, vmx = float(s['a'].min()), float(s['a'].max())
        if vmx <= vmn:
            vmx = vmn + 1.0
        sc = ax.scatter(pos, t, c=s['a'], cmap=cmap, s=55,
                        vmin=vmn, vmax=vmx, zorder=3,
                        edgecolor='k', linewidth=0.4)
        fit = s['fit']
        if np.isfinite(fit['slope_ns_per_mm']) and fit['slope_ns_per_mm'] != 0:
            pl, tl = _fit_line_xy(fit, pos)
            ax.plot(pl, tl - t0, 'r--', lw=1.6, zorder=2,
                    label=f'anchored fit\nθ={s.get("theta_deg", np.nan):.1f}°')
        # mark the earliest-strip anchor (the mesh crossing point)
        ax.plot(fit['mesh_position_mm'], fit['earliest_time_ns'] - t0, 'r*',
                ms=15, zorder=4, label='mesh anchor')
        ax.set_xlabel(f'{lab} position [mm]')
        ax.set_title(f'{lab}-plane ({col})   n={fit["n_strips"]}  '
                     f'ext={fit["extent_mm"]:.1f} mm\n'
                     f'dur={fit["duration_ns"]:.0f} ns   r={s["r"]:+.2f}   '
                     f'θ={s.get("theta_deg", np.nan):+.1f}°', fontsize=10)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02,
                     label=f'{lab} amp [ADC] (own scale)')
    axes[0].set_ylabel('drift time − t0 [ns]   (deeper into gap ↑)')
    fig.suptitle(f'{run}/{block} — detector A — event {ev}\n'
                 f'micro-TPC track: drift-time gradient in BOTH planes of one chamber',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, 'tpc_display.png')


def render_hitmap(ev, sx, sy, det, block, run, out_dir):
    """(x,y) spatial view — localized coincident cluster."""
    fig, ax = plt.subplots(figsize=(6.6, 6.8))
    norm = plt.Normalize(*AMP_CLIP); cmap = plt.get_cmap('viridis')
    xs = sx['pos'] if sx is not None else np.array([])
    ax_ = sx['a'] if sx is not None else np.array([])
    ys = sy['pos'] if sy is not None else np.array([])
    ay_ = sy['a'] if sy is not None else np.array([])
    for x, a in zip(xs, ax_):
        ax.axvline(x, color=cmap(norm(a)), lw=2.0, alpha=0.9)
    for y, a in zip(ys, ay_):
        ax.axhline(y, color=cmap(norm(a)), lw=2.0, alpha=0.9)
    if len(xs) and len(ys):
        ax.plot(np.mean(xs), np.mean(ys), 'r+', ms=18, mew=2)
    allx = [px for (a, _, _), (px, py) in det.strip_map.map.items() if a == 'x']
    ally = [py for (a, _, _), (px, py) in det.strip_map.map.items() if a == 'y']
    ax.set_xlim(min(allx), max(allx)); ax.set_ylim(min(ally), max(ally))
    ax.set_aspect('equal'); ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label='amplitude [ADC]')
    if len(xs) and len(ys):
        cx, cy = np.mean(xs), np.mean(ys)
        axin = ax.inset_axes([0.60, 0.60, 0.38, 0.38])
        for x, a in zip(xs, ax_):
            axin.axvline(x, color=cmap(norm(a)), lw=2.5)
        for y, a in zip(ys, ay_):
            axin.axhline(y, color=cmap(norm(a)), lw=2.5)
        axin.set_xlim(cx - 25, cx + 25); axin.set_ylim(cy - 25, cy + 25)
        axin.set_aspect('equal'); axin.tick_params(labelsize=6)
        axin.set_title('zoom ±25 mm', fontsize=7)
        for sp in axin.spines.values():
            sp.set_edgecolor('red')
    ax.set_title(f'{run}/{block} — detector A — event {ev}\nspatial hit map '
                 f'(nx={len(xs)}, ny={len(ys)})', fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, 'hitmap_2d.png')


def render_diagnostics(ev, sx, sy, t0, block, run, out_dir):
    """amp vs position and amp vs drift-time, per plane."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, s, lab in [(0, sx, 'X'), (1, sy, 'Y')]:
        axp, axt = axes[row]
        if s is None:
            for a in (axp, axt):
                a.text(0.5, 0.5, f'no {lab}-plane', ha='center', va='center',
                       transform=a.transAxes, color='gray')
            continue
        order = np.argsort(s['pos'])
        axp.stem(s['pos'][order], s['a'][order])
        axp.set_xlabel(f'{lab} position [mm]'); axp.set_ylabel('amplitude [ADC]')
        axp.set_title(f'{lab}: amplitude vs position'); axp.grid(alpha=0.3)
        sct = axt.scatter(s['t'] - t0, s['a'], c=s['pos'], cmap='plasma', s=45)
        axt.set_xlabel('drift time − t0 [ns]'); axt.set_ylabel('amplitude [ADC]')
        axt.set_title(f'{lab}: amplitude vs drift time'); axt.grid(alpha=0.3)
        fig.colorbar(sct, ax=axt, fraction=0.046, pad=0.02,
                     label=f'{lab} position [mm]')
    fig.suptitle(f'{run}/{block} — detector A — event {ev} — amplitude diagnostics',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, 'diagnostics.png')


def _plane_qa(s, lab):
    """Text block for one plane's QA."""
    if s is None:
        return [f'  {lab}-plane: NO reconstructable cluster / no gradient']
    fit, feat = s['fit'], s['feat']
    L = [f'  {lab}-plane:']
    L.append(f'    cluster : n_strips={fit["n_strips"]}  n_dropped={fit["n_dropped"]}  '
             f'extent={fit["extent_mm"]:.2f} mm  q_sum={fit["q_sum"]:.0f} ADC')
    L.append(f'    anchor  : mesh_position={fit["mesh_position_mm"]:.2f} mm  '
             f'earliest_t={fit["earliest_time_ns"]:.1f} ns  latest_t={fit["latest_time_ns"]:.1f} ns')
    L.append(f'    gradient: duration={fit["duration_ns"]:.1f} ns  '
             f'slope={fit["slope_ns_per_mm"]:+.2f} ns/mm  S={fit["S_um_ns"]:+.1f} um/ns')
    L.append(f'    red_chi2={fit["red_chi2"]:.3g}  (bench convention: amplitude-'
             f'weighted, NOT physical σ — a RELATIVE metric, large is normal; '
             f'use pearson r for linearity)')
    L.append(f'    linearity: pearson r(pos,time)={s["r"]:+.3f}   '
             f'is_track={s["is_track"]} (dur>{mtpc.TRK_DUR_MIN:.0f}ns & |r|>{mtpc.TRK_R_MIN})')
    # depth read-out (flagged: gas not in garfield set)
    dlo = fit['duration_ns'] * V_DRIFT_LO_UM_NS / 1000.0
    dhi = fit['duration_ns'] * V_DRIFT_HI_UM_NS / 1000.0
    L.append(f'    depth   : ~{dlo:.1f}-{dhi:.1f} mm of the 30 mm gap '
             f'(v={V_DRIFT_LO_UM_NS:.0f}-{V_DRIFT_HI_UM_NS:.0f} um/ns ASSUMED — '
             f'Ar/iso 80/20 not in garfield tables, UNCALIBRATED)')
    if feat:
        L.append(f'    hits6   : a_lead={feat["a_lead"]:.0f}  tot_lead={feat["tot_lead"]:.0f}  '
                 f'n_raw={feat["n_raw"]}  q_frac={feat["q_frac"]:.3f}  '
                 f'a_asym={feat["a_asym"]:.3f}  t_delay={feat["t_delay"]:.1f} ns')
        L.append(f'    signed  : a_asym_sgn={feat["a_asym_sgn"]:+.3f}  '
                 f't_asym_sgn={feat.get("t_asym_sgn", np.nan):+.1f} ns  '
                 f'pos_lead={feat["pos_lead_mm"]:.2f} mm')
    else:
        L.append(f'    hits6   : NO valid lead strip (a_lead < {bc.A_LEAD_MIN:.0f}) — '
                 f'features unavailable, angle from fallback sign only')
    th = s.get('theta_deg', np.nan)
    L.append(f'    ANGLE   : tan_reg={s.get("tan_reg", np.nan):+.3f}  '
             f'theta_reco={th:+.2f} deg  (frozen det3 hits6 model, restd on block pop.)')
    # flags
    flags = []
    if fit['mesh_position_mm'] < bc.EDGE_ANGLE_FIDUCIAL_MM or \
       fit['mesh_position_mm'] > (bc.N_STRIPS_PLANE * bc.PITCH_MM - bc.EDGE_ANGLE_FIDUCIAL_MM):
        flags.append(f'EDGE(<{bc.EDGE_ANGLE_FIDUCIAL_MM:.0f}mm from an edge: angle unreliable)')
    if s['a'].max() >= 3500:
        flags.append('SATURATION?')
    if fit['n_dropped'] > 0:
        flags.append(f'n_dropped={fit["n_dropped"]} (under-split?)')
    L.append(f'    flags   : {", ".join(flags) if flags else "none"}')
    return L


def qa_report(ev, sx, sy, dt_info, block, run, out_dir):
    """Full text QA -> QA.txt (also returned as a string for stdout)."""
    L = []
    L.append('=' * 72)
    L.append(f'  MICRO-TPC TRACK QA   {run}/{block}   event {ev}')
    L.append(f'  detector A = mx17_3 (bench det3)   FEU 3 = X, FEU 4 = Y')
    L.append('=' * 72)
    if dt_info is not None:
        dt_s, burst = dt_info
        L.append(f'  timing : Δt = {fmt_dt(dt_s)} after the flash  (burst {burst}) '
                 f'{"[FLASH ITSELF]" if dt_s == 0 else "[post-flash]"}')
    L.append(f'  config : gas Ar/iso 80/20   drift gap 30 mm   drift HV ~800 V '
             f'(E ~267 V/cm)   sample 20 ns')
    L.append('')

    # coincidence between the two planes (the 3-D track requirement)
    L.append('  COINCIDENCE (the both-plane requirement for a 3-D track):')
    if sx is not None and sy is not None:
        ex, ey = sx['fit']['earliest_time_ns'], sy['fit']['earliest_time_ns']
        lx, ly = sx['fit']['latest_time_ns'], sy['fit']['latest_time_ns']
        iou = mt.time_iou(ex, lx, ey, ly)
        L.append(f'    earliest_t: X={ex:.1f} ns  Y={ey:.1f} ns  |Δt0|={abs(ex-ey):.1f} ns '
                 f'({"OK <150 ns" if abs(ex-ey) < 150 else "WIDE — check"})')
        L.append(f'    time-span IoU(X,Y) = {iou:.2f}   '
                 f'(overlapping drift windows ⇒ same particle)')
        qx, qy = sx['fit']['q_sum'], sy['fit']['q_sum']
        f = qx / (qx + qy)
        fb = bc.F_BALANCE[DET_MX]
        pull = abs(f - fb['med']) / fb['s68']
        L.append(f'    charge balance f=qX/(qX+qY)={f:.3f}  vs det3 {fb["med"]:.3f}±{fb["s68"]:.2f}  '
                 f'pull={pull:.1f} ({"compatible" if pull < 2 else "off — check pairing"})')
        thx, thy = sx.get('theta_deg', np.nan), sy.get('theta_deg', np.nan)
        L.append(f'    reco angles: θx={thx:+.1f}°  θy={thy:+.1f}°  '
                 f'(independent projections of the 3-D track; both real ⇒ 3-D track)')
    else:
        have = 'X' if sx is not None else ('Y' if sy is not None else 'neither')
        L.append(f'    ONLY {have}-plane has a gradient — this is a single-plane '
                 f'segment, NOT a confirmed 3-D micro-TPC track.')
    L.append('')

    L.append('  PER-PLANE RECONSTRUCTION:')
    L += _plane_qa(sx, 'X')
    L += _plane_qa(sy, 'Y')
    L.append('')

    # raw cluster hit tables
    for s, lab in [(sx, 'X'), (sy, 'Y')]:
        if s is None:
            continue
        L.append(f'  {lab}-plane cluster hits ({s["fit"]["n_strips"]} strips, position-sorted):')
        L.append(f'    {"pos[mm]":>9} {"time[ns]":>9} {"amp[ADC]":>9} {"tot[ns]":>8}')
        o = np.argsort(s['pos'])
        for i in o:
            L.append(f'    {s["pos"][i]:9.2f} {s["t"][i]:9.1f} {s["a"][i]:9.0f} {s["q"][i]:8.1f}')
        L.append('')

    text = '\n'.join(L)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'QA.txt'), 'w') as f:
        f.write(text + '\n')
    return text


def process(run, block, events):
    cfg = load_config(BASE_PATH, run)
    sm = Mx17StripMap(MAP_CSV_PATH)
    det = Detector(DET, [d for d in cfg['detectors'] if d['name'] == DET][0], sm)
    model = mt.load_model(mtpc.MODEL_PATH)
    mtpc.RUN = run                                  # so collect()/labels use it

    df = build_df(run, block, det)
    if df is None or df.empty:
        print(f'  [skip] {block} — no hits'); return

    # reconstruct the WHOLE block so restandardization matches run30_microtpc
    segx = mtpc.collect(df, det, 'x_mm')
    segy = mtpc.collect(df, det, 'y_mm')
    if segx:
        mtpc.reco_angles(segx, model, 'x')
    if segy:
        mtpc.reco_angles(segy, model, 'y')
    bx = {s['ev']: s for s in segx}
    by = {s['ev']: s for s in segy}
    try:
        times = event_times(block) if run == 'run_30' else {}
    except Exception:
        times = {}

    root = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', 'run30_track_deepdive', block)
    print(f'\n{run}/{block}: {len(segx)} X / {len(segy)} Y compact clusters in block; '
          f'deep-diving events {events}')
    for ev in events:
        sx, sy = bx.get(ev), by.get(ev)
        if sx is None and sy is None:
            print(f'  evt {ev}: no compact cluster in either plane — skipped'); continue
        out_dir = os.path.join(root, f'evt_{ev:06d}')
        # common t0 across both planes so the shared time axis shows coincidence
        t0s = [s['t'].min() for s in (sx, sy) if s is not None]
        t0 = min(t0s)
        render_tpc(ev, sx, sy, t0, block, run, out_dir)
        render_hitmap(ev, sx, sy, det, block, run, out_dir)
        render_diagnostics(ev, sx, sy, t0, block, run, out_dir)
        text = qa_report(ev, sx, sy, times.get(ev), block, run, out_dir)
        print('\n' + text)
        print(f'  -> {out_dir}  (tpc_display.png, hitmap_2d.png, diagnostics.png, QA.txt)')


if __name__ == '__main__':
    args = sys.argv[1:]
    run = 'run_30'
    if args and re.fullmatch(r'run_\d+', args[0]):
        run = args[0]; args = args[1:]
    block = DEFAULT_BLOCK
    if args and not args[0].isdigit():
        block = args[0]; args = args[1:]
    events = [int(a) for a in args] if args else DEFAULT_EVENTS
    process(run, block, events)
    print('\ndone')
