#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run79_event_display.py -- event displays for the PRELIMINARY run_145
waveform-first tracks, drawn on the current Geant4 geometry.

Three views of one event, all of the same fit:

  <evt>_waveforms.png    what the reconstruction actually works on: the X and
                         Y waveform windows (strip x time) beside the forward
                         model that was fitted to them, plus the fitted charge
                         column.
  <evt>_projections.png  the track in the global frame -- top-down, both side
                         views, and an arm close-up -- with the SiPM bar group
                         and the plastic bar that fired the trigger drawn
                         opaque.
  <evt>_3d.png           the same track through the 3D detector model
                         (pyvista), geometry as built in MX17_Full_Geant.

"Golden" events (`pick`): both planes fitted and quality-ok, an n_TOF wall AND
plastic tag in the SAME arm, the track pointing back through the He-3 capsule,
and its extrapolation landing inside both the wall group and the plastic bar
that actually fired -- i.e. events where the chamber, the trigger and the
target all agree. That is a display selection, not an efficiency: see
RUN79_PRELIM_2026-07-30.md for what is and is not established.

Usage:
    python -m ntof_tracking.run79_event_display pick [--n 15]
    python -m ntof_tracking.run79_event_display make --best 4
    python -m ntof_tracking.run79_event_display make --event 1234567
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# the fit has to be reproduced under the same configuration that produced the
# table (WFT_* are read at import time inside wft.reco / wft.model)
for _k, _v in (('WFT_MODEL_FRAC', '0.03'), ('WFT_PRESCAN', '1'),
               ('WFT_CHI2DOF_BAD', '250'), ('WFT_PAIR_SELECT', '1'),
               ('OMP_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1')):
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ntof_tracking.reco import geometry as geo
from ntof_tracking.reco import display as rdisp
from ntof_tracking.run79_merge_prelim import (
    TARGET_TO_STRIPS, STRIPS_TO_WALL, STRIPS_TO_PLASTIC, STRIP_MAP_HALF,
    PINWHEEL, N_WALL_SEG, wall_segment_u, plastic_bar_u, OUT_BASE)

MERGED = (OUT_BASE / 'stat090_0000' / 'mx17_A' / 'merged_prelim.parquet')
INK, MUTED = '#222222', '#888888'
TRACK_C = '#d62728'
HOT = '#ff7f0e'

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
})


# ---------------------------------------------------------------- geometry
def transforms(run='run_145'):
    cfg = json.load(open(f'/media/dylan/data/x17/beam_july/runs/{run}/run_config.json'))
    return geo.detector_transforms(cfg)


def track_line(row, arm, tr):
    """The fitted track as a global-frame line.

    The fit gives, per plane, a position at the mesh and tan = d(pos)/d(depth)
    with depth increasing TOWARD the target. So the local point at depth w is
    (u0 + tan_x w, v0 + tan_y w, -w) and the whole track is one straight line;
    w < 0 is the outward extrapolation past the strips (wall, plastics).
    """
    u0 = float(row['x_p0']) - STRIP_MAP_HALF
    v0 = float(row['y_p0']) - STRIP_MAP_HALF
    tx, ty = float(row['x_tan_theta']), float(row['y_tan_theta'])

    def at(w):
        w = np.asarray(w, float)
        return tr.local_to_global(u0 + tx * w, v0 + ty * w, w)

    p0, p1 = at(0.0), at(100.0)
    d = p1 - p0
    d /= np.linalg.norm(d)
    return dict(at=at, u0=u0, v0=v0, tan_x=tx, tan_y=ty, point=p0, dir=d)


def event_geometry(row, arm, tr, mapping='descending'):
    """Everything the displays and the selection need about one event."""
    L = track_line(row, arm, tr)
    at = L['at']
    g = dict(L)
    # --- crossing points of the scintillators, in the arm's own (u, v)
    for tag, depth in (('wall', STRIPS_TO_WALL), ('plastic', STRIPS_TO_PLASTIC[arm])):
        g[f'u_{tag}'] = L['u0'] - depth * L['tan_x'] - PINWHEEL[arm]   # structure
        g[f'v_{tag}'] = L['v0'] - depth * L['tan_y']
        g[f'p_{tag}'] = at(-depth)
    # --- which elements fired, and does the track go through them
    seg = int((row['wal_detn'] - 1) // 2) if np.isfinite(row['wal_detn']) else -1
    g['seg'] = seg
    if seg >= 0:
        grp = (N_WALL_SEG - 1 - seg) if mapping == 'descending' else seg
        g['wall_group'] = grp
        g['wall_bars'] = [grp * 4 + 1 + i for i in range(4)]
        lo, hi = wall_segment_u(grp)
        g['wall_span'] = (lo, hi)
        g['in_wall'] = bool(lo <= g['u_wall'] <= hi
                            and abs(g['v_wall']) <= geo.SIPM_HALF_V)
    else:
        g['wall_group'], g['wall_bars'] = None, []
        g['wall_span'], g['in_wall'] = None, False
    dn = int(row['pss_detn']) if np.isfinite(row['pss_detn']) else 0
    g['pss_detn'] = dn
    if dn:
        lo, hi = plastic_bar_u(dn, arm, mapping)
        g['plastic_span'] = (lo, hi)
        # detn 1 sits at positive u under the descending order (README + fig 2)
        g['plastic_name'] = 'plastic R' if (lo + hi) / 2 > 0 else 'plastic L'
        g['in_plastic'] = bool(lo <= g['u_plastic'] <= hi
                               and abs(g['v_plastic']) <= geo.PLASTIC_HALF_V)
    else:
        g['plastic_span'], g['plastic_name'], g['in_plastic'] = None, None, False
    # --- pointing back at the source: closest approach to the beam axis
    p, d = L['point'], L['dir']
    g['dca_beam_mm'] = geo.line_line_dist(np.zeros(3), geo.V_HAT, p, d)
    b = float(d @ geo.V_HAT)
    den = 1.0 - b * b
    g['beam_y_mm'] = float((float(p @ geo.V_HAT) - b * float(p @ d)) / den) \
        if den > 1e-9 else np.nan
    # inside the He-3 gas: radius < 10 mm and within the capsule's y span
    g['from_capsule'] = bool(g['dca_beam_mm'] < geo.HE3_R_MAX
                             and geo.HE3_GAS_Y[0] <= g['beam_y_mm'] <= geo.HE3_GAS_Y[-1])
    # the drift depth at which the track is closest to the beam axis: how far
    # back the inward extrapolation is worth drawing
    m = at(1.0) - at(0.0)
    q = at(0.0)
    den = m[0] ** 2 + m[2] ** 2
    g['w_closest'] = float(-(q[0] * m[0] + q[2] * m[2]) / den) if den > 1e-12 \
        else 0.0
    g['theta_x_deg'] = float(np.degrees(np.arctan(L['tan_x'])))
    g['theta_y_deg'] = float(np.degrees(np.arctan(L['tan_y'])))
    return g


# --------------------------------------------------------------- selection
def pick_events(d, arm='A', n=15, mapping='descending', tr=None):
    """Rank display candidates. Everything here is a LOOKS-GOOD cut."""
    tr = tr or transforms()[f'mx17_{arm}']
    m = (d['x_ok'] & d['y_ok'] & d['x_quality_ok'] & d['y_quality_ok']
         & np.isfinite(d['wal_dt']) & np.isfinite(d['pss_dt'])
         & (d['x_tan_theta'].abs() < 0.9) & (d['y_tan_theta'].abs() < 0.9)
         & (d['x_n_strips'].between(8, 45)) & (d['y_n_strips'].between(8, 45)))
    rows = []
    for _, r in d[m].iterrows():
        g = event_geometry(r, arm, tr, mapping)
        if not (g['in_wall'] and g['in_plastic'] and g['from_capsule']):
            continue
        cx = float(r['x_chi2'] / max(r['x_dof'], 1))
        cy = float(r['y_chi2'] / max(r['y_dof'], 1))
        # prefer: points at the capsule centre, clean fits, and an angle you
        # can actually see (a track along the normal is a boring picture)
        score = (-g['dca_beam_mm'] / 10.0 - 0.5 * np.log10(max(cx, 1))
                 - 0.5 * np.log10(max(cy, 1))
                 + min(abs(g['theta_x_deg']), 30) / 15.0)
        rows.append(dict(event_id=int(r['event_id']), score=float(score),
                         dca_beam_mm=g['dca_beam_mm'], beam_y_mm=g['beam_y_mm'],
                         theta_x_deg=g['theta_x_deg'], theta_y_deg=g['theta_y_deg'],
                         chi2dof_x=cx, chi2dof_y=cy,
                         n_strips_x=int(r['x_n_strips']), n_strips_y=int(r['y_n_strips']),
                         seg=g['seg'], pss=g['pss_detn'],
                         u_wall=g['u_wall'], u_plastic=g['u_plastic'],
                         t_ms=float(r['t_since_flash_ns']) / 1e6,
                         bunch=int(r['BunchNumber'])))
    out = pd.DataFrame(rows).sort_values('score', ascending=False)
    return out.head(n).reset_index(drop=True)


# ------------------------------------------------------------- the waveforms
def refit_event(event_id, arm='A', run='run_145', sub_run='stat090_0000'):
    """Re-run the fit for one event and hand back the winning window, the fit
    and the NNLS charge column -- the pieces the waveform display needs and
    the summary table does not keep."""
    from ntof_tracking import wft_beam as wb
    from wft import io as wio, reco as wreco, model as wm
    from wft.calib import CalibrationBundle

    cfg = wb.beam_config(arm, run, sub_run)
    bundle = str(Path(cfg.OUT_BASE) / 'calib_bundle_prelim')
    cal = CalibrationBundle.load(bundle)
    wm.use_calibration(cal)
    pos_maps = wio.strip_position_map(cfg)

    for tag in wb.subrun_tags(cfg):
        hp = wb.hits_file_for_tag(cfg, tag)
        if hp is None:
            continue
        hits = wb.read_hits_tag(hp, (cfg.MX17_FEU_X, cfg.MX17_FEU_Y))
        if event_id not in set(hits['eventId'].unique()):
            continue
        seeds = wb.seeds_from_hits_beam(hits, pos_maps, cfg.MX17_FEU_X,
                                        cfg.MX17_FEU_Y)
        del hits
        if event_id not in seeds:
            break
        payloads = wb._windows_for_tag(cfg, tag, pos_maps, seeds, {event_id}, 3)
        if not payloads:
            break
        _eid, wins, sds, _nh, _sp, _fd = payloads[0]
        out = {'tag': tag, 'cal': cal, 'planes': {}}
        for plane in ('x', 'y'):
            cand = wins.get(plane) or []
            best, best_key, best_P = None, None, None
            for i, P in enumerate(cand):
                f = wreco.fit_plane(P, plane, cal)
                if f is None:
                    continue
                plaus, dchi2 = wreco._candidate_score(P, plane, f)
                # frozen HEAD dropped _cand_key; fit_plane_candidates ranks by
                # (plausible, dchi2) — reproduce that key here
                key = (1 if plaus else 0, dchi2)
                if best_key is None or key > best_key:
                    best, best_key, best_P = f, key, P
            if best is None:
                continue
            # the charge column at the fitted parameters (fit_plane keeps only
            # its summary statistics)
            r = wm.fit_plane_raw(best_P, plane, best.p0, best.w, best.t0,
                                 hyper=cal.hyper, fix_p0w=(best.p0, best.w))
            W, noise, pos, sat = wm.prep_plane(best_P, plane)
            model = wm.model_waveforms(plane, pos, best.p0, best.w, best.t0,
                                       r['q'], cal.hyper)
            out['planes'][plane] = dict(fit=best, W=W, model=model, pos=pos,
                                        q=np.asarray(r['q'], float),
                                        ch=np.asarray(best_P['ch'], int))
        return out
    return None


def fig_waveforms(ev, event_id, arm, out_path, v_drift=36.0):
    """Data vs forward model, per plane, plus the fitted charge column.

    This is the picture behind "waveform-first": nothing here is a hit time.
    The fit adjusts one track (position at the mesh + transverse speed) and a
    non-negative charge-vs-depth profile until the MODEL panel looks like the
    DATA panel, on every strip at once.
    """
    from wft import model as wm
    planes = ev['planes']
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4),
                             gridspec_kw=dict(width_ratios=[1, 1, .62]))
    for ri, plane in enumerate(('x', 'y')):
        P = planes.get(plane)
        if P is None:
            for ax in axes[ri]:
                ax.axis('off')
            continue
        f, W, model, pos = P['fit'], P['W'], P['model'], P['pos']
        t = np.arange(W.shape[1]) * wm.SNS      # 60 ns sampling
        vmax = float(np.nanpercentile(W, 99.5)) or 1.0
        order = np.argsort(pos)
        for ci, (M, lab) in enumerate(((W, 'data'), (model, 'forward model'))):
            ax = axes[ri][ci]
            im = ax.pcolormesh(pos[order], t, M[order].T, cmap='magma',
                               vmin=0, vmax=vmax, shading='nearest')
            # the fitted track: charge that arrives at time t was born at
            # depth (t - t0) v and sits at p0 + w (t - t0)
            tt = np.linspace(f.t0, f.t0 + max(f.q_uend, 200.0), 50)
            ax.plot(f.p0 + f.w * (tt - f.t0), tt, '-', color='#00e5ff', lw=2,
                    alpha=.9, label='fitted track' if ci == 0 else None)
            ax.plot([f.p0], [f.t0], 'o', ms=6, mfc='none', mec='#00e5ff', mew=2)
            ax.set_xlabel(f'{plane.upper()} strip position [mm]')
            ax.set_ylim(t.max() + wm.SNS / 2, -wm.SNS / 2)   # data fills the axes
            ax.set_facecolor('k')
            ax.text(.025, .965, (f'{plane.upper()} plane — ' if ci == 0 else '')
                    + lab.upper(), transform=ax.transAxes, va='top',
                    color='w', fontsize=9.5, fontweight='bold')
            if ci == 0:
                ax.set_ylabel('time in the window [ns]')
                ax.legend(frameon=False, fontsize=8, loc='lower right',
                          labelcolor='w')
            else:
                ax.tick_params(labelleft=False)
        fig.colorbar(im, ax=axes[ri][1], fraction=.046, pad=.02,
                     label='amplitude [ADC]')
        ax = axes[ri][2]
        q = P['q']
        u = wm.UK[:len(q)]
        depth = u * v_drift / 1e3            # ns -> mm at the assumed v
        ax.step(depth, q / max(q.max(), 1e-9), where='mid', color=TRACK_C, lw=2)
        ax.fill_between(depth, 0, q / max(q.max(), 1e-9), step='mid',
                        color=TRACK_C, alpha=.25)
        ax.set_xlabel('depth below the mesh [mm]')
        ax.set_ylabel('fitted charge (norm.)')
        ax.grid(alpha=.25)
        ax.set_title(f'charge column: {f.q_uend:.0f} ns '
                     f'= {f.q_uend * v_drift / 1e3:.1f} mm', fontsize=8.5,
                     loc='left')
        ax.text(.98, .95, f'p0 = {f.p0:.1f} mm\ntan = {f.tan_theta:+.3f}'
                          f'\nchi2/dof = {f.chi2 / max(f.dof, 1):.0f}'
                          f'\n{f.n_strips} strips',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color=MUTED)
    fig.suptitle(f'run_145 / mx17_{arm}  event {event_id}: the waveforms and the '
                 f'forward model fitted to them   [PRELIMINARY]',
                 ha='left', x=0.01, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out_path)


# ------------------------------------------------------------- projections
VIEWS = [
    ('top-down   (screen x = Z, y = X;  beam out of the page)', (2, 0),
     {'A': 'topdown', 'B': 'topdown', 'C': 'topdown', 'D': 'topdown'}),
    ('side  Z-Y   (arms A/C in the plane, beam up)', (2, 1),
     {'A': 'side', 'C': 'side', 'B': 'face', 'D': 'face'}),
    ('side  X-Y   (arms B/D in the plane, beam up)', (0, 1),
     {'B': 'side', 'D': 'side', 'A': 'face', 'C': 'face'}),
]


def _hot_names(g):
    names = [f'SiPM bar {b:02d}' for b in g['wall_bars']]
    if g['plastic_name']:
        names.append(g['plastic_name'])
    return set(names)


def _pp(pt, idx):
    return pt[idx[0]], pt[idx[1]]


def fig_projections(row, g, arm, event_id, out_path):
    """The three-panel track display, upgraded for one event: top-down, the
    side view in whose plane this arm lies, and a close-up of the arm itself.
    The elements that fired are opaque with a heavy edge; the track is drawn
    from its closest approach to the beam axis outward.

    (The X-Y / Z-Y side view that collapses this arm's drift direction is
    dropped -- for a single event it adds nothing the other three do not say,
    and three wide panels are what fits on a slide.)"""
    hot = _hot_names(g)
    fig, axs = plt.subplots(1, 3, figsize=(19.2, 6.6))
    axs = np.atleast_1d(axs)
    at = g['at']
    views = [VIEWS[0], VIEWS[1] if arm in ('A', 'C') else VIEWS[2]]
    for ax, (vt, idx, modes) in zip(axs.ravel()[:2], views):
        for a, mode in modes.items():
            rdisp._draw_arm_2d(ax, a, idx, mode,
                               highlight=hot if a == arm else (),
                               dim=1.0 if a == arm else .45)
        rdisp._draw_target(ax, idx)
        _draw_track(ax, g, idx, arm=arm)
        ax.set_aspect('equal')
        ax.set_xlim(-560, 560)
        ax.set_ylim(-560, 560)
        ax.grid(alpha=.25, lw=.3)
        ax.set_title(vt, fontsize=9.5, loc='left')
        xl = {0: 'X [mm]', 1: 'Y (beam) [mm]', 2: 'Z [mm]'}
        ax.set_xlabel(xl[idx[0]])
        ax.set_ylabel(xl[idx[1]])
        if ax is axs[0]:
            for a in geo.ARMS:
                p = geo.arm_front_face(a) + geo.W_HAT[a] * 340
                ax.text(p[2], p[0], a, ha='center', va='center', fontsize=12,
                        fontweight='bold', color=MUTED)
            handles = [Rectangle((0, 0), 1, 1, facecolor=rdisp.KIND_COLOR[k],
                                 alpha=.6, edgecolor='k', lw=.3, label=lab)
                       for k, lab in (('mm', 'MM drift gas'),
                                      ('sipm', 'SiPM wall (16 bars)'),
                                      ('plastic', 'plastics'),
                                      ('ls', 'liquid scintillator'))]
            handles += [plt.Line2D([], [], color=TRACK_C, lw=3,
                                   label='measured segment'),
                        plt.Line2D([], [], color=TRACK_C, lw=1.2, ls='--',
                                   label='extrapolation')]
            ax.legend(handles=handles, frameon=False, fontsize=7.5,
                      loc='lower left')

    # --- close-up: the arm's own (u, w) plane, the micro-TPC's view
    ax = axs[2]
    idx = (2, 0) if arm in ('A', 'C') else (0, 2)
    for a, mode in (( arm, 'topdown'),):
        rdisp._draw_arm_2d(ax, a, idx, mode, highlight=hot)
    _draw_track(ax, g, idx, lw=2.4, arm=arm)
    ff = geo.arm_front_face(arm)
    w, u = geo.W_HAT[arm], geo.U_HAT[arm]
    c = ff + w * 110.0 + u * 0.5 * (g['u_wall'] + g['v_wall'] * 0)
    cx, cy = _pp(c, idx)
    ax.set_xlim(cx - 165, cx + 165)
    ax.set_ylim(cy - 165, cy + 165)
    ax.set_aspect('equal')
    ax.grid(alpha=.25, lw=.3)
    ax.set_title(f'arm {arm} close-up: drift gap -> SiPM wall -> plastics',
                 fontsize=9.5, loc='left')
    xl = {0: 'X [mm]', 1: 'Y (beam) [mm]', 2: 'Z [mm]'}
    ax.set_xlabel(xl[idx[0]])
    ax.set_ylabel(xl[idx[1]])
    for tag, txt in (('wall', f'wall segment {g["seg"]}'),
                     ('plastic', f'PSS {g["pss_detn"]}')):
        p = _pp(g[f'p_{tag}'], idx)
        ax.plot(*p, marker='*', ms=15, color=HOT, mec='k', mew=.6, zorder=9)
        ax.annotate(txt, p, textcoords='offset points', xytext=(8, 8),
                    fontsize=8.5, color='#a03a26')

    txt = (f'bunch {int(row["BunchNumber"])}, '
           f't = {row["t_since_flash_ns"] / 1e6:.2f} ms after the flash   |   '
           f'theta_u = {g["theta_x_deg"]:+.1f} deg, '
           f'theta_v = {g["theta_y_deg"]:+.1f} deg   |   '
           f'passes {g["dca_beam_mm"]:.1f} mm from the beam axis at '
           f'y = {g["beam_y_mm"]:+.0f} mm   |   '
           f'crosses the wall at u = {g["u_wall"]:+.0f} mm (segment {g["seg"]} '
           f'= bars {g["wall_bars"][0]}-{g["wall_bars"][-1]}, '
           f'{g["wall_span"][0]:+.0f}..{g["wall_span"][1]:+.0f} mm) and the '
           f'plastic at u = {g["u_plastic"]:+.0f} mm (PSS {g["pss_detn"]}, '
           f'{g["plastic_span"][0]:+.0f}..{g["plastic_span"][1]:+.0f} mm)')
    fig.suptitle(f'run_145 / mx17_{arm}  event {event_id}: waveform-first track '
                 f'through the MX17 geometry   [PRELIMINARY]\n' + txt,
                 ha='left', x=0.01, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out_path)


def _draw_track(ax, g, idx, lw=1.8, arm='A'):
    """Measured segment (inside the drift gap) thick, extrapolation thin:
    back to the point of closest approach to the beam axis, and out past the
    plastics. Nothing is drawn on the far side of the beam axis — a particle
    from the target never went there."""
    at = g['at']
    w_in = float(np.clip(g['w_closest'], 0.0, 420.0))
    p_in, p_out = at(w_in), at(-STRIPS_TO_PLASTIC[arm] - 45)
    ax.plot([_pp(p_in, idx)[0], _pp(p_out, idx)[0]],
            [_pp(p_in, idx)[1], _pp(p_out, idx)[1]],
            '--', color=TRACK_C, lw=lw * .7, alpha=.8, zorder=8)
    a, b = at(0.0), at(geo.DRIFT_GAP)
    ax.plot([_pp(a, idx)[0], _pp(b, idx)[0]], [_pp(a, idx)[1], _pp(b, idx)[1]],
            '-', color=TRACK_C, lw=lw * 2.2, alpha=.95, zorder=9,
            solid_capstyle='round')


# ---------------------------------------------------------------------- 3D
def fig_3d(g, arm, event_id, out_path,
           views=(('across the arm: target -> drift gap -> wall -> plastic',
                   74.0, 430.0, 1250.0, 'mid'),
                  ('from above: the wall bars and the plastic behind them',
                   58.0, 1450.0, 850.0, 'mid'))):
    """The track through the 3D model, rendered with pyvista.

    The geometry is the Geant build's (ntof_tracking.reco.geometry is kept in
    sync with SimConfig.hh); the He-3 capsule profile is taken from the Geant
    scripts directly. The triggering arm is solid, its neighbours are ghosts,
    and the far arms are left out entirely — an event display, not a geometry
    drawing. Views are (label, azimuth from the arm's normal toward +u, camera
    height above the beam plane [mm], distance [mm], focus) -- looking ACROSS
    the stack, so capsule -> drift gap -> wall -> plastic all stay visible.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    def _box(pl, c0, c1, color, opacity, edge=False):
        lo, hi = np.minimum(c0, c1), np.maximum(c0, c1)
        pl.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])),
                    color=color, opacity=opacity, smooth_shading=True,
                    show_edges=edge, edge_color='#333333', line_width=1.5)

    hot = _hot_names(g)
    C = {'mm': '#4a90d9', 'sipm': '#f0c040', 'plastic': '#e07820', 'ls': '#b0b0b0'}
    w_hat, u_hat = geo.W_HAT[arm], geo.U_HAT[arm]
    at = g['at']
    w_in = float(np.clip(g['w_closest'], 0.0, 420.0))
    # the middle of the story: from where the track left the beam axis out to
    # the plastic it fired
    focus_mid = 0.5 * (at(w_in) + g['p_plastic'])
    imgs, labs = [], []
    for tag, az_deg, height, dist, foc in views:
        pl = pv.Plotter(off_screen=True, window_size=(1450, 1150))
        pl.set_background('white')
        pl.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
        for a in geo.ARMS:
            near = (a == arm)
            if not near:
                continue                 # the other arms only get in the way
            for el in geo.arm_active_volumes(a):
                if el['kind'] == 'ls':
                    continue             # the LS slab hides everything behind it
                ff = geo.arm_front_face(a, el['on'])
                w, u = geo.W_HAT[a], geo.U_HAT[a]
                c0 = ff + w * el['w0'] + u * el['u_lo'] - geo.V_HAT * el['half_v']
                c1 = ff + w * el['w1'] + u * el['u_hi'] + geo.V_HAT * el['half_v']
                fired = near and el['name'] in hot
                op = (0.98 if fired else (0.26 if near else 0.06))
                _box(pl, c0, c1, HOT if fired else C[el['kind']], op,
                     edge=fired)
        # He-3 capsule: Al vessel + gas, profiles from the Geant build
        for prof_r, prof_y, col, op in _capsule_profiles():
            pts = np.column_stack([prof_r, np.zeros_like(prof_r), prof_y])
            mesh = pv.lines_from_points(pts).extrude_rotate(resolution=64,
                                                            capping=True)
            mesh.rotate_x(-90.0, inplace=True)     # profile axis -> +Y (beam)
            pl.add_mesh(mesh, color=col, opacity=op, smooth_shading=True)
        pl.add_mesh(pv.Arrow(start=(0, -250, 0), direction=(0, 1, 0), scale=500,
                             tip_length=0.05, tip_radius=0.010,
                             shaft_radius=0.003), color='firebrick')
        # the track: thick where it was MEASURED (inside the drift gap),
        # thin where it is extrapolated (back to the beam, out to the plastics)
        pl.add_mesh(pv.Tube(pointa=tuple(at(w_in)), pointb=tuple(at(0.0)),
                            radius=1.7), color=TRACK_C, opacity=.75)
        pl.add_mesh(pv.Tube(pointa=tuple(at(0.0)), pointb=tuple(at(geo.DRIFT_GAP)),
                            radius=5.0), color=TRACK_C)
        pl.add_mesh(pv.Tube(pointa=tuple(at(0.0)),
                            pointb=tuple(at(-STRIPS_TO_PLASTIC[arm] - 40)),
                            radius=1.7), color=TRACK_C, opacity=.75)
        for t2 in ('wall', 'plastic'):
            pl.add_mesh(pv.Sphere(radius=9.0, center=tuple(g[f'p_{t2}'])),
                        color=TRACK_C)
        labels = [(g['p_wall'] - u_hat * 45 + geo.V_HAT * 150,
                   f'wall segment {g["seg"]}'),
                  (g['p_plastic'] + u_hat * 40 - geo.V_HAT * 110,
                   f'plastic {g["pss_detn"]}'),
                  (at(15.0) - u_hat * 60 + geo.V_HAT * 120, 'MM drift gap'),
                  (np.array([0., 95., 0.]), 'He-3 target')]
        pl.add_point_labels([tuple(p) for p, _ in labels],
                            [t for _, t in labels], text_color='#222222',
                            font_size=22, bold=False, shape=None,
                            show_points=False, always_visible=True)
        focus = focus_mid if foc == 'mid' else g[f'p_{foc}']
        az = np.radians(az_deg)
        eye = focus + (w_hat * np.cos(az) + u_hat * np.sin(az)) * dist \
            + geo.V_HAT * height
        pl.camera_position = [tuple(eye), tuple(focus), (0, 1, 0)]
        img = str(out_path).replace('.png', f'_{len(imgs)}.png')
        pl.screenshot(img, return_img=False)
        pl.close()
        _crop_white(img)
        imgs.append(img)
        labs.append(tag)
    _compose_3d(imgs, labs, g, arm, event_id, out_path)
    for i in imgs:
        os.remove(i)
    print('wrote', out_path)


def _opposite(arm):
    return {'A': 'C', 'C': 'A', 'B': 'D', 'D': 'B'}[arm]


def _capsule_profiles():
    """(r, y, colour, opacity) polycone profiles of the He-3 capsule. The Al
    vessel comes from the Geant scripts; the active gas is already in
    reco.geometry."""
    out = []
    try:
        sys.path.insert(0, os.path.expanduser('~/CLionProjects/MX17_Full_Geant/scripts'))
        import plot_geometry as pg           # noqa: E402
        out.append((pg.RO_AL * 10.0, pg.Z_AL * 10.0, '#aaaaaa', 0.55))
    except Exception:
        pass
    out.append((geo.HE3_GAS_R, geo.HE3_GAS_Y, '#3fb8e8', 0.95))
    return out


def _crop_white(path, margin=12):
    """Trim the white border pyvista leaves around the scene."""
    a = plt.imread(path)
    ink = (a[..., :3] < 0.985).any(axis=-1)
    if not ink.any():
        return
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, a.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, a.shape[1])
    plt.imsave(path, a[y0:y1, x0:x1])


def _compose_3d(imgs, labs, g, arm, event_id, out_path):
    fig, axes = plt.subplots(1, len(imgs), figsize=(7.6 * len(imgs), 5.8))
    axes = np.atleast_1d(axes)
    for ax, img, lab in zip(axes, imgs, labs):
        ax.imshow(plt.imread(img))
        ax.axis('off')
        ax.set_title(lab, loc='left', fontsize=9, color=MUTED)
    fig.suptitle(f'run_145 / mx17_{arm}  event {event_id}: the same track in the '
                 f'3D model — MM drift gas (blue), SiPM wall (yellow), plastics '
                 f'(orange), He-3 capsule\nthe bar group and the plastic bar '
                 f'that fired the trigger are highlighted; stars mark where the '
                 f'track crosses them   [PRELIMINARY]',
                 ha='left', x=0.01, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# -------------------------------------------------------------------- driver
def make_event(d, event_id, arm, out_dir, mapping='descending', tr=None,
               skip_waveforms=False):
    tr = tr or transforms()[f'mx17_{arm}']
    row = d[d['event_id'] == event_id]
    if not len(row):
        raise SystemExit(f'event {event_id} not in the merged table')
    row = row.iloc[0]
    g = event_geometry(row, arm, tr, mapping)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'evt{event_id}'
    fig_projections(row, g, arm, event_id, out_dir / f'{stem}_projections.png')
    fig_3d(g, arm, event_id, out_dir / f'{stem}_3d.png')
    if not skip_waveforms:
        ev = refit_event(event_id, arm)
        if ev is None or not ev['planes']:
            print(f'[display] no waveform window recovered for {event_id}')
        else:
            fig_waveforms(ev, event_id, arm, out_dir / f'{stem}_waveforms.png')
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pick', 'make'])
    ap.add_argument('--merged', default=str(MERGED))
    ap.add_argument('--arm', default='A')
    ap.add_argument('--n', type=int, default=15)
    ap.add_argument('--event', type=int, action='append')
    ap.add_argument('--best', type=int, default=0,
                    help='make displays for the N best-ranked events')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--no-waveforms', action='store_true')
    a = ap.parse_args()

    d = pd.read_parquet(a.merged)
    sj = Path(a.merged).with_name('merged_prelim.summary.json')
    mapping = 'descending'
    if sj.exists():
        mapping = (json.load(open(sj)).get('wall_pointing') or {}).get(
            'mapping', 'descending')
    tr = transforms()[f'mx17_{a.arm}']
    out = Path(a.outdir or (Path(a.merged).parent / 'event_displays'))

    if a.cmd == 'pick':
        t = pick_events(d, a.arm, a.n, mapping, tr)
        pd.set_option('display.width', 200)
        print(t.round(2).to_string())
        out.mkdir(parents=True, exist_ok=True)
        t.to_csv(out / 'candidates.csv', index=False)
        print('wrote', out / 'candidates.csv')
        return 0

    ids = list(a.event or [])
    if a.best:
        ids += pick_events(d, a.arm, a.best, mapping, tr)['event_id'].tolist()
    if not ids:
        raise SystemExit('nothing to draw: pass --event or --best')
    for eid in ids:
        make_event(d, int(eid), a.arm, out, mapping, tr,
                   skip_waveforms=a.no_waveforms)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
