#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_event_displays_3d.py

3-D cosmic-muon event displays from the MX17 detector-3 micro-TPC data
(Saturday 6-27 long run, resist 490 V / drift 1000 V), companion to the 2-D
``make_event_displays.py``.

Each display draws, in the detector-local frame:
  * X-strip hits  (x measured, y predicted from the Y fit)   — reds / crimson
  * Y-strip hits  (x predicted from the X fit, y measured)   — blues / steelblue
  * the M3 reference track as a straight LINE
        x(z) = ref_mesh_x_mm + z * ref_tan_theta_x
        y(z) = ref_mesh_y_mm + z * ref_tan_theta_y
  where z is the electron drift distance from the mesh.

The drawing is the canonical implementation in
``cosmic_bench_analysis/cosmic_micro_tpc_analysis.py``
(``plot_event_display_3d`` / ``plot_event_display_3d_rotating``); this script
just reuses the QA data-loading + event-selection from make_event_displays.py
and feeds the matched EventResults (which carry the reference-track anchor and
slopes) into those functions.

Produces (in engineer_package/event_displays_3d/):
  event_<eid>_3d_display.png         static 3-D view (per picked band)
  event_<eid>_3d_rotating.gif        turntable animation (unless --no-gif)
  DISPLAYS_3D.md                     event/band/angle index

Usage (from mx_june_cosmic_qa/):
  ../.venv/bin/python engineer_package/make_event_displays_3d.py
      [--refresh]                    rebuild the candidate cache
      [--pick 1=<eid> 2=<eid> ...]   override auto-picked event per band
      [--bands 1,2,5,6]              which BANDS to render (default: ref-matched)
      [--no-gif]                     skip the rotating GIFs (faster)
      [--gif-frames N] [--gif-fps N]

Only events matched to a single clean M3 reference ray get a 3-D display,
since the whole point is to show the reference track as a line; the steep
telescope-free bands (3/4) have no reference and are skipped by default.
"""
import os
import sys
import pickle
import argparse

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                          # noqa: E402
import matplotlib.patheffects as pe                      # noqa: E402
from matplotlib import colors as mcolors                 # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
QA_DIR = os.path.dirname(HERE)
sys.path.insert(0, QA_DIR)
from qa_config import get_config, setup_paths            # noqa: E402
setup_paths()
import cosmic_micro_tpc_analysis as cm                   # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles   # noqa: E402

# Reuse the exact loading + selection machinery from the 2-D display script so
# the 3-D displays land on the very same hand-tuned "nice" events per band.
# Importing it also applies the package matplotlib style (rcParams).
import make_event_displays as m2                         # noqa: E402
from make_event_displays import (                        # noqa: E402
    CFG, VETO, CHI2_CUT, M3_MIN_NCLUS, RES_CUT_MM, V_UM_NS, BANDS,
    CHARGE_CMAP, INK, DATE_STR,
    load_hits, build_candidates, pick_events,
)

OUT = os.path.join(HERE, 'event_displays_3d')
FIG_DIR = os.path.join(HERE, 'figures')
# reference-matched bands only (BANDS 3/4 are the telescope-free steep tier,
# which have no M3 reference track to draw as a line).
DEFAULT_BANDS = [1, 2, 5, 6]

REF_GREEN = '#1a9850'          # M3 reference-track line
DRIFT_WINDOW_MM = 30.0         # nominal drift gap (top depth-cue plane)

# --- M3 reference-track pointing resolution at the DUT plane (aligned frame).
# Measured by the M3 self-resolution study (m3_self_resolution/results.json,
# "pointing at_DUT 702 mm"); fallbacks are that study's published values.  The
# M3 angular error adds <~15 µm over the 30 mm gap, so a constant width is
# accurate; the display band uses these rotated into the raw strip frame.
_M3RES_JSON = os.path.join(QA_DIR, 'm3_self_resolution', 'results.json')
REF_SIGMA_ALIGNED_X_MM, REF_SIGMA_ALIGNED_Y_MM = 0.206, 0.242
try:
    import json as _json
    with open(_M3RES_JSON) as _f:
        _p = _json.load(_f)['pointing']
    REF_SIGMA_ALIGNED_X_MM = _p['X']['at_DUT']['702 (mid-gap)'] / 1000.0
    REF_SIGMA_ALIGNED_Y_MM = _p['Y']['at_DUT']['702 (mid-gap)'] / 1000.0
except Exception as _exc:                                    # pragma: no cover
    print(f'note: using fallback M3 pointing sigmas ({_exc})')


def ref_band_sigmas(best):
    """Per-axis 1σ M3 pointing resolution in the RAW strip frame [mm]."""
    return cm.ref_sigma_raw_frame(best, REF_SIGMA_ALIGNED_X_MM,
                                  REF_SIGMA_ALIGNED_Y_MM)


# ----------------------------------------------------------------- data load
def load_full_reference():
    """Load the cached EventResults, alignment and M3 rays.

    Mirrors ``make_event_displays.load_reference`` but returns the *full*
    EventResult list and the AlignmentParams (both of which that helper
    discards) — the 3-D drawing needs ``ref_mesh_x/y_mm`` and the per-axis
    reference tangents that ``attach_reference_positions`` writes onto each
    EventResult.

    Returns
    -------
    results : list[EventResult]      with ref_mesh_x/y_mm + ref_tan_theta_x/y
    best    : AlignmentParams        (z_x, z_y, rotation, ...) for the subtitle
    ref     : dict eid -> (tan_x_det, tan_y_det)   single clean-ray matches
    by_eid  : dict eid -> EventResult
    """
    cache = os.path.join(CFG.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}',
                              'alignment.json')
    if not os.path.exists(cache):
        raise SystemExit(
            f'missing EventResult cache: {cache}\n'
            'Run the micro-TPC alignment/QA pipeline first (the same cache the '
            '2-D make_event_displays.py consumes).')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)

    ray_counts = pd.Series(anum).value_counts()
    single_ray = set(ray_counts[ray_counts == 1].index.astype(int))
    th = np.deg2rad(best.theta_deg)
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM \
                or np.isnan(r.ref_tan_theta_x) or r.event_id not in single_ray:
            continue
        # rotate reference tangents into the detector frame (matches 2-D script)
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
    by_eid = {int(r.event_id): r for r in results}
    print(f'{len(ref):,} events matched to a single clean reference ray')
    return results, best, ref, by_eid


def reconcile_candidates(cand, ref, by_eid):
    """Drop/repair candidate rows against the *current* reference set.

    ``candidates.csv`` is shared with the 2-D display script and can predate a
    change in the M3 recipe (e.g. the χ²<1 & NClus≥4 reprocess), so some cached
    picks no longer have a reference track. Keep only events that currently
    carry a finite reference anchor, and overwrite the (possibly stale)
    reference-angle columns from the live ``ref`` dict so band selection is
    consistent with what will actually be drawn.
    """
    def _has_anchor(eid):
        r = by_eid.get(int(eid))
        return (r is not None and r.has_both
                and eid in ref
                and np.isfinite(r.ref_mesh_x_mm) and np.isfinite(r.ref_mesh_y_mm))

    keep = cand[cand['eid'].apply(_has_anchor)].copy()
    tx = keep['eid'].map(lambda e: ref[int(e)][0])
    ty = keep['eid'].map(lambda e: ref[int(e)][1])
    keep['theta_ref_x'] = np.degrees(np.arctan(tx))
    keep['theta_ref_y'] = np.degrees(np.arctan(ty))
    keep['theta_sel'] = keep['theta_ref_x']
    keep['radial_resid'] = keep['eid'].map(
        lambda e: float(by_eid[int(e)].radial_residual_mm))
    n_drop = len(cand) - len(keep)
    print(f'{len(keep):,} candidates with a live reference '
          f'({n_drop:,} cached picks dropped as reference-less)')
    return keep


def pick_by_residual(cand, bands, overrides, max_resid=1.2):
    """Per band, pick the display-clean event whose reference track best
    overlays the charge cloud — i.e. the smallest ref-vs-detector radial
    residual (tightest match), rather than the 2-D fit-quality score.

    The build_candidates gates already guarantee a clean single cluster and a
    full drift span, so among those the min-residual event gives the most
    convincing overlay while still being aesthetically clean. ``max_resid``
    keeps the pick near/below the population median (~0.8 mm) so the display
    honestly reflects the ~0.6-0.8 mm/axis resolution.
    """
    picks = {}
    for band in bands:
        lo, hi = BANDS[band]
        if band in overrides:
            row = cand[cand['eid'] == overrides[band]]
            if len(row):
                picks[band] = row.iloc[0]
            continue
        sel = cand[cand['theta_sel'].abs().between(lo, hi)]
        tight = sel[sel['radial_resid'] <= max_resid]
        pool = tight if len(tight) else sel
        if len(pool) == 0:
            print(f'  band {band} ({lo:.0f}-{hi:.0f} deg): no candidate')
            continue
        picks[band] = pool.nsmallest(1, 'radial_resid').iloc[0]
        r = picks[band]
        print(f'  band {band} ({lo:.0f}-{hi:.0f} deg): eid {int(r["eid"])}  '
              f'resid={r["radial_resid"]:.2f} mm  thx_ref={r["theta_ref_x"]:+.1f}  '
              f'ny={r["y_n"]:.0f}')
    return picks


# ------------------------------------------------- depth-resolved diagnostic
def _clip_sigma(a, k=4, n_iter=5):
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    m, s = np.median(a), a.std()
    for _ in range(n_iter):
        keep = np.abs(a - m) < k * s
        if keep.sum() < 20:
            break
        m, s = a[keep].mean(), a[keep].std()
    return m, s


def render_depth_diagnostic(ref, by_eid, hits, best, out_dir=OUT):
    """Depth-resolved residual: distance of each strip signal from the M3
    reference *line*, as a function of drift depth, over the whole matched
    population.  Separates three things the single-point mesh residual cannot:

      (A) a depth-dependent BIAS  -> the raw charge-sharing angle bias (slope of
          the trend) and its non-linearity;
      (B) resolution vs depth     -> diffusion + attachment (sigma grows deep);
      (C) whether the mesh anchor sits at the low-bias pivot (position metric OK).

    Everything is in the raw strip frame (as the 3-D display), using the rotated
    reference tangents.  Signed so a *steeper* raw ladder reads positive.
    """
    eids = set(int(e) for e in ref)
    Z, D, DS, AMP = [], [], [], []
    ev_steep = []                                  # |tan_raw| - |tan_ref| per event/plane
    mesh_res = []                                  # earliest-hit residual (raw frame)
    hv = hits[hits['eventId'].isin(eids)]
    for eid, df in hv.groupby('eventId'):
        r = by_eid[int(eid)]
        tanx, tany = cm._rotate_ref_tangents(r, best)
        t0 = min(r.x_fit.earliest_time_ns, r.y_fit.earliest_time_ns)
        dfx = df[df['x_position_mm'].notna()]
        dfy = df[df['y_position_mm'].notna()]
        zx = (dfx['time'].values - t0) * V_UM_NS / 1000.0
        zy = (dfy['time'].values - t0) * V_UM_NS / 1000.0
        dx = dfx['x_position_mm'].values - (r.ref_mesh_x_mm + zx * tanx)
        dy = dfy['y_position_mm'].values - (r.ref_mesh_y_mm + zy * tany)
        Z += list(zx) + list(zy)
        D += list(dx) + list(dy)
        DS += list(np.sign(tanx) * dx) + list(np.sign(tany) * dy)
        AMP += list(dfx['amplitude'].values) + list(dfy['amplitude'].values)
        mesh_res += [r.x_fit.mesh_position_mm - r.ref_mesh_x_mm,
                     r.y_fit.mesh_position_mm - r.ref_mesh_y_mm]
        if len(zx) >= 4:
            ev_steep.append(abs(np.polyfit(zx, dfx['x_position_mm'].values, 1)[0]) - abs(tanx))
        if len(zy) >= 4:
            ev_steep.append(abs(np.polyfit(zy, dfy['y_position_mm'].values, 1)[0]) - abs(tany))
    Z, D, DS, AMP = map(np.array, (Z, D, DS, AMP))
    ev_steep = np.array(ev_steep)
    mesh_res = np.array(mesh_res)

    bins = np.arange(0, 30.01, 2.5)
    zc, bias, sem, sig = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (Z >= lo) & (Z < hi)
        if sel.sum() < 50:
            continue
        mb, _ = _clip_sigma(DS[sel])
        _, ss = _clip_sigma(D[sel])
        zc.append((lo + hi) / 2); bias.append(mb); sig.append(ss)
        sem.append(ss / np.sqrt(sel.sum()))
    zc, bias, sem, sig = map(np.array, (zc, bias, sem, sig))
    steep_med = np.median(ev_steep[np.isfinite(ev_steep)])
    ang_med = np.degrees(np.arctan(0.17 + steep_med)) - np.degrees(np.arctan(0.17))
    mm, sm = _clip_sigma(mesh_res)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax[0].axhline(0, color='k', lw=0.7)
    ax[0].errorbar(zc, bias, yerr=sem, fmt='o-', color='#c0392b', lw=1.8, ms=5)
    ax[0].set_xlabel('drift depth z  [mm]')
    ax[0].set_ylabel('signed residual from reference line  [mm]')
    ax[0].set_title('(A) Position bias vs depth\n= raw charge-sharing angle bias', fontsize=12)
    ax[0].annotate(f'raw ladder ~{ang_med:+.1f}° steeper than reference\n'
                   f'(removed by hit-level unsharing in analysis)',
                   xy=(0.03, 0.95), xycoords='axes fraction', va='top', fontsize=9,
                   color='#c0392b')
    ax[1].plot(zc, sig, 'o-', color='#2e86c1', lw=1.8, ms=5)
    ax[1].set_ylim(0, None)
    ax[1].set_xlabel('drift depth z  [mm]')
    ax[1].set_ylabel('residual core σ  [mm]')
    ax[1].set_title('(B) Resolution vs depth\n(diffusion + attachment)', fontsize=12)
    good = ev_steep[np.isfinite(ev_steep)]
    ax[2].hist(good[np.abs(good) < 0.3], bins=70, color='#666')
    ax[2].axvline(0, color='k', lw=0.8)
    ax[2].axvline(steep_med, color='#c0392b', lw=1.8,
                  label=f'median {steep_med:+.3f}  (~{ang_med:+.1f}°)')
    ax[2].set_xlabel('|tan(raw ladder)| − |tan(reference)|  per event')
    ax[2].set_title('(C) Raw-ladder steepening\n(per event)', fontsize=12)
    ax[2].legend(fontsize=9)
    fig.suptitle(f'Depth-resolved reference-track residual — detector A, '
                 f'{len(eids):,} matched muons   '
                 f'[mesh anchor: mean {mm:+.02f}, σ {sm:.2f} mm]',
                 fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, '_depth_residual_diagnostic.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  depth diagnostic -> {p}')
    print(f'    mesh anchor residual: mean {mm:+.3f} mm, sigma {sm:.3f} mm (centered => no misalignment)')
    print(f'    raw-ladder steepening: median {steep_med:+.3f} tan (~{ang_med:+.1f} deg); '
          f'bias grows ~0 at mesh to ~{np.max(bias):+.1f} mm mid-drift')
    return p


# ------------------------------------------------------------------- drawing
def make_3d_for_event(row, by_eid, hits, best, *, make_gif=True,
                      gif_frames=180, gif_fps=20):
    """Render the static 3-D display (+ optional rotating GIF) for one event."""
    eid = int(row['eid'])
    result = by_eid.get(eid)
    if result is None or not result.has_both:
        print(f'  eid {eid}: no EventResult with both planes — skipped')
        return None
    if not (np.isfinite(result.ref_mesh_x_mm) and np.isfinite(result.ref_mesh_y_mm)
            and np.isfinite(result.ref_tan_theta_x)
            and np.isfinite(result.ref_tan_theta_y)):
        print(f'  eid {eid}: no valid reference-track anchor — skipped')
        return None

    df_event = hits[hits['eventId'] == eid].copy()
    ref_sx, ref_sy = ref_band_sigmas(best)

    # static display (reds/blues, colorbar, green reference line ± pointing σ)
    cm.plot_event_display_3d(
        df_event, result, best, v_drift_um_per_ns=V_UM_NS,
        event_id=eid, out_dir=OUT,
        ref_sigma_x_mm=ref_sx, ref_sigma_y_mm=ref_sy)

    # clean turntable version (amplitude-sized points, mesh/drift planes,
    # limegreen reference line) + optional rotating GIF
    gif_path = os.path.join(OUT, f'event_{eid}_3d_rotating.gif') if make_gif else None
    cm.plot_event_display_3d_rotating(
        df_event, result, best, v_drift_um_per_ns=V_UM_NS,
        event_id=eid, gif_path=gif_path,
        gif_frames=gif_frames, gif_fps=gif_fps,
        ref_sigma_x_mm=ref_sx, ref_sigma_y_mm=ref_sy)

    # aligned telescope angles (band-defining, matches the 2-D display captions)
    thx = float(row['theta_ref_x'])
    thy = float(row['theta_ref_y'])
    print(f'  eid {eid}: 3-D display  telescope(thx,thy)=({thx:+.1f},{thy:+.1f}) deg'
          + ('  + GIF' if make_gif else ''))
    return dict(eid=eid, thx=thx, thy=thy)


# --------------------------------------------------- polished package figures
def _event_geometry(df_event, result, best):
    """Reconstruct the 3-D point cloud + reference-track polyline for one event.

    Mirrors the coordinate construction in ``cm.plot_event_display_3d`` (x/y =
    strip position, z = drift distance from the mesh; complementary coordinate
    predicted from the other plane's fit) but extends the reference line across
    the *full* depth spanned by the hits so it reads as a track through them.
    The reference tangents are rotated from the aligned/M3 frame into the raw
    strip frame (via cm._rotate_ref_tangents) so the line's slope matches the
    charge cloud — the mesh anchor is already in raw coordinates.
    """
    x_fit, y_fit = result.x_fit, result.y_fit
    t0 = min(x_fit.earliest_time_ns, y_fit.earliest_time_ns)

    def _z(t):
        return (np.asarray(t, float) - t0) * V_UM_NS / 1000.0

    dfx = df_event[df_event['x_position_mm'].notna()]
    dfy = df_event[df_event['y_position_mm'].notna()]
    x_meas, t_x, amp_x = (dfx['x_position_mm'].values, dfx['time'].values,
                          dfx['amplitude'].values)
    y_meas, t_y, amp_y = (dfy['y_position_mm'].values, dfy['time'].values,
                          dfy['amplitude'].values)
    z_x, z_y = _z(t_x), _z(t_y)
    y_at_x = y_fit.mesh_position_mm + (t_x - y_fit.earliest_time_ns) * y_fit.slope_mm_per_ns
    x_at_y = x_fit.mesh_position_mm + (t_y - x_fit.earliest_time_ns) * x_fit.slope_mm_per_ns

    tan_x, tan_y = cm._rotate_ref_tangents(result, best)
    z_hi = max(float(np.max(np.concatenate([z_x, z_y]))), 1.0)
    z_track = np.linspace(0.0, z_hi, 200)
    x_track = result.ref_mesh_x_mm + z_track * tan_x
    y_track = result.ref_mesh_y_mm + z_track * tan_y
    ref_sx, ref_sy = ref_band_sigmas(best)
    return dict(x_meas=x_meas, y_at_x=y_at_x, z_x=z_x, amp_x=amp_x,
                x_at_y=x_at_y, y_meas=y_meas, z_y=z_y, amp_y=amp_y,
                x_track=x_track, y_track=y_track, z_track=z_track, z_hi=z_hi,
                ref_sx=ref_sx, ref_sy=ref_sy)


def _draw_event_3d(ax, g, *, norm, planes=True, s_scale=1.0, label=False):
    """Render one event onto a 3-D axis in the package style.

    X-strip hits are circles, Y-strip hits are squares, both colored by pulse
    amplitude (viridis); the M3 reference track is a green line. Optional faint
    mesh (z=0) and drift-window (z=30 mm) planes give depth cues.
    """
    all_x = np.concatenate([g['x_meas'], g['x_at_y'], g['x_track']])
    all_y = np.concatenate([g['y_at_x'], g['y_meas'], g['y_track']])
    pad = 3.0
    xlo, xhi = all_x.min() - pad, all_x.max() + pad
    ylo, yhi = all_y.min() - pad, all_y.max() + pad
    zhi = max(g['z_hi'] * 1.05, DRIFT_WINDOW_MM)

    if planes:
        xx, yy = np.meshgrid([xlo, xhi], [ylo, yhi])
        ax.plot_surface(xx, yy, np.zeros_like(xx), color='silver',
                        alpha=0.13, linewidth=0, zorder=0)
        ax.plot_surface(xx, yy, np.full_like(xx, DRIFT_WINDOW_MM),
                        color='lightsteelblue', alpha=0.10, linewidth=0, zorder=0)

    def _sz(amp):
        m = amp.max() if amp.size else 1.0
        return s_scale * (34.0 + 210.0 * (amp / m if m else amp))

    sc = ax.scatter(g['x_meas'], g['y_at_x'], g['z_x'], marker='o',
                    c=g['amp_x'], cmap=CHARGE_CMAP, norm=norm, s=_sz(g['amp_x']),
                    edgecolors='white', linewidths=0.8, depthshade=False, zorder=4)
    ax.scatter(g['x_at_y'], g['y_meas'], g['z_y'], marker='s',
               c=g['amp_y'], cmap=CHARGE_CMAP, norm=norm, s=_sz(g['amp_y']),
               edgecolors='white', linewidths=0.8, depthshade=False, zorder=4)

    if 'ref_sx' in g:
        cm._draw_ref_uncertainty_tube(ax, g['x_track'], g['y_track'],
                                      g['z_track'], g['ref_sx'], g['ref_sy'],
                                      color=REF_GREEN)
    ax.plot(g['x_track'], g['y_track'], g['z_track'], color=REF_GREEN, lw=3.0,
            zorder=5, solid_capstyle='round',
            path_effects=[pe.Stroke(linewidth=5.0, foreground='white'),
                          pe.Normal()])

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(0, zhi)
    ax.set_box_aspect((1, 1, 1.35))
    ax.view_init(elev=17, azim=-58)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#cccccc')
    ax.tick_params(colors='#666666', labelsize=8)
    return sc


def _fig_legend(fig, ax=None, anchor=None):
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#7a7a7a',
               markeredgecolor='white', markersize=8, label='X strips'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#7a7a7a',
               markeredgecolor='white', markersize=8, label='Y strips'),
        Line2D([0], [0], color=REF_GREEN, lw=3,
               label='M3 reference track ±1σ/2σ pointing (σ ≈ 0.2 mm)'),
    ]
    kw = dict(handles=handles, loc='upper left', fontsize=10, frameon=False)
    if anchor is not None:
        kw['bbox_to_anchor'] = anchor
    (ax or fig).legend(**kw)


def _save_twins(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, f'{stem}.png')
    pdf = os.path.join(FIG_DIR, f'{stem}.pdf')
    fig.savefig(png, dpi=220, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote figures/{stem}.png + .pdf')


def _geom_from_hits(df_event, result, best):
    """Like _event_geometry but derives the per-plane fit (for the predicted
    complementary coordinate and t0) from the GIVEN hits, so it works for either
    the raw combined hits or the SOTA (unshared) hits. The reference line uses
    the M3 anchor + rotated tangents (independent of the chamber hits).
    """
    dfx = df_event[df_event['x_position_mm'].notna()]
    dfy = df_event[df_event['y_position_mm'].notna()]
    x_meas, t_x, amp_x = (dfx['x_position_mm'].values, dfx['time'].values,
                          dfx['amplitude'].values)
    y_meas, t_y, amp_y = (dfy['y_position_mm'].values, dfy['time'].values,
                          dfy['amplitude'].values)
    t0 = min(t_x.min() if len(t_x) else np.inf, t_y.min() if len(t_y) else np.inf)

    def _z(t):
        return (np.asarray(t, float) - t0) * V_UM_NS / 1000.0

    z_x, z_y = _z(t_x), _z(t_y)

    def _pred(t_here, z_other, pos_other):
        # predict the complementary coordinate from a linear fit pos(z) of the
        # OTHER plane, evaluated at this hit's depth
        if len(z_other) >= 2 and np.ptp(z_other) > 0:
            m, b = np.polyfit(z_other, pos_other, 1)
            return b + m * _z(t_here)
        return np.full(len(np.atleast_1d(t_here)),
                       pos_other.mean() if len(pos_other) else np.nan)

    y_at_x = _pred(t_x, z_y, y_meas)
    x_at_y = _pred(t_y, z_x, x_meas)

    tan_x, tan_y = cm._rotate_ref_tangents(result, best)
    zs = np.concatenate([z_x, z_y]) if (len(z_x) + len(z_y)) else np.array([0.0, 1.0])
    z_hi = max(float(np.max(zs)), 1.0)
    z_track = np.linspace(0.0, z_hi, 200)
    x_track = result.ref_mesh_x_mm + z_track * tan_x
    y_track = result.ref_mesh_y_mm + z_track * tan_y
    ref_sx, ref_sy = ref_band_sigmas(best)
    return dict(x_meas=x_meas, y_at_x=y_at_x, z_x=z_x, amp_x=amp_x,
                x_at_y=x_at_y, y_meas=y_meas, z_y=z_y, amp_y=amp_y,
                x_track=x_track, y_track=y_track, z_track=z_track, z_hi=z_hi,
                ref_sx=ref_sx, ref_sy=ref_sy)


def _core_filter(df, frac=0.30):
    """Keep only core strips (amp >= frac*max) per event/plane — the strips the
    ladder fit uses; drops the low-amplitude edge strips where the deconvolution
    amplifies noise."""
    import pandas as pd
    if df.empty:
        return df
    keep = [g[g['amplitude'] >= frac * g['amplitude'].max()]
            for _, g in df.groupby(['eventId', 'plane'])]
    return pd.concat(keep) if keep else df


def _ladder_angle(df, result, best):
    """Per-event combined ladder angle (deg) from a hit DataFrame, in the raw
    strip frame, for comparison with the reference."""
    dfx = df[df['plane'] == 'x']; dfy = df[df['plane'] == 'y']
    t0 = df['time'].min()
    out = {}
    for pl, g in (('x', dfx), ('y', dfy)):
        if len(g) >= 3:
            z = (g['time'].values - t0) * V_UM_NS / 1000.0
            if np.ptp(z) > 0:
                out[pl] = np.polyfit(z, g[f'{pl}_position_mm'].values, 1)[0]
    return out


def render_unsharing_comparison(row, result, best, det, wf_cache=None):
    """Single-event 3-D comparison: RAW hits (left) vs UNSHARED hits (right),
    core strips, both vs the M3 reference line.  HONEST framing: the unsharing
    corrects the ladder *angle* (fitted slope moves toward the reference) but at
    the single-hit level the deconvolution adds per-strip scatter — the win is
    the angle, shown quantitatively at the population level in
    render_unsharing_depth_proof.  Saved as a secondary diagnostic.
    """
    import sota_reco as sota
    eid = int(row['eid'])
    if wf_cache is None:
        wf_cache = sota.load_waveforms([eid], CFG, det)
    raw = _core_filter(sota.sota_hits([eid], CFG, det, do_unshare=False, wf_cache=wf_cache))
    uns = _core_filter(sota.sota_hits([eid], CFG, det, do_unshare=True, wf_cache=wf_cache))
    if raw.empty or uns.empty:
        print(f'  eid {eid}: no waveform hits — comparison skipped')
        return None

    tanx, tany = cm._rotate_ref_tangents(result, best)
    ang_ref = np.degrees(np.arctan(np.hypot(tanx, tany)))
    fig = plt.figure(figsize=(13.6, 6.8))
    amps = np.concatenate([raw['amplitude'].values, uns['amplitude'].values])
    norm = mcolors.Normalize(vmin=0, vmax=np.percentile(amps, 99))
    ang_txt = []
    for i, (title, df) in enumerate([('RAW hits (production path)', raw),
                                     ('UNSHARED hits (hit-level correction)', uns)]):
        g = _geom_from_hits(df, result, best)
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        _draw_event_3d(ax, g, norm=norm, s_scale=0.9)
        ax.set_xlabel('X [mm]', fontsize=9, labelpad=2)
        ax.set_ylabel('Y [mm]', fontsize=9, labelpad=2)
        ax.set_zlabel('drift depth [mm]', fontsize=9, labelpad=2)
        ax.set_title(title, fontsize=12, color=INK, pad=0)
        sl = _ladder_angle(df, result, best)
        a = np.degrees(np.arctan(np.hypot(sl.get('x', 0.0), sl.get('y', 0.0))))
        ang_txt.append(a)
        ax.text2D(0.5, -0.02, f'fitted ladder angle: {a:.1f}°   (reference {ang_ref:.1f}°)',
                  transform=ax.transAxes, ha='center', fontsize=10, color=INK)
    _fig_legend(fig, anchor=(0.005, 0.93))
    fig.suptitle('Charge-sharing unsharing corrects the ladder ANGLE — '
                 f'detector A, event {eid}',
                 fontsize=14, fontweight='bold', color=INK, y=1.0)
    fig.text(0.5, 0.945,
             f'raw ladder {ang_txt[0]:.1f}° (~{ang_txt[0]-ang_ref:+.1f}° vs reference) → '
             f'unshared {ang_txt[1]:.1f}° ({ang_txt[1]-ang_ref:+.1f}°). '
             f'Per-hit scatter rises (deconvolution noise) — the win is statistical, '
             f'see the population proof.',
             ha='center', fontsize=10, color='#666666')
    fig.subplots_adjust(left=0.0, right=0.99, top=0.90, bottom=0.06, wspace=0.02)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'_unsharing_event_compare_{eid}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  single-event comparison -> {p}  (raw {ang_txt[0]:.1f}° / unshared '
          f'{ang_txt[1]:.1f}° / ref {ang_ref:.1f}°)')
    return ang_txt


def render_unsharing_depth_proof(ref, by_eid, best, det, n_events=600):
    """POPULATION proof that hit-level unsharing removes the charge-sharing
    angle bias: the depth-resolved signed residual (core strips) BEFORE vs AFTER
    unsharing.  The bias slope (residual angle bias) drops sharply.  figures/09.
    """
    import sota_reco as sota
    eids = sorted(int(e) for e in ref)[:n_events]
    print(f'  unsharing depth proof: processing {len(eids)} events (waveforms)...')
    wf = sota.load_waveforms(eids, CFG, det)

    def profile(do_unshare):
        df = _core_filter(sota.sota_hits(eids, CFG, det, do_unshare=do_unshare, wf_cache=wf))
        Z, DS = [], []
        for eid, d in df.groupby('eventId'):
            r = by_eid[int(eid)]
            tanx, tany = cm._rotate_ref_tangents(r, best)
            t0 = d['time'].min()
            for pl, col, tr, mr in (('x', 'x_position_mm', tanx, r.ref_mesh_x_mm),
                                    ('y', 'y_position_mm', tany, r.ref_mesh_y_mm)):
                g = d[d.plane == pl]
                if len(g) < 3:
                    continue
                z = (g['time'].values - t0) * V_UM_NS / 1000.0
                Z += list(z)
                DS += list(np.sign(tr) * (g[col].values - (mr + z * tr)))
        Z, DS = np.array(Z), np.array(DS)
        bins = np.arange(0, 27.51, 2.5); zc, mb = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (Z >= lo) & (Z < hi)
            if sel.sum() < 30:
                continue
            a = DS[sel]; m = np.median(a)
            for _ in range(4):
                k = np.abs(a - m) < 3 * a.std()
                if k.sum() < 10:
                    break
                m = a[k].mean()
            zc.append((lo + hi) / 2); mb.append(m)
        return np.array(zc), np.array(mb)

    zr, br = profile(False); zu, bu = profile(True)
    sr = np.polyfit(zr, br, 1)[0] * 1000; su = np.polyfit(zu, bu, 1)[0] * 1000
    print('    depth[mm]  raw-bias  unshared-bias')
    for i in range(min(len(zr), len(zu))):
        print(f'      {zr[i]:5.1f}    {br[i]:+.3f}     {bu[i]:+.3f}')

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.axhline(0, color='k', lw=0.8)
    ax.plot(zr, br, 'o-', color='#9a9a9a', lw=2.4, ms=7,
            label=f'RAW production ladder   ({sr:+.0f} µm/mm)')
    ax.plot(zu, bu, 'o-', color=REF_GREEN, lw=2.4, ms=7,
            label=f'UNSHARED (state-of-the-art)   ({su:+.0f} µm/mm)')
    ax.set_xlabel('drift depth  z  [mm]', fontsize=12)
    ax.set_ylabel('signed distance of hits from reference track  [mm]', fontsize=12)
    ax.set_title('Hit-level charge-sharing unsharing removes the depth bias\n'
                 f'detector A · {len(eids)} muons · core strips', fontsize=13,
                 fontweight='bold', color=INK)
    ax.legend(fontsize=10.5, loc='upper left')
    ax.annotate(f'depth-dependent bias reduced {100*(1-abs(su)/abs(sr)):.0f}%\n'
                '(residual = diffusion floor, removed by the\n'
                'per-plane angle calibration, script 28)',
                xy=(0.97, 0.05), xycoords='axes fraction', ha='right', va='bottom',
                fontsize=9.5, color='#555555')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_twins(fig, '09-det3A-unsharing-depth-bias')
    print(f'    raw bias slope {sr:+.0f} µm/mm → unshared {su:+.0f} µm/mm '
          f'({100*(1-abs(su)/abs(sr)):.0f}% reduction)')
    return sr, su


def render_hero(row, result, df_event, best):
    """Single large 3-D figure for the report/slide (figures/07-...)."""
    eid = int(row['eid'])
    g = _event_geometry(df_event, result, best)
    amps = np.concatenate([g['amp_x'], g['amp_y']])
    norm = mcolors.Normalize(vmin=0, vmax=amps.max())

    fig = plt.figure(figsize=(9.2, 7.6))
    ax = fig.add_axes([0.0, 0.02, 0.86, 0.86], projection='3d')
    sc = _draw_event_3d(ax, g, norm=norm)
    ax.set_xlabel('strip position X [mm]', fontsize=11, labelpad=6)
    ax.set_ylabel('strip position Y [mm]', fontsize=11, labelpad=6)
    ax.set_zlabel('drift depth [mm]', fontsize=11, labelpad=4)

    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.0, aspect=18)
    cb.set_label('signal amplitude [ADC]', fontsize=10.5)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)

    thx, thy = float(row['theta_ref_x']), float(row['theta_ref_y'])
    fig.text(0.5, 0.965,
             'A cosmic muon in 3-D: charge cloud vs the reference track',
             ha='center', fontsize=15, fontweight='bold', color=INK)
    fig.text(0.5, 0.922,
             f'detector A · event {eid} · telescope angle '
             f'{abs(thx):.0f}° (X) / {abs(thy):.0f}° (Y) · '
             f'match {result.radial_residual_mm:.1f} mm',
             ha='center', fontsize=11, color='#666666')
    _fig_legend(fig, anchor=(0.015, 0.86))
    fig.text(0.985, 0.015, f'{DATE_STR}', ha='right', fontsize=8.5,
             color='#999999')
    _save_twins(fig, '07-det3A-event-3d-display')


def render_gallery(picked, best):
    """2x2 gallery of 3-D events across angle bands (figures/08-...)."""
    picked = picked[:4]
    fig = plt.figure(figsize=(12.6, 10.4))
    for i, (row, result, df_event) in enumerate(picked):
        g = _event_geometry(df_event, result, best)
        amps = np.concatenate([g['amp_x'], g['amp_y']])
        norm = mcolors.Normalize(vmin=0, vmax=amps.max())
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        _draw_event_3d(ax, g, norm=norm, s_scale=0.62)
        ax.set_xlabel('X [mm]', fontsize=8, labelpad=1)
        ax.set_ylabel('Y [mm]', fontsize=8, labelpad=1)
        ax.set_zlabel('depth [mm]', fontsize=8, labelpad=1)
        thx = abs(float(row['theta_ref_x']))
        ax.set_title(f'event {int(row["eid"])} · {thx:.0f}° (telescope)',
                     fontsize=11.5, color=INK, pad=0)
    _fig_legend(fig, ax=fig)
    fig.suptitle('Cosmic muons in 3-D — charge cloud vs M3 reference track',
                 fontsize=16, fontweight='bold', color=INK, y=0.98)
    fig.text(0.5, 0.045,
             'Each point is one strip signal (circle = X plane, square = Y '
             'plane) placed at its strip position and drift depth, colored by '
             'amplitude; the green line is the independent M3 telescope track, '
             'drawn with its measured ±1σ/2σ pointing envelope (~0.2 mm).',
             ha='center', fontsize=10.5, color='#666666')
    fig.text(0.985, 0.012, f'detector A · {DATE_STR}', ha='right',
             fontsize=8.5, color='#999999')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.09,
                        wspace=0.02, hspace=0.10)
    _save_twins(fig, '08-det3A-event-3d-gallery')


def write_md(rendered):
    lines = [
        '# 3-D micro-TPC event displays (detector 3)\n',
        'Companion to the 2-D `make_event_displays.py`. Each event is drawn in '
        'the detector-local frame with the electron drift distance as the '
        'vertical axis; the **green line is the M3 reference track**.\n',
        '| band | event | ref θx [deg] | ref θy [deg] | files |',
        '|------|-------|--------------|--------------|-------|',
    ]
    for band, rec in sorted(rendered.items()):
        eid = rec['eid']
        lines.append(
            f"| {band} | {eid} | {rec['thx']:+.1f} | {rec['thy']:+.1f} | "
            f"`event_{eid}_3d_display.png`, `event_{eid}_3d_rotating.gif` |")
    with open(os.path.join(OUT, 'DISPLAYS_3D.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  wrote DISPLAYS_3D.md ({len(rendered)} events)')


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true',
                    help='rebuild the candidate cache')
    ap.add_argument('--pick', nargs='*', default=[],
                    help='band=eid overrides, e.g. --pick 1=12345')
    ap.add_argument('--bands', default=None,
                    help='comma-separated bands to render (default: 1,2,5,6)')
    ap.add_argument('--no-gif', action='store_true',
                    help='skip the rotating GIFs (much faster)')
    ap.add_argument('--gif-frames', type=int, default=180)
    ap.add_argument('--gif-fps', type=int, default=20)
    ap.add_argument('--figures', action='store_true',
                    help='also build the polished package figures '
                         '(figures/07 hero + 08 gallery, PNG+PDF twins)')
    ap.add_argument('--hero-band', type=int, default=6,
                    help='which band supplies the hero figure (default 6, ~20 deg)')
    ap.add_argument('--gallery-bands', default='1,5,2,6',
                    help='ordered bands for the 2x2 gallery (default 1,5,2,6)')
    ap.add_argument('--max-resid', type=float, default=1.2,
                    help='prefer events with radial match <= this [mm] '
                         '(tightest overlay; ~median is 0.8 mm)')
    ap.add_argument('--diagnostic', action='store_true',
                    help='also build the depth-resolved residual diagnostic '
                         '(_depth_residual_diagnostic.png): bias/resolution vs '
                         'drift depth over the whole matched population')
    ap.add_argument('--compare', action='store_true',
                    help='build the unsharing figures: population depth-bias '
                         'proof (figures/09) + single-event 3-D compare '
                         '(needs decoded_root waveforms)')
    ap.add_argument('--compare-n', type=int, default=600,
                    help='number of events for the population unsharing proof')
    args = ap.parse_args()

    overrides = {int(k): int(v) for k, v in
                 (s.split('=') for s in args.pick)}
    want_bands = ([int(b) for b in args.bands.split(',')]
                  if args.bands else DEFAULT_BANDS)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(m2.CACHE, exist_ok=True)

    results, best, ref, by_eid = load_full_reference()
    hits, det = load_hits()
    cand = build_candidates(hits, ref, refresh=args.refresh)
    cand = reconcile_candidates(cand, ref, by_eid)
    # pick the tightest-matching clean event per band, so the drawn reference
    # track overlays the charge cloud as well as the ~0.6-0.8 mm/axis resolution
    # actually allows (the old 2-D fit-score pick landed above the median match).
    gal_bands = [int(b) for b in args.gallery_bands.split(',')]
    all_bands = sorted(set(want_bands) | set(gal_bands) | {args.hero_band})
    print('selecting tightest-match events per band:')
    picks = pick_by_residual(cand, all_bands, overrides, max_resid=args.max_resid)

    def _valid(row):
        r = by_eid.get(int(row['eid']))
        return (r is not None and r.has_both
                and np.isfinite(r.ref_mesh_x_mm) and np.isfinite(r.ref_mesh_y_mm)
                and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y))

    rendered = {}
    for band in want_bands:
        if band not in picks:
            print(f'band {band}: no picked candidate — skipped')
            continue
        row = picks[band]
        lo, hi = BANDS[band]
        print(f'band {band} ({lo:.0f}-{hi:.0f} deg): eid {int(row["eid"])}')
        rec = make_3d_for_event(
            row, by_eid, hits, best, make_gif=not args.no_gif,
            gif_frames=args.gif_frames, gif_fps=args.gif_fps)
        if rec is not None:
            rendered[band] = rec

    if rendered:
        write_md(rendered)

    if args.diagnostic:
        print('\nbuilding depth-resolved residual diagnostic:')
        render_depth_diagnostic(ref, by_eid, hits, best)

    if args.compare:
        print('\nbuilding unsharing figures (needs decoded_root waveforms):')
        render_unsharing_depth_proof(ref, by_eid, best, det, n_events=args.compare_n)
        hrow = picks.get(args.hero_band)
        if hrow is not None and _valid(hrow):
            render_unsharing_comparison(hrow, by_eid[int(hrow['eid'])], best, det)
        else:
            print(f'  hero band {args.hero_band}: no valid pick — single-event compare skipped')

    if args.figures:
        print('\nbuilding polished package figures:')
        gal_bands = [int(b) for b in args.gallery_bands.split(',')]
        gallery = []
        for band in gal_bands:
            row = picks.get(band)
            if row is None or not _valid(row):
                print(f'  gallery band {band}: no valid pick — skipped')
                continue
            eid = int(row['eid'])
            gallery.append((row, by_eid[eid], hits[hits['eventId'] == eid].copy()))
        if gallery:
            render_gallery(gallery, best)
        hrow = picks.get(args.hero_band)
        if hrow is not None and _valid(hrow):
            heid = int(hrow['eid'])
            render_hero(hrow, by_eid[heid], hits[hits['eventId'] == heid].copy(), best)
        else:
            print(f'  hero band {args.hero_band}: no valid pick — hero skipped')

    print(f'\nOutputs in {OUT}  ({len(rendered)} events rendered)')


if __name__ == '__main__':
    main()
