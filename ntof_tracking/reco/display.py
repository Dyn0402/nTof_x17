#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
display.py — event displays: per-plane micro-TPC views with reconstruction
overlays, and global-frame views of extrapolated 3D tracks through the
MX17 experiment elements.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import MaxNLocator

from . import geometry as geo

CLS_COLOR = {'track': 'crimson', 'point': 'royalblue',
             'band_fragment': 'gray', 'blob': 'darkorange'}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# per-plane event display with reco overlays
# ---------------------------------------------------------------------------
def plot_event_planes(hits_ev, segs, title, out_dir, name,
                      amp_clip=(100, 2500)):
    """4 detectors x 2 planes; per row: drift TIME on the shared y axis,
    strip coordinate on x (X plane left, Y plane right, no gap between
    them — the micro-TPC convention: the vertical axis is the drift/depth
    direction). Noise-flagged hits greyed, clean hits coloured by
    amplitude; track fits drawn translucent BEHIND the hit points."""
    dets = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
    fig, axes = plt.subplots(4, 2, figsize=(12, 15), sharey=True)
    fig.subplots_adjust(wspace=0.0, hspace=0.22)
    sc = None
    for ri, det in enumerate(dets):
        for ci, plane in enumerate(['x', 'y']):
            ax = axes[ri][ci]
            g = hits_ev[(hits_ev['det'] == det) & (hits_ev['plane'] == plane)]
            band = g[g['in_band']]
            iso = g[g['isolated']]
            clean = g[g['clean']]
            ax.scatter(band['pos_mm'], band['time'] / 1e3, c='0.85', s=7,
                       zorder=3, label='band' if ri == ci == 0 else None)
            ax.scatter(iso['pos_mm'], iso['time'] / 1e3, c='0.7', s=7,
                       marker='x', zorder=3,
                       label='isolated' if ri == ci == 0 else None)
            if len(clean):
                sc = ax.scatter(clean['pos_mm'], clean['time'] / 1e3,
                                c=np.clip(clean['amplitude'], *amp_clip),
                                cmap='viridis', vmin=amp_clip[0],
                                vmax=amp_clip[1], s=16, zorder=4)
            for s in segs:
                if s['det'] != det or s['plane'] != plane:
                    continue
                col = CLS_COLOR.get(s['cls'], 'k')
                if s['cls'] == 'track' and np.isfinite(s.get('slope_mm_ns', np.nan)):
                    tt = np.array([s['t0_ns'], s['t1_ns']])
                    pp = s['slope_mm_ns'] * tt + s['intercept_mm']
                    ax.plot(pp, tt / 1e3, '-', color=col, lw=2.5, alpha=.45,
                            zorder=2)
                    ax.annotate(f"track n={s['n_strips']} r2={s.get('r2', 0):.2f}",
                                xy=(pp[0], tt[0] / 1e3), fontsize=7, color=col,
                                xytext=(2, 6), textcoords='offset points')
                else:
                    ax.add_patch(Rectangle(
                        (s['pos_lo_mm'], s['t0_ns'] / 1e3),
                        max(s['pos_hi_mm'] - s['pos_lo_mm'], 1.0),
                        max((s['t1_ns'] - s['t0_ns']) / 1e3, .01),
                        fill=False, edgecolor=col, lw=1.0, ls='--', alpha=.6,
                        zorder=2))
            ax.grid(alpha=.25)
            ax.set_xlim(-205, 205)
            ax.set_xlabel(f'{det} {plane} [mm]', fontsize=9)
            # panels touch: drop the tick label at the shared edge
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=8, prune='upper' if ci == 0 else 'lower'))
            if ci == 0:
                ax.set_ylabel('hit time [us]', fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            ax.text(.02, .96, f'n={len(g)} clean={len(clean)}',
                    transform=ax.transAxes, va='top', fontsize=7,
                    bbox=dict(boxstyle='round', fc='w', alpha=.7))
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=.015,
                     label='amplitude [ADC]')
    fig.suptitle(title, fontsize=13)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# global-frame extrapolation views
# ---------------------------------------------------------------------------
def _proj(pt, axes_idx):
    return pt[axes_idx[0]], pt[axes_idx[1]]


KIND_COLOR = {'mm': '#4a90d9', 'sipm': '#f0c040',
              'plastic': '#e07820', 'ls': '#d9534f'}


def _draw_arm_2d(ax, arm, axes_idx, mode: str, highlight=(), dim=1.0):
    """Draw one arm's ACTIVE volumes projected on (axes_idx) screen axes.

    mode: 'topdown' — u x w footprint (resolves SiPM/plastic bars);
          'side'    — w x v profile (arm's normal lies in the plane);
          'face'    — u x v face-on GHOST of the MM drift gas only, very
                      transparent: locates a measured segment drawn in a
                      panel whose projection collapses this arm's drift
                      direction (out-of-plane arm in a side view).

    `highlight` is a collection of element names (as in arm_active_volumes,
    e.g. 'SiPM bar 13', 'plastic R') to draw opaque with a heavy edge — for an
    event display, the elements that actually fired. `dim` scales the alpha of
    everything else, so one arm can be shown against ghosted neighbours.
    """
    for el in geo.arm_active_volumes(arm):
        if mode == 'face' and el['kind'] != 'mm':
            continue
        hot = el['name'] in highlight
        ff = geo.arm_front_face(arm, el['on'])
        w, u = geo.W_HAT[arm], geo.U_HAT[arm]
        if mode == 'topdown':
            corner0 = ff + w * el['w0'] + u * el['u_lo']
            corner1 = ff + w * el['w1'] + u * el['u_hi']
        elif mode == 'side':
            corner0 = ff + w * el['w0'] - geo.V_HAT * el['half_v']
            corner1 = ff + w * el['w1'] + geo.V_HAT * el['half_v']
        else:  # face
            wm = 0.5 * (el['w0'] + el['w1'])
            corner0 = ff + w * wm + u * el['u_lo'] - geo.V_HAT * el['half_v']
            corner1 = ff + w * wm + u * el['u_hi'] + geo.V_HAT * el['half_v']
        x0, y0 = _proj(corner0, axes_idx)
        x1, y1 = _proj(corner1, axes_idx)
        if mode == 'face':
            ax.add_patch(Rectangle((min(x0, x1), min(y0, y1)),
                                   abs(x1 - x0), abs(y1 - y0),
                                   facecolor=KIND_COLOR['mm'], alpha=.10,
                                   edgecolor=KIND_COLOR['mm'], lw=.6,
                                   zorder=1))
        else:
            ax.add_patch(Rectangle((min(x0, x1), min(y0, y1)),
                                   abs(x1 - x0), abs(y1 - y0),
                                   facecolor=KIND_COLOR[el['kind']],
                                   alpha=(.95 if hot else .6 * dim),
                                   edgecolor=('k' if hot else 'k'),
                                   lw=(1.6 if hot else .3), zorder=3 if hot else 2))


def _draw_target(ax, axes_idx):
    """He-3 ACTIVE gas: circle (top-down) or polycone profile (side views)."""
    if axes_idx[1] == 0:      # top-down (screen y = X): circular bore
        ax.add_patch(Circle((0, 0), geo.HE3_R_MAX, facecolor='#99d8f5',
                            edgecolor='k', lw=.5, zorder=5))
    else:                     # side view (screen y = Y): profile along beam
        r, y = geo.HE3_GAS_R, geo.HE3_GAS_Y
        poly = np.vstack([np.column_stack([r, y]),
                          np.column_stack([-r[::-1], y[::-1]])])
        ax.add_patch(plt.Polygon(poly, closed=True, facecolor='#99d8f5',
                                 edgecolor='k', lw=.5, zorder=5))
        ax.annotate('', xy=(0, -60), xytext=(0, -170),
                    arrowprops=dict(arrowstyle='-|>', color='firebrick'))
        ax.text(8, -140, 'beam (+Y)', color='firebrick', fontsize=8)


GLOBAL_VIEWS = [
    # (title, (axis_x, axis_y), {arm: draw mode})
    ('top-down  (screen x=Z East, y=X North, beam out of page)', (2, 0),
     {'A': 'topdown', 'B': 'topdown', 'C': 'topdown', 'D': 'topdown'}),
    ('side  Z-Y  (A/C arms in plane, beam up)', (2, 1),
     {'A': 'side', 'C': 'side', 'B': 'face', 'D': 'face'}),
    ('side  X-Y  (B/D arms in plane, beam up)', (0, 1),
     {'B': 'side', 'D': 'side', 'A': 'face', 'C': 'face'}),
]


def plot_global_tracks(gsegs: List[dict], title, out_dir, name,
                       run_cfg: Optional[dict] = None):
    """Three projections (top-down X-Z, side Z-Y, side X-Y) of the geometry
    with extrapolated 3D track lines. gsegs: geometry.segment_to_global out."""
    views = GLOBAL_VIEWS
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.2))
    for ax, (vt, idx, arm_modes) in zip(axs, views):
        for arm, mode in arm_modes.items():
            _draw_arm_2d(ax, arm, idx, mode)
        _draw_target(ax, idx)
        # tracks: extrapolation in every panel (outward path dashed, backward
        # extension solid + very faint); measured segment thick everywhere —
        # out-of-plane arms show as face-on MM ghosts so it has a home
        # (run_48 _00 evt 87 lesson, v2).
        for i, s in enumerate(gsegs):
            p0, p1 = s['p_lo_global'], s['p_hi_global']
            # outward path (beam-axis closest approach -> chamber -> beyond)
            # solid-ish; backward line extension (opposite side) very faint —
            # a beamline-origin particle never goes there.
            p_mid, d_out, s_beam = geo.orient_outward(s)
            for s_a, s_b, ls, alpha in ((s_beam, 800.0, '--', .85),
                                        (-800.0, s_beam, '-', .15)):
                e0 = p_mid + s_a * d_out
                e1 = p_mid + s_b * d_out
                ax.plot([_proj(e0, idx)[0], _proj(e1, idx)[0]],
                        [_proj(e0, idx)[1], _proj(e1, idx)[1]],
                        ls, color=f'C{i}', lw=1.0, alpha=alpha, zorder=6,
                        label=(f"{s['det']} evt {s['eventId']}  "
                               f"dca_beam={s['dca_beam_axis_mm']:.0f} mm  "
                               f"vert={s['angle_to_vertical_deg']:.0f}°")
                        if alpha > .5 else None)
            ax.plot([_proj(p0, idx)[0], _proj(p1, idx)[0]],
                    [_proj(p0, idx)[1], _proj(p1, idx)[1]],
                    '-', color=f'C{i}', lw=3.0, zorder=7)
        ax.set_aspect('equal')
        ax.set_xlim(-560, 560)
        ax.set_ylim(-560, 560)
        ax.grid(alpha=.3, lw=.3)
        ax.axhline(0, color='.7', lw=.4)
        ax.axvline(0, color='.7', lw=.4)
        ax.set_title(vt, fontsize=10)
        xl = {0: 'X [mm]', 1: 'Y (beam) [mm]', 2: 'Z [mm]'}
        ax.set_xlabel(xl[idx[0]])
        ax.set_ylabel(xl[idx[1]])
        if ax is axs[0]:
            for arm in geo.ARMS:
                p = geo.arm_front_face(arm) + geo.W_HAT[arm] * 330
                ax.text(p[2], p[0], arm, ha='center', va='center',
                        fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='lower left')
    fig.suptitle(title, fontsize=13)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# ensemble views: MANY tracks from many events on one geometry
# ---------------------------------------------------------------------------
DET_COLOR = {'mx17_A': '#e41a1c', 'mx17_B': '#377eb8',
             'mx17_C': '#4daf4a', 'mx17_D': '#984ea3'}


def plot_global_ensemble(gsegs: List[dict], title, out_dir, name):
    """Same three projections as plot_global_tracks, but for an ENSEMBLE of
    3D pairs pooled across events/subruns: one thin outward-extrapolation
    line per pair, coloured by measuring detector (measured segment drawn
    slightly thicker). Backward line extensions are omitted — at ensemble
    scale they only fog the picture."""
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.2))
    n_det = {}
    for ax, (vt, idx, arm_modes) in zip(axs, GLOBAL_VIEWS):
        for arm, mode in arm_modes.items():
            _draw_arm_2d(ax, arm, idx, mode)
        _draw_target(ax, idx)
        for s in gsegs:
            col = DET_COLOR.get(s['det'], 'k')
            n_det[s['det']] = n_det.get(s['det'], 0) + 1
            p_mid, d_out, s_beam = geo.orient_outward(s)
            e0, e1 = p_mid + s_beam * d_out, p_mid + 800.0 * d_out
            ax.plot([_proj(e0, idx)[0], _proj(e1, idx)[0]],
                    [_proj(e0, idx)[1], _proj(e1, idx)[1]],
                    '-', color=col, lw=.7, alpha=.35, zorder=6)
            p0, p1 = s['p_lo_global'], s['p_hi_global']
            ax.plot([_proj(p0, idx)[0], _proj(p1, idx)[0]],
                    [_proj(p0, idx)[1], _proj(p1, idx)[1]],
                    '-', color=col, lw=2.2, alpha=.9, zorder=7)
        ax.set_aspect('equal')
        ax.set_xlim(-560, 560)
        ax.set_ylim(-560, 560)
        ax.grid(alpha=.3, lw=.3)
        ax.axhline(0, color='.7', lw=.4)
        ax.axvline(0, color='.7', lw=.4)
        ax.set_title(vt, fontsize=10)
        xl = {0: 'X [mm]', 1: 'Y (beam) [mm]', 2: 'Z [mm]'}
        ax.set_xlabel(xl[idx[0]])
        ax.set_ylabel(xl[idx[1]])
        if ax is axs[0]:
            for arm in geo.ARMS:
                p = geo.arm_front_face(arm) + geo.W_HAT[arm] * 330
                ax.text(p[2], p[0], arm, ha='center', va='center',
                        fontsize=11, fontweight='bold')
    handles = [plt.Line2D([], [], color=DET_COLOR[d], lw=2,
                          label=f'{d} ({n_det.get(d, 0) // len(GLOBAL_VIEWS)})')
               for d in sorted(DET_COLOR) if d in n_det]
    axs[0].legend(handles=handles, fontsize=8, loc='lower left')
    fig.suptitle(title, fontsize=13)
    return _save(fig, out_dir, name)


def _p3(pt):
    """Global (X, Y, Z) -> 3D plot axes (Z East, X North, Y beam-up)."""
    return pt[2], pt[0], pt[1]


def _box_faces(corner_lo, corner_hi):
    """6 quad faces of the axis-aligned box spanning the two corners
    (given in PLOT coordinates)."""
    lo = np.minimum(corner_lo, corner_hi)
    hi = np.maximum(corner_lo, corner_hi)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces_idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                 (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [v[list(f)] for f in faces_idx]


KIND_ALPHA_3D = {'mm': .16, 'sipm': .30, 'plastic': .30, 'ls': .18}


def _draw_geometry_3d(ax):
    """The MX17 ACTIVE volumes (as in the Geant4 build: MM drift gas,
    16 instrumented SiPM bars, PVT plastics, LS layer, He-3 gas polycone)
    on a 3D axis in plot coords (Z, X, Y-up)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for arm in geo.ARMS:
        for el in geo.arm_active_volumes(arm):
            ff = geo.arm_front_face(arm, el['on'])
            w, u = geo.W_HAT[arm], geo.U_HAT[arm]
            c0 = ff + w * el['w0'] + u * el['u_lo'] - geo.V_HAT * el['half_v']
            c1 = ff + w * el['w1'] + u * el['u_hi'] + geo.V_HAT * el['half_v']
            pc = Poly3DCollection(
                _box_faces(np.array(_p3(c0)), np.array(_p3(c1))),
                facecolor=KIND_COLOR[el['kind']],
                alpha=KIND_ALPHA_3D[el['kind']],
                edgecolor='k', linewidths=.15)
            ax.add_collection3d(pc)
    # He-3 gas: surface of revolution about the beam (plot-z) axis
    th = np.linspace(0, 2 * np.pi, 36)
    r, y = geo.HE3_GAS_R, geo.HE3_GAS_Y
    px = np.outer(r, np.cos(th))          # plot x = global Z
    py = np.outer(r, np.sin(th))          # plot y = global X
    pz = np.outer(y, np.ones_like(th))    # plot z = global Y (beam)
    ax.plot_surface(px, py, pz, color='#3fb8e8', alpha=.8, linewidth=0,
                    shade=True)
    # beam arrow
    ax.quiver(0, 0, -260, 0, 0, 150, color='firebrick', lw=1.8,
              arrow_length_ratio=.25)
    ax.text(0, 0, -300, 'beam (+Y)', color='firebrick', fontsize=9)
    for arm in geo.ARMS:
        p = geo.arm_front_face(arm) + geo.W_HAT[arm] * 340
        ax.text(*_p3(p), arm, ha='center', va='center', fontsize=12,
                fontweight='bold')


def plot_global_3d(gsegs: List[dict], title, out_dir, name,
                   views=((22, -55), (68, -90))):
    """3D model of the active Geant4 geometry with an ensemble of measured
    segments (thick) + their outward beam-axis->beyond extrapolations (thin),
    coloured by detector. One panel per (elev, azim) view."""
    fig = plt.figure(figsize=(9.5 * len(views), 9.5))
    n_det = {}
    for vi, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), vi + 1, projection='3d')
        _draw_geometry_3d(ax)
        for s in gsegs:
            col = DET_COLOR.get(s['det'], 'k')
            if vi == 0:
                n_det[s['det']] = n_det.get(s['det'], 0) + 1
            p_mid, d_out, s_beam = geo.orient_outward(s)
            e0, e1 = p_mid + s_beam * d_out, p_mid + 700.0 * d_out
            ax.plot(*zip(_p3(e0), _p3(e1)), '-', color=col, lw=.7,
                    alpha=.4)
            p0, p1 = s['p_lo_global'], s['p_hi_global']
            ax.plot(*zip(_p3(p0), _p3(p1)), '-', color=col, lw=2.4,
                    alpha=.95)
        ax.set_xlim(-560, 560)
        ax.set_ylim(-560, 560)
        ax.set_zlim(-560, 560)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('Z (East) [mm]')
        ax.set_ylabel('X (North) [mm]')
        ax.set_zlabel('Y (beam) [mm]')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'elev {elev}°, azim {azim}°', fontsize=9)
        if vi == 0:
            handles = [plt.Line2D([], [], color=DET_COLOR[d], lw=2,
                                  label=f'{d} ({n_det.get(d, 0)})')
                       for d in sorted(DET_COLOR) if d in n_det]
            ax.legend(handles=handles, fontsize=9, loc='upper left')
    fig.suptitle(title, fontsize=13)
    return _save(fig, out_dir, name)
