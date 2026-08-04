#!/usr/bin/env python3
"""
17_det4_board_with_beam.py — det4's board map with the H4 beam painted on it.

The inverse of `16_sps_beam_lab_frame.py`: instead of putting det4 into the beam's
frame, this puts the beam into det4's frame. Same content as
`board_map_g_det4_rot90ccw.png` — the within-5 mm efficiency map on a scale
drawing of the MX17 readout board, bands horizontal — with the triggered flux
measured on the P2 telescope overlaid where it would actually land.

One difference from `board_map_g_det4_rot90ccw.png`: it carries the measured
triggered flux. The orientation is the same — detector-local Y increases to the
LEFT, so with the X cards on the right X1 is at the bottom and with the Y cards
on top **Y1 is on the right**, which is what a 90 deg CCW rotation of the bench
view (X cards bottom / X1 left, Y cards right / Y1 bottom) gives. The board
margin and the X bank are mirrored about the active-area centre so the wide
50.32 mm connector margin and the X cards land together on the right, exactly as
`14_board_map.py` does.

The beam is placed by the mounting of `16_...`: det4 on a 30 mm riser with its
bare -X PCB edge down, readout square centred on the beam horizontally. Under
that placement the beam centre lands at detector-local ~(200, 199) mm.

Products (written next to this script):
  det4_board_with_beam.png
  det4_board_with_beam.json

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/17_det4_board_with_beam.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]
from qa_config import get_config, setup_paths                # noqa: E402
setup_paths()
import cosmic_micro_tpc_analysis as cm                       # noqa: E402

sys.path.insert(0, HERE)
from importlib import import_module                          # noqa: E402
_bm = import_module('14_board_map')
_b = import_module('15_sps_beam_board')
_lab = import_module('16_sps_beam_lab_frame')

INK, MUTED, BEAM, DET4 = _b.INK, _b.MUTED, _b.BEAM, _b.DET4
LEVELS = (0.1, 0.5, 0.9)


def det4_efficiency(key, kernel, grid, minn):
    """The within-5 mm efficiency map in detector-local mm (as 14_board_map)."""
    cfg = get_config(key)
    params = cm.load_alignment(os.path.join(cfg.OUT_BASE, 'alignment_tpc_veto50',
                                            'alignment.json'))
    d = pd.read_csv(os.path.join(cfg.OUT_BASE, 'efficiency',
                                 'ray_hit_miss_list.csv'))
    d = d[np.isfinite(d.x) & np.isfinite(d.y)]
    lx, ly = _bm.ref_to_det(d.x.to_numpy(), d.y.to_numpy(), params)
    within = d['within'].astype(str).str.lower().isin(
        ('true', '1')).to_numpy(float)
    gx = np.arange(0, 398.58 + grid, grid)
    gy = np.arange(0, 398.58 + grid, grid)
    return gx, gy, _bm.sliding(lx, ly, within, gx, gy, kernel, minn)


def beam_in_det4(maps, gap, shim, mirror):
    """Triggered flux, per pad, expressed in det4 detector-local coordinates.

    Inverse of `16_sps_beam_lab_frame.d4_to_lab`:
        local X = height - 20.32 - shim ,  local Y = x - x0 + 199.29
    with x0 (where det4 sits horizontally) set so the readout square is centred
    on the beam.
    """
    m, polys = _b.load_board()
    apex = _b.fan_apex(m)
    (_, r_hi), (phi_lo, phi_hi) = _b.active_extent(polys, apex)
    bis = 0.5 * (phi_lo + phi_hi)
    ill, files = _b.load_illumination(maps)
    keep = ill.n_tag.values > 0
    w = ill.n_tag.values[keep].astype(float)
    rows = m.set_index('channel_id').index.get_indexer(
        ill.channel_id.values[keep])
    corners = _lab.to_lab(polys[rows].reshape(-1, 2), apex, bis, gap + r_hi,
                          mirror).reshape(-1, 4, 2)
    cen = _lab.to_lab(np.stack([ill.pad_cx.values[keep],
                                ill.pad_cy.values[keep]], 1),
                      apex, bis, gap + r_hi, mirror)
    bx = float(np.average(cen[:, 0], weights=w))
    y0 = np.mean(_lab.D4_SQUARE)
    # lab -> det4-local, per pad corner
    lo_y = corners[:, :, 1].min(axis=1) - (-_lab.D4_PCB[0]) - shim
    hi_y = corners[:, :, 1].max(axis=1) - (-_lab.D4_PCB[0]) - shim
    lo_x = corners[:, :, 0].min(axis=1) - bx + y0
    hi_x = corners[:, :, 0].max(axis=1) - bx + y0
    return w, (lo_x, hi_x), (lo_y, hi_y), files


def density(ext_y, ext_x, w, gy, gx):
    """Pad weights spread over their true extents onto a (localX, localY) grid."""
    oy = _lab._overlap(ext_y[0][:, None], ext_y[1][:, None],
                       gy[None, :-1], gy[None, 1:])
    ox = _lab._overlap(ext_x[0][:, None], ext_x[1][:, None],
                       gx[None, :-1], gx[None, 1:])
    return ox.T @ (w[:, None] * oy)          # (nY, nX) -> [localY, localX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='g_det4')
    ap.add_argument('--maps', default='eff_map_P2_MID_eff_nominal_*.csv')
    ap.add_argument('--gap', type=float, default=_lab.TABLE_GAP)
    ap.add_argument('--shim', type=float, default=_lab.SHIM_MM)
    ap.add_argument('--p2-mirror', type=int, choices=(1, -1), default=-1)
    ap.add_argument('--kernel', type=float, default=3.0)
    ap.add_argument('--grid', type=float, default=1.33)
    ap.add_argument('--minn', type=int, default=1)
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()

    gx, gy, eff = det4_efficiency(args.key, args.kernel, args.grid, args.minn)
    w, ext_x, ext_y, files = beam_in_det4(args.maps, args.gap, args.shim,
                                          args.p2_mirror)

    # beam density on a coarse grid in the same coordinates
    by_ = np.arange(-40, 460, 8.0)          # local X (vertical)
    bx_ = np.arange(-40, 460, 8.0)          # local Y (horizontal)
    dens = density(ext_y, ext_x, w, by_, bx_)
    # light 3x3 box smoothing: the pads are ~12 mm, so anything finer than that
    # in the contours is binning noise, not structure
    k = np.ones(3) / 3.0
    dens = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 0, dens)
    dens = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 1, dens)
    cy = 0.5 * (by_[:-1] + by_[1:])
    cx = 0.5 * (bx_[:-1] + bx_[1:])
    tot = w.sum()

    # beam centre and the trigger slab, in det4-local mm
    mid_x = 0.5 * (ext_x[0] + ext_x[1])
    mid_y = 0.5 * (ext_y[0] + ext_y[1])
    b_locY = float(np.average(mid_x, weights=w))
    b_locX = float(np.average(mid_y, weights=w))
    prof_x = dens.sum(axis=0)
    slab = _lab.slab_edges(cy, prof_x) if prof_x.max() > 0 else (np.nan, np.nan)

    band = _lab.D4_BAND
    sq = _lab.D4_SQUARE

    # what det4 would deliver: efficiency folded with the triggered flux
    ex, ey = np.meshgrid(gx, gy, indexing='ij')       # eff is [localX, localY]
    fy = np.interp(gx, cy, prof_x, left=0, right=0)
    prof_y = dens.sum(axis=1)
    fx = np.interp(gy, cx, prof_y, left=0, right=0)
    wgt = fy[:, None] * fx[None, :]
    ok = np.isfinite(eff)
    in_band = (ex >= band[0]) & (ex <= band[1]) & ok
    in_sq = ((ex >= sq[0]) & (ex <= sq[1]) & (ey >= sq[0]) & (ey <= sq[1]) & ok)
    eff_band = float(np.sum(eff[in_band] * wgt[in_band]) / np.sum(wgt[in_band]))
    eff_sq = float(np.sum(eff[in_sq] * wgt[in_sq]) / np.sum(wgt[in_sq]))

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(11.4, 9.6))
    pm = ax.pcolormesh(gy, gx, np.ma.masked_invalid(eff), cmap='viridis',
                       vmin=0, vmax=1, shading='auto', zorder=2)
    cb = fig.colorbar(pm, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label('det4 efficiency within 5 mm of the M3 reference '
                 f'({args.kernel:.0f} mm sliding kernel)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # board outline, active area, the bare edge that sits on the riser
    pcb_y = sorted(_lab.mirror_y(np.array(_lab.D4_PCB)))
    ax.add_patch(Rectangle((pcb_y[0], _lab.D4_PCB[0]), pcb_y[1] - pcb_y[0],
                           _lab.D4_PCB[1] - _lab.D4_PCB[0], fill=False,
                           ec='#444444', lw=1.8, zorder=4))
    ax.add_patch(Rectangle((0, 0), 399.36, 399.36, fill=False, ec='#888888',
                           lw=1.0, ls='--', zorder=4))

    # connector banks, on the Gerber's +X (top) and +Y (right) edges
    for k, lo, hi in _lab.D4_CONN:
        on = k in _lab.D4_HILITE
        kw = dict(facecolor=DET4 if on else '#dfe5ea',
                  edgecolor=DET4 if on else '#7f8c99',
                  alpha=0.85 if on else 0.8, lw=1.4 if on else 0.7, zorder=5)
        tkw = dict(fontsize=7, ha='center', va='center',
                   color='white' if on else '#55606b', zorder=6)
        xb0, xb1 = sorted(_lab.mirror_y(np.array(_lab.D4_BANK)))
        ax.add_patch(Rectangle((xb0, lo), xb1 - xb0, hi - lo, **kw))
        ax.text(0.5 * (xb0 + xb1), 0.5 * (lo + hi), f'X{k}', rotation=90,
                **tkw)
        ax.add_patch(Rectangle((lo, _lab.D4_BANK[0]), hi - lo,
                               _lab.D4_BANK[1] - _lab.D4_BANK[0], **kw))
        ax.text(0.5 * (lo + hi), np.mean(_lab.D4_BANK), f'Y{k}', **tkw)

    # the live band and the four-cable readout square
    ax.add_patch(Rectangle((0, band[0]), 399.36, band[1] - band[0], fill=False,
                           ec=DET4, lw=1.8, ls=(0, (5, 2.5)), zorder=6))
    ax.text(392, band[1] + 5, f'live band  X {band[0]:.0f}–{band[1]:.0f} mm',
            fontsize=9, color=DET4, va='bottom', ha='left', zorder=7,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none',
                      alpha=0.8))
    ax.add_patch(Rectangle((sq[0], sq[0]), sq[1] - sq[0], sq[1] - sq[0],
                           fill=False, ec=DET4, lw=2.2, zorder=6))
    ax.text(sq[1] + 6, sq[0] - 6, 'X4+X5 / Y4+Y5\nreadout square, 99 mm',
            fontsize=8.5, color=DET4, ha='left', va='top', zorder=7,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none',
                      alpha=0.8))

    # the beam
    cs = ax.contour(cx, cy, dens.T / dens.max(), levels=LEVELS, colors=BEAM,
                    linewidths=(1.0, 1.9, 1.0), zorder=8)
    ax.clabel(cs, fmt=lambda v: f'{v:.0%}', fontsize=7, inline=True)
    ax.plot([b_locY], [b_locX], '+', color=BEAM, ms=14, mew=2.4, zorder=9)
    for e in slab:
        ax.axhline(e, color=BEAM, lw=1.0, ls=(0, (6, 4)), zorder=8)
    ax.text(392, slab[1] + 5,
            f'trigger slab, local X {slab[0]:.0f}–{slab[1]:.0f} mm',
            fontsize=8.5, color=BEAM, ha='left', va='bottom', zorder=9,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none',
                      alpha=0.8))
    ax.annotate(f'beam centre\nlocal ({b_locX:.0f}, {b_locY:.0f}) mm',
                xy=(b_locY, b_locX), xytext=(96, 78),
                textcoords='offset points', fontsize=9, color=BEAM, ha='left',
                va='bottom', zorder=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BEAM,
                          alpha=0.9, lw=0.8),
                arrowprops=dict(arrowstyle='-', color=BEAM, lw=0.9))

    ax.set_xlim(_lab.D4_PCB[1] + 12, min(pcb_y[0], -60) - 12)   # Y increases LEFT
    ax.set_ylim(_lab.D4_PCB[0] - 12, _lab.D4_PCB[1] + 12)
    ax.set_aspect('equal')
    ax.set_xlabel('detector-local Y [mm], increasing LEFT   —   '
                  'same orientation as board_map_..._rot90ccw')
    ax.set_ylabel('detector-local X [mm]   (the coordinate the bands live in)')
    ax.set_title(f'det4 board map with the H4 beam on it — {args.key}, '
                 f'{args.shim:.0f} mm riser\n'
                 f'contours are the triggered flux; beam-weighted efficiency '
                 f'{100 * eff_band:.0f} % on the band, {100 * eff_sq:.0f} % '
                 'in the square', fontsize=11, loc='left')
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    png = os.path.join(args.out, 'det4_board_with_beam.png')
    fig.savefig(png, dpi=170)
    print('wrote', png)

    out = {
        'run_key': args.key, 'shim_mm': args.shim, 'gap_mm': args.gap,
        'source_maps': [os.path.basename(f) for f in files],
        'beam_centre_local_mm': {'X': b_locX, 'Y': b_locY},
        'trigger_slab_local_X_mm': [float(slab[0]), float(slab[1])],
        'live_band_local_X_mm': list(band),
        'readout_square_local_mm': list(sq),
        'beam_weighted_efficiency_on_band': eff_band,
        'beam_weighted_efficiency_in_square': eff_sq,
        'n_tagged_tracks': float(tot),
    }
    js = os.path.join(args.out, 'det4_board_with_beam.json')
    with open(js, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', js)
    print(f'\nbeam centre at detector-local ({b_locX:.1f}, {b_locY:.1f}) mm')
    print(f'trigger slab spans local X {slab[0]:.0f}-{slab[1]:.0f} mm')
    print(f'beam-weighted efficiency: band {100 * eff_band:.1f} %, '
          f'square {100 * eff_sq:.1f} %')


if __name__ == '__main__':
    main()
