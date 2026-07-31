#!/usr/bin/env python3
"""
15_sps_beam_board.py — the SPS H4 beam spot drawn on the P2 BASKET board.

Reconstructs where the H4 beam sits on the P2 telescope's readout board, and how
big it is, from the P2 group's own products (read off `banco_cern`, copied into
`sps_beam_data/`):

  * `P2_BASKET_mapping.csv`      — the Gerber-derived pad map: 1280 pads, each
    with its true centre, size and rotation, plus fan polar coordinates. This is
    the board.
  * `eff_map_<det>_*.csv`        — stage 22's per-pad `n_tag` counts. `n_tag` is
    the number of times the *other* planes tagged a track pointing at that pad,
    so it is the illumination, independent of the plane's own efficiency.

Only channels 384-895 (sectors 3-6 of 10) are instrumented, so the beam is
measured over the middle four sectors of a larger board.

Products (written next to this script):
  sps_beam_board.png    board + spot + dimensions + profiles + band containment
  sps_beam_board.json   every number in the figure

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/15_sps_beam_board.py
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.collections import PolyCollection            # noqa: E402
from matplotlib.patches import Ellipse, Polygon              # noqa: E402
import matplotlib.colors as mcolors                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'sps_beam_data')

DET4_BAND_MM = 38.0        # det4's usable live band, X 177-215 mm
BAND_WIDTHS = (25, 38, 50, 60, 80, 100)

INK = '#1a1a1a'
MUTED = '#8a8a8a'
BEAM = '#c2410c'           # warm accent for everything beam-related
DET4 = '#0f766e'           # teal for the det4 band


# --------------------------------------------------------------------------- #
# Board and illumination
# --------------------------------------------------------------------------- #
def load_board():
    """The 1280-pad map, with each pad's true rectangle corners."""
    m = pd.read_csv(os.path.join(DATA, 'P2_BASKET_mapping.csv'))
    a = np.radians(m.pad_angle.values)
    hw, hh = m.pad_w.values / 2, m.pad_h.values / 2
    cx, cy = m.pad_cx.values, m.pad_cy.values
    # corners of an axis-aligned pad, rotated by pad_angle about the centre
    ux, uy = np.cos(a), np.sin(a)
    corners = []
    for sx, sy in ((-1, -1), (+1, -1), (+1, +1), (-1, +1)):
        corners.append(np.stack([cx + sx * hw * ux - sy * hh * uy,
                                 cy + sx * hw * uy + sy * hh * ux], axis=1))
    polys = np.stack(corners, axis=1)          # (npad, 4, 2)
    return m, polys


def fan_apex(m):
    """The fan apex, back-solved from the map's own polar columns.

    `radius`/`phi` are built from the map's `x`/`y` (the strip endpoint), not
    from `pad_cx`/`pad_cy` — using the pad centroids instead puts the apex 11 mm
    off. With `x`/`y` the back-solve is exact (zero scatter over all 1280 pads).
    """
    ax = m.x.values - m.radius.values * np.cos(m.phi.values)
    ay = m.y.values - m.radius.values * np.sin(m.phi.values)
    assert ax.std() < 1e-3 and ay.std() < 1e-3, 'apex back-solve is inconsistent'
    return np.array([ax.mean(), ay.mean()])


def active_extent(polys, apex):
    """(radius, phi) range of the metallised area, from the true pad corners."""
    v = polys.reshape(-1, 2) - apex
    r = np.hypot(v[:, 0], v[:, 1])
    ph = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return (float(r.min()), float(r.max())), (float(ph.min()), float(ph.max()))


def load_illumination(pattern):
    """Sum stage-22 n_tag over every matching sub-run map."""
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    if not files:
        raise SystemExit(f'no illumination maps matching {pattern} in {DATA}')
    tot = None
    for f in files:
        d = pd.read_csv(f)[['channel_id', 'pad_cx', 'pad_cy', 'n_tag']]
        tot = d if tot is None else tot.assign(n_tag=tot.n_tag + d.set_index(
            'channel_id').loc[tot.channel_id, 'n_tag'].values)
    return tot, files


# --------------------------------------------------------------------------- #
# Spot
# --------------------------------------------------------------------------- #
def moments(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cxx = np.average((x - mx) ** 2, weights=w)
    cyy = np.average((y - my) ** 2, weights=w)
    cxy = np.average((x - mx) * (y - my), weights=w)
    C = np.array([[cxx, cxy], [cxy, cyy]])
    ev, evec = np.linalg.eigh(C)
    ang = math.degrees(math.atan2(evec[1, -1], evec[0, -1])) % 180.0
    return (mx, my), C, math.sqrt(ev[-1]), math.sqrt(ev[0]), ang


def core(x, y, w, nsig=2.0, iters=15):
    """Iterative elliptical core, with the truncation bias divided out."""
    (cx, cy), C, *_ = moments(x, y, w)
    k = nsig ** 2
    for _ in range(iters):
        Ci = np.linalg.inv(C)
        dx, dy = x - cx, y - cy
        r2 = Ci[0, 0] * dx ** 2 + 2 * Ci[0, 1] * dx * dy + Ci[1, 1] * dy ** 2
        m = r2 < k
        (cx, cy), C, *_ = moments(x[m], y[m], w[m])
    frac = w[m].sum() / w.sum()
    # E[r^2 | r^2 < k] / 2 for a 2D Gaussian
    f2 = (1 - (1 + k / 2) * math.exp(-k / 2)) / (1 - math.exp(-k / 2))
    ev, evec = np.linalg.eigh(C / f2)
    ang = math.degrees(math.atan2(evec[1, -1], evec[0, -1])) % 180.0
    return (cx, cy), C / f2, math.sqrt(ev[-1]), math.sqrt(ev[0]), ang, frac


def band_fraction(x, y, w, normal_deg, width, centre=None):
    """Weight fraction inside a straight band of the given width."""
    a = math.radians(normal_deg)
    u = x * math.cos(a) + y * math.sin(a)
    if centre is not None:
        m = np.abs(u - centre) <= width / 2
        return w[m].sum() / w.sum(), centre
    o = np.argsort(u)
    us, ws = u[o], w[o]
    cw = np.cumsum(ws)
    j = np.clip(np.searchsorted(us, us + width, side='right') - 1, 0, len(us) - 1)
    encl = cw[j] - cw + ws
    i = int(np.argmax(encl))
    return encl[i] / w.sum(), us[i] + width / 2


def profile(x, y, w, angle_deg, centre_uv, bin_mm=12.0):
    """1D illumination profile along the axis at angle_deg through the spot."""
    a = math.radians(angle_deg)
    u = (x - centre_uv[0]) * math.cos(a) + (y - centre_uv[1]) * math.sin(a)
    lo, hi = u.min(), u.max()
    edges = np.arange(math.floor(lo / bin_mm) * bin_mm, hi + bin_mm, bin_mm)
    h, _ = np.histogram(u, bins=edges, weights=w)
    return 0.5 * (edges[:-1] + edges[1:]), h


def fwhm(centres, h):
    """Full width at half maximum of a binned profile, by linear crossing."""
    half = h.max() / 2
    above = np.where(h >= half)[0]
    if len(above) < 2:
        return float('nan')
    i, j = above[0], above[-1]
    left = centres[i] if i == 0 else np.interp(
        half, [h[i - 1], h[i]], [centres[i - 1], centres[i]])
    right = centres[j] if j == len(h) - 1 else np.interp(
        half, [h[j + 1], h[j]], [centres[j + 1], centres[j]])
    return right - left


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='eff_map_P2_MID_eff_nominal_*.csv',
                    help='glob for the stage-22 maps to sum')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()

    m, polys = load_board()
    apex = fan_apex(m)
    ill, files = load_illumination(args.maps)

    live = ill.n_tag.values > 0
    x, y, w = (ill.pad_cx.values[live], ill.pad_cy.values[live],
               ill.n_tag.values[live].astype(float))
    (mx, my), C, s_maj, s_min, ang = moments(x, y, w)
    (kx, ky), Ck, k_maj, k_min, k_ang, k_frac = core(x, y, w)

    # --- where that is on the board, in the fan's own coordinates ------------
    r_beam = math.hypot(mx - apex[0], my - apex[1])
    phi_beam = math.degrees(math.atan2(my - apex[1], mx - apex[0]))
    sel = m.channel_id.between(384, 895).values
    (r_lo, r_hi), (phi_lo, phi_hi) = active_extent(polys, apex)
    (ri_lo, ri_hi), (phii_lo, phii_hi) = active_extent(polys[sel], apex)

    edges = {
        'to_outer_arc_mm': r_hi - r_beam,
        'to_inner_arc_mm': r_beam - r_lo,
        'to_low_phi_edge_mm': r_beam * math.radians(phi_beam - phi_lo),
        'to_high_phi_edge_mm': r_beam * math.radians(phi_hi - phi_beam),
        'to_instrumented_outer_arc_mm': ri_hi - r_beam,
        'to_instrumented_inner_arc_mm': r_beam - ri_lo,
        'to_instrumented_low_phi_mm': r_beam * math.radians(phi_beam - phii_lo),
        'to_instrumented_high_phi_mm': r_beam * math.radians(phii_hi - phi_beam),
    }

    # --- band containment ---------------------------------------------------
    best_normal = min(range(0, 180),
                      key=lambda a: -band_fraction(x, y, w, a, DET4_BAND_MM)[0])
    along, across = (best_normal - 90) % 180, best_normal
    contain = {int(width): {
        'along_beam': band_fraction(x, y, w, along + 90, width)[0],
        'across_beam': band_fraction(x, y, w, along, width)[0],
    } for width in BAND_WIDTHS}
    f_det4, u_det4 = band_fraction(x, y, w, best_normal, DET4_BAND_MM)
    centring = {int(d): [band_fraction(x, y, w, best_normal, DET4_BAND_MM,
                                       u_det4 + s * d)[0] for s in (+1, -1)]
                for d in (0, 5, 10, 15, 20, 25, 30)}

    # --- profiles along the two spot axes -----------------------------------
    cu_maj, h_maj = profile(x, y, w, ang, (mx, my))
    cu_min, h_min = profile(x, y, w, ang + 90, (mx, my))
    fw_maj, fw_min = fwhm(cu_maj, h_maj), fwhm(cu_min, h_min)

    # ----------------------------------------------------------------------- #
    # Figure
    # ----------------------------------------------------------------------- #
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], hspace=0.52,
                          wspace=0.20, left=0.055, right=0.965,
                          top=0.90, bottom=0.085)
    axb = fig.add_subplot(gs[:, 0])
    axp1 = fig.add_subplot(gs[0, 1])
    axp2 = fig.add_subplot(gs[1, 1])
    axc = fig.add_subplot(gs[2, 1])

    # ---- the board ---------------------------------------------------------
    counts = np.zeros(len(m))
    idx = m.set_index('channel_id').index.get_indexer(ill.channel_id.values)
    counts[idx] = ill.n_tag.values
    instrumented = m.channel_id.between(384, 895).values

    axb.add_collection(PolyCollection(
        polys[~instrumented], facecolors='#f2f2f2', edgecolors='#dcdcdc',
        linewidths=0.3, zorder=1))
    hot = counts[instrumented]
    norm = mcolors.LogNorm(vmin=max(hot[hot > 0].min(), 1), vmax=hot.max())
    pc = PolyCollection(polys[instrumented], array=hot, cmap='Blues',
                        norm=norm, edgecolors='#c8d4e0', linewidths=0.25,
                        zorder=2)
    axb.add_collection(pc)
    cb = fig.colorbar(pc, ax=axb, pad=0.015, fraction=0.045)
    cb.set_label('tagged tracks per pad', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # spot: 1 and 2 sigma from the moments, plus the 2-sigma core
    for n, lw, ls in ((1, 2.0, '-'), (2, 1.4, '-')):
        axb.add_patch(Ellipse((mx, my), 2 * n * s_maj, 2 * n * s_min, angle=ang,
                              fill=False, ec=BEAM, lw=lw, ls=ls, zorder=5))
    axb.add_patch(Ellipse((kx, ky), 2 * k_maj, 2 * k_min, angle=k_ang,
                          fill=False, ec=BEAM, lw=1.2, ls=(0, (4, 3)), zorder=5))
    axb.plot([mx], [my], '+', color=BEAM, ms=13, mew=2.2, zorder=6)

    # det4's live band at its best placement
    a = math.radians(best_normal)
    n_hat, t_hat = np.array([math.cos(a), math.sin(a)]), np.array(
        [-math.sin(a), math.cos(a)])
    half, length = DET4_BAND_MM / 2, 168.0
    ctr = np.array([mx, my]) + n_hat * (u_det4 - (mx * n_hat[0] + my * n_hat[1]))
    quad = [ctr + s1 * half * n_hat + s2 * length * t_hat
            for s1, s2 in ((-1, -1), (+1, -1), (+1, +1), (-1, +1))]
    axb.add_patch(Polygon(quad, closed=True, facecolor='none',
                          edgecolor=DET4, lw=1.8, ls=(0, (5, 2.5)), zorder=8))
    tip = ctr + t_hat * length
    axb.annotate(f'det4 band, {DET4_BAND_MM:.0f} mm', xy=tuple(tip),
                 xytext=(-8, -12), textcoords='offset points', fontsize=8.5,
                 color=DET4, ha='right', va='top', zorder=9)

    # fan geometry: apex, the radial line through the spot, the two arcs
    axb.plot([apex[0]], [apex[1]], 'o', color=MUTED, ms=5, zorder=6)
    th = np.linspace(math.radians(phi_lo), math.radians(phi_hi), 200)
    for rr in (r_lo, r_hi):
        axb.plot(apex[0] + rr * np.cos(th), apex[1] + rr * np.sin(th),
                 color=MUTED, lw=0.9, ls=':', zorder=3)
    pb = math.radians(phi_beam)
    axb.plot([apex[0], apex[0] + r_hi * math.cos(pb)],
             [apex[1], apex[1] + r_hi * math.sin(pb)],
             color=MUTED, lw=0.9, ls=':', zorder=3)
    for pe in (phii_lo, phii_hi):                     # readout aperture
        axb.plot([apex[0] + r_lo * math.cos(math.radians(pe)),
                  apex[0] + r_hi * math.cos(math.radians(pe))],
                 [apex[1] + r_lo * math.sin(math.radians(pe)),
                  apex[1] + r_hi * math.sin(math.radians(pe))],
                 color='#9db4c8', lw=1.0, ls='--', zorder=3)

    # dimension callouts: radially to the outer arc, azimuthally to the edge
    outer = np.array([apex[0] + r_hi * math.cos(pb), apex[1] + r_hi * math.sin(pb)])
    axb.annotate('', xy=tuple(outer), xytext=(mx, my),
                 arrowprops=dict(arrowstyle='<->', color=INK, lw=1.1), zorder=7)
    axb.annotate(f'{edges["to_outer_arc_mm"]:.0f} mm to the outer arc',
                 xy=tuple(outer), xytext=(-12, 20),
                 textcoords='offset points', fontsize=8.5, color=INK,
                 ha='right', va='bottom', zorder=9)
    arc = np.linspace(pb, math.radians(phi_hi), 80)
    axb.plot(apex[0] + r_beam * np.cos(arc), apex[1] + r_beam * np.sin(arc),
             color=INK, lw=1.1, zorder=7)
    end = np.array([apex[0] + r_beam * math.cos(arc[-1]),
                    apex[1] + r_beam * math.sin(arc[-1])])
    axb.annotate(f'{edges["to_high_phi_edge_mm"]:.0f} mm along the arc',
                 xy=tuple(end), xytext=(-4, 14), textcoords='offset points',
                 fontsize=8.5, color=INK, ha='center', va='bottom', zorder=9)

    axb.annotate(f'beam centre  pad ({mx:.1f}, {my:.1f}) mm\n'
                 f'r = {r_beam:.0f} mm,  $\\varphi$ = {phi_beam:.2f}$\\degree$',
                 xy=(mx, my), xytext=(24, -62), textcoords='offset points',
                 fontsize=9, color=BEAM, ha='left', va='top', zorder=9,
                 arrowprops=dict(arrowstyle='-', color=BEAM, lw=0.8))
    axb.text(apex[0] + 10, apex[1] - 4, 'fan apex', fontsize=8.5, color=MUTED,
             ha='left', va='top')

    axb.text(0.015, 0.985,
             f'the spot sits on the fan centre-line: $\\varphi$ = '
             f'{phi_beam:.2f}$\\degree$ vs {0.5 * (phi_lo + phi_hi):.2f}$\\degree$\n'
             'clearance from the spot centre to the board:\n'
             f'  outer arc {edges["to_outer_arc_mm"]:.0f} mm  ·  '
             f'inner arc {edges["to_inner_arc_mm"]:.0f} mm\n'
             f'  fan edges {edges["to_low_phi_edge_mm"]:.0f} / '
             f'{edges["to_high_phi_edge_mm"]:.0f} mm along the arc\n'
             'to the readout aperture (dashed):\n'
             f'  {edges["to_instrumented_low_phi_mm"]:.0f} / '
             f'{edges["to_instrumented_high_phi_mm"]:.0f} mm along the arc',
             transform=axb.transAxes, fontsize=8.5, color=INK, va='top',
             ha='left', zorder=9,
             bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#d8d8d8'))

    axb.set_xlim(m.pad_cx.min() - 25, m.pad_cx.max() + 25)
    axb.set_ylim(m.pad_cy.min() - 30, m.pad_cy.max() + 95)
    axb.set_aspect('equal')
    axb.set_xlabel('board x [mm]  (P2 BASKET pad frame)')
    axb.set_ylabel('board y [mm]')
    axb.set_title('H4 beam on the P2 BASKET board — grey pads are not read out',
                  fontsize=10.5, loc='left')
    axb.tick_params(labelsize=8.5)
    for s in ('top', 'right'):
        axb.spines[s].set_visible(False)

    handles = [
        plt.Line2D([], [], color=BEAM, lw=2.0, label='beam 1$\\sigma$, 2$\\sigma$'),
        plt.Line2D([], [], color=BEAM, lw=1.2, ls=(0, (4, 3)),
                   label='2$\\sigma$ core (bias-corrected)'),
        plt.Line2D([], [], color=DET4, lw=1.8, ls=(0, (5, 2.5)),
                   label=f'det4 live band, {DET4_BAND_MM:.0f} mm, best placement'),
        plt.Line2D([], [], color='#9db4c8', lw=1.0, ls='--',
                   label='readout aperture (sectors 3-6)'),
    ]
    axb.legend(handles=handles, fontsize=8.5, loc='upper right',
               framealpha=0.95, edgecolor='#d8d8d8')

    # ---- profiles ----------------------------------------------------------
    for ax, (c, h, s, fw, lab) in zip(
            (axp1, axp2),
            ((cu_min, h_min, s_min, fw_min, f'across the spot ({(ang + 90) % 180:.0f}$\\degree$)'),
             (cu_maj, h_maj, s_maj, fw_maj, f'along the spot ({ang:.0f}$\\degree$)'))):
        ax.step(c, h / h.max(), where='mid', color=BEAM, lw=1.6)
        ax.fill_between(c, 0, h / h.max(), step='mid', color=BEAM, alpha=0.12)
        ax.axvspan(-DET4_BAND_MM / 2, DET4_BAND_MM / 2, color=DET4, alpha=0.15,
                   lw=0)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xlim(-170, 170)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('illumination\n(peak = 1)', fontsize=8.5)
        ax.set_xlabel(f'distance from beam centre [mm], {lab}', fontsize=9)
        ax.text(0.985, 0.90, f'$\\sigma$ = {s:.1f} mm    FWHM = {fw:.0f} mm',
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                color=INK)
        ax.tick_params(labelsize=8.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axp1.set_title('Beam profile through the spot centre, and det4\'s 38 mm band',
                   fontsize=10.5, loc='left')

    # ---- containment -------------------------------------------------------
    ws_ = np.array(BAND_WIDTHS, float)
    axc.plot(ws_, [100 * contain[int(v)]['along_beam'] for v in ws_], 'o-',
             color=DET4, lw=1.8, ms=6, label='band along the spot')
    axc.plot(ws_, [100 * contain[int(v)]['across_beam'] for v in ws_], 's--',
             color=MUTED, lw=1.5, ms=5, label='band across the spot')
    axc.plot([DET4_BAND_MM], [100 * f_det4], '*', color=BEAM, ms=15, zorder=5)
    axc.annotate(f'det4: {100 * f_det4:.0f}%', (DET4_BAND_MM, 100 * f_det4),
                 textcoords='offset points', xytext=(8, -12), fontsize=9,
                 color=BEAM)
    axc.set_xlabel('band width [mm]')
    axc.set_ylabel('% of triggers\non the band', fontsize=8.5)
    axc.set_ylim(20, 100)
    axc.set_title('Fraction of the triggered beam a live band collects',
                  fontsize=10.5, loc='left')
    axc.legend(fontsize=8.5, loc='lower right', frameon=False)
    axc.grid(alpha=0.18, lw=0.6)
    axc.tick_params(labelsize=8.5)
    for s in ('top', 'right'):
        axc.spines[s].set_visible(False)

    fig.suptitle('SPS H4 beam spot reconstructed on the P2 telescope board  '
                 f'({len(files)} sub-runs, {w.sum() / 1e6:.1f} M tagged tracks)',
                 fontsize=12, x=0.055, ha='left')

    png = os.path.join(args.out, 'sps_beam_board.png')
    fig.savefig(png, dpi=170)
    print('wrote', png)

    out = {
        'source_maps': [os.path.basename(f) for f in files],
        'n_tagged_tracks': float(w.sum()),
        'board': {
            'n_pads': int(len(m)), 'n_instrumented': int(instrumented.sum()),
            'instrumented_channels': [384, 895],
            'fan_apex_pad_mm': [float(apex[0]), float(apex[1])],
            'radius_range_mm': [float(r_lo), float(r_hi)],
            'phi_range_deg': [float(phi_lo), float(phi_hi)],
            'instrumented_radius_range_mm': [float(ri_lo), float(ri_hi)],
            'instrumented_phi_range_deg': [float(phii_lo), float(phii_hi)],
        },
        'spot': {
            'centre_pad_mm': [float(mx), float(my)],
            'radius_mm': float(r_beam), 'phi_deg': float(phi_beam),
            'sigma_major_mm': float(s_maj), 'sigma_minor_mm': float(s_min),
            'major_axis_deg': float(ang),
            'fwhm_along_mm': float(fw_maj), 'fwhm_across_mm': float(fw_min),
            'core_centre_pad_mm': [float(kx), float(ky)],
            'core_sigma_major_mm': float(k_maj),
            'core_sigma_minor_mm': float(k_min),
            'core_axis_deg': float(k_ang), 'core_fraction': float(k_frac),
        },
        'distance_to_board_edge': {k: float(v) for k, v in edges.items()},
        'band': {
            'best_normal_deg': int(best_normal),
            'det4_band_mm': DET4_BAND_MM,
            'det4_fraction': float(f_det4),
            'containment_vs_width': contain,
            'centring_offset_mm': {str(k): [float(a) for a in v]
                                   for k, v in centring.items()},
        },
    }
    js = os.path.join(args.out, 'sps_beam_board.json')
    with open(js, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', js)

    print(f'\nbeam centre  pad ({mx:.1f}, {my:.1f}) mm   '
          f'r={r_beam:.1f} mm  phi={phi_beam:.2f} deg')
    print(f'sigma        major {s_maj:.1f} @ {ang:.1f} deg, minor {s_min:.1f} mm'
          f'   FWHM {fw_maj:.0f} x {fw_min:.0f} mm')
    print('to board edge: ' + '  '.join(f'{k}={v:.0f}' for k, v in edges.items()))
    print(f'det4 {DET4_BAND_MM:.0f} mm band, best placement: '
          f'{100 * f_det4:.1f}% of triggers')


if __name__ == '__main__':
    main()
