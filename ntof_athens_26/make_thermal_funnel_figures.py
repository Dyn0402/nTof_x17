#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_thermal_funnel_figures.py -- slide 40 onward: where the thermal neutrons go.

    ../.venv/bin/python make_thermal_funnel_figures.py
    ../.venv/bin/python make_thermal_funnel_figures.py --only F3
    ../.venv/bin/python make_thermal_funnel_figures.py --accounting <accounting.json>

Five figures, one step of the funnel each (HANDOFF_THERMAL_ACCOUNTING.md §0),
written as figures/thermal_funnel_F{1..5}.{png,pdf,csv}.  Every figure has the
same layout: a diagram of the step on the left, and on the right ONE split of
its parent -- fractions and counts -- with the segment the next figure opens up
in copper.

    F1  what a neutron entering the capsule does          (n,p) vs (n,γ)
    F2  does the capture put charge in a drift gap        γ only vs charged
    F3  where that charged particle was made              the capsule's pairs vs the rest
    F4  which reaction made the capsule pair              Al / CFRP / ³He, external + internal
    F5  what fires a trigger leg                          capsule pair vs other, other broken down

NOTHING IS COMPUTED HERE.  The numbers are the contract
`data/thermal_accounting/accounting.json`, written by
`MX17_Full_Geant/scripts/thermal_accounting.py merge` from the Geant4 truth of the
nose-first thermal campaign plus the analytic internal-pair yields.  Counts are
per 30 days at slide 39's flux (the rate table's neutrons on the cell), the
normalisation `thermal_branching` panel (c) uses, so the slides agree; the
Geant4 campaign's own normalisation is in the JSON beside it.  F5 is per pulse
of the Geant4 campaign, and its fractions are the result: the sim makes 2.1x
fewer legs than the data.

The diagrams are drawn from the same survey as the fans slide
(`make_overhead_figure`: run_145's run_config transforms, SiPM wall, plastics)
and the capsule from the STEP-derived profiles in MX17_Full_Geant's
DetectorConstruction.cc.  Tracks in them are illustrations, not events.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle           # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plotstyle as P                                               # noqa: E402
import make_overhead_figure as M                                    # noqa: E402

ACCOUNTING = HERE / 'data' / 'thermal_accounting' / 'accounting.json'
OUT = HERE / 'figures'
FIGSIZE = (13.35, 6.22)            # the slide's 2.15:1 hole, as the fans slide

# ---- colour, by the job it does ------------------------------------------- #
HL = P.COPPER                      # the segment the next figure opens up
INVIS = '#e3e7ec'                  # what we cannot see
GREY = '#b3bcc7'                   # the rest, not followed further
GREY2 = '#cfd5dc'
WALL = '#5b6b7d'                   # the capsule wall
CFRP_INK = '#2f3640'
GAS = '#eadcec'                    # the ³He, as a pale fill of P.ACCENT
#: nucleus identity -- a categorical set, validated (dataviz validate_palette,
#: light, #fbfcfe): ALL CHECKS PASS, worst adjacent CVD ΔE 16.0.  Every use
#: carries a text label.
C_AL, C_CFRP, C_HE3 = '#0072B2', '#009E73', P.ACCENT

# ---- the capsule, nose-first: world y = local z ---------------------------- #
#: MX17_Full_Geant src/DetectorConstruction.cc (STEP sectioning), mm
Z_VESSEL = np.array([-35, -34, -33, -31, -29, -27, -25, -23, -21, -20, -15, -5, 5,
                     15, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 40, 45, 50, 51],
                    float)
R_AL = np.array([0.000, 3.803, 5.287, 7.206, 8.480, 9.375, 9.994, 10.386, 10.600,
                 10.600, 10.600, 10.600, 10.600, 10.600, 10.600, 10.600, 10.386,
                 9.994, 9.375, 8.480, 7.206, 5.747, 4.708, 4.015, 3.621, 3.500,
                 3.500, 3.500, 3.500])
R_CFRP = np.array([0.000, 4.703, 6.187, 8.106, 9.380, 10.275, 10.894, 11.286,
                   11.500, 11.500, 11.500, 11.500, 11.500, 11.500, 11.500, 11.500,
                   11.286, 10.894, 10.275, 9.380, 8.106, 6.647, 5.608, 4.915,
                   4.521, 4.400, 4.400, 4.400, 4.400])


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def sci(v: float, d: int = 1) -> str:
    """2.79e12 -> '2.8×10¹²'; plain between 0.01 and 10⁴; '10⁹' for a power."""
    if not v:
        return '0'
    ex = int(np.floor(np.log10(abs(v))))
    man = v / 10 ** ex
    if round(man, d) >= 10:
        man, ex = man / 10, ex + 1
    if 0 <= ex <= 3:
        return f'{v:,.0f}'.replace(',', ' ')
    if -2 <= ex < 0:
        return f'{v:.{d - ex}f}'
    if abs(man - 1.0) < 1e-9:
        return f'10{str(ex).translate(SUP)}'
    return f'{man:.{d}f}×10{str(ex).translate(SUP)}'


def pct(f: float) -> str:
    p = 100.0 * f
    if p == 0:
        return '0 %'
    if 99.5 <= p < 100:
        return f'{p:.2f} %'
    if p >= 1:
        return f'{p:.1f} %'
    if p >= 0.01:
        return f'{p:.2f} %'
    return f'{sci(p)} %'


class Accounting:
    def __init__(self, path: Path):
        self.d = json.loads(Path(path).read_text(encoding='utf-8'))
        self.by = {n['id']: n for n in self.d['nodes'] + self.d['trigger_nodes']}

    def v(self, ids, key: str) -> float:
        ids = [ids] if isinstance(ids, str) else ids
        return float(sum((self.by.get(i) or {}).get(key) or 0.0 for i in ids))


def _ids(ids):
    return ids if isinstance(ids, list) else [ids]


def seg(A, ids, label, color, parent_pn, sub='', hl=False, hatch=None, keep=False):
    """One segment of a per-neutron split (F1-F4): fraction from per_neutron, so
    analytic and Geant4 nodes are on the same footing."""
    pn = A.v(ids, 'per_neutron')
    return dict(label=label, sub=sub, color=color, hatch=hatch, hl=hl, keep=keep,
                ids=_ids(ids), frac=pn / parent_pn, count=A.v(ids, 'per30d_ratetable'),
                per_neutron=pn, mc=A.v(ids, 'mc_count'),
                source='+'.join(sorted({(A.by.get(i) or {}).get('source', '-')
                                        for i in _ids(ids)})))


def seg5(A, ids, label, color, parent_mc, sub='', hl=False, keep=False):
    """One segment of the trigger split (F5): fraction of legs, count per pulse."""
    mc = A.v(ids, 'mc_count')
    return dict(label=label, sub=sub, color=color, hatch=None, hl=hl, keep=keep,
                ids=_ids(ids), frac=mc / parent_mc if parent_mc else 0.0,
                count=A.v(ids, 'per_pulse'), mc=mc, per_neutron=A.v(ids, 'per_neutron'),
                source='geant4')


# --------------------------------------------------------------------------- #
# the right-hand split
# --------------------------------------------------------------------------- #
def split_panel(fig, x0, y_top, w, title, segs, unit, small=False):
    """A 100 % bar and one row per segment beneath it.  Returns the y it ends at.

    Every segment's identity is its text row (swatch + label); the numbers are
    in ink, never in the segment colour.  A sliver too thin to see is drawn at
    a floor width taken from the largest segment.
    """
    segs = [s for s in segs if s['frac'] > 0 or s['keep']]
    fs = 11.0 if small else 12.5
    xp = x0 + 0.78 * w                                  # the share column
    fig.text(x0, y_top, title, ha='left', va='top', fontsize=fs, fontweight='bold',
             color=P.INK)
    for x, t in ((xp, 'share'), (x0 + w, unit)):
        fig.text(x, y_top, t, ha='right', va='top', fontsize=9.5, color=P.MUTED)
    bh = 0.045 if small else 0.075
    yb = y_top - 0.042 - bh
    ax = fig.add_axes([x0, yb, w, bh])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    floor = 0.004
    widths = np.array([max(s['frac'], floor) for s in segs])
    widths[int(np.argmax(widths))] -= widths.sum() - 1.0
    left = 0.0
    for s, wd in zip(segs, widths):
        ax.add_patch(Rectangle((left, 0), wd, 1, facecolor=s['color'],
                               edgecolor=C_AL if s['hatch'] else 'none',
                               hatch=s['hatch'], lw=0))
        left += wd
    for x in np.cumsum(widths)[:-1]:                 # 2 px surface gaps
        ax.axvline(x, color=P.SURFACE, lw=2.2)

    rh = 0.044 if small else 0.058
    y = yb - 0.022
    for s in segs:
        sub, sub_col = s['sub'], P.MUTED
        if s['hl']:
            sub, sub_col = '→ the next figure opens this up', HL
        fig.add_artist(Rectangle((x0, y - 0.024), 0.016, 0.024, transform=fig.transFigure,
                                 facecolor=s['color'], hatch=s['hatch'],
                                 edgecolor=C_AL if s['hatch'] else 'none', lw=0))
        wt = 'bold' if s['hl'] else 'normal'
        fig.text(x0 + 0.024, y, s['label'], ha='left', va='top', fontsize=fs,
                 fontweight=wt, color=P.INK)
        if sub:
            fig.text(x0 + 0.024, y - (0.027 if small else 0.030), sub, ha='left', va='top',
                     fontsize=fs - 2.0, color=sub_col, fontweight='bold' if s['hl'] else 'normal')
        fig.text(xp, y, pct(s['frac']), ha='right', va='top', fontsize=fs,
                 fontweight=wt, color=P.INK)
        fig.text(x0 + w, y, sci(s['count']), ha='right', va='top', fontsize=fs,
                 fontweight=wt, color=P.INK)
        y -= rh * (1.55 if sub else 1.0)
    return y


def frame(headline, sub):
    P.use()
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.006, 0.975, headline, ha='left', va='top', fontsize=19,
             fontweight='bold', color=P.INK)
    fig.text(0.006, 0.905, sub, ha='left', va='top', fontsize=11, color=P.MUTED)
    return fig


def diagram_axes(fig, rect=(0.006, 0.02, 0.46, 0.84)):
    ax = fig.add_axes(list(rect))
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def save(fig, stem, segs_by_panel):
    OUT.mkdir(exist_ok=True)
    M.save(fig, str(OUT / stem))
    rows = []
    for panel, segs in segs_by_panel.items():
        for s in segs:
            rows.append(dict(panel=panel, label=s['label'].replace('\n', ' '),
                             node_ids=' + '.join(s['ids']), frac_of_parent=s['frac'],
                             count=s['count'], mc_count=s['mc'],
                             per_neutron=s['per_neutron'], source=s['source'],
                             highlighted=s['hl']))
    pd.DataFrame(rows).to_csv(OUT / f'{stem}.csv', index=False)
    print(f'  -> {OUT / stem}.csv')


def n_sim(A) -> str:
    return sci(A.d['normalisation']['neutrons_simulated'])


# --------------------------------------------------------------------------- #
# diagram pieces
# --------------------------------------------------------------------------- #
def _mirror(r, z):
    return np.r_[r, -r[::-1]], np.r_[z, z[::-1]]


def capsule_side(ax, al=WALL, cfrp=CFRP_INK, gas=GAS, z=2):
    """The capsule from the side, beam up the page, to scale."""
    from ntof_tracking.reco import geometry as G
    for r, zz, c in ((R_CFRP, Z_VESSEL, cfrp), (R_AL, Z_VESSEL, al),
                     (np.asarray(G.HE3_GAS_R), np.asarray(G.HE3_GAS_Y), gas)):
        xs, ys = _mirror(r, zz)
        ax.fill(xs, ys, facecolor=c, edgecolor='none', zorder=z)


#: Where ³He(n,p) happens: Geant4 truth (EventTree cap_x, cap_y, cap_z of
#: He3Gas neutronInelastic), 10 files of neutrons_thermal_trig_2cm_nose =
#: 8.3e7 absorptions, histogrammed on lxplus 2026-09-15.  H_slice is a
#: |z| < 1 mm section through the capsule axis in 0.2 mm bins; H_depth is the
#: distance along the beam past the lower gas dome, over all captures.
#: ``*.npz`` is gitignored, so a fresh clone (the Windows box) has no repo copy:
#: it falls back to the staged copy on the data disk, found through `paths` --
#: set X17_ROOT (or X17_SEPT26_OUT) to where that disk is mounted.
GAS_NP = HERE / 'data' / 'thermal_accounting' / 'gas_np_positions_nose.npz'
GAS_NP_STAGED = ('athens_thermal_accounting', 'gas_np_positions_nose.npz')


def gas_np_path() -> Path:
    if GAS_NP.is_file():
        return GAS_NP
    from sept26_prelim_analysis import paths
    staged = paths.spell('out', *GAS_NP_STAGED)
    if staged.is_file():
        return staged
    raise FileNotFoundError(f'no gas (n,p) histogram: tried {GAS_NP} and {staged} '
                            '(set X17_ROOT to the data disk)')


def gas_stopping(ax, path=None):
    """The gas coloured by where the (n,p) happens.  Returns (mappable, quantiles)."""
    from ntof_tracking.reco import geometry as G
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    d = np.load(path or gas_np_path())
    H = d['H_slice'].T                                   # (y, x)
    # empty bins sit below the colour range and take the gas's own pale fill,
    # so smoothing never invents an edge between "no data" and "little"
    rel = np.where(H > 0, H / H.max(), 1e-9)
    # one hue, light -> dark, starting at the gas's own pale fill
    cmap = LinearSegmentedColormap.from_list('np_density',
                                             [GAS, '#c99ccd', P.ACCENT, '#3f1a43'])
    cmap.set_under(GAS)
    xs, ys = _mirror(np.asarray(G.HE3_GAS_R), np.asarray(G.HE3_GAS_Y))
    clip = Polygon(np.c_[xs, ys], closed=True, facecolor='none', edgecolor='none')
    ax.add_patch(clip)
    # 2.5 decades: wide enough to show the tail into the gas, narrow enough
    # that the sub-millimetre skin is what the eye lands on
    im = ax.imshow(rel, extent=(d['xe'][0], d['xe'][-1], d['ye'][0], d['ye'][-1]),
                   origin='lower', cmap=cmap, norm=LogNorm(3e-3, 1.0),
                   interpolation='bilinear', zorder=3, rasterized=True, aspect='auto')
    im.set_clip_path(clip)
    c = np.cumsum(d['H_depth']) / d['H_depth'].sum()
    q = {p: float(np.interp(p, c, d['de'][1:])) for p in (0.5, 0.9, 0.99)}
    return im, q


def gas_surface_y(r):
    """The lower gas dome: y of the gas surface a neutron at radius r meets."""
    from ntof_tracking.reco import geometry as G
    return float(np.interp(r, np.asarray(G.HE3_GAS_R)[:6], np.asarray(G.HE3_GAS_Y)[:6]))


def capsule_top(ax, c=(0.0, 0.0), al=WALL, cfrp=CFRP_INK, gas=GAS, z=8):
    """The capsule barrel from above: CFRP 11.5, Al 10.6, gas 10.0 mm."""
    for r, col in ((11.5, cfrp), (10.6, al), (10.0, gas)):
        ax.add_patch(Circle(c, r, facecolor=col, edgecolor='none', zorder=z))


def ray(ax, p0, p1, color, dashed=False, lw=1.8, head=True, z=9, ms=13):
    ax.annotate('', xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle='-|>' if head else '-', color=color,
                                lw=lw, linestyle=(0, (4, 3)) if dashed else '-',
                                shrinkA=0, shrinkB=0, mutation_scale=ms))


def star(ax, xy, color, ms=15, z=11):
    ax.plot(*xy, marker=(8, 2, 0), ms=ms, mew=2.0, color=color, zorder=z)


def label(ax, xy, text, color=P.INK, ha='left', va='center', fs=11, bold=False, z=12,
          box=True):
    ax.text(*xy, text, ha=ha, va=va, fontsize=fs, color=color, zorder=z,
            fontweight='bold' if bold else 'normal', linespacing=1.25,
            bbox=dict(facecolor=P.SURFACE, edgecolor='none', alpha=0.85, pad=1.5)
            if box else None)


def leader(ax, p0, p1, color=P.MUTED):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=0.8, zorder=9)


def scale_bar(ax, x, y, length, text):
    ax.plot([x, x + length], [y, y], color=P.INK, lw=2.4, solid_capstyle='butt', zorder=12)
    ax.text(x + length / 2, y + 0.012 * np.ptp(ax.get_ylim()), text, ha='center',
            va='bottom', fontsize=10, color=P.INK, zorder=12)


_STATION = {}


def station():
    """run_145's surveyed transforms, SiPM walls and plastic bars (cached)."""
    if not _STATION:
        from sept26_prelim_analysis import source_imaging as SI
        trs = SI.transforms(M.RUN)
        _STATION.update(trs=trs, wall=M.sipm_wall(trs), bars=M.plastics(trs))
    return _STATION


def arm_pt(arm, u, d):
    """(u along the strips, d outward past the strip plane) -> overhead (Z, X)."""
    tr = station()['trs'][f'mx17_{arm}']
    return M._local_poly(tr, [(u, 0.0, d)], M.OVERHEAD)[0]


def draw_station(ax, arms=('A', 'B', 'C', 'D'), scint=True, lit=None):
    """Drift gaps, and behind them the SiPM wall and plastics; ``lit`` =
    {arm: [(wall_group_index, plastic_index), ...]} outlines what legs fired."""
    S = station()
    lit = lit or {}
    for a in arms:
        tr = S['trs'][f'mx17_{a}']
        ax.add_patch(Polygon(M._box(tr, (-M.PLANE_HALF, M.PLANE_HALF), (0, 0),
                                    (-M.DRIFT_GAP_MM, 0.0), M.OVERHEAD), closed=True,
                             facecolor=P.LINE, edgecolor=P.MUTED, lw=0.8, zorder=3))
        if not scint or a not in S['bars']:
            continue
        on_g = {g for g, _ in lit.get(a, [])}
        on_b = {b for _, b in lit.get(a, [])}
        for g, poly in enumerate(S['wall'][a]['groups']):
            ax.add_patch(Polygon(poly, closed=True, facecolor='#c3cad3',
                                 edgecolor=P.INK if g in on_g else 'none',
                                 lw=1.6 if g in on_g else 0, zorder=4))
        for b, poly in enumerate(S['bars'][a]['polys']):
            ax.add_patch(Polygon(poly, closed=True, facecolor='#dde2e8',
                                 edgecolor=P.INK if b in on_b else '#aab3be',
                                 lw=1.6 if b in on_b else 0.7, zorder=4))


# --------------------------------------------------------------------------- #
# F1
# --------------------------------------------------------------------------- #
def f1(A):
    root = A.v('root', 'per_neutron')
    ng = A.v('F1.ncapture', 'per_neutron')
    segs = [
        seg(A, 'F1.he3_np', '³He(n,p) → p + t', INVIS, root,
            sub='invisible to us: both stop in the gas'),
        seg(A, 'F1.ncapture', '(n,γ) capture', HL, root, hl=True),
        seg(A, 'F1.no_absorption', 'scatters off the wall, escapes', GREY, root),
        seg(A, 'F1.other', 'other', GREY2, root),
    ]
    zoom = [
        seg(A, 'F1.ncapture.wall_al', '²⁷Al vessel', C_AL, ng),
        seg(A, 'F1.ncapture.wall_cfrp', 'carbon-fibre wrap', C_CFRP, ng),
        seg(A, 'F1.ncapture.elsewhere', 'scattered out, captured elsewhere', GREY, ng),
        seg(A, 'F1.ncapture.he3', '³He(n,γ)', C_HE3, ng, keep=True,
            sub='analytic, self-shielded'),
    ]
    fig = frame('Almost every neutron makes a proton we cannot see',
                f'Geant4, {n_sim(A)} EAR2 neutrons below 2 eV · per neutron entering the '
                'capsule · counts per 30 days at slide 39’s flux')

    ax = diagram_axes(fig)
    ax.set_xlim(-66, 66)
    ax.set_ylim(-68, 60)
    capsule_side(ax)
    im, q = gas_stopping(ax)
    y0 = -58
    # 1: through the nose -> (n,p) in the first fraction of a millimetre of gas
    ray(ax, (-3.5, y0), (-3.5, gas_surface_y(3.5)), P.INK, lw=1.4)
    label(ax, (-64, -20),
          f'³He(n,p) happens here:\nhalf within {q[0.5]:.1f} mm\nof the gas surface,\n'
          f'90 % within {q[0.9]:.1f} mm', color=P.INK, fs=10.5, bold=True)
    leader(ax, (-33, -26), (-8.2, gas_surface_y(8.2) + 0.4), P.INK)
    # the stopping layer is a few pixels at this scale: magnify the dome
    axin = ax.inset_axes([0.585, 0.075, 0.40, 0.16])
    capsule_side(axin)
    gas_stopping(axin)
    axin.set_xlim(-6.0, 6.0)
    axin.set_ylim(-30.6, -26.1)
    axin.set_aspect('equal')
    axin.set_xticks([])
    axin.set_yticks([])
    for sp in axin.spines.values():
        sp.set_color(P.MUTED)
        sp.set_linewidth(0.8)
    axin.plot([3.4, 5.4], [-26.6, -26.6], color=P.INK, lw=2.0, solid_capstyle='butt',
              zorder=12)
    axin.text(4.4, -26.75, '1 mm', ha='center', va='top', fontsize=9, color=P.INK, zorder=12)
    ax.indicate_inset_zoom(axin, edgecolor=P.MUTED, alpha=0.9, lw=0.8)
    # the colour key: relative (n,p) density in a 2 mm section through the axis
    cax = ax.inset_axes([0.015, 0.74, 0.27, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='neither')
    cb.set_ticks([1e-2, 1e-1, 1.0])
    cb.set_ticklabels(['10⁻²', '10⁻¹', '1'])
    cb.ax.tick_params(labelsize=9.5, length=2, color=P.MUTED, labelcolor=P.MUTED)
    cb.outline.set_visible(False)
    cax.set_title('(n,p) density, relative · log\na 2 mm section, Geant4', fontsize=9.5,
                  color=P.MUTED, loc='left', pad=4)
    # 2: into the Al -> capture, 7.7 MeV γ out
    ray(ax, (10.3, y0), (10.3, -8), P.INK, lw=1.4)
    star(ax, (10.3, -8), HL)
    ray(ax, (10.3, -8), (50, 18), HL, dashed=True)
    label(ax, (64, 30), '(n,γ) in the wall:\n7.7 MeV γ cascade', color=P.INK, ha='right',
          fs=10.5, bold=True)
    # 3: scatters off the nose and leaves
    ray(ax, (3.5, y0), (3.5, -32.2), GREY, lw=1.2, head=False)
    ray(ax, (3.5, -32.2), (-46, -50), GREY, lw=1.2)
    label(ax, (-64, -54), 'scatters off the nose,\nescapes', color=P.MUTED, fs=10.5)
    ax.text(0, 25, '³He\n500 atm', ha='center', va='center', fontsize=10.5,
            color=P.ACCENT, fontweight='bold', zorder=12)
    label(ax, (-26, 8), 'Al', color=WALL, fs=10.5, bold=True, ha='right')
    leader(ax, (-25, 8), (-10.8, 4), WALL)
    label(ax, (22, 46), 'carbon fibre', color=CFRP_INK, fs=10.5, bold=True)
    leader(ax, (21, 44), (11.3, 18), CFRP_INK)
    ax.text(0, -64, 'neutrons ↑', ha='center', va='center', fontsize=10.5, color=P.INK)
    scale_bar(ax, -64, 50, 20, '20 mm')

    y = split_panel(fig, 0.52, 0.80, 0.46, 'What the neutron does', segs, 'per 30 days')
    split_panel(fig, 0.52, y - 0.03, 0.46, 'where the (n,γ) happens', zoom,
                'per 30 days', small=True)
    save(fig, 'thermal_funnel_F1', dict(main=segs, zoom=zoom))


# --------------------------------------------------------------------------- #
# F2
# --------------------------------------------------------------------------- #
def f2(A):
    ng = A.v('F1.ncapture', 'per_neutron')
    ch = A.v('F2.charged_in_detector', 'per_neutron')
    segs = [
        seg(A, 'F2.gamma_only', 'γ only: no charge in a chamber', INVIS, ng),
        seg(A, 'F2.charged_in_detector', 'charge in a drift gap', HL, ng, hl=True),
    ]
    zoom = [
        seg(A, 'F2.by.al27.charged', '²⁷Al vessel', C_AL, ch),
        seg(A, 'F2.by.cfrp.charged', 'carbon-fibre wrap', C_CFRP, ch),
        seg(A, 'F2.by.capture_elsewhere.charged', 'captured elsewhere', GREY, ch),
    ]
    delayed = A.v('F2.delayed_only', 'per_neutron')
    fig = frame('Most capture γ fly straight through',
                'Geant4 · per (n,γ) capture · charge = ≥ 1 keV from charged tracks in one '
                'Micromegas drift gap, within 100 ms')
    ax = diagram_axes(fig)
    L = 455
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    draw_station(ax)
    capsule_top(ax)
    for ang in (28, 68, 118, 162, 212, 250, 298, 335):
        t = np.radians(ang)
        ray(ax, (0, 0), (430 * np.cos(t), 430 * np.sin(t)), GREY, dashed=True, lw=1.4)
    # one that stops in A's drift gap and knocks out an electron
    e0, e1 = arm_pt('A', 55, -24), arm_pt('A', 120, 6)
    ray(ax, (0, 0), e0, HL, dashed=True, lw=1.6)
    ray(ax, e0, e1, HL, lw=2.6)
    label(ax, (e0[0] - 20, e0[1] - 95), 'an e⁻ in\nthe drift gap', color=P.INK, ha='right',
          bold=True, fs=10.5)
    for a in 'ABCD':                     # between each arm's SiPM wall and plastics
        label(ax, arm_pt(a, 0, 145), f'arm {a}', color=P.MUTED, ha='center', fs=10.5,
              bold=True)
    label(ax, (0, -45), 'capsule', color=P.INK, ha='center', fs=10.5, bold=True)
    label(ax, (-450, -430), 'from above, to scale · beam into the page',
          color=P.MUTED, fs=9.5)
    scale_bar(ax, 330, -440, 100, '100 mm')

    y = split_panel(fig, 0.52, 0.80, 0.46, 'Does the capture put charge in a chamber?',
                    segs, 'per 30 days')
    y = split_panel(fig, 0.52, y - 0.03, 0.46, 'the ones that do, by capture site', zoom,
                    'per 30 days', small=True)
    fig.text(0.52, y - 0.02, f'Not counted: charge only from ²⁸Al β decays seconds to hours '
             f'later ({pct(delayed / ng)} of captures),\nwhich the data sees at random '
             'times, not with the pulse.', ha='left', va='top', fontsize=9.5, color=P.MUTED)
    save(fig, 'thermal_funnel_F2', dict(main=segs, zoom=zoom))


# --------------------------------------------------------------------------- #
# F3
# --------------------------------------------------------------------------- #
def f3(A):
    ch = A.v('F2.charged_in_detector', 'per_neutron')
    segs = [
        seg(A, 'F3.pair_near_capsule', 'e⁺e⁻ pair made in the capsule', HL, ch, hl=True),
        seg(A, 'F3.compton_near_capsule', 'Compton e⁻ made in the capsule', WALL, ch),
        seg(A, ['F3.compton_elsewhere', 'F3.pair_elsewhere'], 'made elsewhere', GREY, ch,
            sub='chamber windows and PCB, the gas itself, air'),
        seg(A, ['F3.neutron_induced', 'F3.other'], 'other', GREY2, ch),
    ]
    fig = frame('What we see: made at the capsule, or somewhere else',
                'Geant4 · per capture that puts charge in a drift gap · attributed to the '
                'track, with its δ-rays, that deposits the most')
    axT = diagram_axes(fig, (0.006, 0.44, 0.46, 0.42))
    axT.set_xlim(-70, 70)
    axT.set_ylim(-30, 30)
    capsule_top(axT)
    star(axT, (-7.6, -7.2), WALL, ms=12)
    ray(axT, (-7.6, -7.2), (8.3, 6.6), WALL, dashed=True, lw=1.4, ms=10)
    ray(axT, (8.3, 6.6), (58, 24), HL, lw=2.4)
    ray(axT, (8.3, 6.6), (60, -4), HL, lw=2.4)
    label(axT, (60, 24), 'e⁺', color=P.INK, ha='left', bold=True)
    label(axT, (62, -4), 'e⁻', color=P.INK, ha='left', bold=True)
    label(axT, (-68, 22), 'made at the capsule:\nthe capture γ converts in the wall',
          fs=10.5, bold=True)
    label(axT, (-68, -22), 'the capsule from above, to scale', color=P.MUTED, fs=9.5)

    axB = diagram_axes(fig, (0.006, 0.02, 0.46, 0.40))
    axB.set_xlim(-70, 70)
    axB.set_ylim(-30, 30)
    # a chamber in section, not to scale: window, drift gap, mesh + PCB
    sx = 14.0
    for x0, w_, c in ((sx, 2.2, '#9aa4b0'), (sx + 2.2, 26, P.LINE),
                      (sx + 28.2, 1.4, P.MUTED), (sx + 29.6, 9, '#c3cad3')):
        axB.add_patch(Rectangle((x0, -24), w_, 48, facecolor=c, edgecolor='none', zorder=3))
    axB.text(sx + 1.1, 25.5, 'window', ha='center', va='bottom', fontsize=9.5, color=P.MUTED)
    axB.text(sx + 15.2, -26.5, 'drift gap', ha='center', va='top', fontsize=9.5,
             color=P.MUTED)
    axB.text(sx + 34.1, -26.5, 'PCB', ha='center', va='top', fontsize=9.5, color=P.MUTED)
    ray(axB, (-40, -16), (sx + 1.1, -4), GREY, dashed=True, lw=1.4, ms=10)
    ray(axB, (sx + 1.1, -4), (sx + 24, 12), P.INK, lw=2.4)
    label(axB, (-68, 16), 'made elsewhere:\na γ Compton-scatters\nin a chamber window',
          fs=10.5, bold=True)
    label(axB, (-68, -24), 'a chamber in section, not to scale', color=P.MUTED, fs=9.5)

    split_panel(fig, 0.52, 0.80, 0.46, 'Where the charged particle was made', segs,
                'per 30 days')
    save(fig, 'thermal_funnel_F3', dict(main=segs))


# --------------------------------------------------------------------------- #
# F4
# --------------------------------------------------------------------------- #
def f4(A):
    par = A.v('F4.capsule_pairs', 'per_neutron')
    segs = [
        seg(A, 'F4.external.al27', '²⁷Al: γ converts in the wall', C_AL, par,
            sub='external conversion · Geant4'),
        seg(A, 'F4.internal.al27', '²⁷Al: internal pair', '#cfe3f1', par, hatch='////',
            sub='analytic: Geant4 makes no internal pairs'),
        seg(A, ['F4.external.cfrp', 'F4.internal.cfrp'], 'carbon fibre: ¹H + ¹²C',
            C_CFRP, par),
        seg(A, 'F4.external.capture_elsewhere', 'γ from a capture elsewhere', GREY, par),
        seg(A, ['F4.external.he3', 'F4.internal.he3'], '³He(n,γ): internal pair',
            C_HE3, par, keep=True, sub='the signal channel · analytic, self-shielded'),
    ]
    fig = frame('Which nucleus made the pair',
                'capsule e⁺e⁻ pairs with a lepton in a drift gap · Geant4 external conversion '
                '+ internal pair creation added analytically')
    ax = diagram_axes(fig)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-68, 60)
    capsule_side(ax, al=C_AL, cfrp=C_CFRP)
    # internal: a pair straight from the nucleus
    star(ax, (-10.3, 6), P.INK, ms=13)
    ray(ax, (-10.3, 6), (-50, 26), C_AL, lw=2.4)
    ray(ax, (-10.3, 6), (-52, 8), C_AL, lw=2.4)
    label(ax, (-78, 44), 'internal pair creation:\ne⁺e⁻ straight from the nucleus',
          fs=10.5, bold=True)
    # external: γ crosses, converts in the far wall
    star(ax, (-10.3, -12), P.INK, ms=13)
    ray(ax, (-10.3, -12), (10.6, -2), P.INK, dashed=True, lw=1.4, ms=10)
    ray(ax, (10.6, -2), (52, 8), C_AL, lw=2.4)
    ray(ax, (10.6, -2), (50, -18), C_AL, lw=2.4)
    label(ax, (78, -34), 'external conversion:\nthe γ converts in the wall',
          fs=10.5, bold=True, ha='right')
    # ³He
    star(ax, (0, -20), C_HE3, ms=12)
    ray(ax, (0, -20), (-7, -46), C_HE3, lw=1.8, ms=9)
    ray(ax, (0, -20), (8, -47), C_HE3, lw=1.8, ms=9)
    label(ax, (-78, -56), '³He(n,γ)⁴He:\n1 per 10⁸ neutrons', color=P.INK, fs=10.5, bold=True)
    label(ax, (22, 34), 'Al', color=C_AL, fs=11, bold=True)
    leader(ax, (20, 33), (10.8, 16), C_AL)
    label(ax, (22, 50), 'carbon fibre', color=C_CFRP, fs=11, bold=True)
    leader(ax, (20, 48), (11.4, 20), C_CFRP)
    ax.text(0, 25, '³He', ha='center', va='center', fontsize=11, color=P.ACCENT,
            fontweight='bold', zorder=12)
    scale_bar(ax, 50, -66, 20, '20 mm')

    split_panel(fig, 0.52, 0.80, 0.46, 'Where the capsule pairs come from', segs,
                'per 30 days')
    save(fig, 'thermal_funnel_F4', dict(main=segs))


# --------------------------------------------------------------------------- #
# F5
# --------------------------------------------------------------------------- #
F5_BUCKETS = ('F5.pair_near_capsule', 'F5.compton_capsule_capture',
              'F5.conv_structure_capsule_capture', 'F5.capture_outside_capsule',
              'F5.neutron_induced', 'F5.mixed', 'F5.other', 'F5.activation')


def f5(A):
    legs = A.v('F5.legs', 'mc_count')
    pair = A.v('F5.pair_near_capsule', 'mc_count')
    rest = [i for i in A.by if i.count('.') == 1 and i in F5_BUCKETS[1:]]
    segs = [
        seg5(A, 'F5.pair_near_capsule', 'e⁺e⁻ pair made in the capsule', HL, legs, hl=True),
        seg5(A, rest, 'anything else', GREY, legs),
    ]
    other = legs - pair
    zoom = [
        seg5(A, 'F5.compton_capsule_capture', 'Compton from a capsule-capture γ', WALL, other),
        seg5(A, 'F5.conv_structure_capsule_capture', 'capsule γ converting in structure',
             GREY, other),
        seg5(A, 'F5.capture_outside_capsule', 'γ from a capture outside', GREY2, other),
        seg5(A, ['F5.neutron_induced', 'F5.mixed', 'F5.other', 'F5.activation'],
             'mixed, neutron-induced, other', INVIS, other),
    ]
    fig = frame('What fires a trigger leg',
                'Geant4 · SiPM-wall bar ∧ plastic bar ≥ 0.5 MIP within 20 ns, in one arm · '
                'the sim makes 2.1× fewer legs than the data: read the fractions')
    ax = diagram_axes(fig)
    ax.set_xlim(-40, 470)
    ax.set_ylim(-255, 255)
    draw_station(ax, arms=('A',), lit={'A': [(3, 1), (1, 0)]})
    capsule_top(ax)
    ray(ax, (0, 0), arm_pt('A', 150, 205), HL, lw=2.6)
    label(ax, arm_pt('A', 160, 10), 'a capsule-pair\nlepton', ha='center', bold=True, fs=10.5)
    g = arm_pt('A', -30, 40)
    ray(ax, (0, 0), g, GREY, dashed=True, lw=1.4)
    ray(ax, g, arm_pt('A', -120, 200), WALL, lw=2.6)
    label(ax, arm_pt('A', -150, 10), 'a Compton e⁻\nfrom a capture γ', ha='center',
          bold=True, fs=10.5)
    label(ax, arm_pt('A', 222, -15), 'drift gap', color=P.MUTED, ha='center', fs=9.5)
    label(ax, arm_pt('A', 222, 97), 'SiPM wall', color=P.MUTED, ha='center', fs=9.5)
    label(ax, arm_pt('A', 222, 200), 'plastics', color=P.MUTED, ha='center', fs=9.5)
    label(ax, (-35, -235), 'one arm, from above, to scale\noutlined: what each leg fired',
          color=P.MUTED, fs=9.5)
    label(ax, (0, -32), 'capsule', color=P.INK, ha='center', fs=10, bold=True)
    scale_bar(ax, 340, -250, 100, '100 mm')

    y = split_panel(fig, 0.52, 0.80, 0.46, 'What made the leg', segs, 'per pulse')
    y = split_panel(fig, 0.52, y - 0.03, 0.46, '“anything else”, broken down', zoom,
                    'per pulse', small=True)
    pt = A.v('F5.pairtags', 'mc_count')
    if pt >= 10:
        one = A.v(['F5.pairtags.relation.one_particle', 'F5.pairtags.relation.genuine_pair'],
                  'mc_count')
        txt = (f'Legs in two arms at once: {sci(A.v("F5.pairtags", "per_pulse"), 2)} per pulse '
               f'({int(pt)} in {n_sim(A)} neutrons). {pct(one / pt)} are one pair or one\n'
               'particle; the rest are two γ of one capture. One neutron per event: the '
               'sim has no accidentals.')
    else:
        txt = f'Legs in two arms at once: {int(pt)} in {n_sim(A)} neutrons, too few to split.'
    fig.text(0.52, y - 0.02, txt, ha='left', va='top', fontsize=9.5, color=P.MUTED)
    save(fig, 'thermal_funnel_F5', dict(main=segs, zoom=zoom))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--accounting', default=str(ACCOUNTING))
    ap.add_argument('--only', choices=('F1', 'F2', 'F3', 'F4', 'F5'))
    a = ap.parse_args()
    A = Accounting(Path(a.accounting))
    for name, fn in (('F1', f1), ('F2', f2), ('F3', f3), ('F4', f4), ('F5', f5)):
        if a.only in (None, name):
            fn(A)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
