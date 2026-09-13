#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vertex3d.py -- one 3D density of every clean pair, all topologies together.

A pair of straight lines always yields a 3D point, but no single pair measures
all three coordinates.  Which direction a pair localises is set by its geometry
(`z_image`):

  A-D, C-D    x from the A/C leg; z from the D leg, which carries little
  A-A, C-C    both lines run along z: x yes, z no
  D-D         both lines run along x: z yes, x no
  A-C         z, weakly
  every pair  y poorly -- weak y views, and the capsule is 80 mm long

So each class on its own images a SLAB -- a band in x, or a band in z -- and
none images a point.  Summed, the slabs cross.  If they cross at the capsule,
the sum is a transverse image built the way a tomograph builds one, by
back-projecting many one-dimensional measurements; if the classes disagree, the
crossing shows it.  That is the question here, and it is answered from the
data rather than assumed.

THE VERTEX of a pair is its transverse crossing ``(vx_xz, vz_xz)`` with
``vy_xz``, the mean of the two legs' y there -- the estimator of the companion
notes, which avoids the 3D closest approach's y drag.

EACH CLASS'S BACKGROUND is its own no-source null (directions shuffled within
chamber x run x flag code, same cuts), normalised to the data in the CORNERS of
the transverse window -- where both |x - cx| and |z - cz| exceed 35 mm, so no
slab of any class reaches it.  Sideband normalisation, not a fitted fraction:
the fitted fractions of `z_image` exist only per coordinate.

THE NOISE FLOOR comes from running the identical construction with shuffle 1
as the "data" and shuffle 2 as its null: a map with no source in it, whose
largest excess is what noise alone makes.

    python -m pair_vertex_imaging.vertex3d            # measure + figures + 3D
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402
from pair_vertex_imaging import vertex_image as VI  # noqa: E402
from pair_vertex_imaging import z_image as Z  # noqa: E402

OUT = HERE / 'figures'
CLASSES = {
    'A–D, C–D': ('A-D', 'C-D'),
    'A–A, C–C': ('A-A', 'C-C'),
    'A–C': ('A-C',),
    'D–D': ('D-D',),
}
CLASS_COLOR = {'A–D, C–D': '#0072B2', 'A–A, C–C': '#009E73', 'A–C': '#E69F00',
               'D–D': '#CC79A7', 'all': '#1b2430', 'balanced': '#7d3a86'}
#: The two classes that each localise one transverse coordinate: x (A-D, C-D)
#: and z (D-D).  A-A/C-C and A-C carry no excess above the no-source map.
BALANCE = ('A–D, C–D', 'D–D')
TIER = 'clean'
CUT = 60.0
EX = np.arange(-64.0, 64.0 + 4.0, 4.0)      # x and z
EY = np.arange(-180.0, 180.0 + 10.0, 10.0)   # y, the beam
SIDEBAND_MM = 35.0
BAND_MM = 12.0          # half-width of the slices the profiles are taken in
SMOOTH_BINS = 1.0


def ctr(e):
    return 0.5 * (e[:-1] + e[1:])


def hist3(h: pd.DataFrame) -> np.ndarray:
    return np.histogramdd(h[['vx_xz', 'vy_xz', 'vz_xz']].to_numpy(float),
                          bins=(EX, EY, EX))[0]


def sideband_mask(cap) -> np.ndarray:
    xc = ctr(EX)
    return ((np.abs(xc - cap[0]) > SIDEBAND_MM)[:, None]
            & (np.abs(xc - cap[1]) > SIDEBAND_MM)[None, :])


def class_maps(d: pd.DataFrame, cap, tier=TIER, cut=CUT):
    """Per class: data, sideband-scaled null, and the shuffle-vs-shuffle noise map."""
    side = sideband_mask(cap)
    maps, rows = {}, []
    for name, pairs in CLASSES.items():
        g = d[d.pair.isin(pairs)]
        ok = ((g.worst_axis < cut).to_numpy()
              & Z.tier_ok(g.code1.to_numpy(), tier) & Z.tier_ok(g.code2.to_numpy(), tier))
        g = g[ok]
        Hd = hist3(g[g.variant == 0])
        H1 = hist3(g[g.variant == 1])
        H2 = hist3(g[g.variant == 2])
        Hn = H1 + H2
        k = Hd.sum(1)[side].sum() / max(Hn.sum(1)[side].sum(), 1)
        k12 = H1.sum(1)[side].sum() / max(H2.sum(1)[side].sum(), 1)
        maps[name] = dict(data=Hd, bkg=Hn * k, noise=H1 - H2 * k12,
                          noise_var=H1 + H2 * k12 ** 2, var=Hd + Hn * k ** 2)
        rows.append(dict(cls=name, n_pairs=int((g.variant == 0).sum()),
                         n_in_window=float(Hd.sum()), k_null=k,
                         bkg_fraction=float((Hn * k).sum() / max(Hd.sum(), 1)),
                         excess=float(Hd.sum() - (Hn * k).sum())))
    tot = {key: sum(m[key] for m in maps.values()) for key in
           ('data', 'bkg', 'noise', 'noise_var', 'var')}
    maps['all'] = tot
    rows.append(dict(cls='all', n_pairs=sum(r['n_pairs'] for r in rows),
                     n_in_window=float(tot['data'].sum()), k_null=np.nan,
                     bkg_fraction=float(tot['bkg'].sum() / max(tot['data'].sum(), 1)),
                     excess=float(tot['data'].sum() - tot['bkg'].sum())))
    # BALANCED: each class in BALANCE scaled to unit excess, so the x slab and
    # the z slab vote equally.  The raw sum is ~84 % A-D/C-D and shows only
    # their slab.  Variances scale with the weight squared.
    w = {c: 1.0 / max(float((maps[c]['data'] - maps[c]['bkg']).sum()), 1.0)
         for c in BALANCE}
    bal = {key: sum(maps[c][key] * (w[c] ** 2 if key in ('var', 'noise_var') else w[c])
                    for c in BALANCE) for key in tot}
    maps['balanced'] = bal
    rows.append(dict(cls='balanced',
                     n_pairs=sum(r['n_pairs'] for r in rows if r['cls'] in BALANCE),
                     n_in_window=np.nan, k_null=np.nan, bkg_fraction=np.nan,
                     excess=float((bal['data'] - bal['bkg']).sum())))
    return maps, pd.DataFrame(rows)


def _fwhm(x, y):
    """Full width at half maximum of a 1D profile around its maximum [mm]."""
    y = np.asarray(y, float)
    if not np.isfinite(y).any() or np.nanmax(y) <= 0:
        return np.nan, np.nan
    i = int(np.nanargmax(y))
    half = 0.5 * y[i]
    lo = i
    while lo > 0 and y[lo] > half:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and y[hi] > half:
        hi += 1
    if y[lo] > half or y[hi] > half:
        return x[i], np.nan      # never falls to half inside the window
    xl = np.interp(half, [y[lo], y[lo + 1]], [x[lo], x[lo + 1]])
    xh = np.interp(half, [y[hi], y[hi - 1]], [x[hi], x[hi - 1]])
    # peak position refined by a parabola through the top three bins
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        xp = x[i] + (0.5 * (a - c) / den * (x[1] - x[0]) if den else 0.0)
    else:
        xp = x[i]
    return xp, xh - xl


def measure(maps: dict, cap) -> pd.DataFrame:
    """Where each map's transverse excess peaks, how wide, how significant."""
    from scipy.ndimage import gaussian_filter
    xc, yc = ctr(EX), ctr(EY)
    rows = []
    for name, m in maps.items():
        for kind, E, V in (('data', m['data'] - m['bkg'], m['var']),
                           ('noise', m['noise'], m['noise_var'])):
            T = gaussian_filter(E.sum(1), SMOOTH_BINS)          # (x, z)
            TV = V.sum(1)
            ix, iz = np.unravel_index(np.argmax(T), T.shape)
            # significance of the raw excess in a 3x3-bin (12 mm) box at the peak
            sl = (slice(max(ix - 1, 0), ix + 2), slice(max(iz - 1, 0), iz + 2))
            box = E.sum(1)[sl].sum()
            # no floor of 1 on the variance: the balanced map is scaled to unit
            # excess, so its variances are ~1e-4 and a count-scale floor would
            # crush its significance to nothing
            bv = TV[sl].sum()
            box_sig = box / np.sqrt(bv) if bv > 0 else np.nan
            # profiles through the CAPSULE, not through the peak: the question is
            # whether the image is at the capsule
            bx = np.abs(xc - cap[1]) < BAND_MM
            bz = np.abs(xc - cap[0]) < BAND_MM
            px = T[:, bx].sum(1)
            pz = T[bz, :].sum(0)
            xpk, xw = _fwhm(xc, px)
            zpk, zw = _fwhm(xc, pz)
            # y profile: every voxel within 16 mm of the capsule transversely.
            # E is (x, y, z); reorder to (x, z, y) so the transverse mask selects
            # whole y columns
            near = ((np.abs(xc - cap[0]) < 16)[:, None] & (np.abs(xc - cap[1]) < 16)[None, :])
            py = gaussian_filter(E.transpose(0, 2, 1)[near].sum(0), SMOOTH_BINS)
            ypk, yw = _fwhm(yc, py)
            rows.append(dict(
                cls=name, map=kind,
                peak_x=xc[ix], peak_z=xc[iz],
                peak_dist_from_capsule=float(np.hypot(xc[ix] - cap[0], xc[iz] - cap[1])),
                box_excess=float(box), box_sigma=float(box_sig),
                xprof_peak=xpk, xprof_fwhm=xw, zprof_peak=zpk, zprof_fwhm=zw,
                yprof_peak=ypk, yprof_fwhm=yw))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def figures(maps, S, M, cap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle
    from scipy.ndimage import gaussian_filter
    import plotstyle as P
    from pair_vertex_imaging.make_image_figures import DIV, foot, save
    P.use()
    xc, yc = ctr(EX), ctr(EY)
    FOOT = ('ntof_athens_26/pair_vertex_imaging  |  33 runs of the condor full '
            'pass, chambers A, C, D; both legs slope-measured and not in a noisy '
            f'column, within {CUT:.0f} mm of the axis')
    yv, rv = VI.capsule_profile()

    names = list(CLASSES) + ['all', 'balanced']
    fig, axes = plt.subplots(len(names), 3, figsize=(14.5, 4.0 * len(names)),
                             gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    for i, name in enumerate(names):
        E = maps[name]['data'] - maps[name]['bkg']
        views = ((E.sum(1), EX, EX, 'x  [mm]', 'z  [mm]', 'transverse'),
                 (E.sum(2), EX, EY, 'x  [mm]', 'y, beam  [mm]', 'x–y'),
                 (E.sum(0).T, EX, EY, 'z  [mm]', 'y, beam  [mm]', 'z–y'))
        srow = S[S.cls == name].iloc[0]
        for j, (H, ea, eb, la, lb, t) in enumerate(views):
            ax = axes[i, j]
            Hs = gaussian_filter(H, SMOOTH_BINS)
            v = max(np.nanpercentile(np.abs(Hs), 99.5), 1e-9)
            ax.imshow(Hs.T, origin='lower', extent=(ea[0], ea[-1], eb[0], eb[-1]),
                      aspect='equal' if j == 0 else 'auto', cmap=DIV,
                      norm=TwoSlopeNorm(0, -v, v), interpolation='nearest')
            if j == 0:
                ax.add_patch(Circle(cap, 10, fill=False, color=P.COPPER, lw=1.5))
                ax.plot(0, 0, 'o', mfc='none', color=P.MUTED, ms=5)
            else:
                c0 = cap[0] if j == 1 else cap[1]
                ax.plot(np.r_[c0 + rv, (c0 - rv)[::-1], c0 + rv[0]],
                        np.r_[yv, yv[::-1], yv[0]], color=P.COPPER, lw=1.2)
            ax.set_xlabel(la, fontsize=9.5)
            ax.set_ylabel(lb, fontsize=9.5)
            if j:
                head = f'{name}  ·  {t}'
            elif name == 'balanced':
                head = f'{name}  ·  {t}\nA–D, C–D and D–D at equal weight'
            else:
                head = (f'{name}  ·  {t}\n{srow.n_pairs:,} pairs, excess '
                        f'{srow.excess:,.0f} ({100 * (1 - srow.bkg_fraction):.0f} %)')
            ax.set_title(head, loc='left', fontsize=10.5, color=P.INK,
                         fontweight='bold' if name == 'all' else 'normal')
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle('Every clean pair, data − null, one topology at a time and summed',
                 x=0.01, ha='left', fontsize=15, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    foot(fig, 'Vertex: transverse crossing, y = mean of the legs’ y there. Each '
         'class’s shuffled null is normalised to its data where |x − cx| and '
         '|z − cz| both exceed 35 mm and subtracted.  Purple: excess; blue: '
         'deficit; smoothed by one 4 mm bin.  Copper: the capsule at the '
         'single-track position.   ' + FOOT)
    save(fig, 'v3d_projections')

    # profiles through the capsule, stacked by class, against the noise map
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    bx = np.abs(xc - cap[1]) < BAND_MM
    bz = np.abs(xc - cap[0]) < BAND_MM
    near = ((np.abs(xc - cap[0]) < 16)[:, None] & (np.abs(xc - cap[1]) < 16)[None, :])
    for ax, (lab, grid, c0, fn) in zip(axes, (
            (f'x, in |z − cz| < {BAND_MM:.0f} mm', xc, cap[0],
             lambda E: gaussian_filter(E.sum(1), SMOOTH_BINS)[:, bx].sum(1)),
            (f'z, in |x − cx| < {BAND_MM:.0f} mm', xc, cap[1],
             lambda E: gaussian_filter(E.sum(1), SMOOTH_BINS)[bz, :].sum(0)),
            ('y, within 16 mm of the capsule transversely', yc, None,
             lambda E: gaussian_filter(E.transpose(0, 2, 1)[near].sum(0), SMOOTH_BINS)))):
        bottom = np.zeros(len(grid))
        for name in CLASSES:
            y = fn(maps[name]['data'] - maps[name]['bkg'])
            ax.fill_between(grid, bottom, bottom + y, step='mid',
                            color=CLASS_COLOR[name], alpha=0.55, lw=0, label=name)
            bottom = bottom + y
        ax.step(grid, fn(maps['all']['data'] - maps['all']['bkg']), where='mid',
                color=P.INK, lw=1.8, label='sum')
        ax.step(grid, fn(maps['all']['noise']), where='mid', color=P.MUTED, lw=1.2,
                ls=(0, (4, 2.5)), label='same construction, no source')
        ax.axhline(0, color=P.LINE, lw=0.8)
        if c0 is not None:
            ax.axvline(c0, color=P.COPPER, lw=1.2)
            ax.axvspan(c0 - 10, c0 + 10, color=P.COPPER, alpha=0.1, lw=0)
        else:
            ax.plot(np.r_[yv[0], yv, yv[-1]] * 0 + np.r_[yv[0], yv, yv[-1]],
                    np.zeros(len(yv) + 2), alpha=0)
            ax.axvspan(yv[0], yv[-1], color=P.COPPER, alpha=0.1, lw=0)
        ax.set_xlabel(f'{lab.split(",")[0]}  [mm]')
        ax.set_title(lab, loc='left', fontsize=11.5, color=P.INK)
        P.strip(ax)
    axes[0].set_ylabel('excess pairs (smoothed)')
    axes[0].legend(fontsize=8.5, loc='upper left')
    fig.suptitle('The summed image, sliced through the capsule', x=0.01, ha='left',
                 fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    foot(fig, 'Filled: each topology’s excess, stacked.  Black: their sum.  Grey '
         'dashed: the identical construction with one shuffled null as the data '
         'and the other as its null — what the map shows with no source at all.  '
         'Copper: the capsule (± 10 mm transversely; its 80 mm gas length in '
         'y).   ' + FOOT)
    save(fig, 'v3d_profiles')

    # significance map of the summed transverse image
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, (kind, E, V) in zip(axes, (
            ('all topologies, raw sum', maps['all']['data'] - maps['all']['bkg'],
             maps['all']['var']),
            ('A–D, C–D and D–D at equal weight',
             maps['balanced']['data'] - maps['balanced']['bkg'],
             maps['balanced']['var']),
            ('equal weight, no source (shuffle 1 − shuffle 2)',
             maps['balanced']['noise'], maps['balanced']['noise_var']))):
        k = np.ones((3, 3))
        from scipy.signal import convolve2d
        num = convolve2d(E.sum(1), k, mode='same')
        var = convolve2d(V.sum(1), k, mode='same')
        # no count-scale floor (see `measure`): empty boxes are left blank
        with np.errstate(divide='ignore', invalid='ignore'):
            Sg = np.where(var > 0, num / np.sqrt(var), np.nan)
        im = ax.imshow(Sg.T, origin='lower', extent=(EX[0], EX[-1], EX[0], EX[-1]),
                       cmap=DIV, norm=TwoSlopeNorm(0, -8, 8), interpolation='nearest')
        ax.add_patch(Circle(cap, 10, fill=False, color=P.COPPER, lw=1.5))
        ax.set_title(f'{kind}\nlargest {np.nanmax(Sg):.1f} σ', loc='left',
                     fontsize=11, color=P.INK)
        ax.set_xlabel('x  [mm]')
        ax.set_ylabel('z  [mm]')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('excess / σ in a 12 mm box', fontsize=9, color=P.MUTED)
        cb.outline.set_visible(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle('Summed transverse image, as a significance', x=0.01, ha='left',
                 fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    foot(fig, 'Excess over the nulls in each 12 × 12 mm box, divided by its '
         'Poisson error (data plus scaled null).  Neighbouring boxes overlap, so '
         'the map is correlated.  Right: the same with no source in it.   ' + FOOT,
         width=150)
    save(fig, 'v3d_significance')


def volume(maps, cap):
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter
    xc, yc = ctr(EX), ctr(EY)
    X, Y, Zg = np.meshgrid(xc, yc, xc, indexing='ij')
    names = ['balanced', 'all'] + list(CLASSES)
    traces = []
    for name in names:
        E = gaussian_filter(maps[name]['data'] - maps[name]['bkg'], 1.0)
        E = np.clip(E, 0, None) / max(E.max(), 1e-9)
        traces.append(go.Volume(
            x=X.ravel().astype(np.float32), y=Zg.ravel().astype(np.float32),
            z=Y.ravel().astype(np.float32), value=E.ravel().astype(np.float32),
            isomin=0.2, isomax=1.0, opacity=0.14, surface_count=14,
            colorscale=[[0, '#e8d6ea'], [0.35, '#b77cbd'], [0.7, '#7d3a86'],
                        [1, '#2a0f30']],
            colorbar=dict(title='relative<br>excess', len=0.55, thickness=12),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=name, visible=(name == 'balanced'), hoverinfo='skip'))
    yv, rv = VI.capsule_profile()
    ph = np.linspace(0, 2 * np.pi, 36)
    Rm, Pm = np.meshgrid(rv, ph, indexing='ij')
    traces.append(go.Surface(x=cap[0] + Rm * np.cos(Pm), y=cap[1] + Rm * np.sin(Pm),
                             z=np.repeat(yv[:, None], len(ph), axis=1),
                             showscale=False, opacity=0.5,
                             colorscale=[[0, '#d18a44'], [1, '#d18a44']],
                             name='He-3 capsule (single-track position)',
                             showlegend=True, hoverinfo='name'))
    traces.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=[EY[0], EY[-1]], mode='lines',
                               line=dict(color='#6a7583', width=4, dash='dash'),
                               name='nominal beam axis', hoverinfo='name'))
    fig = go.Figure(traces)
    nv = len(names)
    labels = {'all': 'all topologies, raw sum',
              'balanced': 'x slab + z slab, equal weight'}
    buttons = [dict(label=labels.get(nm, nm), method='update',
                    args=[{'visible': [i == k for i in range(nv)] + [True, True]}])
               for k, nm in enumerate(names)]
    fig.update_layout(
        updatemenus=[dict(type='buttons', direction='right', x=0.0, y=1.06,
                          xanchor='left', buttons=buttons, showactive=True)],
        scene=dict(xaxis_title='x [mm]', yaxis_title='z [mm]',
                   zaxis_title='y, beam [mm]', aspectmode='manual',
                   aspectratio=dict(x=1, y=1, z=2.2),
                   camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))),
        margin=dict(l=0, r=0, t=46, b=0), height=720,
        legend=dict(x=0.01, y=0.02), paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, Helvetica, Arial, sans-serif'))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT / 'v3d_volume.html', include_plotlyjs='cdn', full_html=True)
    div = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='vertex3dclean')
    (OUT / 'v3d_volume.div.html').write_text(div, encoding='utf-8')
    print(f'  -> v3d_volume.html  ({len(div) / 1e6:.1f} MB)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()
    od = paths.out('pair_vertex')
    cap = VI.capsule_centre()
    d = Z.load(od)
    maps, S = class_maps(d, cap)
    del d
    M = measure(maps, cap)
    S.to_csv(od / 'v3d_classes.csv', index=False)
    M.to_csv(od / 'v3d_measure.csv', index=False)
    np.savez_compressed(od / 'v3d_maps.npz', ex=EX, ey=EY,
                        **{f'{c}|{k}': v for c, m in maps.items()
                           for k, v in m.items()})
    json.dump(dict(schema='athens26/vertex3d/1', tier=TIER, cut_mm=CUT,
                   sideband_mm=SIDEBAND_MM, band_mm=BAND_MM,
                   smooth_bins=SMOOTH_BINS, capsule_xz=list(cap),
                   classes={k: list(v) for k, v in CLASSES.items()}),
              open(od / 'v3d.meta.json', 'w'), indent=1)
    pd.set_option('display.width', 220)
    print(S.to_string(index=False, float_format=lambda x: f'{x:10.3f}'))
    print()
    print(M.to_string(index=False, float_format=lambda x: f'{x:8.2f}'))
    print(f'\ncapsule (single tracks): x {cap[0]:+.2f}  z {cap[1]:+.2f}')
    if not a.no_figures:
        figures(maps, S, M, cap)
        volume(maps, cap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
