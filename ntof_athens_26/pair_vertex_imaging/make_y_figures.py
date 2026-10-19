#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_y_figures.py -- the figures for the y study (`y_image.py`).

Draws from ``pairs_y.parquet``, ``tracks_y.parquet``, ``y_fits.csv``,
``y_band_scale.csv``, ``y_focus.csv`` and ``y_map3d.npz``; the fit model is
re-evaluated at the stored parameters, never refitted.  ``y_<name>.png`` +
``.pdf`` into ``figures/``, and ``y_volume.html`` / ``y_volume.div.html``.

    python -m pair_vertex_imaging.make_y_figures
    python -m pair_vertex_imaging.make_y_figures --only split,single
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402
import plotstyle as P  # noqa: E402
from pair_vertex_imaging import vertex_image as VI  # noqa: E402
from pair_vertex_imaging import y_image as Y  # noqa: E402
from pair_vertex_imaging.make_image_figures import DIV, SEQ, foot, save  # noqa: E402

FOOT = ('ntof_athens_26/pair_vertex_imaging  |  33 runs of the condor full pass, '
        'stage-3 tracks, chambers A, C, D')
ARM_COLOR = {'A': '#0072B2', 'C': '#009E73', 'D': '#CC79A7'}
SCALE_STYLE = {'raw (s = 1)': ((0, (4, 2.5)), 1.2), 'band scale': ('-', 2.0),
               'focus scale': ((0, (1, 1.5)), 1.6)}


def read(od, name):
    return pd.read_csv(od / name, keep_default_na=False, na_values=[''])


def frow(F, kind, selection, scale_set, tier=None):
    r = F[(F.kind == kind) & (F.selection == selection) & (F.scale_set == scale_set)]
    if tier is not None:
        r = r[r.tier == tier]
    return r.iloc[0] if len(r) else None


def gas_band(ax, centroid_at=None):
    """Shade the gas's y extent, placed with its volume centroid at ``centroid_at``
    (nominal position if None)."""
    y0, _ = VI.capsule_profile()
    _, _, cen = Y.capsule_y_profile()
    off = 0.0 if centroid_at is None else centroid_at - cen
    ax.axvspan(y0[0] + off, y0[-1] + off, color=P.COPPER, alpha=0.12, lw=0)


def profile_panel(ax, yd, yn, fr, hue, title):
    M = Y.YProfileModel(np.asarray(yd, float), np.asarray(yn, float))
    x = M.centres
    gas_band(ax)
    ax.errorbar(x, M.n, np.sqrt(M.n), fmt='o', ms=2.8, color=hue, lw=0.9, label='data')
    ax.plot(x, M.N * M.B, color=P.MUTED, lw=1.2, ls=(0, (4, 2.5)), label='null, same cuts')
    if fr is not None and np.isfinite(fr.get('c', np.nan)):
        S = M.N * fr.f * M.source(fr.c, fr.s)
        B = M.N * (1 - fr.f) * M.B
        ax.plot(x, S + B, color=P.INK, lw=1.6, label='fit: gas ⊗ blur + null')
        ax.fill_between(x, B, S + B, color=hue, alpha=0.18, lw=0)
        sub = (f'centroid {fr.c:+.1f} ± {fr.c_err:.1f}   σ {fr.s:.0f} mm   '
               f'term {100 * fr.f:.0f} %   χ²/ndf {fr.chi2 / max(fr.ndf, 1):.1f}')
    else:
        sub = 'the fit wants no capsule term' if fr is not None and 'f' in fr else ''
    ax.set_xlim(-Y.Y_WIN, Y.Y_WIN)
    ax.set_ylim(bottom=0)
    ax.set_title(f'{title}\n{sub}', loc='left', fontsize=9.5, color=P.INK)
    P.strip(ax)


# --------------------------------------------------------------------------- #
def split(d, F, cap, scales):
    """Why the old vertex y had three peaks: mean of legs vs each leg."""
    raw = scales['raw (s = 1)']
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.8))
    for i, (pair, arm) in enumerate((('A-D', 'A'), ('C-D', 'C'))):
        for j, (which, ylegs, lab, hue, sel) in enumerate((
                ('mean', (), 'mean of both legs — the old vertex y', P.INK,
                 f'{pair[0]}–D, mean of both legs (as before)'),
                (1, (1,), f'{arm} leg alone (y-clean)', ARM_COLOR[arm], f'{pair[0]}–D, {arm} leg'),
                (2, (2,), 'D leg alone (y-clean)', ARM_COLOR['D'], f'{pair[0]}–D, D leg alone'))):
            g = Y.pair_select(d, pair, ylegs, 'x', cap)
            y = Y.pair_y(g, which, raw)
            nul = g.null.to_numpy()
            profile_panel(axes[i, j], y[~nul], y[nul],
                          frow(F, 'pair', sel, 'raw (s = 1)'), hue,
                          f'{pair} · {lab} · {int((~nul).sum()):,} pairs')
            axes[i, j].set_xlabel('y at the transverse crossing  [mm]')
        axes[i, 0].set_ylabel('pairs per 10 mm')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Where the three peaks came from: averaging a good y with a junk one',
                 x=0.01, ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    foot(fig, f'Pairs whose crossing is within {Y.WIN_X:.0f} mm of the capsule in x, both '
         f'legs slope-measured and not noisy in x, within {Y.CUT:.0f} mm of the axis; '
         'y angle scale as reconstructed.  Left: the mean of the two legs’ y, as '
         'plotted before.  Middle: the A or C leg’s own y.  Right: the D leg’s own y. '
         'Copper: the gas’s nominal extent along the beam.   ' + FOOT)
    save(fig, 'y_split')


def bands(tr):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), sharex=True, sharey=True)
    for j, arm in enumerate(Y.ARMS):
        g = tr[tr.arm == arm]
        for i, (lab, m) in enumerate((('x-clean, pointing < 30 mm', np.ones(len(g), bool)),
                                      ('+ y slope measured, not a noisy y column', Y.ok_y(g.code)))):
            h = g[m]
            q = (h.ty * h.dw).to_numpy(float) / Y.L_MM
            ax = axes[i, j]
            H, xe, ye = np.histogram2d(h.py, -q, bins=(200, 160),
                                       range=((-200, 200), (-0.8, 0.8)))
            ax.imshow(H.T + 0.5, origin='lower', aspect='auto', extent=(xe[0], xe[-1], ye[0], ye[-1]),
                      cmap=SEQ, norm=LogNorm(vmin=0.5, vmax=max(H.max(), 2)), interpolation='nearest')
            ax.set_title(f'{arm} · {lab}\n{len(h):,} tracks', loc='left', fontsize=10.5, color=P.INK)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 1:
                ax.set_xlabel('impact y on the strip plane  [mm]')
            if j == 0:
                ax.set_ylabel('− tan y · dw / L   (y slope toward the axis)')
    fig.suptitle('The pointing band in y, before and after the y-plane cuts', x=0.01,
                 ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    foot(fig, 'A track from a source at y_s lies on a line of slope s/L through (y_s, 0), '
         'where s is the y angle scale; the capsule is 80 mm long, so the band is a '
         'smear of such lines.  Horizontal line at 0: y slopes the timing did not '
         'measure; vertical stripes: noisy y columns.   ' + FOOT)
    save(fig, 'y_bands')


def gas_robust_sigma() -> float:
    """1.4826 x MAD of the gas's own y distribution (density ~ pi R^2)."""
    u, a, _ = Y.capsule_y_profile()
    cdf = np.cumsum(a) / a.sum()
    med = np.interp(0.5, cdf, u)
    dev = np.abs(u - med)
    o = np.argsort(dev)
    mad = dev[o][np.searchsorted(np.cumsum(a[o]) / a.sum(), 0.5)]
    return float(1.4826 * mad)


def single(tr, F, BS, FS, scales):
    tyn = Y.shuffled_ty(tr)
    gas_rsig = gas_robust_sigma()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.0))
    for j, arm in enumerate(Y.ARMS):
        m = (tr.arm == arm).to_numpy() & Y.ok_y(tr.code)
        py, ty, dw = (tr.py.to_numpy(float)[m], tr.ty.to_numpy(float)[m], tr.dw.to_numpy(float)[m])
        s = scales['raw (s = 1)'].get(arm, 1.0)
        profile_panel(axes[0, j], py + ty / s * dw, py + tyn[m] / s * dw,
                      frow(F, 'single track', f'chamber {arm}', 'raw (s = 1)', 'y-clean'),
                      ARM_COLOR[arm],
                      f'chamber {arm} · single tracks · y scale as reconstructed')
        axes[0, j].set_xlabel('single-track y at the beam axis  [mm]')
        ax = axes[1, j]
        f = FS[FS.arm == arm]
        ax.plot(f.scale, f.rsig_data, color=ARM_COLOR[arm], lw=2, label='data')
        ax.plot(f.scale, f.rsig_null, color=P.MUTED, lw=1.2, ls=(0, (4, 2.5)), label='null')
        bs = BS[(BS.arm == arm) & (BS.tier == 'y-clean')]
        if len(bs):
            ax.axvline(float(bs.band_scale.iloc[0]), color=P.INK, lw=1, ls=':',
                       label=f'band scale {float(bs.band_scale.iloc[0]):.2f}')
        fmin = f.loc[f.rsig_data.idxmin()]
        ax.axvline(fmin.scale, color=ARM_COLOR[arm], lw=1, ls='--',
                   label=f'focus minimum {fmin.scale:.2f}')
        ax.axhline(gas_rsig, color=P.COPPER, lw=1)
        ax.text(f.scale.min(), gas_rsig + 2, ' the gas alone', color=P.COPPER, fontsize=8.5)
        ax.set_xlabel('y angle scale s applied (tan y ÷ s)')
        ax.set_ylabel('robust σ of y at the axis  [mm]')
        ax.set_title(f'chamber {arm}: y focus scan', loc='left', fontsize=11, color=P.INK)
        ax.legend(fontsize=8)
        P.strip(ax)
    axes[0, 0].set_ylabel('tracks per 10 mm')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Single tracks already image the capsule along the beam', x=0.01,
                 ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, 'Top: y of each track where it passes the beam axis, x and y planes cleaned, '
         'x pointing < 30 mm, y angle scale as reconstructed; null: tan y shuffled among tracks '
         'of the same chamber, run and cuts.  Bottom: robust width against the y scale; the '
         'copper line is the robust σ of the gas’s own y distribution.  The band scale sits '
         'low (background flattens the band) and the focus minimum high (a larger scale also '
         'shrinks the tan y noise), so neither is applied.   ' + FOOT)
    save(fig, 'y_single')


def pairs(d, F, cap, scales):
    band = scales['raw (s = 1)']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    for ax, (label, pair, which, ylegs, win, hue) in zip(axes, (
            ('A–D, A leg', 'A-D', 1, (1,), 'x', ARM_COLOR['A']),
            ('C–D, C leg', 'C-D', 1, (1,), 'x', ARM_COLOR['C']),
            ('D–D, mean of both D legs', 'D-D', 'mean', (1, 2), 'z', ARM_COLOR['D']))):
        g = Y.pair_select(d, pair, ylegs, win, cap)
        y = Y.pair_y(g, which, band)
        nul = g.null.to_numpy()
        profile_panel(ax, y[~nul], y[nul], frow(F, 'pair', label, 'raw (s = 1)'), hue,
                      f'{label} · {int((~nul).sum()):,} pairs')
        ax.set_xlabel('pair vertex y  [mm]')
    axes[0].set_ylabel('pairs per 10 mm')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('The pair vertex y, taken from the legs that measure it', x=0.01, ha='left',
                 fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    foot(fig, f'A–D and C–D: y of the A or C leg at the crossing, crossing within {Y.WIN_X:.0f} mm '
         f'of the capsule in x.  D–D: mean of the two D legs, crossing within {Y.WIN_X:.0f} mm in '
         'z.  Both legs x-clean, the y legs y-clean, y angle scale as reconstructed.   '
         + FOOT)
    save(fig, 'y_pairs')


def maps(od, cap):
    from scipy.ndimage import gaussian_filter
    from matplotlib.patches import Circle
    Z3 = np.load(od / 'y_map3d.npz')
    ex, ey = Z3['ex'], Z3['ey']
    names = [k.split('|')[0] for k in Z3.files if k.endswith('|excess')]
    yv, rv = VI.capsule_profile()
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 4.3 * len(names)),
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    for i, name in enumerate(names):
        E = Z3[f'{name}|excess']
        for j, (H, ea, eb, la, lb) in enumerate((
                (E.sum(1), ex, ex, 'x  [mm]', 'z  [mm]'),
                (E.sum(2), ex, ey, 'x  [mm]', 'y, beam  [mm]'),
                (E.sum(0).T, ex, ey, 'z  [mm]', 'y, beam  [mm]'))):
            ax = axes[i, j]
            Hs = gaussian_filter(H, 1.0)
            v = max(np.nanpercentile(np.abs(Hs), 99.5), 1e-12)
            ax.imshow(Hs.T, origin='lower', extent=(ea[0], ea[-1], eb[0], eb[-1]),
                      aspect='equal' if j == 0 else 'auto', cmap=DIV,
                      norm=TwoSlopeNorm(0, -v, v), interpolation='nearest')
            if j == 0:
                ax.add_patch(Circle(cap, 10, fill=False, color=P.COPPER, lw=1.5))
            else:
                c0 = cap[0] if j == 1 else cap[1]
                ax.plot(np.r_[c0 + rv, (c0 - rv)[::-1], c0 + rv[0]],
                        np.r_[yv, yv[::-1], yv[0]], color=P.COPPER, lw=1.2)
            ax.set_xlabel(la, fontsize=9.5)
            ax.set_ylabel(lb, fontsize=9.5)
            ax.set_title(name if j == 0 else '', loc='left', fontsize=11, color=P.INK,
                         fontweight='bold' if name == 'balanced' else 'normal')
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle('The pair vertex in 3D, with y from the legs that measure it', x=0.01,
                 ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    foot(fig, 'Data minus each class’s shuffled null, normalised in the transverse corners; '
         '"balanced" gives the x slab and the z slab equal weight.  Copper: the capsule at the '
         'single-track transverse position and its NOMINAL y — the y axis is not shifted to '
         'where the data put it.   ' + FOOT)
    save(fig, 'y_maps')


def volume(od, cap):
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter
    Z3 = np.load(od / 'y_map3d.npz')
    ex, ey = Z3['ex'], Z3['ey']
    xc, yc = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])
    X, Yg, Zg = np.meshgrid(xc, yc, xc, indexing='ij')
    names = ['balanced'] + [k.split('|')[0] for k in Z3.files
                            if k.endswith('|excess') and not k.startswith('balanced')]
    traces = []
    for name in names:
        E = gaussian_filter(Z3[f'{name}|excess'], 1.0)
        E = np.clip(E, 0, None) / max(E.max(), 1e-12)
        traces.append(go.Volume(
            x=X.ravel().astype(np.float32), y=Zg.ravel().astype(np.float32),
            z=Yg.ravel().astype(np.float32), value=E.ravel().astype(np.float32),
            isomin=0.25, isomax=1.0, opacity=0.14, surface_count=14,
            colorscale=[[0, '#e8d6ea'], [0.35, '#b77cbd'], [0.7, '#7d3a86'], [1, '#2a0f30']],
            colorbar=dict(title='relative<br>excess', len=0.55, thickness=12),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=name, visible=(name == 'balanced'), hoverinfo='skip'))
    yv, rv = VI.capsule_profile()
    ph = np.linspace(0, 2 * np.pi, 36)
    Rm, Pm = np.meshgrid(rv, ph, indexing='ij')
    traces.append(go.Surface(x=cap[0] + Rm * np.cos(Pm), y=cap[1] + Rm * np.sin(Pm),
                             z=np.repeat(yv[:, None], len(ph), axis=1), showscale=False,
                             opacity=0.5, colorscale=[[0, '#d18a44'], [1, '#d18a44']],
                             name='He-3 capsule (nominal y, single-track x/z)',
                             showlegend=True, hoverinfo='name'))
    traces.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=[ey[0], ey[-1]], mode='lines',
                               line=dict(color='#6a7583', width=4, dash='dash'),
                               name='nominal beam axis', hoverinfo='name'))
    fig = go.Figure(traces)
    nv = len(names)
    buttons = [dict(label=nm, method='update',
                    args=[{'visible': [i == k for i in range(nv)] + [True, True]}])
               for k, nm in enumerate(names)]
    fig.update_layout(
        updatemenus=[dict(type='buttons', direction='right', x=0.0, y=1.06, xanchor='left',
                          buttons=buttons, showactive=True)],
        scene=dict(xaxis_title='x [mm]', yaxis_title='z [mm]', zaxis_title='y, beam [mm]',
                   aspectmode='manual', aspectratio=dict(x=1, y=1, z=2.6),
                   camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))),
        margin=dict(l=0, r=0, t=46, b=0), height=760, legend=dict(x=0.01, y=0.02),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(family='IBM Plex Sans, Helvetica, Arial, sans-serif'))
    fig.write_html(HERE / 'figures' / 'y_volume.html', include_plotlyjs='cdn', full_html=True)
    div = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='vertexy3d')
    (HERE / 'figures' / 'y_volume.div.html').write_text(div, encoding='utf-8')
    print(f'  -> y_volume.html  ({len(div) / 1e6:.1f} MB)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    want = set(x for x in a.only.split(',') if x)
    wants = lambda k: not want or k in want  # noqa: E731
    od = paths.out('pair_vertex')
    cap = VI.capsule_centre()
    P.use()
    F = read(od, 'y_fits.csv')
    BS = read(od, 'y_band_scale.csv')
    FS = read(od, 'y_focus.csv')
    scales = json.loads((od / 'y_derive.meta.json').read_text())['scales']
    if wants('bands') or wants('single'):
        tr = pd.read_parquet(od / 'tracks_y.parquet')
        if wants('bands'):
            bands(tr)
        if wants('single'):
            single(tr, F, BS, FS, scales)
        del tr
    if wants('split') or wants('pairs'):
        d = Y.pair_load(od)
        if wants('split'):
            split(d, F, cap, scales)
        if wants('pairs'):
            pairs(d, F, cap, scales)
        del d
    if wants('maps'):
        maps(od, cap)
    if wants('volume'):
        volume(od, cap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
