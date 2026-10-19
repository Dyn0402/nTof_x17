#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_image_figures.py -- the figures for the pair-vertex IMAGE.

Reads ``<out>/pair_vertex/pairs_image.parquet`` and the tables
`vertex_image.py` wrote; draws, and does not measure (the fit model is
re-evaluated at the stored parameters, never refitted).  Every figure is
written as ``.png`` + ``.pdf`` into ``figures/``, and the interactive 3D
density as ``image_3d.html`` (standalone) plus ``image_3d.div.html`` (a
fragment for the note, plotly loaded from its CDN).

    python -m pair_vertex_imaging.make_image_figures
    python -m pair_vertex_imaging.make_image_figures --only profiles,per_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402
import plotstyle as P  # noqa: E402
from pair_vertex_imaging import vertex_image as VI  # noqa: E402

OUT = HERE / 'figures'
SEL_TOPO, SEL_CUT = 'perpendicular', 60.0
CLASS_COLOR = {'perpendicular': '#0072B2', 'intra': '#009E73',
               'opposing': '#CC79A7'}
SEQ = LinearSegmentedColormap.from_list(
    'seq', ['#fbfcfe', '#d9c6dc', '#a86aae', '#6d2c75', '#2a0f30'])
DIV = LinearSegmentedColormap.from_list(
    'div', ['#1f5f8b', '#9cc3dc', '#eceef1', '#d7a9da', '#6d2c75'])
FOOT = ('ntof_athens_26/pair_vertex_imaging  |  33 runs of the condor full pass, '
        'stage-3 tracks, chambers A, C, D')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {name}.png')


def foot(fig, text, width=190):
    import textwrap
    fig.text(0.005, 0.005, textwrap.fill(text, width), ha='left', va='bottom',
             fontsize=8.5, color=P.MUTED, linespacing=1.4)


def capsule_side(ax, centre, color=P.COPPER, **kw):
    y, r = VI.capsule_profile()
    ax.plot(np.r_[centre + r, (centre - r)[::-1], centre + r[0]],
            np.r_[y, y[::-1], y[0]], color=color, lw=1.3, **kw)


def fit_row(F, sel, cut, coord):
    return F[(F.selection == sel) & (F.cut_mm == cut) & (F.coord == coord)].iloc[0]


# --------------------------------------------------------------------------- #
def projections(d, cap, F):
    fx = fit_row(F, SEL_TOPO, SEL_CUT, 'x')
    g = VI.select(d, SEL_TOPO, SEL_CUT)
    D, N = g[~g.null], g[g.null]
    # the background the x fit claims, in data-pair units
    scale = (1 - float(fx.f)) * len(D) / max(len(N), 1)
    ex = np.arange(-64, 66, 2.0)
    ey = np.arange(-200, 205, 8.0)
    views = [('vx_xz', 'vz_xz', ex, ex, 'x  [mm]   (set by chamber A or C)',
              'z  [mm]   (set by chamber D)', 'transverse, looking along the beam'),
             ('vx_xz', 'vy_xz', ex, ey, 'x  [mm]', 'y, beam  [mm]', 'side view, x–y'),
             ('vz_xz', 'vy_xz', ex, ey, 'z  [mm]', 'y, beam  [mm]', 'side view, z–y')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.2),
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    rows = []
    for c, (a, b, ea, eb, la, lb, title) in enumerate(views):
        Hd = np.histogram2d(D[a], D[b], bins=(ea, eb))[0]
        Hn = np.histogram2d(N[a], N[b], bins=(ea, eb))[0] * scale
        for r, (H, lab) in enumerate(((Hd, 'data'),
                                      (Hd - Hn, 'data − null × (1 − f)'))):
            ax = axes[r, c]
            ext = (ea[0], ea[-1], eb[0], eb[-1])
            if r == 0:
                im = ax.imshow(H.T, origin='lower', extent=ext, aspect='auto',
                               cmap=SEQ, interpolation='nearest')
            else:
                v = np.nanpercentile(np.abs(H), 99.5)
                im = ax.imshow(H.T, origin='lower', extent=ext, aspect='auto',
                               cmap=DIV, norm=TwoSlopeNorm(0, -v, v),
                               interpolation='nearest')
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=8, colors=P.MUTED)
            cb.outline.set_visible(False)
            cb.set_label('pairs per bin', fontsize=8.5, color=P.MUTED)
            if c == 0:
                ax.add_patch(Circle(cap, 10, fill=False, color=P.COPPER, lw=1.6))
                ax.plot(*cap, marker='+', color=P.COPPER, ms=11, mew=1.6)
                ax.axvline(fx.c, color=P.INK, lw=1.1, ls=(0, (5, 3)))
                ax.plot(0, 0, marker='o', mfc='none', color=P.MUTED, ms=6)
                ax.set_aspect('equal')
            else:
                capsule_side(ax, cap[0] if a == 'vx_xz' else cap[1])
                if a == 'vx_xz':
                    ax.axvline(fx.c, color=P.INK, lw=1.1, ls=(0, (5, 3)))
            ax.set_xlabel(la, fontsize=10)
            ax.set_ylabel(lb, fontsize=10)
            ax.set_title(f'{title}  ·  {lab}', loc='left', fontsize=11,
                         color=P.INK)
            for s in ax.spines.values():
                s.set_visible(False)
            xc = 0.5 * (ea[:-1] + ea[1:])
            yc = 0.5 * (eb[:-1] + eb[1:])
            ii, jj = np.nonzero(H)
            rows += [dict(view=title, panel=lab, a=xc[i], b=yc[j], value=H[i, j])
                     for i, j in zip(ii, jj)]
    axes[0, 0].plot([], [], color=P.COPPER, lw=1.6,
                    label='capsule at the single-track position')
    axes[0, 0].plot([], [], color=P.INK, lw=1.1, ls=(0, (5, 3)),
                    label='fitted x centre of the pair image')
    axes[0, 0].plot([], [], 'o', mfc='none', color=P.MUTED,
                    label='nominal beam axis')
    axes[0, 0].legend(loc='upper left', fontsize=8.5, frameon=True,
                      framealpha=0.92)
    fig.suptitle(f'Where perpendicular pairs meet  —  {len(D):,} pairs, '
                 f'each leg within {SEL_CUT:.0f} mm of the beam axis',
                 x=0.01, ha='left', fontsize=15, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    foot(fig, 'Vertex = the two tracks’ crossing in the transverse plane; y = '
         'mean of the two legs’ y at that crossing.  Bottom row: the no-source '
         'null (directions shuffled within each chamber, same cut) scaled by the '
         'x fit’s background fraction and subtracted.  The excess is a vertical '
         'band at the capsule’s x: x is imaged, z (chamber D) is not.   ' + FOOT)
    save(fig, 'image_projections')
    pd.DataFrame(rows).to_csv(OUT / 'image_projections.csv', index=False)


# --------------------------------------------------------------------------- #
def profiles(d, cap, F):
    panels = (('A-D', 'x', 'chamber A sets x'), ('C-D', 'x', 'chamber C sets x'),
              ('perpendicular', 'z', 'chamber D sets z'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    rows = []
    for ax, (sel, coord, sub) in zip(axes, panels):
        fr = fit_row(F, sel, SEL_CUT, coord)
        g = VI.select(d, sel, SEL_CUT)
        col = 'vx_xz' if coord == 'x' else 'vz_xz'
        M = VI.ProfileModel(g.loc[~g.null, col].to_numpy(float),
                            g.loc[g.null, col].to_numpy(float))
        S = M.N * fr.f * M.source(fr.c, fr.s)
        B = M.N * (1 - fr.f) * M.B
        B0 = M.N * M.B
        x = M.centres
        hue = '#0072B2' if coord == 'x' else '#CC79A7'
        ax.axvspan(fr.capsule - 10, fr.capsule + 10, color=P.COPPER, alpha=0.12,
                   lw=0, zorder=0)
        ax.axvline(fr.capsule, color=P.COPPER, lw=1.0, zorder=1)
        ax.errorbar(x, M.n, np.sqrt(M.n), fmt='o', ms=3.2, color=hue, lw=1,
                    zorder=5, label='data')
        ax.plot(x, B0, color=P.MUTED, lw=1.2, ls=(0, (4, 2.5)), zorder=3,
                label='no-source null, same pairs')
        if fr.f > 0.002:
            ax.plot(x, S + B, color=P.INK, lw=1.8, zorder=4,
                    label='fit: capsule ⊗ blur + null')
            ax.fill_between(x, B, S + B, color=hue, alpha=0.18, lw=0, zorder=2,
                            label='the capsule term')
        ax.set_xlim(-64, 64)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f'{coord} of the transverse crossing  [mm]')
        if coord == 'x':
            head = (f'{sel}: {sub}\ncentre {fr.c:+.1f} ± {fr.c_err:.1f}   '
                    f'σ {fr.s:.1f} mm   capsule term {100 * fr.f:.0f} %')
        else:
            # at the f = 0 boundary the curvature error is meaningless, so it
            # is not printed rather than printed as a false +-0.0
            err = (f' ± {100 * fr.f_err:.1f}'
                   if np.isfinite(fr.f_err) and fr.f_err > 1e-4 else '')
            head = (f'{sel}: {sub}\ncentre fixed at the capsule   '
                    f'capsule term {100 * fr.f:.1f}{err} %')
        ax.set_title(head, loc='left', fontsize=11.5, color=P.INK)
        P.strip(ax)
        for xi, ni, si, bi in zip(x, M.n, S, B):
            rows.append(dict(selection=sel, coord=coord, x=xi, data=ni,
                             source=si, background=bi))
    axes[0].set_ylabel('pairs per 2 mm')
    axes[0].legend(loc='upper left', fontsize=8.5)
    fig.suptitle(f'One coordinate at a time  (legs within {SEL_CUT:.0f} mm of '
                 'the axis)', x=0.01, ha='left', fontsize=14, fontweight='bold',
                 color=P.INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    foot(fig, 'Points: data.  Dashed: the shuffled null through the same cut, '
         'normalised to the data.  Solid: the fit — the He-3 gas projected onto '
         'this coordinate, blurred by a Gaussian, plus the null shape.  Copper: '
         'the single-track capsule position ± 10 mm.  In z the centre is not '
         'free; the fit only asks whether a capsule term is wanted there.   '
         + FOOT)
    save(fig, 'image_profiles')
    pd.DataFrame(rows).to_csv(OUT / 'image_profiles.csv', index=False)


# --------------------------------------------------------------------------- #
def agreement(d):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    rows = []
    es = np.geomspace(0.5, 600, 46)
    ed = np.linspace(0, 400, 41)
    for c, topo in enumerate(VI.CLASSES):
        col = CLASS_COLOR[topo]
        for r, (var, edges, lab, scale) in enumerate((
                ('sep_mm', es, 'sep: distance between the two lines at closest approach  [mm]', 'log'),
                ('dy_abs', ed, '|y₁ − y₂| where they cross in the transverse plane  [mm]', 'linear'))):
            ax = axes[r, c]
            for cut, lw, alpha in ((SEL_CUT, 2.1, 1.0), (30.0, 1.3, 0.5)):
                h = VI.select(d, topo, cut)
                for null, ls in ((False, '-'), (True, (0, (4, 2.5)))):
                    v = h[h.null == null]
                    x = (np.abs(v.dy_cross) if var == 'dy_abs' else v[var]).to_numpy(float)
                    x = x[np.isfinite(x)]
                    if len(x) < 30:
                        continue
                    y = np.histogram(x, edges)[0] / len(x)
                    ax.stairs(y, edges, color=col if not null else P.MUTED,
                              lw=lw if not null else lw * 0.8, ls=ls,
                              alpha=alpha,
                              label=(f'legs < {cut:.0f} mm, '
                                     f'{"null" if null else "data"}:  '
                                     f'median {np.median(x):.0f}'))
                    rows += [dict(topology=topo, variable=var, cut_mm=cut,
                                  sample='null' if null else 'data', lo=lo,
                                  hi=hi, fraction=val)
                             for lo, hi, val in zip(edges[:-1], edges[1:], y)]
            ax.set_xscale(scale)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(lab, fontsize=9.5)
            if c == 0:
                ax.set_ylabel('fraction of pairs per bin')
            ax.set_title(topo, loc='left', fontsize=13, fontweight='bold',
                         color=P.INK)
            ax.legend(loc='upper left' if r == 0 else 'upper right', fontsize=8)
            P.strip(ax)
    fig.suptitle('How closely do the two legs actually meet?', x=0.01,
                 ha='left', fontsize=15, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    foot(fig, 'Top: the 3D distance of closest approach between the two fitted '
         'lines.  Bottom: the y disagreement at their transverse crossing (the '
         'transverse separation there is zero by construction).  Grey dashed: '
         'the shuffled-direction null through the same cut.   ' + FOOT)
    save(fig, 'image_agreement')
    pd.DataFrame(rows).to_csv(OUT / 'image_agreement.csv', index=False)


# --------------------------------------------------------------------------- #
def cut_scan(S, cap):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    series = (('A-D', 'x', '#0072B2', 'x from A'), ('C-D', 'x', '#009E73', 'x from C'),
              ('perpendicular', 'z', '#CC79A7', 'z from D'))
    for sel, coord, col, name in series:
        s = S[(S.topology == sel) & (S.estimator == 'xz')]
        ref = cap[0] if coord == 'x' else cap[1]
        for centre, ls, mk in (('axis', '-', 'o'), ('capsule', (0, (4, 2)), 's')):
            h = s[(s.cut_centre == centre)]
            hd = h[h['sample'] == 'data'].sort_values('cut_mm')
            hn = h[h['sample'] == 'null'].sort_values('cut_mm')
            lab = f'{name}, cut about the {centre}'
            axes[0].errorbar(hd.cut_mm, hd[f'{coord}_med'] - ref,
                             hd[f'{coord}_med_err'], color=col, ls=ls, marker=mk,
                             ms=4, label=lab)
            axes[1].plot(hd.cut_mm, hd[f'{coord}_rsig'] / hn[f'{coord}_rsig'].to_numpy(),
                         color=col, ls=ls, marker=mk, ms=4, label=lab)
            band = 'f_xband' if coord == 'x' else 'f_zband'
            axes[2].plot(hd.cut_mm, 100 * (hd[band].to_numpy() - hn[band].to_numpy()),
                         color=col, ls=ls, marker=mk, ms=4, label=lab)
    axes[0].axhline(0, color=P.COPPER, lw=1)
    axes[1].axhline(1, color=P.MUTED, lw=0.8)
    axes[2].axhline(0, color=P.MUTED, lw=0.8)
    for ax, t, yl in (
            (axes[0], 'Image centre, relative to the capsule', 'median − capsule  [mm]'),
            (axes[1], 'Image width, relative to the null', 'robust σ data / robust σ null'),
            (axes[2], 'Excess at the capsule', f'within ±{VI.BAND_MM:.0f} mm: data − null  [% of pairs]')):
        ax.set_xscale('log')
        ax.set_xticks([10, 20, 30, 60, 150])
        ax.set_xticklabels(['10', '20', '30', '60', 'none'])
        ax.minorticks_off()
        ax.set_xlabel('per-leg pointing cut  [mm]')
        ax.set_ylabel(yl)
        ax.set_title(t, loc='left', fontsize=12, color=P.INK)
        P.strip(ax)
    h, lab = axes[2].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.06), frameon=False)
    fig.suptitle('Against the per-leg pointing cut', x=0.01, ha='left',
                 fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    foot(fig, 'Solid: the cut is on each leg’s miss distance from the nominal '
         'beam axis.  Dashed: from the measured capsule position instead.  Robust '
         'σ = 1.4826 × MAD.  "none" is the 150 mm build ceiling.   ' + FOOT)
    save(fig, 'image_cut_scan')


# --------------------------------------------------------------------------- #
def per_run(PR):
    if PR.empty:
        return
    PR = PR.dropna(subset=['c']).copy()
    PR['num'] = PR.run.str.replace('run_', '').astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9), sharey=False)
    for ax, ch, col in ((axes[0], 'A', '#0072B2'), (axes[1], 'C', '#009E73')):
        g = PR[PR.chamber == ch].sort_values('num')
        x = np.arange(len(g))
        ax.errorbar(x, g.c, g.c_err, fmt='o', ms=4, color=col, lw=1,
                    label=f'pair image (x from chamber {ch})')
        ax.plot(x, g.single_track_x, 'D', ms=4.5, mfc='white', color=P.COPPER,
                label=f'single-track band crossing, chamber {ch}')
        ax.axhline(g.c.median(), color=col, lw=0.8, ls=':')
        ax.axhline(g.single_track_x.median(), color=P.COPPER, lw=0.8, ls=':')
        ax.set_xticks(x)
        ax.set_xticklabels(g.num, rotation=90, fontsize=7.5)
        ax.set_xlabel('run')
        ax.set_ylabel('capsule x  [mm]')
        ax.set_title(f'chamber {ch}: pairs {g.c.median():+.1f} (run spread '
                     f'{g.c.std():.1f}, typical error {g.c_err.median():.1f})   '
                     f'single tracks {g.single_track_x.median():+.1f} mm',
                     loc='left', fontsize=11, color=P.INK)
        ax.legend(fontsize=8.5, loc='lower left')
        P.strip(ax)
    fig.suptitle('Run by run: chamber A’s pair image is steady and ~2 mm off '
                 'its band crossing; chamber C’s scatters beyond its errors',
                 x=0.01, ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, f'Pairs with legs within {SEL_CUT:.0f} mm.  Per run only the x '
         'centre is fitted; blur and capsule fraction are fixed to that arm '
         'pair’s pooled fit and the null is pooled.   ' + FOOT)
    save(fig, 'image_per_run')


# --------------------------------------------------------------------------- #
def volume_3d(d, cap, F):
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter
    fx = fit_row(F, SEL_TOPO, SEL_CUT, 'x')
    g = VI.select(d, SEL_TOPO, SEL_CUT)
    D, N = g[~g.null], g[g.null]
    scale = (1 - float(fx.f)) * len(D) / max(len(N), 1)
    ex = np.arange(-60, 64, 4.0)
    ey = np.arange(-180, 190, 10.0)
    cols = ['vx_xz', 'vy_xz', 'vz_xz']
    Hd = np.histogramdd(D[cols].to_numpy(float), bins=(ex, ey, ex))[0]
    Hn = np.histogramdd(N[cols].to_numpy(float), bins=(ex, ey, ex))[0] * scale
    cx, cy, cz = [0.5 * (e[:-1] + e[1:]) for e in (ex, ey, ex)]
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')
    names, traces = [], []
    for H, name in ((Hd, 'data'), (Hd - Hn, 'data − no-source null')):
        Hs = gaussian_filter(H, 0.8)
        Hs = np.clip(Hs, 0, None) / max(Hs.max(), 1e-9)
        # plot axes: x -> x, z -> horizontal y, beam y -> vertical, so the
        # scene stands the way the apparatus does (beam upward)
        traces.append(go.Volume(
            x=X.ravel().astype(np.float32), y=Z.ravel().astype(np.float32),
            z=Y.ravel().astype(np.float32),
            value=Hs.ravel().astype(np.float32), isomin=0.15, isomax=1.0,
            opacity=0.14, surface_count=14, colorscale=[
                [0, '#e8d6ea'], [0.35, '#b77cbd'], [0.7, '#7d3a86'],
                [1, '#2a0f30']],
            colorbar=dict(title='relative<br>density', len=0.55, thickness=12),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=name, visible=(name == 'data'), hoverinfo='skip'))
        names.append(name)
    yv, rv = VI.capsule_profile()
    ph = np.linspace(0, 2 * np.pi, 36)
    Rm, Pm = np.meshgrid(rv, ph, indexing='ij')
    Ym = np.repeat(yv[:, None], len(ph), axis=1)
    traces.append(go.Surface(x=cap[0] + Rm * np.cos(Pm), y=cap[1] + Rm * np.sin(Pm),
                             z=Ym, showscale=False, opacity=0.5,
                             colorscale=[[0, '#d18a44'], [1, '#d18a44']],
                             name='He-3 capsule (single-track position)',
                             showlegend=True, hoverinfo='name'))
    traces.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-180, 180], mode='lines',
                               line=dict(color='#6a7583', width=4, dash='dash'),
                               name='nominal beam axis', hoverinfo='name'))
    fig = go.Figure(traces)
    nvol = len(names)
    buttons = [dict(label=nm, method='update',
                    args=[{'visible': [i == k for i in range(nvol)] + [True, True]}])
               for k, nm in enumerate(names)]
    fig.update_layout(
        updatemenus=[dict(type='buttons', direction='right', x=0.0, y=1.06,
                          xanchor='left', buttons=buttons, showactive=True)],
        scene=dict(xaxis_title='x [mm] (A/C)', yaxis_title='z [mm] (D)',
                   zaxis_title='y, beam [mm]', aspectmode='manual',
                   aspectratio=dict(x=1, y=1, z=2.2),
                   camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))),
        margin=dict(l=0, r=0, t=46, b=0), height=720,
        legend=dict(x=0.01, y=0.02), paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, Helvetica, Arial, sans-serif'))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT / 'image_3d.html', include_plotlyjs='cdn', full_html=True)
    div = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='vertex3d')
    (OUT / 'image_3d.div.html').write_text(div, encoding='utf-8')
    print(f'  -> image_3d.html  ({len(div) / 1e6:.1f} MB)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    want = set(x for x in a.only.split(',') if x)
    od = paths.out('pair_vertex')
    cap = VI.capsule_centre()
    P.use()
    d = VI.load(od)
    # keep_default_na=False: the `sample` column holds the literal 'null',
    # which read_csv otherwise turns into NaN and silently drops every null row
    S = pd.read_csv(od / 'image_stats.csv', keep_default_na=False,
                    na_values=[''])
    F = pd.read_csv(od / 'image_fit.csv')
    PR = pd.read_csv(od / 'image_fit_per_run.csv')
    jobs = dict(projections=lambda: projections(d, cap, F),
                profiles=lambda: profiles(d, cap, F),
                agreement=lambda: agreement(d),
                cut_scan=lambda: cut_scan(S, cap),
                per_run=lambda: per_run(PR),
                volume=lambda: volume_3d(d, cap, F))
    for k, fn in jobs.items():
        if not want or k in want:
            fn()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
