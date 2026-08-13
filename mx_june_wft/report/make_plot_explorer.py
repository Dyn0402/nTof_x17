#!/usr/bin/env python3
"""
make_plot_explorer.py — one plot per figure, zoomable, with its data beside it.

The fleet report's figures are multi-panel composites at fixed scale: fine as a
summary, useless for looking closely at one thing. This renders each panel as
its OWN figure, and writes the exact numbers behind it as a small CSV, so a
plot can be examined, or rebuilt differently, without re-running anything.

    plots/<key>/<name>.svg|png   the figure, one subject each
    plots/<key>/<name>.csv       exactly what is drawn (bins, profile points,
                                 grid cells) -- not the whole ray table
    explorer.html                card grid + full-screen zoom/pan viewer

1-D plots are SVG so they stay sharp at any zoom; 2-D densities and maps are
PNG at 200 dpi, where vector output would be enormous for no gain.

Input is plot_data/rays.csv (export_plot_data.py). Run that first.

    ../../.venv/bin/python mx_june_wft/report/make_plot_explorer.py [keys...]
"""
import base64
import html
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, HERE, os.path.join(REPO, 'mx_june_cosmic_qa')]

from qa_config import get_config, setup_paths                 # noqa: E402
setup_paths()
import matplotlib                                             # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.colors import LogNorm                         # noqa: E402

FLEET_REPORT = '/home/dylan/x17/cosmic_bench/Analysis/fleet_report'
KEYS = ['g_det3_wknd', 'o22_long_det2', 'g_det6_long', 'g_det7_long', 'g_det4']
LETTER = {'g_det3_wknd': 'A', 'o22_long_det2': 'B', 'g_det6_long': 'C',
          'g_det7_long': 'D', 'g_det4': 'E'}
CAT_COLOUR = {'within': '#1c6b3f', 'reco_far': '#c9761d',
              'hit_no_reco': '#a8322d', 'no_hit': '#8b8f94', 'spark': '#6b3fa0'}

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 260, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': .25, 'axes.axisbelow': True,
    'figure.autolayout': True, 'svg.fonttype': 'none',
})


class Plots:
    """Collects rendered figures and the data behind each."""

    def __init__(self, out_dir, rel_dir):
        self.out_dir, self.rel_dir = out_dir, rel_dir
        os.makedirs(out_dir, exist_ok=True)
        self.items = []

    def add(self, fig, name, title, caption, data=None, group='', png=False,
            dpi=None):
        ext = 'png' if png else 'svg'
        fig.savefig(os.path.join(self.out_dir, f'{name}.{ext}'),
                    bbox_inches='tight', **({'dpi': dpi} if dpi else {}))
        plt.close(fig)
        csv = None
        if data is not None:
            data.to_csv(os.path.join(self.out_dir, f'{name}.csv'), index=False)
            csv = f'{self.rel_dir}/{name}.csv'
        self.items.append(dict(name=name, title=title, caption=caption,
                               src=f'{self.rel_dir}/{name}.{ext}', csv=csv,
                               group=group))


def _hist(ax, v, bins, label, colour='#1d5fa8'):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n, edges = np.histogram(v, bins=bins)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ax.step(ctr, n, where='mid', color=colour, lw=1.4)
    ax.fill_between(ctr, n, step='mid', alpha=.18, color=colour)
    ax.set_xlabel(label)
    ax.set_ylabel('rays')
    return pd.DataFrame({'bin_centre': ctr, 'bin_lo': edges[:-1],
                         'bin_hi': edges[1:], 'count': n})


def _profile(x, y, edges, min_n=20):
    """Median of y in bins of x, with a bootstrap-free median error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() < min_n:
            continue
        yy = y[s]
        rows.append(dict(bin_lo=lo, bin_hi=hi, centre=.5 * (lo + hi),
                         n=int(s.sum()), median=float(np.median(yy)),
                         err=float(1.253 * np.std(yy, ddof=1) / np.sqrt(s.sum())),
                         mean=float(np.mean(yy))))
    return pd.DataFrame(rows)


def _frac_profile(x, flag, edges, min_n=20):
    """Fraction of `flag` true in bins of x, with a binomial error."""
    x = np.asarray(x, float)
    f = np.asarray(flag, bool)
    m = np.isfinite(x)
    x, f = x[m], f[m]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() < min_n:
            continue
        p = float(f[s].mean())
        rows.append(dict(bin_lo=lo, bin_hi=hi, centre=.5 * (lo + hi),
                         n=int(s.sum()), fraction=p,
                         err=float(np.sqrt(max(p * (1 - p), 1e-9) / s.sum()))))
    return pd.DataFrame(rows)


def _grid_csv(H, xe, ye, value='value'):
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    return pd.DataFrame({'x': X.ravel(), 'y': Y.ravel(),
                         value: np.asarray(H).ravel()})


def s68(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(np.abs(v - np.median(v)), 68.27)) if len(v) else np.nan


# ---------------------------------------------------------------- sliding scan
STEP_MM = 2.0          # grid pitch of the scan
TARGET_N = 150         # rays per circle the default radius aims for


def _disc(radius_mm, step):
    k = int(np.ceil(radius_mm / step))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return ((xx ** 2 + yy ** 2) * step ** 2 <= radius_mm ** 2).astype(float)


def _smear(H, disc):
    """Sum of H over the disc centred on each cell."""
    from scipy.signal import fftconvolve
    return fftconvolve(H, disc, mode='same')


def auto_radius(n_rays, box):
    """Radius holding ~TARGET_N rays on average, rounded to something sane.

    A literally 2 mm circle is not a statistics question anyone can answer: at
    this run's density it holds ~2 rays, so efficiency in it is 0, 50 or 100 %.
    The SCAN steps every 2 mm; the circle has to be wide enough to measure.
    """
    area = max((box['x1'] - box['x0']) * (box['y1'] - box['y0']), 1.0)
    r = np.sqrt(TARGET_N * area / (np.pi * max(n_rays, 1)))
    return float(np.clip(round(r), 6, 40))


def sliding_maps(d, box, radius_mm, step=STEP_MM):
    """Every sliding-circle quantity in one pass over the binned sums.

    Binning once and convolving with a disc gives the same answer as asking
    "which rays are within radius of this point" at every grid point, but in
    seconds rather than minutes: at a 2 mm step that is ~35,000 circles per
    map per detector.
    """
    xe = np.arange(box['x0'], box['x1'] + step, step)
    ye = np.arange(box['y0'], box['y1'] + step, step)
    disc = _disc(radius_mm, step)
    x, y = d['x'].to_numpy(float), d['y'].to_numpy(float)

    def binned(w=None, sel=None):
        xs, ys = (x, y) if sel is None else (x[sel], y[sel])
        ws = None if w is None else (w if sel is None else w[sel])
        H, _, _ = np.histogram2d(xs, ys, bins=[xe, ye], weights=ws)
        return _smear(H, disc)

    out = {'x_edges': xe, 'y_edges': ye, 'radius_mm': radius_mm, 'step': step}
    n_tot = binned()
    out['n_rays'] = n_tot

    r = d['r_mm'].to_numpy(float)
    for cut in (2.0, 5.0):
        hit = np.isfinite(r) & (r < cut)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[f'eff{cut:g}'] = np.where(n_tot >= 40,
                                          100 * binned(sel=hit) / n_tot, np.nan)

    def sigma_map(comps, window, min_n=60, passes=2):
        """Width of `comps` per circle, with the window re-tightened once.

        A fixed wide window measures the RMS INCLUDING the non-Gaussian
        shoulder, which is not what "resolution" means anywhere else in this
        analysis -- on detector A it reads 0.8 mm against the report's 0.44 mm
        core sigma. Re-running with a window of 2.5 sigma converges on the
        core, matching the iterative Gaussian used by the accounting.
        """
        sel_var = np.zeros(1)
        sig = n = None
        for _ in range(passes):
            sel = np.ones(len(d), bool)
            for c in comps:
                sel &= np.isfinite(c) & (np.abs(c - np.nanmedian(c)) < window)
            n = binned(sel=sel)
            with np.errstate(invalid='ignore', divide='ignore'):
                var = np.zeros_like(n)
                for c in comps:
                    cc = np.where(np.isfinite(c), c, 0.0)
                    m1 = binned(w=cc, sel=sel) / n
                    m2 = binned(w=cc ** 2, sel=sel) / n
                    var += np.maximum(m2 - m1 ** 2, 0)
                sig = np.where(n >= min_n, np.sqrt(var / len(comps)), np.nan)
            if np.isfinite(sig).any():
                window = float(np.clip(2.5 * np.nanmedian(sig),
                                       0.3 * window, window))
            sel_var = sig
        return sel_var, n, window

    # Spatial: the mis-association tail is not resolution, so start inside 5 mm
    # and let the window tighten onto the core.
    dx, dy = d['dx'].to_numpy(float), d['dy'].to_numpy(float)
    rr = np.where(np.isfinite(r), r, np.nan)
    keep = np.abs(rr - 0) < 5.0
    out['sigma_pos'], out['n_core'], out['win_pos'] = sigma_map(
        [np.where(keep, dx, np.nan), np.where(keep, dy, np.nan)], 5.0)

    # Angles, per plane
    for p in ('x', 'y'):
        col = f'dtheta_{p}_deg'
        if col in d:
            (out[f'sigma_theta_{p}'], _,
             out[f'win_theta_{p}']) = sigma_map([d[col].to_numpy(float)], 10.0)

    # Time: the two planes timestamp the same event, so their difference is
    # free of trigger jitter; per-plane sigma is that spread over root two.
    if {'x_t0', 'y_t0'} <= set(d.columns):
        dt = (d['x_t0'] - d['y_t0']).to_numpy(float)
        st, _, win = sigma_map([dt], 250.0)
        out['sigma_t'] = st / np.sqrt(2)
        out['win_t'] = win
    return out


def build_detector(key, root):
    cfg = get_config(key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    pd_dir = os.path.join(W, 'plot_data')
    d = pd.read_csv(os.path.join(pd_dir, 'rays.csv'))
    summary = json.load(open(os.path.join(pd_dir, 'summary.json')))
    L = LETTER.get(key, '?')
    rel = f'plots/{key}'
    P = Plots(os.path.join(root, rel), rel)
    box = summary['active_box']
    ok = d['within'].astype(bool)
    print(f'== {L} {key}: {len(d):,} rays')

    # ---------- where the rays went ----------
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for cat in ('no_hit', 'hit_no_reco', 'spark', 'reco_far', 'within'):
        s = d['category'] == cat
        if not s.any():
            continue
        ax.scatter(d.loc[s, 'x'], d.loc[s, 'y'], s=(1.5 if cat == 'within' else 6),
                   c=CAT_COLOUR[cat], label=f'{cat} ({s.sum():,})',
                   alpha=(.25 if cat == 'within' else .85), lw=0,
                   rasterized=True)
    ax.set_xlabel('reference x [mm]'); ax.set_ylabel('reference y [mm]')
    ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8, framealpha=.9)
    ax.set_title(f'{L} · reference crossings by outcome')
    P.add(fig, 'hitmiss_scatter', 'Hit / miss map',
          'Every reference ray in the active box, coloured by what the detector '
          'did there. Misses cluster where the detector is dead, not uniformly.',
          d[['event_id', 'x', 'y', 'category', 'r_mm']], 'Geometry', png=True)

    # ---------- efficiency map ----------
    nb = 40
    xe = np.linspace(box['x0'], box['x1'], nb + 1)
    ye = np.linspace(box['y0'], box['y1'], nb + 1)
    tot, _, _ = np.histogram2d(d['x'], d['y'], bins=[xe, ye])
    hit, _, _ = np.histogram2d(d.loc[ok, 'x'], d.loc[ok, 'y'], bins=[xe, ye])
    with np.errstate(invalid='ignore', divide='ignore'):
        effmap = np.where(tot >= 5, 100 * hit / tot, np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.pcolormesh(xe, ye, effmap.T, cmap='viridis', vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label='within 5 mm [%]')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_aspect('equal')
    ax.set_title(f'{L} · efficiency map ({nb}x{nb} bins, >=5 rays)')
    g = _grid_csv(effmap, xe, ye, 'efficiency_pct')
    g['n_rays'] = _grid_csv(tot, xe, ye, 'n')['n']
    P.add(fig, 'eff_map', 'Efficiency map',
          f'Fraction reconstructed within 5 mm, in {nb}x{nb} bins of the active '
          'box. Bins with fewer than 5 rays are blank.', g, 'Geometry', png=True)

    for ax_name, other in (('x', 'y'), ('y', 'x')):
        edges = np.linspace(box[f'{ax_name}0'], box[f'{ax_name}1'], 41)
        prof = _frac_profile(d[ax_name], ok, edges)
        fig, a = plt.subplots(figsize=(7.2, 4.2))
        a.errorbar(prof['centre'], 100 * prof['fraction'],
                   yerr=100 * prof['err'], fmt='o-', ms=3.5, lw=1.2,
                   color='#1d5fa8')
        a.set_xlabel(f'{ax_name} [mm]'); a.set_ylabel('within 5 mm [%]')
        a.set_ylim(0, 100)
        a.set_title(f'{L} · efficiency vs {ax_name}')
        P.add(fig, f'eff_vs_{ax_name}', f'Efficiency vs {ax_name}',
              f'Efficiency projected onto {ax_name}, collapsing {other}. '
              'Structure here is detector geometry (dead strips, edges).',
              prof, 'Geometry')

    # ---------- residuals ----------
    for comp, lab in (('dx', 'x'), ('dy', 'y')):
        v = d.loc[np.isfinite(d[comp]), comp]
        lim = np.nanpercentile(np.abs(v), 99) if len(v) else 5
        fig, a = plt.subplots(figsize=(7.2, 4.2))
        data = _hist(a, v, np.linspace(-lim, lim, 161),
                     f'{comp} = reconstructed - reference [mm]')
        a.axvline(0, color='#888', lw=.8)
        a.set_title(f'{L} · {lab} residual  (median {np.median(v):+.3f} mm, '
                    f'sigma68 {s68(v):.3f} mm)')
        P.add(fig, f'resid_{comp}', f'{lab.upper()} residual',
              f'Reconstructed minus reference along {lab}. Width here is the '
              'position resolution folded with the M3 pointing error.',
              data, 'Residuals')

    v = d.loc[np.isfinite(d['r_mm']), 'r_mm']
    fig, a = plt.subplots(figsize=(7.2, 4.2))
    data = _hist(a, v, np.linspace(0, min(30, np.nanmax(v)), 121),
                 'radial residual |r| [mm]')
    a.set_yscale('log')
    a.axvline(5, color='#a8322d', lw=1.1, ls='--')
    a.text(5.2, a.get_ylim()[1] * .4, '5 mm', color='#a8322d', fontsize=9)
    a.set_title(f'{L} · radial residual (median {np.median(v):.3f} mm)')
    P.add(fig, 'resid_r', 'Radial residual',
          'Log scale, so the tail beyond the 5 mm efficiency cut is visible. '
          'The tail is mis-association, not resolution.', data, 'Residuals')

    # Median, not mean, and a coarser grid: the mean over a handful of rays is
    # dominated by the mis-association tail, which renders as speckle and hides
    # the thing the map is for (is the resolution uniform across the face).
    good = d[np.isfinite(d['r_mm'])]
    nbr = 26
    xer = np.linspace(box['x0'], box['x1'], nbr + 1)
    yer = np.linspace(box['y0'], box['y1'], nbr + 1)
    ix = np.clip(np.digitize(good['x'], xer) - 1, 0, nbr - 1)
    iy = np.clip(np.digitize(good['y'], yer) - 1, 0, nbr - 1)
    rmap = np.full((nbr, nbr), np.nan)
    cmap_n = np.zeros((nbr, nbr), int)
    rv = good['r_mm'].to_numpy()
    for i in range(nbr):
        for j in range(nbr):
            s = (ix == i) & (iy == j)
            cmap_n[i, j] = s.sum()
            if s.sum() >= 10:
                rmap[i, j] = np.median(rv[s])
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    # Span the 5-95 percentile of the map, not 0-max: these values cluster in a
    # narrow band well above zero, so anchoring at zero spends the whole
    # colormap on a range nothing occupies and flattens the real structure.
    finite = rmap[np.isfinite(rmap)]
    lo95, hi95 = ((float(np.percentile(finite, 5)), float(np.percentile(finite, 95)))
                  if finite.size else (0.0, 1.0))
    im = ax.pcolormesh(xer, yer, rmap.T, cmap='magma_r', vmin=lo95, vmax=hi95)
    fig.colorbar(im, ax=ax, label='median |r| [mm]')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_aspect('equal')
    ax.set_title(f'{L} · residual map ({nbr}x{nbr} bins, >=10 rays)')
    g = _grid_csv(rmap, xer, yer, 'median_r_mm')
    g['n_rays'] = _grid_csv(cmap_n, xer, yer, 'n')['n']
    P.add(fig, 'resid_map', 'Residual map',
          'Median radial residual across the face — shows whether the '
          'resolution is uniform or degrades in a region. Median, so the '
          'mis-association tail does not set the colour.',
          g, 'Residuals', png=True)

    # ---------- sliding-circle scans ----------
    R = auto_radius(len(d), box)
    sm = sliding_maps(d, box, R)
    sxe, sye = sm['x_edges'], sm['y_edges']
    per_circle = float(np.nanmedian(sm['n_rays']))
    p_eff = float(ok.mean())
    stat_pct = 100 * np.sqrt(max(p_eff * (1 - p_eff), 1e-9) / max(per_circle, 1))
    grid_note = (f'Circle r = {R:.0f} mm stepped every {STEP_MM:g} mm '
                 f'(~{per_circle:.0f} rays per circle, so neighbouring circles '
                 f'overlap heavily). Statistical spread at this circle size is '
                 f'about +/-{stat_pct:.1f}% on efficiency — structure smaller '
                 f'than that is noise, not the detector.')

    def slide(field, name, title, cbar, caption, cmap='viridis',
              lo_pct=2, hi_pct=98, vmin=None, vmax=None, group='Sliding scans'):
        M = sm.get(field)
        if M is None or not np.isfinite(M).any():
            return
        fin = M[np.isfinite(M)]
        v0 = vmin if vmin is not None else float(np.percentile(fin, lo_pct))
        v1 = vmax if vmax is not None else float(np.percentile(fin, hi_pct))
        fig, a = plt.subplots(figsize=(6.6, 5.6))
        im = a.pcolormesh(sxe, sye, M.T, cmap=cmap, vmin=v0, vmax=v1)
        fig.colorbar(im, ax=a, label=cbar)
        a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]'); a.set_aspect('equal')
        a.set_title(f'{L} · {title}')
        g2 = _grid_csv(M, sxe, sye, field)
        g2['n_rays_in_circle'] = _grid_csv(sm['n_rays'], sxe, sye, 'n')['n']
        P.add(fig, name, title, f'{caption} {grid_note}', g2, group, png=True,
              dpi=170)

    slide('n_rays', 'slide_stats', 'Ray statistics (sliding)', 'rays in circle',
          'How many reference rays each circle contains — the exposure behind '
          'every other sliding map, and the reason the edges are noisier.',
          cmap='cividis', lo_pct=0, hi_pct=100)
    slide('eff2', 'slide_eff2', 'Efficiency, 2 mm match (sliding)',
          'within 2 mm [%]',
          'Fraction of rays reconstructed within 2 mm of the reference.')
    slide('eff5', 'slide_eff5', 'Efficiency, 5 mm match (sliding)',
          'within 5 mm [%]',
          'The same scan at the 5 mm match radius the headline numbers use.')
    slide('sigma_pos', 'slide_sigma_pos', 'Spatial resolution (sliding)',
          'sigma position [mm]',
          'Width of the residual per axis, on a window that re-tightens onto '
          'the core (~2.5 sigma) so the mis-association shoulder does not set '
          'it. This is a windowed RMS and sits BETWEEN the two widths the '
          'report quotes — on detector A, 0.53 mm here against 0.44 mm for the '
          'Gaussian core fit and 0.63 mm for sigma68. Compare map to map, not '
          'to the headline number.',
          cmap='magma_r')
    for p in ('x', 'y'):
        slide(f'sigma_theta_{p}', f'slide_sigma_theta_{p}',
              f'Angle resolution {p.upper()} (sliding)', 'sigma theta [deg]',
              f'RMS of the {p}-plane angle residual over |dtheta| < 10 deg.',
              cmap='magma_r')
    slide('sigma_t', 'slide_sigma_t', 'Time resolution (sliding)',
          'sigma t per plane [ns]',
          'From the X-Y plane time difference (trigger jitter cancels), '
          'divided by root two. Sampling-limited: the DAQ samples every 60 ns, '
          'so this is fit granularity, NOT the scintillator-referenced time '
          'resolution of the June timing study.', cmap='magma_r')

    # ---------- position correlation ----------
    for p in ('x', 'y'):
        s = np.isfinite(d[f'det_{p}'])
        fig, a = plt.subplots(figsize=(6.0, 5.6))
        # 140 bins over a diagonal locus leaves most cells empty, which reads
        # as speckle rather than a correlation. Scale the binning to the
        # sample so the diagonal is a continuous ridge.
        nb2 = int(np.clip(np.sqrt(max(s.sum(), 1)) / 1.6, 40, 110))
        h = a.hist2d(d.loc[s, p], d.loc[s, f'det_{p}'], bins=nb2,
                     cmap='viridis', norm=LogNorm(vmin=1))
        fig.colorbar(h[3], ax=a, label='rays')
        lo = min(a.get_xlim()[0], a.get_ylim()[0])
        hi = max(a.get_xlim()[1], a.get_ylim()[1])
        a.plot([lo, hi], [lo, hi], color='#e8e8e8', lw=.9, ls='--')
        a.set_xlabel(f'reference {p} [mm]')
        a.set_ylabel(f'reconstructed {p} [mm]')
        a.set_title(f'{L} · {p} position correlation')
        P.add(fig, f'poscorr_{p}', f'{p.upper()} position correlation',
              f'Reconstructed against reference {p}. Slope is the position '
              'scale; offset is residual misalignment.',
              _grid_csv(h[0], h[1], h[2], 'rays'), 'Correlation', png=True)

    # ---------- angles ----------
    for p in ('x', 'y'):
        s = np.isfinite(d[f'{p}_theta_deg']) & np.isfinite(d[f'ref_theta_{p}_deg'])
        if s.sum() < 50:
            continue
        fig, a = plt.subplots(figsize=(6.0, 5.6))
        lim = float(np.nanpercentile(np.abs(d.loc[s, f'ref_theta_{p}_deg']), 99))
        nba = int(np.clip(np.sqrt(max(s.sum(), 1)) / 1.6, 40, 110))
        h = a.hist2d(d.loc[s, f'ref_theta_{p}_deg'], d.loc[s, f'{p}_theta_deg'],
                     bins=nba, range=[[-lim, lim], [-lim, lim]],
                     cmap='viridis', norm=LogNorm(vmin=1))
        fig.colorbar(h[3], ax=a, label='planes')
        a.plot([-lim, lim], [-lim, lim], color='#e8e8e8', lw=.9, ls='--')
        a.set_xlabel(f'reference theta_{p} [deg]')
        a.set_ylabel(f'reconstructed theta_{p} [deg]')
        a.set_title(f'{L} · {p} angle correlation (n={s.sum():,})')
        P.add(fig, f'angcorr_{p}', f'{p.upper()} angle correlation',
              'Full coverage — every fitted plane, including head-on tracks '
              'that the slope_reliable gate used to discard.',
              _grid_csv(h[0], h[1], h[2], 'planes'), 'Angles', png=True)

        v = d.loc[s, f'dtheta_{p}_deg']
        fig, a = plt.subplots(figsize=(7.2, 4.2))
        data = _hist(a, v, np.linspace(-12, 12, 161),
                     f'theta_{p} residual [deg]', colour='#6b3fa0')
        a.axvline(0, color='#888', lw=.8)
        a.set_title(f'{L} · {p} angle residual (bias {np.median(v):+.2f}, '
                    f'sigma68 {s68(v):.2f} deg)')
        P.add(fig, f'dtheta_{p}', f'{p.upper()} angle residual',
              'Reconstructed minus reference angle. The bias is the number the '
              'w0/kw constants fix; it should sit within a few hundredths.',
              data, 'Angles')

        edges = np.linspace(-lim, lim, 25)
        prof = _profile(d.loc[s, f'ref_theta_{p}_deg'], v, edges)
        if len(prof):
            fig, a = plt.subplots(figsize=(7.2, 4.2))
            a.errorbar(prof['centre'], prof['median'], yerr=prof['err'],
                       fmt='o-', ms=3.5, lw=1.2, color='#1d5fa8')
            a.axhline(0, color='#888', lw=.8)
            a.set_xlabel(f'reference theta_{p} [deg]')
            a.set_ylabel('angle bias [deg]')
            a.set_title(f'{L} · {p} angle bias vs incidence')
            P.add(fig, f'bias_vs_theta_{p}', f'{p.upper()} bias vs incidence',
                  'Angle bias against true incidence. Flat means the mapping '
                  'is right at every angle, not just on average.',
                  prof, 'Angles')

    # ---------- fit quality ----------
    for col, name, lab, rng in (
            ('x_n_strips', 'nstrips_x', 'strips in the X cluster', (0, 30)),
            ('y_n_strips', 'nstrips_y', 'strips in the Y cluster', (0, 30)),
            ('x_chi2', 'chi2_x', 'X fit chi2', None),
            ('y_chi2', 'chi2_y', 'Y fit chi2', None),
            ('x_q_sum', 'qsum_x', 'X cluster charge [ADC]', None),
            ('y_q_sum', 'qsum_y', 'Y cluster charge [ADC]', None)):
        if col not in d:
            continue
        v = d.loc[np.isfinite(d[col]), col]
        if not len(v):
            continue
        hi = rng[1] if rng else float(np.nanpercentile(v, 99))
        lo = rng[0] if rng else float(max(np.nanmin(v), 0))
        fig, a = plt.subplots(figsize=(7.2, 4.2))
        data = _hist(a, v, np.linspace(lo, hi, 81), lab, colour='#1c6b3f')
        a.set_title(f'{L} · {lab} (median {np.median(v):.1f})')
        P.add(fig, name, lab.capitalize(),
              'Distribution over all fitted planes in the active box.',
              data, 'Fit quality')

    # ---------- how the answer moves with the reference cut ----------
    if {'chi2_x', 'chi2_y'} <= set(d.columns) and d['chi2_x'].notna().any():
        c2 = np.maximum(d['chi2_x'].to_numpy(float), d['chi2_y'].to_numpy(float))
        far = (d['category'] == 'reco_far').to_numpy(bool)
        hnr = (d['category'] == 'hit_no_reco').to_numpy(bool)
        cuts = np.round(np.arange(0.05, float(np.nanmax(c2)) + 1e-9, 0.05), 3)
        rows = []
        for c in cuts:
            s = np.isfinite(c2) & (c2 < c)
            if s.sum() < 100:
                continue
            rows.append(dict(chi2_cut=float(c), n_rays=int(s.sum()),
                             frac_rays_kept=float(s.mean()),
                             within_5mm_pct=float(100 * ok.to_numpy(bool)[s].mean()),
                             reco_far_pct=float(100 * far[s].mean()),
                             hit_no_reco_pct=float(100 * hnr[s].mean())))
        scan = pd.DataFrame(rows)
        if len(scan) > 2:
            fig, a = plt.subplots(figsize=(7.6, 4.4))
            a.plot(scan['chi2_cut'], scan['within_5mm_pct'], 'o-', ms=3.5,
                   lw=1.4, color='#1c6b3f', label='within 5 mm')
            a.set_xlabel('M3 reference cut: max(chi2_x, chi2_y) <')
            a.set_ylabel('within 5 mm [%]', color='#1c6b3f')
            a.tick_params(axis='y', labelcolor='#1c6b3f')
            a2 = a.twinx()
            a2.plot(scan['chi2_cut'], scan['reco_far_pct'], 's--', ms=3.5,
                    lw=1.4, color='#c9761d', label='reco_far')
            a2.set_ylabel('reco_far [%]', color='#c9761d')
            a2.tick_params(axis='y', labelcolor='#c9761d')
            a2.grid(False)
            a.axvline(1.0, color='#888', lw=1.0, ls=':')
            a.text(0.995, a.get_ylim()[0], ' frozen recipe', rotation=90,
                   va='bottom', ha='right', fontsize=8, color='#666')
            a.set_title(f'{L} · efficiency and reco_far vs reference chi2 cut')
            P.add(fig, 'chi2_scan', 'Efficiency vs reference chi2 cut',
                  'Both quantities against the M3 track-quality cut. The cut '
                  'can only be TIGHTENED: reconstruction ran on rays already '
                  'passing chi2 < 1 and NClus >= 4, so there is nothing to '
                  'score beyond the frozen recipe. A flat curve means the '
                  'reference selection is not manufacturing the efficiency; '
                  'a rise toward tighter cuts means some of what is called '
                  'inefficiency is reference mis-pointing.',
                  scan, 'Reference cut')

    return dict(key=key, letter=L, detector=cfg.DET_NAME,
                run=f'{cfg.RUN}/{cfg.SUB_RUN}', summary=summary,
                n_rays=int(len(d)), items=P.items,
                rays_csv=os.path.relpath(os.path.join(pd_dir, 'rays.csv'),
                                         FLEET_REPORT))


def build_fleet(root, dets):
    rel = 'plots/fleet'
    P = Plots(os.path.join(root, rel), rel)
    fl = pd.read_csv(os.path.join(FLEET_REPORT, 'plot_data', 'fleet.csv'))
    fl = fl.sort_values('letter')
    lbl = [f'{r.letter} ({r.detector.replace("mx17_", "det")})'
           for r in fl.itertuples()]
    for col, name, ylab, title in (
            ('within_5mm_pct', 'fleet_eff', 'within 5 mm [%]',
             'Reconstruction efficiency'),
            ('core_sigma_mm', 'fleet_sigma', 'core sigma [mm]',
             'Position resolution (core)'),
            ('s68_lt5_x_deg', 'fleet_theta_x', 'sigma68 [deg]',
             'Angle resolution X (|theta|<5 deg)'),
            ('s68_lt5_y_deg', 'fleet_theta_y', 'sigma68 [deg]',
             'Angle resolution Y (|theta|<5 deg)')):
        if col not in fl or fl[col].isna().all():
            continue
        fig, a = plt.subplots(figsize=(7.0, 4.2))
        a.bar(lbl, fl[col], color='#1d5fa8', width=.62)
        for i, v in enumerate(fl[col]):
            if np.isfinite(v):
                a.text(i, v, f'{v:.2f}' if v < 10 else f'{v:.1f}',
                       ha='center', va='bottom', fontsize=9)
        a.set_ylabel(ylab); a.set_title(title)
        a.grid(axis='x', alpha=0)
        P.add(fig, name, title, 'One bar per detector, from fleet.csv.',
              fl[['letter', 'detector', col]], 'Fleet')
    return dict(key='fleet', letter='Fleet', detector='all five',
                run='June 2026 cosmic bench', summary={}, n_rays=int(fl['n_rays'].sum()),
                items=P.items, rays_csv='plot_data/fleet.csv')


PAGE_CSS = """
:root{--bg:#fff;--fg:#16191c;--muted:#5b6167;--rule:#e1e5e9;--card:#f7f9fb;
--accent:#1d5fa8;--shadow:0 1px 3px rgba(0,0,0,.08);}
@media (prefers-color-scheme:dark){:root{--bg:#111417;--fg:#e7eaed;
--muted:#98a1a9;--rule:#272d33;--card:#191d22;--accent:#79b3f2;
--shadow:0 1px 3px rgba(0,0,0,.5);}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:1.5rem 1.25rem .9rem;max-width:1500px;margin:0 auto}
h1{margin:0 0 .3rem;font-size:1.45rem;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:.9rem}
nav{position:sticky;top:0;z-index:5;background:var(--bg);
border-bottom:1px solid var(--rule);padding:.55rem 1.25rem;
display:flex;gap:.4rem;flex-wrap:wrap;align-items:center}
nav a{color:var(--fg);text-decoration:none;padding:.28rem .7rem;border-radius:20px;
font-size:.87rem;font-weight:600;border:1px solid var(--rule)}
nav a:hover{border-color:var(--accent);color:var(--accent)}
main{max-width:1500px;margin:0 auto;padding:0 1.25rem 4rem}
section{padding-top:1.6rem}
h2{font-size:1.15rem;margin:.2rem 0 .1rem}
.meta{color:var(--muted);font-size:.85rem;margin-bottom:.8rem}
.meta a{color:var(--accent)}
h3.group{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;
color:var(--muted);margin:1.3rem 0 .5rem;padding-bottom:.3rem;
border-bottom:1px solid var(--rule)}
.grid{display:grid;gap:.9rem;
grid-template-columns:repeat(auto-fill,minmax(290px,1fr))}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;
overflow:hidden;cursor:zoom-in;box-shadow:var(--shadow);transition:transform .12s}
.card:hover{transform:translateY(-2px);border-color:var(--accent)}
.card img{width:100%;display:block;background:#fff;aspect-ratio:4/3;
object-fit:contain}
.card .t{padding:.5rem .65rem .6rem}
.card .t b{display:block;font-size:.92rem}
.card .t span{color:var(--muted);font-size:.8rem;display:block;margin-top:.15rem}
#viewer{position:fixed;inset:0;background:rgba(10,12,14,.94);z-index:50;
display:none;flex-direction:column}
#viewer.on{display:flex}
#vbar{display:flex;gap:.5rem;align-items:center;padding:.6rem .9rem;
color:#eee;font-size:.9rem;border-bottom:1px solid #333;flex-wrap:wrap;
background:#15181b}
#vbar b{font-size:.98rem}
#vbar .sp{flex:1}
#vbar button,#vbar a{background:#23282e;color:#eee;border:1px solid #3a4149;
border-radius:5px;padding:.3rem .65rem;font-size:.83rem;cursor:pointer;
text-decoration:none}
#vbar button:hover,#vbar a:hover{border-color:#79b3f2;color:#79b3f2}
#vstage{flex:1;overflow:hidden;position:relative;cursor:grab;background:#fff}
#vstage.drag{cursor:grabbing}
#vimg{position:absolute;transform-origin:0 0;user-select:none;
-webkit-user-drag:none}
#vcap{color:#b9c0c7;font-size:.85rem;padding:.55rem .9rem;border-top:1px solid #333;
background:#15181b}
"""

PAGE_JS = """
const PLOTS = __PLOTS__;
let cur = -1, scale = 1, tx = 0, ty = 0, natural = [0,0];
const V = document.getElementById('viewer'), IMG = document.getElementById('vimg'),
      ST = document.getElementById('vstage');

function fit(){
  const r = ST.getBoundingClientRect();
  if(!natural[0]) return;
  scale = Math.min(r.width/natural[0], r.height/natural[1]) * 0.97;
  tx = (r.width - natural[0]*scale)/2; ty = (r.height - natural[1]*scale)/2;
  apply();
}
function apply(){
  IMG.style.width = natural[0]+'px'; IMG.style.height = natural[1]+'px';
  IMG.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`;
}
function open_(i){
  cur = (i + PLOTS.length) % PLOTS.length;
  const p = PLOTS[cur];
  document.getElementById('vtitle').textContent = p.title;
  document.getElementById('vsub').textContent = p.det;
  document.getElementById('vcap').textContent = p.caption;
  const dl = document.getElementById('vdata');
  if(p.csv){ dl.href = p.csv; dl.style.display=''; } else { dl.style.display='none'; }
  document.getElementById('vimgdl').href = p.src;
  IMG.onload = () => { natural = [IMG.naturalWidth||1200, IMG.naturalHeight||900]; fit(); };
  IMG.src = p.src;
  V.classList.add('on');
}
function close_(){ V.classList.remove('on'); IMG.src=''; }
document.querySelectorAll('.card').forEach(c => {
  c.addEventListener('click', () => open_(+c.dataset.i));
  // Portable build ships each image once, in PLOTS; hydrate the thumbnails
  // from it rather than repeating every data URI in the markup.
  const im = c.querySelector('img'), p = PLOTS[+c.dataset.i];
  if(im && !im.getAttribute('src') && p) im.src = p.src;
});
document.getElementById('vclose').onclick = close_;
document.getElementById('vfit').onclick = fit;
document.getElementById('vprev').onclick = () => open_(cur-1);
document.getElementById('vnext').onclick = () => open_(cur+1);
V.addEventListener('click', e => { if(e.target === V) close_(); });
document.addEventListener('keydown', e => {
  if(!V.classList.contains('on')) return;
  if(e.key === 'Escape') close_();
  if(e.key === 'ArrowRight') open_(cur+1);
  if(e.key === 'ArrowLeft') open_(cur-1);
  if(e.key === '0') fit();
});
ST.addEventListener('wheel', e => {
  e.preventDefault();
  const r = ST.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const f = Math.exp(-e.deltaY * 0.0016);
  const ns = Math.min(Math.max(scale * f, 0.05), 60);
  tx = mx - (mx - tx) * (ns/scale); ty = my - (my - ty) * (ns/scale);
  scale = ns; apply();
}, {passive:false});
let down = null;
ST.addEventListener('pointerdown', e => {
  down = {x:e.clientX, y:e.clientY, tx, ty}; ST.classList.add('drag');
  ST.setPointerCapture(e.pointerId);
});
ST.addEventListener('pointermove', e => {
  if(!down) return;
  tx = down.tx + (e.clientX - down.x); ty = down.ty + (e.clientY - down.y); apply();
});
ST.addEventListener('pointerup', () => { down = null; ST.classList.remove('drag'); });
ST.addEventListener('dblclick', fit);
window.addEventListener('resize', () => { if(V.classList.contains('on')) fit(); });
"""


INLINE_CSV_MAX = 60 * 1024      # keep the page portable; big tables stay on disk


def _data_uri(path, mime):
    with open(path, 'rb') as f:
        return f'data:{mime};base64,' + base64.b64encode(f.read()).decode()


def build_page(root, dets, selfcontained=False, out_name='explorer.html'):
    """selfcontained: inline every image so the page works offline, off any
    filesystem, with nothing beside it. Small per-plot CSVs are inlined as
    download links too; the big ray tables and grids stay as files, since
    embedding them would multiply the page size for data nobody reads on a
    phone."""
    flat, cards = [], {}
    for det in dets:
        cards[det['key']] = []
        for it in det['items']:
            src, csv = it['src'], it['csv'] or ''
            if selfcontained:
                p = os.path.join(root, it['src'])
                src = _data_uri(p, 'image/svg+xml' if p.endswith('.svg')
                                else 'image/png')
                cp = os.path.join(root, csv) if csv else None
                csv = (_data_uri(cp, 'text/csv')
                       if cp and os.path.getsize(cp) <= INLINE_CSV_MAX else '')
            flat.append(dict(title=f"{det['letter']} · {it['title']}",
                             det=f"{det['detector']} — {det['run']}",
                             caption=it['caption'], src=src, csv=csv))
            cards[det['key']].append((len(flat) - 1, it))

    nav = ''.join(f'<a href="#{d["key"]}">{d["letter"]}</a>' for d in dets)
    body = []
    for det in dets:
        s = det['summary'] or {}
        eff = (s.get('efficiency') or {}).get('within_R')
        head = (f'{det["n_rays"]:,} rays'
                + (f' · {eff:.1f}% within 5 mm' if eff else '')
                + (f' · <a href="{det["rays_csv"]}">full data (CSV)</a>'
                   if not selfcontained else
                   ' · full per-ray CSV in the Analysis tree'))
        body.append(f'<section id="{det["key"]}"><h2>{html.escape(det["letter"])} '
                    f'— {html.escape(det["detector"])}</h2>'
                    f'<div class="meta">{html.escape(det["run"])} · {head}</div>')
        group, open_grid = None, False
        for idx, it in cards[det['key']]:
            if it['group'] != group:
                if open_grid:
                    body.append('</div>')
                group = it['group']
                body.append(f'<h3 class="group">{html.escape(group)}</h3>'
                            '<div class="grid">')
                open_grid = True
            body.append(
                f'<div class="card" data-i="{idx}">'
                f'<img loading="lazy" alt=""'
                f'{"" if selfcontained else " src=" + chr(34) + html.escape(it["src"]) + chr(34)}>'
                f'<div class="t"><b>{html.escape(it["title"])}</b>'
                f'<span>{html.escape(it["caption"][:96])}'
                f'{"…" if len(it["caption"]) > 96 else ""}</span></div></div>')
        if open_grid:
            body.append('</div>')
        body.append('</section>')

    extra_nav = ('' if selfcontained else
                 '<a href="report.html">summary report</a>'
                 '<a href="plot_data/fleet.csv">fleet.csv</a>')
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>June fleet — plot explorer</title>
<style>{PAGE_CSS}</style></head><body>
<header>
  <h1>June cosmic bench — plot explorer</h1>
  <div class="sub">One plot per subject, click to zoom. Every figure has the
  numbers behind it as CSV, and each detector links its full per-ray table.
  Waveform-first reconstruction, w0/kw applied in reco.</div>
</header>
<nav>{nav}<span style="flex:1"></span>{extra_nav}</nav>
<main>{''.join(body)}</main>
<div id="viewer">
  <div id="vbar">
    <b id="vtitle"></b><span id="vsub" style="color:#9aa3ab"></span>
    <span class="sp"></span>
    <button id="vprev">&larr;</button><button id="vnext">&rarr;</button>
    <button id="vfit">fit</button>
    <a id="vdata" download>data CSV</a><a id="vimgdl" download>image</a>
    <button id="vclose">close</button>
  </div>
  <div id="vstage"><img id="vimg" alt=""></div>
  <div id="vcap"></div>
</div>
<script>{PAGE_JS.replace('__PLOTS__', json.dumps(flat))}</script>
</body></html>"""
    out = os.path.join(root, out_name)
    with open(out, 'w') as f:
        f.write(doc)
    print(f'\nwrote {out} ({len(doc) // 1024} kB, {len(flat)} plots)')
    return out


def main():
    keys = [a for a in sys.argv[1:] if not a.startswith('-')] or KEYS
    root = FLEET_REPORT
    dets = [build_detector(k, root) for k in keys]
    dets.append(build_fleet(root, dets))
    build_page(root, dets)
    if '--selfcontained' in sys.argv:
        build_page(root, dets, selfcontained=True,
                   out_name='explorer_selfcontained.html')


if __name__ == '__main__':
    main()
