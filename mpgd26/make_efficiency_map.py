#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_efficiency_map.py -- the SLIDING-KERNEL efficiency map for the MPGD2026
talk's efficiency slide.

    ../.venv/bin/python make_efficiency_map.py
    ../.venv/bin/python make_efficiency_map.py --edge-hits 20 --print
    ../.venv/bin/python make_efficiency_map.py --no-slide

Writes ``figures/efficiency_map_sliding.png`` (+ the companion
``_rays.png``) and, unless --no-slide, the deck copy
``slides/assets/img/efficiency_map_sliding.png``.

HOW WE GOT HERE (2026-08-26, one afternoon):

1. "it's overlapping transparent circles" -- it wasn't; it was already a
   grid-of-x,y-under-a-circular-kernel imshow map (mx_june_wft/report/
   make_june_figs.py:sliding_map, a FIXED-radius circular kernel, ported here
   as an FFT convolution for speed).

2. "I have the old efficiency maps ... r=4.1mm kernel and go almost to the
   corner" (``june_detectors_overview.pdf``, 2026-07-12). That map is a
   one-off run of ``mx_june_cosmic_qa/12_efficiency_map_sliding.py
   --edge-hits=10``, which derives ONE radius from the mean ray density so
   an edge pixel holds ~10 rays: ``kernel = sqrt(2*edge_hits/(pi*density))``.
   Ported it (``--edge-hits``, below) -- on today's data (21,953 rays, the
   current project-standard M3 cut chi2<1.0 & NClus=4, vs. the PDF's looser
   "v2: NClus>=3 & chi2<5" giving 52,006) it derives r=6.3mm and still looks
   more textured than the PDF, because there are 2.4x fewer total rays.

3. Tried the June script's OTHER "more advanced" mode (k-NN adaptive kernel)
   and, when that still looked textured, a Gaussian-weighted soft kernel
   (smooth by construction, no hard boundary) -- Dylan didn't want either:
   "I still don't believe we should get so many circle structures with the
   hard in/out scan ... I don't think the change in stats alone caused the
   circle structures." Fair to ask for evidence rather than an assertion, so:

   TESTED FOR A BUG, DIRECTLY, RATHER THAN ARGUING FROM DENSITY ARITHMETIC.
   At r=6.3mm, sampled INDEPENDENT (non-overlapping, spacing = 2r) circles
   and compared the observed circle-to-circle efficiency variance to the
   binomial prediction p(1-p)/n:
     - all circles (edges + interior): var ratio 1.64x binomial -- real excess.
     - circles >=40mm from the box edge (away from the corner dip): ratio
       1.07x -- consistent with pure Poisson noise, no bug.
   Also checked event-ID order for time-clustered misses (a bug -- e.g. a
   dead-time window -- would show as bursts): mean/max consecutive-miss run
   lengths matched the plain-iid-Bernoulli prediction almost exactly (1.073
   observed vs. 1.075 expected for the measured miss rate). No clustering in
   time either. So: the interior "circle" texture IS explained by the lower
   ray count (n~20/circle at this radius) -- verified by test, not asserted
   -- and the 1.64x excess seen INCLUDING the edges is the already-documented
   real corner/edge dip, not a second bug on top of it.

DEFAULT: the hard circular kernel, radius auto-derived from the ray density
(``--edge-hits``, default 10, same target as the PDF's own derivation) so it
tightens or loosens automatically if the reference dataset is reprocessed.
``--kernel`` overrides it with a literal radius instead.

DATA -- the highest-statistics det3 set on disk (see above for why this is
fewer rays than the July PDF used):

    g_det3_wknd = mx17_det3_p2_det1_overnight_6-27-26 /
                  long_run_p2_det1_sanity_check / mx17_3
    21,953 reference rays (wft basis, M3 chi2<1.0 & NClus=4), 93.1% within 5mm

    <OUT_BASE>/wft/efficiency/ray_hit_miss_list.csv

one row per reference ray with its reference position, whether the chamber
fired, and the distance to the reconstructed point. Regenerate with:

    .venv/bin/python mx_june_wft/report/make_june_figs.py g_det3_wknd

COLOUR. This is the one figure in the deck that does NOT use
plotstyle.efficiency_cmap(): the house ramp is a bad->good ramp ending in the
deck's green, and Dylan asked for yellow-for-efficient, i.e. viridis, which is
what the June report's own sliding maps use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as PS                                # noqa: E402

FIG = os.path.join(HERE, 'figures')
SLIDES = os.path.join(HERE, 'slides', 'assets', 'img')

# The highest-statistics det3 reconstruction on disk (see the docstring).
RUN_KEY = 'g_det3_wknd'
BASE = os.path.expanduser(
    '~/x17/cosmic_bench/Analysis/mx17_det3_p2_det1_overnight_6-27-26/'
    'long_run_p2_det1_sanity_check/mx17_3/wft/efficiency')
CSV = os.path.join(BASE, 'ray_hit_miss_list.csv')
META = os.path.join(BASE, 'efficiency_map_sliding.json')

EDGE_HITS = 10           # target rays at the box edge -- derives the kernel
                        # radius from the actual ray density (see docstring);
                        # this is the same target the July PDF's r=4.1mm map
                        # used, just re-solved for today's lower density
STEP_MM = 0.5             # step; far finer than the kernel on purpose
MIN_FILL = 0.0            # off by default -- see the 2026-08-26 note in git
                        # history for why the "circle must be mostly
                        # on-chamber" guard was dropped in favour of the
                        # box-centre mask below
R_MATCH = 5.0             # the match radius that defines "reconstructed"


def load():
    """The per-ray table and the active box the analysis cut it on."""
    import pandas as pd
    if not os.path.exists(CSV):
        sys.exit(f'missing {CSV}\n  regenerate: .venv/bin/python '
                 f'mx_june_wft/report/make_june_figs.py {RUN_KEY}')
    d = pd.read_csv(CSV)
    with open(META) as f:
        meta = json.load(f)
    return d, meta


def auto_kernel(x, y, box, edge_hits):
    """The July-PDF derivation: the ONE kernel radius such that a pixel at
    the box edge (roughly half its circle on-chamber) collects ~edge_hits
    rays, given the mean ray density over the box.
    """
    area = (box['x1'] - box['x0']) * (box['y1'] - box['y0'])
    dens = len(x) / area
    kernel = float(np.sqrt(2 * edge_hits / (np.pi * dens)))
    min_rays = max(5, edge_hits // 2)
    return kernel, min_rays, dens


def sliding(x, y, within, box, kernel, step=STEP_MM, min_rays=5,
           min_fill=MIN_FILL):
    """Efficiency under a hard circular kernel swept across the face.

    Identical in definition to make_june_figs.sliding_map -- every ray whose
    reference position falls inside the circle counts once in the denominator
    and, if it was reconstructed within R_MATCH, once in the numerator -- but
    evaluated as a convolution so the step can be fine.

    Returns (eff, cnt, extent) with eff NaN wherever the circle holds fewer
    than ``min_rays``, i.e. wherever the map would be reporting noise, OR
    whose centre falls outside the active box (a circle centred just outside
    the chamber can still clear min_rays from a sliver of real chamber, on n
    small enough that the ratio is close to pure noise -- unmasked, that
    noise sits on a ring of solid colour outside the detector).
    """
    from scipy.signal import fftconvolve
    pad = kernel
    x0, x1 = box['x0'] - pad, box['x1'] + pad
    y0, y1 = box['y0'] - pad, box['y1'] + pad
    nx = int(round((x1 - x0) / step)) + 1
    ny = int(round((y1 - y0) / step)) + 1
    edges_x = x0 - step / 2 + step * np.arange(nx + 1)
    edges_y = y0 - step / 2 + step * np.arange(ny + 1)
    cx = x0 + step * np.arange(nx)
    cy = y0 + step * np.arange(ny)

    den, _, _ = np.histogram2d(x, y, bins=(edges_x, edges_y))
    num, _, _ = np.histogram2d(x, y, bins=(edges_x, edges_y), weights=within)

    k = int(np.ceil(kernel / step))
    gx, gy = np.meshgrid(np.arange(-k, k + 1) * step,
                         np.arange(-k, k + 1) * step, indexing='ij')
    disc = ((gx ** 2 + gy ** 2) <= kernel ** 2).astype(float)

    cnt = np.round(fftconvolve(den, disc, mode='same'))
    hit = np.round(fftconvolve(num, disc, mode='same'))

    box_mask = ((cx[:, None] >= box['x0']) & (cx[:, None] <= box['x1']) &
                (cy[None, :] >= box['y0']) & (cy[None, :] <= box['y1']))
    dens = len(x) / ((box['x1'] - box['x0']) * (box['y1'] - box['y0']))
    full = dens * np.pi * kernel ** 2
    floor = max(min_rays, min_fill * full)
    eff = np.where(box_mask & (cnt >= floor), np.divide(hit, np.maximum(cnt, 1)), np.nan)
    return eff, cnt, [x0 - step / 2, x1 + step / 2, y0 - step / 2, y1 + step / 2]


def gaussian_sliding(x, y, within, box, sigma, step=STEP_MM, min_w=5, trunc=4.0):
    """Same estimator as sliding(), but the hard 0/1 disc is replaced by a
    Gaussian weight exp(-d^2/2 sigma^2) -- every ray still contributes, just
    smoothly less as it gets farther from the pixel, so a single miss no
    longer draws a sharp-edged disc of radius=kernel (see the 2026-08-26
    isolated-miss check in git history): its effect fades out over ~sigma
    instead of cutting off dead at one radius. Truncated at `trunc` sigma for
    speed; at trunc=4 the truncation error is <1e-4 of the weight.

    `sigma` is deliberately passed in as the SAME auto-derived length as the
    hard kernel's radius (rather than the much larger sigma=25mm tried
    earlier and rejected) -- same small-scale resolution, no hard edges.
    """
    from scipy.signal import fftconvolve
    pad = trunc * sigma
    x0, x1 = box['x0'] - pad, box['x1'] + pad
    y0, y1 = box['y0'] - pad, box['y1'] + pad
    nx = int(round((x1 - x0) / step)) + 1
    ny = int(round((y1 - y0) / step)) + 1
    edges_x = x0 - step / 2 + step * np.arange(nx + 1)
    edges_y = y0 - step / 2 + step * np.arange(ny + 1)
    cx = x0 + step * np.arange(nx)
    cy = y0 + step * np.arange(ny)

    den, _, _ = np.histogram2d(x, y, bins=(edges_x, edges_y))
    num, _, _ = np.histogram2d(x, y, bins=(edges_x, edges_y), weights=within)

    k = int(np.ceil(trunc * sigma / step))
    gx, gy = np.meshgrid(np.arange(-k, k + 1) * step,
                         np.arange(-k, k + 1) * step, indexing='ij')
    g = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))

    cnt = fftconvolve(den, g, mode='same')     # effective (weighted) ray count
    hit = fftconvolve(num, g, mode='same')

    box_mask = ((cx[:, None] >= box['x0']) & (cx[:, None] <= box['x1']) &
                (cy[None, :] >= box['y0']) & (cy[None, :] <= box['y1']))
    eff = np.where(box_mask & (cnt >= min_w), np.divide(hit, np.maximum(cnt, 1e-9)), np.nan)
    return eff, cnt, [x0 - step / 2, x1 + step / 2, y0 - step / 2, y1 + step / 2]


def draw(eff, cnt, extent, box, meta, kernel, out_base, slide=True, step=STEP_MM,
         label=None, vmin=85, vmax=100):
    PS.use()
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(PS.SURFACE)            # too few muons: draw nothing, not a value

    im = ax.imshow(np.ma.masked_invalid(eff * 100.0).T, origin='lower',
                   extent=extent, aspect='equal', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.add_patch(plt.Rectangle((box['x0'], box['y0']),
                               box['x1'] - box['x0'], box['y1'] - box['y0'],
                               fill=False, ec=PS.MUTED, lw=1.0, ls=(0, (4, 3)),
                               zorder=4))
    ax.set_xlabel('reference x  [mm]')
    ax.set_ylabel('reference y  [mm]')
    ax.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                      extend='min' if vmin > 0 else 'neither')
    cb.set_label(f'reconstructed within {R_MATCH:g} mm  [%]', color=PS.MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=PS.MUTED)

    if label is None:
        label = (f'sliding circular kernel r = {kernel:g} mm stepped {step:g} mm '
                 f'(≈{int(round(np.nanmedian(cnt[np.isfinite(eff)]))):d} muons per '
                 f'circle) · blank where the circle is off the chamber')
    PS.note(fig,
            f'det3 · {meta["run"]}/{meta["sub_run"]} · '
            f'{meta["n_rays"]:,} M3 reference rays · {label}'.replace(',', ' '))
    PS.save(fig, out_base + '.png')
    if slide:
        import shutil
        os.makedirs(SLIDES, exist_ok=True)
        dest = os.path.join(SLIDES, os.path.basename(out_base) + '.png')
        shutil.copyfile(out_base + '.png', dest)
        print(f'  -> {dest}')


def draw_rays(eff, cnt, extent, box, meta, kernel, out_base, step=STEP_MM,
              label=None, cbar_label='muons in kernel'):
    """The companion 'rays per kernel' panel -- same thing as the third panel
    of mx_june_wft/report/make_june_figs.py:fig_sliding, so the blob texture
    in the efficiency map can be checked directly against the muon-count
    field that drives it. Same mask as the efficiency map, so pixel-for-
    pixel the two figures show the same footprint.
    """
    PS.use()
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad(PS.SURFACE)
    cnt_m = np.where(np.isfinite(eff), cnt, np.nan)
    im = ax.imshow(np.ma.masked_invalid(cnt_m).T, origin='lower',
                   extent=extent, aspect='equal', cmap=cmap,
                   interpolation='nearest')
    ax.add_patch(plt.Rectangle((box['x0'], box['y0']),
                               box['x1'] - box['x0'], box['y1'] - box['y0'],
                               fill=False, ec=PS.MUTED, lw=1.0, ls=(0, (4, 3)),
                               zorder=4))
    ax.set_xlabel('reference x  [mm]')
    ax.set_ylabel('reference y  [mm]')
    ax.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, color=PS.MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=PS.MUTED)
    if label is None:
        label = (f'same circular kernel r = {kernel:g} mm stepped {step:g} mm as '
                 f'the efficiency map, same mask -- this is the denominator '
                 f'behind every pixel there')
    PS.note(fig,
            f'det3 · {meta["run"]}/{meta["sub_run"]} · '
            f'{meta["n_rays"]:,} M3 reference rays · {label}'.replace(',', ' '))
    PS.save(fig, out_base + '_rays.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel', type=float, default=None,
                    help='kernel RADIUS in mm, overrides --edge-hits')
    ap.add_argument('--edge-hits', type=float, default=EDGE_HITS,
                    help='derive the kernel radius so a box-edge pixel holds '
                         '~this many rays (default 10, the PDF\'s own target)')
    ap.add_argument('--step', type=float, default=STEP_MM)
    ap.add_argument('--min-fill', type=float, default=MIN_FILL,
                    help='extra floor on circle fill fraction (default 0, off)')
    ap.add_argument('--no-slide', dest='slide', action='store_false')
    ap.add_argument('--print', dest='show', action='store_true')
    ap.add_argument('--suffix', default='',
                    help='appended to the output basename, e.g. _v2, so a '
                         'variant does not overwrite efficiency_map_sliding.png')
    ap.add_argument('--gaussian', action='store_true',
                    help='replace the hard circular kernel with a Gaussian '
                         'weight at the SAME auto-derived length scale -- '
                         'smooths out the per-miss disc edges without '
                         'blurring across the larger sigma=25mm tried '
                         'earlier and rejected')
    ap.add_argument('--sigma', type=float, default=None,
                    help='gaussian mode only: override sigma directly '
                         '(default: the auto/--kernel radius, same length '
                         'as the hard-kernel run)')
    ap.add_argument('--vmin', type=float, default=85,
                    help='colour scale floor, in %% (default 85; use 0 for '
                         'the full 0-100%% scale)')
    ap.add_argument('--min-rays', type=int, default=None,
                    help='override the min-rays/min-effective-weight floor '
                         '(default: max(5, edge_hits//2))')
    args = ap.parse_args()

    d, meta = load()
    box = meta['active_box']
    x = d['x'].to_numpy(float)
    y = d['y'].to_numpy(float)
    within = d['within'].to_numpy(bool).astype(float)

    if args.kernel is not None:
        kernel, min_rays = args.kernel, max(5, int(args.edge_hits) // 2)
    else:
        kernel, min_rays, dens = auto_kernel(x, y, box, args.edge_hits)
        print(f'  auto-kernel for ~{args.edge_hits:g} rays at the edge: '
              f'density={dens:.3f}/mm^2 -> kernel={kernel:.2f} mm, '
              f'min_rays={min_rays:g}')

    if args.min_rays is not None:
        min_rays = args.min_rays

    if args.gaussian:
        sigma = args.sigma if args.sigma is not None else kernel
        eff, cnt, extent = gaussian_sliding(x, y, within, box, sigma=sigma,
                                            step=args.step, min_w=min_rays)
        kernel = sigma  # for the caption/labels below
    else:
        eff, cnt, extent = sliding(x, y, within, box, kernel=kernel, step=args.step,
                                   min_rays=min_rays, min_fill=args.min_fill)
    live = np.isfinite(eff)
    if args.show or True:
        print(f'{RUN_KEY}: {len(d):,} rays, integrated within {R_MATCH:g} mm = '
              f'{100 * within.mean():.2f} %'.replace(',', ' '))
        kind = f'gaussian sigma = {kernel:g}' if args.gaussian else f'kernel r = {kernel:g}'
        print(f'  {kind} mm, step {args.step:g} mm, '
              f'{live.sum():,} live grid points'.replace(',', ' '))
        print(f'  {"effective" if args.gaussian else ""} muons per '
              f'{"footprint" if args.gaussian else "circle"}: median '
              f'{np.median(cnt[live]):.1f}, min {cnt[live].min():.1f}')
        q = np.percentile(eff[live] * 100, [5, 50, 95])
        print(f'  efficiency across the face: p5 {q[0]:.1f} %, '
              f'median {q[1]:.1f} %, p95 {q[2]:.1f} %')
    if args.show:
        return
    os.makedirs(FIG, exist_ok=True)
    out_base = os.path.join(FIG, 'efficiency_map_sliding' + args.suffix)
    if args.gaussian:
        eff_label = (f'gaussian-weighted kernel σ = {kernel:g} mm (same length as the '
                    f'auto-derived hard-kernel radius) stepped {args.step:g} mm, '
                    f'truncated at 4σ -- no hard disc edge')
        rays_label = (f'same gaussian kernel σ = {kernel:g} mm as the efficiency map, '
                     f'same mask -- effective (weighted) ray count behind every pixel')
        draw(eff, cnt, extent, box, meta, kernel, out_base, slide=args.slide,
             step=args.step, label=eff_label, vmin=args.vmin)
        draw_rays(eff, cnt, extent, box, meta, kernel, out_base, step=args.step,
                  label=rays_label, cbar_label='effective muons (gaussian-weighted)')
    else:
        draw(eff, cnt, extent, box, meta, kernel, out_base, slide=args.slide,
             step=args.step, vmin=args.vmin)
        draw_rays(eff, cnt, extent, box, meta, kernel, out_base, step=args.step)


if __name__ == '__main__':
    main()
