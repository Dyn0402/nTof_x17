#!/usr/bin/env python3
"""
make_bench_effmaps.py -- one sliding-kernel efficiency map per cosmic-bench
chamber (A..E), one per PDF page, in the MPGD2026 deck's format (slide 27):
viridis 0-100 %, Gaussian-weighted kernel stepped 0.5 mm, dashed active box,
"reconstructed within 5 mm of the M3 reference track".

    .venv/bin/python mx_june_wft/report/make_bench_effmaps.py [--sigma 4] [--keys ...]

Per-ray accounting is make_june_figs.per_ray_table (== 02_efficiency.py, same
M3 recipe chi2<1 & NClus=4); the kernel is mpgd26/make_efficiency_map.gaussian_sliding,
imported rather than copied so the maps cannot drift from the deck's.
Output (not in the repo): ~/x17/cosmic_bench/Analysis/bench_efficiency_maps/
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mpgd26'), HERE]

import make_june_figs as MJ                       # noqa: E402
import make_efficiency_map as ME                  # noqa: E402
import plotstyle as PS                            # noqa: E402
from qa_config import get_config                  # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                   # noqa: E402
import pandas as pd                               # noqa: E402

OUT = os.path.expanduser('~/x17/cosmic_bench/Analysis/bench_efficiency_maps')

# page order A..E; (letter, detector, primary key)
PAGES = [('A', 'det3', 'g_det3_wknd'), ('B', 'det2', 'g_det2'),
         ('C', 'det6', 'g_det6_long'), ('D', 'det7', 'g_det7_long'),
         ('E', 'det4', 'g_det4')]
OPTIONS = ['sat_det3', 'o22_long_det2']


def ray_table(key):
    """Cached per-ray table + active box for a run key."""
    os.makedirs(OUT, exist_ok=True)
    csv, js = os.path.join(OUT, f'{key}_rays.csv'), os.path.join(OUT, f'{key}_box.json')
    if os.path.exists(csv) and os.path.exists(js):
        return pd.read_csv(csv), json.load(open(js))
    d, box, _ = MJ.per_ray_table(get_config(key))
    d.to_csv(csv, index=False)
    json.dump(box, open(js, 'w'))
    return d, box


def to_detector_frame(key, x, y, box):
    """Aligned/M3-frame (x, y) -> detector-local (u, v), the exact inverse of
    alignment_transform (rotation by theta about the centre, then the offset).
    At theta ~ 90 deg, u ~ reference y and v ~ -reference x. Returns the points,
    the active-box polygon in the same frame and theta."""
    al = json.load(open(os.path.join(get_config(key).OUT_BASE, 'wft', 'alignment',
                                     'alignment.json')))
    th = np.radians(al['theta_deg'])
    c, s_ = np.cos(th), np.sin(th)
    cx, cy = al['centre_x'], al['centre_y']

    def inv(px, py):
        dx, dy = np.asarray(px) - cx - al['x_offset'], np.asarray(py) - cy - al['y_offset']
        return c * dx + s_ * dy + cx, -s_ * dx + c * dy + cy

    corners = [(box['x0'], box['y0']), (box['x1'], box['y0']),
               (box['x1'], box['y1']), (box['x0'], box['y1'])]
    pu, pv = inv([q[0] for q in corners], [q[1] for q in corners])
    return inv(x, y), np.column_stack([pu, pv]), al['theta_deg']


def make_map(key, sigma, step=0.5):
    from matplotlib.path import Path
    d, box_ref = ray_table(key)
    (x, y), poly, theta = to_detector_frame(key, d['x'].to_numpy(float),
                                            d['y'].to_numpy(float), box_ref)
    # axis-aligned bounding box for the kernel; the true (slightly rotated)
    # active polygon is applied as a mask afterwards
    box = dict(x0=poly[:, 0].min(), x1=poly[:, 0].max(),
               y0=poly[:, 1].min(), y1=poly[:, 1].max(), poly=poly, theta=theta)
    w = d['within'].to_numpy(bool).astype(float)
    eff, cnt, extent = ME.gaussian_sliding(x, y, w, box, sigma=sigma, step=step, min_w=1)
    nx_, ny_ = eff.shape
    gx = extent[0] + (np.arange(nx_) + 0.5) * step
    gy = extent[2] + (np.arange(ny_) + 0.5) * step
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    inside = Path(poly).contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(eff.shape)
    eff = np.where(inside, eff, np.nan)
    live = np.isfinite(eff)
    q = np.percentile(eff[live] * 100, [5, 50, 95])
    stats = dict(key=key, n_rays=int(len(d)), integrated=float(w.mean() * 100),
                 sigma=sigma, p5=float(q[0]), p50=float(q[1]), p95=float(q[2]),
                 eff_muons_median=float(np.median(cnt[live])),
                 eff_muons_p5=float(np.percentile(cnt[live], 5)),
                 masked_frac=float(1 - live[(cnt >= 0)].mean()))
    stats['theta_deg'] = theta
    return d, box, eff, cnt, extent, stats


AX_LIM = (-20.0, 420.0)       # identical detector-frame limits on every page
FIG_W, FIG_H = 8.6, 10.6


def layout():
    """Fixed figure, axes and colour-bar rectangles: pages differ only in content,
    so flicking through the PDF the axes never move.  Axes are square (equal
    aspect on equal limits)."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    side = 0.70 * FIG_W
    h = side / FIG_H
    ax = fig.add_axes([0.11, 0.345, 0.70, h])
    cax = fig.add_axes([0.845, 0.345, 0.024, h])
    return fig, ax, cax


def footer(fig, paras, y=0.285, width=100, size=9.0):
    """Wrapped provenance/method text in the figure, below the axes."""
    import textwrap
    lines = []
    for para in paras:
        lines += textwrap.wrap(para, width) + ['']
    fig.text(0.04, y, '\n'.join(lines), ha='left', va='top', fontsize=size,
             color=PS.MUTED, linespacing=1.35)


def draw(fig, ax, cax, eff, extent, box, title, sub):
    from matplotlib.patches import Polygon, FancyArrowPatch
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(PS.SURFACE)
    im = ax.imshow(np.ma.masked_invalid(eff * 100.0).T, origin='lower', extent=extent,
                   aspect='equal', cmap=cmap, vmin=0, vmax=100, interpolation='nearest')
    ax.add_patch(Polygon(box['poly'], closed=True, fill=False, ec=PS.MUTED, lw=1.0,
                         ls=(0, (4, 3)), zorder=4))
    ax.set_xlim(*AX_LIM); ax.set_ylim(*AX_LIM)
    ax.set_xlabel('detector x  [mm]')
    ax.set_ylabel('detector y  [mm]')
    ax.grid(False)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('reconstructed within 5 mm  [%]', color=PS.MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=PS.MUTED)
    ax.set_title(title + '\n', fontsize=15, color=PS.INK)
    ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=11.5, color=PS.MUTED)

    # where the M3 reference axes lie in this picture
    th = np.radians(box['theta'])
    ins = fig.add_axes([0.80, 0.215, 0.17, 0.10])
    ins.set_xlim(-1.5, 1.5); ins.set_ylim(-1.5, 1.5); ins.set_aspect('equal'); ins.axis('off')
    for name, vec in (('ref. x', (np.cos(th), -np.sin(th))), ('ref. y', (np.sin(th), np.cos(th)))):
        ins.annotate('', xy=vec, xytext=(0, 0),
                     arrowprops=dict(arrowstyle='-|>', lw=1.3, color=PS.MUTED))
        ins.text(vec[0] * 1.32, vec[1] * 1.32, name, fontsize=8.5, color=PS.MUTED,
                 ha='center', va='center')
    ins.text(0, -1.55, 'M3 reference axes', fontsize=7.5, color=PS.MUTED, ha='center', va='top')


def page(letter, det, key, sigma, out_png, pdf=None):
    cfg = get_config(key)
    d, box, eff, cnt, extent, st = make_map(key, sigma)
    PS.use()
    matplotlib.rcParams['savefig.bbox'] = 'standard'    # keep the fixed page size
    fig, ax, cax = layout()
    draw(fig, ax, cax, eff, extent, box, f'Detector {letter} ({det}): efficiency across the face',
         f'Gaussian-weighted circle, σ = {sigma:g} mm, swept 500 µm at a time')
    footer(fig, [
        'How it is calculated. Every point comes from a cosmic muon whose position is taken from the M3 '
        'reference telescope (χ² < 1 and 4 clusters per view, inside this chamber\'s active area), not from '
        'this detector. Each muon is coloured by whether the detector saw it: DETECTED (yellow) if the chamber '
        'reconstructed an X+Y point within 5 mm of the reference track position; MISSED (dark) if there is no '
        'reconstructed point, the point lies more than 5 mm away, or the event was a spark (> 50 strips fired).',
        f'The map. Each pixel is the detected fraction of the muons under a Gaussian-weighted circle '
        f'(σ = {sigma:g} mm) swept across the face in 0.5 mm steps, so it reads 100 % where every muon nearby '
        f'was detected and 0 % where none was. Pixel-to-pixel scatter is counting noise at ≈ '
        f'{st["eff_muons_median"]:.0f} effective muons per kernel (median).',
        f'Dataset. {det} · {cfg.RUN} / {cfg.SUB_RUN} · {st["n_rays"]:,} reference muons · integrated '
        f'efficiency {st["integrated"]:.1f} % · alignment θ = {st["theta_deg"]:.2f}°. Axes are the detector\'s '
        f'own x/y; the inset shows the M3 reference axes. Position from the waveform-first reconstruction '
        f'(hits are used only to decide whether the chamber fired at all).'.replace(',', ' ')])
    PS.save(fig, out_png)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    st.update(letter=letter, det=det, run=cfg.RUN, sub_run=cfg.SUB_RUN)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, default=4.0)
    ap.add_argument('--only-stats', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages
    rows = []
    pdf = PdfPages(os.path.join(OUT, f'cosmic_bench_efficiency_maps_s{a.sigma:g}mm.pdf'))
    for letter, det, key in PAGES:
        png = os.path.join(OUT, f'map_{letter}_{det}_{key}_s{a.sigma:g}.png')
        rows.append(page(letter, det, key, a.sigma, png, pdf))
        print(rows[-1])
    pdf.close()
    for key in OPTIONS:
        _, _, _, _, _, st = make_map(key, a.sigma)
        print('option', st)
    json.dump(rows, open(os.path.join(OUT, f'pages_s{a.sigma:g}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
