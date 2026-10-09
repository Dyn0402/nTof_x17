#!/usr/bin/env python3
"""
make_bench_gainmaps.py -- relative-gain map across the face of each cosmic-bench
chamber (A..E), same layout/orientation/kernel machinery as make_bench_effmaps.py.

OBSERVABLE (per matched muon, no detector position involved -- the position is the
M3 pointing): the waveform-fit charge  Q = sqrt(x_q_sum * y_q_sum)  from
<OUT_BASE>/wft/events.parquet, divided by the track path length through the drift
gap  sqrt(1 + tan_x^2 + tan_y^2)  (|tan| clipped to 1).  Why not the peak-strip
amplitude the HV scans use: at the 490-495 V operating points ~40 % of det3 events
have a saturated peak strip, so it clips exactly where the gain is highest; the
wft fit leaves saturated samples out.  Selection: M3 muon inside the active box,
reconstructed within 5 mm, not a spark -- so the map exists only where the chamber
detects the muon (read it next to the efficiency map; where efficiency collapses
the gain is a biased-up survivor estimate).

MAP: Gaussian-weighted MEAN OF ln Q (geometric mean; the Landau tail would wreck an
arithmetic mean) under the kernel, exponentiated and divided by the chamber's own
median => 1.0 = chamber median.  Q is in fitted-amplitude units with no ADC->electron
conversion, so chambers are NOT comparable with each other, only within a page.

    .venv/bin/python mx_june_wft/report/make_bench_gainmaps.py [--sigma 8]
Output: ~/x17/cosmic_bench/Analysis/bench_efficiency_maps/gain/
"""
import argparse
import json
import os
import sys

import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import matplotlib
import make_bench_effmaps as BE                   # noqa: E402
ME, PS, get_config, plt, pd = BE.ME, BE.PS, BE.get_config, BE.plt, BE.pd

OUT = os.path.join(BE.OUT, 'gain')
MIN_EFF_MUONS = 3.0


def muon_charge(key):
    """Matched, non-spark muons with a path-normalised charge."""
    d, box_ref = BE.ray_table(key)
    cfg = get_config(key)
    ev = pd.read_parquet(os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet'))
    m = d.merge(ev, on='event_id', how='inner', suffixes=('', '_w'))
    m = m[m['within'] & ~m['spark'].astype(bool) & m['x_ok'] & m['y_ok']]
    qx, qy = m['x_q_sum'].to_numpy(float), m['y_q_sum'].to_numpy(float)
    ok = np.isfinite(qx) & np.isfinite(qy) & (qx > 0) & (qy > 0) & (qx < 1e7) & (qy < 1e7)
    m = m[ok]
    tx = np.clip(m['x_tan_theta'].fillna(0).to_numpy(float), -1, 1)
    ty = np.clip(m['y_tan_theta'].fillna(0).to_numpy(float), -1, 1)
    q = np.sqrt(m['x_q_sum'].to_numpy(float) * m['y_q_sum'].to_numpy(float)) / np.sqrt(1 + tx**2 + ty**2)
    # independent cross-check: summed hit integral per view (hits are QA-grade for charge)
    import uproot
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'integral'], library='pd')
    qi = np.sqrt(raw[raw.feu == cfg.MX17_FEU_X].groupby('eventId').integral.sum()
                 .reindex(m['event_id']).to_numpy() *
                 raw[raw.feu == cfg.MX17_FEU_Y].groupby('eventId').integral.sum()
                 .reindex(m['event_id']).to_numpy())
    good = np.isfinite(qi) & (qi > 0)
    m.attrs['r_hits'] = float(np.corrcoef(np.log(q[good]), np.log(qi[good]))[0, 1])
    return m, q, box_ref, len(d)


def make_gain(key, sigma, step=0.5):
    from matplotlib.path import Path
    m, q, box_ref, n_rays = muon_charge(key)
    (x, y), poly, theta = BE.to_detector_frame(key, m['x'].to_numpy(float),
                                               m['y'].to_numpy(float), box_ref)
    box = dict(x0=poly[:, 0].min(), x1=poly[:, 0].max(), y0=poly[:, 1].min(),
               y1=poly[:, 1].max(), poly=poly, theta=theta)
    lq = np.log(q)
    med = float(np.median(q))
    lmap, cnt, extent = ME.gaussian_sliding(x, y, lq - np.log(med), box, sigma=sigma,
                                            step=step, min_w=MIN_EFF_MUONS)
    g = np.exp(lmap)
    gx = extent[0] + (np.arange(g.shape[0]) + 0.5) * step
    gy = extent[2] + (np.arange(g.shape[1]) + 0.5) * step
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    inside = Path(poly).contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(g.shape)
    g = np.where(inside, g, np.nan)
    live = np.isfinite(g)
    st = dict(key=key, n_muons=int(len(q)), n_rays=int(n_rays), median_q=med,
              p5=float(np.percentile(g[live], 5)), p50=float(np.percentile(g[live], 50)),
              p95=float(np.percentile(g[live], 95)),
              eff_muons_median=float(np.median(cnt[live])), theta_deg=theta,
              per_muon_sigma_lnq=float(np.std(lq)), r_hits=m.attrs.get('r_hits'))
    return g, extent, box, st


def draw(key, letter, det, sigma, pdf):
    from matplotlib.patches import Polygon
    g, extent, box, st = make_gain(key, sigma)
    cfg = get_config(key)
    PS.use()
    matplotlib.rcParams['savefig.bbox'] = 'standard'
    fig, ax, cax = BE.layout()
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(PS.SURFACE)
    im = ax.imshow(np.ma.masked_invalid(g).T, origin='lower', extent=extent, aspect='equal',
                   cmap=cmap, norm=LogNorm(0.3, 3.0), interpolation='nearest')
    ax.add_patch(Polygon(box['poly'], closed=True, fill=False, ec=PS.MUTED, lw=1.0,
                         ls=(0, (4, 3)), zorder=4))
    ax.set_xlim(*BE.AX_LIM); ax.set_ylim(*BE.AX_LIM)
    ax.set_xlabel('detector x  [mm]'); ax.set_ylabel('detector y  [mm]'); ax.grid(False)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.ax.minorticks_off()
    cb.set_ticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3]); cb.set_ticklabels(['0.3', '0.5', '0.7', '1', '1.5', '2', '3'])
    cb.set_label('relative gain  (1 = chamber median)', color=PS.MUTED)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=PS.MUTED)
    ax.set_title(f'Detector {letter} ({det}): relative gain across the face\n',
                 fontsize=15, color=PS.INK)
    ax.text(0.5, 1.012, f'path-normalised waveform charge per muon, Gaussian σ = {sigma:g} mm',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=11.5, color=PS.MUTED)
    th = np.radians(box['theta'])
    ins = fig.add_axes([0.80, 0.215, 0.17, 0.10])
    ins.set_xlim(-1.5, 1.5); ins.set_ylim(-1.5, 1.5); ins.set_aspect('equal'); ins.axis('off')
    for name, vec in (('ref. x', (np.cos(th), -np.sin(th))), ('ref. y', (np.sin(th), np.cos(th)))):
        ins.annotate('', xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=1.3, color=PS.MUTED))
        ins.text(vec[0] * 1.32, vec[1] * 1.32, name, fontsize=8.5, color=PS.MUTED, ha='center', va='center')
    ins.text(0, -1.55, 'M3 reference axes', fontsize=7.5, color=PS.MUTED, ha='center', va='top')
    BE.footer(fig, [
        'How it is calculated. Each point is a cosmic muon located by the M3 reference telescope and detected '
        'by this chamber (reconstructed within 5 mm of the reference track, not a spark). Its charge is the '
        'waveform-fit charge, X and Y combined as a geometric mean and divided by the track path length through '
        'the drift gap. The map is the Gaussian-weighted mean of ln Q under a '
        f'σ = {sigma:g} mm kernel, divided by this chamber\'s median, so 1 = the chamber median (log colour scale).',
        f'Reading it. Relative within this chamber only (median fitted charge {st["median_q"]:.0f}, arbitrary '
        f'units); chambers cannot be compared with each other. Measured on detected muons only, so where the '
        f'efficiency collapses the value is a survivor estimate. Per-muon ln Q agrees with the summed hit '
        f'integral at r = {st["r_hits"]:.2f}. Map p5 / median / p95 = {st["p5"]:.2f} / {st["p50"]:.2f} / '
        f'{st["p95"]:.2f}; ≈ {st["eff_muons_median"]:.0f} effective muons per kernel.',
        f'Dataset. {det} · {cfg.RUN} / {cfg.SUB_RUN} · {st["n_muons"]:,} detected muons of {st["n_rays"]:,} '
        f'reference muons.'.replace(',', ' ')])
    PS.save(fig, os.path.join(OUT, f'gain_{letter}_{det}_{key}_s{sigma:g}.png'))
    pdf.savefig(fig)
    plt.close(fig)
    st.update(letter=letter, det=det)
    return st


def main():
    from matplotlib.backends.backend_pdf import PdfPages
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, default=8.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pdf = PdfPages(os.path.join(OUT, f'cosmic_bench_gain_maps_s{a.sigma:g}mm.pdf'))
    rows = [draw(k, l, d, a.sigma, pdf) for l, d, k in BE.PAGES]
    pdf.close()
    for r in rows:
        print(r)
    json.dump(rows, open(os.path.join(OUT, 'pages.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
