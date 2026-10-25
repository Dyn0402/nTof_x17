#!/usr/bin/env python3
"""make_figures.py [--examples] -- figures for report.html into ~/x17/cosmic_bench/outliers/report/figures/.

  classes.png        outlier classes per chamber, % of all M3-matched events (stacked)
  farmap.png         far fraction (|dθ| > 5°) vs the reference position, every chamber and view
  edge_response.png  angle response k and far rate vs distance to the edge (edge_response.py)
  bent.png           det3: MX17 - M3 residual against the M3 doublet kink, and straight fit vs doublet vertex
  sps.png            head-on angle residual: det4 at H4 (run_71) vs det4 and det3 on the bench, same fitter
  ex_<det>_<cls>.png example displays (--examples; montage.py)
"""
import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE]
from olib import BASE, DETS, HEADON, FAR, load  # noqa: E402

OUT = os.path.join(BASE, 'report', 'figures')
COL = {'REJECTED_OK': '#bdbdbd', 'OUTSIDE': '#9ecae1', 'EDGE': '#4292c6', 'HW_KNOWN': '#08306b',
       'M3_ELSEWHERE': '#fdae6b', 'POS_TAIL_BENT': '#e6550d', 'MULTI': '#74c476', 'DELTA': '#238b45',
       'MX17_DEAD': '#dadaeb', 'ONE_VIEW_ELSEWHERE': '#bcbddc', 'POS_TAIL_MX17': '#9e9ac8', 'FIT_SAT': '#756bb1',
       'FIT': '#54278f'}


def classes(R, plt):
    fig, ax = plt.subplots(figsize=(11, 5))
    keys = list(COL); x = np.arange(len(DETS)); bot = np.zeros(len(DETS))
    for k in keys:
        h = np.array([R['classes'][d]['cls'].get(k, {'pct': 0})['pct'] for d in DETS])
        ax.bar(x, h, bottom=bot, color=COL[k], label=k, width=0.6); bot += h
    ax.set_xticks(x); ax.set_xticklabels([f"{d}\n({R['classes'][d]['n_events']} ev)" for d in DETS])
    ax.set_ylabel('% of M3-matched events'); ax.set_title('Events that disagree with M3, by cause')
    ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1), loc='upper left'); ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'classes.png'), dpi=90); plt.close(fig)


def farmap(plt):
    fig, axs = plt.subplots(2, 5, figsize=(22, 8.5))
    for j, det in enumerate(DETS):
        d = load(det)
        for i, v in enumerate('xy'):
            u = d[d[f'use_{v}']]; far = (u[f'dl_{v}'].abs() > FAR).to_numpy()
            X, Y = u.ref_mesh_x.to_numpy(), u.ref_mesh_y.to_numpy(); b = np.linspace(0, 400, 21)
            n, _, _ = np.histogram2d(X, Y, [b, b]); f, _, _ = np.histogram2d(X[far], Y[far], [b, b])
            with np.errstate(invalid='ignore', divide='ignore'):
                r = np.where(n >= 8, f / n, np.nan)
            ax = axs[i, j]; im = ax.imshow(r.T, origin='lower', extent=[0, 400, 0, 400], vmin=0, vmax=0.5, cmap='magma')
            ax.set_title(f'{det} {v} view: far share'); ax.set_xlabel('reference x at mesh [mm]'); ax.set_ylabel('y [mm]')
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'farmap.png'), dpi=70); plt.close(fig)


def bent(plt):
    V = pd.read_parquet(os.path.join(BASE, 'kinkvtx_det3.parquet'))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for i, v in enumerate('xy'):
        g = V[(V.view == v) & (V.cls == 'GOOD') & (V.dl.abs() < 3)]; p = V[(V.view == v) & (V.cls == 'POS_TAIL') & (V.r_full.abs() > 3)]
        axs[i].plot(g.kink, g.r_full, '.', ms=1.5, color='0.6', label='clean events')
        axs[i].plot(p.kink, p.r_full, 'o', ms=4, color='#e6550d', label=f'POS_TAIL (corr {np.corrcoef(p.kink, p.r_full)[0, 1]:+.2f})')
        axs[i].set_xlabel('M3 doublet kink [mrad]'); axs[i].set_ylabel('MX17 − M3 straight fit [mm]')
        axs[i].set_xlim(-25, 25); axs[i].set_ylim(-8, 8); axs[i].grid(alpha=.3); axs[i].legend(fontsize=8)
        axs[i].set_title(f'det3 {v}: residual follows the kink')
    p = V[(V.cls == 'POS_TAIL') & (V.r_full.abs() > 3)]; g = V[(V.cls == 'GOOD') & (V.dl.abs() < 3)]
    b = np.linspace(0, 8, 33)
    axs[2].hist(np.abs(p.r_full), b, histtype='step', lw=2, label='POS_TAIL: vs straight 4-layer fit')
    axs[2].hist(np.abs(p.r_vtx), b, histtype='step', lw=2, label='POS_TAIL: vs doublet vertex')
    axs[2].hist(np.abs(g.r_vtx), b, histtype='step', lw=1, ls='--', density=False, weights=np.full(len(g), len(p) / len(g)),
                label='clean: vs doublet vertex (its own spread, scaled)')
    axs[2].set_xlabel('|MX17 − reference| [mm]'); axs[2].legend(fontsize=8); axs[2].set_title('det3: which M3 prediction MX17 agrees with')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'bent.png'), dpi=80); plt.close(fig)


def sps(plt):
    S = os.path.join(BASE, 'sps')
    sets = [('sps700_dp015', 'det4 H4 run_71 (700 V, Dp 0.015)', 14.0), ('sps700_dp025', 'det4 H4 run_71 (Dp 0.025)', 14.0),
            ('bench_det4_headon', 'det4 bench, head-on', 33.0), ('bench_det3_headon', 'det3 bench, head-on', 35.8)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for f, lab, vl in sets:
        p = os.path.join(S, f + '.json')
        if not os.path.exists(p):
            continue
        Z = json.load(open(p))
        for i, v in enumerate('xy'):
            R = [e[v] for e in Z['events'] if v in e and abs(e[v]['tr']) < 0.1]
            th = np.degrees(np.arctan(np.array([e['w'] for e in R]) / vl) - np.arctan(np.array([e['tr'] for e in R])))
            axs[i].hist(np.clip(th, -20, 20), np.linspace(-20, 20, 81), histtype='step', density=True, lw=1.5,
                        label=f"{lab}: far {Z['summary'][v]['far_pct']:.1f} %")
    for i, v in enumerate('xy'):
        axs[i].set_yscale('log'); axs[i].axvline(-5, color='k', lw=.5); axs[i].axvline(5, color='k', lw=.5)
        axs[i].set_xlabel('reconstructed − true angle [deg] (head-on)'); axs[i].legend(fontsize=7); axs[i].set_title(f'{v} view')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'sps.png'), dpi=80); plt.close(fig)


def edgechar(plt):
    F = json.load(open(os.path.join(BASE, 'report', 'fiducial.json')))
    bands = ['edge 0-5', 'edge 5-10', 'edge 10-15', 'edge 15-20', 'edge 20-30', 'interior']
    xl = ['0-5', '5-10', '10-15', '15-20', '20-30', '≥30']
    fig, axs = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for j, v in enumerate('xy'):
        for det in DETS:
            g = lambda b, k: F.get(f'{det}|{v}|accepted|{b}', {}).get(k, np.nan)
            for i, (k, yl) in enumerate((('far', 'far share [%]'), ('k_all', 'k = fitted tan / M3 tan'),
                                         ('tilt_in_headon', 'head-on tilt, + = inward [tan]'))):
                axs[i, j].plot(range(len(bands)), [g(b, k) for b in bands], 'o-', label=det)
                axs[i, j].set_ylabel(yl); axs[i, j].grid(alpha=.3)
        axs[0, j].set_title(f'{v} view (distance to the nearest {v} active edge)'); axs[0, j].set_yscale('log')
        axs[1, j].set_ylim(0, 1.8); axs[1, j].axhline(1, color='k', lw=.6); axs[2, j].axhline(0, color='k', lw=.6)
        axs[2, j].set_xticks(range(len(bands))); axs[2, j].set_xticklabels(xl); axs[2, j].set_xlabel('footprint distance to edge [mm]')
        axs[0, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'edgechar.png'), dpi=80); plt.close(fig)


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(); ap.add_argument('--examples', action='store_true'); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    R = json.load(open(os.path.join(BASE, 'report', 'results.json')))
    classes(R, plt); farmap(plt); bent(plt); sps(plt); edgechar(plt)
    subprocess.run([sys.executable, os.path.join(HERE, 'edge_response.py'), '--out', os.path.join(OUT, 'edge_response.png')], check=True,
                   stdout=subprocess.DEVNULL)
    if a.examples:
        for det, cls, seed in [('det3', 'MULTI', 0), ('det3', 'DELTA', 0), ('det3', 'POS_TAIL', 3), ('det3', 'EDGE', 0),
                               ('det3', 'FIT_SAT', 0), ('det3', 'FIT', 3), ('det3', 'OUTSIDE', 0), ('det3', 'REJECTED_OK', 0),
                               ('det4', 'HW_KNOWN', 0), ('det4', 'FIT', 0), ('det7', 'HW_KNOWN', 0), ('det7', 'MULTI', 1)]:
            subprocess.run([sys.executable, os.path.join(HERE, 'montage.py'), det, cls, '--n', '4', '--seed', str(seed),
                            '--out', OUT], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            src = os.path.join(OUT, f'{det}_{cls}_{seed}.png'); os.replace(src, os.path.join(OUT, f'ex_{det}_{cls}.png'))
    print('figures in', OUT)


if __name__ == '__main__':
    main()
