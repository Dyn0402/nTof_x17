#!/usr/bin/env python3
"""Figures for the flash-timing calibration report."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = BASE / 'data'
FIGS = BASE / 'figures'
FIGS.mkdir(exist_ok=True)

INK = '#1b1b1b'
WALLC = {'WALA': '#2f6f9f', 'WALB': '#c0632c', 'WALC': '#4a8a5a', 'WALD': '#8a5a9a'}
EPOCH = {224356: '07-11', 224357: '07-11', 224358: '07-11', 224359: '07-11',
         224360: '07-11', 224464: '07-16', 224466: '07-16'}

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.titlesize': 9.5,
                     'axes.labelsize': 9, 'axes.edgecolor': '#999999',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'axes.titlelocation': 'left'})


def load_csv(p):
    import csv
    rows = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if k in ('tree',):
                    out[k] = v
                elif v == '':
                    out[k] = np.nan
                else:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return rows


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def fig_per_channel(rows):
    """32 wall channels: flash arrival vs PKUP, every run overlaid."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    walls = ['WALA', 'WALB', 'WALC', 'WALD']
    runs = sorted({int(r['run']) for r in rows})
    x, labels = [], []
    for i, w in enumerate(walls):
        for ch in range(1, 9):
            x.append(i * 9 + ch - 1)
            labels.append(f'{w[-1]}{ch}')
    grand = np.nanmean([r['dt_mean'] for r in rows if r['tree'] in walls])
    for run in runs:
        xs, ys, es = [], [], []
        for i, w in enumerate(walls):
            for ch in range(1, 9):
                m = sel(rows, tree=w, run=float(run), ch=float(ch))
                if not m:
                    continue
                xs.append(i * 9 + ch - 1); ys.append(m[0]['dt_mean']); es.append(m[0]['dt_err'])
        mk = 'o' if EPOCH.get(run) == '07-11' else 's'
        axes[0].errorbar(xs, ys, yerr=es, fmt=mk, ms=3.4, lw=0, elinewidth=0.9,
                         alpha=0.85, label=f'{run} ({EPOCH.get(run,"")})')
    for i, w in enumerate(walls):
        axes[0].axvspan(i * 9 - 0.5, i * 9 + 7.5, color=WALLC[w], alpha=0.06)
        axes[0].text(i * 9 + 3.5, axes[0].get_ylim()[1], w, ha='center', va='bottom',
                     color=WALLC[w], fontsize=9)
    axes[0].set_ylabel('flash arrival − PKUP  [ns]')
    axes[0].set_title('1. per-channel γ-flash arrival time, divert-off runs')
    axes[0].legend(ncol=4, fontsize=7.5, loc='lower right')

    # bottom: within-epoch reproducibility vs the epoch-to-epoch shift
    xs, ys, zs = [], [], []
    for i, w in enumerate(walls):
        for ch in range(1, 9):
            m = sel(rows, tree=w, ch=float(ch))
            e1 = [r['dt_mean'] for r in m if EPOCH.get(int(r['run'])) == '07-11']
            e2 = [r['dt_mean'] for r in m if EPOCH.get(int(r['run'])) == '07-16']
            if len(e1) > 1:
                xs.append(i * 9 + ch - 1)
                ys.append(np.std(e1))
                zs.append(np.mean(e2) - np.mean(e1) if e2 else np.nan)
    axes[1].bar([x - 0.19 for x in xs], ys, color='#4a8a5a', width=0.38,
                label='within 07-11 epoch (σ over 5 runs)')
    axes[1].bar([x + 0.19 for x in xs], np.abs(zs), color='#c0632c', width=0.38,
                label='|07-16 − 07-11| shift')
    axes[1].legend(fontsize=7.5, ncol=2)
    axes[1].set_ylabel('[ns]')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=6.5, rotation=90)
    axes[1].set_xlabel('wall channel')
    axes[1].set_title('2. reproducibility within an epoch, vs the real 07-11→07-16 hardware shift')
    fig.tight_layout()
    fig.savefig(FIGS / '01_per_channel_arrival.png', bbox_inches='tight')
    plt.close(fig)


def fig_jitter_and_intensity(rows, series):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    walls = ['WALA', 'WALB', 'WALC', 'WALD']

    # (a) per-bunch sigma per channel
    for w in walls:
        v = [r['dt_sigma'] for r in sel(rows, tree=w)]
        axes[0].scatter([w[-1]] * len(v), v, s=9, alpha=0.6, color=WALLC[w])
    axes[0].set_ylabel('per-bunch σ of (t − PKUP) [ns]')
    axes[0].set_title('3. single-bunch timing spread')
    axes[0].set_xlabel('wall')

    # (b) intensity dependence: dt_hi - dt_lo
    d = [r['dt_hi'] - r['dt_lo'] for r in rows if r['tree'] in walls
         and np.isfinite(r.get('dt_hi', np.nan)) and np.isfinite(r.get('dt_lo', np.nan))]
    if d:
        axes[1].hist(d, bins=np.arange(-12, 12.5, 1), color='#2f6f9f', alpha=0.85)
        axes[1].axvline(0, color=INK, lw=0.8)
        axes[1].axvline(np.mean(d), color='#c0632c', lw=1.2,
                        label=f'mean {np.mean(d):+.2f} ns')
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel('Δt(dedicated) − Δt(parasitic)  [ns]')
    axes[1].set_ylabel('channels')
    axes[1].set_title('4. time walk vs beam intensity')

    # (c) amplitude vs intensity
    lo = [r['amp_lo'] for r in rows if r['tree'] in walls and np.isfinite(r.get('amp_lo', np.nan))]
    hi = [r['amp_hi'] for r in rows if r['tree'] in walls and np.isfinite(r.get('amp_hi', np.nan))]
    if lo:
        axes[2].scatter(lo, hi, s=10, alpha=0.7, color='#4a8a5a')
        lim = [min(lo + hi) * 0.95, max(lo + hi) * 1.05]
        axes[2].plot(lim, lim, color=INK, lw=0.8, ls='--', label='equal')
        axes[2].plot(lim, [2 * v for v in lim], color='#c0632c', lw=0.8, ls=':', label='×2 (linear)')
        axes[2].set_xlim(lim); axes[2].set_ylim(lim[0], lim[1])
        axes[2].legend(fontsize=8)
    axes[2].set_xlabel('flash amp, parasitic 4.1e12 [ADC]')
    axes[2].set_ylabel('flash amp, dedicated 8.5e12')
    axes[2].set_title('5. amplitude saturates')
    fig.tight_layout()
    fig.savefig(FIGS / '02_jitter_intensity.png', bbox_inches='tight')
    plt.close(fig)


def fig_stability(series):
    """Δt vs bunch index, one panel per run (drift within run)."""
    runs = sorted({int(k.split('_')[0]) for k in series.files})
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, runs):
        allb, alld = [], []
        for w in ['WALA', 'WALB', 'WALC', 'WALD']:
            k = f'{run}_{w}'
            if f'{k}_dt' not in series.files:
                continue
            b, dt = series[f'{k}_bunch'], series[f'{k}_dt']
            good = np.isfinite(dt) & (np.abs(dt - np.nanmedian(dt)) < 100)
            allb.append(b[good]); alld.append(dt[good])
        if not allb:
            continue
        b = np.concatenate(allb); dt = np.concatenate(alld)
        nb = 40
        edges = np.linspace(b.min(), b.max(), nb + 1)
        idx = np.digitize(b, edges) - 1
        med = np.array([np.median(dt[idx == i]) if (idx == i).sum() > 20 else np.nan
                        for i in range(nb)])
        ax.plot(0.5 * (edges[:-1] + edges[1:]), med - np.nanmedian(med), lw=1.2,
                color='#2f6f9f')
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(f'{run}')
        ax.set_xlabel('bunch')
    axes[0].set_ylabel('Δt − run median [ns]')
    axes[0].set_ylim(-8, 8)
    fig.suptitle('6. stability of the flash time within each run (all walls pooled)',
                 fontsize=9.5, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(FIGS / '03_stability.png', bbox_inches='tight')
    plt.close(fig)


def fig_distributions(series):
    """Δt distribution, one channel, and channel-difference (PKUP-free)."""
    runs = sorted({int(k.split('_')[0]) for k in series.files})
    run = runs[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    k = f'{run}_WALA'
    dt, ch = series[f'{k}_dt'], series[f'{k}_ch']
    m = (ch == 1) & np.isfinite(dt)
    x = dt[m]; x = x[np.abs(x - np.median(x)) < 60]
    axes[0].hist(x, bins=60, color='#2f6f9f', alpha=0.85)
    axes[0].set_xlabel('t(WALA ch1) − t(PKUP) [ns]')
    axes[0].set_ylabel('bunches')
    axes[0].set_title(f'7. single channel vs pickup (run {run})\n'
                      f'median {np.median(x):.1f}, σ {1.4826*np.median(np.abs(x-np.median(x))):.1f} ns')

    # channel difference removes the pickup/beam jitter
    b1, d1 = series[f'{k}_bunch'][ch == 1], dt[ch == 1]
    b2, d2 = series[f'{k}_bunch'][ch == 5], dt[ch == 5]
    common = np.intersect1d(b1, b2)
    i1 = {int(b): i for i, b in enumerate(b1)}
    i2 = {int(b): i for i, b in enumerate(b2)}
    diff = np.array([d1[i1[int(b)]] - d2[i2[int(b)]] for b in common])
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff - np.median(diff)) < 60]
    axes[1].hist(diff, bins=60, color='#c0632c', alpha=0.85)
    axes[1].set_xlabel('t(WALA ch1) − t(WALA ch5) [ns]')
    axes[1].set_title(f'8. channel−channel (pickup jitter cancels)\n'
                      f'median {np.median(diff):.1f}, σ {1.4826*np.median(np.abs(diff-np.median(diff))):.1f} ns')
    fig.tight_layout()
    fig.savefig(FIGS / '04_distributions.png', bbox_inches='tight')
    plt.close(fig)


def main():
    rows = load_csv(DATA / 'per_channel_flash_timing.csv')
    series = np.load(DATA / 'per_bunch_series.npz')
    fig_per_channel(rows)
    fig_jitter_and_intensity(rows, series)
    fig_stability(series)
    fig_distributions(series)
    print('figures written to', FIGS)


if __name__ == '__main__':
    main()


def fig_transport():
    """LIQ / PSS flash constant run by run: does the time base transport?"""
    import csv
    p = DATA / 'plastic_liq_flash_by_run.csv'
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))

    def col(t):
        v = [(int(r['run']), float(r[t])) for r in rows if r.get(t) not in (None, '')]
        return zip(*v) if v else ([], [])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharex=True)
    for t, c in zip(('LIQA', 'LIQB', 'LIQC', 'LIQD'),
                    ('#2f6f9f', '#c0632c', '#4a8a5a', '#8a5a9a')):
        x, y = col(t)
        if len(x):
            axes[0].plot(x, y, 'o-', ms=3, lw=0.8, color=c, label=t)
    axes[0].set_title('9. liquid scintillators: flash time vs PKUP, run by run')
    axes[0].set_ylabel('C [ns]'); axes[0].set_xlabel('run')
    axes[0].legend(ncol=4, fontsize=8)
    for t, c in zip(('PSSA', 'PSSB', 'PSSC', 'PSSD'),
                    ('#2f6f9f', '#c0632c', '#4a8a5a', '#8a5a9a')):
        x, y = col(t)
        if len(x):
            axes[1].plot(x, y, 'o-', ms=3, lw=0.8, color=c, label=t)
    for ax in axes:
        for r, lab in ((224464, 'divert-off 07-16'),):
            ax.axvline(r, color='#999999', lw=0.8, ls='--')
    axes[1].set_title('10. plastics: same measurement — not stable')
    axes[1].set_xlabel('run'); axes[1].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / '05_transport.png', bbox_inches='tight')
    plt.close(fig)


_orig_main = main


def main():  # noqa: F811
    _orig_main()
    fig_transport()
    print('transport figure written')


if __name__ == '__main__':
    main()
