#!/usr/bin/env python3
"""Figures for the after-pulse veto: what it removes, and what it costs.

    python veto_figures.py            # reads veto_scan.json / .npz
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / 'figures'
KEEP, CUT, MUTED2 = '#0072B2', '#D55E00', '#009E73'
SURFACE, INK, MUTED = '#fcfcfb', '#20242b', '#6b7280'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.edgecolor': '#c9ccd1',
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#e6e8ea',
    'grid.linewidth': 0.7, 'axes.axisbelow': True, 'font.size': 9,
    'axes.titlesize': 10, 'legend.frameon': False, 'lines.linewidth': 1.6,
})


T_REC, R_REC = 1000.0, 0.05


def fig_dt(d):
    dt, ctrl = d['dt'], d['ctrl'].astype(bool)
    # rebuild the flag at the recommended operating point rather than whatever
    # veto_on_dream.py happened to be invoked with
    w = int(np.argmin(np.abs(d['t_holds'] - T_REC)))
    fl = d['amp'] < R_REC * d['pmax'][w]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    e = np.arange(-1000, 1001, 5.0)
    c = 0.5 * (e[:-1] + e[1:])
    h_sig = np.histogram(dt[~ctrl], bins=e)[0]
    h_ctl = np.histogram(dt[ctrl], bins=e)[0]
    h_keep = np.histogram(dt[~ctrl & ~fl], bins=e)[0]
    h_ckeep = np.histogram(dt[ctrl & ~fl], bins=e)[0]
    ax1.step(c, np.maximum(h_sig - h_ctl, 0.3), where='mid', color=CUT,
             label='all plastic hits')
    ax1.step(c, np.maximum(h_keep - h_ckeep, 0.3), where='mid', color=KEEP,
             label='after the after-pulse veto')
    ax1.set_yscale('log')
    ax1.set_xlim(-1000, 1000)
    ax1.set_ylim(3, None)
    ax1.set_xlabel('dt to the corrected DREAM prediction  [ns]')
    ax1.set_ylabel('background-subtracted hits per 5 ns')
    ax1.set_title(f'The veto flattens the late tail and leaves the core '
                  f'(T={T_REC:.0f} ns, R={R_REC:.2f})',
                  loc='left')
    ax1.legend(fontsize=8)

    e2 = np.arange(30, 161, 1.0)
    c2 = 0.5 * (e2[:-1] + e2[1:])
    s2 = np.histogram(dt[~ctrl], bins=e2)[0] - np.histogram(dt[ctrl], bins=e2)[0]
    k2 = (np.histogram(dt[~ctrl & ~fl], bins=e2)[0]
          - np.histogram(dt[ctrl & ~fl], bins=e2)[0])
    ax2.step(c2, s2, where='mid', color=CUT, label='all plastic hits')
    ax2.step(c2, k2, where='mid', color=KEEP, label='after the veto')
    ax2.set_xlim(30, 160)
    ax2.set_xlabel('dt to the corrected DREAM prediction  [ns]')
    ax2.set_ylabel('background-subtracted hits per ns')
    ax2.set_title('The late side is not featureless: a bump near 80 ns',
                  loc='left')
    pk = c2[np.argmax(s2 * ((c2 > 65) & (c2 < 95)))]
    ax2.annotate(f'bump at ~{pk:.0f} ns\n(the 81 ns echo, smeared by\nthe '
                 f"parent's own residual)",
                 xy=(pk, s2[np.argmin(abs(c2 - pk))]),
                 xytext=(pk + 16, s2.max() * 0.72), fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.0))
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'veto_dt.png', dpi=150)
    plt.close(fig)


def fig_roc(scan):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    t_holds = sorted({r['t_hold'] for r in scan})
    marks = ['o', 's', '^', 'D', 'v']
    cmap = plt.get_cmap('cividis')
    for i, th in enumerate(t_holds):
        rows = sorted([r for r in scan if r['t_hold'] == th],
                      key=lambda r: r['ratio'])
        x = [r['control_vetoed'] * 100 for r in rows]
        y = [r['late_removed'] * 100 for r in rows]
        ax.plot(x, y, '-', marker=marks[i % len(marks)], ms=6,
                color=cmap(0.12 + 0.72 * i / max(len(t_holds) - 1, 1)),
                label=f'lookback {th:.0f} ns')
        for r, xi, yi in zip(rows, x, y):
            if r['ratio'] in (0.05, 0.20):
                ax.annotate(f"R={r['ratio']:.2f}", (xi, yi), fontsize=7,
                            color=MUTED, xytext=(4, -9),
                            textcoords='offset points')
    ax.set_xlabel('hits vetoed in the +100 µs accidental control  [%]\n(an upper bound on the loss: control hits are small singles, core hits are MIPs)')
    ax.set_ylabel('late tail removed  [% of the 150–1000 ns excess]')
    ax.set_title('What the shadow flag buys against what it costs\n'
                 'lookback, not ratio, is what moves it up', loc='left')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG / 'veto_roc.png', dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    res = json.loads((HERE / 'veto_scan.json').read_text())
    fig_roc(res['scan'])
    fig_dt(np.load(HERE / 'veto_scan.npz'))
    print(f'wrote figures to {FIG}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
