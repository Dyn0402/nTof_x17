#!/usr/bin/env python3
"""Figures for the plastic-scintillator after-pulse study.

Reads the artifacts the analysis scripts leave behind and writes figures/*.png.

    python make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / 'figures'
PLASTIC, WALL, LIQUID = '#0072B2', '#D55E00', '#009E73'
STYLES = ['-', '--', '-.', ':']
SURFACE, INK, MUTED = '#fcfcfb', '#20242b', '#6b7280'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.edgecolor': '#c9ccd1',
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#e6e8ea',
    'grid.linewidth': 0.7, 'axes.axisbelow': True, 'font.size': 9,
    'axes.titlesize': 10, 'legend.frameon': False, 'lines.linewidth': 1.6,
})


def density(d, edges):
    """Excess followers per leader per ns, and the mixed control."""
    w = np.diff(edges)
    obs = np.array(d['counts']) / d['n_leaders'] / w
    mix = np.array(d['mixed']) * d['mix_scale'] / d['n_leaders'] / w
    return obs, mix


def rebin(res, det, edges, target):
    """Counts per leader per ns on a coarser set of edges."""
    d = res['dets'][det]
    ctr = 0.5 * (edges[:-1] + edges[1:])
    obs = np.array(d['counts'], float)
    mix = np.array(d['mixed'], float) * d['mix_scale']
    idx = np.digitize(ctr, target) - 1
    ok = (idx >= 0) & (idx < target.size - 1)
    o = np.bincount(idx[ok], obs[ok], target.size - 1)
    m = np.bincount(idx[ok], mix[ok], target.size - 1)
    w = np.diff(target) * d['n_leaders']
    return o / w, m / w


def fig_deltat(res):
    edges = np.array(res['edges'])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    target = np.geomspace(14, 2e4, 55)
    tctr = np.sqrt(target[:-1] * target[1:])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    def draw(ax, det, style, col, lab, lw=1.4):
        o, m = rebin(res, det, edges, target)
        y = o - m
        ax.plot(np.where(y > 0, tctr, np.nan), np.where(y > 0, y, np.nan),
                style, color=col, lw=lw, label=lab)

    for i, det in enumerate(['PSSA', 'PSSB', 'PSSC', 'PSSD']):
        draw(ax1, det, STYLES[i], PLASTIC, f'plastic {det[-1]}')
    for det, col, lab in [('WALA', WALL, 'SiPM wall A'),
                          ('LIQA', LIQUID, 'liquid A')]:
        if det in res['dets']:
            draw(ax1, det, '-', col, lab, lw=1.8)
    _o, m = rebin(res, 'PSSB', edges, target)
    ax1.plot(tctr, m, '-', color=MUTED, lw=1.0,
             label='accidental level (event-mixed, PSSB)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(14, 2e4)
    ax1.set_ylim(2e-6, 4e-1)
    ax1.set_xlabel('delay after the leading pulse  [ns]')
    ax1.set_ylabel('excess follower hits per leader per ns')
    ax1.set_title('The plastics carry a correlated hit tail; the walls do not',
                  loc='left')
    ax1.legend(fontsize=7.5, loc='lower left', ncol=1)

    for i, det in enumerate(['PSSA', 'PSSB', 'PSSC', 'PSSD']):
        obs, mix = density(res['dets'][det], edges)
        m = ctr < 160
        ax2.plot(ctr[m], (obs - mix)[m], STYLES[i], color=PLASTIC, lw=1.4,
                 label=f'plastic {det[-1]}')
    obs, mix = density(res['dets']['WALA'], edges)
    m = ctr < 160
    ax2.plot(ctr[m], (obs - mix)[m], '-', color=WALL, lw=1.8, label='SiPM wall A')
    ax2.set_xlim(0, 160)
    ax2.set_xlabel('delay after the leading pulse  [ns]')
    ax2.set_ylabel('excess follower hits per leader per ns')
    ax2.set_title('A 2 ns-wide echo at 81 ns, on all four plastics', loc='left')
    ax2.annotate('81-82 ns\ncable echo', xy=(82, 0.125), xytext=(104, 0.105),
                 fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.0))
    ax2.annotate('PSA two-pulse\nresolution ~18 ns', xy=(19, 0.004),
                 xytext=(26, 0.052), fontsize=8, color=MUTED,
                 arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.0))
    ax2.legend(fontsize=7.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(FIG / 'deltat_spectrum.png', dpi=150)
    plt.close(fig)


def fig_same_block(sb):
    dets = ['PSSB', 'WALA']
    fig, axes = plt.subplots(1, len(dets), figsize=(10.4, 4.0), sharey=True)
    for ax, det in zip(np.atleast_1d(axes), dets):
        rows = sb['dets'][det]
        x = np.arange(len(rows))
        n = np.array([r['n_leaders'] for r in rows], float)
        same = np.array([r['same'] for r in rows]) / n
        new = np.array([r['new'] for r in rows]) / n
        # grouped, not stacked: on a log axis a stacked segment's height is not
        # its value, so the two mechanisms have to sit side by side
        ax.bar(x - 0.19, np.where(same > 0, same, np.nan), 0.34, color=PLASTIC,
               label="inside the leader's own block")
        ax.bar(x + 0.19, np.where(new > 0, new, np.nan), 0.34, color=WALL,
               label='a NEW threshold crossing')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['lo']}-{r['hi']}" for r in rows], rotation=60,
                           ha='right', fontsize=7.5)
        ax.set_yscale('log')
        ax.set_ylim(2e-4, 30)
        ax.set_xlabel('delay band  [ns]')
        ax.set_title(f'{det}', loc='left')
    np.atleast_1d(axes)[0].set_ylabel('follower hits per leader')
    np.atleast_1d(axes)[0].legend(fontsize=8, loc='upper left')
    fig.suptitle("The whole correlated tail sits inside the primary's own "
                 'recorded block, where the raw samples can be checked; beyond '
                 '~1 us\nthe followers are separate records at the accidental '
                 'level', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / 'same_block.png', dpi=150)
    plt.close(fig)


def fig_tail(stack):
    pre = int(stack['pre'])
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    def draw(det, style, col, lab, lw):
        if f'{det}_median' not in stack:
            return
        y, n = stack[f'{det}_median'], stack[f'{det}_n']
        # the zero-suppressed block ends at its own moment on every pulse, so
        # stop each curve where fewer than 60 % of the traces still exist
        y = np.where(n >= 0.6 * n.max(), y, np.nan)
        ax.plot(np.arange(y.size) - pre, y, style, color=col, lw=lw, label=lab)

    for i, det in enumerate(['PSSA', 'PSSB', 'PSSC', 'PSSD']):
        draw(det, STYLES[i], PLASTIC, f'plastic {det[-1]}', 1.4)
    draw('WALA', '-', WALL, 'SiPM wall A', 1.8)
    draw('LIQA', '-', LIQUID, 'liquid A', 1.8)
    ax.set_yscale('log')
    ax.set_xlim(-20, 400)
    ax.set_ylim(3e-4, 1.6)
    ax.set_xlabel('time from the pulse peak  [ns]')
    ax.set_ylabel('median trace, normalised to the peak')
    ax.set_title('The MEDIAN tail is smooth and monotonic: the after-pulses are '
                 'sporadic,\nnot a fixed ring riding on every pulse',
                 loc='left')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'pulse_tails.png', dpi=150)
    plt.close(fig)


def fig_echo(cond):
    pre = int(cond['pre'])
    a, b = cond['A_mean'], cond['B_mean']
    t = np.arange(a.size) - pre
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True,
                                   height_ratios=[2, 1])
    ax1.plot(t, a, '-', color=PLASTIC, lw=1.8,
             label=f"has a PSA hit at 79-85 ns  (n={int(cond['A_n'])})")
    ax1.plot(t, b, '-', color=MUTED, lw=1.6,
             label=f"has none within 120 ns  (n={int(cond['B_n'])})")
    ax1.set_yscale('log')
    ax1.set_ylim(2e-3, 0.1)
    ax1.set_ylabel('mean trace / peak')
    ax1.set_title('The 81 ns hit sits on a real bump in the raw trace  (PSSB)',
                  loc='left')
    ax1.legend(fontsize=8)
    ax2.plot(t, a - b, '-', color=WALL, lw=1.8)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_xlim(40, 160)
    ax2.set_ylim(-0.002, 0.017)
    ax2.set_xlabel('time from the pulse peak  [ns]')
    ax2.set_ylabel('difference')
    ax2.annotate('+1.5 % of the peak, at 82 ns', xy=(82, 0.0145),
                 xytext=(96, 0.0125), fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.0))
    fig.tight_layout()
    fig.savefig(FIG / 'echo_conditional.png', dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    res = json.loads((HERE / 'afterpulse.json').read_text())
    fig_deltat(res)
    fig_same_block(json.loads((HERE / 'same_block.json').read_text()))
    fig_tail(np.load(HERE / 'stack_head8.npz'))
    fig_echo(np.load(HERE / 'echo_cond_PSSB.npz'))
    print(f'wrote figures to {FIG}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
