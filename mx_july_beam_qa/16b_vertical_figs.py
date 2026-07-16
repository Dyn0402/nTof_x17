"""
16b_vertical_figs.py — Figures for the top/bottom combination + vertical-position
analysis (cache from 16_wall_vertical.py).

Usage: python 16b_vertical_figs.py [run_stem] [tag]
  tag: '' for full run, or e.g. '_n150' for a smoke-test cache.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
TAG = sys.argv[2] if len(sys.argv) > 2 else ''
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '16_vertical'
OUT.mkdir(parents=True, exist_ok=True)

V_EFF = 15.0      # cm/ns assumed effective light velocity along the bar
L_BAR = 50.0      # cm bar length

d = np.load(BASE / 'cache' / f'16_vertical_{RUN_STEM}{TAG}.npz')
sc = float(d['sb_scale'])
AE = d['amp_edges']; AC = np.sqrt(AE[:-1] * AE[1:])
LRE = d['lr_edges']; LRC = 0.5 * (LRE[:-1] + LRE[1:])
DTE = d['dt_edges']; DTC = 0.5 * (DTE[:-1] + DTE[1:])
A2 = d['a2_edges']
DT2 = d['dt2_edges']; LR2 = d['lr2_edges']

GOOD_ARMS = 'BC'
CH_COLORS = plt.get_cmap('tab10').colors
KERN = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); KERN /= KERN.sum()


def sub(h2):                      # signal - scale*sideband
    return h2[0] - sc * h2[1]


def peak(spec):
    sm = np.convolve(spec, KERN, mode='same')
    m = AC > 150
    return AC[m][np.argmax(sm[m])]


# ---- (1) estimator comparison: top / bottom / arith / geo, per arm (summed groups)
def fig_estimators():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, st in zip(axes.ravel(), 'ABCD'):
        for key, lab, col in (('top', 'top only', '#60a5fa'),
                              ('bot', 'bottom only', '#a78bfa'),
                              ('arith', 'arithmetic mean', '#9ca3af'),
                              ('geo', 'geometric mean', '#c2410c')):
            s = sub(d[f'{st}_{key}']).sum(axis=0)      # sum over groups
            pk = peak(s)
            ax.step(AC, s, where='mid', color=col, lw=1.6 if key == 'geo' else 1.1,
                    label=f'{lab}  (pk {pk:.0f})')
        ax.set_xscale('log'); ax.set_xlim(80, 3e4)
        ax.set_title(f'arm {st}')
        ax.set_xlabel('amplitude estimator [ADC]')
        ax.grid(alpha=0.25, which='both')
        ax.legend(fontsize=8)
        ax.axhline(0, color='#999', lw=0.7)
    axes[0, 0].set_ylabel('true coincidences / bin')
    axes[1, 0].set_ylabel('true coincidences / bin')
    fig.suptitle(f'{RUN_STEM}: top/bottom amplitude estimators (sideband-subtracted, '
                 'summed over 4 groups)')
    fig.tight_layout()
    fig.savefig(OUT / 'estimator_comparison.png', dpi=140)
    plt.close(fig)


# ---- (2) resolution table: relative MIP peak width for each estimator
def peak_width(spec):
    """Gaussian-ish sigma/peak from the subtracted spectrum above 150 ADC using
    a half-max span; returns (peak, fwhm/peak)."""
    sm = np.convolve(np.clip(spec, 0, None), KERN, mode='same')
    m = AC > 150
    x, y = AC[m], sm[m]
    if y.max() <= 0:
        return np.nan, np.nan
    pk = x[np.argmax(y)]
    half = y.max() / 2
    above = y >= half
    if above.sum() < 2:
        return pk, np.nan
    xr = x[above]
    return pk, (xr.max() - xr.min()) / pk


def report_widths():
    print(f'{"arm":4s} {"estimator":12s} {"peak[ADC]":>10s} {"FWHM/peak":>10s}')
    for st in 'ABCD':
        for key in ('top', 'bot', 'arith', 'geo'):
            s = sub(d[f'{st}_{key}']).sum(axis=0)
            pk, w = peak_width(s)
            print(f'{st:4s} {key:12s} {pk:>10.0f} {w:>10.2f}')


# ---- (3) geo-mean spectrum split by plastic bar, good arms
def fig_bybar():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, st in zip(axes, GOOD_ARMS):
        gb = d[f'{st}_geobar']                 # (2 sig/side, 4 g, 2 bar, NA)
        for bar, col in ((0, '#2563eb'), (1, '#dc2626')):
            s = sub(gb[:, :, bar, :]).sum(axis=0)
            ax.step(AC, s, where='mid', color=col, lw=1.5,
                    label=f'plastic bar {bar + 1}  (pk {peak(s):.0f})')
        ax.set_xscale('log'); ax.set_xlim(80, 3e4)
        ax.set_title(f'arm {st}: geo-mean MIP by tagging bar')
        ax.set_xlabel('geometric-mean amplitude [ADC]')
        ax.grid(alpha=0.25, which='both'); ax.legend(fontsize=9)
        ax.axhline(0, color='#999', lw=0.7)
    axes[0].set_ylabel('true coincidences / bin')
    fig.suptitle(f'{RUN_STEM}: wall MIP spectrum split by which plastic bar tagged it')
    fig.tight_layout()
    fig.savefig(OUT / 'geo_by_bar.png', dpi=140)
    plt.close(fig)


# ---- (4) vertical position: amplitude ratio and timing, per group (arm C)
def fig_position(st='C'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for g in range(4):
        col = CH_COLORS[g]
        s_lr = sub(d[f'{st}_lr'][:, g, :])
        s_dt = sub(d[f'{st}_dt'][:, g, :])
        pos = np.clip(s_dt, 0, None)
        skew = np.average(DTC, weights=pos) if pos.sum() > 0 else 0.0   # cable skew
        axes[0].step(LRC, s_lr / max(s_lr.sum(), 1), where='mid', color=col, lw=1.3,
                     label=f'group {g + 1}')
        axes[1].step(DTC - skew, s_dt / max(s_dt.sum(), 1), where='mid', color=col, lw=1.3,
                     label=f'group {g + 1} (skew {skew:+.1f} ns)')
    axes[0].set_xlabel('ln(A_top / A_bot)   →  vertical position')
    axes[0].set_title(f'arm {st}: amplitude asymmetry')
    axes[1].set_xlabel('t_bot − t_top  [ns]   →  vertical position')
    axes[1].set_title(f'arm {st}: timing asymmetry')
    # secondary axis: cm from timing, y = v_eff/2 * dt
    ax2 = axes[1].twiny()
    ax2.set_xlim(DTC[0] * V_EFF / 2, DTC[-1] * V_EFF / 2)
    ax2.set_xlabel(f'≈ vertical offset [cm]  (v_eff={V_EFF:.0f} cm/ns)')
    for ax in axes:
        ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_ylabel('norm. true coinc.')
    fig.suptitle(f'{RUN_STEM}: vertical-position observables (sideband-subtracted, arm {st})')
    fig.tight_layout()
    fig.savefig(OUT / f'position_1d_{st}.png', dpi=140)
    plt.close(fig)


# ---- (5) the money plot: do timing and amplitude agree on vertical position?
def _principal_slope(h, ce_x, ce_y):
    """Total-least-squares (principal-axis) slope dy/dx of a 2D weighted blob."""
    w = np.clip(h, 0, None)
    W = w.sum()
    X, Y = np.meshgrid(ce_x, ce_y, indexing='ij')
    mx, my = (w * X).sum() / W, (w * Y).sum() / W
    cxx = (w * (X - mx) ** 2).sum() / W
    cyy = (w * (Y - my) ** 2).sum() / W
    cxy = (w * (X - mx) * (Y - my)).sum() / W
    # larger-eigenvalue eigenvector of [[cxx,cxy],[cxy,cyy]]
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    return np.tan(theta), (mx, my), cxy / np.sqrt(cxx * cyy)


def fig_correlation(st='C', g=2):
    h = sub(d[f'{st}_dtlr'][:, g, :, :])       # (dt2, lr2)
    ce_dt = 0.5 * (DT2[:-1] + DT2[1:]); ce_lr = 0.5 * (LR2[:-1] + LR2[1:])
    # subtract cable-skew: shift dt axis so the blob centroid sits at dt0
    slope, (mx, my), r = _principal_slope(h, ce_dt, ce_lr)
    lam = V_EFF / slope if slope > 0 else np.nan     # ln_ratio/dt = v_eff/lambda
    fig, ax = plt.subplots(figsize=(7.8, 6))
    im = ax.pcolormesh(DT2 - mx, LR2, np.clip(h, 0, None).T, cmap='viridis', rasterized=True)
    fig.colorbar(im, ax=ax, label='true coincidences')
    xs = np.array([DT2[0] - mx, DT2[-1] - mx])
    ax.plot(xs, my + slope * xs, 'w--', lw=1.6,
            label=f'principal axis  {slope:.3f} ln/ns  (r={r:.2f})\n'
                  f'$\\Rightarrow \\lambda_{{eff}}\\approx${lam:.0f} cm  '
                  f'(v_eff={V_EFF:.0f} cm/ns)')
    ax.legend(loc='upper left', fontsize=9, labelcolor='white', framealpha=0.35)
    ax.set_xlabel('t_bot − t_top − skew  [ns]  →  vertical y')
    ax.set_ylabel('ln(A_top / A_bot)  →  vertical y')
    ax.set_title(f'{RUN_STEM} arm {st} group {g + 1}: timing vs amplitude vertical position')
    fig.tight_layout()
    fig.savefig(OUT / f'position_corr_{st}_g{g + 1}.png', dpi=140)
    plt.close(fig)
    return slope, lam, r


# ---- (6) attenuation / Landau cancellation: A_top vs A_bot 2D
def fig_attenuation(st='C', g=2):
    h = sub(d[f'{st}_ab'][:, g, :, :])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(A2, A2, np.clip(h, 1, None).T, norm=LogNorm(), cmap='viridis',
                       rasterized=True)
    ax.plot([A2[0], A2[-1]], [A2[0], A2[-1]], 'w:', lw=1, alpha=0.7)
    fig.colorbar(im, ax=ax, label='true coincidences')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(150, 3e4); ax.set_ylim(150, 3e4)
    ax.set_xlabel('A_top [ADC]'); ax.set_ylabel('A_bot [ADC]')
    ax.set_title(f'{RUN_STEM} arm {st} group {g + 1}: A_top vs A_bot\n'
                 '(spread ALONG diagonal = Landau; ACROSS = vertical position)')
    fig.tight_layout()
    fig.savefig(OUT / f'attenuation_{st}_g{g + 1}.png', dpi=140)
    plt.close(fig)


def report_lambda():
    print(f'\n{"arm":4s} {"group":6s} {"slope[ln/ns]":>12s} {"r":>6s} {"lambda[cm]":>11s}')
    for st in GOOD_ARMS:
        for g in range(4):
            h = sub(d[f'{st}_dtlr'][:, g, :, :])
            ce_dt = 0.5 * (DT2[:-1] + DT2[1:]); ce_lr = 0.5 * (LR2[:-1] + LR2[1:])
            sl, _, r = _principal_slope(h, ce_dt, ce_lr)
            lam = V_EFF / sl if sl > 0 else np.nan
            print(f'{st:4s} g{g + 1:<5d} {sl:>12.3f} {r:>6.2f} {lam:>11.0f}')


if __name__ == '__main__':
    fig_estimators()
    fig_bybar()
    fig_position('C')
    fig_position('B')
    for g in range(4):
        fig_correlation('C', g)
    fig_attenuation('C', 2)
    print()
    report_widths()
    report_lambda()
    print(f'\nFigures -> {OUT}')
