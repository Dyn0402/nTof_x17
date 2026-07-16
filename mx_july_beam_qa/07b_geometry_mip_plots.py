"""
07b_geometry_mip_plots.py — Figures for 06 (wall geometry test) and 07 (MIP spectra).

Usage: python 07b_geometry_mip_plots.py [run_stem]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '06_07_geometry_mip'
OUT.mkdir(parents=True, exist_ok=True)

CH_COLORS = plt.get_cmap('tab10').colors


def excess_matrix(h, cen, side):
    n_a, n_b = h.shape[:2]
    m = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            hh = h[i, j]
            base = hh[side].mean()
            sub = np.clip(hh - base, 0, None)
            win = np.abs(cen - cen[np.argmax(sub)]) <= 10
            m[i, j] = sub[win].sum()
    return m


def geometry_figs():
    d = np.load(BASE / 'cache' / f'06_wallgeom_{RUN_STEM}.npz')
    cen = 0.5 * (d['dt_edges'][:-1] + d['dt_edges'][1:])
    side = np.abs(cen) > 60

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (a, b) in enumerate([('A', 'D'), ('B', 'C')]):
        for col, key in enumerate([f'intra_{a}', f'intra_{b}', f'cross_{a}{b}']):
            ax = axes[row][col]
            m = excess_matrix(d[key], cen, side)
            if key.startswith('intra'):
                np.fill_diagonal(m, np.nan)  # same-channel retriggers, not geometry
            im = ax.imshow(m, cmap='viridis', norm=LogNorm(vmin=max(np.nanmin(m[m > 0]), 1)))
            fig.colorbar(im, ax=ax, shrink=0.8)
            st_a, st_b = (key[6], key[6]) if key.startswith('intra') else (key[6], key[7])
            ax.set_xlabel(f'WAL{st_b} channel')
            ax.set_ylabel(f'WAL{st_a} channel')
            ax.set_xticks(range(8), [str(i + 1) for i in range(8)])
            ax.set_yticks(range(8), [str(i + 1) for i in range(8)])
            ttl = f'within arm {st_a}' if key.startswith('intra') else f'cross {st_a}-{st_b}'
            ax.set_title(f'{ttl}: coincidence excess')
    fig.suptitle(f'{RUN_STEM}: wall channel-channel coincidence matrices '
                 '(expect (1,2),(3,4),(5,6),(7,8) top-bottom blocks within arm)')
    fig.tight_layout()
    fig.savefig(OUT / 'wall_geometry_matrices.png', dpi=140)
    plt.close(fig)


def mip_figs():
    from adc_mv import mv_factors
    fac = mv_factors()
    d = np.load(BASE / 'cache' / f'07_mip_{RUN_STEM}.npz')
    cen = np.sqrt(d['amp_edges'][:-1] * d['amp_edges'][1:])
    wid = np.diff(d['amp_edges'])
    sc = float(d['sb_scale'])

    # wall/pss spectra in mV: one panel per arm; log and linear versions
    for det, n_ch, base in (('wal', 8, 'mip_wall_spectra'),
                            ('pss', 2, 'mip_pss_spectra')):
        for scale, xmax in (('log', None), ('linear', 150)):
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=False)
            for ax, st in zip(axes, 'ABCD'):
                tree = ('WAL' if det == 'wal' else 'PSS') + st
                h = d[f'{st}_{det}']
                for ch in range(n_ch):
                    f_mv = fac[tree][ch]
                    sub = (h[0, ch] - sc * h[1, ch]) / (wid * f_mv)  # hits / mV
                    ax.plot(cen * f_mv, sub, lw=1, color=CH_COLORS[ch],
                            label=f'ch{ch + 1}')
                if scale == 'log':
                    ax.set_xscale('log')
                else:
                    ax.set_xlim(0, xmax)
                    ax.set_ylim(bottom=0)
                ax.set_title(tree)
                ax.set_xlabel('amplitude [mV]')
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7, ncol=2, frameon=False)
                ax.axhline(0, color='#9ca3af', lw=0.8)
            axes[0].set_ylabel('true-coincidence hits / mV')
            fig.suptitle(f'{RUN_STEM}: sideband-subtracted coincidence amplitude spectra '
                         f'({"SiPM wall" if det == "wal" else "plastic bars"}, '
                         f'tof > 0.1 ms, {scale} axes)')
            fig.tight_layout()
            suffix = '' if scale == 'log' else '_linear'
            fig.savefig(OUT / f'{base}{suffix}.png', dpi=140)
            plt.close(fig)

    # 2D wall vs pss amplitude (subtracted), per arm, in mV (mean factor per tree)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
    e2 = d['amp2d_edges']
    for ax, st in zip(axes, 'ABCD'):
        fw, fp = fac[f'WAL{st}'].mean(), fac[f'PSS{st}'].mean()
        h2 = d[f'{st}_2d']
        sub = h2[0] - sc * h2[1]
        im = ax.pcolormesh(e2 * fw, e2 * fp, np.clip(sub, 1, None).T, norm=LogNorm(),
                           cmap='viridis', rasterized=True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'arm {st}')
        ax.set_xlabel('wall amp [mV]')
    axes[0].set_ylabel('plastic amp [mV]')
    fig.colorbar(im, ax=axes, label='true pairs / bin', shrink=0.85)
    fig.suptitle(f'{RUN_STEM}: wall vs plastic amplitude of true coincidences')
    fig.savefig(OUT / 'mip_2d_amp.png', dpi=140)
    plt.close(fig)

    # MIP peak table in mV (mode of smoothed subtracted spectrum above ~4.5 mV)
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    kern /= kern.sum()
    print(f'{"channel":10s} {"peak [mV]":>10s} {"peak [ADC]":>11s}')
    for st in 'ABCD':
        for det, n_ch in (('wal', 8), ('pss', 2)):
            tree = ('WAL' if det == 'wal' else 'PSS') + st
            h = d[f'{st}_{det}']
            for ch in range(n_ch):
                sub = h[0, ch] - sc * h[1, ch]
                sm = np.convolve(sub, kern, mode='same')
                m = cen > 150
                pk = cen[m][np.argmax(sm[m])]
                print(f'{tree}{ch + 1:<3d} {pk * fac[tree][ch]:>10.1f} {pk:>11.0f}')


if __name__ == '__main__':
    geometry_figs()
    mip_figs()
    print(f'Figures -> {OUT}')
