#!/usr/bin/env python3
"""T5 figures + pointmap.npz (collects the per-cell arrays)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pm

plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': False, 'figure.dpi': 130})
H = pm.HERE
BLUE, ORANGE, GREEN, GREY = '#2a6fb0', '#d9822b', '#3a9a5b', '#7a7a7a'


def mapimg(ax, Z, edges, title, vmax, cmap='RdBu_r', label='x 10$^{-3}$'):
    im = ax.imshow(Z.T * 1000, origin='lower', extent=[edges[0], edges[-1], edges[0], edges[-1]], cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x_local [mm]')
    return im


def hmaps():
    for arm in 'ACD':
        o = np.load(H / f'hmap_{arm}_25.npz')
        e, v = o['edges'], o['valid']
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
        im = mapimg(axs[0], np.where(v, o['c'], np.nan), e, f'{arm}: pointing centre c', 50)
        mapimg(axs[1], np.where(v, o['pred'], np.nan), e, f'{arm}: cosmic map a + s t$_p$', 50)
        fig.colorbar(im, ax=axs[:2], shrink=0.8, label='tan residual x 10$^{-3}$')
        pull = np.where(v, (o['c'] - o['pred']) / np.hypot(o['err'], o['pred_err']), np.nan)
        im2 = axs[2].imshow(pull.T, origin='lower', extent=[e[0], e[-1], e[0], e[-1]], cmap='RdBu_r', vmin=-20,
                            vmax=20, interpolation='nearest')
        axs[2].set_title('(c - cosmic) / error', fontsize=10)
        axs[2].set_xlabel('x_local [mm]')
        fig.colorbar(im2, ax=axs[2], shrink=0.8, label='pull')
        axs[0].set_ylabel('y_local [mm]')
        fig.savefig(H / f'fig_hmap_{arm}.png')
        plt.close(fig)


def scatter():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    T = pd.read_csv(H / 'hmap_compare.csv')
    for ax, arm in zip(axs, 'ACD'):
        o = np.load(H / f'hmap_{arm}_25.npz')
        v = o['valid']
        ax.errorbar(o['pred'][v] * 1000, o['c'][v] * 1000, xerr=o['pred_err'][v] * 1000, yerr=o['err'][v] * 1000,
                    fmt='o', ms=3.5, color=BLUE, ecolor='#9bbbd9', lw=0.8)
        lim = [-70, 50]
        ax.plot(lim, lim, color=GREY, lw=1, ls='--', label='1:1')
        r = T[(T.arm == arm) & (T.cell == 25)].iloc[0]
        ax.set_title(f'{arm}: r = {r['corr']:.2f}, slope {r['slope']:.2f}, rms diff {r['rms_diff'] * 1000:.1f}e-3', fontsize=9.5)
        ax.set_xlabel('cosmic map a + s t$_p$ [x10$^{-3}$]')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    axs[0].set_ylabel('pointing centre c [x10$^{-3}$]')
    axs[0].legend(frameon=False)
    fig.savefig(H / 'fig_scatter_c_vs_cosmic.png')
    plt.close(fig)


def c_vs_e():
    mp = pm.cosmic_maps()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True, sharey=True)
    for ext, col, lab in ((False, BLUE, '|tan| < 0.5 (gated)'), (True, ORANGE, '|tan| < 0.6')):
        d = pm.load(ext=ext)
        for ax, arm in zip(axs, 'ACD'):
            g = d[d.arm == arm]
            A = pm.arm_arrays(g, arm)
            a, s = pm.interp(mp[arm + '_x'], A['x'], A['y'])
            r = A['sh'] - A['e']
            cy = np.abs(A['y']) < 100
            eb = np.arange(-0.6, 0.61, 0.02)
            b = np.digitize(A['e'], eb)
            xs, cs, ps = [], [], []
            for k in range(1, len(eb)):
                m = cy & (b == k)
                if m.sum() < 500:
                    continue
                xs.append(eb[k - 1] + 0.01)
                cs.append(pm.wmedian(r[m]) * 1000)
                ps.append(np.mean((a + s * A['e'])[m]) * 1000)
            ax.plot(xs, cs, '-', color=col, lw=2, label=f'pointing, {lab}')
            if ext:
                ax.plot(xs, ps, '-', color=GREY, lw=1.5, label='cosmic map')
            ax.set_title(arm, fontsize=10)
            ax.set_xlabel('expected tan t$_p$ (|y_local| < 100 mm)')
            ax.axvspan(-0.6, -0.44, color='#eeeeee', zorder=0)
            ax.axvspan(0.44, 0.6, color='#eeeeee', zorder=0)
    axs[0].set_ylabel('residual centre [x10$^{-3}$]')
    axs[0].set_ylim(-130, 130)
    axs[0].legend(frameon=False, fontsize=8.5)
    fig.savefig(H / 'fig_c_vs_tp.png')
    plt.close(fig)


def validation():
    T = pd.read_csv(H / 'validation_cosmic_lines.csv')
    T = T[(T.cell == 25) & (T.region.isin(['covered', 'all']))]
    meth = [('none', 'no map'), ('cosmic', 'cosmic map (held out)'), ('P0', 'pointing c only'),
            ('P1', 'pointing c, cosmic s'), ('cos+Pd', 'cosmic + local pointing')]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True, sharey=True)
    cols = [GREY, BLUE, ORANGE, '#b5651d', GREEN]
    for ax, reg in zip(axs, ['covered', 'all']):
        S = T[T.region == reg].set_index('arm')
        x = np.arange(3)
        w = 0.16
        for i, ((k, lab), c) in enumerate(zip(meth, cols)):
            ax.bar(x + (i - 2) * w, [S.loc[a, f'sig_{k}'] * 1000 for a in 'ACD'], w * 0.9, color=c, label=lab,
                   yerr=[S.loc[a, f'err_{k}'] * 1000 for a in 'ACD'], error_kw=dict(lw=0.8))
        ax.set_xticks(x, [f'{a} ({int(S.loc[a, "n_lines"])} lines)' for a in 'ACD'])
        ax.set_title('lines in valid pointing cells' if reg == 'covered' else 'all clean lines', fontsize=10)
        ax.set_ylim(10, 40)
    axs[0].set_ylabel('clean single-view sigma (MAD) x10$^{-3}$')
    axs[1].legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.savefig(H / 'fig_validation_cosmic_lines.png')
    plt.close(fig)


def sysfig():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    for ax, arm in zip(axs, 'ACD'):
        nom = np.load(H / f'hmap_{arm}_50.npz')
        v = nom['valid']
        e = nom['e'][v]
        for lab, col in (('xs+1', BLUE), ('dist+1%', ORANGE), ('midgap', GREY), ('q_low', GREEN), ('first_half', '#8e5bb5')):
            o = np.load(H / f'hsys_{arm}_{lab}.npz')
            dc = (o['c'] - nom['c'])[v] * 1000
            ax.plot(e, dc, 'o', ms=4, color=col, label=lab if arm == 'A' else None)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_title(f'{arm}: change of c (50 mm cells)', fontsize=10)
        ax.set_xlabel('cell mean t$_p$')
    axs[0].set_ylabel('c(variant) - c(nominal) [x10$^{-3}$]')
    axs[0].legend(frameon=False, fontsize=8.5)
    fig.savefig(H / 'fig_systematics.png')
    plt.close(fig)


def vfig():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    V = pd.read_csv(H / 'vmap_compare.csv')
    for ax, arm in zip(axs, 'ACD'):
        o = np.load(H / f'vmap_{arm}_25.npz')
        if 'cen_detr' not in o:
            continue
        v = np.isfinite(o['cen_detr'])
        ax.errorbar(o['pred_cen_detr'][v] * 1000, o['cen_detr'][v] * 1000, yerr=o['cen_err'][v] * 1000, fmt='o',
                    ms=3.5, color=BLUE, ecolor='#9bbbd9', lw=0.8)
        ax.plot([-15, 15], [-15, 15], ls='--', color=GREY, lw=1)
        r = V[(V.arm == arm) & (V.cell == 25) & (V.est == 'cen')].iloc[0]
        ax.set_title(f'{arm} vertical: r = {r.corr_detr:.2f}, slope {r.slope_detr:.1f}', fontsize=10)
        ax.set_xlabel('cosmic y map, quadratic removed [x10$^{-3}$]')
    axs[0].set_ylabel('box centre, quadratic removed [x10$^{-3}$]')
    fig.savefig(H / 'fig_vertical.png')
    plt.close(fig)


def collect():
    out = {}
    for arm in 'ACD':
        for cs in ('25', '50'):
            o = np.load(H / f'hmap_{arm}_{cs}.npz')
            for k in o.files:
                if k in ('holes_x',) or o[k].ndim == 2 or k == 'edges':
                    out[f'h_{arm}_{cs}_{k}'] = o[k]
            ov = np.load(H / f'vmap_{arm}_{cs}.npz')
            for k in ov.files:
                out[f'v_{arm}_{cs}_{k}'] = ov[k]
    out['README'] = np.array('keys h_<arm>_<cell>_<field> horizontal (c = pointing residual centre, err, n, sig, '
                             'c_map = after cosmic map, pred = cosmic a+s t_p, pred_err bootstrap, valid, e = mean t_p, '
                             'c_ext06 = |tan|<0.6 sample), v_<arm>_<cell>_<field> vertical (cen, hi, lo edges, '
                             'pred_*, *_detr). Arrays [ix, iy] over x_local, y_local edges -200..200; global slopes '
                             '(A,C d_x/d_z; D d_z/d_x), unmapped; source (-10.9, -4.5) mm, y_c -28.5 mm.')
    np.savez(H / 'pointmap.npz', **out)


if __name__ == '__main__':
    hmaps()
    scatter()
    validation()
    sysfig()
    vfig()
    collect()
    c_vs_e()
