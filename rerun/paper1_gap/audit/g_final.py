"""g_final.py -- figures and markdown tables of audit test G from the products of g_compare.py, g_direct.py, g_bow.py."""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import g_common as G
import g_bow as GB

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'figure.dpi': 130})
OUT = G.OUT
CMP = json.load(open(f'{OUT}/compare.json'))
BM = pickle.load(open(f'{OUT}/bench_maps.pkl', 'rb'))
ARMS = [('A', 'x'), ('A', 'y'), ('C', 'x'), ('C', 'y')]
EXT = [-200, 200, -200, 200]


def img(ax, Z, vmax, title, cb=True, cmap='RdBu_r'):
    im = ax.imshow(Z.T, origin='lower', extent=EXT, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('x_local [mm]'); ax.set_ylabel('y_local [mm]')
    return im


def fig_maps(margin=0.0):
    fig, axs = plt.subplots(4, 4, figsize=(13, 12.5))
    for r, (arm, p) in enumerate(ARMS):
        sg = G.SGN[(arm, p)]
        nt = G.ntof_map(arm, p)
        good = nt['n'] >= G.N_MIN_NTOF
        B = BM[margin][arm]['all'][p]
        at, st = np.where(good, nt['a'], np.nan), np.where(good, nt['s'], np.nan)
        ab, sb = sg * B['a'], B['s']
        va = 0.05 if arm == 'A' else 0.055
        vs = 0.12
        im = img(axs[r, 0], at - np.nanmedian(at), va, f'{arm} {p}: n_TOF tilt a (in situ)'); plt.colorbar(im, ax=axs[r, 0], shrink=0.8, label='tan')
        im = img(axs[r, 1], ab - np.nanmedian(ab), va, f'{arm} {p}: bench tilt a'); plt.colorbar(im, ax=axs[r, 1], shrink=0.8, label='tan')
        im = img(axs[r, 2], st - np.nanmedian(st), vs, f'{arm} {p}: n_TOF scale s'); plt.colorbar(im, ax=axs[r, 2], shrink=0.8, label='relative')
        im = img(axs[r, 3], sb - np.nanmedian(sb), vs, f'{arm} {p}: bench scale s'); plt.colorbar(im, ax=axs[r, 3], shrink=0.8, label='relative')
    fig.suptitle('Cell maps in the n_TOF local frame (median over cells removed; grey = no data). Left pair: tilt a, right pair: scale s.')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f'{OUT}/fig1_maps.png'); plt.close(fig)


def fig_scatter(margin=0.0):
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.5))
    for c, (arm, p) in enumerate(ARMS):
        d = CMP[f'm{margin:.0f}_{arm}_{p}']
        sg = G.SGN[(arm, p)]
        nt = G.ntof_map(arm, p)
        good = nt['n'] >= G.N_MIN_NTOF
        B = BM[margin][arm]['all'][p]
        for r, (k, lab) in enumerate((('a', 'tilt a [tan]'), ('s', 'scale s'))):
            ax = axs[r, c]
            u = (sg * B['a'] if k == 'a' else B['s'])
            v = np.where(good, nt[k], np.nan)
            e = (B['ea'] if k == 'a' else B['es'])
            ax.errorbar(u.ravel(), v.ravel(), xerr=e.ravel(), fmt='o', ms=3, lw=0.6, color='C0')
            ok = np.isfinite(u) & np.isfinite(v)
            xs = np.array([np.nanmin(u[ok]), np.nanmax(u[ok])])
            sl, ic = d[f'slope_{k}'], d[f'icpt_{k}']
            ax.plot(xs, ic + sl * xs, 'k-', lw=1)
            ax.plot(xs, xs, 'r:', lw=0.8)
            ax.set_xlabel(f'bench {lab}'); ax.set_ylabel(f'n_TOF {lab}')
            ax.set_title(f"{arm} {p}: r = {d['r_'+k]:+.2f} +- {d['r_'+k+'_err']:.2f}, slope {sl:+.2f}")
    fig.suptitle(f'n_TOF map against bench map, cell by cell (bench interior margin {margin:.0f} mm); red dotted = identity, black = OLS')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'{OUT}/fig2_scatter_m{margin:.0f}.png'); plt.close(fig)


def fig_sigma(margin=0.0):
    R = pd.read_csv(f'{OUT}/direct_m{margin:.0f}.csv')
    T = R.groupby(['arm', 'view', 'model']).agg(sig=('sig_all', 'mean'), err=('err_all', 'mean')).reset_index()
    models = ['none', 'insitu', 'bench_raw', 'bench_1', 'bench_fit']
    labs = ['no map', 'in-situ map', 'bench map', 'global + bench shape', 'bench, free amplitude']
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.16
    for i, (m, lb) in enumerate(zip(models, labs)):
        y = [T[(T.arm == a) & (T.view == p) & (T.model == m)].sig.iloc[0] * 1e3 for a, p in ARMS]
        e = [T[(T.arm == a) & (T.view == p) & (T.model == m)].err.iloc[0] * 1e3 for a, p in ARMS]
        ax.bar(np.arange(4) + (i - 2) * w, y, w, yerr=e, label=lb)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{a} {p}' for a, p in ARMS])
    ax.set_ylabel('held-out single-view sigma [1e-3 in tan]'); ax.set_ylim(15, 40)
    ax.legend(ncol=3, fontsize=8, loc='upper left')
    ax.set_title('Held-out sigma of the A-C cosmic lines (train one run parity, test the other)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig3_sigma_m{margin:.0f}.png'); plt.close(fig)


def total_gradients():
    X, Y = np.meshgrid(G.CENTRES, G.CENTRES, indexing='ij')
    rows = []
    for arm in 'AC':
        bm = GB.bow_maps(arm)
        for p in 'xy':
            sg = G.SGN[(arm, p)]
            nt = G.ntof_map(arm, p)
            good = nt['n'] >= G.N_MIN_NTOF
            ab = sg * BM[0.0][arm]['all'][p]['a']
            abow = sg * bm[p]['a']
            both = good & np.isfinite(ab)
            comp = {'block': abow, 'bench total (block + bench residual)': abow + ab, 'n_TOF total (block + n_TOF residual)': abow + nt['a']}
            for k, Z in comp.items():
                Zm = np.where(both, Z, np.nan)
                co = GB.plane(Zm, both.astype(float), X, Y)
                rng = np.random.default_rng(2)
                idx = np.flatnonzero(both.ravel())
                bs = []
                for _ in range(500):
                    q = rng.choice(idx, len(idx))
                    A = np.c_[np.ones(len(q)), X.ravel()[q], Y.ravel()[q]]
                    bs.append(np.linalg.lstsq(A, Zm.ravel()[q], rcond=None)[0][1:] * 100)
                bs = np.array(bs)
                rows.append(dict(arm=arm, view=p, what=k, gx=co[1] * 100, gx_err=bs[:, 0].std(), gy=co[2] * 100, gy_err=bs[:, 1].std()))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/total_gradients.csv', index=False)
    return T


def fig_bow(T):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False)
    for c, (arm, p) in enumerate(ARMS):
        ax = axs[c]
        q = T[(T.arm == arm) & (T.view == p)]
        ax_ = 'gx' if p == 'x' else 'gy'
        names = list(q.what)
        vals = [q[q.what == n][ax_].iloc[0] * 1e3 for n in names]
        errs = [q[q.what == n][ax_ + '_err'].iloc[0] * 1e3 for n in names]
        ax.bar(range(3), vals, yerr=errs, color=['0.6', 'C0', 'C1'])
        ax.set_xticks(range(3)); ax.set_xticklabels(['bow\nblock', 'bench\ntotal', 'n_TOF\ntotal'])
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(f'{arm} {p}: d tilt / d {p}_local' if True else '')
        ax.set_ylabel('gradient [1e-3 tan per 100 mm]')
    fig.suptitle('Plane-fit tilt gradient along the view\'s own axis, common cells: block, block + bench residual, block + n_TOF residual')
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f'{OUT}/fig4_bow_gradient.png'); plt.close(fig)


def tables():
    pd.set_option('display.width', 250)
    out = []
    for margin in (0.0, 30.0):
        out.append(f'### correlation table, bench interior margin {margin:.0f} mm')
        out.append('| chamber view | cells | r(a) | slope a_ntof/a_bench | rms a bench / n_TOF [1e-3] | ceilings a (bench rows / bench tags / n_TOF parity) | r(a) / ceiling | cross r(a) (bench half x n_TOF half) | r(s) | slope s | ceilings s | cross r(s) | derived-combo r(a) | best-combo r(a) |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for arm, p in ARMS:
            d = CMP[f'm{margin:.0f}_{arm}_{p}']
            out.append(f"| {arm} {p} | {d['n_cells_a']} | {d['r_a']:+.2f} +- {d['r_a_err']:.2f} | {d['slope_a']:+.2f} +- {d['slope_a_err']:.2f} | {d['rms_a_bench']*1e3:.1f} / {d['rms_a_ntof']*1e3:.1f} | "
                       f"{d['rb_a']:.2f} / {d['rh_a']:.2f} / {d['rn_a']:.2f} | {d['att_a']:+.2f} | {d['cross_a']:+.2f} | {d['r_s']:+.2f} +- {d['r_s_err']:.2f} | {d['slope_s']:+.2f} +- {d['slope_s_err']:.2f} | "
                       f"{d['rb_s']:.2f} / {d['rh_s']:.2f} / {d['rn_s']:.2f} | {d['cross_s']:+.2f} | {d['derived_r']:+.2f} ({d['derived']}) | {d['combos'][d['best']]:+.2f} ({d['best']}) |")
        out.append('')
    for margin in (0.0, 30.0):
        R = pd.read_csv(f'{OUT}/direct_m{margin:.0f}.csv')
        T = R.groupby(['arm', 'view', 'model'], sort=False).agg(sig_all=('sig_all', 'mean'), err_all=('err_all', 'mean'), sig_clean=('sig_clean', 'mean'),
                                                                  err_clean=('err_clean', 'mean')).reset_index()
        out.append(f'### held-out sigma [1e-3 in tan], bench map from interior margin {margin:.0f} mm (mean of the two train/test directions; error = bootstrap over test lines)')
        out.append('| chamber view | no map | global only | in-situ map | bench map as is | global + bench shape | bench free amplitude | bench a only, free amplitude | in-situ shape of the other parity | gain in-situ | gain bench as is | gain bench free amp | fraction of in-situ gain recovered (free amp) |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for arm, p in ARMS:
            t = T[(T.arm == arm) & (T.view == p)].set_index('model')
            f = lambda m: f"{t.loc[m, 'sig_all']*1e3:.1f} +- {t.loc[m, 'err_all']*1e3:.1f}"  # noqa: E731
            g = lambda m: 1 - t.loc[m, 'sig_all'] / t.loc['none', 'sig_all']  # noqa: E731
            out.append(f"| {arm} {p} | {f('none')} | {f('global')} | {f('insitu')} | {f('bench_raw')} | {f('bench_1')} | {f('bench_fit')} | {f('bench_fit_a')} | {f('insitu_other_parity_shape')} | "
                       f"{g('insitu')*100:.1f} % | {g('bench_raw')*100:+.1f} % | {g('bench_fit')*100:+.1f} % | {g('bench_fit')/g('insitu')*100:.0f} % |")
        out.append('Same on the clean lines only:')
        out.append('| chamber view | no map | in-situ | bench as is | bench free amplitude |')
        out.append('|---|---|---|---|---|')
        for arm, p in ARMS:
            t = T[(T.arm == arm) & (T.view == p)].set_index('model')
            out.append(f"| {arm} {p} | " + ' | '.join(f"{t.loc[m, 'sig_clean']*1e3:.1f}" for m in ('none', 'insitu', 'bench_raw', 'bench_fit')) + ' |')
        out.append('')
        P = pd.read_csv(f'{OUT}/direct_amps_m{margin:.0f}.csv')
        out.append(f'Fitted amplitudes (margin {margin:.0f}), train parity 0 / 1:')
        out.append('| chamber view | alpha (tilt shape) | beta (scale shape) |')
        out.append('|---|---|---|')
        for arm, p in ARMS:
            q = P[(P.arm == arm) & (P.view == p)]
            out.append(f"| {arm} {p} | " + ' / '.join(f'{v:.2f} +- {e:.2f}' for v, e in zip(q.alpha, q.alpha_err)) + ' | ' + ' / '.join(f'{v:.2f} +- {e:.2f}' for v, e in zip(q.beta, q.beta_err)) + ' |')
        out.append('')
    B = pd.read_csv(f'{OUT}/bow_level.csv')
    B = B[B['map'] == 'r2']
    out.append('### quadratic level (map r2)')
    out.append('| chamber view | block tilt gradient along own axis [1e-3 /100 mm] | n_TOF residual gradient | ratio | f-1 from regression of n_TOF residual on block | same with bench residual | n_TOF minus bench residual |')
    out.append('|---|---|---|---|---|---|---|')
    for arm, p in ARMS:
        q = B[(B.arm == arm) & (B.view == p)].iloc[0]
        ax_ = 'x' if p == 'x' else 'y'
        out.append(f"| {arm} {p} | {q['grad_bow_'+ax_]*1e3:+.1f} | {q['grad_resid_'+ax_]*1e3:+.1f} | {q['ratio_'+ax_]:+.2f} | {q.fminus1_a:+.2f} +- {q.fminus1_a_err:.2f} | {q.bench_fm1_a:+.2f} +- {q.bench_fm1_a_err:.2f} | {q.diff_fm1_a:+.2f} +- {q.diff_fm1_a_err:.2f} |")
    out.append('')
    T = total_gradients()
    out.append('### tilt gradient (plane fit over common cells, 1e-3 per 100 mm along the view own axis)')
    out.append('| chamber view | block | bench total | n_TOF total |')
    out.append('|---|---|---|---|')
    for arm, p in ARMS:
        q = T[(T.arm == arm) & (T.view == p)]
        ax_ = 'gx' if p == 'x' else 'gy'
        out.append(f'| {arm} {p} | ' + ' | '.join(f'{q.iloc[i][ax_]*1e3:+.1f} +- {q.iloc[i][ax_+"_err"]*1e3:.1f}' for i in range(3)) + ' |')
    fig_bow(T)
    open(f'{OUT}/tables.md', 'w').write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    fig_maps(0.0)
    fig_scatter(0.0)
    fig_scatter(30.0)
    fig_sigma(0.0)
    fig_sigma(30.0)
    tables()
