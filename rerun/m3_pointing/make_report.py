#!/usr/bin/env python3
"""make_report.py -- figures and report.html of the M3 per-track pointing study, from the products of
decompose.py, fit.py, crosscov.py, validate.py, efficiency.py and extend.py (all under $WORK).

Writes $WORK/report/{report.html, figures/*.png + *.csv}.  Numbers in the text are read from the products, so a rerun
updates text and figures together.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'sept26_prelim_analysis')]
import figstyle as fs  # noqa: E402
from m3common import WORK, core_sigma  # noqa: E402
from rows import ROWS, PUBLIC  # noqa: E402
from decompose import comp_pdf  # noqa: E402

OUT = os.path.join(WORK, 'report')
FIG = os.path.join(OUT, 'figures')
HELD = ('A_run1', 'A_run2', 'E_624')
BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', fs.MUTED
J = lambda name: json.load(open(os.path.join(WORK, name)))


def fig_decompose(par):
    fs.use()
    fig, axs = fs.figure(figsize=(9.6, 3.6), ncols=2)
    ax = axs[0]
    p = par['X']; edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 1.5, 2, 3, 5, 8, 12, 20, 35, 60])
    from decompose import X_EDGES
    rows = []
    for j, col, mk in ((0, BLUE, 'o'), (5, GREEN, '^'), (9, ORANGE, 's')):
        H = np.array(p['H'][j]); n = p['n'][j]
        xe = np.array(X_EDGES); c = 0.5 * (xe[1:] + xe[:-1]); d = H / np.diff(xe) / n
        lo, hi = p['t_edges'][j], p['t_edges'][j + 1]
        ax.plot(c, d, mk, color=col, ms=3.5, label=f'|tan| {lo:.2f}–{hi:.2f}')
        xx = np.geomspace(0.005, 60, 300); mu = p['mu'][j]
        f = np.array(p['f']); lam = np.array(p['lam'])
        model = (1 - f.sum()) * comp_pdf(xx, mu, None) + sum(fm * comp_pdf(xx, mu, lm) for fm, lm in zip(f, lam))
        anom = sum(fm * comp_pdf(xx, mu, lm) for fm, lm in zip(f, lam))
        ax.plot(xx, model, '-', color=col, lw=1.3)
        ax.plot(xx, anom, ':', color=col, lw=1.1)
        rows += [dict(t_lo=lo, t_hi=hi, chi2=a, density=b) for a, b in zip(c, d)]
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-4, 30); ax.set_xlim(0.005, 60)
    ax.set_xlabel('M3 χ² of the X coordinate  [mm²]'); ax.set_ylabel('density  [1/mm²]')
    ax.legend(loc='lower left', fontsize=8.5)
    fs.title(ax, '(a) Ordinary core + anomalous tail', 'points: data; solid: fit; dotted: anomalous part alone')
    ax = axs[1]
    rows2 = []
    for c, col, mk in (('X', BLUE, 'o'), ('Y', ORANGE, 's')):
        t = np.array(par[c]['t_centres']); s = np.sqrt(np.array(par[c]['mu']) / 2) * 1000
        ax.plot(t, s, '-', marker=mk, color=col, label=f'M3 {c} planes')
        rows2 += [dict(coord=c, tan=a, sigma_plane_um=b) for a, b in zip(t, s)]
    ax.set_xlabel('|tan θ| of the track in that coordinate'); ax.set_ylabel('plane resolution √(μ/2)  [µm]')
    ax.set_ylim(0, None); ax.legend(loc='upper left')
    fs.title(ax, '(b) Plane resolution from the ordinary part', 'μ(t) = mean of the ordinary χ² = 2 σ²')
    fig.tight_layout()
    fs.save(fig, os.path.join(FIG, 'decompose'), data={'a': pd.DataFrame(rows), 'b': pd.DataFrame(rows2)})


def fig_crosscov(cc):
    fs.use()
    fig, ax = fs.figure(figsize=fs.FIG)
    rows = []
    for v, col, mk in (('x', BLUE, 'o'), ('y', ORANGE, 's')):
        r = cc[v]; x = np.array(r['chi2_bins']); y = np.array(r['cov']); e = np.array(r['err'])
        ax.errorbar(x, y, e, fmt=mk, color=col, ms=5, capsize=0, label=f"view {v} (M3 {r['m3']})")
        xx = np.linspace(0, x.max() * 1.05, 50)
        ax.plot(xx, r['intercept'] + r['slope'] * xx, '-', color=col, lw=1.1, alpha=0.8)
        ax.plot([0], [r['gauss_pred']], marker='*', ms=12, color=col, mec='white', mew=0.8, ls='none')
        rows += [dict(view=v, chi2=a, cov=b, err=c) for a, b, c in zip(x, y, e)]
    ax.set_xlim(-0.02, None); ax.set_ylim(0, None)
    ax.set_xlabel('M3 χ² of the matched coordinate  [mm²]'); ax.set_ylabel('cov(r_top, r_bottom)  [mm²]')
    ax.legend(loc='upper left')
    fs.title(ax, 'Two chambers share only the reference error',
             'tracks within 5° of normal incidence; stars: Gaussian pointing from the per-plane σ')
    fs.preliminary(ax, 'lower right')
    fs.save(fig, os.path.join(FIG, 'crosscov'), data=pd.DataFrame(rows))


def pooled_pulls():
    S = pd.concat([pd.read_parquet(os.path.join(WORK, 'validate', f"sigma_{r['tag']}.parquet")) for r in ROWS])
    S = S[S.res_ok]
    S['held'] = S.tag.isin(HELD)
    return S


def fig_closure(S):
    fs.use()
    fig, axs = fs.figure(figsize=(9.6, 3.4), ncols=3, sharey=True)
    specs = [('k', 'anomalous χ² E[k | χ², t]', [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0], '(a) anomaly'),
             ('t', '|tan θ| in the view', [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35], '(b) angle'),
             ('chi2', 'M3 χ² of the matched coordinate', [0, 0.03, 0.08, 0.15, 0.3, 0.5, 0.75, 1.0], '(c) χ²')]
    rows = []
    for ax, (var, lab, edges, ttl) in zip(axs, specs):
        for held, ls in ((True, '-'), (False, '--')):
            for kind, col, mk in (('model', BLUE, 'o'), ('const', ORANGE, 's')):
                xs, ys = [], []
                sfx = '' if kind == 'model' else '0'
                for lo, hi in zip(edges[:-1], edges[1:]):
                    pp, xx = [], []
                    for v, c in (('x', 'Y'), ('y', 'X')):     # both views pooled per bin
                        col_x = {'k': f'k_v{v}', 't': None, 'chi2': f'chi2_{c}'}[var]
                        xcol = S[col_x] if col_x else S[f'tan_{v}'].abs()
                        m = (xcol >= lo) & (xcol < hi) & (S.held == held)
                        s = S[m]
                        pp.append((s[f'res_{v}'] - s[f'bias{sfx}_{v}']) / s[f'sig{sfx}_{v}']); xx.append(xcol[m])
                    p = pd.concat(pp); xv = pd.concat(xx)
                    if len(p) < 150:
                        continue
                    xs.append(float(xv.mean())); ys.append(core_sigma(p)[0])
                    rows.append(dict(var=var, held_out=held, kind=kind, x=xs[-1], pull_core=ys[-1], n=len(p)))
                o = np.argsort(xs)
                ax.plot(np.array(xs)[o], np.array(ys)[o], ls, marker=mk, color=col, ms=3.5, lw=1.0,
                        label=f"{'per track' if kind == 'model' else 'constant (Paper I)'}, "
                              f"{'held-out rows' if held else 'pair rows'}")
        ax.axhline(1, color=GREY, lw=0.8)
        ax.set_xlabel(lab); fs.title(ax, ttl)
    axs[0].set_ylabel('core width of the pull r / σ_i'); axs[0].set_ylim(0.4, 2.2)
    axs[2].legend(loc='upper left', fontsize=7.5)
    fig.tight_layout()
    fs.save(fig, os.path.join(FIG, 'closure'), data=pd.DataFrame(rows))


def fig_eff(curves):
    fs.use()
    fig, ax = fs.figure(figsize=fs.FIG)
    C = curves[curves.group == 'A (Runs 1+2)']
    rows = []
    for kind, col, mk, lab in (('R', GREY, 'o', 'fixed radius R'), ('n', BLUE, 's', 'per-track n σ_i'),
                               ('n0', ORANGE, '^', 'constant-pointing n σ')):
        s = C[C.kind == kind]
        ax.plot(s.acc, s.eff, '-', color=col, lw=1.3, label=lab)
        for cut in ((5.0,) if kind == 'R' else (3.0, 5.0)):
            k = s[s.cut == cut]
            ax.plot(k.acc, k.eff, mk, color=col, ms=6)
            ax.annotate(f"{'R=' if kind == 'R' else 'n='}{cut:g}", (k.acc.iloc[0], k.eff.iloc[0]),
                        textcoords='offset points', xytext=(5, -10), fontsize=8, color=fs.INK)
        rows.append(s)
    ax.set_xscale('log'); ax.set_xlim(3e-3, 1); ax.set_ylim(85, 97)
    ax.set_xlabel('accidental match probability  [%]'); ax.set_ylabel('efficiency  [%]')
    ax.legend(loc='lower right')
    fs.title(ax, 'Per-track windows gain only where the window is tight', 'chamber A, Runs 1+2, interior, any-candidate rule')
    fs.preliminary(ax, 'upper left')
    fs.save(fig, os.path.join(FIG, 'eff_vs_acc'), data=pd.concat(rows))


def fig_extend(P, E):
    fs.use()
    fig, axs = fs.figure(figsize=(9.6, 3.5), ncols=3)
    ax = axs[0]
    for v, col, mk in (('x', BLUE, 'o'), ('y', ORANGE, 's')):
        s = P[P.view == v]
        ax.plot(s.chi2, s.res_s68, mk, color=col, ms=5, label=f'view {v}: residual s68')
        ax.plot(s.chi2, s.sig_med, '-', color=col, lw=1.2, label=f'view {v}: model σ_i (median)')
    ax.axvline(1, color=GREY, lw=0.8, ls=':'); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('M3 χ² of the matched coordinate'); ax.set_ylabel('[mm]'); ax.legend(fontsize=7.5, loc='upper left')
    fs.title(ax, '(a) Residual width')
    ax = axs[1]
    for v, col, mk in (('x', BLUE, 'o'), ('y', ORANGE, 's')):
        s = P[P.view == v]
        ax.plot(s.chi2, s.s68_1, '-', marker=mk, color=col, ms=4, label=f'per track, view {v}')
        ax.plot(s.chi2, s.s68_0, '--', marker=mk, color=col, ms=3, alpha=0.6, label=f'constant, view {v}')
    ax.axhline(1, color=GREY, lw=0.8); ax.axvline(1, color=GREY, lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('M3 χ² of the matched coordinate')
    ax.set_ylabel('s68 of the pull'); ax.legend(fontsize=7.5, loc='upper left')
    fs.title(ax, '(b) Pull width')
    ax = axs[2]
    x = np.sqrt(E.lo.clip(lower=0.03) * E.hi.clip(upper=100))
    ax.plot(x, 100 * E.eff_R5, 'o-', color=GREY, label='R = 5 mm')
    ax.plot(x, 100 * E.eff_n5, 's-', color=BLUE, label='d ≤ 5 σ_i')
    ax.plot(x, 100 * E.eff_n8, '^--', color=GREEN, label='d ≤ 8 σ_i')
    ax.axvline(1, color=GREY, lw=0.8, ls=':'); ax.set_xscale('log')
    ax.set_xlabel('max M3 χ² of the two coordinates'); ax.set_ylabel('efficiency  [%]'); ax.set_ylim(40, 100)
    ax.legend(fontsize=7.5, loc='lower left')
    fs.title(ax, '(c) Efficiency')
    fs.preliminary(ax, 'upper right')
    fig.tight_layout()
    fs.save(fig, os.path.join(FIG, 'extend'), data={'pulls': P, 'eff': E})


def main():
    os.makedirs(FIG, exist_ok=True)
    par, fr, cc = J('decompose.json'), J('fit.json'), J('crosscov.json')
    fig_decompose(par); fig_crosscov(cc)
    S = pooled_pulls(); fig_closure(S)
    curves = pd.read_parquet(os.path.join(WORK, 'efficiency', 'curves.parquet')); fig_eff(curves)
    ext = os.path.exists(os.path.join(WORK, 'extend', 'eff.parquet'))
    if ext:
        P = pd.read_parquet(os.path.join(WORK, 'extend', 'pulls.parquet'))
        E = pd.read_parquet(os.path.join(WORK, 'extend', 'eff.parquet'))
        fig_extend(P, E)
    import report_text
    html = report_text.build(par, fr, cc, J('validate/chamber.json'), J('efficiency/summary.json'), S,
                             P if ext else None, E if ext else None,
                             J('extend/closure.json') if ext else None)
    open(os.path.join(OUT, 'report.html'), 'w').write(html)
    print('wrote', os.path.join(OUT, 'report.html'))


if __name__ == '__main__':
    main()
