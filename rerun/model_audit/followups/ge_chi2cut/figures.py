#!/usr/bin/env python3
"""figures.py -- GE figures from collect.py's table (M3-pointed interior views, `fe` gate).

  figures/chi2_vs_charge.png   median chi2/dof (and p10-p90) vs q_sum / median(q_sum), good vs bad views, per chamber
  figures/chi2_vs_angle.png    the same against |tan_ref| (M3)
  figures/purity_eff.png       bad views removed vs all views removed, scoring by chi2/dof, by q_sum, by the
                               brightness-corrected chi2 (chi2_resid); markers: CHI2DOF_BAD = 300 and 5 x median
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scores import add_scores, SRC

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
C_GOOD, C_BAD, C3 = '#2a78d6', '#eb6834', '#1baf7a'
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']


def band(ax, x, y, edges, color, label):
    b = np.digitize(x, edges)
    xs, m, lo, hi = [], [], [], []
    for i in range(1, len(edges)):
        s = y[b == i]
        if len(s) < 15:
            continue
        xs.append(np.sqrt(edges[i - 1] * edges[i]) if edges[0] > 0 else 0.5 * (edges[i - 1] + edges[i]))
        m.append(np.median(s)); lo.append(np.percentile(s, 10)); hi.append(np.percentile(s, 90))
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(xs, m, color=color, lw=2, marker='o', ms=4, label=label)


def main():
    os.makedirs(FIG, exist_ok=True)
    A = pd.read_parquet(SRC); A = A[A.region == 'interior']
    A['bad'] = (A.dth.abs() > 5) | (A.res.abs() > 5)
    for var, fname, edges, xl, logx in (
            ('qrel', 'chi2_vs_charge.png', np.geomspace(0.05, 20, 16), 'q_sum / median q_sum (this chamber, view)', True),
            ('atan', 'chi2_vs_angle.png', np.array([0, .03, .06, .1, .15, .2, .25, .3, .4, .55]), '|tan θ| (M3)', False)):
        fig, axs = plt.subplots(2, 5, figsize=(19, 7.5), sharex=True)
        for j, d in enumerate(DETS):
            for i, v in enumerate('xy'):
                ax = axs[i, j]; g = A[(A.det == d) & (A.view == v)]
                x = (g.q_sum / g.q_sum.median()).to_numpy() if var == 'qrel' else g.tan_ref.abs().to_numpy()
                for m, c, lab in ((~g.bad.to_numpy(), C_GOOD, 'good (|Δθ|≤5°, |r|≤5 mm)'), (g.bad.to_numpy(), C_BAD, 'bad')):
                    band(ax, x[m], g.chi2dof.to_numpy()[m], edges, c, lab)
                ax.axhline(300, color='0.4', ls='--', lw=1)
                ax.axhline(5 * g.chi2dof.median(), color=C3, ls=':', lw=1.5)
                ax.set_yscale('log')
                if logx:
                    ax.set_xscale('log')
                ax.set_title(f'{d} {v}  (bad {100 * g.bad.mean():.1f} %)', fontsize=10)
                ax.grid(alpha=0.25, lw=0.5)
                if j == 0:
                    ax.set_ylabel('χ²/dof (median, p10–p90)')
                if i == 1:
                    ax.set_xlabel(xl)
        axs[0, 0].legend(fontsize=8, loc='upper left')
        axs[0, 1].text(0.02, 0.97, '-- CHI2DOF_BAD = 300\n·· 5 × median (proposed)', transform=axs[0, 1].transAxes,
                       va='top', fontsize=8, color='0.3')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=110); plt.close(fig)

    fig, axs = plt.subplots(2, 5, figsize=(19, 7.5), sharex=True, sharey=True)
    for j, d in enumerate(DETS):
        for i, v in enumerate('xy'):
            ax = axs[i, j]
            g, _a = add_scores(A[(A.det == d) & (A.view == v)])
            bad = g.bad.to_numpy(); fr = np.linspace(0, 0.6, 61)
            for s, c, lab in (('chi2dof', C_GOOD, 'χ²/dof'), ('q', C_BAD, 'q_sum'), ('chi2_resid', C3, 'χ² at fixed q, sat, angle')):
                x = g[s].to_numpy(); o = np.argsort(-x)
                cum = np.cumsum(bad[o]) / max(bad.sum(), 1)
                n = np.clip((fr * len(x)).astype(int) - 1, 0, len(x) - 1)
                ax.plot(fr * 100, np.where(fr > 0, cum[n], 0) * 100, color=c, lw=2, label=lab)
            for rule, mk in ((g.chi2dof >= 300, 's'), (g.chi2dof > 5 * g.chi2dof.median(), 'o')):
                r = rule.to_numpy()
                ax.plot(100 * r.mean(), 100 * r[bad].mean(), mk, color='k', ms=7, mfc='none', mew=1.5)
            ax.plot([0, 60], [0, 60], color='0.6', lw=0.8, ls='--')
            ax.set_title(f'{d} {v}  (n {len(g)}, bad {100 * bad.mean():.1f} %)', fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            if j == 0:
                ax.set_ylabel('bad views removed (%)')
            if i == 1:
                ax.set_xlabel('all views removed (%)')
    axs[0, 0].legend(fontsize=8, loc='lower right')
    axs[0, 1].text(0.35, 0.08, '□ χ²/dof ≥ 300   ○ χ²/dof > 5×median\n-- random removal', transform=axs[0, 1].transAxes, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'purity_eff.png'), dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
