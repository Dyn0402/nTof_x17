#!/usr/bin/env python3
"""scores.py -- GE: is chi2/dof a good discriminator of bad reconstructions, or a brightness cut?

On the M3-pointed interior views of the `fe` gate (collect.py), "bad" = |d_theta| > 5 deg OR |position residual| > 5 mm
(the gate's own 'far' definition, per view).  For each candidate score: ROC AUC for bad, and at a fixed removed
fraction (5, 10, 20 % and the fraction CHI2DOF_BAD=300 removes) the share of bad views removed and the far rate left.

Scores (larger = more suspect):
  chi2dof         chi2/dof, as the quality flag reads it (a per-chamber quantile cut has the SAME ordering)
  q_sum           brightness alone
  n_sat           number of saturated strips
  chi2_resid      log chi2/dof minus its per-chamber/view median in bins of (q_sum decile x n_sat bucket x |tan| bin):
                  the part of chi2 that brightness/saturation/angle do not explain (model-error-aware proxy)
  chi2_qnorm      chi2/dof / q_sum^a, a fitted per chamber/view (log-log slope of the median)
  mult            hit multiplicity in the window (candidate-finding level, hits allowed)
  split_fstat     the two-track split statistic (0 when no attempt)
  other_chi2      the other view's chi2/dof
  tan_err         the fit's own angle error
-> results/scores.json, results/scores.md
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser('~/x17/cosmic_bench/model_audit/ge_chi2cut/bench_views_fe.parquet')


def auc(score, y):
    s = pd.Series(score).rank().to_numpy()
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((s[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def add_scores(g):
    g = g.copy()
    g['l_chi'] = np.log(g.chi2dof.clip(lower=1e-3))
    g['n_sat'] = g.n_sat_strips
    g['qdec'] = pd.qcut(g.q_sum.rank(method='first'), 10, labels=False)
    g['sbin'] = np.digitize(g.n_sat_strips, [0.5, 1.5, 3.5, 6.5])
    g['tbin'] = np.digitize(g.tan_ref.abs(), [0.1, 0.2, 0.3])
    med = g.groupby(['qdec', 'sbin', 'tbin']).l_chi.transform('median')
    g['chi2_resid'] = g.l_chi - med
    lq = np.log(g.q_sum.clip(lower=1))
    # slope of the median log chi2/dof vs log q in q deciles
    m = g.groupby('qdec').agg(lq=('q_sum', lambda s: np.log(s.clip(lower=1)).median()), lc=('l_chi', 'median'))
    a = float(np.polyfit(m.lq, m.lc, 1)[0])
    g['chi2_qnorm'] = g.l_chi - a * lq
    g['split_fstat'] = g.split_fstat.fillna(0.0)
    g['other_chi2'] = g.chi2dof_other.fillna(0.0)
    g['q'] = g.q_sum
    return g, a


SCORES = ['chi2dof', 'q', 'n_sat', 'chi2_resid', 'chi2_qnorm', 'mult', 'split_fstat', 'other_chi2', 'tan_err']


def main():
    A = pd.read_parquet(SRC)
    A = A[(A.region == 'interior')]
    out, L = {}, []
    L.append('| chamber | view | n | bad % | slope a | ' + ' | '.join(f'AUC {s}' for s in SCORES) + ' |')
    L.append('|' + '---|' * (5 + len(SCORES)))
    L2 = ['| chamber | view | removed | rule | bad removed % | far5 kept % | s68 kept (deg) | pos>5 kept % |',
          '|---|---|---|---|---|---|---|---|']
    for (d, v), g in A.groupby(['det', 'view']):
        g, a = add_scores(g)
        bad = ((g.dth.abs() > 5) | (g.res.abs() > 5)).to_numpy()
        r = dict(n=len(g), bad=float(bad.mean()), slope=a,
                 auc={s: auc(g[s].to_numpy(), bad) for s in SCORES})
        L.append(f'| {d} | {v} | {len(g)} | {100 * bad.mean():.2f} | {a:.2f} | '
                 + ' | '.join(f'{r["auc"][s]:.2f}' for s in SCORES) + ' |')
        f300 = float((g.chi2dof >= 300).mean())
        r['cut'] = {}
        for frac in (0.0, 0.05, 0.10, 0.20, f300):
            for s in ('chi2dof', 'q', 'chi2_resid', 'mult', 'split_fstat'):
                if frac == 0 and s != 'chi2dof':
                    continue
                if frac == 0:
                    rem = np.zeros(len(g), bool)
                else:
                    thr = np.quantile(g[s], 1 - frac)
                    rem = (g[s] > thr).to_numpy() if s != 'split_fstat' else (g[s].rank(method='first', ascending=False) <= frac * len(g)).to_numpy()
                k = ~rem
                dk = g.dth[k]
                q = dict(removed=float(rem.mean()), bad_removed=float(rem[bad].mean()) if bad.any() else np.nan,
                         far5_kept=float((dk.abs() > 5).mean()), pos5_kept=float((g.res[k].abs() > 5).mean()),
                         s68_kept=float(np.percentile(np.abs(dk[dk.abs() < 5]), 68)))
                tag = 'chi2>=300' if frac == f300 and s == 'chi2dof' else s
                if frac == f300 and s != 'chi2dof':
                    tag = s + ' @300-frac'
                r['cut'][f'{tag}|{frac:.3f}'] = q
                L2.append(f'| {d} | {v} | {100 * frac:.1f} % | {tag} | {100 * q["bad_removed"]:.1f} | '
                          f'{100 * q["far5_kept"]:.2f} | {q["s68_kept"]:.3f} | {100 * q["pos5_kept"]:.2f} |')
        out[f'{d}|{v}'] = r
    md = '\n'.join(L) + '\n\n' + '\n'.join(L2) + '\n'
    print(md)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'scores.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', 'scores.md'), 'w').write(md)


if __name__ == '__main__':
    main()
