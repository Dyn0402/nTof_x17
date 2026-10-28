#!/usr/bin/env python3
"""rules.py -- GE: candidate rules for the per-view quality flag, judged on the M3-pointed interior views of the `fe` gate.

"bad" = |d_theta| > 5 deg OR |position residual vs M3| > 5 mm (the gate's 'far', per view).  Every rule is scale-free
(in units of the table's own median per view), so the same rule reads the same on every chamber and dataset:

  fixed300     chi2/dof >= 300 (today's CHI2DOF_BAD)
  rel5         chi2/dof > 5 x median(chi2/dof) of this table and view
  rel5_both    rel5 in this view AND in the other view (a second particle / shower brightens both)
  q3           q_sum > 3 x median(q_sum)
  wide         n_strips > median(n_strips | |tan_fit| bin) + 6  (excess width; no chi2)
  tanerr       tan_err > 5 x median(tan_err)

Per rule: removed %, bad removed %, far5 among kept, s68 among kept (|d|<5), and the angle s68 of the good views the
rule removes (the efficiency it costs: are the removed good tracks well measured?).  Also the AUC of each underlying
score.  -> results/rules.json, results/rules.md
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser('~/x17/cosmic_bench/model_audit/ge_chi2cut/bench_views_fe.parquet')


def auc(score, y):
    s = pd.Series(score).rank().to_numpy(); n1 = int(y.sum()); n0 = len(y) - n1
    return float((s[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def s68(d):
    d = d[np.abs(d) < 5]
    return float(np.percentile(np.abs(d), 68)) if len(d) > 10 else np.nan


def rules_for(g):
    med = g.chi2dof.median(); medo = g.chi2dof_other[g.other_ok].median()
    tb = np.digitize(g.tan_fit.abs(), [0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    nexp = g.groupby(tb).n_strips.transform('median')
    sc = dict(chi2dof=g.chi2dof / med,
              both=np.minimum(g.chi2dof / med, np.where(g.other_ok, g.chi2dof_other / medo, np.inf)),
              q=g.q_sum / g.q_sum.median(), wide=g.n_strips - nexp, tanerr=g.tan_err / g.tan_err.median())
    R = dict(fixed300=g.chi2dof >= 300, rel5=sc['chi2dof'] > 5, rel10=sc['chi2dof'] > 10, rel5_both=sc['both'] > 5,
             q3=sc['q'] > 3, wide=sc['wide'] > 6, tanerr=sc['tanerr'] > 5)
    return sc, {k: v.to_numpy() if hasattr(v, 'to_numpy') else np.asarray(v) for k, v in R.items()}


def main():
    A = pd.read_parquet(SRC); A = A[A.region == 'interior']
    out = {}
    L = ['| chamber | view | n | bad % | rule | removed % | bad removed % | far5 kept % | s68 kept | s68 removed-good |',
         '|---|---|---|---|---|---|---|---|---|---|']
    LA = ['| chamber | view | ' + ' | '.join(f'AUC {k}' for k in ('chi2dof', 'both', 'q', 'wide', 'tanerr')) + ' |',
          '|---|---|---|---|---|---|---|']
    for (d, v), g in A.groupby(['det', 'view']):
        g = g.reset_index(drop=True)
        bad = ((g.dth.abs() > 5) | (g.res.abs() > 5)).to_numpy()
        sc, R = rules_for(g)
        o = dict(n=len(g), bad=float(bad.mean()), auc={k: auc(np.asarray(x, float), bad) for k, x in sc.items()}, rules={})
        LA.append(f'| {d} | {v} | ' + ' | '.join(f'{o["auc"][k]:.2f}' for k in ('chi2dof', 'both', 'q', 'wide', 'tanerr')) + ' |')
        dth = g.dth.to_numpy()
        for name, rem in [('none', np.zeros(len(g), bool))] + list(R.items()):
            k = ~rem
            r = dict(removed=float(rem.mean()), bad_removed=float(rem[bad].mean()), far5_kept=float((np.abs(dth[k]) > 5).mean()),
                     s68_kept=s68(dth[k]), s68_removed_good=s68(dth[rem & ~bad]))
            o['rules'][name] = r
            L.append(f'| {d} | {v} | {len(g)} | {100 * bad.mean():.1f} | {name} | {100 * r["removed"]:.1f} | '
                     f'{100 * r["bad_removed"]:.1f} | {100 * r["far5_kept"]:.2f} | {r["s68_kept"]:.3f} | {r["s68_removed_good"]:.3f} |')
        out[f'{d}|{v}'] = o
    md = '\n'.join(LA) + '\n\n' + '\n'.join(L) + '\n'
    print(md)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'rules.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', 'rules.md'), 'w').write(md)


if __name__ == '__main__':
    main()
