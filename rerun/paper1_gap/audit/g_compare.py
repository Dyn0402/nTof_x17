"""g_compare.py -- bench residual maps (beyond the bow block) vs the n_TOF in-situ map, cell by cell (audit test G, steps 2-3).

    python g_compare.py     -> $WORK/audit/g/{bench_maps.pkl, ntof_halves.pkl, compare.json, compare_tables.md}
"""
import json
import os
import pickle
import sys

os.environ.setdefault('X17_TQ_OUT', '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/g/tq')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools

import numpy as np
import pandas as pd

import g_common as G
from rerun.ntof.improve import angle_map as AM

os.makedirs(G.OUT, exist_ok=True)
LINES = '/media/ucla/work/ntof_pass1/ana_r2/track_quality/lines.parquet'


def ntof_halves(n_min=40, nboot=40):
    """the n_TOF map refitted on the two run parities (independent noise) and on all lines, cell-wise, with bootstrap errors."""
    M = pd.read_parquet(LINES)
    rn = M.run.str.replace('run_', '').astype(int)
    out = {}
    for s, arm in (('a', 'A'), ('c', 'C')):
        for p in 'xy':
            Gt = AM.clean_lines(M, s, p)
            par = Gt.run.str.replace('run_', '').astype(int) % 2
            d = {}
            for lab, m in (('even', par == 0), ('odd', par == 1), ('all', par >= 0)):
                g = Gt[m]
                d[lab] = G.fit_cells(g.x.to_numpy(), g.y.to_numpy(), g.j.to_numpy(), g.r.to_numpy(), n_min if lab != 'all' else 80,
                                     nboot=nboot if lab == 'all' else 0)
            out[(arm, p)] = d
            print(arm, p, 'clean lines', len(Gt), flush=True)
    return out


def flips(M, fx, fy):
    M = M[::-1, :] if fx else M
    return M[:, ::-1] if fy else M


def main():
    res = {}
    for margin in (30.0, 0.0):
        res[margin] = {arm: G.bench_maps(arm, margin=margin) for arm in 'AC'}
        for arm in 'AC':
            print('margin', margin, arm, res[margin][arm]['n_events'], flush=True)
    pickle.dump(res, open(f'{G.OUT}/bench_maps.pkl', 'wb'))
    NH = ntof_halves()
    pickle.dump(NH, open(f'{G.OUT}/ntof_halves.pkl', 'wb'))

    out = {}
    lines = []
    for margin in (30.0, 0.0):
        for arm in 'AC':
            for p in 'xy':
                B = res[margin][arm]
                sg = G.SGN[(arm, p)]
                nt = G.ntof_map(arm, p, 'r2')
                at = np.where(nt['n'] >= G.N_MIN_NTOF, nt['a'], np.nan)
                st = np.where(nt['n'] >= G.N_MIN_NTOF, nt['s'], np.nan)
                ab = sg * B['all'][p]['a']
                sb = B['all'][p]['s']
                # de-meaned versions (remove the common-cell mean over the common support)
                key = f'm{margin:.0f}_{arm}_{p}'
                ok = np.isfinite(at) & np.isfinite(ab)
                oks = np.isfinite(st) & np.isfinite(sb)
                r_a, r_s = G.pearson(at, ab), G.pearson(st, sb)
                ea, es = G.boot_corr(at, ab), G.boot_corr(st, sb)
                sl_a = G.ols(ab, at); sl_s = G.ols(sb, st)
                # noise ceilings: bench rows split, n_TOF parity split
                rb_a = G.pearson(B['r1'][p]['a'], B['r2'][p]['a']); rb_s = G.pearson(B['r1'][p]['s'], B['r2'][p]['s'])
                nh = NH[(arm, p)]
                rn_a = G.pearson(nh['even']['a'], nh['odd']['a']); rn_s = G.pearson(nh['even']['s'], nh['odd']['s'])
                rh_a = G.pearson(B['h0'][p]['a'], B['h1'][p]['a']); rh_s = G.pearson(B['h0'][p]['s'], B['h1'][p]['s'])
                sb_ = lambda r: 2 * r / (1 + r)       # Spearman-Brown: reliability of the pooled data
                att_a = r_a / np.sqrt(max(sb_(rh_a), 1e-3) * max(sb_(rn_a), 1e-3))
                att_s = r_s / np.sqrt(max(sb_(rh_s), 1e-3) * max(sb_(rn_s), 1e-3))
                # cross products free of within-map noise: bench row1 vs n_TOF even, row2 vs odd, etc.
                cross_a = np.mean([G.pearson(sg * B[r][p]['a'], nh[h]['a']) for r in ('r1', 'r2') for h in ('even', 'odd')])
                cross_s = np.mean([G.pearson(B[r][p]['s'], nh[h]['s']) for r in ('r1', 'r2') for h in ('even', 'odd')])
                # the 8 sign/mirror combinations for a (flip x position, flip y position, flip sign of a) and 4 for s
                combos = {}
                for fx, fy, fa in itertools.product((0, 1), repeat=3):
                    bb = flips(B['all'][p]['a'], fx, fy) * (-1 if fa else 1)
                    combos[f'fx{fx}fy{fy}fa{fa}'] = G.pearson(at, bb)
                combos_s = {f'fx{fx}fy{fy}': G.pearson(st, flips(B['all'][p]['s'], fx, fy)) for fx, fy in itertools.product((0, 1), repeat=2)}
                derived = 'fx0fy0fa' + ('1' if sg < 0 else '0')
                best = max(combos, key=lambda k: combos[k] if np.isfinite(combos[k]) else -9)
                d = dict(n_cells_a=int(ok.sum()), n_cells_s=int(oks.sum()), r_a=r_a, r_a_err=ea, r_s=r_s, r_s_err=es,
                         slope_a=sl_a[0], slope_a_err=sl_a[2], icpt_a=sl_a[1], slope_s=sl_s[0], slope_s_err=sl_s[2], icpt_s=sl_s[1],
                         rb_a=rb_a, rb_s=rb_s, rh_a=rh_a, rh_s=rh_s, rn_a=rn_a, rn_s=rn_s, att_a=att_a, att_s=att_s, cross_a=cross_a, cross_s=cross_s,
                         rms_a_bench=float(np.sqrt(np.nanmean((ab - np.nanmean(ab)) ** 2))), rms_a_ntof=float(np.sqrt(np.nanmean((at - np.nanmean(at)) ** 2))),
                         rms_s_bench=float(np.sqrt(np.nanmean((sb - np.nanmean(sb)) ** 2))), rms_s_ntof=float(np.sqrt(np.nanmean((st - np.nanmean(st)) ** 2))),
                         combos=combos, combos_s=combos_s, derived=derived, best=best, derived_r=combos[derived],
                         a0_bench=B['all'][p]['a0'] * sg, s0_bench=B['all'][p]['s0'], a0_ntof=nt['a0'], s0_ntof=nt['s0'])
                out[key] = d
    json.dump(out, open(f'{G.OUT}/compare.json', 'w'), indent=1, default=float)
    for k, d in out.items():
        print(k, f"cells {d['n_cells_a']}/{d['n_cells_s']}  r_a {d['r_a']:+.2f}+-{d['r_a_err']:.2f} (ceil bench rows {d['rb_a']:.2f} tags {d['rh_a']:.2f} ntof {d['rn_a']:.2f}, cross {d['cross_a']:+.2f}) "
                     f"slope_a {d['slope_a']:+.2f}+-{d['slope_a_err']:.2f} | r_s {d['r_s']:+.2f}+-{d['r_s_err']:.2f} (ceil {d['rb_s']:.2f},{d['rh_s']:.2f},{d['rn_s']:.2f}, cross {d['cross_s']:+.2f}) "
                     f"slope_s {d['slope_s']:+.2f}+-{d['slope_s_err']:.2f} | derived {d['derived']} r={d['derived_r']:+.2f}; best {d['best']} r={d['combos'][d['best']]:+.2f}")


if __name__ == '__main__':
    main()
