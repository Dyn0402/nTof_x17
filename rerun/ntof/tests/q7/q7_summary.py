#!/usr/bin/env python3
"""q7_summary.py -- overlay truth summary per configuration (reads q7_overlay.py score output) -> overlay_summary.json"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402
OUT = os.path.join(L.DATA, 'overlay')
SEPC = [0, 6, 12, 24, 400]


def summ(arm):
    S = pd.read_parquet(os.path.join(OUT, f'score_donors_{arm}.parquet'))
    E = pd.read_parquet(os.path.join(OUT, f'score_events_{arm}.parquet'))
    ov, ev = S[S['mode'] == 'overlay'].copy(), E[E['mode'] == 'overlay'].copy()
    sg = S[S['mode'] == 'single']
    ov['sep_min'] = np.minimum(ov.sep_x, ov.sep_y)
    ov['sepc'] = pd.cut(ov.sep_min, SEPC, labels=[f'{a}-{b}' for a, b in zip(SEPC[:-1], SEPC[1:])]).astype(str)
    res = {}
    for cf, g in ov.groupby('cfg'):
        e = ev[ev.cfg == cf]
        both = g.groupby('oid').paired.all()
        r = dict(n_overlays=int(g.oid.nunique()),
                 both_paired=float(both.mean()),
                 donor_paired=float(g.paired.mean()),
                 shallow_paired=float(g[~g.steep].paired.mean()), steep_paired=float(g[g.steep].paired.mean()),
                 ghost_pair_per_ov=float((e.ghost_swap_pair + e.ghost_junk_pair).mean()),
                 ghost_swap_pair_per_ov=float(e.ghost_swap_pair.mean()))
        for acc in ('0.6', '1'):
            m = g[f'truth_in_acc{acc}']
            r[f'acc{acc}'] = dict(eff=float(g[m][f'found_acc{acc}'].mean()), n=int(m.sum()),
                                  ghosts_per_ov=float((e[f'ghost_swap_acc{acc}'] + e[f'ghost_junk_acc{acc}']).mean()),
                                  swap_per_ov=float(e[f'ghost_swap_acc{acc}'].mean()),
                                  both_in_acc_eff=float(g[m].groupby('oid')[f'found_acc{acc}'].all()
                                                        [g[m].groupby('oid').size() == 2].mean()))
        # by tan class and separation: the shallow member of an SX pair is the canary for the gate failure
        r['by_tcls'] = {}
        for t, h in g.groupby('tcls'):
            e2 = e[e.tcls == t]
            r['by_tcls'][t] = dict(n_ov=int(h.oid.nunique()), both_paired=float(h.groupby('oid').paired.all().mean()),
                                   shallow_paired=float(h[~h.steep].paired.mean()) if (~h.steep).any() else None,
                                   shallow_found_acc06=float(h[~h.steep & h['truth_in_acc0.6']]['found_acc0.6'].mean())
                                   if (~h.steep & h['truth_in_acc0.6']).any() else None,
                                   steep_paired=float(h[h.steep].paired.mean()) if h.steep.any() else None,
                                   ghost_acc06_per_ov=float((e2['ghost_swap_acc0.6'] + e2['ghost_junk_acc0.6']).mean()),
                                   ghost_acc1_per_ov=float((e2['ghost_swap_acc1'] + e2['ghost_junk_acc1']).mean()))
        r['by_sep'] = {s: dict(n=int(h.oid.nunique()), both_paired=float(h.groupby('oid').paired.all().mean()),
                               found_acc06=float(h[h['truth_in_acc0.6']]['found_acc0.6'].mean()))
                       for s, h in g.groupby('sepc')}
        s1 = sg[sg.cfg == cf]
        r['single'] = dict(n=int(len(s1)), paired=float(s1.paired.mean()) if len(s1) else None,
                           found_acc06=float(s1[s1['truth_in_acc0.6']]['found_acc0.6'].mean()) if len(s1) else None)
        res[cf] = r
    return res


if __name__ == '__main__':
    arms = sys.argv[1:] or ['A', 'C']
    out = {a: summ(a) for a in arms if os.path.exists(os.path.join(OUT, f'score_donors_{a}.parquet'))}
    json.dump(out, open(os.path.join(OUT, 'overlay_summary.json'), 'w'), indent=1)
    for a, res in out.items():
        print('==', a)
        print(f"{'cfg':20s} both  shalP steepP ghost | acc0.6 eff ghost swap | acc1 eff ghost | SX shalAcc06 ghost06 | single")
        for cf, r in sorted(res.items()):
            sx = r['by_tcls'].get('SX', {})
            print(f"{cf:20s} {r['both_paired']:.3f} {r['shallow_paired']:.3f} {r['steep_paired']:.3f} {r['ghost_pair_per_ov']:.3f} | "
                  f"{r['acc0.6']['eff']:.3f} {r['acc0.6']['ghosts_per_ov']:.3f} {r['acc0.6']['swap_per_ov']:.3f} | "
                  f"{r['acc1']['eff']:.3f} {r['acc1']['ghosts_per_ov']:.3f} | {sx.get('shallow_found_acc06') or 0:.3f} {sx.get('ghost_acc06_per_ov', 0):.3f} | {r['single']['paired']}")
