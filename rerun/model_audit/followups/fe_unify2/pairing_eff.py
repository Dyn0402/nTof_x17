#!/usr/bin/env python3
"""pairing_eff.py --arm fe [--base-bundle-arm fe] -- FE item 11: x-y pairing efficiency per event class.

Data-driven toy (as rerun/satstudy/pairing.py): two real single-track events A, B of one golden row (gate table of the
arm; single gated track, both views, n_dropped <= 2) stand for two particles crossing the chamber together.  The
reconstruction must pair x_A with y_A and x_B with y_B; it swaps when cost(xA,yB) + cost(xB,yA) < cost(xA,yA) +
cost(xB,yB), with wft.reco.xy_pair_cost and the bundle's xy_pairing (features lq, u50, u90, tc).  The pair's event class
is the worse of its two tracks' (clean < cens < sat, wft.reco.pairing_class).  Rules compared:
  class   the bundle as finalised (lq judged against its event class's own width, lq_class)
  one     the same bundle with lq_class removed (one charge width per chamber, the 10-27 state)
  charge  lq alone, one width (the satstudy toy's rule)
Pairs are drawn independently within class combinations; 4000 per (chamber, pair class).
-> results/pairing_eff.json, results/pairing_eff.md
"""
import argparse, json, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
from wft import reco as wr  # noqa: E402

T = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/gate_tree')
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
N = 4000


def fits(df, v):
    return [SimpleNamespace(q_sum=r[f'{v}_q_sum'], q_u50=r[f'{v}_q_u50'], q_u90=r[f'{v}_q_u90'],
                            t0=r[f'{v}_t0'] + 10.0 * r[f'{v}_ftst'],      # phase-corrected: compare across triggers
                            n_sat_strips=r[f'{v}_n_sat_strips'], n_flagged_strips=r[f'{v}_n_flagged_strips'],
                            n_band_strips=r[f'{v}_n_band_strips'], _q=None)
            for r in df.to_dict('records')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='fe'); ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); out = {}; L = []
    L += [f'| chamber | pair class | n | correct, per-class widths | one width | charge only | lq rsig used (class / one) |',
          '|---|---|---|---|---|---|---|']
    for det, row in ROWS.items():
        w = os.path.join(T, row, 'wft')
        tp = os.path.join(w, a.arm, 'events.parquet'); bp = os.path.join(w, f'calib_bundle_{a.arm}', 'bundle.json')
        if not (os.path.exists(tp) and os.path.exists(bp)):
            continue
        P = json.load(open(bp))['xy_pairing']
        one = {k: v for k, v in P.items() if k != 'lq_class'}
        chg = dict(features=['lq'], median={'lq': P['median']['lq']}, rsig={'lq': P['rsig']['lq']})
        d = pd.read_parquet(tp)
        d = d[d.x_ok & d.y_ok & (d.x_n_dropped <= 2) & (d.y_n_dropped <= 2) & (d.n_tracks <= 1)
              & (d.x_ftst >= 0) & (d.y_ftst >= 0)].reset_index(drop=True)
        FX, FY = fits(d, 'x'), fits(d, 'y')
        cls = np.array([wr.pairing_class(fx, fy) for fx, fy in zip(FX, FY)])
        out[det] = {}
        for pc in ('clean', 'cens', 'sat'):
            # pairs whose worse member is pc: A of class pc, B of any class not worse than pc
            ia = np.flatnonzero(cls == pc)
            ib = np.flatnonzero(np.array([wr.PAIR_CLASSES.index(c) <= wr.PAIR_CLASSES.index(pc) for c in cls]))
            if len(ia) < 30:
                continue
            A = rng.choice(ia, N); B = rng.choice(ib, N); k = A != B; A, B = A[k], B[k]
            res = {}
            for name, p in (('class', P), ('one', one), ('charge', chg)):
                ok = 0
                for i, j in zip(A, B):
                    c = lambda x, y: wr.xy_pair_cost(x, y, p, 0.0, 0)
                    keep = c(FX[i], FY[i]) + c(FX[j], FY[j]); swap = c(FX[i], FY[j]) + c(FX[j], FY[i])
                    ok += keep <= swap
                res[name] = ok / len(A)
            lc = wr.lq_balance(P, pc)[1]
            out[det][pc] = dict(n_pairs=int(len(A)), n_class_events=int(len(ia)), **res, lq_rsig_class=lc,
                                lq_rsig_one=P['rsig']['lq'])
            L.append(f'| {det} | {pc} | {len(ia)} ev | {100 * res["class"]:.1f} % | {100 * res["one"]:.1f} % | '
                     f'{100 * res["charge"]:.1f} % | {lc:.3f} / {P["rsig"]["lq"]:.3f} |')
    md = '\n'.join(L) + '\n'; print(md)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'pairing_eff.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', 'pairing_eff.md'), 'w').write(md)


if __name__ == '__main__':
    main()
