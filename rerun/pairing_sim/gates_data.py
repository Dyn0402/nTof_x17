#!/usr/bin/env python3
"""gates_data.py -- the data side of the x<->y pairing validation gates (PLAN.md §4), on every bench chamber.

    ../../.venv/bin/python gates_data.py            -> ~/x17/cosmic_bench/satstudy/pairing_sim/gates_data.json

Per chamber (golden row of rerun/effbudget, production final_phys candidate table, M3 rays): clean single tracks are
interior M3 rays that production detects (cat_xy_5 == 'detected') and where the view holds exactly ONE candidate
(rank >= 0) and it lies within 5 mm of the ray at the mesh. That is stricter than satstudy's class S (which drops only
particle-like extra clusters, from a waveform pass that exists for det3 alone): an event with a noise band is lost
here, which costs statistics, not bias in the x/y quantities. det3 is reported with both selections as the check.

Gate quantities (all from the production fit, as wft.reco.xy_pair_cost uses them):
  lq           ln(q_sum_x / q_sum_y): median and robust sigma (1.4826 MAD)
  u50, u90     x minus y charge-arrival quantile [ns]: median and robust sigma
  lnQ          0.5 ln(q_sum_x q_sum_y): the Landau width, as robust sigma and as Q quantiles / median Q
  vs angle     robust sigma of lq in bins of the M3 inclination (max |theta_x|, |theta_y|), for angle-matched sim
  toy          pairing.py's toy (odd ids calibrate, even ids are paired at random, 200 k pairs): right fraction by
               charge (q), charge + profile (q_u), and right vs |ln Q1/Q2| (by_ratio)
det7 x: only connector 5-8 (FEU 6 ch 256-511, x >= 200 mm) is quoted (memory det7-x-readout-faults: the M1 cards were
badly seated); the det7 rows use events whose x track lies in that half.
"""
import json, os
import numpy as np
import pandas as pd

BUDGET = os.path.expanduser('~/x17/cosmic_bench/effbudget')
OUT = os.path.expanduser('~/x17/cosmic_bench/satstudy/pairing_sim')
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
R_MM = 5.0
ANG_EDGES = [0, 5, 10, 15, 20, 30, 90]
RATIO_EDGES = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 10]
FEATS = dict(q=['lq'], q_u=['lq', 'u50', 'u90'], u=['u50', 'u90'])
N_PAIRS = 200_000


def rsig(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else float('nan')


def singles(det):
    b = json.load(open(os.path.join(BUDGET, f'budget_{det}.json')))
    R = pd.read_parquet(os.path.join(BUDGET, f'rays_{det}.parquet'))
    R = R[(R.region == 'interior') & ~R.not_measured & (R.cat_xy_5 == 'detected')].set_index('event_id')
    C = pd.read_parquet(b['table'].replace('.parquet', '.candidates.parquet'))
    C = C[(C['rank'] >= 0) & C.event_id.isin(R.index)]
    n = C.groupby(['event_id', 'plane']).size().unstack(fill_value=0)
    one = n.index[(n.get('x', 0) == 1) & (n.get('y', 0) == 1)]
    C = C[C.event_id.isin(one)]
    w = C.pivot(index='event_id', columns='plane', values=['p0', 'tan_theta', 'q_sum', 'q_u50', 'q_u90', 't0'])
    R = R.loc[w.index]
    near = ((w.p0.x - R.mesh_x).abs() <= R_MM) & ((w.p0.y - R.mesh_y).abs() <= R_MM)
    w, R = w[near], R[near]
    if det == 'det7':      # x from FEU 6 ch 256-511 only
        keep = (w.p0.x >= 200.0) & (w.p0.x + 30 * w.tan_theta.x >= 200.0)
        w, R = w[keep], R[keep]
    ok = (w.q_sum.x > 0) & (w.q_sum.y > 0)
    return b['run_key'], w[ok], R[ok], int(len(n))


def satstudy_singles():
    """det3 with satstudy's class S selection (pairing.singles), for the cross-check."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'satstudy'))
    import pairing as P
    return P.singles()


def feats(w):
    return dict(lq=np.log(w.q_sum.x / w.q_sum.y).to_numpy(), u50=(w.q_u50.x - w.q_u50.y).to_numpy(),
                u90=(w.q_u90.x - w.q_u90.y).to_numpy())


def cost(d, cal, keys):
    return sum(np.minimum(((d[k] - cal[k][0]) / cal[k][1]) ** 2, 25.0) for k in keys)


def toy(w, seed=1):
    """pairing.py toy on a singles table (columns q_sum/q_u50/q_u90 x x/y)."""
    odd = np.asarray(w.index) % 2 == 1
    f = feats(w[odd]); cal = {k: (float(np.median(v)), rsig(v)) for k, v in f.items()}
    T = w[~odd]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(T), N_PAIRS); j = rng.integers(0, len(T), N_PAIRS); k = i != j; i, j = i[k], j[k]
    qx, qy = T.q_sum.x.to_numpy(), T.q_sum.y.to_numpy()
    ux5, uy5, ux9, uy9 = (T.q_u50.x.to_numpy(), T.q_u50.y.to_numpy(), T.q_u90.x.to_numpy(), T.q_u90.y.to_numpy())
    g = lambda a, b: dict(lq=np.log(qx[a] / qy[b]), u50=ux5[a] - uy5[b], u90=ux9[a] - uy9[b])
    lqr = np.abs(0.5 * np.log(qx[i] * qy[i]) - 0.5 * np.log(qx[j] * qy[j]))
    b = np.digitize(lqr, RATIO_EDGES) - 1
    out = dict(n_cal=int(odd.sum()), n_test=int(len(T)), n_pairs=int(len(i)), cal=cal, right={}, by_ratio={})
    for nm, ks in FEATS.items():
        ok = (cost(g(i, i), cal, ks) + cost(g(j, j), cal, ks)) < (cost(g(i, j), cal, ks) + cost(g(j, i), cal, ks))
        out['right'][nm] = float(ok.mean())
        out['by_ratio'][nm] = [dict(lo=RATIO_EDGES[q], hi=RATIO_EDGES[q + 1], n=int((b == q).sum()),
                                    right=float(ok[b == q].mean()) if (b == q).any() else None)
                               for q in range(len(RATIO_EDGES) - 1)]
    out['frac_ratio_lt_0p1'] = float((lqr < 0.1).mean())
    return out


def gate(w, R=None):
    f = feats(w)
    lnQ = 0.5 * np.log(w.q_sum.x.to_numpy() * w.q_sum.y.to_numpy())
    Q = np.exp(lnQ); qm = float(np.median(Q))
    o = dict(n=int(len(w)))
    for k, v in f.items():
        o[k] = dict(median=float(np.median(v)), rsig=rsig(v), s68=float(np.percentile(np.abs(v - np.median(v)), 68)))
    o['lnQ'] = dict(median_Q=qm, rsig=rsig(lnQ), q_over_median={str(p): float(np.percentile(Q, p) / qm)
                                                                  for p in (5, 10, 25, 75, 90, 95)})
    # bootstrap stat error of rsig(lq)
    rng = np.random.default_rng(3); lq = f['lq']
    o['lq']['rsig_err'] = float(np.std([rsig(rng.choice(lq, len(lq))) for _ in range(200)]))
    if R is not None:
        ang = np.degrees(np.arctan(np.maximum(np.abs(R.tan_x.to_numpy()), np.abs(R.tan_y.to_numpy()))))
        ax = np.degrees(np.arctan(np.abs(R.tan_x.to_numpy()))); ay = np.degrees(np.arctan(np.abs(R.tan_y.to_numpy())))
        o['angle_hist'] = np.histogram(ang, bins=ANG_EDGES)[0].tolist()
        o['vs_angle'] = []
        for lo, hi in zip(ANG_EDGES[:-1], ANG_EDGES[1:]):
            s = (ang >= lo) & (ang < hi)
            o['vs_angle'].append(dict(lo=lo, hi=hi, n=int(s.sum()), lq_rsig=rsig(lq[s]) if s.sum() > 20 else None,
                                      lq_med=float(np.median(lq[s])) if s.sum() > 20 else None,
                                      u50_rsig=rsig(f['u50'][s]) if s.sum() > 20 else None,
                                      lnQ_rsig=rsig(lnQ[s]) if s.sum() > 20 else None))
        # per inclined view (the other view within 5 deg): sim points thx<a> / thy<a> are directly comparable
        o['vs_view_angle'] = {}
        for nm, a, other in (('x', ax, ay), ('y', ay, ax)):
            rows = []
            for c in (0, 5, 10, 15, 20, 30):
                s = (np.abs(a - c) < 2.5) & (other < 5)
                rows.append(dict(deg=c, n=int(s.sum()), lq_rsig=rsig(lq[s]) if s.sum() > 20 else None,
                                 lq_med=float(np.median(lq[s])) if s.sum() > 20 else None,
                                 u50_rsig=rsig(f['u50'][s]) if s.sum() > 20 else None,
                                 lnQ_rsig=rsig(lnQ[s]) if s.sum() > 20 else None))
            o['vs_view_angle'][nm] = rows
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for det in DETS:
        key, w, R, n_one = singles(det)
        g = gate(w, R); g['run_key'] = key; g['n_detected_interior'] = n_one
        g['toy'] = toy(w)
        res[det] = g
        print(f"{det} {key:14s} n {g['n']:5d}  lq {g['lq']['median']:+.3f} sigma {g['lq']['rsig']:.4f}+-{g['lq']['rsig_err']:.4f}"
              f"  u50 {g['u50']['rsig']:.0f}  u90 {g['u90']['rsig']:.0f} ns  lnQ {g['lnQ']['rsig']:.3f}"
              f"  toy q {g['toy']['right']['q']:.3f} q+u {g['toy']['right']['q_u']:.3f}", flush=True)
    w = satstudy_singles()
    g = gate(w); g['toy'] = toy(w)
    res['det3_satstudyS'] = g
    print(f"det3 class-S check n {g['n']} lq sigma {g['lq']['rsig']:.4f}  toy q {g['toy']['right']['q']:.3f} "
          f"q+u {g['toy']['right']['q_u']:.3f}")
    json.dump(res, open(os.path.join(OUT, 'gates_data.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
