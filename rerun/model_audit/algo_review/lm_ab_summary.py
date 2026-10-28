#!/usr/bin/env python3
"""lm_ab_summary.py <rows.pkl> ... -- tables for lm_ab.py: per chamber and view, each arm's angle vs M3 (s68 after
the median, far > 5 deg), the paired change against its NM arm (same w, |dtan|, dchi2, dt0), errors, cost.
Writes results/lm_ab_summary.json and prints a markdown table."""
import os, sys, json, pickle
import numpy as np

PAIRS = [('nm', 'lm'), ('nm', 'lm_nohop'), ('nm_free', 'lm_free'), ('nm_free', 'lm_free_nohop'), ('nm', 'lm_grid')]


def s68(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(np.abs(x), 68)) if len(x) else np.nan


def arm_stats(R, a):
    R = [r for r in R if np.isfinite(r.get(f'{a}_tan', np.nan)) and np.isfinite(r['tan_ref'])]
    if not R:
        return None
    tr = np.array([r['tan_ref'] for r in R])
    d = np.degrees(np.arctan([r[f'{a}_tan'] for r in R])) - np.degrees(np.arctan(tr))
    d = d - np.median(d)
    keep = np.abs(tr) < 0.6
    return dict(n=int(keep.sum()), s68=s68(d[keep]), far5=float(100 * np.mean(np.abs(d[keep]) > 5)),
                chi2dof=float(np.median([r[f'{a}_chi2'] / max(r[f'{a}_dof'], 1) for r in R])),
                nsolve=float(np.median([r[f'{a}_nsolve'] for r in R])),
                time=float(np.median([r[f'{a}_time'] for r in R])),
                p0_err=float(np.nanmedian([r[f'{a}_p0_err'] for r in R])),
                w_err=float(np.nanmedian([r[f'{a}_w_err'] for r in R])),
                t0_err=float(np.nanmedian([r[f'{a}_t0_err'] for r in R])),
                tan_err=float(np.nanmedian([r[f'{a}_tan_err'] for r in R])),
                t0_vs_pred_p50=float(np.nanmedian(np.abs([r[f'{a}_t0'] - r.get('t0_pred', np.nan) for r in R]))),
                t0_vs_pred_gt30=float(np.nanmean(np.abs([r[f'{a}_t0'] - r.get('t0_pred', np.nan) for r in R]) > 30)))


def pair_stats(R, a, b):
    R = [r for r in R if np.isfinite(r.get(f'{a}_w', np.nan)) and np.isfinite(r.get(f'{b}_w', np.nan))]
    if not R:
        return None
    g = lambda k: np.array([r[k] for r in R], float)
    dw = np.abs(g(f'{b}_w') - g(f'{a}_w')) * 1e3
    dth = np.abs(np.degrees(np.arctan(g(f'{b}_tan'))) - np.degrees(np.arctan(g(f'{a}_tan'))))
    dchi = g(f'{b}_chi2') - g(f'{a}_chi2')
    dt0 = g(f'{b}_t0') - g(f'{a}_t0')
    # position referee: p0 against the M3 point at the mesh (offset removed per arm by the median); on the planes
    # where the two arms disagree by more than 0.3 mm (a lattice hop), which one is closer to M3?
    rm = g('ref_mesh') if all('ref_mesh' in r for r in R) else np.full(len(R), np.nan)
    # the M3 frame is not the strip frame: a linear map ref -> p0, fitted on arm A with a 3-sigma clip, for both
    pa_, pb_ = g(f'{a}_p0'), g(f'{b}_p0')
    fin = np.isfinite(rm) & np.isfinite(pa_)
    if fin.sum() > 10:
        sel = fin.copy()
        for _ in range(3):
            c = np.polyfit(rm[sel], pa_[sel], 1)
            res = pa_ - np.polyval(c, rm)
            sel = fin & (np.abs(res) < 3 * 1.4826 * np.nanmedian(np.abs(res[sel])))
        ra, rb = pa_ - np.polyval(c, rm), pb_ - np.polyval(c, rm)
    else:
        ra, rb = np.full(len(R), np.nan), np.full(len(R), np.nan)
    hop = (np.abs(g(f'{b}_p0') - g(f'{a}_p0')) > 0.3) & np.isfinite(rm)
    tp = g('t0_pred') if all('t0_pred' in r for r in R) else np.full(len(R), np.nan)
    hop_t = (np.abs(g(f'{b}_t0') - g(f'{a}_t0')) > 30) & np.isfinite(tp)
    if np.isfinite(tp).any():
        # the bundle's t0 prediction and the fit's t0 differ by a constant (the model's t0 definition, F5): referee
        # each arm against the prediction plus arm A's median offset
        tp = tp + np.nanmedian(g(f'{a}_t0') - tp)
    return dict(n_hop_p0=int(hop.sum()),
                b_closer_m3=float(np.mean(np.abs(rb[hop]) < np.abs(ra[hop]))) if hop.any() else np.nan,
                pos_s68_a=s68(ra), pos_s68_b=s68(rb),
                n_hop_t0=int(hop_t.sum()),
                b_closer_trigger=float(np.mean(np.abs(g(f'{b}_t0') - tp)[hop_t] < np.abs(g(f'{a}_t0') - tp)[hop_t]))
                if hop_t.any() else np.nan,
                n=len(R), same_w=float(np.mean(dw < 1.0)), dth_p50=float(np.median(dth)),
                dth_p95=float(np.percentile(dth, 95)), dth_gt1=float(np.mean(dth > 1.0)),
                dchi_p50=float(np.median(dchi)), better=float(np.mean(dchi < -1)), worse=float(np.mean(dchi > 1)),
                t0_shift60=float(np.mean(np.abs(dt0) > 30)), dp0_p50=float(np.median(np.abs(g(f'{b}_p0') - g(f'{a}_p0')))),
                cost=float(np.median(g(f'{b}_nsolve')) / max(np.median(g(f'{a}_nsolve')), 1)),
                time=float(np.median(g(f'{b}_time')) / max(np.median(g(f'{a}_time')), 1e-9)))


def main():
    rows = [r for fn in sys.argv[1:] for r in pickle.load(open(fn, 'rb'))]
    out = {}
    for det in sorted({r['det'] for r in rows}):
        for plane in 'xy':
            R = [r for r in rows if r['det'] == det and r['plane'] == plane]
            arms = sorted({k[:-4] for r in R for k in r if k.endswith('_tan') and k != 'tan_ref'})
            out[f'{det}_{plane}'] = dict(arms={a: arm_stats(R, a) for a in arms},
                                          pairs={f'{a}>{b}': pair_stats(R, a, b) for a, b in PAIRS
                                                 if a in arms and b in arms})
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lm_ab_summary.json')
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(out)
    json.dump(old, open(fn, 'w'), indent=1)
    print('| row | pair | n | same w | dθ p50 / p95 (°) | dθ>1° | Δχ² p50 | better/worse | t0 moved >30 ns | '
          'solves ratio | time ratio | s68 A→B (°) | far5 A→B (%) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for key, v in out.items():
        for pk, p in v['pairs'].items():
            if p is None:
                continue
            a, b = pk.split('>')
            A, B = v['arms'][a], v['arms'][b]
            print(f"| {key} | {pk} | {p['n']} | {p['same_w']:.1%} | {p['dth_p50']:.3f} / {p['dth_p95']:.3f} | "
                  f"{p['dth_gt1']:.1%} | {p['dchi_p50']:+.2f} | {p['better']:.0%}/{p['worse']:.0%} | "
                  f"{p['t0_shift60']:.1%} | {p['cost']:.2f} | {p['time']:.2f} | {A['s68']:.3f}→{B['s68']:.3f} | "
                  f"{A['far5']:.1f}→{B['far5']:.1f} |")
    print()
    print('| row | pair | p0 hops (> 0.3 mm) | B closer to M3 | position s68 A→B (mm) | t0 hops (> 30 ns) | '
          'B closer to trigger |')
    print('|---|---|---|---|---|---|---|')
    for key, v in out.items():
        for pk, p in v['pairs'].items():
            if p is not None:
                print(f"| {key} | {pk} | {p['n_hop_p0']} | {p['b_closer_m3']:.0%} | {p['pos_s68_a']:.3f}→"
                      f"{p['pos_s68_b']:.3f} | {p['n_hop_t0']} | {p['b_closer_trigger']:.0%} |")
    print()
    print('| row | arm | s68 (°) | far5 (%) | χ²/dof | solves | time (s) | p0_err (mm) | w_err (µm/ns) | t0_err (ns) | '
          'tan_err | t0 − trigger p50 (ns) | > 30 ns |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for key, v in out.items():
        for a, A in v['arms'].items():
            if A:
                print(f"| {key} | {a} | {A['s68']:.3f} | {A['far5']:.1f} | {A['chi2dof']:.1f} | {A['nsolve']:.0f} | "
                      f"{A['time']:.2f} | {A['p0_err']:.4f} | {A['w_err'] * 1e3:.4f} | {A['t0_err']:.2f} | "
                      f"{A['tan_err']:.4f} | {A['t0_vs_pred_p50']:.1f} | {A['t0_vs_pred_gt30']:.1%} |")


if __name__ == '__main__':
    main()
