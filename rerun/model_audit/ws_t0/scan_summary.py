#!/usr/bin/env python3
"""scan_summary.py [--dir D] -- summarise the t0 latency scan (t0_scan.py outputs <det>_C<C>.npz).

Per chamber and view, for the free arm and every pinned latency constant C (t0 = C - 10 ns * ftst), on the events
fitted in EVERY arm (event-paired):
  chi2      sum of chi2 / sum of dof, relative to the free fit
  ladder    median (w - c) / tan_ref over 0.10 < |tan_ref| < 0.45 (c = head-on median), the reco angle scale [um/ns]
  s68, far  angle resolution (68 % of |dev - median| after the ladder) and |dev| > 5 deg fraction
  pos       robust sigma of p0 - ref_mesh (M3 extrapolated to the mesh) and its slope against tan_ref [mm per unit tan]
            (a mean t0 error dt moves the fitted mesh point along the track by w dt = tan v dt: slope = v dt)
Output: scan_summary.json beside this script, and the per-C table on stdout.
"""
import argparse, glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['eid', 'view', 'C', 'tan_ref', 'ref', 'w', 'p0', 't0', 'chi2', 'dof', 'uend', 'u50', 'sig', 'ftst']


def rsig(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan
    q = np.percentile(a, [16, 84]); return 0.5 * (q[1] - q[0])


def angle_stats(tr, w):
    m = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
    h = np.abs(tr) < 0.05
    c = np.median(w[h]) if h.sum() > 20 else 0.0
    s = float(np.median((w[m] - c) / tr[m]))
    d = np.degrees(np.arctan((w - c) / s) - np.arctan(tr)); far = np.abs(d) > 5
    k = ~far
    s68 = float(np.percentile(np.abs(d[k] - np.median(d[k])), 68))
    return s, s68, float(100 * far.mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='/home/dylan/x17/cosmic_bench/model_audit/ws_t0/scan')
    a = ap.parse_args()
    files = sorted(glob.glob(f'{a.dir}/det*_C*.npz'))
    by = {}
    for f in files:
        m = re.match(r'(det\d)_C(.+)\.npz', os.path.basename(f))
        per = np.load(f)['per']
        if per.size:
            by.setdefault(m.group(1), []).append(per)
    out = {}
    for det, parts in sorted(by.items()):
        P = np.concatenate(parts); out[det] = {}
        for vi, view in enumerate('xy'):
            V = P[P[:, 1] == vi]
            Cs = sorted(set(np.where(np.isnan(V[:, 2]), -1, V[:, 2])))
            # events present in every arm
            sets = [set(V[(np.isnan(V[:, 2]) if c == -1 else (V[:, 2] == c)), 0].astype(int)) for c in Cs]
            common = set.intersection(*sets)
            rows = {}
            for c in Cs:
                R = V[(np.isnan(V[:, 2]) if c == -1 else (V[:, 2] == c))]
                R = R[np.isin(R[:, 0].astype(int), list(common))]
                R = R[np.argsort(R[:, 0])]
                tr, w = R[:, 3], R[:, 5]
                lad, s68, far = angle_stats(tr, w)
                res = R[:, 6] - R[:, 4]
                ok = np.abs(res - np.median(res)) < 5
                cf = np.polyfit(tr[ok], res[ok], 1); slope = float(cf[0])
                # the same residual after re-fitting the mesh z for this arm (the linear tan term = a z shift):
                # the fair position resolution when each arm gets its own alignment
                rz = res - np.polyval(cf, tr)
                rows['free' if c == -1 else f'{c:g}'] = dict(
                    n=int(len(R)), chi2=float(R[:, 8].sum()), chi2dof=float(R[:, 8].sum() / R[:, 9].sum()),
                    ladder=lad, s68=s68, far=far, pos_rsig=float(rsig(res[ok])), pos_rsig_z=float(rsig(rz[ok])), pos_med=float(np.median(res[ok])),
                    pos_slope=slope, t0c_med=float(np.median(R[:, 7] + 10 * R[:, 13])),
                    uend_med=float(np.median(R[:, 10])), wide_frac=float(np.mean(R[:, 12] > 0.41)))
            ref = rows['free']['chi2']
            for k, r in rows.items():
                r['dchi2_rel'] = r['chi2'] / ref - 1.0
            out[det][view] = rows
            print(f'== {det} {view}  n={len(common)}')
            for k, r in rows.items():
                print(f'  C={k:>5s} chi2 {100 * r["dchi2_rel"]:+6.2f} %  ladder {r["ladder"]:6.2f}  s68 {r["s68"]:5.2f}  far {r["far"]:4.1f}  '
                      f'pos rsig {r["pos_rsig"]:.3f} z-refit {r["pos_rsig_z"]:.3f} med {r["pos_med"]:+.3f} slope {r["pos_slope"]:+.2f}  t0c {r["t0c_med"]:6.1f}  '
                      f'uend {r["uend_med"]:6.0f}  wide {r["wide_frac"]:.2f}')
    json.dump(out, open(os.path.join(HERE, 'scan_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
