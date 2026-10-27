#!/usr/bin/env python3
"""heldout_compare.py --dir DIR --base base --arms t0i mf03 ... [--dets det2 det3 det4 det6 det7] [--json out]

Event-paired comparison of heldout_arm.py outputs <dir>/<det>_<arm>.npz against <dir>/<det>_<base>.npz: per view the
ladder, s68 (|Δθ| < 5°, each arm on its own ladder), far %, median χ²/dof, the steep-bin bias (tan_ref >= +0.28 and
<= −0.28) and the x–y t0 coincidence width, arm − base with a bootstrap over events.  Markdown table + JSON.
"""
import argparse, json, os
import numpy as np


def load(path, vi):
    P = np.load(path)['per']; P = P[P[:, 1] == vi]
    return {int(r[0]): r for r in P}


def ladder(tr, w):
    m = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
    c = np.median(w[np.abs(tr) < 0.05]) if (np.abs(tr) < 0.05).sum() > 20 else 0.0
    return (float(np.median((w[m] - c) / tr[m])) if m.sum() > 30 else np.nan), c


def stats(R, s, c):
    tr, w, chi, dof = R[:, 2], R[:, 3], R[:, 4], R[:, 5]
    d = np.degrees(np.arctan((w - c) / s) - np.arctan(tr)); far = np.abs(d) > 5; k = ~far
    def s68(m):
        mm = m & k
        return float(np.percentile(np.abs(d[mm] - np.median(d[mm])), 68)) if mm.sum() > 10 else np.nan
    def bias(m):
        mm = m & k
        return float(np.median(d[mm])) if mm.sum() > 5 else np.nan
    at = np.abs(tr)
    return np.array([s68(at < 9), s68(at < 0.1), s68(at >= 0.1), 100 * far.mean(), np.median(chi / np.maximum(dof, 1)),
                     bias(tr >= 0.28), bias(tr <= -0.28), bias(at < 0.1)])


FIELDS = ['s68', 's68_headon', 's68_incl', 'far%', 'chi2dof', 'bias_steep+', 'bias_steep-', 'bias_headon']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--base', default='base')
    ap.add_argument('--arms', nargs='+', required=True); ap.add_argument('--dets', nargs='*', default=['det2', 'det3', 'det4', 'det6', 'det7'])
    ap.add_argument('--nboot', type=int, default=300); ap.add_argument('--json', default=None); ap.add_argument('--prefix', default='')
    a = ap.parse_args(); rng = np.random.default_rng(1); out = {}; lines = []
    hdr = '| det view arm | n | ladder | Δs68 all | Δs68 head-on | Δs68 incl | Δfar pp | χ²/dof | Δbias steep+ | Δbias steep− | Δt0xy rσ |'
    lines += [hdr, '|' + '---|' * 11]
    for det in a.dets:
        fb = os.path.join(a.dir, f'{a.prefix}{det}_{a.base}.npz')
        if not os.path.exists(fb):
            continue
        for arm in a.arms:
            ft = os.path.join(a.dir, f'{a.prefix}{det}_{arm}.npz')
            if not os.path.exists(ft):
                continue
            # x-y t0 width per arm (events with both views)
            def t0w(path):
                X, Y = load(path, 0), load(path, 1); ids = sorted(set(X) & set(Y))
                if len(ids) < 20:
                    return np.nan
                dt = np.array([X[e][8] - Y[e][8] for e in ids]); return float(1.4826 * np.median(np.abs(dt - np.median(dt))))
            wb, wt = t0w(fb), t0w(ft)
            for vi, view in enumerate('xy'):
                B, T = load(fb, vi), load(ft, vi); ids = sorted(set(B) & set(T))
                if len(ids) < 30:
                    continue
                RB = np.array([B[i] for i in ids]); RT = np.array([T[i] for i in ids])
                sb, cb = ladder(RB[:, 2], RB[:, 3]); st, ct = ladder(RT[:, 2], RT[:, 3])
                x0, x1 = stats(RB, sb, cb), stats(RT, st, ct); dd = x1 - x0
                bs = []
                for _ in range(a.nboot):
                    k = rng.integers(0, len(ids), len(ids))
                    bs.append(stats(RT[k], st, ct) - stats(RB[k], sb, cb))
                e = np.nanstd(np.array(bs), 0)
                out[f'{det}|{view}|{arm}'] = dict(n=len(ids), ladder=[sb, st], base=dict(zip(FIELDS, x0.tolist())), arm=dict(zip(FIELDS, x1.tolist())),
                                                 delta=dict(zip(FIELDS, dd.tolist())), err=dict(zip(FIELDS, e.tolist())), t0xy_rsig=[wb, wt])
                f = lambda i, fmt='+.3f': f'{dd[i]:{fmt}}±{e[i]:.3f}'
                lines.append(f'| {det} {view} {arm} | {len(ids)} | {sb:.2f}→{st:.2f} | {f(0)} | {f(1)} | {f(2)} | {dd[3]:+.1f}±{e[3]:.1f} | '
                             f'{x0[4]:.0f}→{x1[4]:.0f} | {f(5, "+.2f")} | {f(6, "+.2f")} | {wb:.0f}→{wt:.0f} |')
    md = '\n'.join(lines) + '\n'
    print(md)
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, default=float); open(a.json.replace('.json', '.md'), 'w').write(md)


if __name__ == '__main__':
    main()
