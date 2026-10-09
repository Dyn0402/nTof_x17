#!/usr/bin/env python3
"""compare_bench.py -- paired, scale-corrected comparison of plane_bench arms.

A kernel that reads every angle a few % flat shrinks the residual spread by
the same few % without resolving anything better; production removes the
scale afterwards with kw.  So besides the raw s68 this reports s68 after
dividing each arm's fitted tan by its own fitted slope (tan_fit vs tan_ref on
the core) -- the spread at the right scale.  Paired bootstrap on the common
events, negative = arm better.

    compare_bench.py bench_det3_rc.json [...] [--start blind]
"""
import argparse, json, os
import numpy as np

LT5 = np.tan(np.radians(5.0))


def arrays(ev):
    a = np.array(ev, float)
    return {int(e): (r, f) for e, r, f in a[:, :3]}


def stats(tr, tf, scale=True):
    core = np.abs(tf - tr) < 0.15
    k = np.polyfit(tr[core], tf[core], 1)[0] if scale else 1.0
    t = tf / k
    core = np.abs(t - tr) < 0.15
    d = np.degrees(np.arctan(t[core])) - np.degrees(np.arctan(tr[core]))
    h = np.abs(tr[core]) < LT5
    s = lambda x: np.percentile(np.abs(x - np.median(x)), 68.27)
    return np.array([s(d), s(d[h]), np.median(d), k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--start', default='blind')
    ap.add_argument('--nboot', type=int, default=500)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261009)
    out = {}
    for f in a.files:
        r = json.load(open(f))
        prod = r['production']['events'][a.start]
        for arm in [k for k in r if k != 'production']:
            ev = r[arm]['events'][a.start]
            for p in ('x', 'y'):
                P, A = arrays(prod[p]), arrays(ev[p])
                common = sorted(set(P) & set(A))
                Pa = np.array([P[e] for e in common]); Aa = np.array([A[e] for e in common])
                row = {}
                for lab, sc in (('raw', False), ('scaled', True)):
                    sp, sa = stats(*Pa.T, scale=sc), stats(*Aa.T, scale=sc)
                    n = len(common)
                    bs = []
                    for _ in range(a.nboot):
                        i = rng.integers(0, n, n)
                        bs.append(stats(*Aa[i].T, scale=sc) - stats(*Pa[i].T, scale=sc))
                    e = np.std(bs, axis=0)
                    row[lab] = dict(prod=sp.tolist(), arm=sa.tolist(), d=(sa - sp).tolist(), err=e.tolist())
                out[f'{arm}_{p}'] = dict(n=len(common), **row)
                q = row['scaled']; qr = row['raw']
                print(f'{arm:11s} {p} n={len(common):4d} | scaled s68 {q["prod"][0]:.3f}->{q["arm"][0]:.3f} '
                      f'({q["d"][0]:+.3f}±{q["err"][0]:.3f})  head {q["prod"][1]:.3f}->{q["arm"][1]:.3f} '
                      f'({q["d"][1]:+.3f}±{q["err"][1]:.3f}) | raw all {qr["d"][0]:+.3f}±{qr["err"][0]:.3f} '
                      f'head {qr["d"][1]:+.3f}±{qr["err"][1]:.3f} | slope {q["prod"][3]:.3f}->{q["arm"][3]:.3f} '
                      f'bias {qr["prod"][2]:+.3f}->{qr["arm"][2]:+.3f}')
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
