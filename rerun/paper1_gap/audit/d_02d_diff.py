"""D part 2d: split the map corrections into the part common to det6 and det7 (Ds = (D6+D7)/2) and the part that differs
(Da = (D6-D7)/2).  d = M6 - M7 sees only 2 Da.  Real chamber-specific distortions: slope of d(const) on (D6-D7) = -1.
Telescope-bound or same-in-both distortions: the common part Ds cancels in d; slope of rs=(r6+r7)/2 on Ds shows how much
of the correction is common to both chambers (and so cannot be tested with d)."""
import json, sys
sys.path.insert(0, '.')
from d_02_part2 import *

def sl(y, x):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (x * x).sum())

def fit(pooled, tests, idxs, name):
    cat = lambda f: np.concatenate([f(t)[i] for t, i in zip(tests, idxs)])
    o = {}
    for ax, a in (('x', 0), ('y', 1)):
        g = lambda det, nm, t: pooled[t]['arrs'][(det, nm)][a]
        m = lambda t: pooled[t]['m3'][a]
        r6 = cat(lambda t: ddeg_arr(g('det6', 'const', t), m(t))); r7 = cat(lambda t: ddeg_arr(g('det7', 'const', t), m(t)))
        D6 = cat(lambda t: ddeg_arr(g('det6', name, t), g('det6', 'const', t))); D7 = cat(lambda t: ddeg_arr(g('det7', name, t), g('det7', 'const', t)))
        ok = np.isfinite(D6) & np.isfinite(D7); r6, r7, D6, D7 = r6[ok], r7[ok], D6[ok], D7[ok]
        d = r6 - r7; Da = 0.5 * (D6 - D7); Ds = 0.5 * (D6 + D7); rs = 0.5 * (r6 + r7)
        o[ax] = dict(b_d_on_2Da=sl(d, 2 * Da), b_rs_on_Ds=sl(rs, Ds), b_ra_on_Da=sl(0.5 * (r6 - r7), Da),
                     rms_Ds=float(Ds.std()), rms_Da=float(Da.std()), frac_var_common=float(Ds.var() / (Ds.var() + Da.var())))
    return o

if __name__ == '__main__':
    res = {}
    for xhalf in (False, True):
        r, pooled = run(xhalf)
        tests = ['long', 'longer']; n = {t: len(pooled[t]['m3'][0]) for t in tests}
        for name in ('w0+v6', 'w0+v8', 'w06', 'w08'):
            pt = fit(pooled, tests, [np.arange(n[t]) for t in tests], name)
            rng = np.random.default_rng(13)
            B = [fit(pooled, tests, [rng.integers(0, n[t], n[t]) for t in tests], name) for _ in range(300)]
            o = {ax: {k: (pt[ax][k], float(np.std([b[ax][k] for b in B]))) for k in pt[ax]} for ax in 'xy'}
            res[f"{'xhalf' if xhalf else 'full'}|{name}"] = o
            print('xhalf' if xhalf else 'full', name, {ax: {k: '%.2f±%.2f' % v for k, v in o[ax].items()} for ax in 'xy'}, flush=True)
    json.dump(res, open(f'{OUT}/part2d.json', 'w'), indent=1)
