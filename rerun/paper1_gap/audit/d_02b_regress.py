"""D part 2b: regression decomposition of the map correction.  Delta_i = M_i(map) - M_i(const) in deg (M3 axis).
Real chamber distortion: r_i(const) = M_i - m3 regresses on Delta_i with slope -1, and d(const) = M6 - M7 regresses on
Delta_6 with slope -1 and on Delta_7 with slope +1.  Telescope-bound: slope of r_i is -1 but slopes of d are 0."""
import json, sys
sys.path.insert(0, '.')
from d_02_part2 import *

def ols(y, X):
    X = np.c_[X]
    return np.linalg.lstsq(X, y, rcond=None)[0]

def fit(pooled, tests, idxs, name):
    cat = lambda f: np.concatenate([f(t)[i] for t, i in zip(tests, idxs)])
    out = {}
    for ax, a in (('x', 0), ('y', 1)):
        g = lambda det, nm, t: pooled[t]['arrs'][(det, nm)][a]
        m = lambda t: pooled[t]['m3'][a]
        r6 = cat(lambda t: ddeg_arr(g('det6', 'const', t), m(t)))
        r7 = cat(lambda t: ddeg_arr(g('det7', 'const', t), m(t)))
        d = cat(lambda t: ddeg_arr(g('det6', 'const', t), g('det7', 'const', t)))
        D6 = cat(lambda t: ddeg_arr(g('det6', name, t), g('det6', 'const', t)))
        D7 = cat(lambda t: ddeg_arr(g('det7', name, t), g('det7', 'const', t)))
        ok = np.isfinite(D6) & np.isfinite(D7) & np.isfinite(r6) & np.isfinite(r7)
        r6, r7, d, D6, D7 = r6[ok], r7[ok], d[ok], D6[ok], D7[ok]
        c6 = ols(r6 - r6.mean(), D6 - D6.mean())[0]
        c7 = ols(r7 - r7.mean(), D7 - D7.mean())[0]
        cd = ols(d - d.mean(), np.c_[D6 - D6.mean(), D7 - D7.mean()])
        out[ax] = dict(b_r6=c6, b_r7=c7, b_d6=cd[0], b_d7=cd[1], rms_D6=D6.std(), rms_D7=D7.std(), corr_D6_D7=np.corrcoef(D6, D7)[0, 1],
                       corr_r6_D6=np.corrcoef(r6, D6)[0, 1], corr_d_D6=np.corrcoef(d, D6)[0, 1], corr_d_D7=np.corrcoef(d, D7)[0, 1])
    return out

if __name__ == '__main__':
    res = {}
    for xhalf in (False, True):
        r, pooled = run(xhalf)
        tests = ['long', 'longer']
        n = {t: len(pooled[t]['m3'][0]) for t in tests}
        for name in ('w0+v6', 'w0+v8', 'w06', 'w08'):
            pt = fit(pooled, tests, [np.arange(n[t]) for t in tests], name)
            rng = np.random.default_rng(9)
            B = [fit(pooled, tests, [rng.integers(0, n[t], n[t]) for t in tests], name) for _ in range(300)]
            o = {}
            for ax in 'xy':
                o[ax] = {k: (float(pt[ax][k]), float(np.std([b[ax][k] for b in B]))) for k in pt[ax]}
            res[f"{'xhalf' if xhalf else 'full'}|{name}"] = o
            print('xhalf' if xhalf else 'full', name, {ax: {k: '%.2f±%.2f' % v for k, v in o[ax].items() if k.startswith('b_')} for ax in 'xy'}, flush=True)
    json.dump(res, open(f'{OUT}/part2b.json', 'w'), indent=1)
