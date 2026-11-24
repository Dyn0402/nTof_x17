"""D part 2c: same budget as part 2 but with the (clipped) variance instead of s68^2, which is only additive for Gaussians."""
import json, sys
sys.path.insert(0, '.')
from d_02_part2 import *
CLIP = 5.0

def stat(pooled, tests, idxs, name, both=True):
    cat = lambda f: np.concatenate([f(t)[i] for t, i in zip(tests, idxs)])
    o = {}
    for ax, a in (('x', 0), ('y', 1)):
        g = lambda det, nm, t: pooled[t]['arrs'][(det, nm)][a]
        m = lambda t: pooled[t]['m3'][a]
        r6c = cat(lambda t: ddeg_arr(g('det6', 'const', t), m(t))); r7c = cat(lambda t: ddeg_arr(g('det7', 'const', t), m(t)))
        keep = (np.abs(r6c) < CLIP) & (np.abs(r7c) < CLIP)
        for nm in ('const', name):
            pass
        def vals(n6, n7):
            r6 = cat(lambda t: ddeg_arr(g('det6', n6, t), m(t))); r7 = cat(lambda t: ddeg_arr(g('det7', n7, t), m(t)))
            k = keep & np.isfinite(r6) & np.isfinite(r7)
            return r6[k], r7[k], k
        r6c_, r7c_, kc = vals('const', 'const')
        res = {}
        for cfg, (n6, n7) in {'both': (name, name), 'det6 only': (name, 'const'), 'det7 only': ('const', name)}.items():
            r6, r7, k = vals(n6, n7)
            # same event set for const: recompute const on k
            r6c2 = r6c[k]; r7c2 = r7c[k]
            d_c = r6c2 - r7c2; d_m = r6 - r7
            fall6 = r6c2.var() - r6.var(); fall7 = r7c2.var() - r7.var()
            res[cfg] = dict(obs=d_c.var() - d_m.var(), pred=fall6 + fall7, f6=fall6, f7=fall7, var_d_const=d_c.var(), var_r6=r6c2.var(), var_r7=r7c2.var(),
                            rms_over_s68_r6=r6c2.std() / s68(r6c2), rms_over_s68_d=d_c.std() / s68(d_c), n=int(k.sum()))
        o[ax] = res
    return o

if __name__ == '__main__':
    out = {}
    for xhalf in (False, True):
        r, pooled = run(xhalf)
        tests = ['long', 'longer']
        n = {t: len(pooled[t]['m3'][0]) for t in tests}
        for name in ('w0+v6', 'w0+v8', 'w06', 'w08'):
            pt = stat(pooled, tests, [np.arange(n[t]) for t in tests], name)
            rng = np.random.default_rng(3)
            B = [stat(pooled, tests, [rng.integers(0, n[t], n[t]) for t in tests], name) for _ in range(300)]
            o = {}
            for ax in 'xy':
                o[ax] = {}
                for cfg in pt[ax]:
                    o[ax][cfg] = {k: (float(v), float(np.std([b[ax][cfg][k] for b in B]))) for k, v in pt[ax][cfg].items()}
                    o[ax][cfg]['diff'] = (float(pt[ax][cfg]['obs'] - pt[ax][cfg]['pred']), float(np.std([b[ax][cfg]['obs'] - b[ax][cfg]['pred'] for b in B])))
            out[f"{'xhalf' if xhalf else 'full'}|{name}"] = o
            print('xhalf' if xhalf else 'full', name, {ax: {c: ('obs %.2f pred %.2f diff %.2f±%.2f' % (o[ax][c]['obs'][0], o[ax][c]['pred'][0], *o[ax][c]['diff'])) for c in o[ax]} for ax in 'xy'}, flush=True)
    json.dump(out, open(f'{OUT}/part2c.json', 'w'), indent=1)
