"""D part 2: M3-free validation with det6 + det7 (same muons).  d = tan(det6) - tan(det7) in the same M3 axis."""
import json, sys
sys.path.insert(0, '.')
from d_lib import *
from scipy.interpolate import RegularGridInterpolator  # noqa
NB = 500
SUBS = ['long', 'longer', 'short']
TESTS = [('long', 'longer'), ('longer', 'long'), ('short', 'long')]   # (test sub-run, train sub-run)
MAPS = [('w0+v', 6), ('w0+v', 8), ('w0', 6), ('w0', 8)]


def arms_full(train_tab, d, p, n, ul):
    """tan of ALL events of table d for view p: const / maps (train = table of another sub-run), constants on |tan_ref|<=0.6."""
    U = d['Urec'].values; W = d['Wrec'].values; w = d[f'{p}_w'].values * 1e3; tr = d[f'tan_ref_{p}'].values
    s = np.abs(tr) <= GM.TAN_LADDER
    out = {'p1': d[f'{p}_tan_theta'].values}
    c = TT.robust_const(tr[s], w[s], 0.0, 1.0); out['const'] = (w - c[0]) / c[1]
    if train_tab is not None:
        Uo, Wo, wo, tro, _ = view(train_tab, p)
        C = cells(Uo, Wo, wo, tro, n, ul)
        if np.sum(np.isfinite([c['v'] for c in C])) < 4:
            out['w0+v'] = out['w0'] = np.full(len(U), np.nan)
            return out
        vr = TT.bilinear(C, 'vrel', U, W); dw = TT.bilinear(C, 'dw0', U, W)
        for arm, (dd, vv) in {'w0+v': (dw, vr), 'w0': (dw, np.ones_like(vr))}.items():
            c = TT.robust_const(tr[s], w[s], dd[s], vv[s])
            out[arm] = (w - c[0] - dd) / (c[1] * vv)
    return out


def rot(tx, ty, th):
    t = np.radians(th)
    return np.cos(t) * tx - np.sin(t) * ty, np.sin(t) * tx + np.cos(t) * ty


def build(test, train, xhalf):
    """matched events of the test sub-run: dict config -> (M6x, M6y), (M7x, M7y); plus reference."""
    D = {}
    T = {}
    for det in ('det6', 'det7'):
        k, kt = f'{det}:{test}', f'{det}:{train}'
        xh = xhalf and det == 'det7'
        d, dt = load(k, xh), load(kt, xh)
        ul = (0.0, 1.7) if xh else UL
        d = d.set_index(['acq_tag', 'event_id'], drop=False)
        T[det] = d
        D[det] = {}
        for n in (6, 8):
            for p in 'xy':
                A = arms_full(dt, d, p, n, ul)
                for arm, v in A.items():
                    D[det][(arm, n if arm in ('w0+v', 'w0') else 0, p)] = v
    idx = T['det6'].index.intersection(T['det7'].index)
    return D, T, idx


def run(xhalf):
    res = {'tests': {}}
    pooled = {}
    for test, train in TESTS:
        D, T, idx = build(test, train, xhalf)
        a6, a7 = T['det6'].loc[idx], T['det7'].loc[idx]
        pos6 = T['det6'].index.get_indexer(idx); pos7 = T['det7'].index.get_indexer(idx)
        th6, th7 = a6.theta_align_deg.values, a7.theta_align_deg.values
        # reference check
        m6 = np.c_[a6.m3_tan_x, a6.m3_tan_y]; m7 = np.c_[a7.m3_tan_x, a7.m3_tan_y]
        r6x, r6y = rot(a6.tan_ref_x.values, a6.tan_ref_y.values, th6)
        r7x, r7y = rot(a7.tan_ref_x.values, a7.tan_ref_y.values, th7)
        chk = dict(n_matched=len(idx), n6=len(T['det6']), n7=len(T['det7']),
                   max_dm3=float(np.abs(m6 - m7).max()),
                   corr_m3x=float(np.corrcoef(m6[:, 0], m7[:, 0])[0, 1]), corr_m3y=float(np.corrcoef(m6[:, 1], m7[:, 1])[0, 1]),
                   max_rot6=float(max(np.abs(r6x - m6[:, 0]).max(), np.abs(r6y - m6[:, 1]).max())),
                   max_rot7=float(max(np.abs(r7x - m7[:, 0]).max(), np.abs(r7y - m7[:, 1]).max())),
                   corr_tanref_x_chamberframe=float(np.corrcoef(a6.tan_ref_x, a7.tan_ref_x)[0, 1]),
                   rms_diff_tanref_x_chamberframe=float(np.std(a6.tan_ref_x.values - a7.tan_ref_x.values)),
                   rms_diff_tanref_y_chamberframe=float(np.std(a6.tan_ref_y.values - a7.tan_ref_y.values)),
                   rms_diff_after_rot_x=float(np.std(r6x - r7x)), th6=float(th6[0]), th7=float(th7[0]))
        # |tan_ref| <= 0.6 in both views of both chambers
        ok = (np.abs(a6.tan_ref_x) <= .6) & (np.abs(a6.tan_ref_y) <= .6) & (np.abs(a7.tan_ref_x) <= .6) & (np.abs(a7.tan_ref_y) <= .6)
        ok = ok.values
        chk['n_used'] = int(ok.sum())
        res['tests'][test] = chk
        print(test, 'train', train, chk, flush=True)

        def M(det, key, pos, th):
            tx = D[det][(key[0], key[1], 'x')][pos]; ty = D[det][(key[0], key[1], 'y')][pos]
            return rot(tx, ty, th)
        m3 = (a6.m3_tan_x.values[ok], a6.m3_tan_y.values[ok])
        keys = {'p1': ('p1', 0), 'const': ('const', 0)}
        for arm, n in MAPS:
            keys[f'{arm}{n}'] = (arm, n)
        arrs = {}
        for name, key in keys.items():
            for det, pos, th in (('det6', pos6, th6), ('det7', pos7, th7)):
                Mx, My = M(det, key, pos, th)
                arrs[(det, name)] = (Mx[ok], My[ok])
        pooled[test] = dict(arrs=arrs, m3=m3)
    return res, pooled


def stats(pooled, tests, idxs):
    """configs -> {s68 d (x, y), s68 r6, r7} on resampled (idxs: dict test -> index array)."""
    names = [n for n in pooled[tests[0]]['arrs'] if n[0] == 'det6']
    names = [n[1] for n in names]
    out = {}
    def cat(f):
        return np.concatenate([f(t)[i] for t, i in zip(tests, idxs)], axis=-1)
    configs = {}
    for nm in names:
        configs[f'{nm}|{nm}'] = (nm, nm)
    for arm in [k for k in names if k.startswith('w0')]:
        configs[f'const|{arm}'] = ('const', arm)       # det6 const, det7 mapped
        configs[f'{arm}|const'] = (arm, 'const')       # det6 mapped, det7 const
    for cname, (n6, n7) in configs.items():
        o = {}
        for ax, a in (('x', 0), ('y', 1)):
            d = cat(lambda t: ddeg_arr(pooled[t]['arrs'][('det6', n6)][a], pooled[t]['arrs'][('det7', n7)][a]))
            e6 = cat(lambda t: ddeg_arr(pooled[t]['arrs'][('det6', n6)][a], pooled[t]['m3'][a]))
            e7 = cat(lambda t: ddeg_arr(pooled[t]['arrs'][('det7', n7)][a], pooled[t]['m3'][a]))
            o[ax] = (s68(d), s68(e6), s68(e7))
        out[cname] = o
    return out


def ddeg_arr(a, b):
    return np.degrees(np.arctan(a) - np.arctan(b))


def summarize(pooled, tests):
    n = {t: len(pooled[t]['m3'][0]) for t in tests}
    full = stats(pooled, tests, [np.arange(n[t]) for t in tests])
    rng = np.random.default_rng(5)
    B = []
    for _ in range(NB):
        B.append(stats(pooled, tests, [rng.integers(0, n[t], n[t]) for t in tests]))
    res = {}
    for c in full:
        res[c] = {}
        for ax in 'xy':
            res[c][ax] = {}
            for j, nm in enumerate(('d', 'r6', 'r7')):
                bs = np.array([b[c][ax][j] for b in B])
                res[c][ax][nm] = (full[c][ax][j], float(bs.std()))
    # predictions: s68^2(d) falls by the sum of the falls of the chambers' s68^2 against M3
    pred = {}
    for c in full:
        n6, n7 = c.split('|')
        if (n6, n7) == ('const', 'const'):
            continue
        pred[c] = {}
        for ax in 'xy':
            def val(src, cc, j):
                return src[cc][ax][j]
            def quantities(src):
                d_c = src['const|const'][ax][0] ** 2
                f6 = (src['const|const'][ax][1] ** 2 - src[c][ax][1] ** 2) if n6 != 'const' else 0.0
                f7 = (src['const|const'][ax][2] ** 2 - src[c][ax][2] ** 2) if n7 != 'const' else 0.0
                obs = src[c][ax][0] ** 2
                return obs - d_c, -(f6 + f7), f6, f7
            full_q = quantities(full)
            bq = np.array([quantities(b) for b in B])
            pred[c][ax] = dict(delta_obs=(full_q[0], float(bq[:, 0].std())), delta_pred=(full_q[1], float(bq[:, 1].std())),
                               fall6=(full_q[2], float(bq[:, 2].std())), fall7=(full_q[3], float(bq[:, 3].std())),
                               diff=(full_q[0] - full_q[1], float((bq[:, 0] - bq[:, 1]).std())))
    return dict(s68=res, pred=pred, n=n)


if __name__ == '__main__':
    out = {}
    for xhalf in (False, True):
        r, pooled = run(xhalf)
        r['main'] = summarize(pooled, ['long', 'longer'])
        r['with_short'] = summarize(pooled, ['long', 'longer', 'short'])
        out['xhalf' if xhalf else 'full'] = r
        for c, o in r['main']['s68'].items():
            print('xhalf' if xhalf else 'full', c, {ax: tuple(round(v[0], 3) for v in o[ax].values()) for ax in 'xy'}, flush=True)
    json.dump(out, open(f'{OUT}/part2.json', 'w'), indent=1, default=float)
