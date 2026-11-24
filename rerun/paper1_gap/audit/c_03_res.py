import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from c_lib import *
D = {k: load(rel) for k, rel in ROWS.items()}
V = {(k, p): view(D[k], p) for k in D for p in 'xy'}
def pool(keys, p):
    return {q: np.concatenate([V[(k, p)][q] for k in keys]) for q in ('U', 'W', 'w', 'tr')}
def dang(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))
def angle(Vt, mp, mode, rate=False):
    """tan from the maps: mode in w0+v, w0, v, plane, fine"""
    ap, af, s, _ = mp.ev(Vt['U'], Vt['W'])
    c0 = TT.robust_const(Vt['tr'], Vt['w'], 0.0, 1.0)
    vp = c0[1]
    scale = (mp.vg if rate else vp)
    A = {'w0+v': ap + af, 'w0': ap + af, 'v': 0 * ap, 'plane': ap, 'fine': af}[mode]
    vr = (1 + s) if mode in ('w0+v', 'v') else np.ones_like(s)
    dd = scale * A
    c = TT.robust_const(Vt['tr'], Vt['w'], dd, vr)
    return (Vt['w'] - c[0] - dd) / (c[1] * vr)
def s68b(d, nb=200, seed=5):
    return TT.s68(d)
rng = np.random.default_rng(11)
out = {}
NBS = 200
for p in 'xy':
    T = V[('r625', p)]; n_ev = len(T['U'])
    bidx = [rng.integers(0, n_ev, n_ev) for _ in range(NBS)]
    arms = {'pass1': T['tp']}
    c = TT.robust_const(T['tr'], T['w'], 0.0, 1.0); arms['const'] = (T['w'] - c[0]) / c[1]
    G = pool(['run1', 'run2'], p)
    for n in (4, 6, 8):
        mp = Maps(G['U'], G['W'], G['w'], G['tr'], n)
        for mode in ('w0+v', 'w0', 'v', 'plane', 'fine'):
            arms[f'gold{n} {mode}'] = angle(T, mp, mode)
        arms[f'gold{n} w0 (fixed rate)'] = angle(T, mp, 'w0', rate=True)
        arms[f'gold{n} w0+v (fixed rate)'] = angle(T, mp, 'w0+v', rate=True)
        # ceiling: own 2-fold, 5 random splits
        for mode in ('w0+v', 'w0'):
            dl = []
            for sp in range(5):
                r = np.random.default_rng(100 + sp); half = r.random(n_ev) < 0.5
                t = np.zeros(n_ev)
                for tr_m, te_m in ((half, ~half), (~half, half)):
                    tr_d = {q: T[q][tr_m] for q in ('U', 'W', 'w', 'tr')}; te_d = {q: T[q][te_m] for q in ('U', 'W', 'w', 'tr')}
                    mo = Maps(tr_d['U'], tr_d['W'], tr_d['w'], tr_d['tr'], n)
                    t[te_m] = angle(te_d, mo, mode)
                dl.append(t)
            arms[f'own2fold{n} {mode}'] = dl   # list of 5 arrays
        # golden-run references within the golden mounting (run1 map -> run2 and vice versa), same arms
    res = {}
    d_p1 = dang(arms['pass1'], T['tr'])
    for name, t in arms.items():
        if isinstance(t, list):
            ds = [dang(x, T['tr']) for x in t]
            ms = [TT.metrics(x, T['tr'], T['U'], T['W']) for x in t]
            s = float(np.mean([m['s68'] for m in ms])); ss = float(np.mean([m['s68_steep'] for m in ms]))
            # paired bootstrap difference with the first split
            diffs = [np.mean([TT.s68(dd[b]) for dd in ds]) - TT.s68(d_p1[b]) for b in bidx[:100]]
            res[name] = dict(s68=s, s68_steep=ss, off_rms=float(np.mean([m['cell_offset_rms'] for m in ms])),
                             dif=float(s - TT.s68(d_p1)), dif_e=float(np.std(diffs)), spread=float(np.std([m['s68'] for m in ms])))
        else:
            m = TT.metrics(t, T['tr'], T['U'], T['W']); dd = dang(t, T['tr'])
            diffs = [TT.s68(dd[b]) - TT.s68(d_p1[b]) for b in bidx]
            res[name] = dict(s68=m['s68'], s68_steep=m['s68_steep'], off_rms=m['cell_offset_rms'], dif=m['s68'] - TT.s68(d_p1),
                             dif_e=float(np.std(diffs)), scale_rms=m['cell_scale_rms_pct'])
        print(p, name, {k: round(v, 4) for k, v in res[name].items()}, flush=True)
    # within-golden references: run1 map -> run2, run2 map -> run1 (and pass-1 baseline there)
    ref = {}
    for src, dst in (('run1', 'run2'), ('run2', 'run1')):
        Td = V[(dst, p)]; Ts = V[(src, p)]
        r = {'pass1': TT.metrics(Td['tp'], Td['tr'], Td['U'], Td['W'])['s68']}
        c = TT.robust_const(Td['tr'], Td['w'], 0.0, 1.0); r['const'] = TT.metrics((Td['w'] - c[0]) / c[1], Td['tr'], Td['U'], Td['W'])['s68']
        for n in (4, 6, 8):
            mp = Maps(Ts['U'], Ts['W'], Ts['w'], Ts['tr'], n)
            for mode in ('w0+v', 'w0'):
                r[f'{n} {mode}'] = TT.metrics(angle(Td, mp, mode), Td['tr'], Td['U'], Td['W'])['s68']
        ref[f'{src}->{dst}'] = r
        print(p, src, dst, {k: round(v, 4) for k, v in r.items()})
    out[p] = dict(r625=res, golden_ref=ref, n=n_ev)
json.dump(out, open(f'{OUT}/res625.json', 'w'), indent=1, default=float)
