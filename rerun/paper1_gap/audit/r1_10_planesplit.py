"""r1_10_planesplit.py: b of the plane part and of the fine part of the gap gradient, with a = 1 fixed (per-view and cross-view h)."""
from r1_lib2 import *
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
rng = np.random.default_rng(3); J = {}
for kind, src, dst in [('T', 'run1', 'run2'), ('T', 'run2', 'run1'), ('q', 'run1', 'run2'), ('q', 'run2', 'run1'), ('q', 'r625', 'r625')]:
    name = f'{kind}_{src}'; fac = G0 / (1 + FLD[name]['beta'])
    for p in 'xy':
        v = V[(dst, p)]; i = 0 if p == 'x' else 1; n = len(v['w'])
        Bd = SPL.basis(v['U'], v['W'], 1 - i, i) / 100
        for hw in ('h', 'perview'):
            gh = hgrad(v['U'], v['W'], hw, BUNDLE625 if dst == 'r625' else BUNDLE)[i]
            def f(idx, rep):
                gg = fac * Bd @ (FLD[name]['c'] if rep is None else FLD[name]['cb'][rep])
                pl = (gg - plane_removed(v['U'], v['W'], gg))[idx]; fn = plane_removed(v['U'], v['W'], gg)[idx]
                X = np.vstack([np.ones(len(idx)), v['tr'][idx] + gh[idx], pl, fn]).T
                c = CL.lsq_clip(X, v['w'][idx])[0]
                return c[2] / c[1], c[3] / c[1]
            c0 = f(np.arange(n), None)
            bs = np.array([f(rng.integers(0, n, n), int(rng.integers(0, 150))) for _ in range(100)])
            J[f'{name}->{dst}:{p}:{hw}'] = dict(b_plane=[c0[0], float(bs[:, 0].std())], b_fine=[c0[1], float(bs[:, 1].std())])
            print(name, dst, p, hw, {k: [round(x, 3) for x in a] for k, a in J[f'{name}->{dst}:{p}:{hw}'].items()}, flush=True)
json.dump(J, open(f'{OUT}/part4_planesplit.json', 'w'), indent=1, default=float)
