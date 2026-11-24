"""r1_05_ab.py: free coefficients  tau = a grad h + b grad(delta g)  (event level, bootstrap over tilt events and gap-field replicates)."""
import sys
from r1_lib2 import *
NB = 150
rng = np.random.default_rng(33)
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
J = {}
CASES = [('T', 'run1', 'run2'), ('T', 'run2', 'run1'), ('T', 'run1', 'run1'), ('T', 'run2', 'run2'),
         ('q', 'run1', 'run2'), ('q', 'run2', 'run1'), ('q', 'run1', 'run1'), ('q', 'run2', 'run2'), ('q', 'r625', 'r625')]
HW = ('h', 'perview')
for kind, src, dst in CASES:
    name = f'{kind}_{src}'
    fac = FLD[name]['beta']; fac = G0 / (1 + fac)
    for p in 'xy':
        v = V[(dst, p)]
        i = 0 if p == 'x' else 1
        Bd = SPL.basis(v['U'], v['W'], 1 - i, i)      # derivative along own axis
        n = len(v['w'])
        for hw in HW:
            gh = hgrad(v['U'], v['W'], hw, BUNDLE625 if dst == 'r625' else BUNDLE)[i]
            for fine in (False, True):
                gg0 = fac * (Bd @ FLD[name]['c']) / 100
                r0 = abfit(v, gh, gg0, fine)
                bs = []
                for _ in range(NB):
                    idx = rng.integers(0, n, n); rep = rng.integers(0, len(FLD[name]['cb']))
                    gg = fac * (Bd @ FLD[name]['cb'][rep]) / 100
                    vv = {k: v[k][idx] for k in ('U', 'W', 'w', 'tr')}
                    bs.append(abfit(vv, gh[idx], gg[idx], fine))
                a_b = np.array([[x['a'], x['b']] for x in bs])
                key = f'{name}->{dst}:{p}:{hw}:{"fine" if fine else "full"}'
                J[key] = dict(a=r0['a'], b=r0['b'], v=r0['v'], a_err=float(a_b[:, 0].std()), b_err=float(a_b[:, 1].std()),
                              corr_ab=float(np.corrcoef(a_b.T)[0, 1]))
                print(key, {k: round(x, 3) for k, x in J[key].items()}, flush=True)
json.dump(J, open(f'{OUT}/part2_ab.json', 'w'), indent=1)
