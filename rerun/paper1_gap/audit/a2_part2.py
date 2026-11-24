"""a2_part2.py -- learning curve of the phi map: s68 against N_train, with the inner CV re-choosing (K, lambda) at every N."""
import sys, json, time
from a2_lib import *
KS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
which = sys.argv[1]            # 'A' (run2+half run1 -> other half run1), 'B' (mirror), 'U' (union 5-fold)
D1, D2 = load('run1'), load('run2')
Ks = (9, 12, 16, 20); lams = 10.0 ** np.arange(-2.5, 2.1, 0.75)
fracs = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
out = []
t0 = time.time()
for seed in range(4):
    rng = np.random.default_rng(100 + seed)
    if which in 'AB':
        full, half = (D1, D2) if which == 'B' else (D2, D1)
        hp = rng.permutation(half.n); hA, hB = hp[:half.n // 2], hp[half.n // 2:]
        pool = concat(full, half.sub(np.sort(hA))); test = half.sub(np.sort(hB))
    else:
        U = concat(D1, D2); perm = rng.permutation(U.n); k = seed % 5
        f = np.arange(U.n) % 5; f = f[np.argsort(perm)]
        pool, test = U.sub(f != k), U.sub(f == k)
    kap = pilot_kappa(pool)
    for fr in fracs:
        n = int(fr * pool.n); idx = np.sort(rng.permutation(pool.n)[:n]); Dn = pool.sub(idx)
        best = None
        for K in Ks:
            spec = Spec('phi', K, KS, kap)
            for lam in lams:
                cx, cy = cv_score(spec, Dn, lam)
                if best is None or cx + cy < best[0]: best = (cx + cy, K, lam)
        spec = Spec('phi', best[1], KS, kap); m = fit_map(spec, Dn, best[2]); sc = scores(apply_map(m, test), test)
        try:
            arms = baseline_arms(test, Dn, grids=(8,)); pc = {a: {p: s68(dth(arms[a][p], test.trx if p == 'x' else test.trY)) for p in 'xy'} for a in arms}
        except Exception:
            pc = {'cells8': {'x': float('nan'), 'y': float('nan')}}
        out.append(dict(which=which, seed=seed, N=n, ntest=test.n, K=best[1], lam=float(best[2]), x=sc['x'], y=sc['y'], base=pc))
        print(which, seed, n, best[1:], {k: round(v, 3) for k, v in sc.items()}, 'cells8', round(pc['cells8']['x'], 3), round(pc['cells8']['y'], 3), round(time.time() - t0), flush=True)
json.dump(out, open(f'{OUT}/part2_{which}.json', 'w'), indent=1)
