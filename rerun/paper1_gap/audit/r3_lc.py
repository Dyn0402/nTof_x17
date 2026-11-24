"""r3_lc.py <mounting> -- learning curve of the direct phi+S map inside a mounting (5 time blocks, train subsampled at random),
(K, lambda) re-chosen by inner CV at every N (restricted grid), s68^2 = s_inf^2 + c/N.  -> $OUT/lc_<mounting>.json"""
import sys, json, time
from r3_lib import *

name = sys.argv[1]
D, info, bundle = load_mounting(name)
NF = 5
f = np.minimum((D.rank.argsort().argsort() * NF) // D.n, NF - 1)
fracs = (1 / 8, 1 / 4, 1 / 2, 1.0)
out = []
t0 = time.time()
for seed in range(2):
    rng = np.random.default_rng(500 + seed)
    pooled = {fr: ([], []) for fr in fracs}
    ntr = {fr: [] for fr in fracs}
    for k in range(NF):
        tr_all, te = D.sub(f != k), D.sub(f == k)
        for fr in fracs:
            n = int(fr * tr_all.n)
            idx = np.sort(rng.permutation(tr_all.n)[:n]); tr_ = tr_all.sub(idx)
            K, KS, lam, _ = select(tr_, 'S', kgrid=(9, 12, 16), ksgrid=(6,), lams=(0.01, 0.1, 1.0), nf=3)
            m = fit_generic(Spec('phi', K, KS, 1.0), tr_, lam)
            tx, ty = tans(te, m)
            pooled[fr][0].append(dth(tx, te.trx)); pooled[fr][1].append(dth(ty, te.trY)); ntr[fr].append(n)
    for fr in fracs:
        out.append(dict(seed=seed, frac=fr, N=float(np.mean(ntr[fr])), x=s68(np.concatenate(pooled[fr][0])), y=s68(np.concatenate(pooled[fr][1]))))
    print(name, seed, round(time.time() - t0), [(o['N'], round(o['x'], 3), round(o['y'], 3)) for o in out[-4:]], flush=True)


def asym(Ns, ss):
    A = np.vstack([np.ones(len(Ns)), 1 / np.asarray(Ns)]).T
    c = np.linalg.lstsq(A, np.asarray(ss) ** 2, rcond=None)[0]
    return float(np.sqrt(max(c[0], 0.0))), float(c[1])


res = {}
for p in 'xy':
    ests = []
    for sd in range(2):
        pts = [(o['N'], o[p]) for o in out if o['seed'] == sd]
        for sl in (slice(0, 4), slice(1, 4), slice(0, 3)):
            ests.append(asym([q[0] for q in pts][sl], [q[1] for q in pts][sl])[0])
    res[p] = dict(s_inf=float(np.mean(ests)), err=float((max(ests) - min(ests)) / 2), all=ests,
                  s_full=float(np.mean([o[p] for o in out if o['frac'] == 1.0])),
                  s_eighth=float(np.mean([o[p] for o in out if o['frac'] == 1 / 8])))
json.dump(dict(name=name, points=out, asym=res), open(f'{OUT}/lc_{name}.json', 'w'), indent=1)
print(name, res)
