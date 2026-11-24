"""D part 1: cross-validated cell-map gain on all chambers (track_grid procedure, ordered pairs of rows of one chamber)."""
import json, sys, itertools
from multiprocessing import Pool
sys.path.insert(0, '.')
from d_lib import *
NB = 300
NS = (4, 6, 8, 10)


def job(a):
    tr_key, te_key, variant = a
    xh = variant == 'xhalf'
    dtr, dte = load(tr_key, xh), load(te_key, xh)
    out = {}
    for p in ('x',) if xh else 'xy':
        ul = (0.0, 1.7) if xh else UL
        S, T = view(dtr, p), view(dte, p)
        D = {'pass1': ddeg(T[4], T[3])}
        valid = {}
        for n in NS:
            A = arms(S, T, n, ul=ul)
            if n == NS[0]:
                D['const'] = ddeg(A['const'], T[3])
            for arm in ('w0+v', 'w0'):
                if arm in A:
                    D[f'{n} {arm}'] = ddeg(A[arm], T[3])
            C = cells(S[0], S[1], S[2], S[3], n, ul)
            valid[n] = (int(np.sum(np.isfinite([c['v'] for c in C]))), n * n, float(np.median([c['n'] for c in C])))
        idx = boot_idx(len(D['pass1']), NB, 1)
        pt = {k: s68(v) for k, v in D.items()}
        bs = {k: np.array([s68(v[i]) for i in idx]) for k, v in D.items()}
        res = {}
        for k in D:
            r = dict(s68=pt[k], err=float(bs[k].std()))
            for ref in ('pass1', 'const'):
                if k != ref:
                    dd = bs[k] - bs[ref]
                    r[f'd_{ref}'] = pt[k] - pt[ref]; r[f'd_{ref}_err'] = float(dd.std())
            res[k] = r
        out[p] = dict(arms=res, valid=valid, ntest=len(D['pass1']), ntrain=len(S[0]))
    return (tr_key, te_key, variant), out


if __name__ == '__main__':
    jobs = []
    for ch, rows in CHAMBERS.items():
        for tr_, te in itertools.permutations(rows, 2):
            jobs.append((tr_, te, 'full'))
            if ch == 'det7':
                jobs.append((tr_, te, 'xhalf'))
    with Pool(4) as P:
        R = P.map(job, jobs, chunksize=1)
    out = {f'{a}>{b}|{v}': o for (a, b, v), o in R}
    json.dump(out, open(f'{OUT}/part1.json', 'w'), indent=1, default=float)
    for k, o in out.items():
        for p, r in o.items():
            a = r['arms']
            print(k, p, 'p1 %.3f c %.3f' % (a['pass1']['s68'], a['const']['s68']), ' '.join('%s %.3f' % (n, a[n]['s68']) for n in a if n[0].isdigit() and n.endswith('w0+v')), flush=True)
