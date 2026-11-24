import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from c_lib import *
D = {k: load(rel) for k, rel in ROWS.items()}
V = {(k, p): view(D[k], p) for k in D for p in 'xy'}
def sub(Vv, idx): return {q: Vv[q][idx] for q in ('U', 'W', 'w', 'tr')}
def pool(keys, p, idxs=None):
    return {q: np.concatenate([V[(k, p)][q] for k in keys]) for q in ('U', 'W', 'w', 'tr')}
def ols(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4: return np.nan, np.nan
    X = np.vstack([np.ones(len(x)), x]).T
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(c[1]), float(np.corrcoef(x, y)[0, 1])
out = {'cells': {}}
rng = np.random.default_rng(3)
sets = {'gold': ('run1', 'run2'), 'r625': ('r625',), 'run1': ('run1',), 'run2': ('run2',)}
NB = 100
imgs = {}
for p in 'xy':
    for n in (4, 6):
        def mk(name, boot=False):
            P = pool(sets[name], p)
            if boot:
                i = rng.integers(0, len(P['U']), len(P['U'])); P = {q: P[q][i] for q in P}
            return Maps(P['U'], P['W'], P['w'], P['tr'], n)
        base = {k: mk(k) for k in sets}
        imgs[f'{p}{n}'] = {k: dict(Afine=base[k].Afine, Aplane=base[k].Aplane, Ae=base[k].Ae, S=base[k].S, Uc=base[k].M['Uc'], Wc=base[k].M['Wc'], N=base[k].M['N'],
                                    A=base[k].A, vg=base[k].vg) for k in base}
        pairs = {'gold_vs_r625': ('gold', 'r625'), 'run1_vs_run2': ('run1', 'run2'), 'run1_vs_r625': ('run1', 'r625'), 'run2_vs_r625': ('run2', 'r625')}
        r = {}
        for nm, (a, b) in pairs.items():
            for q in ('Afine', 'Aplane', 'S', 'A'):
                s0, c0 = ols(getattr(base[a], q).ravel(), getattr(base[b], q).ravel())
                bs = []
                for _ in range(NB):
                    ma, mb = mk(a, True), mk(b, True)
                    bs.append(ols(getattr(ma, q).ravel(), getattr(mb, q).ravel()))
                bs = np.array(bs)
                r[f'{nm}:{q}'] = dict(slope=s0, slope_e=float(np.nanstd(bs[:, 0])), corr=c0, corr_e=float(np.nanstd(bs[:, 1])))
            print(p, n, nm, {q: (round(r[f'{nm}:{q}']['slope'], 2), round(r[f'{nm}:{q}']['slope_e'], 2), round(r[f'{nm}:{q}']['corr'], 2)) for q in ('Afine', 'Aplane', 'S')}, flush=True)
        # noise reliability of fine maps
        for k in base:
            out.setdefault('noise', {})[f'{p}{n}:{k}'] = dict(var_fine=float(np.nanvar(base[k].Afine)), mean_err2=float(np.nanmean(base[k].Ae ** 2)),
                                                          sd_fine=float(np.nanstd(base[k].Afine)), sd_plane=float(np.nanstd(base[k].Aplane)))
        out['cells'][f'{p}{n}'] = r
np.savez(f'{OUT}/maps_imgs.npz', **{f'{k}__{s}__{q}': v for k, d in imgs.items() for s, dd in d.items() for q, v in dd.items()})
json.dump(out, open(f'{OUT}/cells.json', 'w'), indent=1, default=float)
