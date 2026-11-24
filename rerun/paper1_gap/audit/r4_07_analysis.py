"""r4_07_analysis: all numbers of R4 -> $WORK/audit/r4/analysis.json
 per-sub-run shape (quadratic) and golden-template amplitude; beta_eff(V) from the drift scan; coefficient against V^2
 and V; mesh scan; time blocks; the other det3 dates (6-22, 6-23, 6-26)."""
import sys, glob, os, json, pickle
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import numpy as np, pandas as pd
from scipy.interpolate import RegularGridInterpolator
import r4_lib as L
T = pickle.load(open(f'{L.OUT}/template.pkl', 'rb'))
GU, GW = T['GRID']
FINE = RegularGridInterpolator((GU, GW), T['f2'], bounds_error=False, fill_value=None)
TOT = RegularGridInterpolator((GU, GW), T['m2'] - T['m2'].mean(), bounds_error=False, fill_value=None)
NB = 200

def fit_unit(d, nb=NB, seed=3):
    """quadratic shape + covariates, and the amplitude of the golden template (fine part, quadratic free; total, plane free)."""
    r = {}
    f = L.fit_shape(d, nboot=nb, seed=seed)
    r.update(N=f['N'], mean_q=f['mean_q'], c=f['c'][:6].tolist(), err=f['err'][:6].tolist(), cov=f['c'][6:9].tolist())
    y = d.lq.values
    P = np.vstack([d.U.values, d.W.values]).T
    for lab, itp, nq in (('fine', FINE, 6), ('tot', TOT, 3)):
        t = itp(P)
        A0 = L.design(d, ('walk', 'tan'))
        if lab == 'tot': A0 = np.delete(A0, [3, 4, 5], axis=1)       # plane free only
        A = np.hstack([A0, t[:, None]])
        c, _ = L.rfit(A, y)
        rng = np.random.default_rng(seed + 1)
        B = []
        for _ in range(nb):
            w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
            B.append(L.rfit(A, y, w)[0][-1])
        r['amp_' + lab] = [float(c[-1]), float(np.std(B))]
    return r

def ivw(vals, errs):
    v, e = np.array(vals, float), np.array(errs, float)
    w = 1 / e ** 2
    m = (w * v).sum() / w.sum()
    chi2 = (((v - m) / e) ** 2).sum()
    return float(m), float(w.sum() ** -0.5), float(chi2), len(v) - 1

out = {'units': {}}
# ---- units: all det3 sub-runs on EOS (events tables), excluding det4-slot 6-23 long files without v info handled below
rels = sorted({os.path.dirname(p)[len(L.EOS) + 1:] for p in glob.glob(L.EOS + '/*/*/events.parquet')})
for rel in rels:
    d = L.load_events(rel)
    if len(d) < 120: continue
    r = fit_unit(d)
    r['run'], r['sub'] = rel.split('/')
    out['units'][rel] = r
    print(f'{rel.split("/")[0][-14:]:14s} {rel.split("/")[1][-34:]:36s} N {r["N"]:5d} gww {r["c"][4]:+.3f}+-{r["err"][4]:.3f} amp_fine {r["amp_fine"][0]:+.2f}+-{r["amp_fine"][1]:.2f}', flush=True)

# ---- time blocks (acq_tag files) in the long runs
out['blocks'] = {}
for key, rel in (('run6-25', L.R625), ('run6-27', L.GOLD1), ('run6-28', L.GOLD2)):
    d = L.load_events(rel)
    tags = sorted(d.acq_tag.unique())
    if key == 'run6-28':      # 12 files of ~1.2k events: keep 12
        groups = [[t] for t in tags]
    else:
        groups = [[t] for t in tags]
    bl = []
    for g in groups:
        dd = d[d.acq_tag.isin(g)]
        if len(dd) < 150: continue
        r = fit_unit(dd, nb=100); r['tag'] = g[0]; bl.append(r)
        print(key, g[0], r['N'], 'gww %+.3f+-%.3f guw %+.3f' % (r['c'][4], r['err'][4], r['c'][5]), flush=True)
    out['blocks'][key] = bl
json.dump(out, open(f'{L.OUT}/analysis.json', 'w'))
