"""a2_part1.py <train_run> -- unbinned spline maps trained on one run (inner time-block CV picks lambda, K_S, K), applied to the other."""
import sys, json, time, itertools
from a2_lib import *
tr_run = sys.argv[1]; te_run = {'run1': 'run2', 'run2': 'run1'}[tr_run]
Dtr, Dte = load(tr_run), load(te_run)
kap = pilot_kappa(Dtr)
KS_list = (0, 4, 6, 8); K_list = (5, 7, 9, 12, 16, 20, 26); lams = 10.0 ** np.arange(-3, 3.6, 0.75)
res = {}; darr = {}
base = baseline_arms(Dtr and Dte, Dtr)
for a, pr in base.items():
    darr[f'{a}|x'] = dth(pr['x'], Dte.trx); darr[f'{a}|y'] = dth(pr['y'], Dte.trY)
grid = {}
t0 = time.time()
for var in VARIANTS:
    for K in K_list:
        for KS in KS_list:
            spec = Spec(var, K, KS, kap, k_extra=min(K, 9))
            cv = []
            for lam in lams:
                cx, cy = cv_score(spec, Dtr, lam)
                cv.append((float(lam), cx, cy))
            cvs = np.array([c[1] + c[2] for c in cv]); j = int(np.argmin(cvs)); lam = cv[j][0]
            m = fit_map(spec, Dtr, lam); pr = apply_map(m, Dte); sc = scores(pr, Dte)
            grid[f'{var}|{K}|{KS}'] = dict(cv=cv, lam=lam, cv_x=cv[j][1], cv_y=cv[j][2], test=sc, edge=(j == 0 or j == len(lams) - 1))
            darr[f'{var}|{K}|{KS}|x'] = dth(pr['x'], Dte.trx); darr[f'{var}|{K}|{KS}|y'] = dth(pr['y'], Dte.trY)
            print(tr_run, var, K, KS, 'lam', lam, 'cv', round(cv[j][1], 3), round(cv[j][2], 3), 'test', {k: round(v, 3) for k, v in sc.items()}, round(time.time() - t0), flush=True)
json.dump(dict(train=tr_run, test=te_run, kappa=kap, lams=list(lams), grid=grid), open(f'{OUT}/part1_{tr_run}.json', 'w'), indent=1)
np.savez_compressed(f'{OUT}/part1_{tr_run}_dth.npz', **darr)
