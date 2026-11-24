"""r3_run.py <mounting> -- cross-validated arms for one mounting.  -> $OUT/cv_<mounting>.npz (angle residuals, deg) and .json"""
import sys, json, time, pickle
from r3_lib import *

name = sys.argv[1]
t0 = time.time()
D, info, bundle = load_mounting(name)
spl, cf, okq = gap_field(D)
attach_field(D, bundle, spl, cf)
print(name, 'N', D.n, 'rows', [i['N'] for i in info], 'field K', spl.K, flush=True)
multi = len(info) > 1
NF = 5
f_time = np.minimum((D.rank.argsort().argsort() * NF) // D.n, NF - 1)
schemes = {'time5': f_time}
if multi:
    schemes['loro'] = D.row.copy()
res = dict(name=name, N=D.n, rows=info, field_K=spl.K)
arrs = {}
SLOPE = {}


def jack(v):
    v = np.asarray(v, float); n = len(v)
    return float(v.mean()), float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def lnv_slope(m):
    a1, a2, s1, s2 = eval_map(m, D.U, D.W)
    out = {}
    X = np.vstack([np.ones(D.n), D.lf]).T
    lx, ly = np.log(s1 / s1.mean()), np.log(s2 / s2.mean())
    for k, y in (('x', lx), ('y', ly), ('xy', 0.5 * (lx + ly))):
        out[k] = float(CL.lsq_clip(X, y)[0][1])
    return out


beta_used = BETA0
for sch, f in schemes.items():
    folds = sorted(set(f))
    P = {a: {'x': np.zeros(D.n), 'y': np.zeros(D.n)} for a in
         ('pass1', 'const', 'dir_phiS', 'dir_tilt', 'rf_mesh', 'rf_tilt_nom', 'rf_tilt_fit', 'rf_tv_nom', 'rf_tv_fit', 'both_phiS', 'both_tilt')}
    P['pass1']['x'][:] = D.tpx; P['pass1']['y'][:] = D.tpy
    meta = {'sel': {}, 'b': [], 'b_xy': [], 'slopes': []}
    tx, ty = tans(D, None)           # const arm: constants per row on the whole row (they are refit per row in production anyway)
    P['const']['x'][:], P['const']['y'][:] = tx, ty
    folddata = []
    # ---- loop 1: direct maps and b
    for k in folds:
        tr_, te = D.sub(f != k), D.sub(f == k)
        te_i = np.where(f == k)[0]
        for var, arm in (('S', 'dir_phiS'), ('T', 'dir_tilt')):
            K, KS, lam, sc = select(tr_, var)
            m = fit_generic(Spec('phi', K, KS, 1.0), tr_, lam)
            tx, ty = tans(te, m)
            P[arm]['x'][te_i], P[arm]['y'][te_i] = tx, ty
            meta['sel'][f'{arm}:{k}'] = [K, KS, lam, sc]
            if var == 'S' and KS > 0:
                meta['slopes'].append(lnv_slope(m))
        bb = fit_b(tr_)
        meta['b'].append(bb)
        print(name, sch, 'fold', k, 'loop1 done', round(time.time() - t0), meta['sel'][f'dir_phiS:{k}'], bb, flush=True)
    # ---- beta from the direct map's S field (once, from the time5 scheme)
    if sch == 'time5':
        sl = meta['slopes']
        if len(sl) >= 3:
            s, se = jack([q['xy'] for q in sl])
            sx_, sex = jack([q['x'] for q in sl]); sy_, sey = jack([q['y'] for q in sl])
            bet = -s / (1 + s) if s > -0.99 else np.nan
            ok = (s < 0) and (abs(s) / se > 3) and (0.2 < bet < 2.0)
            beta_used = float(bet) if ok else BETA0
            SLOPE = dict(slope=s, err=se, slope_x=sx_, err_x=sex, slope_y=sy_, err_y=sey, beta_meas=float(bet), measurable=bool(ok),
                         beta_used=beta_used, nfold=len(sl))
        else:
            SLOPE = dict(slope=None, measurable=False, beta_used=BETA0, nfold=len(sl))
        print(name, 'beta', SLOPE, flush=True)
        res['beta'] = SLOPE
    # ---- loop 2: reference-free arms and both
    for ik, k in enumerate(folds):
        tr_, te = D.sub(f != k), D.sub(f == k)
        te_i = np.where(f == k)[0]
        bk = meta['b'][ik]['xy']
        for arm, (b, vel) in {'rf_mesh': (0.0, False), 'rf_tilt_nom': (0.5, False), 'rf_tilt_fit': (bk, False),
                              'rf_tv_nom': (0.5, True), 'rf_tv_fit': (bk, True)}.items():
            # b is quoted at beta 0.68; the amplitude of the gap term scales as 1/(1+beta): keep b * G0/(1+beta_used) * (1+beta_used)/(1+0.68)
            bb = b if arm.endswith('fit') is False else b * (1 + beta_used) / (1 + BETA0)
            ax, ay, sx, sy = rf_base(te, bb, beta_used, vel)
            te2 = with_base(te, ax, ay, sx, sy)
            tx, ty = tans(te2, None)
            P[arm]['x'][te_i], P[arm]['y'][te_i] = tx, ty
        # both: base on train, remainder map
        for arm, (vel, var) in {'both_phiS': (True, 'S'), 'both_tilt': (False, 'T')}.items():
            bb = bk * (1 + beta_used) / (1 + BETA0)
            ax, ay, sx, sy = rf_base(tr_, bb, beta_used, vel)
            tr2 = with_base(tr_, ax, ay, sx, sy)
            K, KS, lam, sc = select(tr2, var)
            m = fit_generic(Spec('phi', K, KS, 1.0), tr2, lam)
            ax, ay, sx, sy = rf_base(te, bb, beta_used, vel)
            te2 = with_base(te, ax, ay, sx, sy)
            tx, ty = tans(te2, m)
            P[arm]['x'][te_i], P[arm]['y'][te_i] = tx, ty
            meta['sel'][f'{arm}:{k}'] = [K, KS, lam, sc]
        print(name, sch, 'fold', k, 'loop2 done', round(time.time() - t0), flush=True)
    # ---- store
    for a in P:
        for p, tr in (('x', D.trx), ('y', D.trY)):
            arrs[f'{sch}|{a}|{p}'] = dth(P[a][p], tr).astype(np.float32)
    bs = [q['xy'] for q in meta['b']]
    meta['b_mean'], meta['b_err'] = jack(bs)
    meta['b_x_mean'], meta['b_x_err'] = jack([q['x'] for q in meta['b']])
    meta['b_y_mean'], meta['b_y_err'] = jack([q['y'] for q in meta['b']])
    res[sch] = meta
for k in ('xm', 'ym', 'row', 'rank', 'satx', 'saty'):
    arrs[k] = getattr(D, k)
arrs['f_time5'] = f_time
np.savez(f'{OUT}/cv_{name}.npz', **arrs)
json.dump(res, open(f'{OUT}/cv_{name}.json', 'w'), indent=1, default=float)
print(name, 'finished', round(time.time() - t0), flush=True)
