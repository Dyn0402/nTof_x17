"""a2_final.py -- map of the union of both runs: CV choice of variant / K / K_S / lambda (5 time blocks over the union, run1 first
then run2), json of the final phi map, and a tiny evaluator."""
import json, sys, time
from a2_lib import *


def evaluate(path_or_dict, U, W):
    """Standalone evaluator (numpy + scipy only): returns (a_x, a_y, s_x, s_y) in um/ns at the reconstructed mid-gap
    position (U, W) = ((x - 200)/100, (y - 200)/100).  a_x = dPhi/dU + cx, a_y = kappa dPhi/dW + cy, s = tr coefficient
    (v (1 + S)).  Use: dw0 = a - mean(a), vrel = s / mean(s) over the run, two constants refit (production recipe)."""
    from scipy.interpolate import BSpline
    M = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
    def b(x, K, lo, hi, nu=0):
        t = np.concatenate([[lo] * 3, np.linspace(lo, hi, K - 2), [hi] * 3]); f = BSpline(t, np.eye(K), 3)
        return (f if nu == 0 else f.derivative(nu))(np.clip(x, lo, hi))
    def t2(Bu, Bw): return (Bu[:, :, None] * Bw[:, None, :]).reshape(len(Bu), -1)
    K, KS = M['K'], M['KS']; (ulo, uhi), (wlo, whi) = M['U_domain'], M['W_domain']
    phi = np.array(M['phi']); kap = M['kappa']
    ax = t2(b(U, K, ulo, uhi, 1), b(W, K, wlo, whi)) @ phi + M['cx']
    ay = kap * (t2(b(U, K, ulo, uhi), b(W, K, wlo, whi, 1)) @ phi) + M['cy']
    if KS == 0:
        sx = np.full(len(U), M['sx'][0]); sy = np.full(len(U), M['sy'][0])
    else:
        T = t2(b(U, KS, ulo, uhi), b(W, KS, wlo, whi)); sx = T @ np.array(M['sx']); sy = T @ np.array(M['sy'])
    return ax, ay, sx, sy


if __name__ == '__main__':
    D1, D2 = load('run1'), load('run2')
    D2.rank = D2.rank + D1.n
    Uni = concat(D1, D2); kap = pilot_kappa(Uni)
    lams = 10.0 ** np.arange(-3, 2.6, 0.75); res = {}
    t0 = time.time()
    plan = [('phi', K, KS) for K in (9, 12, 16, 20, 26) for KS in (0, 4, 6, 8)] + \
           [(v, K, 6) for v in ('indep', 'phiplus') for K in (9, 12, 16, 20, 26)]
    for var, K, KS in plan:
        spec = Spec(var, K, KS, kap, k_extra=min(K, 9))
        cv = [(float(l),) + cv_score(spec, Uni, l) for l in lams]
        j = int(np.argmin([c[1] + c[2] for c in cv]))
        res[f'{var}|{K}|{KS}'] = dict(lam=cv[j][0], cv_x=cv[j][1], cv_y=cv[j][2], edge=(j == 0 or j == len(lams) - 1), curve=cv)
        print(var, K, KS, cv[j], round(time.time() - t0), flush=True)
    # cells reference in the same CV
    best = min((k for k in res if k.startswith('phi|')), key=lambda k: res[k]['cv_x'] + res[k]['cv_y'])
    var, K, KS = best.split('|'); K, KS = int(K), int(KS); lam = res[best]['lam']
    spec = Spec('phi', K, KS, kap); m = fit_map(spec, Uni, lam)
    c = m['c']; o = spec.off
    cx0 = TT.robust_const(Uni.trx, Uni.wx, 0.0, 1.0); cy0 = TT.robust_const(Uni.trY, Uni.wy, 0.0, 1.0)
    M = dict(K=K, KS=KS, lam=lam, kappa=kap, U_domain=list(UD), W_domain=list(WD), degree=3,
             knots_U=knots(K, *UD).tolist(), knots_W=knots(K, *WD).tolist(), knots_S_U=(knots(KS, *UD).tolist() if KS else None),
             knots_S_W=(knots(KS, *WD).tolist() if KS else None),
             phi=c[slice(*o['phi'])].tolist(), cx=float(c[o['cx'][0]]), cy=float(c[o['cy'][0]]),
             sx=c[slice(*o['sx'])].tolist(), sy=c[slice(*o['sy'])].tolist(),
             v_used=dict(x=float(cx0[1]), y=float(cy0[1]), note='robust ladder slopes of the union (um/ns per unit tan); w0 not used, refit per run'),
             recipe='dw0 = a - mean(a); vrel = s/mean(s) over the run; refit two constants (TT.robust_const); tan = (w - c0 - dw0)/(c1 vrel)',
             ntrain=Uni.n, nclip_kept=m['nkeep'], cv_s68_deg=dict(x=res[best]['cv_x'], y=res[best]['cv_y']),
             convention='U=(x-200)/100, W=(y-200)/100 at reconstructed mid-gap (x_p0+14 tan - 200)/100; +y up as in the pass-1 tables')
    json.dump(M, open(f'{OUT}/final_map_det3.json', 'w'))
    ax, ay, sx, sy = evaluate(M, Uni.U, Uni.W); a2, b2, c2, d2 = eval_map(m, Uni.U, Uni.W)
    M['check_max_abs_diff'] = float(max(np.abs(ax - a2).max(), np.abs(ay - b2).max(), np.abs(sx - c2).max(), np.abs(sy - d2).max()))
    json.dump(M, open(f'{OUT}/final_map_det3.json', 'w'))
    # the CV reference arms on the union (same time blocks): pass1, cells 8x8
    nf = 5; f = np.minimum((Uni.rank.argsort().argsort() * nf) // Uni.n, nf - 1); ref = {a: ([], []) for a in ('pass1', 'pass1c', 'cells8', 'cells6', 'cells10')}
    for k in range(nf):
        tr_, te = Uni.sub(f != k), Uni.sub(f == k)
        try:
            arms = baseline_arms(te, tr_, grids=(6, 8, 10))
        except Exception as e:
            print('cells fail', e); continue
        for a in ref:
            ref[a][0].append(dth(arms[a]['x'], te.trx)); ref[a][1].append(dth(arms[a]['y'], te.trY))
    refs = {a: (s68(np.concatenate(v[0])), s68(np.concatenate(v[1]))) for a, v in ref.items()}
    json.dump(dict(cvgrid=res, best=best, refs=refs, check=M['check_max_abs_diff']), open(f'{OUT}/final_cv.json', 'w'), indent=1)
    print('best', best, res[best]['cv_x'], res[best]['cv_y'], refs, M['check_max_abs_diff'])
