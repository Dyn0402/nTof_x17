"""r1_07_part3_625.py: the 6-25 row: golden direct map, mesh term, own q_u90 gap field (nominal and free amplitude), own 2-fold ceiling."""
import sys
sys.argv = sys.argv[:1]
from r1_lib2 import *
import runpy
NBS = 200
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
rng = np.random.default_rng(77)
NSPLIT = 5

def dang(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))

def apply(v, dw, vr):
    c = TT.robust_const(v['tr'], v['w'], dw, vr)
    return (v['w'] - c[0] - dw) / (c[1] * vr)

def sub(v, m):
    return {k: v[k][m] for k in ('U', 'W', 'w', 'tr')}

def halves(n, k):
    r = np.random.default_rng(100 + k); h = r.random(n) < 0.5
    return h, ~h

def fit_model(v, gh, gg, mode):
    """mode: 'b' (a = 1), 'ab', 'abp' (a, b and a free plane).  returns function dw(vtest, ghT, ggT)."""
    n = len(v['w'])
    if mode == 'b':
        X = np.vstack([np.ones(n), v['tr'] + gh, gg]).T
    elif mode == 'ab':
        X = np.vstack([np.ones(n), v['tr'], gh, gg]).T
    else:
        X = np.vstack([np.ones(n), v['tr'], gh, gg, v['U'], v['W']]).T
    c = CL.lsq_clip(X, v['w'])[0]
    c1 = c[1]
    if mode == 'b':
        a, b, pl = 1.0, c[2] / c1, (0, 0)
    else:
        a, b = c[2] / c1, c[3] / c1
        pl = (c[4], c[5]) if mode == 'abp' else (0, 0)
    return dict(a=a, b=b, c1=c1, pl=pl)

def dwof(par, vt, gh, gg, vv=None):
    return par['c1'] * (par['a'] * gh + par['b'] * gg) + par['pl'][0] * vt['U'] + par['pl'][1] * vt['W']

J = {}
for p in 'xy':
    v = V[('r625', p)]; n = len(v['w']); i = 0 if p == 'x' else 1
    gh = hgrad(v['U'], v['W'], 'h', BUNDLE625)[i]
    fq = Fld('q_r625')
    gg = fq.grad(v['U'], v['W'])[i]
    vr_nom = fq.vrel(v['U'], v['W'])
    c0 = TT.robust_const(v['tr'], v['w'], 0.0, 1.0); vv = c0[1]
    arms = {'pass1': v['tp'], 'const': apply(v, 0.0, 1.0)}
    # golden direct maps (pooled run1 + run2)
    G = {q: np.concatenate([V[(r, p)][q] for r in ('run1', 'run2')]) for q in ('U', 'W', 'w', 'tr')}
    for nn in (4, 8):
        mp = CL.Maps(G['U'], G['W'], G['w'], G['tr'], nn)
        ap, af, s, _ = mp.ev(v['U'], v['W'])
        dd = vv * (ap + af); vr = 1 + s
        arms[f'golden direct {nn}x{nn} w0'] = apply(v, dd, 1.0)
        arms[f'golden direct {nn}x{nn} w0+v'] = apply(v, dd, vr)
    arms['mesh'] = apply(v, vv * gh, 1.0)
    arms['mesh+gap own q_u90, nominal (b=0.5/(1+1.28) scale)'] = apply(v, vv * (gh + 0.5 * gg), 1.0)
    ghp = hgrad(v['U'], v['W'], 'perview', BUNDLE625)[i]
    arms['mesh (per-view h)'] = apply(v, vv * ghp, 1.0)
    arms['mesh+gap own q_u90, nominal, per-view h'] = apply(v, vv * (ghp + 0.5 * gg), 1.0)
    arms['mesh+gap + v_rel (beta 1.28), per-view h'] = apply(v, vv * (ghp + 0.5 * gg), vr_nom)
    arms['mesh+gap + v_rel (beta 1.28)'] = apply(v, vv * (gh + 0.5 * gg), vr_nom)
    # free amplitude, fitted on one half, tested on the other (5 random splits)
    for mode, lab in (('b', 'free b (a=1)'), ('ab', 'free a, b'), ('abp', 'free a, b + plane (uses reference)')):
        L = []
        pars = []
        for k in range(NSPLIT):
            h1, h2 = halves(n, k)
            t = np.zeros(n)
            for tr_m, te_m in ((h1, h2), (h2, h1)):
                par = fit_model(sub(v, tr_m), gh[tr_m], gg[tr_m], mode); pars.append(par)
                vt = sub(v, te_m)
                dw = dwof(par, vt, gh[te_m], gg[te_m])
                t[te_m] = apply(vt, dw, 1.0)
            L.append(t)
        arms[f'mesh+gap own q_u90, {lab}'] = L
        J[f'{p}:amp:{mode}'] = dict(a=float(np.mean([q['a'] for q in pars])), a_sd=float(np.std([q['a'] for q in pars])),
                                    b=float(np.mean([q['b'] for q in pars])), b_sd=float(np.std([q['b'] for q in pars])))
        print(p, mode, J[f'{p}:amp:{mode}'], flush=True)
    # + v_rel with the free-b arm
    L = []
    for k in range(NSPLIT):
        h1, h2 = halves(n, k); t = np.zeros(n)
        for tr_m, te_m in ((h1, h2), (h2, h1)):
            par = fit_model(sub(v, tr_m), gh[tr_m], gg[tr_m], 'b'); vt = sub(v, te_m)
            t[te_m] = apply(vt, dwof(par, vt, gh[te_m], gg[te_m]), vr_nom[te_m])
        L.append(t)
    arms['mesh+gap own q_u90, free b + v_rel (beta 1.28)'] = L
    # own 2-fold direct ceilings
    for nn in (4, 6, 8):
        for mode in ('w0+v', 'w0'):
            L = []
            for k in range(NSPLIT):
                h1, h2 = halves(n, k); t = np.zeros(n)
                for tr_m, te_m in ((h1, h2), (h2, h1)):
                    dtr = sub(v, tr_m); dte = sub(v, te_m)
                    mp = CL.Maps(dtr['U'], dtr['W'], dtr['w'], dtr['tr'], nn)
                    ap, af, s, _ = mp.ev(dte['U'], dte['W'])
                    c0_ = TT.robust_const(dte['tr'], dte['w'], 0.0, 1.0)
                    dd = c0_[1] * (ap + af); vr = (1 + s) if mode == 'w0+v' else np.ones_like(s)
                    t[te_m] = apply(dte, dd, vr)
                L.append(t)
            arms[f'own 2-fold direct {nn}x{nn} {mode}'] = L
    # metrics + paired bootstrap
    bidx = [rng.integers(0, n, n) for _ in range(NBS)]
    Dd = {k: ([dang(x, v['tr']) for x in t] if isinstance(t, list) else [dang(t, v['tr'])]) for k, t in arms.items()}
    def s(k, idx=None):
        return float(np.mean([TT.s68(d if idx is None else d[idx]) for d in Dd[k]]))
    base = {k: np.array([s(k, b) for b in bidx]) for k in arms}
    res = {}
    for k in arms:
        res[k] = dict(s68=s(k), sd=float(base[k].std()))
        for ref in ('pass1', 'mesh', 'const'):
            dd = base[k] - base[ref]
            res[k][f'd_{ref}'] = res[k]['s68'] - s(ref); res[k][f'd_{ref}_err'] = float(dd.std())
    for c in ('own 2-fold direct 4x4 w0+v', 'own 2-fold direct 8x8 w0+v', 'own 2-fold direct 4x4 w0', 'golden direct 4x4 w0+v'):
        den0 = res['pass1']['s68'] - res[c]['s68']
        for k in arms:
            fr = (base['pass1'] - base[k]) / (base['pass1'] - base[c])
            res[k][f'frac_{c}'] = float((res['pass1']['s68'] - res[k]['s68']) / den0); res[k][f'frac_{c}_err'] = float(fr.std())
    J[f'{p}:arms'] = res
    print(p, {k: round(x['s68'], 3) for k, x in res.items()}, flush=True)
json.dump(J, open(f'{OUT}/part3_625.json', 'w'), indent=1)
