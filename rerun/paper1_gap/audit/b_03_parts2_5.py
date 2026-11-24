#!/usr/bin/env python3
"""b_03_parts2_5.py: gap-or-gas slopes (Part 2), electrode regressions (Part 3), focusing (Part 4), model line (Part 5)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from b_lib import *

G0 = 28.2
J = {}
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

def lnmap(M, k):
    return np.log(M[k]), M[k + '_err'] / M[k]

def fields(n, run):
    """smooth + fine parts of the log maps and of hp, w0."""
    M = MAPS[run][n]
    F = {}
    for k in ('T', 'vx', 'vy'):
        y, e = lnmap(M, k)
        fit, res, c = wfit(y, e, n, 2)
        F[k] = dict(tot=y - np.nanmean(y), smooth=fit - np.nanmean(y), fine=res, coef=c)
    # hp: mm
    for k in ('hpx', 'hpy'):
        fit, res, c = wfit(M[k], M[k + '_err'], n, 2)
        F[k] = dict(tot=M[k] - np.nanmean(M[k]), smooth=fit - np.nanmean(M[k]), fine=res, coef=c)
    # tilt tau = w0 / v (chamber v) ; plane removed for fine
    for p in 'xy':
        vch = MAPS[run]['glob'][f'{p}_t06'][0]
        fit, res, c = wfit(M[f'w0{p}'], M[f'w0{p}_err'], n, 1)
        F['tau' + p] = dict(tot=(M['w0' + p] - np.nanmean(M['w0' + p])) / vch * 1.0, smooth=(fit - np.nanmean(M['w0' + p])) / vch,
                           fine=res / vch, coef=c / vch, v=vch)
    F['lnvm'] = {pt: 0.5 * (F['vx'][pt] + F['vy'][pt]) for pt in ('tot', 'smooth', 'fine')}
    return F

def beta_of(s):
    return -s / (1 + s)

# ============================================================ PART 2
P('\n##### PART 2: d ln v / d ln T')
EXPECT = -0.68 / 1.68
for n in (4, 6, 8):
    F1, F2 = fields(n, 'run1'), fields(n, 'run2')
    for vv in ('vx', 'vy'):
        for part in ('tot', 'smooth', 'fine'):
            a = [F1[vv][part].ravel(), F2[vv][part].ravel(), F1['T'][part].ravel(), F2['T'][part].ravel()]
            f = lambda v1, v2, t1, t2: iv_slope(v1, v2, t1, t2)
            s, se, lo, hi, N = boot_cells(f, a, nb=3000)
            # naive within-run (noise-biased) slope for comparison
            nv = 0.5 * (cov(a[0], a[2]) / cov(a[2], a[2]) + cov(a[1], a[3]) / cov(a[3], a[3]))
            bt = beta_of(s); blo, bhi = beta_of(hi), beta_of(lo)
            P(f'n={n} {vv} {part:6s} slope {s:+.3f} ± {se:.3f} [{lo:+.2f},{hi:+.2f}]  within-run {nv:+.3f}  beta {bt:.2f} [{blo:.2f},{bhi:.2f}] N={N}')
            J[f'p2_{n}_{vv}_{part}'] = dict(s=s, err=se, lo=lo, hi=hi, within=nv, beta=bt, beta_lo=blo, beta_hi=bhi, N=N)
    # average of both views
    for part in ('tot', 'smooth', 'fine'):
        a = [F1['lnvm'][part].ravel(), F2['lnvm'][part].ravel(), F1['T'][part].ravel(), F2['T'][part].ravel()]
        s, se, lo, hi, N = boot_cells(lambda v1, v2, t1, t2: iv_slope(v1, v2, t1, t2), a, nb=3000)
        P(f'n={n} v(x+y)/2 {part:6s} slope {s:+.3f} ± {se:.3f}   beta {beta_of(s):.2f} [{beta_of(hi):.2f},{beta_of(lo):.2f}]')
        J[f'p2_{n}_vm_{part}'] = dict(s=s, err=se, lo=lo, hi=hi)
P(f'expected: gap change beta=0.68 -> {EXPECT:.3f}; gas -> -1')

# tan-range dependence
P('\n-- tan range: chamber-wide v (um/ns)')
for run in RUNS:
    g = MAPS[run]['glob']
    P(run, {k: f'{v[0]:.3f}±{v[1]:.3f} (n={v[3]})' for k, v in g.items()})
    J[f'glob_{run}'] = {k: [float(v[0]), float(v[1])] for k, v in g.items()}
for n in (4, 6, 8):
    for p in 'xy':
        for lab in ('t03', 't016'):
            for run in RUNS:
                M = MAPS[run][n]
                a, b = M[f'v{p}'], M[f'v{p}_{lab}']
                d = (b - a)
                P(f'n={n} {p} {lab} {run}: r(map t06, map {lab}) {pear(a.ravel(), b.ravel()):.3f}; rms(diff) {np.nanstd(d):.3f} vs mean err of the narrower-range v {np.nanmean(M[f"v{p}_{lab}_err"]):.3f}; mean diff {np.nanmean(d):+.3f}')
            d1 = (MAPS['run1'][n][f'v{p}_{lab}'] - MAPS['run1'][n][f'v{p}']); d2 = (MAPS['run2'][n][f'v{p}_{lab}'] - MAPS['run2'][n][f'v{p}'])
            c = cov(d1.ravel(), d2.ravel())
            P(f'      reproducible range difference (run1 x run2): cov {c:+.4f}  r {pear(d1.ravel(), d2.ravel()):+.2f}  signal rms {np.sqrt(max(c,0)):.3f} (um/ns)')

# ============================================================ PART 3
P('\n##### PART 3: which electrode')
def grad(A, axis, n, size_mm):
    """central difference along axis (0 = U/x, 1 = W/y); edge cells NaN. A in units / ; returns per mm."""
    d = size_mm / n
    out = np.full_like(A, np.nan)
    if axis == 0:
        out[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * d)
    else:
        out[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2 * d)
    return out
SIZE = (340.0, 302.0)

def gapmaps(F, part, beta):
    """returns dict of delta-g maps in mm: from T (beta), from v*T."""
    return {'T': G0 * F['T'][part] / (1 + beta), 'vT': G0 * (F['lnvm'][part] + F['T'][part])}

res3 = {}
for n in (4, 6, 8):
    F1, F2 = fields(n, 'run1'), fields(n, 'run2')
    for part in ('fine', 'tot'):
        for gk, beta in (('T', 0.68), ('T', 1.0), ('vT', None)):
            g1 = gapmaps(F1, part, beta or 0.68)[gk]; g2 = gapmaps(F2, part, beta or 0.68)[gk]
            lab = f'{gk}' + (f'(b={beta})' if beta else '')
            for p, ax in (('x', 0), ('y', 1)):
                t1, t2 = F1['tau' + p][part], F2['tau' + p][part]
                gg1, gg2 = grad(g1, ax, n, SIZE[ax]), grad(g2, ax, n, SIZE[ax])
                s = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (t1.ravel(), t2.ravel(), gg1.ravel(), gg2.ravel()), nb=3000)
                rr = pear(t1.ravel(), gg2.ravel()), pear(t2.ravel(), gg1.ravel())
                P(f'n={n} {part:4s} tau_{p} vs grad_{p}(dg from {lab:9s}): k = {s[0]:+.3f} ± {s[1]:.3f} [{s[2]:+.2f},{s[3]:+.2f}] N={s[4]}  raw r (cross) {rr[0]:+.2f} {rr[1]:+.2f}')
                res3[(n, part, lab, p)] = s
                J[f'p3_{n}_{part}_{lab}_{p}'] = dict(k=s[0], err=s[1], lo=s[2], hi=s[3], N=s[4])
    # hp vs dg (fine) direct, hp from both views and avg
    for part in ('fine', 'tot'):
        for gk, beta in (('T', 0.68), ('vT', None)):
            g1 = gapmaps(F1, part, beta or 0.68)[gk]; g2 = gapmaps(F2, part, beta or 0.68)[gk]
            for hk in ('hpx', 'hpy', 'hpm'):
                if hk == 'hpm':
                    h1, h2 = 0.5 * (F1['hpx'][part] + F1['hpy'][part]), 0.5 * (F2['hpx'][part] + F2['hpy'][part])
                else:
                    h1, h2 = F1[hk][part], F2[hk][part]
                s = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (h1.ravel(), h2.ravel(), g1.ravel(), g2.ravel()), nb=3000)
                P(f'n={n} {part:4s} {hk} vs dg({gk}): slope {s[0]:+.3f} ± {s[1]:.3f} [{s[2]:+.2f},{s[3]:+.2f}]  (mesh -1, cathode 0)')
                J[f'p3h_{n}_{part}_{gk}_{hk}'] = dict(k=s[0], err=s[1], lo=s[2], hi=s[3])
    # tau vs grad hp (mean of both views): mesh-only => +0.5 (tau = grad h /2 ... see text), also tau vs [grad hp, grad dg] 2-regressor
    for part in ('fine',):
        h1, h2 = 0.5 * (F1['hpx'][part] + F1['hpy'][part]), 0.5 * (F2['hpx'][part] + F2['hpy'][part])
        g1 = gapmaps(F1, part, 0.68)['vT']; g2 = gapmaps(F2, part, 0.68)['vT']
        for p, ax in (('x', 0), ('y', 1)):
            t1, t2 = F1['tau' + p][part], F2['tau' + p][part]
            gh1, gh2 = grad(h1, ax, n, SIZE[ax]), grad(h2, ax, n, SIZE[ax])
            gg1, gg2 = grad(g1, ax, n, SIZE[ax]), grad(g2, ax, n, SIZE[ax])
            s = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (t1.ravel(), t2.ravel(), gh1.ravel(), gh2.ravel()), nb=3000)
            P(f'n={n} fine tau_{p} vs grad hp: {s[0]:+.3f} ± {s[1]:.3f}  (model tau = grad h + grad dg/2)')
            J[f'p3t_{n}_{p}_gradhp'] = dict(k=s[0], err=s[1])
            # two-regressor: tau = a grad h + b grad dg
            def two(t1, t2, h1_, h2_, g1_, g2_):
                C = np.array([[0.5 * (cov(h1_, h2_) + cov(h2_, h1_)), 0.5 * (cov(h1_, g2_) + cov(h2_, g1_))],
                              [0.5 * (cov(g1_, h2_) + cov(g2_, h1_)), 0.5 * (cov(g1_, g2_) + cov(g2_, g1_))]])
                c = np.array([0.5 * (cov(h1_, t2) + cov(h2_, t1)), 0.5 * (cov(g1_, t2) + cov(g2_, t1))])
                return np.linalg.solve(C, c)[0], np.linalg.solve(C, c)[1]
            ok = np.all([np.isfinite(x.ravel()) for x in (t1, t2, gh1, gh2, gg1, gg2)], axis=0)
            arr = [x.ravel()[ok] for x in (t1, t2, gh1, gh2, gg1, gg2)]
            a0, b0 = two(*arr)
            rng = np.random.default_rng(5); bs = []
            for _ in range(2000):
                i = rng.integers(0, ok.sum(), ok.sum())
                try: bs.append(two(*[x[i] for x in arr]))
                except Exception: pass
            bs = np.array(bs)
            P(f'n={n} fine tau_{p} = a grad h + b grad dg: a {a0:+.2f} ± {np.std(bs[:,0]):.2f}, b {b0:+.2f} ± {np.std(bs[:,1]):.2f}   (all-mesh: 1, 0.5 ; all-cathode: 0, 0.5 .. sign see text)')
            J[f'p3t2_{n}_{p}'] = dict(a=a0, a_err=float(np.std(bs[:, 0])), b=b0, b_err=float(np.std(bs[:, 1])))

# (i) quadratic parts, analytic, bootstrap over cells with refits
P('\n-- (i) quadratic parts: k = 100 dtau/dU / G_uu')
def smooth_k(n, beta, gk, rng=None):
    out = {}
    for run in RUNS:
        M = MAPS[run][n]
        N_ = n * n
        idx = np.arange(N_) if rng is None else rng.integers(0, N_, N_)
        UU, WW = grid(n)
        def fit(y, e, deg):
            y, e, U, W = y.ravel()[idx], e.ravel()[idx], UU.ravel()[idx], WW.ravel()[idx]
            ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
            A = basis(U[ok], W[ok], deg); w = 1 / e[ok]
            return np.linalg.lstsq(A * w[:, None], y[ok] * w, rcond=None)[0]
        yT, eT = lnmap(M, 'T')
        cT = fit(yT, eT, 2)
        if gk == 'T':
            cg = cT * G0 / (1 + beta)
        else:
            cvs = []
            for p in 'xy':
                yv, ev = lnmap(M, 'v' + p); cvs.append(fit(yv, ev, 2))
            cg = (cT + 0.5 * (cvs[0] + cvs[1])) * G0
        Guu, Gww = 2 * cg[3], 2 * cg[4]
        Gu, Gw = cg[1], cg[2]
        ck = {}
        for p in 'xy':
            c1 = fit(M['w0' + p], M['w0' + p + '_err'], 1)
            v = MAPS[run]['glob'][f'{p}_t06'][0]
            ck[p] = c1[1] / v if p == 'x' else c1[2] / v     # d tau_p / d(own coord) per 100 mm
        out[run] = dict(Guu=Guu, Gww=Gww, Gu=Gu, Gw=Gw, tx=ck['x'], ty=ck['y'], kx=100 * ck['x'] / Guu, ky=100 * ck['y'] / Gww)
    return out
for n in (4, 6, 8):
    for gk, beta in (('T', 0.68), ('T', 1.0), ('vT', 0.68)):
        o = smooth_k(n, beta, gk)
        rng = np.random.default_rng(2); bs = [smooth_k(n, beta, gk, rng) for _ in range(300)]
        for run in RUNS:
            e = lambda key: np.nanstd([b[run][key] for b in bs])
            P(f'n={n} {gk}(b={beta}) {run}: G_uu {o[run]["Guu"]:+.2f}±{e("Guu"):.2f}  G_ww {o[run]["Gww"]:+.2f}±{e("Gww"):.2f} mm/(100mm)^2;  '
              f'tau_x slope {o[run]["tx"]*1e3:+.2f}e-3 /100mm, tau_y slope {o[run]["ty"]*1e3:+.2f}e-3;  k_x {o[run]["kx"]:+.2f}  k_y {o[run]["ky"]:+.2f}  (tau slope in mm/(100mm)^2 = 100*slope: x {100*o[run]["tx"]:.2f}, y {100*o[run]["ty"]:.2f})')
            J[f'p3i_{n}_{gk}{beta}_{run}'] = dict(Guu=o[run]['Guu'], Gww=o[run]['Gww'], tx=o[run]['tx'], ty=o[run]['ty'], Guu_e=e('Guu'), Gww_e=e('Gww'))
P('audited joint fit: G_uu +0.28, G_ww -1.09 mm/(100mm)^2; (h+c)/2 curvature 0.92 (x), 0.77 (y): k_x = 3.3, k_y = -0.7')

# ============================================================ PART 4
P('\n##### PART 4: focusing')
P('derivation: tau_audit(x) = tau0 + a x, drift line slope dx/dz = -tau, charge uniform in z, time ~ z.  Landing x_L(z) = x_c + (t + tau_c) z + a (t + tau0/2) z^2,')
P('with x_c the track at the mesh, tau_c = tau(x_c).  Linear fit slope over z in [0,g] = coefficient of z + g * (coef of z^2).  x_c = x_m - t g/2 for a cell indexed by the mid-gap position x_m.')
P('=> dx_L/d(vt_a) = t (1 + a g) + tau(x_m) - a g t/2 + const => v_fit = v (1 + a g / 2): f = 1/2 exactly (f = 1 if cells were indexed at the mesh).')
for n in (4, 6, 8):
    F1, F2 = fields(n, 'run1'), fields(n, 'run2')
    for beta in (0.68, 1.0):
        for part in ('tot', 'fine'):
            for p, ax in (('x', 0), ('y', 1)):
                R = []; X = []
                for F in (F1, F2):
                    Rr = F['v' + p][part] + beta / (1 + beta) * F['T'][part]
                    X.append(G0 * grad(F['tau' + p][part], ax, n, SIZE[ax]) * 1.0)    # g dtau/dx (dimensionless)
                    R.append(Rr)
                s = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (R[0].ravel(), R[1].ravel(), X[0].ravel(), X[1].ravel()), nb=3000)
                sigR = cov(R[0].ravel(), R[1].ravel()); sigX = cov(X[0].ravel(), X[1].ravel())
                frac = s[0] ** 2 * sigX / sigR if sigR > 0 else np.nan
                frac_f = 0.25 * sigX / sigR if sigR > 0 else np.nan
                P(f'n={n} beta={beta} {part:4s} view {p}: f = {s[0]:+.2f} ± {s[1]:.2f} [{s[2]:+.2f},{s[3]:+.2f}]  signal rms of R {np.sqrt(max(sigR,0))*100:.2f} %  rms of g dtau/dx {np.sqrt(max(sigX,0))*100:.3f} %  '
                  f'variance of R removed at f=0.5: {frac_f*100:.1f} %, at fitted f: {frac*100:.1f} %')
                J[f'p4_{n}_{beta}_{part}_{p}'] = dict(f=s[0], err=s[1], sigR=sigR, sigX=sigX, frac_f05=frac_f, frac_fit=frac)

# ============================================================ PART 5
P('\n##### PART 5: how much reproducible variance is explained by a common (dh, dc) pair')
def explained(M1, M2, regs1, regs2, nb=1500, fixed=None):
    """fraction of cov(M1,M2) explained by a cross-run regression on regressors (list of arrays); or by a fixed prediction."""
    def f(*arrs):
        m1, m2 = arrs[0], arrs[1]; r = arrs[2:]
        k = len(r) // 2
        r1, r2 = r[:k], r[k:]
        tot = cov(m1, m2)
        if fixed is not None:
            p1 = sum(c * x for c, x in zip(fixed, r1)); p2 = sum(c * x for c, x in zip(fixed, r2))
            return 1 - cov(m1 - p1, m2 - p2) / tot
        C = np.array([[0.5 * (cov(r1[i], r2[j]) + cov(r2[i], r1[j])) for j in range(k)] for i in range(k)])
        c = np.array([0.5 * (cov(r1[i], m2) + cov(r2[i], m1)) for i in range(k)])
        return float(c @ np.linalg.solve(C, c)) / tot
    return boot_cells(f, [M1, M2] + list(regs1) + list(regs2), nb=nb)
for n in (6, 8):
    F1, F2 = fields(n, 'run1'), fields(n, 'run2')
    f = lambda F, k: F[k]['fine'].ravel()
    hm = lambda F: 0.5 * (F['hpx']['fine'] + F['hpy']['fine']).ravel()
    tests = [
        ('T <- v_x, v_y', 'T', ['vx', 'vy']), ('v_x <- T', 'vx', ['T']), ('v_y <- T', 'vy', ['T']),
        ('v_x <- T, v_y', 'vx', ['T', 'vy']), ('v_y <- T, v_x', 'vy', ['T', 'vx']),
        ('hp_x <- hp_y', 'hpx', ['hpy']), ('hp_y <- hp_x', 'hpy', ['hpx']),
        ('hp_x <- T', 'hpx', ['T']), ('hp_x <- T, hp_y', 'hpx', ['T', 'hpy']), ('hp_y <- T, hp_x', 'hpy', ['T', 'hpx']),
    ]
    for lab, tgt, regs in tests:
        s = explained(f(F1, tgt), f(F2, tgt), [f(F1, r) for r in regs], [f(F2, r) for r in regs])
        P(f'n={n} fine: {lab:18s}: explained fraction of the reproducible variance {s[0]:.2f} ± {s[1]:.2f} [{s[2]:.2f},{s[3]:.2f}] (N={s[4]})')
        J[f'p5_{n}_{lab}'] = dict(frac=s[0], err=s[1])
    # tau gradients: fixed model grad h + grad dg/2, with h from hp mean (mm) and dg from vT (mm); free 2 regressors
    for p, ax in (('x', 0), ('y', 1)):
        gh = lambda F: grad(0.5 * (F['hpx']['fine'] + F['hpy']['fine']), ax, n, SIZE[ax]).ravel()
        gg = lambda F: grad(G0 * (F['lnvm']['fine'] + F['T']['fine']), ax, n, SIZE[ax]).ravel()
        t1, t2 = F1['tau' + p]['fine'].ravel(), F2['tau' + p]['fine'].ravel()
        s = explained(t1, t2, [gh(F1), gg(F1)], [gh(F2), gg(F2)], fixed=(1.0, 0.5))
        s2 = explained(t1, t2, [gh(F1), gg(F1)], [gh(F2), gg(F2)])
        s3 = explained(t1, t2, [gg(F1)], [gg(F2)])
        s4 = explained(t1, t2, [gh(F1)], [gh(F2)])
        P(f'n={n} fine tau_{p}: fixed model grad h + grad dg/2 explains {s[0]:.2f} ± {s[1]:.2f}; free (grad h, grad dg) {s2[0]:.2f} ± {s2[1]:.2f}; grad dg alone {s3[0]:.2f} ± {s3[1]:.2f}; grad h alone {s4[0]:.2f} ± {s4[1]:.2f}')
        J[f'p5_{n}_tau{p}'] = dict(fixed=s[0], fixed_e=s[1], free=s2[0], free_e=s2[1], gg=s3[0], gh=s4[0])
json.dump(J, open(f'{OUT}/part2_5.json', 'w'), indent=1, default=float)
open(f'{OUT}/part2_5.log', 'w').write('\n'.join(LOG))

# ============================================================ figures
n = 8
F = {r: fields(n, r) for r in RUNS}
fig, ax = plt.subplots(2, 6, figsize=(15, 5.2))
cols = [('T fine (%)', lambda f: 100 * f['T']['fine'], 3), ('v*T fine (%)', lambda f: 100 * (f['lnvm']['fine'] + f['T']['fine']), 3),
        ('hp (x+y)/2 fine (mm)', lambda f: 0.5 * (f['hpx']['fine'] + f['hpy']['fine']), 1.2), ('v_x*... v mean fine (%)', lambda f: 100 * f['lnvm']['fine'], 1.5),
        ('tau_x fine (1e-3)', lambda f: 1e3 * f['taux']['fine'], 4), ('tau_y fine (1e-3)', lambda f: 1e3 * f['tauy']['fine'], 4)]
for i, r in enumerate(RUNS):
    for j, (t, fn, vm) in enumerate(cols):
        A = fn(F[r]); M = MAPS[r][n]
        im = ax[i, j].imshow(A.T, origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm, extent=[M['U'][0] - .21, M['U'][-1] + .21, M['W'][0] - .19, M['W'][-1] + .19])
        ax[i, j].set_title(f'{t}\n{r}', fontsize=8); ax[i, j].tick_params(labelsize=7)
        if j == 0: ax[i, j].set_ylabel('W (100 mm)')
        ax[i, j].set_xlabel('U (100 mm)')
        plt.colorbar(im, ax=ax[i, j], fraction=0.046)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_fine_maps_n8.png', dpi=130); plt.close()

fig, ax = plt.subplots(2, 3, figsize=(11, 6.5))
for j, p in enumerate('xy'):
    axx = 0 if p == 'x' else 1
    gs = [grad(G0 * (F[r]['lnvm']['fine'] + F[r]['T']['fine']), axx, n, SIZE[axx]) for r in RUNS]
    ts = [F[r]['tau' + p]['fine'] for r in RUNS]
    a = ax[0, j]
    a.plot(gs[1].ravel() * 1e3, ts[0].ravel() * 1e3, 'o', ms=3, label='tau run1 vs grad dg run2')
    a.plot(gs[0].ravel() * 1e3, ts[1].ravel() * 1e3, 's', ms=3, label='tau run2 vs grad dg run1')
    xx = np.linspace(-15, 15, 2); a.plot(xx, 0.5 * xx, 'k--', lw=.8, label='k=+0.5 (cathode)'); a.plot(xx, -0.5 * xx, 'k:', lw=.8, label='k=-0.5 (mesh)')
    a.set_xlabel(f'd(dg)/d{p}  from v*T  (1e-3)'); a.set_ylabel(f'tau_{p} fine (1e-3)'); a.legend(fontsize=6); a.set_title(f'n=8 fine, view {p}', fontsize=9)
# hp vs dg
a = ax[0, 2]
g = [G0 * (F[r]['lnvm']['fine'] + F[r]['T']['fine']) for r in RUNS]; h = [0.5 * (F[r]['hpx']['fine'] + F[r]['hpy']['fine']) for r in RUNS]
a.plot(g[1].ravel(), h[0].ravel(), 'o', ms=3); a.plot(g[0].ravel(), h[1].ravel(), 's', ms=3)
xx = np.linspace(-1.5, 1.5, 2); a.plot(xx, -xx, 'k:', lw=.8, label='mesh (-1)'); a.plot(xx, 0 * xx, 'k--', lw=.8, label='cathode (0)')
a.set_xlabel('fine dg from v*T, other run (mm)'); a.set_ylabel('fine hp (x+y)/2 (mm)'); a.legend(fontsize=6)
# part 2 scatter
for j, vv in enumerate(('vx', 'vy')):
    a = ax[1, j]
    a.plot(F['run2']['T']['fine'].ravel() * 100, F['run1'][vv]['fine'].ravel() * 100, 'o', ms=3, label='v run1 vs T run2')
    a.plot(F['run1']['T']['fine'].ravel() * 100, F['run2'][vv]['fine'].ravel() * 100, 's', ms=3, label='v run2 vs T run1')
    xx = np.array([-6, 6]); a.plot(xx, -0.405 * xx, 'k--', lw=.8, label='gap, beta=0.68'); a.plot(xx, -xx, 'k:', lw=.8, label='gas')
    a.set_xlabel('fine ln T (%)'); a.set_ylabel(f'fine ln {vv} (%)'); a.legend(fontsize=6); a.set_title('n=8 fine', fontsize=9)
a = ax[1, 2]
for p, c in (('x', 'C0'), ('y', 'C1')):
    axx = 0 if p == 'x' else 1
    Rs = [F[r]['v' + p]['fine'] + 0.68 / 1.68 * F[r]['T']['fine'] for r in RUNS]
    Xs = [G0 * grad(F[r]['tau' + p]['fine'], axx, n, SIZE[axx]) for r in RUNS]
    a.plot(Xs[1].ravel() * 100, Rs[0].ravel() * 100, 'o', ms=3, color=c, label=f'view {p} run1 vs run2')
    a.plot(Xs[0].ravel() * 100, Rs[1].ravel() * 100, 's', ms=3, color=c, mfc='none')
xx = np.array([-.5, .5]); a.plot(xx, 0.5 * xx, 'k--', lw=.8, label='f=0.5')
a.set_xlabel('g dtau/dx (%)'); a.set_ylabel('v residual after gap term (%)'); a.legend(fontsize=6); a.set_title('focusing, n=8 fine', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_scatter_n8.png', dpi=130); plt.close()
