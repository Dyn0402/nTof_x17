"""r2_06_cells.py -- cell-level numbers (fine-part test with the plane removed, field-sample comparison), gap-map summary, figures 1-2."""
import sys, pickle, json
from r2_lib import *
import r2_04_compare as C4
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FL = C4.FL
CP = pickle.load(open(f'{OUT}/compare.pkl', 'rb'))
rng = np.random.default_rng(8)
X, Y = np.meshgrid(C4.CEN, C4.CEN, indexing='ij')


def plane_rm(Z, w):
    A = np.c_[np.ones(Z.size), X.ravel(), Y.ravel()]
    sw = np.sqrt(w.ravel())
    c = np.linalg.lstsq(A * sw[:, None], Z.ravel() * sw, rcond=None)[0]
    return Z - (A @ c).reshape(Z.shape)


def wcorr(a, b, w):
    am, bm = np.average(a, weights=w), np.average(b, weights=w)
    return float(np.sum(w * (a - am) * (b - bm)) / np.sqrt(np.sum(w * (a - am) ** 2) * np.sum(w * (b - bm) ** 2)))


rows = []
for arm in 'AC':
    for view in 'xy':
        nt = C4.ntof_map(arm, view)
        w = nt['n'].astype(float)
        for fk in [(arm, 'late', 'all'), (arm, 'cosmic', 'all'), (arm, 'late', 'par0'), (arm, 'late', 'par1'), (arm, 'late', 'early'),
                   (arm, 'late', 'lateC'), (arm, 'late', 'tan>0.3'), (arm, 'late', 'notan'), (arm, 'late', 'x_q_u90'), (arm, 'late', 'y_q_u90'),
                   (arm, 'late', 'K8_l0.1'), (arm, 'late', 'K12_l0.1')]:
            if fk not in FL: continue
            c = FL[fk]['c']
            i = 0 if view == 'x' else 1
            sg = SGN[(arm, view)]
            Kf = FL[fk].get('K', 10)
            G = (sg * C4.NOM * (lambda g: g[i])(C4.grad_lnT(c))) if Kf == 10 else None
            if G is None:
                C4.SP = Spl(Kf); sp_old = C4.SP
                G = sg * C4.NOM * C4.grad_lnT(c)[i]
                C4.SP = Spl(10)
            a_fine, G_fine = plane_rm(nt["a"], w).ravel(), plane_rm(G, w).ravel()
            r_fine = wcorr(a_fine, G_fine, w.ravel())
            # slope of a_fine on G_fine (b')
            sl = float(np.sum(w.ravel() * a_fine * G_fine) / np.sum(w.ravel() * G_fine ** 2))
            bs_r, bs_s = [], []
            for _ in range(300):
                ix = rng.integers(0, w.size, w.size)
                af, gf, ww_ = a_fine[ix], G_fine[ix], w.ravel()[ix]
                bs_r.append(wcorr(af, gf, ww_)); bs_s.append(np.sum(ww_ * af * gf) / np.sum(ww_ * gf ** 2))
            rows.append(dict(arm=arm, view=view, field='/'.join(fk[1:]), r_fine=r_fine, r_fine_err=float(np.std(bs_r)), slope_fine=sl,
                             slope_fine_err=float(np.std(bs_s)), r_full=wcorr(nt["a"].ravel(), G.ravel(), w.ravel()),
                             sd_ain_fine=float(np.sqrt(np.average(a_fine ** 2, weights=w.ravel()))), sd_G_fine=float(np.sqrt(np.average(G_fine ** 2, weights=w.ravel())))))
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/cells_fine.csv', index=False)
pd.set_option('display.width', 250, 'display.max_rows', 200)
print(df.round(3).to_string())

# ---- gap-map summary (late and cosmic) per chamber
uu, ww = grid(60)
INS = (np.abs(uu.ravel()) < 1.8) & (np.abs(ww.ravel()) < 1.8)
SP = Spl(10)
BG = SP.basis(uu.ravel(), ww.ravel())
fac = C4.G0 / (1 + C4.BETA)
summ = {}
maps = {}
for arm in 'ABCD':
    for w_ in ('late', 'cosmic'):
        if (arm, w_, 'all') not in FL: continue
        m = fac * (BG @ FL[(arm, w_, 'all')]['c'])
        maps[(arm, w_)] = m
        U, W = uu.ravel()[INS] * 100, ww.ravel()[INS] * 100
        A_ = np.c_[np.ones(INS.sum()), U, W, U * U / 2, W * W / 2, U * W]
        cc = np.linalg.lstsq(A_, m[INS], rcond=None)[0]
        res = m[INS] - A_ @ cc
        cb = np.array([fac * (BG @ c) for c in FL[(arm, w_, 'boot')]['cb']])
        summ[f'{arm} {w_}'] = dict(sd_mm=float(m[INS].std()), p2p_mm=float(np.ptp(m[INS])), tilt_x_mm_per100=float(cc[1]), tilt_y_mm_per100=float(cc[2]),
                                    curv_xx=float(cc[3]), curv_yy=float(cc[4]), curv_xy=float(cc[5]), sd_after_quadratic_mm=float(res.std()),
                                    noise_mm=float(cb.std(0)[INS].mean()), n=FL[(arm, w_, 'all')]['n'], dlnT_sd=float(m[INS].std() / fac))
json.dump(summ, open(f'{OUT}/gapmap_summary.json', 'w'), indent=1)
for k, v in summ.items(): print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})

# ---- Figure 1: gap maps, local frame (x_local = -100U - 0.71 ; y_local = 100 W + 0.71) -> plot against local coordinates
fig, ax = plt.subplots(2, 4, figsize=(13, 6.6), constrained_layout=True)
for j, arm in enumerate('ABCD'):
    for i, w_ in enumerate(('late', 'cosmic')):
        a = ax[i, j]
        if (arm, w_) not in maps: a.axis('off'); continue
        m = maps[(arm, w_)].reshape(uu.shape)
        xl, yl = -100 * uu - 0.71, 100 * ww + 0.71
        im = a.pcolormesh(xl, yl, np.where((np.abs(uu) < 1.95) & (np.abs(ww) < 1.95), m, np.nan), vmin=-1.2, vmax=1.2, cmap='RdBu_r', shading='auto')
        a.set_aspect('equal'); a.set_title(f'{arm}  {"late beam" if w_ == "late" else "beam-off"}  (n = {FL[(arm, w_, "all")]["n"]:,})', fontsize=9)
        a.set_xlabel('x_local (mm)'); a.set_ylabel('y_local (mm)')
fig.colorbar(im, ax=ax, shrink=0.8, label=r'$\delta g$ (mm), $\beta$ = 0.68, $g_0$ = 28.2 mm')
fig.savefig(f'{OUT}/fig1_gapmaps.png', dpi=110)

# ---- Figure 2: in-situ tilt, timing-predicted (G alone), and best model, A and C, x and y
fig, ax = plt.subplots(4, 4, figsize=(12.5, 11.5), constrained_layout=True)
for r_, (arm, view) in enumerate([('A', 'x'), ('A', 'y'), ('C', 'x'), ('C', 'y')]):
    o = CP[f'{arm}_{view}_late']['_arrays']
    # in-situ residual, pred G nominal, pred full k=1, pred full fitted (res_fit)
    sg = SGN[(arm, view)]
    ain, G, Hh, ap = o['a_in'], o['G'], o['H'], o['apoly']
    w = C4.ntof_map(arm, view)['n'].astype(float)
    fitc = CP[f'{arm}_{view}_late']
    pk1 = Hh + G - ap
    pf = fitc['res_fit_k'] * Hh + fitc['res_fit_bratio'] * G - ap
    pan = [(ain, 'in-situ residual map a (cosmic lines)'), (G - np.average(G, weights=w), 'timing: 0.5 grad g (nominal)'),
           (pk1 - np.average(pk1, weights=w), 'timing + dish k = 1 - block'), (pf - np.average(pf, weights=w), f'fitted k = {fitc["res_fit_k"]:.2f}, b\' = {fitc["res_fit_bratio"]:.2f}')]
    v = 0.04
    for c_, (Z, t) in enumerate(pan):
        a = ax[r_, c_]
        im = a.imshow((Z - np.nanmedian(Z) if c_ == 0 else Z).T, origin='lower', extent=[-200, 200, -200, 200], vmin=-v, vmax=v, cmap='RdBu_r')
        a.set_title(f'{arm} {view}: {t}', fontsize=8)
        a.set_xlabel('x_local (mm)'); a.set_ylabel('y_local (mm)')
fig.colorbar(im, ax=ax, shrink=0.5, label='tilt (tan)')
fig.savefig(f'{OUT}/fig2_tilt_maps.png', dpi=100)
