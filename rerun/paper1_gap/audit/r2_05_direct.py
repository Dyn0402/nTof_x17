"""r2_05_direct.py -- held-out single-view sigma on the 31,387 A-C cosmic lines (exactly g_direct.py's protocol), with timing-predicted tilt models.
Train on one run parity, test on the other (both ways, averaged).  Models (per chamber and view; all start from the unmapped v(t) slope):
  none; global (c + gamma j); insitu (8x8 map, parity-trained);
  tim_nom_k0 / tim_nom_k1: global + [k H + NOM*G - AP] with k = 0 / 1 (bench dish absent / present at full size), b nominal (0.5, beta 0.68)
  tim_fit_k0 / tim_fit_k1: same with b free (fitted on the training parity)
  tim_fit_kb: k and b both free
  tim_fit_kb + bench fine (A only): + alpha * bench fine shape (G's bench map, free amplitude)
 with H = SGN dh_p/dstrip (bench h block), AP = SGN a_poly (pass-1 bow block, subtracted from the template since the lines are built on slopes that already include it:
 r + AP = c + gamma j + k H + b G),  G = SGN * NOM * dlnT/dstrip_p from the n_TOF drift-time field of the chosen sample (late beam = independent of the lines)."""
import sys, os, pickle, json
os.environ.setdefault('X17_TQ_OUT', '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/g/tq')
from r2_lib import *
import r2_04_compare as C4
import g_direct as GD, g_common as GC
from rerun.ntof.improve import angle_map as AM
FIELD = sys.argv[1] if len(sys.argv) > 1 else 'late'
NBOOT = 200
LINES = '/media/ucla/work/ntof_pass1/ana_r2/track_quality/lines.parquet'
FL = C4.FL
SPL = C4.SP
M = pd.read_parquet(LINES)
rn = M.run.str.replace('run_', '').astype(int)
rng = np.random.default_rng(21)
BOWJ = {a: json.load(open(f'{C4.BUND}/{a}/bundle.json'))['bow'] for a in 'AC'}


def feats(Mr, s, p, arm, c):
    """H, AP, G at the lines' positions of chamber s (rows Mr)"""
    U = (Mr[f'x_p0_{s}'].to_numpy(float) + MID * Mr[f'x_tan_theta_{s}'].to_numpy(float) - 200) / 100
    Wc = (Mr[f'y_p0_{s}'].to_numpy(float) + MID * Mr[f'y_tan_theta_{s}'].to_numpy(float) - 200) / 100
    U = np.clip(U, -2.0, 2.0); Wc = np.clip(Wc, -2.0, 2.0)
    sg = SGN[(arm, p)]
    b = BOWJ[arm]
    AP = sg * C4.R._bow_poly(b[f'a_{p}'], U, Wc)
    h = b[f'h_{p}']
    H = sg * ((h['gu'] + h['kuu'] * U + h['kuw'] * Wc) / 100 if p == 'x' else (h['gw'] + h['kww'] * Wc + h['kuw'] * U) / 100)
    i = 1 if p == 'y' else 0
    G = sg * C4.NOM * ((SPL.basis(U, Wc, 1, 0) @ c) / 100 if p == 'x' else (SPL.basis(U, Wc, 0, 1) @ c) / 100)
    return H, AP, G


def fit_lin(cols, y, clip=3.0):
    X = np.vstack(cols).T
    keep = np.ones(len(y), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        res = y - X @ coef
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < clip * sig
    cov = np.linalg.pinv(X[keep].T @ X[keep]) * sig ** 2
    return coef, np.sqrt(np.diag(cov))


def sig_boot(v, rng):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    b = [GC.mad(v[rng.integers(0, len(v), len(v))]) for _ in range(NBOOT // 2)]
    return GC.mad(v), float(np.std(b))


rows, amps = [], []
for s, arm in (('a', 'A'), ('c', 'C')):
    fkey = (arm, FIELD, 'all')
    c = FL[fkey]['c']
    for p in 'xy':
        Gc = AM.clean_lines(M, s, p)
        parc = Gc.run.str.replace('run_', '').astype(int) % 2
        # features for ALL lines (test) and for clean lines (train / clean test)
        Hall, APall, Gall = feats(M, s, p, arm, c)
        Hc, APc, Gcl = (z[M.index.get_indexer(Gc.index)] for z in (Hall, APall, Gall))
        a_b, s_b, a_sh, s_sh, a0b, s0b = GD.shape(arm, p, 0.0)
        ash_all = AM.interp(dict(a=a_sh, s=np.zeros_like(a_sh), centres=GC.CENTRES, a0=0.0, s0=0.0), M[f'x_local_{s}'].to_numpy(), M[f'y_local_{s}'].to_numpy())[0]
        ash_c = ash_all[M.index.get_indexer(Gc.index)]
        for tr in (0, 1):
            trm = (parc == tr).to_numpy()
            tem = ((rn % 2) != tr).to_numpy()
            j = M.loc[tem, f'j{p}'].to_numpy()
            slope = (M.loc[tem, f'd_{p}_{s}'] / M.loc[tem, f'd_z_{s}']).to_numpy()
            g = np.abs(j) < 0.45
            Gs = Gc[~trm]
            slope_c = Gs.r.to_numpy() + Gs.j.to_numpy()
            Ht, APt, Gt_ = Hall[tem], APall[tem], Gall[tem]
            Hs, APs, Gs_ = Hc[~trm], APc[~trm], Gcl[~trm]
            ash_t, ash_s = ash_all[tem], ash_c[~trm]
            jt, rt = Gc.j.to_numpy()[trm], Gc.r.to_numpy()[trm]
            a_g, s_g, _ = AM.robust_line(jt, rt)

            def evals(tag, corr_all, corr_clean):
                sa, ea = sig_boot((corr_all - j)[g], rng)
                sc_, ec = sig_boot(corr_clean - Gs.j.to_numpy(), rng)
                rows.append(dict(arm=arm, view=p, train=tr, model=tag, sig_all=sa, err_all=ea, sig_clean=sc_, err_clean=ec,
                                 tail_all=float(np.mean(np.abs(corr_all - j)[g] > 0.15)), n_all=int(g.sum()), n_clean=len(Gs)))

            const = lambda cc, gm: dict(a=np.full((8, 8), cc), s=np.full((8, 8), gm), centres=GC.CENTRES, a0=cc, s0=gm)  # noqa: E731
            evals('none', slope, slope_c)
            mp = const(a_g, s_g)
            evals('global', AM.correct(slope, M.loc[tem, f'x_local_{s}'].to_numpy(), M.loc[tem, f'y_local_{s}'].to_numpy(), mp),
                  AM.correct(slope_c, Gs.x.to_numpy(), Gs.y.to_numpy(), mp))
            mi = AM.fit_map(Gc[trm]) | dict(centres=GC.CENTRES)
            evals('insitu', AM.correct(slope, M.loc[tem, f'x_local_{s}'].to_numpy(), M.loc[tem, f'y_local_{s}'].to_numpy(), mi),
                  AM.correct(slope_c, Gs.x.to_numpy(), Gs.y.to_numpy(), mi))
            y_tr = rt + APc[trm]          # r + AP
            Htr, Gtr, Atr = Hc[trm], Gcl[trm], ash_c[trm]

            def model(tag, cols_tr, cols_te, cols_s):
                """cols: list of regressor arrays for [one, j, ...]; fits y_tr = X coef, applies slope_corr = (slope - tilt)/(1+gamma)"""
                coef, err = fit_lin([np.ones_like(jt), jt] + cols_tr, y_tr)
                tilt = lambda cc, cols: coef[0] + sum(coef[2 + q] * cols[q] for q in range(len(cols))) - 0.0   # noqa: E731
                ca = (slope - (coef[0] + sum(coef[2 + q] * cols_te[q] for q in range(len(cols_te))) - APt)) / (1 + coef[1])
                cc = (slope_c - (coef[0] + sum(coef[2 + q] * cols_s[q] for q in range(len(cols_s))) - APs)) / (1 + coef[1])
                evals(tag, ca, cc)
                amps.append(dict(arm=arm, view=p, train=tr, model=tag, coef=[float(x) for x in coef], err=[float(x) for x in err]))

            # nominal b = 1 (G already carries NOM), k = 0 / 1: coefficients on H and G fixed -> fit only const and gamma on y - kH - G
            for kf in (0, 1):
                y2 = y_tr - kf * Htr - Gtr
                coef, _ = fit_lin([np.ones_like(jt), jt], y2)
                ca = (slope - (coef[0] + kf * Ht + Gt_ - APt)) / (1 + coef[1])
                cc = (slope_c - (coef[0] + kf * Hs + Gs_ - APs)) / (1 + coef[1])
                evals(f'tim_nom_k{kf}', ca, cc)
                # dish only (no gap term): isolates the H part
                y3 = y_tr - kf * Htr
                coef, _ = fit_lin([np.ones_like(jt), jt], y3)
                evals(f'dish_only_k{kf}', (slope - (coef[0] + kf * Ht - APt)) / (1 + coef[1]), (slope_c - (coef[0] + kf * Hs - APs)) / (1 + coef[1]))
            for kf in (0, 1):
                y2 = y_tr - kf * Htr
                coef, err = fit_lin([np.ones_like(jt), jt, Gtr], y2)
                ca = (slope - (coef[0] + kf * Ht + coef[2] * Gt_ - APt)) / (1 + coef[1])
                cc = (slope_c - (coef[0] + kf * Hs + coef[2] * Gs_ - APs)) / (1 + coef[1])
                evals(f'tim_fit_k{kf}', ca, cc)
                amps.append(dict(arm=arm, view=p, train=tr, model=f'tim_fit_k{kf}', coef=[float(x) for x in coef], err=[float(x) for x in err]))
            model('tim_fit_kb', [Htr, Gtr], [Ht, Gt_], [Hs, Gs_])
            model('tim_fit_kb_bench', [Htr, Gtr, Atr], [Ht, Gt_, ash_t], [Hs, Gs_, ash_s])
            # bench fine shape with k fixed 0/1 and b nominal (what the bench map adds to the timing prediction)
            for kf in (0, 1):
                y2 = y_tr - kf * Htr - Gtr
                coef, err = fit_lin([np.ones_like(jt), jt, Atr], y2)
                ca = (slope - (coef[0] + kf * Ht + Gt_ + coef[2] * ash_t - APt)) / (1 + coef[1])
                cc = (slope_c - (coef[0] + kf * Hs + Gs_ + coef[2] * ash_s - APs)) / (1 + coef[1])
                evals(f'tim_nom_k{kf}_bench', ca, cc)
            print(arm, p, tr, 'done', flush=True)
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/direct_{FIELD}.csv', index=False)
json.dump(amps, open(f'{OUT}/direct_amps_{FIELD}.json', 'w'))
pd.set_option('display.width', 250)
T = R.groupby(['arm', 'view', 'model'], sort=False)[['sig_all', 'err_all', 'sig_clean', 'err_clean', 'tail_all']].mean()
print(T.round(4).to_string())
