"""T9 tasks 1, 3, 4: correlations of the (mapped) cosmic line residual with per-track timing predictors."""
import json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prep import D, PRED, DESC, cosmic_long, beam_long

plt.rcParams.update({'font.size': 10.5})
JB = {'all': (0, 0.45), 'shallow': (0, 0.1), 'mid': (0.1, 0.25), 'steep': (0.25, 0.45)}
CTRL = ['lnq', 'nstr', 'chi2', 'edge', 'absj', 'absjq']


def clipq(v, lo=0.01, hi=0.99):
    v = np.asarray(v, float)
    a, b = np.nanquantile(v, [lo, hi])
    return np.clip(v, a, b)


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 50 or np.std(x) == 0:
        return dict(n=n, pear=np.nan, pear_e=np.nan, spear=np.nan, spear_e=np.nan)
    pr = stats.pearsonr(x, y)[0]
    sr = stats.spearmanr(x, y)[0]
    return dict(n=n, pear=pr, pear_e=(1 - pr ** 2) / np.sqrt(n - 1), spear=sr, spear_e=1.03 / np.sqrt(n - 3))


def residualize(y, X):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    A = np.c_[np.ones(ok.sum()), X[ok], X[ok] ** 2]
    out = np.full(len(y), np.nan)
    b = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    out[ok] = y[ok] - A @ b
    return out


def main():
    L = cosmic_long()
    rows, prof = [], {}
    for (arm, p), F in L.items():
        G = F[F.clean].copy()
        G['absj'], G['absjq'] = G.j.abs(), G.jq.abs()
        sg = np.sign(G.j.to_numpy())
        for jb, (lo, hi) in JB.items():
            m = ((G.j.abs() >= lo) & (G.j.abs() < hi)).to_numpy()
            r = G.r.to_numpy()[m]
            rs = (G.r.to_numpy() * sg)[m]
            rj = (G.r / G.j).to_numpy()[m]
            Xc = G[CTRL].to_numpy(float)[m]
            for f in PRED:
                x = clipq(G[f].to_numpy(float))[m]
                d = dict(arm=arm, view=p, jbin=jb, pred=f)
                for lab, y in (('r', r), ('rsgn', rs)) + ((('rj', np.clip(rj, -1.5, 1.5)),) if lo >= 0.1 else ()):
                    rr = rho(x, y)
                    d.update({f'{lab}_{k}': v for k, v in rr.items()})
                # partial (controls: charge, strips, chi2, edge, |j|, |j_other|; the predictor itself excluded)
                keep = [i for i, c in enumerate(CTRL) if c != f]
                Xk = Xc[:, keep]
                for lab, y in (('r', r), ('rsgn', rs)):
                    pr = rho(residualize(x, Xk), residualize(y, Xk))
                    d[f'{lab}_partial'] = pr['pear']
                    d[f'{lab}_partial_spear'] = pr['spear']
                rows.append(d)
        # profiles (binned median, deciles of the predictor), r for all |j| and r*sign(j) for |j| >= 0.25
        for f in PRED:
            x = G[f].to_numpy(float)
            ok = np.isfinite(x)
            e = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, 11)))
            k = np.digitize(x, e[1:-1])
            st = G.j.abs().to_numpy() >= 0.25
            pr = []
            for b in range(len(e) - 1):
                mm = ok & (k == b)
                rr = G.r.to_numpy()[mm]
                rs = (G.r.to_numpy() * sg)[mm & st]
                pr.append(dict(xmed=float(np.median(x[mm])), r_med=float(np.median(rr)),
                               r_med_e=float(1.2533 * 1.4826 * np.median(np.abs(rr - np.median(rr))) / np.sqrt(len(rr))),
                               rs_steep=float(np.median(rs)) if len(rs) > 20 else np.nan,
                               rs_steep_e=float(1.2533 * 1.4826 * np.median(np.abs(rs - np.median(rs))) / np.sqrt(len(rs))) if len(rs) > 20 else np.nan,
                               n=int(mm.sum())))
            prof[(arm, p, f)] = pr
    T = pd.DataFrame(rows)
    T.to_csv(D / 'corr_table.csv', index=False)
    json.dump({f'{a}_{p}_{f}': v for (a, p, f), v in prof.items()}, open(D / 'profiles.json', 'w'), indent=1)

    # significance summary
    T['sig_r'] = T.r_spear / T.r_spear_e
    T['sig_rsgn'] = T.rsgn_spear / T.rsgn_spear_e
    pd.set_option('display.width', 260, 'display.max_rows', 600)
    show = ['arm', 'view', 'jbin', 'pred', 'r_n', 'r_pear', 'r_spear', 'r_spear_e', 'r_partial', 'rsgn_pear',
            'rsgn_spear', 'rsgn_partial', 'rj_spear']
    print(T[show].round(4).to_string())

    # figure: profiles of the predictors, one panel per predictor, 4 lines (A x, A y, C x, C y)
    fig, axs = plt.subplots(4, 4, figsize=(13, 10.5))
    for ax, f in zip(axs.flat, PRED):
        for (arm, p), col in zip([('A', 'x'), ('A', 'y'), ('C', 'x'), ('C', 'y')], ['C0', 'C1', 'C2', 'C3']):
            pr = pd.DataFrame(prof[(arm, p, f)])
            ax.errorbar(pr.xmed, pr.r_med * 1e3, pr.r_med_e * 1e3, color=col, marker='o', ms=3, label=f'{arm} {p}: r')
            ax.errorbar(pr.xmed, pr.rs_steep * 1e3, pr.rs_steep_e * 1e3, color=col, ls='--', marker='s', ms=3,
                        mfc='none', label=f'{arm} {p}: r sgn(j), |j|>0.25')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel(DESC[f], fontsize=9)
        ax.set_ylabel('median residual x1000')
    axs.flat[0].legend(fontsize=6.5, ncol=2)
    for ax in axs.flat[len(PRED):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(D / 'profiles_cosmic.png', dpi=120)

    # task 3: same-track x-y correlation of the scale-form residual
    out3 = []
    for arm in 'AC':
        X, Y = L[(arm, 'x')], L[(arm, 'y')]
        both = ((X.j.abs() > 0.1) & (Y.j.abs() > 0.1) & (X.j.abs() < 0.45) & (Y.j.abs() < 0.45)
                & (X.ro.abs() < 0.1) & (X.roq.abs() < 0.1) & (X.r.abs() < 0.15) & (Y.r.abs() < 0.15))
        a, b = (X.r / X.j)[both].to_numpy(), (Y.r / Y.j)[both].to_numpy()
        ra, rb = X.r[both].to_numpy(), Y.r[both].to_numpy()
        # control: the same quantity across chambers (shares only j, truth error)
        oa = (X.ro / X.j)[both].to_numpy()
        ca, cb = np.clip(a, -1, 1), np.clip(b, -1, 1)
        cov = np.cov(ca, cb)
        # robust: Spearman, and the sign-split covariance of r with sign
        sgn = np.sign(X.j[both] * Y.j[both]).to_numpy()
        out3.append(dict(arm=arm, n=int(both.sum()), **{f'scale_{k}': v for k, v in rho(ca, cb).items()},
                         **{f'offset_{k}': v for k, v in rho(ra, rb).items()},
                         **{f'offset_samesign_{k}': v for k, v in rho(ra[sgn > 0], rb[sgn > 0]).items()},
                         **{f'offset_oppsign_{k}': v for k, v in rho(ra[sgn < 0], rb[sgn < 0]).items()},
                         common_frac_x=float(cov[0, 1] / cov[0, 0]), common_frac_y=float(cov[0, 1] / cov[1, 1]),
                         ctrl_AC_scale_pear=rho(ca, np.clip(oa, -1, 1))['pear']))
    T3 = pd.DataFrame(out3)
    T3.to_csv(D / 'xy_common.csv', index=False)
    print(T3.round(4).T.to_string())

    # beam: the same predictors against the pointing residual of the x view (clean late tracks, |s| >= 0.1 for rsgn)
    Bl = beam_long()
    rowsb = []
    for arm, F in Bl.items():
        core = F.r.abs() < 0.15
        G = F[core]
        sg = np.sign(G.s.to_numpy())
        for jb, (lo, hi) in JB.items():
            m = ((G.s.abs() >= lo) & (G.s.abs() < hi)).to_numpy()
            for f in PRED:
                x = clipq(G[f].to_numpy(float))[m]
                d = dict(arm=arm, jbin=jb, pred=f)
                d.update({f'r_{k}': v for k, v in rho(x, G.r.to_numpy()[m]).items()})
                d.update({f'rsgn_{k}': v for k, v in rho(x, (G.r.to_numpy() * sg)[m]).items()})
                rowsb.append(d)
    TB = pd.DataFrame(rowsb)
    TB.to_csv(D / 'corr_table_beam.csv', index=False)
    print(TB[['arm', 'jbin', 'pred', 'r_n', 'r_pear', 'r_spear', 'rsgn_pear', 'rsgn_spear']].round(4).to_string())


if __name__ == '__main__':
    main()
