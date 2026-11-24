#!/usr/bin/env python3
"""q12_cells.py -- Q1 (scale equality between views, curl of the tilt field) and Q2 (a-s degeneracy per cell) on the
free 50 mm cell maps of angle_map.py, with bootstrap errors per cell.  -> q12.json, cells_free50.npz, q12_*.png"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from t6lib import rows_ac, OUT, AM

NB = 300
E = np.arange(-200.0, 201.0, 50.0)
C = 0.5 * (E[1:] + E[:-1])
NC = len(C)
rng = np.random.default_rng(11)


def cellfit(j, r):
    """robust_line + its covariance (a, s) on the kept points."""
    keep = np.ones(len(j), bool)
    for _ in range(3):
        s, a = np.polyfit(j[keep], r[keep], 1)
        res = r - (a + s * j)
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < 3 * sig
    X = np.column_stack([np.ones(keep.sum()), j[keep]])
    cov = np.mean((r[keep] - X @ [a, s]) ** 2) * np.linalg.inv(X.T @ X)
    return a, s, cov


def maps(D):
    out = {}
    for w in (0, 1):
        Dw = D[(D.view == w) & D.clean]
        g = Dw.g.iloc[0]
        ix = np.clip(np.digitize(Dw.x, E) - 1, 0, NC - 1)
        iy = np.clip(np.digitize(Dw.y, E) - 1, 0, NC - 1)
        A, S, EA, ES, RHO, N, JM, JR, AB, SB = [np.full((NC, NC), np.nan) for _ in range(10)]
        Ab, Sb = np.full((NB, NC, NC), np.nan), np.full((NB, NC, NC), np.nan)
        jj, rr = Dw.j.to_numpy(), Dw.r.to_numpy()
        for a in range(NC):
            for b in range(NC):
                m = (ix == a) & (iy == b)
                N[a, b] = m.sum()
                if m.sum() < 5:
                    continue
                JM[a, b], JR[a, b] = jj[m].mean() * g, jj[m].std()
                if m.sum() < AM.N_MIN:
                    continue
                aa, ss, cov = cellfit(jj[m], rr[m])
                A[a, b], S[a, b] = g * aa, ss
                EA[a, b], ES[a, b] = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
                RHO[a, b] = g * cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])   # local frame
                idx = np.flatnonzero(m)
                for k in range(NB):
                    bi = rng.choice(idx, len(idx))
                    a2, s2, _ = cellfit(jj[bi], rr[bi])
                    Ab[k, a, b], Sb[k, a, b] = g * a2, s2
                AB[a, b], SB[a, b] = np.std(Ab[:, a, b]), np.std(Sb[:, a, b])
        out[w] = dict(a=A, s=S, ea_fit=EA, es_fit=ES, rho_as=RHO, n=N, jmean_loc=JM, jrms=JR, ea=AB, es=SB,
                      Ab=Ab, Sb=Sb)
    return out


def curl(ax, ay, h=50.0):
    """at the (NC-1)^2 inner corners: d ax/dy - d ay/dx (central over the 2x2 block)."""
    dax_dy = 0.5 * ((ax[:-1, 1:] - ax[:-1, :-1]) + (ax[1:, 1:] - ax[1:, :-1])) / h
    day_dx = 0.5 * ((ay[1:, :-1] - ay[:-1, :-1]) + (ay[1:, 1:] - ay[:-1, 1:])) / h
    return dax_dy, day_dx


def wls_slope(x, y, ex, ey):
    """y = alpha x with errors on both (effective variance), and chi2 of y - x."""
    al = 1.0
    for _ in range(20):
        w = 1 / (ey ** 2 + al ** 2 * ex ** 2)
        al = np.sum(w * x * y) / np.sum(w * x * x)
    w = 1 / (ey ** 2 + al ** 2 * ex ** 2)
    eal = 1 / np.sqrt(np.sum(w * x * x))
    return al, eal


def main():
    res, npz = {}, {}
    for arm in 'AC':
        D = rows_ac(arm)
        mp = maps(D)
        X, Y = mp[0], mp[1]
        R = {}
        # ---- Q1a: scale map the same in both views?
        ok = np.isfinite(X['s']) & np.isfinite(Y['s'])
        sx, sy, ex, ey = X['s'][ok], Y['s'][ok], X['es'][ok], Y['es'][ok]
        d = sy - sx
        ed = np.sqrt(ex ** 2 + ey ** 2)
        mu = np.sum(d / ed ** 2) / np.sum(1 / ed ** 2)
        al, eal = wls_slope(sx - sx.mean(), sy - sy.mean(), ex, ey)
        R['scale'] = dict(n_cells=int(ok.sum()), corr=float(np.corrcoef(sx, sy)[0, 1]),
                          rms_sx=float(sx.std()), rms_sy=float(sy.std()), mean_sx=float(sx.mean()),
                          mean_sy=float(sy.mean()), median_err_sx=float(np.median(ex)),
                          median_err_sy=float(np.median(ey)),
                          slope_sy_vs_sx=float(al), slope_err=float(eal),
                          chi2_equal=float(np.sum(d ** 2 / ed ** 2)), chi2_equal_plus_offset=float(
                              np.sum((d - mu) ** 2 / ed ** 2)), offset_y_minus_x=float(mu),
                          chi2_sx_flat=float(np.sum((sx - np.average(sx, weights=1 / ex ** 2)) ** 2 / ex ** 2)),
                          chi2_sy_flat=float(np.sum((sy - np.average(sy, weights=1 / ey ** 2)) ** 2 / ey ** 2)))
        # null: shuffle the cells of one view
        cs = [np.corrcoef(sx, rng.permutation(sy))[0, 1] for _ in range(2000)]
        R['scale']['corr_shuffled_p95'] = float(np.percentile(np.abs(cs), 95))
        # ---- Q1b: curl in the local frame
        dxy, dyx = curl(X['a'], Y['a'])
        cb = np.array([np.subtract(*curl(X['Ab'][k], Y['Ab'][k])) for k in range(NB)])
        pb = np.array([np.add(*curl(X['Ab'][k], Y['Ab'][k])) for k in range(NB)])
        bxy = np.array([curl(X['Ab'][k], Y['Ab'][k])[0] for k in range(NB)])
        byx = np.array([curl(X['Ab'][k], Y['Ab'][k])[1] for k in range(NB)])
        cu, pl = dxy - dyx, dxy + dyx
        okc = np.isfinite(cu) & np.all(np.isfinite(cb), axis=0)
        ecu, epl = np.std(cb, axis=0)[okc], np.std(pb, axis=0)[okc]
        exy, eyx = np.std(bxy, axis=0)[okc], np.std(byx, axis=0)[okc]
        # full-covariance chi2 (bootstrap covariance, NB=300 > n corners)
        def chi2full(v, B):
            Cv = np.cov(B[:, okc].T)
            return float(v @ np.linalg.solve(Cv, v))
        # shuffled map: permute the cells of a_y (within cells with data), compute the curl rms
        sh = []
        cells = np.argwhere(np.isfinite(Y['a']))
        for _ in range(500):
            ay = Y['a'].copy()
            p = rng.permutation(len(cells))
            ay[tuple(cells.T)] = Y['a'][tuple(cells[p].T)]
            c2 = np.subtract(*curl(X['a'], ay))
            sh.append(np.sqrt(np.nanmean(c2 ** 2)))
        R['curl'] = dict(n_corners=int(okc.sum()), unit='tan per mm',
                         rms_daxdy=float(np.sqrt(np.mean(dxy[okc] ** 2))), rms_daydx=float(np.sqrt(np.mean(dyx[okc] ** 2))),
                         chi2_daxdy_zero=float(np.sum((dxy[okc] / exy) ** 2)),
                         chi2_daydx_zero=float(np.sum((dyx[okc] / eyx) ** 2)),
                         corr_daxdy_daydx=float(np.corrcoef(dxy[okc], dyx[okc])[0, 1]),
                         rms_curl=float(np.sqrt(np.mean(cu[okc] ** 2))), rms_curl_noise=float(np.sqrt(np.mean(ecu ** 2))),
                         chi2_curl_diag=float(np.sum((cu[okc] / ecu) ** 2)), chi2_curl_full=chi2full(cu[okc], cb),
                         rms_plus=float(np.sqrt(np.mean(pl[okc] ** 2))), chi2_plus_diag=float(np.sum((pl[okc] / epl) ** 2)),
                         chi2_plus_full=chi2full(pl[okc], pb),
                         rms_curl_shuffled_median=float(np.median(sh)), rms_curl_shuffled_p05=float(np.percentile(sh, 5)))
        # ---- Q2: degeneracy
        R['degeneracy'] = {}
        for w, M in ((0, X), (1, Y)):
            okw = np.isfinite(M['a'])
            R['degeneracy'][f'view{w}'] = dict(
                median_rho_as=float(np.median(M['rho_as'][okw])), max_abs_rho_as=float(np.max(np.abs(M['rho_as'][okw]))),
                median_jrms=float(np.median(M['jrms'][okw])), range_jmean=[float(np.min(M['jmean_loc'][okw])),
                                                                          float(np.max(M['jmean_loc'][okw]))],
                median_ea=float(np.median(M['ea'][okw])), median_es=float(np.median(M['es'][okw])),
                boot_over_fit_ea=float(np.median(M['ea'][okw] / M['ea_fit'][okw])),
                boot_over_fit_es=float(np.median(M['es'][okw] / M['es_fit'][okw])),
                corr_jmean_s_over_cells=float(np.corrcoef(M['jmean_loc'][okw], M['s'][okw])[0, 1]),
                corr_jmean_y=float(np.corrcoef(M['jmean_loc'][okw], np.broadcast_to(C[None, :], (NC, NC))[okw])[0, 1]))
        res[arm] = R
        for w, M in ((0, 'x'), (1, 'y')):
            for k in ('a', 's', 'ea', 'es', 'rho_as', 'n', 'jmean_loc', 'jrms'):
                npz[f'{arm}_{M}_{k}'] = mp[w][k]
        npz[f'{arm}_daxdy'], npz[f'{arm}_daydx'] = dxy, dyx
        # ---- figures
        fig, axs = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        for w, vn in ((0, 'x'), (1, 'y')):
            M = mp[w]
            for c, (key, lab, vm) in enumerate((('a', 'tilt a (local tan)', 0.03), ('s', 'scale s', 0.15),
                                                ('jmean_loc', 'mean j (local)', 0.4), ('rho_as', 'corr(a, s)', 1))):
                ax = axs[w, c]
                im = ax.imshow(M[key].T, origin='lower', extent=(-200, 200, -200, 200), cmap='RdBu_r', vmin=-vm, vmax=vm)
                ax.set_title(f'{arm} {vn} view: {lab}', fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.8)
                if w == 1:
                    ax.set_xlabel('x_local (mm)')
                if c == 0:
                    ax.set_ylabel('y_local (mm)')
        fig.savefig(OUT / f'q12_cells_{arm}.png', dpi=110)
        plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        axs[0].errorbar(sx, sy, xerr=ex, yerr=ey, fmt='o', ms=3)
        lo, hi = min(sx.min(), sy.min()), max(sx.max(), sy.max())
        axs[0].plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        axs[0].set_xlabel('scale s, x view'); axs[0].set_ylabel('scale s, y view'); axs[0].set_title(f'{arm}: scale per cell')
        axs[1].errorbar(dxy[okc] * 1e3, dyx[okc] * 1e3, xerr=exy * 1e3, yerr=eyx * 1e3, fmt='o', ms=3)
        lo, hi = np.nanmin([dxy[okc], dyx[okc]]) * 1e3, np.nanmax([dxy[okc], dyx[okc]]) * 1e3
        axs[1].plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        axs[1].set_xlabel('d a_x / d y  (1e-3 / mm)'); axs[1].set_ylabel('d a_y / d x  (1e-3 / mm)')
        axs[1].set_title(f'{arm}: curl-free test (local frame)')
        fig.savefig(OUT / f'q1_scale_curl_{arm}.png', dpi=110)
        plt.close(fig)
    json.dump(res, open(OUT / 'q12.json', 'w'), indent=1)
    np.savez(OUT / 'cells_free50.npz', edges=E, **npz)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
