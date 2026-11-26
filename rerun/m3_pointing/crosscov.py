#!/usr/bin/env python3
"""crosscov.py -- the M3 pointing floor without any chamber: the covariance of two chambers' residuals.

Two chambers on the same track share the M3 error, and nothing else: cov(r_top, r_bottom) = cov(e_M3(z_top),
e_M3(z_bottom)).  As Paper I does for the variance (Fig. posres a), we take tracks within 5 deg of normal incidence in
the view, bin them in the chi2 of the matched M3 coordinate, fit cov linear in chi2 and extrapolate to chi2 -> 0.  The
intercept is compared with the Gaussian prediction from the per-plane sigma (m3_self_resolution):
sum_k g_k(z_top) g_k(z_bottom) sigma_k^2.  Robust covariance per bin: (s68(a+b)^2 - s68(a-b)^2) / 4, 68.27 % of a
Gaussian; error by bootstrap.  Writes $WORK/crosscov.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m3common import load, WORK, s68, Z_LAYERS, SIG_PLANE, hat, pointing_gauss  # noqa: E402
from fit import pair_table, MATCH  # noqa: E402

TAN5 = np.tan(np.deg2rad(5.0))
NB = 6
rng = np.random.default_rng(1)


def rcov(a, b):
    return (s68(a + b) ** 2 - s68(a - b) ** 2) / 4


def main():
    T = pair_table(load())
    out = {}
    for v, c in MATCH.items():
        S = T[(T[f'tan_{v}_t'].abs() < TAN5)]
        x = S[f'chi2_{c}_t'].to_numpy(); a = S[f'res_{v}_t'].to_numpy(); b = S[f'res_{v}_b'].to_numpy()
        q = np.quantile(x, np.linspace(0, 1, NB + 1))
        xs, cs, es = [], [], []
        for lo, hi in zip(q[:-1], q[1:]):
            m = (x >= lo) & (x <= hi)
            xs.append(x[m].mean()); cs.append(rcov(a[m], b[m]))
            bs = [rcov(a[m][i], b[m][i]) for i in (rng.integers(0, m.sum(), m.sum()) for _ in range(200))]
            es.append(np.std(bs))
        xs, cs, es = map(np.array, (xs, cs, es))
        w = 1 / es ** 2
        A = np.vstack([np.ones_like(xs), xs]).T
        C = np.linalg.inv(A.T @ (A * w[:, None])); p = C @ A.T @ (w * cs)
        chi2 = float(np.sum(w * (cs - A @ p) ** 2))
        zt, zb = float(S[f'z_{v}_t'].median()), float(S[f'z_{v}_b'].median())
        pred = float(np.sum(hat(Z_LAYERS, zt) * hat(Z_LAYERS, zb) * SIG_PLANE[c] ** 2))
        out[v] = dict(m3=c, n=int(len(S)), chi2_bins=xs.tolist(), cov=cs.tolist(), err=es.tolist(),
                      intercept=float(p[0]), intercept_err=float(np.sqrt(C[0, 0])), slope=float(p[1]),
                      slope_err=float(np.sqrt(C[1, 1])), fit_chi2=chi2, ndf=NB - 2, gauss_pred=pred,
                      ratio=float(p[0] / pred), ratio_err=float(np.sqrt(C[0, 0]) / pred),
                      P_top=pointing_gauss(zt, c), P_bottom=pointing_gauss(zb, c))
        print(f"view {v} (M3 {c}) n {len(S)}: cov(chi2->0) = {p[0]:.4f} ± {np.sqrt(C[0, 0]):.4f} mm^2, slope "
              f"{p[1]:.3f} ± {np.sqrt(C[1, 1]):.3f}, chi2 {chi2:.1f}/{NB - 2};  Gaussian prediction {pred:.4f} -> "
              f"ratio {p[0] / pred:.2f} ± {np.sqrt(C[0, 0]) / pred:.2f}")
    json.dump(out, open(os.path.join(WORK, 'crosscov.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
