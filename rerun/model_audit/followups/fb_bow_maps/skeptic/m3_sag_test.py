#!/usr/bin/env python3
"""Is the M3 telescope itself bowed?  Two tests on the per-station rays (sat_perplane_0to3.root, det3 weekend run).

Planes 0..3 at z = 1302, 1185, 144, 24 mm.  A plane whose surface sits at z_k + d_k(x) measures a track of slope tan at
x_track(z_k) + tan * d_k(x).
 RELATIVE sag: unbiased residual of plane k (3-plane fit of the others, predicted at z_k) against tan, in bins of the
   track position on plane k -> slope = d_k(x) - [interpolation of the other planes' d at their crossing points].
   A plane bowed differently from the others shows a position-dependent slope.
 COMMON dish d(x) = kM3 x^2 on all four planes: the extra term tan*kM3*(x_c + tan*dz_k)^2 has a tan^3 * dz_k^2 part
   that a straight line cannot absorb -> 4-point residual of plane k = kM3 * tan^3 * (dz_k^2 - mean) with
   (dz_k^2 - mean) = (+0.67, -0.69, -0.70, +0.70) x 1e5 mm^2 about the centroid z = 664.
   Regress the biased residual on [1, tan, tan^2, tan^3, x_c, x_c*tan] per plane and read the tan^3 coefficient.
   The chamber-side dish is kuu/2 ~ 0.75 mm per (100 mm)^2 = 7.5e-5 /mm -> coefficient 7.5e-5 * 0.7e5 = 5.2 mm per unit tan^3.
"""
import numpy as np
import uproot
import awkward as ak

RAYS = '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/m3_self_resolution/sat_perplane_0to3.root'
Z = np.array([1302.0, 1185.0, 144.0, 24.0])


def load():
    t = uproot.open(RAYS)['T']
    br = ['Chi2X', 'Chi2Y', 'NClusX', 'NClusY'] + [f'm{c}{k}' for c in 'XY' for k in range(4)] + \
         [f'z{c}{k}' for c in 'XY' for k in range(4)]
    a = t.arrays(br, library='ak')
    a = a[ak.num(a['Chi2X'], axis=1) == 1]
    d = {k: ak.to_numpy(ak.flatten(a[k])) for k in br}
    return d


def linefit(zz, yy):
    """Weighted-equal straight line y = a + b z through the columns of yy (n x m)."""
    zm = zz.mean(); z0 = zz - zm
    b = (yy * z0).sum(1) / (z0 ** 2).sum()
    a = yy.mean(1)
    return a, b, zm


def main():
    d = load()
    for c in 'XY':
        m = np.isfinite(d[f'm{c}0']) & np.isfinite(d[f'm{c}1']) & np.isfinite(d[f'm{c}2']) & np.isfinite(d[f'm{c}3'])
        m &= (d[f'NClus{c}'] >= 4) & (d[f'Chi2{c}'] < 5)
        Y = np.c_[[d[f'm{c}{k}'][m] for k in range(4)]].T          # n x 4
        zz = np.array([np.nanmedian(d[f'z{c}{k}'][m]) for k in range(4)])
        n = len(Y)
        a, b, zm = linefit(zz, Y)
        tan = b
        xc = a                                                     # position at the z centroid (664 mm)
        res = Y - (a[:, None] + b[:, None] * (zz - zm)[None, :])   # biased 4-point residuals
        print(f'\n=== {c}: n={n}, planes z={zz}, tan rms {tan.std():.3f}, |tan|>0.3: {(np.abs(tan) > 0.3).sum()}')
        dz2 = (zz - zm) ** 2; pat = dz2 - dz2.mean()
        print(f'   dz^2 - mean = {pat / 1e5} x1e5 mm^2')
        # COMMON dish: regress per-plane biased residual on [1, tan, tan^2, tan^3, xc, xc*tan]
        X = np.c_[np.ones(n), tan, tan ** 2, tan ** 3, (xc - np.nanmedian(xc)) / 100, (xc - np.nanmedian(xc)) / 100 * tan]
        for k in range(4):
            r = res[:, k]
            g = np.abs(r) < 2.0
            beta, *_ = np.linalg.lstsq(X[g], r[g], rcond=None)
            rr = r[g] - X[g] @ beta
            cov = np.linalg.inv(X[g].T @ X[g]) * (rr ** 2).mean()
            e = np.sqrt(np.diag(cov))
            kM3 = beta[3] / pat[k]                                   # /mm
            print(f'   plane {k} z={zz[k]:.0f}: tan^3 coef {beta[3]:+.3f}±{e[3]:.3f} mm -> kM3 {kM3 * 1e4:+.2f}±{e[3] / abs(pat[k]) * 1e4:.2f} e-4/mm '
                  f'(chamber dish kuu/2 ~ 0.75 e-4/mm)   [tan {beta[1]:+.3f}±{e[1]:.3f}, tan^2 {beta[2]:+.3f}±{e[2]:.3f}]')
        # combined estimate: weight by pattern
        num = 0; den = 0
        for k in range(4):
            r = res[:, k]; g = np.abs(r) < 2.0
            beta, *_ = np.linalg.lstsq(X[g], r[g], rcond=None)
            rr = r[g] - X[g] @ beta
            cov = np.linalg.inv(X[g].T @ X[g]) * (rr ** 2).mean(); e3 = np.sqrt(cov[3, 3])
            wgt = (pat[k] / e3) ** 2
            num += wgt * beta[3] / pat[k]; den += wgt
        print(f'   combined kM3 = {num / den * 1e4:+.3f} ± {1 / np.sqrt(den) * 1e4:.3f} e-4/mm  -> common M3 dish at 150 mm: {num / den * 150 ** 2:+.2f} ± {150 ** 2 / np.sqrt(den):.2f} mm')
        # RELATIVE sag: unbiased residual of plane k vs tan, binned in x on plane k (8 bins)
        print('   relative sag d_k(x) [mm] = slope of unbiased residual vs tan, bins of position on the plane:')
        for k in range(4):
            others = [j for j in range(4) if j != k]
            a3, b3, zm3 = linefit(zz[others], Y[:, others])
            pred = a3 + b3 * (zz[k] - zm3)
            ur = Y[:, k] - pred
            xk = Y[:, k]
            edges = np.nanpercentile(xk, np.linspace(0, 100, 9))
            row = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (xk >= lo) & (xk < hi) & (np.abs(ur) < 3)
                if s.sum() < 200:
                    row.append('   nan   '); continue
                A = np.c_[np.ones(s.sum()), b3[s]]
                bb, *_ = np.linalg.lstsq(A, ur[s], rcond=None)
                rr = ur[s] - A @ bb; ee = np.sqrt(np.linalg.inv(A.T @ A)[1, 1] * (rr ** 2).mean())
                row.append(f'{bb[1]:+5.2f}±{ee:.2f}')
            print(f'     plane {k} (x bins {edges[0]:.0f}..{edges[-1]:.0f} mm): ' + ' '.join(row))


if __name__ == '__main__':
    main()
