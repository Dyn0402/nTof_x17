import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_sag_test as M
d = M.load()
for c in 'XY':
    for lab, chi, tmax in (('chi2<5', 5, 9), ('chi2<2', 2, 9), ('chi2<5 |tan|<0.3', 5, 0.3), ('chi2<5 |tan|<0.25', 5, 0.25)):
        m = np.isfinite(d[f'm{c}0']) & np.isfinite(d[f'm{c}1']) & np.isfinite(d[f'm{c}2']) & np.isfinite(d[f'm{c}3'])
        m &= (d[f'NClus{c}'] >= 4) & (d[f'Chi2{c}'] < chi)
        Y = np.c_[[d[f'm{c}{k}'][m] for k in range(4)]].T
        zz = np.array([np.nanmedian(d[f'z{c}{k}'][m]) for k in range(4)])
        a, b, zm = M.linefit(zz, Y); tan = b; xc = a
        s = np.abs(tan) < tmax; Y, tan, xc = Y[s], tan[s], xc[s]
        res = Y - (xc[:, None] + tan[:, None] * (zz - zm)[None, :])
        dz2 = (zz - zm) ** 2; pat = dz2 - dz2.mean()
        n = len(Y); X = np.c_[np.ones(n), tan, tan**2, tan**3, (xc - np.nanmedian(xc))/100, (xc - np.nanmedian(xc))/100*tan]
        num = den = 0; per = []
        for k in range(4):
            r = res[:, k]; g = np.abs(r) < 2.0
            beta, *_ = np.linalg.lstsq(X[g], r[g], rcond=None); rr = r[g] - X[g] @ beta
            e3 = np.sqrt(np.linalg.inv(X[g].T @ X[g])[3, 3] * (rr**2).mean())
            wgt = (pat[k]/e3)**2; num += wgt*beta[3]/pat[k]; den += wgt; per.append(beta[3]/pat[k]*1e4)
        print(f'{c} {lab:20s} n={n:6d}  common dish at 150 mm: {num/den*150**2:+.2f} ± {150**2/np.sqrt(den):.2f} mm   per-plane kM3 e-4/mm: ' + ' '.join(f'{x:+.2f}' for x in per))
