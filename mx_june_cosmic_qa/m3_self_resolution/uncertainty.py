#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical uncertainties on the M3 pointing curve P(z) and on the
quadrature-deconvolved DUT resolution sigma_DUT = sqrt(sigma_resid^2 - P^2).

Two independent bootstraps:
  (A) pointing P(z=232,702) from the per-plane residual sample
      (sat_perplane_0to3.root, reused via analyze.py);
  (B) the DUT core residual sigma_resid at chi2<1 & NClus4, from the det3
      trade-off cache (m3_cut_tradeoff_data_g_det3_wknd.npz), rstd estimator.
Then sigma_DUT is formed replica-by-replica (independent samples) so the
error bar already carries the quadrature-subtraction non-linearity.

Prints a quotable table and writes uncertainty.json.
"""
import json, numpy as np
import analyze as A            # reuse load(), residuals(), hat_coeffs(), gauss_core_sigma(), Z

B = 2000                        # bootstrap replicas
DUT_Z = {'232': 232.0, '702': 702.0}
TRADEOFF = '../det3_recofar_analysis/m3_cut_tradeoff_data_g_det3_wknd.npz'
CHI2, NCLUS = 1.0, 4            # the operating recipe
rng = np.random.default_rng(12345)


def core_sigma(v):
    return A.gauss_core_sigma(v)[1]


def pointing_from_rows(biased, unbias, rows, z):
    """sigma_k (geo-mean core) on a bootstrap row-sample -> P(z)."""
    g = A.hat_coeffs(A.Z, z)
    var = 0.0
    for k in range(4):
        s_i = core_sigma(biased[rows, k])
        s_e = core_sigma(unbias[rows, k])
        var += g[k] ** 2 * (s_i * s_e)          # sigma_k^2 = s_incl*s_excl
    return np.sqrt(var)


def boot_pointing():
    """Return dict coord-> {z-> (P_hat, P_err, samples)} and the X/Y-averaged P."""
    d = A.load()
    per = {}
    for coord in ['X', 'Y']:
        res = A.residuals(coord, d)                # the heavy per-track loop, once
        biased, unbias, n = res['biased'], res['unbias'], res['n']
        # point estimate on the full sample
        full = {name: pointing_from_rows(biased, unbias, np.arange(n), z)
                for name, z in DUT_Z.items()}
        # bootstrap
        samp = {name: np.empty(B) for name in DUT_Z}
        for b in range(B):
            rows = rng.integers(0, n, n)
            for name, z in DUT_Z.items():
                samp[name][b] = pointing_from_rows(biased, unbias, rows, z)
        per[coord] = {'n': n, 'full': full, 'samp': samp}
        for name in DUT_Z:
            print(f'  P_{coord}(z={name}) = {full[name]*1000:6.1f} '
                  f'+- {samp[name].std()*1000:4.1f} um   (n={n:,})')
    # X/Y average (matches analyze.py's deconvolution convention), paired replicas
    avg = {}
    for name in DUT_Z:
        s = 0.5 * (per['X']['samp'][name] + per['Y']['samp'][name])
        f = 0.5 * (per['X']['full'][name] + per['Y']['full'][name])
        avg[name] = {'hat': f, 'err': s.std(), 'samp': s}
        print(f'  P_avg(z={name}) = {f*1000:6.1f} +- {s.std()*1000:4.1f} um')
    return per, avg


def boot_resid():
    """DUT core residual sigma at chi2<1 & NClus4, rstd, bootstrapped."""
    z = np.load(TRADEOFF)
    cat, r = z['cat'], z['r']
    cmax = np.maximum(z['chi2x'], z['chi2y'])
    ncl = np.minimum(z['nclx'], z['ncly'])
    CAT = json.loads(str(z['cat_codes']))
    is_reco = (cat == CAT['reco_near']) | (cat == CAT['reco_far'])
    keep = (cmax <= CHI2) & (ncl >= NCLUS) & is_reco & (r < 15) & np.isfinite(r)
    rr = r[keep]
    m = rr.size
    hat = A_rstd(rr)
    samp = np.empty(B)
    for b in range(B):
        samp[b] = A_rstd(rr[rng.integers(0, m, m)])
    print(f'  sigma_resid(|r|, chi2<{CHI2} & NClus>={NCLUS}) = '
          f'{hat*1000:6.1f} +- {samp.std()*1000:4.1f} um   (n={m:,})')
    return {'hat': hat, 'err': samp.std(), 'samp': samp, 'n': int(m)}


def A_rstd(v, ns=3, it=5):
    """Iterated 3-sigma-clipped std -- identical to m3_cut_tradeoff.rstd."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size < 3:
        return float('nan')
    for _ in range(it):
        mm, ss = np.median(v), np.std(v); k = np.abs(v - mm) <= ns * ss
        if k.all() or k.sum() < 10:
            break
        v = v[k]
    return float(np.std(v))


def main():
    print('=== (A) pointing bootstrap (sat_perplane_0to3) ===')
    per, avg = boot_pointing()
    print('\n=== (B) DUT residual bootstrap (g_det3_wknd trade-off cache) ===')
    resid = boot_resid()

    print('\n=== (C) deconvolved sigma_DUT = sqrt(sigma_resid^2 - P^2) ===')
    out = {'B': B, 'chi2_cut': CHI2, 'nclus': NCLUS,
           'sigma_resid_um': {'hat': resid['hat']*1000, 'err': resid['err']*1000,
                              'n': resid['n']},
           'pointing_um': {}, 'sigma_DUT_um': {}}
    for name in DUT_Z:
        P = avg[name]
        out['pointing_um'][name] = {'hat': P['hat']*1000, 'err': P['err']*1000}
        # replica-by-replica quadrature subtraction (independent samples)
        d2 = resid['samp']**2 - P['samp']**2
        sdut = np.sqrt(np.clip(d2, 0, None))
        hat = float(np.sqrt(max(resid['hat']**2 - P['hat']**2, 0.0)))
        err = float(sdut.std())
        frac_neg = float(np.mean(d2 <= 0))
        out['sigma_DUT_um'][name] = {'hat': hat*1000, 'err': err*1000,
                                     'frac_unphysical': frac_neg,
                                     'ref_var_frac': float((P['hat']/resid['hat'])**2)}
        print(f'  z={name}:  P={P["hat"]*1000:5.0f}+-{P["err"]*1000:3.0f}  '
              f'sigma_resid={resid["hat"]*1000:5.0f}+-{resid["err"]*1000:3.0f}  '
              f'-> sigma_DUT = {hat*1000:5.0f} +- {err*1000:3.0f} um '
              f'(ref {(P["hat"]/resid["hat"])**2*100:.0f}% of var)')

    with open('uncertainty.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nwrote uncertainty.json')


if __name__ == '__main__':
    main()
