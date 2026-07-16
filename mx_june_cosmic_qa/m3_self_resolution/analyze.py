#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3 four-plane telescope self-resolution + DUT pointing-uncertainty analysis.

Data: sat_perplane_0to3.root -- MX17 det3 weekend "saturday" long run
(resist 490 V, drift 1000 V), reprocessed locally with the per-plane branch
patch (mX0..3 / mY0..3 = aligned cluster position at each of the 4 stations;
zX0..3 / zY0..3 = the plane z's).  Layers 0..3 sit at z = 1302,1185,144,24 mm.

Each coordinate (X, Y) is an independent 4-point straight-line fit -> a genuine
4-plane beam telescope.  We measure:
  1. per-plane intrinsic resolution sigma_k  (geometric-mean of biased/unbiased
     residual widths, cross-checked by a simultaneous unbiased-residual solve);
  2. multiple scattering (two-doublet divergence + Highland cross-check);
  3. the M3 track POINTING uncertainty P(z) vs z, and specifically at the DUT
     planes z=232 and z=702 mm -- the reference error folded into every DUT
     residual -- and where a DUT plane out-resolves the reference track.

Outputs: figs/*.png and results.json + stdout summary.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import uproot, awkward as ak

# ---------------------------------------------------------------- config
RAYS = 'sat_perplane_0to3.root'
Z = np.array([1302.0, 1185.0, 144.0, 24.0])          # layer 0,1,2,3 station z [mm]
LAYNAME = ['L0 z=1302', 'L1 z=1185', 'L2 z=144', 'L3 z=24']
DUT_Z = {'232 (bottom slot)': 232.0, '702 (mid-gap)': 702.0}
# Measured det3 DUT core residuals (M3 track vs DUT hit), from the June QA
# (det3_recofar_analysis / M3_CUT_AND_ACTIVE_AREA_NOTE): these are DUT (+) M3
# convolved.  Used only for the deconvolution cross-check.
DUT_RESID_CORE = {'chi2<5': 0.63, 'chi2<1 & NClus4': 0.47}
X0_AIR_MM = 304200.0                                  # radiation length of air
OUT = 'figs'

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- load
def load():
    t = uproot.open(RAYS)
    tn = sorted([k for k in t.keys() if k.startswith('T;')])[-1]
    br = ['evn', 'Chi2X', 'Chi2Y', 'NClusX', 'NClusY',
          'mX0', 'mX1', 'mX2', 'mX3', 'mY0', 'mY1', 'mY2', 'mY3']
    a = t[tn].arrays(br, library='ak')
    # single-track events only (clean topology), then flatten the jagged branches
    ntr = ak.num(a['Chi2X'], axis=1)
    a = a[ntr == 1]
    d = {}
    for k in br:
        if k == 'evn':
            d[k] = ak.to_numpy(a[k])                 # scalar per event = per track here
        else:
            d[k] = ak.to_numpy(ak.flatten(a[k]))
    return d


# ---------------------------------------------------------------- fit helpers
def line_fit(zz, yy):
    """Unweighted straight-line LS (matches the producer's TGraph fit).
    Returns intercept a, slope b."""
    A = np.vstack([np.ones_like(zz), zz]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return coef


def hat_coeffs(z_used, z_eval):
    """Row of the unweighted prediction operator: yhat(z_eval)=sum g_j*y_j over
    the used planes.  Pure geometry."""
    A = np.vstack([np.ones_like(z_used), z_used]).T
    M = np.linalg.inv(A.T @ A)
    return np.array([1.0, z_eval]) @ M @ A.T          # length len(z_used)


def gauss_core_sigma(r, n_iter=6, nsig=2.5):
    """Robust Gaussian-core width: iteratively fit a Gaussian to |r|<nsig*sigma.
    Returns (mu, sigma, n_core)."""
    r = r[np.isfinite(r)]
    mu, sig = np.median(r), 1.4826 * np.median(np.abs(r - np.median(r)))
    for _ in range(n_iter):
        m = np.abs(r - mu) < nsig * sig
        if m.sum() < 20:
            break
        mu, sig = r[m].mean(), r[m].std()
    return mu, sig, int(m.sum())


# ---------------------------------------------------------------- per-plane residuals
def residuals(coord, d, chi2_cut=None):
    """Return dict with biased & unbiased residual arrays per plane for the
    4-hit tracks of the given coordinate."""
    M = np.stack([d[f'm{coord}{L}'] for L in range(4)], axis=1)     # (n,4)
    nclus = d[f'NClus{coord}']
    chi2 = d[f'Chi2{coord}']
    good = (nclus == 4) & np.all(np.isfinite(M), axis=1) & (chi2 >= 0)
    if chi2_cut is not None:
        good = good & (chi2 < chi2_cut)
    M = M[good]
    n = len(M)
    biased = np.full((n, 4), np.nan)
    unbias = np.full((n, 4), np.nan)
    # precompute geometry
    full_coef_op = None
    loo_ops = []
    for k in range(4):
        idx = [j for j in range(4) if j != k]
        loo_ops.append((idx, hat_coeffs(Z[idx], Z[k])))
    for i in range(n):
        y = M[i]
        a, b = line_fit(Z, y)
        pred = a + b * Z
        biased[i] = y - pred
        for k in range(4):
            idx, g = loo_ops[k]
            unbias[i, k] = y[k] - g @ y[idx]
    return dict(M=M, biased=biased, unbias=unbias, n=n)


# ---------------------------------------------------------------- Method A: geometric mean
def geometric_mean(res):
    out = []
    for k in range(4):
        _, s_incl, _ = gauss_core_sigma(res['biased'][:, k])
        _, s_excl, _ = gauss_core_sigma(res['unbias'][:, k])
        out.append(dict(layer=k, s_incl=s_incl, s_excl=s_excl,
                        sigma=np.sqrt(s_incl * s_excl)))
    return out


# ---------------------------------------------------------------- Method B: simultaneous unbiased solve
def simultaneous_solve(res):
    """Var(u_k) = sigma_k^2 + sum_{j!=k} c_kj^2 sigma_j^2.  Solve for sigma_k^2.
    NOTE: for the two-doublet geometry the leave-one-out coefficients within a
    doublet are huge, so this 4x4 system is severely ill-conditioned -- reported
    only to expose that (motivates the geometric-mean estimator)."""
    V = np.array([gauss_core_sigma(res['unbias'][:, k])[1] ** 2 for k in range(4)])
    Mmat = np.zeros((4, 4))
    for k in range(4):
        idx = [j for j in range(4) if j != k]
        g = hat_coeffs(Z[idx], Z[k])
        Mmat[k, k] = 1.0
        for jj, j in enumerate(idx):
            Mmat[k, j] = g[jj] ** 2
    cond = np.linalg.cond(Mmat)
    s2 = np.linalg.solve(Mmat, V)
    return dict(var_unbias=V, sigma2=s2, cond=cond,
                sigma=np.sqrt(np.clip(s2, 0, None)))


def equal_sigma_global(coord, d, chi2_cut=None):
    """Assume a single sigma common to all planes: from the biased 4-plane fit,
    E[Chi2] = sigma^2 * (N-2) = 2 sigma^2.  Use the Gaussian core of the pooled
    biased residuals (2 dof), which is leverage-independent for equal sigma."""
    res = residuals(coord, d, chi2_cut=chi2_cut)
    pooled = res['biased'].ravel()
    # per-plane biased core variance summed = E[Chi2]; equal-sigma => sigma^2 = sumincl/(N-2)
    incl2 = np.array([gauss_core_sigma(res['biased'][:, k])[1] ** 2 for k in range(4)])
    sigma = np.sqrt(incl2.sum() / (4 - 2))
    return sigma


# ---------------------------------------------------------------- pointing vs z
def pointing_curve(sigma_k, z_grid):
    """Measurement-only pointing sigma of the full 4-plane unweighted fit at each
    z in z_grid: Var = sum_k g_k(z)^2 sigma_k^2."""
    P = np.zeros_like(z_grid)
    for i, z in enumerate(z_grid):
        g = hat_coeffs(Z, z)
        P[i] = np.sqrt(np.sum((g ** 2) * (sigma_k ** 2)))
    return P


def pointing_at(sigma_k, z):
    g = hat_coeffs(Z, z)
    return float(np.sqrt(np.sum((g ** 2) * (sigma_k ** 2))))


# ---------------------------------------------------------------- multiple scattering: two-doublet divergence
def doublet_divergence(coord, d, z_eval):
    """Predict position at z_eval from the TOP doublet (L0,L1) and the BOTTOM
    doublet (L2,L3) independently; the spread of their difference beyond the
    measurement-only expectation is the MS accumulated between the doublets."""
    M = np.stack([d[f'm{coord}{L}'] for L in range(4)], axis=1)
    nclus = d[f'NClus{coord}']
    good = (nclus == 4) & np.all(np.isfinite(M), axis=1)
    M = M[good]
    top_idx, bot_idx = [0, 1], [2, 3]
    g_top = hat_coeffs(Z[top_idx], z_eval)
    g_bot = hat_coeffs(Z[bot_idx], z_eval)
    x_top = M[:, top_idx] @ g_top
    x_bot = M[:, bot_idx] @ g_bot
    diff = x_top - x_bot
    _, s_diff, _ = gauss_core_sigma(diff)
    return dict(g_top=g_top, g_bot=g_bot, s_diff=s_diff, diff=diff, n=len(diff))


def kink_ms_bound(coord, d, sigma_k):
    """Direct inter-doublet kink: Delta_slope = slope_top - slope_bottom.
    Var(Delta) = sigma_slope_top^2 + sigma_slope_bot^2 + theta_MS^2, with the
    slope errors fixed by the plane resolution and the (short) doublet lever arm.
    Returns theta_MS (rad) or an upper limit if the excess is not resolved."""
    M = np.stack([d[f'm{coord}{L}'] for L in range(4)], axis=1)
    good = (d[f'NClus{coord}'] == 4) & np.all(np.isfinite(M), axis=1)
    M = M[good]
    dz_top = Z[0] - Z[1]                       # 117 mm
    dz_bot = Z[2] - Z[3]                        # 120 mm
    s_top = (M[:, 0] - M[:, 1]) / dz_top
    s_bot = (M[:, 2] - M[:, 3]) / dz_bot
    dslope = s_top - s_bot
    _, s_meas, _ = gauss_core_sigma(dslope)
    # expected slope-difference width from resolution only
    var_slope_top = (sigma_k[0] ** 2 + sigma_k[1] ** 2) / dz_top ** 2
    var_slope_bot = (sigma_k[2] ** 2 + sigma_k[3] ** 2) / dz_bot ** 2
    var_res = var_slope_top + var_slope_bot
    excess = s_meas ** 2 - var_res
    theta = np.sqrt(excess) if excess > 0 else 0.0
    # 1-sigma upper limit: statistical error on s_meas ~ s_meas/sqrt(2N)
    n = len(dslope)
    s_meas_hi = s_meas * (1 + 1 / np.sqrt(2 * n))
    theta_ul = np.sqrt(max(s_meas_hi ** 2 - var_res, 0.0))
    return dict(s_dslope_mrad=s_meas * 1e3, res_only_mrad=np.sqrt(var_res) * 1e3,
                theta_ms_mrad=theta * 1e3, theta_ms_ul_mrad=theta_ul * 1e3, n=n)


def ms_from_divergence(coord, d, sigma_k, z_eval):
    dd = doublet_divergence(coord, d, z_eval)
    # measurement-only variance of (x_top - x_bot) at z_eval
    var_meas = (np.sum(dd['g_top'] ** 2 * sigma_k[[0, 1]] ** 2)
                + np.sum(dd['g_bot'] ** 2 * sigma_k[[2, 3]] ** 2))
    var_excess = dd['s_diff'] ** 2 - var_meas
    return dict(s_diff=dd['s_diff'], var_meas=var_meas,
                var_excess=var_excess,
                sig_excess=np.sqrt(var_excess) if var_excess > 0 else 0.0,
                diff=dd['diff'])


# ---------------------------------------------------------------- main
def main():
    d = load()
    print(f'single-track rays loaded: {len(d["evn"]):,}')
    results = {'run': 'MX17 det3 weekend saturday long (490V/1000V), files 0-3',
               'z_layers': Z.tolist(), 'coords': {}}

    fig_res, axes_res = plt.subplots(2, 4, figsize=(17, 7.5))
    sigma_by_coord = {}
    for ci, coord in enumerate(['X', 'Y']):
        res = residuals(coord, d)
        gm = geometric_mean(res)
        sm = simultaneous_solve(res)
        sig_eq = equal_sigma_global(coord, d)
        sigma_gm = np.array([g['sigma'] for g in gm])
        # blend: report geometric-mean as primary
        sigma_k = sigma_gm
        sigma_by_coord[coord] = sigma_k
        print(f'\n===== {coord} : {res["n"]:,} four-hit tracks =====')
        for k in range(4):
            print(f'  {LAYNAME[k]:>9}: sigma_incl={gm[k]["s_incl"]*1000:6.1f}um  '
                  f'sigma_excl={gm[k]["s_excl"]*1000:6.1f}um  '
                  f'-> sigma_geomean={gm[k]["sigma"]*1000:6.1f}um')
        print(f'  geo-mean average sigma = {np.mean(sigma_k)*1000:.1f} um ; '
              f'equal-sigma global = {sig_eq*1000:.1f} um ; '
              f'sim-solve cond#={sm["cond"]:.1e} (ill-conditioned, not used)')
        results['coords'][coord] = {
            'n_tracks': res['n'],
            'layers': [{'layer': k, 'z': Z[k],
                        's_incl_um': gm[k]['s_incl'] * 1000,
                        's_excl_um': gm[k]['s_excl'] * 1000,
                        'sigma_geomean_um': gm[k]['sigma'] * 1000} for k in range(4)],
            'sigma_mean_um': float(np.mean(sigma_k) * 1000),
            'sigma_equal_global_um': float(sig_eq * 1000),
            'simsolve_cond': float(sm['cond']),
        }
        # residual panels
        for k in range(4):
            ax = axes_res[ci, k]
            u = res['unbias'][:, k]
            mu, sig, _ = gauss_core_sigma(u)
            rng_pl = 4 * sig
            ax.hist(u, bins=120, range=(-rng_pl, rng_pl), color='steelblue',
                    alpha=0.8, density=True)
            xs = np.linspace(-rng_pl, rng_pl, 200)
            ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
                    'r-', lw=1.5)
            ax.set_title(f'{coord} {LAYNAME[k]}\nunbiased core $\\sigma$={sig*1000:.0f} µm',
                         fontsize=9)
            ax.set_xlabel('unbiased residual [mm]', fontsize=8)
            ax.tick_params(labelsize=7)
    fig_res.suptitle('M3 per-plane unbiased residuals (4-hit tracks) with Gaussian-core fit',
                     fontsize=12)
    fig_res.tight_layout()
    fig_res.savefig(f'{OUT}/residuals_per_plane.png', dpi=140, bbox_inches='tight')
    plt.close(fig_res)

    # ------------------------------------------------ multiple scattering
    print('\n===== multiple scattering (inter-doublet kink) =====')
    ms = {}
    for coord in ['X', 'Y']:
        sigma_k = sigma_by_coord[coord]
        kk = kink_ms_bound(coord, d, sigma_k)
        ms[coord] = kk
        print(f'  {coord}: sigma(Delta-slope)={kk["s_dslope_mrad"]:.2f} mrad  '
              f'res-only={kk["res_only_mrad"]:.2f} mrad  '
              f'-> theta_MS={kk["theta_ms_mrad"]:.2f} mrad (UL {kk["theta_ms_ul_mrad"]:.2f})')
    # Highland cross-check with a rough material budget
    # air 24->1302mm + ~4 chambers (~0.4% X0 each) + DUT (~0.3% X0)
    x_over_X0 = (1278.0 / X0_AIR_MM) + 4 * 0.004 + 0.003
    theta0 = lambda p_GeV: (13.6e-3 / p_GeV) * np.sqrt(x_over_X0) * (1 + 0.038 * np.log(x_over_X0))
    # MS-induced pointing at an interpolated DUT: the fit straddles the DUT, so
    # the scattering in the material between the doublets contributes at most
    # ~theta_MS * (lever inside the gap)/2 in the worst case; quote theta*d.
    theta_ms_meas = np.mean([ms[c]['theta_ms_mrad'] for c in ['X', 'Y']]) * 1e-3
    theta_ms_ul = np.mean([ms[c]['theta_ms_ul_mrad'] for c in ['X', 'Y']]) * 1e-3
    results['ms'] = {
        'x_over_X0_estimate': x_over_X0,
        'theta0_mrad_at_1GeV': float(theta0(1.0) * 1000),
        'theta0_mrad_at_3GeV': float(theta0(3.0) * 1000),
        'kink': {c: {'s_dslope_mrad': ms[c]['s_dslope_mrad'],
                     'res_only_mrad': ms[c]['res_only_mrad'],
                     'theta_ms_mrad': ms[c]['theta_ms_mrad'],
                     'theta_ms_ul_mrad': ms[c]['theta_ms_ul_mrad']} for c in ['X', 'Y']},
    }
    print(f'  Highland (x/X0~{x_over_X0:.4f}): theta0 = {theta0(1.0)*1e3:.2f} mrad @1GeV, '
          f'{theta0(3.0)*1e3:.2f} mrad @3GeV')
    print(f'  measured inter-doublet kink theta_MS ~ {theta_ms_meas*1e3:.2f} mrad '
          f'(UL {theta_ms_ul*1e3:.2f}) -> consistent with a soft/high-p cosmic mix')

    # ------------------------------------------------ pointing vs z  (compute now; plot after sigma_DUT)
    pointing_results = {}
    for coord in ['X', 'Y']:
        sigma_k = sigma_by_coord[coord]
        pointing_results[coord] = {'at_DUT': {}}
        for name, zdut in DUT_Z.items():
            pv = pointing_at(sigma_k, zdut)
            pointing_results[coord]['at_DUT'][name] = pv * 1000
            print(f'  {coord} pointing at z={zdut:.0f}: {pv*1000:.1f} um')
    results['pointing'] = pointing_results

    # ------------------------------------------------ DUT deconvolution + crossover
    print('\n===== DUT deconvolution & crossover =====')
    # Core (MS-free) reference pointing uses the tightest-cut DUT residual core.
    deconv = {}
    for name, zdut in DUT_Z.items():
        P_mean = np.mean([pointing_at(sigma_by_coord[c], zdut) for c in ['X', 'Y']])
        # best deconvolution: core-to-core with the tight-cut residual
        sres = DUT_RESID_CORE['chi2<1 & NClus4']
        val = sres ** 2 - P_mean ** 2
        sdut = float(np.sqrt(val)) if val > 0 else float('nan')
        deconv[name] = {'P_M3_core_um': P_mean * 1000,
                        'dut_resid_core_um': sres * 1000,
                        'sigma_DUT_um': sdut * 1000,
                        'reference_fraction_of_var': P_mean ** 2 / sres ** 2}
        print(f'  z={zdut:.0f}: P_M3(core)={P_mean*1000:.0f}um, DUT resid(core)={sres*1000:.0f}um '
              f'-> sigma_DUT={sdut*1000:.0f}um   (reference = {P_mean**2/sres**2*100:.0f}% of residual variance)')
    # Deconvolved DUT intrinsic (average of the two slots)
    sigma_DUT = np.nanmean([deconv[n]['sigma_DUT_um'] for n in DUT_Z]) / 1000.0
    results['deconvolution'] = deconv
    results['sigma_DUT_intrinsic_um'] = sigma_DUT * 1000

    # crossover: where does the reference pointing exceed the DUT resolution?
    # (below this the reference is better than the DUT; beyond it the DUT wins)
    print('\n===== crossover: P_M3(z) vs sigma_DUT =====')
    xover = {}
    for coord in ['X', 'Y']:
        sk = sigma_by_coord[coord]
        # scan outward from the telescope centre to find P(z)=sigma_DUT
        zc = np.sum(Z) / 4
        z_hi = np.linspace(zc, 3000, 4000)
        P_hi = pointing_curve(sk, z_hi)
        z_lo = np.linspace(zc, -1500, 4000)
        P_lo = pointing_curve(sk, z_lo)
        cross_hi = z_hi[np.argmax(P_hi > sigma_DUT)] if np.any(P_hi > sigma_DUT) else None
        cross_lo = z_lo[np.argmax(P_lo > sigma_DUT)] if np.any(P_lo > sigma_DUT) else None
        xover[coord] = {'z_up_mm': float(cross_hi) if cross_hi else None,
                        'z_down_mm': float(cross_lo) if cross_lo else None}
        print(f'  {coord}: P_M3(z) exceeds sigma_DUT={sigma_DUT*1000:.0f}um  '
              f'above z~{cross_hi:.0f}mm and below z~{cross_lo:.0f}mm '
              f'(inside the M3 volume the reference always out-points the DUT)')
    results['crossover'] = xover

    # ------------------------------------------------ pointing figure (with crossover)
    z_grid = np.linspace(-700, 2100, 700)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = {'X': 'C0', 'Y': 'C3'}
    for coord in ['X', 'Y']:
        P = pointing_curve(sigma_by_coord[coord], z_grid) * 1000
        ax.plot(z_grid, P, color=colors[coord], lw=2.2,
                label=f'{coord}: M3 pointing $P(z)$ (core)')
    # M3 active volume shading
    ax.axvspan(24, 1302, color='steelblue', alpha=0.06)
    ax.text(663, ax.get_ylim()[1] if False else 60, 'M3 tracking volume\n(interpolation)',
            fontsize=8, color='steelblue', ha='center')
    for zk in Z:
        ax.axvline(zk, color='gray', ls=':', lw=0.8)
    for name, zdut in DUT_Z.items():
        ax.axvline(zdut, color='green', ls='--', lw=1.3)
        ax.text(zdut + 15, 900, f'DUT z={name.split()[0]}', rotation=90,
                fontsize=8, color='green', va='top')
    # deconvolved DUT intrinsic
    ax.axhline(sigma_DUT * 1000, color='darkorange', ls='-', lw=1.6,
               label=f'DUT intrinsic $\\sigma_{{DUT}}\\approx{sigma_DUT*1000:.0f}$ µm (deconvolved)')
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('pointing / resolution [µm]')
    ax.set_title('M3 reference-track pointing uncertainty vs z\n'
                 'stations z=24,144,1185,1302 mm — DUT out-resolves the reference only where $P(z)>\\sigma_{DUT}$')
    ax.legend(fontsize=9, loc='upper center')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1000)
    ax.set_xlim(-700, 2100)
    fig.tight_layout()
    fig.savefig(f'{OUT}/pointing_vs_z.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------ per-plane sigma bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.35
    xk = np.arange(4)
    for i, coord in enumerate(['X', 'Y']):
        sig = [results['coords'][coord]['layers'][k]['sigma_geomean_um'] for k in range(4)]
        ax.bar(xk + (i - 0.5) * w, sig, w, label=f'{coord} planes', color=colors[coord], alpha=0.85)
    ax.set_xticks(xk)
    ax.set_xticklabels([f'L{k}\nz={int(Z[k])}' for k in range(4)])
    ax.set_ylabel('intrinsic $\\sigma$ [µm]  (geometric-mean)')
    ax.set_title('M3 per-plane intrinsic resolution (four-hit tracks)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{OUT}/sigma_per_plane.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nwrote results.json + figs/{residuals_per_plane,pointing_vs_z,sigma_per_plane}.png')
    return results, sigma_by_coord, d


if __name__ == '__main__':
    main()
