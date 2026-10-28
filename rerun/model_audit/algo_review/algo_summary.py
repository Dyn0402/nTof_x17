#!/usr/bin/env python3
"""Summarise algo_review rows: prod vs lm vs kgap, and the 60 ns lattice degeneracy."""
import sys, json, pickle
import numpy as np

rows = pickle.load(open(sys.argv[1], 'rb'))
out = {}


def rsig(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    q = np.percentile(x, [16, 84]); return 0.5 * (q[1] - q[0])


def s68(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.percentile(np.abs(x), 68) if len(x) else np.nan


def angle_table(arm):
    res = {}
    for plane in 'xy':
        R = [r for r in rows if r['plane'] == plane and f'{arm}_w' in r and np.isfinite(r.get(f'{arm}_w', np.nan))]
        tr = np.array([r['tan_ref'] for r in R]); w = np.array([r[f'{arm}_w'] for r in R]) * 1e3   # um/ns
        sel = (np.abs(tr) > 0.10) & (np.abs(tr) < 0.45)
        v_lad = np.median(w[sel] / tr[sel]) if sel.sum() > 10 else np.nan
        # w0 offset (chamber tilt) removed as in production: median residual at small angle
        th_fit = np.degrees(np.arctan(w / v_lad)); th_ref = np.degrees(np.arctan(tr))
        d = th_fit - th_ref
        d = d - np.median(d)
        keep = np.abs(tr) < 0.6
        res[plane] = dict(n=int(keep.sum()), v_ladder=float(v_lad), s68_deg=float(s68(d[keep])),
                          rsig_deg=float(rsig(d[keep])), far5_pct=float(100 * np.mean(np.abs(d[keep]) > 5)),
                          chi2dof_median=float(np.median([r[f'{arm}_chi2'] / max(r[f'{arm}_dof'], 1) for r in R])))
    return res


for arm in ('prod', 'lm', 'kgap'):
    out[arm] = angle_table(arm)

# ---- prod vs lm: same minimum?
both = [r for r in rows if 'lm_chi2' in r and 'prod_chi2' in r]
dchi = np.array([r['lm_chi2'] - r['prod_chi2'] for r in both])
scale = np.array([r['prod_chi2'] / max(r['prod_dof'], 1) for r in both])
dw = np.array([abs(r['lm_w'] - r['prod_w']) * 1e3 for r in both])
dp = np.array([abs(r['lm_p0'] - r['prod_p0']) for r in both])
dt = np.array([abs(r['lm_t0'] - r['prod_t0']) for r in both])
out['prod_vs_lm'] = dict(
    n=len(both),
    lm_lower_pct=float(100 * np.mean(dchi < -1.0)), lm_higher_pct=float(100 * np.mean(dchi > 1.0)),
    same_pct=float(100 * np.mean(np.abs(dchi) <= 1.0)),
    lm_lower_by_1sig_pct=float(100 * np.mean(dchi < -scale)), lm_higher_by_1sig_pct=float(100 * np.mean(dchi > scale)),
    same_w_1umns_pct=float(100 * np.mean(dw < 1.0)), same_p0_50um_pct=float(100 * np.mean(dp < 0.05)),
    same_t0_5ns_pct=float(100 * np.mean(dt < 5.0)),
    same_basin_pct=float(100 * np.mean((dw < 1.0) & (dp < 0.1) & (dt < 30.0))),
    nsolve_prod_median=float(np.median([r['prod_nsolve'] for r in both])),
    nsolve_lm_median=float(np.median([r['lm_nsolve'] for r in both])),
    time_prod_median_s=float(np.median([r['prod_time'] for r in both])),
    time_lm_median_s=float(np.median([r['lm_time'] for r in both])),
    lm_iters_median=float(np.median([r['lm_nit'] for r in both])),
    # resolution on the SAME rows when both agree vs disagree
)
# where they disagree, who is closer to the reference?
dis = [r for r in both if abs(r['lm_w'] - r['prod_w']) * 1e3 >= 1.0]
if dis:
    vl = {p: out['prod'][p]['v_ladder'] for p in 'xy'}
    e_prod = np.array([abs(np.degrees(np.arctan(r['prod_w'] * 1e3 / vl[r['plane']])) - np.degrees(np.arctan(r['tan_ref']))) for r in dis])
    e_lm = np.array([abs(np.degrees(np.arctan(r['lm_w'] * 1e3 / vl[r['plane']])) - np.degrees(np.arctan(r['tan_ref']))) for r in dis])
    chi_lm_lower = np.array([r['lm_chi2'] < r['prod_chi2'] for r in dis])
    out['prod_vs_lm']['disagree'] = dict(n=len(dis), lm_closer_to_ref_pct=float(100 * np.mean(e_lm < e_prod)),
                                         lm_lower_chi2_pct=float(100 * np.mean(chi_lm_lower)),
                                         median_err_prod_deg=float(np.median(e_prod)), median_err_lm_deg=float(np.median(e_lm)))

# ---- errors
out['errors'] = {}
for plane in 'xy':
    R = [r for r in both if r['plane'] == plane]
    curv = np.array([np.sqrt(max(r['prod_p0_err'] ** 2 - 0.33 ** 2, 0)) for r in R])
    out['errors'][plane] = dict(
        prod_p0_curvature_mm_median=float(np.median(curv)),
        lm_p0_err_mm_median=float(np.median([r['lm_p0_err'] for r in R])),
        prod_t0_err_ns_median=float(np.nanmedian([r['prod_t0_err'] for r in R])),
        lm_t0_err_ns_median=float(np.nanmedian([r['lm_t0_err'] for r in R])),
        prod_w_err_umns_median=float(np.nanmedian([r['prod_w_err'] * 1e3 for r in R])),
        lm_w_err_umns_median=float(np.nanmedian([r['lm_w_err'] * 1e3 for r in R])),
        rho_p0t0_median=float(np.nanmedian([r['lm_rho_p0t0'] for r in R])),
        rho_p0t0_abs_p90=float(np.nanpercentile(np.abs([r['lm_rho_p0t0'] for r in R]), 90)),
    )

# ---- lattice shift degeneracy
S = [r for r in rows if 'shift_chi2_ref' in r and np.isfinite(r['shift_chi2_ref'])]
d_raw = np.array([(r['shift_chi2'] - r['prod_chi2']) / r['prod_chi2'] for r in S])
d_ref = np.array([r['shift_chi2_ref'] - r['prod_chi2'] for r in S])
sc = np.array([r['prod_chi2'] / max(r['prod_dof'], 1) for r in S])
stayed = np.array([abs(r['shift_t0_ref'] - r['prod_t0']) > 30.0 for r in S])     # refinement stayed in the other basin
out['lattice_shift'] = dict(
    n=len(S),
    raw_shift_dchi2_over_chi2_median=float(np.median(d_raw)),
    refined_stays_in_other_basin_pct=float(100 * np.mean(stayed)),
    refined_dchi2_over_chi2_median=float(np.median(d_ref[stayed] / np.array([r['prod_chi2'] for r in S])[stayed])),
    refined_dchi2_in_sigma_units_median=float(np.median(d_ref[stayed] / sc[stayed])),
    other_basin_within_1sig_pct=float(100 * np.mean((d_ref < sc) & stayed)),
    other_basin_within_3sig_pct=float(100 * np.mean((d_ref < 3 * sc) & stayed)),
    other_basin_BETTER_pct=float(100 * np.mean((d_ref < -1.0) & stayed)),
)

# ---- kgap vs prod
KG = [r for r in rows if 'kgap_chi2' in r and 'prod_chi2' in r]
out['kgap_vs_prod'] = dict(
    n=len(KG),
    chi2_ratio_median=float(np.median([r['kgap_chi2'] / r['prod_chi2'] for r in KG])),
    same_w_1umns_pct=float(100 * np.mean([abs(r['kgap_w'] - r['prod_w']) * 1e3 < 1.0 for r in KG])),
    dt0_median_ns=float(np.median([r['kgap_t0'] - r['prod_t0'] for r in KG])),
    dt0_rsig_ns=float(rsig([r['kgap_t0'] - r['prod_t0'] for r in KG])),
    qsum_ratio_median=float(np.median([r['kgap_q_sum'] / r['prod_q_sum'] for r in KG])),
    prod_qsum_gt_1e5_pct=float(100 * np.mean([r['prod_q_sum'] > 1e5 for r in KG])),
    kgap_qsum_gt_1e5_pct=float(100 * np.mean([r['kgap_q_sum'] > 1e5 for r in KG])),
)
# x-y t0 consistency per arm (both planes present), by ftst-pair class
ev = {}
for r in rows:
    ev.setdefault(r['eid'], {})[r['plane']] = r
for arm in ('prod', 'lm', 'kgap'):
    dts = {}
    for e in ev.values():
        if 'x' in e and 'y' in e and f'{arm}_t0' in e['x'] and f'{arm}_t0' in e['y']:
            key = f"{e['x']['ftst']}-{e['y']['ftst']}"
            dts.setdefault(key, []).append(e['x'][f'{arm}_t0'] - e['y'][f'{arm}_t0'])
    big = {k: v for k, v in dts.items() if len(v) >= 40}
    out[f'xy_t0_{arm}'] = {k: dict(n=len(v), median=float(np.median(v)), rsig=float(rsig(v)),
                                   within_60_pct=float(100 * np.mean(np.abs(np.array(v) - np.median(v)) < 60)))
                           for k, v in big.items()}

json.dump(out, open(sys.argv[1].replace('.pkl', '_summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
