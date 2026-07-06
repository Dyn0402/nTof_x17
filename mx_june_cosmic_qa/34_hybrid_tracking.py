#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
34_hybrid_tracking.py

HYBRID micro-TPC angle estimator: time-fit tracking where it works,
signature-based regression where it doesn't (|theta| < ~5 deg).

The time-fit angle needs a spatial lever arm and fails/blows up for
near-vertical tracks. But the head-on signature features (33) are
CONTINUOUS, monotonic functions of |theta| near zero:

    tot_lead   (lead time-over-threshold): whole column on one strip at
               theta=0, shrinks as the column spreads over strips
    q_frac     (lead-charge fraction): same geometry, charge-normalised
    n_raw/n_u  footprint before/after unsharing
    a_asym     grows with |theta| (deep-side neighbour attenuated)

So instead of a binary head-on TAG, regress |tan theta| from the features
(linear least squares on standardized features -> monotonic binned-median
calibration; trained on EVEN eids, everything evaluated on ODD eids).

The SIGN at low angle comes from the signed left/right asymmetries --
the shallow (mesh-side) neighbour has MORE charge and fires EARLIER:
    sign features: (a_r - a_l)/(a_r + a_l),  (t_r - t_l)
(a small Fisher discriminant trained on 2-10 deg tracks), with the
segment sign taking priority when a segment exists.

HYBRID rule (per plane):
    segment exists and |tan_seg| > TAN_SWITCH  ->  tan_seg (time fit)
    otherwise                                  ->  sign_hat * |tan_reg|

Benchmarked against (1) the production time-fit angle (current algorithm)
and (2) the unshared+calibrated time-fit alone: bias, sigma68 and COVERAGE
vs theta_ref, with the headline being the |theta|<5 deg band where the
track-only estimators have no (or nonsense) answer.

Inputs (all cached, no waveform pass):
    <headon>/headon_features.csv        (33)
    <microtpc_metrics>/microtpc_segments.csv  (31)
    event cache + alignment + v2 rays

Usage: ../.venv/bin/python 34_hybrid_tracking.py sat_det3 [--veto=50]
Output: <alignment_tpc_vetoN>/hybrid/ hybrid_tracking.png,
        hybrid_summary.csv + stdout tables
"""
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from qa_config import config_from_argv, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
CHI2_CUT = 5.0
RES_CUT_MM = 10.0
TAN_SWITCH = 0.09          # ~5 deg: below this the time fit has no lever arm
PLATEAU_TAN = (0.12, 0.55)
MAX_TAN = 0.7
FEATS = ['tot_lead', 'q_frac', 'n_u', 'n_raw', 'a_asym', 'a_lead', 't_delay']

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'hybrid')
SEG_CSV = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'microtpc_metrics',
                       'microtpc_segments.csv')
FEAT_CSV = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'headon',
                        'headon_features.csv')


def robust_line(x, y, n_iter=4, clip=3.0):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    keep = np.ones(len(x), bool)
    p = (np.nan, np.nan)
    for _ in range(n_iter):
        if keep.sum() < 10:
            return np.nan
        p = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < clip * s
    return float(p[0])


def profile(x, y, edges, min_n=40, stat='med'):
    ctr, val, err = [], [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = (x >= b0) & (x < b1) & np.isfinite(y)
        n = m.sum()
        if n < min_n:
            continue
        ctr.append(0.5 * (b0 + b1))
        if stat == 'med':
            q = np.percentile(y[m], [16, 50, 84])
            val.append(q[1]); err.append(0.5 * (q[2] - q[0]) / np.sqrt(n))
        elif stat == 's68':
            q = np.percentile(y[m], [16, 50, 84])
            val.append(0.5 * (q[2] - q[0])); err.append(0.5 * (q[2] - q[0]) / np.sqrt(2 * n))
        elif stat == 'eff':
            k = float(np.sum(y[m]))
            val.append(k / n); err.append(np.sqrt(max(k, 0.25) * (1 - k / n)) / n)
    return np.array(ctr), np.array(val), np.array(err)


def main():
    # ---- reference + production angles per plane ----
    cache_res = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache_res, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)

    rows = []
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        rows.append(dict(eid=r.event_id, tan_ref_x=tx, tan_ref_y=ty,
                         S_prod_x=r.x_fit.slope_mm_per_ns * 1000.0,
                         S_prod_y=r.y_fit.slope_mm_per_ns * 1000.0))
    ev = pd.DataFrame(rows).set_index('eid')
    print(f'{len(ev):,} matched events')

    # ---- per-plane long table: ref, production, segment, features ----
    seg = pd.read_csv(SEG_CSV)
    ft = pd.read_csv(FEAT_CSV)
    ft = ft[np.isfinite(ft['t_lead'])].copy()
    ft['a_asym'] = np.abs(ft['a_r'] - ft['a_l']) / (ft['a_r'] + ft['a_l'])
    ft['a_asym_sgn'] = (ft['a_r'] - ft['a_l']) / (ft['a_r'] + ft['a_l'])
    ft['t_asym_sgn'] = ft['t_r'] - ft['t_l']
    ft['t_delay'] = np.minimum(ft['t_l'], ft['t_r']) - ft['t_lead']

    planes = {}
    for p in 'xy':
        d = pd.DataFrame(index=ev.index)
        d['tan_ref'] = ev[f'tan_ref_{p}']
        d['S_prod'] = ev[f'S_prod_{p}']
        s = seg[seg['plane'] == p].drop_duplicates('eid').set_index('eid')
        d['S_seg'] = s['S_um_ns'].reindex(d.index)
        f = ft[ft['plane'] == p].drop_duplicates('eid').set_index('eid')
        for c in FEATS + ['a_asym_sgn', 't_asym_sgn']:
            d[c] = f[c].reindex(d.index)
        d = d[np.abs(d['tan_ref']) < MAX_TAN]
        planes[p] = d

    # ---- estimator construction, per plane ----
    est = {}
    consts = {}
    for p, d in planes.items():
        train = (d.index.values % 2 == 0)
        test = ~train

        # (1) production angle: own robust ridge v (no calibration = current algo)
        T, Sp = d['tan_ref'].to_numpy(), d['S_prod'].to_numpy()
        vs = [robust_line(T[m], Sp[m]) for m in
              [(T > 0.06) & (T < 0.55), (T < -0.06) & (T > -0.55)]]
        v_prod = np.nanmean(vs)
        d['tan_prod'] = d['S_prod'] / v_prod

        # (2) unshared + calibrated segment angle (as 31)
        Ss = d['S_seg'].to_numpy()
        vs = [robust_line(T[m & np.isfinite(Ss)], Ss[m & np.isfinite(Ss)]) for m in
              [(T > 0.06) & (T < 0.55), (T < -0.06) & (T > -0.55)]]
        v_seg = np.nanmean(vs)
        tan_det = Ss / v_seg
        m_pl = (np.abs(T) > PLATEAU_TAN[0]) & (np.abs(T) < PLATEAU_TAN[1]) & np.isfinite(tan_det)
        b = float(np.median(np.abs(tan_det[m_pl]) - np.abs(T[m_pl])))
        d['tan_seg'] = tan_det - np.sign(tan_det) * b

        # (3) |tan| regression from features (train on even eids only)
        F = d[FEATS].to_numpy()
        ok = np.isfinite(F).all(axis=1)
        mu, sd = np.nanmean(F[ok & train], axis=0), np.nanstd(F[ok & train], axis=0)
        Z = (F - mu) / sd
        y = np.abs(d['tan_ref'].to_numpy())
        A = np.c_[Z[ok & train], np.ones((ok & train).sum())]
        w, *_ = np.linalg.lstsq(A, y[ok & train], rcond=None)
        s_lin = np.full(len(d), np.nan)
        s_lin[ok] = Z[ok] @ w[:-1] + w[-1]
        # monotonic binned-median calibration on the training half
        qs = np.nanquantile(s_lin[ok & train], np.linspace(0.02, 0.98, 25))
        ctr_s, med_t = [], []
        for lo_, hi_ in zip(qs[:-1], qs[1:]):
            m = ok & train & (s_lin >= lo_) & (s_lin < hi_)
            if m.sum() > 50:
                ctr_s.append(np.median(s_lin[m])); med_t.append(np.median(y[m]))
        ctr_s, med_t = np.array(ctr_s), np.maximum.accumulate(np.array(med_t))
        tan_reg_abs = np.full(len(d), np.nan)
        tan_reg_abs[ok] = np.interp(s_lin[ok], ctr_s, med_t)

        # (4) sign at low angle: Fisher on signed asymmetries, trained on
        #     2-10 deg tracks (even eids); fallback sign = production slope
        G = d[['a_asym_sgn', 't_asym_sgn']].to_numpy()
        okg = np.isfinite(G).all(axis=1)
        band = (np.abs(y) > np.tan(np.radians(2))) & (np.abs(y) < np.tan(np.radians(10)))
        sgn_true = np.sign(d['tan_ref'].to_numpy())
        mtr = okg & train & band
        mu1 = G[mtr & (sgn_true > 0)].mean(axis=0)
        mu0 = G[mtr & (sgn_true < 0)].mean(axis=0)
        Sw = np.cov(G[mtr & (sgn_true > 0)].T) + np.cov(G[mtr & (sgn_true < 0)].T)
        wg = np.linalg.solve(Sw, mu1 - mu0)
        g = np.full(len(d), np.nan)
        g[okg] = G[okg] @ wg
        sign_feat = np.sign(g)
        sign_hat = np.where(np.isfinite(d['tan_seg']), np.sign(d['tan_seg']),
                            np.where(np.isfinite(g) & (g != 0), sign_feat,
                                     np.sign(d['tan_prod'])))
        # sign accuracy vs angle (odd eids)
        acc_ctr, acc_val, _ = profile(
            np.degrees(np.arctan(np.abs(y)))[test & okg],
            (sign_feat == sgn_true).astype(float)[test & okg],
            np.arange(0, 20, 2), 40, 'eff')

        d['tan_reg'] = sign_hat * tan_reg_abs

        # (5) HYBRID -- the REGRESSOR decides the regime (it is the low-angle
        # expert; switching on the segment angle mis-assigns true low-angle
        # tracks whose time fit fluctuated high): regression below the switch
        # or when no segment exists, time fit above it.
        use_seg = np.isfinite(d['tan_seg']) & (np.abs(d['tan_seg']) < 1.5) \
            & (tan_reg_abs > TAN_SWITCH)
        d['tan_hyb'] = np.where(use_seg, d['tan_seg'], d['tan_reg'])

        consts[p] = dict(v_prod=v_prod, v_seg=v_seg, b=b, w=w, wg=wg,
                         sign_acc=(acc_ctr, acc_val))
        est[p] = d
        print(f'plane {p}: v_prod={v_prod:.1f} v_seg={v_seg:.1f} b={b:+.4f}  '
              f'regression trained on {int((ok & train).sum()):,} events '
              f'(feature weights {np.round(w[:-1], 3)})')

    # ---- evaluation (ODD eids only — honest holdout) ----
    def collect(col):
        dth, thr = [], []
        for p, d in est.items():
            te = d.index.values % 2 == 1
            t_est = d[col].to_numpy()[te]
            t_ref = d['tan_ref'].to_numpy()[te]
            m = np.isfinite(t_est) & (np.abs(t_est) < 1.5)
            dth.append(np.degrees(np.arctan(t_est[m])) - np.degrees(np.arctan(t_ref[m])))
            thr.append(np.degrees(np.arctan(t_ref[m])))
        return np.concatenate(thr), np.concatenate(dth)

    def coverage(col):
        cov_th, cov_v = [], []
        for p, d in est.items():
            te = d.index.values % 2 == 1
            t_est = d[col].to_numpy()[te]
            t_ref = d['tan_ref'].to_numpy()[te]
            cov_th.append(np.degrees(np.arctan(t_ref)))
            cov_v.append((np.isfinite(t_est) & (np.abs(t_est) < 1.5)).astype(float))
        return np.concatenate(cov_th), np.concatenate(cov_v)

    estimators = [('production time-fit (current)', 'tan_prod', 'tab:blue'),
                  ('unshared+cal time-fit (track-only)', 'tan_seg', 'tab:cyan'),
                  ('signature regression only', 'tan_reg', 'tab:orange'),
                  ('HYBRID [regression <5° / segment >5°]', 'tan_hyb', 'crimson')]

    edges = np.arange(-27.5, 28, 2.5)
    print(f'\n== |θ_ref| < 5° band (holdout) ==')
    print(f'{"estimator":40s} {"coverage":>9s} {"bias":>7s} {"σ68":>6s}')
    summary = []
    band_stats = {}
    for name, col, c in estimators:
        thr, dth = collect(col)
        cth, cv = coverage(col)
        mb = np.abs(thr) < 5
        mcb = np.abs(cth) < 5
        cov = cv[mcb].mean()
        q = np.percentile(dth[mb], [16, 50, 84]) if mb.sum() > 50 else [np.nan]*3
        s68 = 0.5 * (q[2] - q[0])
        print(f'{name:40s} {100*cov:8.1f}% {q[1]:+6.2f}° {s68:5.2f}°')
        band_stats[name] = (cov, q[1], s68)
        summary.append(dict(estimator=name, band='lt5', coverage=cov,
                            bias_deg=q[1], s68_deg=s68))
        # plateau too
        mp = np.abs(thr) > 8
        qp = np.percentile(dth[mp], [16, 50, 84])
        summary.append(dict(estimator=name, band='gt8',
                            coverage=cv[np.abs(cth) > 8].mean(),
                            bias_deg=qp[1], s68_deg=0.5 * (qp[2] - qp[0])))
    pd.DataFrame(summary).to_csv(os.path.join(OUT, 'hybrid_summary.csv'), index=False)

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.5))

    ax = axes[0, 0]
    thr, dth = collect('tan_hyb')
    ax.hist2d(thr, thr + dth, bins=[70, 70], range=[[-30, 30], [-30, 30]],
              norm=LogNorm(), cmap='viridis')
    ax.plot([-30, 30], [-30, 30], 'r--', lw=1)
    ax.axvspan(-5, 5, color='w', alpha=0.12)
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('θ_hybrid [deg]')
    ax.set_title('hybrid: correlation (shaded = regression regime)')

    ax = axes[0, 1]
    for name, col, c in estimators:
        thr, dth = collect(col)
        ctr, med, err = profile(thr, dth, edges, 60, 'med')
        ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=4, color=c, label=name.split(' (')[0])
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('median Δθ [deg]')
    ax.set_title('bias'); ax.set_ylim(-6, 6); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    for name, col, c in estimators:
        thr, dth = collect(col)
        ctr, s68, err = profile(thr, dth, edges, 60, 's68')
        ax.errorbar(ctr, s68, yerr=err, fmt='s-', ms=4, color=c, label=name.split(' (')[0])
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('σ68(Δθ) [deg]')
    ax.set_title('resolution — hybrid closes the |θ|<5° hole')
    ax.set_ylim(0, 10); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, col, c in estimators:
        cth, cv = coverage(col)
        ctr, v, e = profile(cth, cv, edges, 60, 'eff')
        ax.errorbar(ctr, 100 * v, yerr=100 * e, fmt='o-', ms=4, color=c,
                    label=name.split(' (')[0])
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('coverage [%]')
    ax.set_title('fraction of matched events with an angle estimate')
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')

    ax = axes[1, 1]
    # |theta| response at low angle: median |theta_est| vs |theta_ref|
    for name, col, c in [('signature regression', 'tan_reg', 'tab:orange'),
                         ('unshared time-fit', 'tan_seg', 'tab:cyan'),
                         ('hybrid', 'tan_hyb', 'crimson')]:
        thr, dth = collect(col)
        ath, aest = np.abs(thr), np.abs(thr + dth)
        ctr, med, err = profile(ath, aest, np.arange(0, 15, 1.0), 60, 'med')
        ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=4, color=c, label=name)
    ax.plot([0, 15], [0, 15], 'k--', lw=1)
    ax.set_xlabel('|θ_ref| [deg]'); ax.set_ylabel('median |θ_est| [deg]')
    ax.set_title('low-angle response (unbiased = diagonal)')
    ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 2]
    ax.axis('off')
    lines = [f'|θ_ref| < 5° BAND (odd-eid holdout)', '',
             f'{"estimator":26s} {"cov":>5s} {"bias":>6s} {"σ68":>5s}']
    for name, col, c in estimators:
        cov, b_, s_ = band_stats[name]
        lines.append(f'{name.split(" (")[0]:26s} {100*cov:4.0f}% {b_:+5.1f}° {s_:4.1f}°')
    accx = consts['x']['sign_acc']
    lines += ['',
              'sign model (feature-based, x plane):',
              '  accuracy ' + '  '.join(f'{a:.0f}°:{100*v:.0f}%'
                                        for a, v in zip(*accx)),
              '',
              f'switch: |tan_seg| > {TAN_SWITCH} (≈{np.degrees(np.arctan(TAN_SWITCH)):.0f}°)',
              'regression features:',
              '  ' + ', '.join(FEATS),
              '',
              'Hybrid gives an angle for EVERY event',
              'with a lead strip (~100% coverage) and',
              'a finite, calibrated response at θ→0,',
              'at the cost of |θ|-folding noise from',
              'the sign model below ~3°.']
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top',
            fontsize=10.5, family='monospace')

    fig.suptitle(f'{CFG.RUN} — hybrid micro-TPC tracking '
                 f'(time fit + head-on signature regression), M3 v2 truth',
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'hybrid_tracking.png'), dpi=155)
    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()
