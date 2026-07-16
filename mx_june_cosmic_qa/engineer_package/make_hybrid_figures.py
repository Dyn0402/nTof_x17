#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_hybrid_figures.py

Presentation-quality figures for the det3 (A) HYBRID tracking result, for the
engineer package. Replicates the estimator pipeline of
``mx_june_cosmic_qa/34_hybrid_tracking.py`` (sat_det3, veto50) exactly —
regression trained on EVEN event ids, everything evaluated on the ODD-id
holdout — so the numbers on the figures match the published ones
(|θ|<5°: σ68 ≈ 1.75° at 97% coverage; plateau |θ|>8°: 1.86° at 98%).
Fig 16 compares the hybrid against the SAME method with the head-on step
disabled ("high-angle-only" = the unshared+calibrated time-fit tan_seg, the
hybrid's own plateau branch): it matches the hybrid above ~5° and rises to
~11° on near-vertical tracks -- the honest ablation of what the head-on
algorithm buys. (NOT the raw production time-fit, which sits ~7° everywhere.)

Inputs (all cached, no waveform pass):
    <alignment_tpc_veto50>/microtpc_metrics/microtpc_segments.csv   (script 31)
    <alignment_tpc_veto50>/headon/headon_features.csv               (script 33)
    event cache + alignment + M3 v2 rays

Outputs (PNG dpi=200 + lossless PDF twin, written next to this script):
    figures/15-det3A-hybrid-angle-correlation.{png,pdf}
    figures/16-det3A-hybrid-angular-resolution-vs-angle.{png,pdf}

Usage:  ../../.venv/bin/python make_hybrid_figures.py
        (or from mx_june_cosmic_qa/:  ../.venv/bin/python engineer_package/make_hybrid_figures.py)
"""
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
QA_DIR = os.path.dirname(HERE)                      # mx_june_cosmic_qa/
if QA_DIR not in sys.path:
    sys.path.insert(0, QA_DIR)
from qa_config import get_config, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

# ---- constants: identical to 34_hybrid_tracking.py -------------------------
RUN_KEY = 'sat_det3'
VETO = 50
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
RES_CUT_MM = 10.0
TAN_SWITCH = 0.09          # ~5 deg: below this the time fit has no lever arm
PLATEAU_TAN = (0.12, 0.55)
MAX_TAN = 0.7
FEATS = ['tot_lead', 'q_frac', 'n_u', 'n_raw', 'a_asym', 'a_lead', 't_delay']

CFG = get_config(RUN_KEY)
tag = f'_veto{VETO}'
SEG_CSV = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'microtpc_metrics',
                       'microtpc_segments.csv')
FEAT_CSV = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'headon',
                        'headon_features.csv')
FIG_DIR = os.path.join(HERE, 'figures')

# ---- style ------------------------------------------------------------------
BLUE = '#2a78d6'          # hybrid accent
GREY = '#7a7975'          # previous method
INK = '#0b0b0b'
INK2 = '#52514e'
SEQ_BLUES = LinearSegmentedColormap.from_list(
    'seq_blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
                  '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({
    'font.size': 15, 'axes.titlesize': 16.5, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.linewidth': 1.0, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})


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


def profile_s68(x, y, edges, min_n=60):
    """sigma68 of y in bins of x (as 34's profile(..., 's68'))."""
    ctr, val, err = [], [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = (x >= b0) & (x < b1) & np.isfinite(y)
        n = m.sum()
        if n < min_n:
            continue
        q = np.percentile(y[m], [16, 50, 84])
        s = 0.5 * (q[2] - q[0])
        ctr.append(0.5 * (b0 + b1)); val.append(s)
        err.append(s / np.sqrt(2 * n))
    return np.array(ctr), np.array(val), np.array(err)


def build_estimators():
    """Replicates 34_hybrid_tracking.py main() up to the estimator tables."""
    cache_res = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache_res, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
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

    est = {}
    for p, d in planes.items():
        train = (d.index.values % 2 == 0)

        # (1) production angle: own robust ridge v (current algorithm)
        T, Sp = d['tan_ref'].to_numpy(), d['S_prod'].to_numpy()
        vs = [robust_line(T[m], Sp[m]) for m in
              [(T > 0.06) & (T < 0.55), (T < -0.06) & (T > -0.55)]]
        v_prod = np.nanmean(vs)
        d['tan_prod'] = d['S_prod'] / v_prod

        # (2) unshared + calibrated segment angle
        Ss = d['S_seg'].to_numpy()
        vs = [robust_line(T[m & np.isfinite(Ss)], Ss[m & np.isfinite(Ss)]) for m in
              [(T > 0.06) & (T < 0.55), (T < -0.06) & (T > -0.55)]]
        v_seg = np.nanmean(vs)
        tan_det = Ss / v_seg
        m_pl = (np.abs(T) > PLATEAU_TAN[0]) & (np.abs(T) < PLATEAU_TAN[1]) \
            & np.isfinite(tan_det)
        b = float(np.median(np.abs(tan_det[m_pl]) - np.abs(T[m_pl])))
        d['tan_seg'] = tan_det - np.sign(tan_det) * b

        # (3) |tan| regression from signature features (train: even eids)
        F = d[FEATS].to_numpy()
        ok = np.isfinite(F).all(axis=1)
        y = np.abs(d['tan_ref'].to_numpy())
        mu, sd = np.nanmean(F[ok & train], axis=0), np.nanstd(F[ok & train], axis=0)
        Z = (F - mu) / sd
        A = np.c_[Z[ok & train], np.ones((ok & train).sum())]
        w, *_ = np.linalg.lstsq(A, y[ok & train], rcond=None)
        s_lin = np.full(len(d), np.nan)
        s_lin[ok] = Z[ok] @ w[:-1] + w[-1]
        qs = np.nanquantile(s_lin[ok & train], np.linspace(0.02, 0.98, 25))
        ctr_s, med_t = [], []
        for lo_, hi_ in zip(qs[:-1], qs[1:]):
            m = ok & train & (s_lin >= lo_) & (s_lin < hi_)
            if m.sum() > 50:
                ctr_s.append(np.median(s_lin[m])); med_t.append(np.median(y[m]))
        ctr_s = np.array(ctr_s)
        med_t = np.maximum.accumulate(np.array(med_t))
        tan_reg_abs = np.full(len(d), np.nan)
        tan_reg_abs[ok] = np.interp(s_lin[ok], ctr_s, med_t)

        # (4) sign at low angle: Fisher on signed asymmetries (2-10 deg, even)
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
        d['tan_reg'] = sign_hat * tan_reg_abs

        # (5) HYBRID: the regressor decides the regime
        use_seg = np.isfinite(d['tan_seg']) & (np.abs(d['tan_seg']) < 1.5) \
            & (tan_reg_abs > TAN_SWITCH)
        d['tan_hyb'] = np.where(use_seg, d['tan_seg'], d['tan_reg'])
        est[p] = d
        print(f'plane {p}: v_prod={v_prod:.1f} v_seg={v_seg:.1f} b={b:+.4f} '
              f'trained on {int((ok & train).sum()):,} events')
    return est


def collect(est, col):
    """(theta_ref, dtheta) on the ODD-eid holdout, both planes (as 34)."""
    dth, thr = [], []
    for p, d in est.items():
        te = d.index.values % 2 == 1
        t_est = d[col].to_numpy()[te]
        t_ref = d['tan_ref'].to_numpy()[te]
        m = np.isfinite(t_est) & (np.abs(t_est) < 1.5)
        dth.append(np.degrees(np.arctan(t_est[m])) - np.degrees(np.arctan(t_ref[m])))
        thr.append(np.degrees(np.arctan(t_ref[m])))
    return np.concatenate(thr), np.concatenate(dth)


def coverage(est, col):
    cov_th, cov_v = [], []
    for p, d in est.items():
        te = d.index.values % 2 == 1
        t_est = d[col].to_numpy()[te]
        t_ref = d['tan_ref'].to_numpy()[te]
        cov_th.append(np.degrees(np.arctan(t_ref)))
        cov_v.append((np.isfinite(t_est) & (np.abs(t_est) < 1.5)).astype(float))
    return np.concatenate(cov_th), np.concatenate(cov_v)


def band_stats(est, col):
    """(coverage, bias, s68) in the |theta|<5 and |theta|>8 bands (as 34)."""
    thr, dth = collect(est, col)
    cth, cv = coverage(est, col)
    out = {}
    for name, m_r, m_c in [('lt5', np.abs(thr) < 5, np.abs(cth) < 5),
                           ('gt8', np.abs(thr) > 8, np.abs(cth) > 8)]:
        q = np.percentile(dth[m_r], [16, 50, 84])
        out[name] = (cv[m_c].mean(), q[1], 0.5 * (q[2] - q[0]))
    return out


def save(fig, stem):
    png = os.path.join(FIG_DIR, stem + '.png')
    pdf = os.path.join(FIG_DIR, stem + '.pdf')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f'wrote {png} (+ .pdf)')


def fig_correlation(est, hyb):
    """Fig 15: hybrid chamber angle vs telescope angle, 2D density."""
    thr, dth = collect(est, 'tan_hyb')
    the = thr + dth
    lim = 30

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    hb = ax.hexbin(thr, the, gridsize=64, extent=(-lim, lim, -lim, lim),
                   cmap=SEQ_BLUES, norm=LogNorm(), mincnt=1, linewidths=0.2)
    ax.plot([-lim, lim], [-lim, lim], ls='--', lw=1.8, color=INK2, zorder=3)
    ax.text(-21.5, -19.7, 'perfect agreement', color=INK2, fontsize=13.5,
            ha='center', va='bottom', rotation=45, rotation_mode='anchor')

    cov, _, s68 = hyb['lt5']
    cov8, _, s688 = hyb['gt8']
    ax.text(0.035, 0.965,
            f'resolution  $\\sigma_{{68}} \\approx$ {0.5*(s68+s688):.1f}$^\\circ$\n'
            f'angle measured for $\\approx$ {100*0.5*(cov+cov8):.0f}% of tracks\n'
            'including straight-down tracks',
            transform=ax.transAxes, va='top', ha='left', fontsize=15,
            color=INK, linespacing=1.55,
            bbox=dict(boxstyle='round,pad=0.55', fc='white', ec=INK2, lw=1.0))

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-30, 31, 10)); ax.set_yticks(np.arange(-30, 31, 10))
    ax.set_xlabel('telescope angle [deg]')
    ax.set_ylabel('chamber angle [deg]')
    ax.set_title('Track angle from the chamber alone\nvs reference telescope (hybrid method)',
                 pad=12)
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.92)
    cb.set_label('muon tracks per cell', fontsize=14.5)
    cb.set_ticks([1, 10, 100])
    cb.set_ticklabels(['1', '10', '100'])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=13.5)
    fig.tight_layout()
    save(fig, '15-det3A-hybrid-angle-correlation')
    plt.close(fig)


def fig_resolution(est, hyb, base):
    """Fig 16: sigma68 vs |theta_ref|, hybrid vs the SAME method without the
    head-on step. The baseline is the unshared+calibrated time-fit (tan_seg) --
    exactly the hybrid's high-angle branch, applied at all angles. By
    construction it equals the hybrid above the ~5 deg switch and only rises at
    low angle where the drift-time fit loses its lever arm; hence
    'high-angle-only'. (The older figure used the raw production time-fit, which
    sits ~7 deg everywhere and never converges -- a misleading baseline.)"""
    edges = np.arange(0, 30.1, 2.5)

    thr_b, dth_b = collect(est, 'tan_seg')
    cb, sb, eb = profile_s68(np.abs(thr_b), dth_b, edges, 80)
    thr_h, dth_h = collect(est, 'tan_hyb')
    ch, sh, eh = profile_s68(np.abs(thr_h), dth_h, edges, 80)

    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    ax.errorbar(cb, sb, yerr=eb, fmt='s--', color=GREY, lw=2.2, ms=7.5,
                capsize=3, label='high-angle-only (no head-on step)', zorder=2)
    ax.errorbar(ch, sh, yerr=eh, fmt='o-', color=BLUE, lw=2.6, ms=8.5,
                capsize=3, label='hybrid (current)', zorder=3)

    ax.set_yscale('log')
    ax.set_ylim(1.0, 20)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    xmax = max(cb.max(), ch.max()) + 1.5
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 0.1, 5))
    ax.grid(True, which='major', axis='both', color='#dddcd8', lw=0.9)
    ax.set_axisbelow(True)

    # shade the head-on regime (< ~5 deg) that the extra step recovers
    ax.axvspan(0, 5, color='#eef4fc', zorder=0)
    ax.text(2.5, 1.06, 'head-on\nregime', color='#4a6fa5', fontsize=11.5,
            ha='center', va='bottom', linespacing=1.15)

    # direct annotations
    ax.annotate('without the head-on step the\ntime-fit fails on near-vertical\n'
                'tracks ($\\approx$11$^\\circ$, only $\\sim$40% covered)',
                xy=(cb[0], sb[0]), xytext=(5.4, 7.2), fontsize=13.5,
                color='#5a5955', ha='left', va='center', linespacing=1.4,
                arrowprops=dict(arrowstyle='->', color='#8a8983', lw=1.6,
                                shrinkB=6, connectionstyle='arc3,rad=0.15'))
    ax.annotate('the two agree here:\nsame method above $\\sim$5$^\\circ$',
                xy=(11.5, np.interp(11.5, ch, sh)), xytext=(13.5, 4.6),
                fontsize=12.5, color=INK2, ha='center', va='center',
                linespacing=1.35,
                arrowprops=dict(arrowstyle='->', color='#8a8983', lw=1.3,
                                shrinkB=5, connectionstyle='arc3,rad=0.2'))
    ax.annotate('hybrid: uniform $\\approx$1.8$^\\circ$ at all angles',
                xy=(6.2, np.interp(6.2, ch, sh)), xytext=(6.0, 2.9),
                fontsize=15.5, color=BLUE, fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.6,
                                shrinkB=5, connectionstyle='arc3,rad=0.2'))

    ax.set_xlabel('telescope angle magnitude [deg]')
    ax.set_ylabel('angular resolution $\\sigma_{68}$ [deg]  (log scale)')
    ax.set_title('Angular resolution vs track angle:\nthe head-on step removes the vertical-track blind spot',
                 pad=12)
    ax.legend(loc='upper right', frameon=True, edgecolor=INK2, framealpha=0.95,
              borderpad=0.7)
    fig.tight_layout()
    save(fig, '16-det3A-hybrid-angular-resolution-vs-angle')
    plt.close(fig)


def main():
    est = build_estimators()
    hyb = band_stats(est, 'tan_hyb')
    seg = band_stats(est, 'tan_seg')      # high-angle-only: hybrid minus head-on step
    prod = band_stats(est, 'tan_prod')    # raw production time-fit (context only)
    print('\nband stats (odd-eid holdout):')
    for nm, st in [('hybrid', hyb), ('high-angle-only', seg), ('production', prod)]:
        for band in ('lt5', 'gt8'):
            cov, bias, s68 = st[band]
            print(f'  {nm:16s} {band}: cov={100*cov:5.1f}%  bias={bias:+5.2f}deg  '
                  f's68={s68:5.2f}deg')
    # sanity vs published (34_hybrid_tracking.py, hybrid_summary.csv)
    assert abs(hyb['lt5'][2] - 1.75) < 0.3, 'hybrid lt5 s68 off vs published 1.75'
    assert abs(hyb['gt8'][2] - 1.86) < 0.3, 'hybrid gt8 s68 off vs published 1.86'
    # the high-angle-only baseline must CONVERGE to the hybrid on the plateau
    # (same estimator there) and DEGRADE in the head-on band.
    assert abs(seg['gt8'][2] - hyb['gt8'][2]) < 0.4, \
        'high-angle-only should match hybrid on the plateau'
    assert seg['lt5'][2] > 4.0, 'high-angle-only should degrade in the head-on band'

    fig_correlation(est, hyb)
    fig_resolution(est, hyb, seg)


if __name__ == '__main__':
    main()
