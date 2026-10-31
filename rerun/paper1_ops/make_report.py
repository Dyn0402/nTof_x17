#!/usr/bin/env python3
"""make_report.py --spark D --telgain D --spsgain D [--out O] -- Paper I §4 (operation: HV, gain, discharges) on pass 1.

Inputs (all derived, read only; on EOS under cosmic_bench/x17rerun/paper1_ops/):
  spark/rows/*.json, spark/rays/*.npz   spark_timeline.py, one per bench pass-1 row
  tel_gain/*.parquet                    tel_gain.py: det4's response on every strict telescope track (SPS)
  sps_gain/rows, sps_gain/events        sps_gain.py: det4's charge on the reconstructed SPS events (the trap)
  rerun/effbudget/pass1/results/rows.csv   the loss budget per row (interior_good efficiency, its loss ladder)
  rerun/paper1_res/results/rows.csv        angle / position resolution per row and view (v(t) arm)
  rerun/paper1_charge/results/rows.csv     the bench charge per row (cens_25, f_sat)
Output (default this directory): results/{summary.json,tradeoff.csv,det4_gain.csv,spark_rows.csv}, figures/*.png|pdf,
report.html, and the Paper I figures hv_scan.pdf and spark_deadtime.pdf (papers/paper1/figures.toml points here).

det4's gain at SPS, the estimator: per plateau, the strict telescope tracks inside det4's LIVE region (back-uRWELL
position bins, BIN_MM, in which det4 saw both views on >= LIVE_FRAC of the tracks at the run's highest-HV plateau,
>= LIVE_N tracks), and on them q = q_x + q_y (summed strip peaks over 60 ADC, oscillator channels out).  A missed
crossing has q = 0 and ranks below every seen one; a crossing with a strip at the 3550 rule ranks above every
unsaturated one (rerun/paper1_charge cens_quantile).  The 50th percentile is the gain proxy.
"""
import argparse
import csv
import glob
import html
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_charge'))
from make_report import cens_quantile, style, COL, NAME, INK, MUTED, GRID  # noqa: E402

DETS = ['mx17_2', 'mx17_3', 'mx17_4', 'mx17_6', 'mx17_7']
BIN_MM = 1.0
DEAD_MAX_S = 0.1
LIVE_FRAC, LIVE_N = 0.9, 30
N_BOOT = 200
RNG = np.random.default_rng(11)
# the HV-scan series of the bench (det, run, drift V) and the rows of each that are skeptical / excluded upstream
SCANS = [('mx17_2', 'mx17_det2_det3_overnight_6-22-26', 1000.0),
         ('mx17_3', 'mx17_det3_saturday_scan_6-27-26', 1000.0),
         ('mx17_6', 'mx17_det6_det7_hv_scan_6-26-26', 700.0),
         ('mx17_7', 'mx17_det6_det7_hv_scan_6-26-26', 700.0)]
# not a scan: the 6-26 overnight resist steps ran while det6's gain rose x10 (the early contaminant,
# rerun/vt_within_rows/README.md), so their efficiency-vs-HV mixes HV and time; shown in the trade-off table only
SPS_SERIES = {'run66 CF4 (flat, 205 s plateaus)': ('run66_flat_resist__r', 'Ar/CF4/iso 88/10/2'),
              'run55/56 Ar/CO2/iso (flat)': (('run55_flatdrift__d700', 'run56_m70V__590V', 'run56_m70V__625V'),
                                             'Ar/CO2/iso 95/3/2')}


def boot(fn, *arrs, n=N_BOOT):
    m = len(arrs[0])
    if m < 30:
        return np.nan
    v = []
    for _ in range(n):
        i = RNG.integers(0, m, m)
        v.append(fn(*(a[i] for a in arrs)))
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    return float(np.std(v)) if len(v) > n // 2 else np.nan


# ---------------------------------------------------------------------------------------------------------- det4 SPS
def det4_gain(tel_dir, reg):
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(tel_dir, '*.parquet')))],
                  ignore_index=True)
    D = D[D.strict].copy()
    D['seen'] = (D.n_x > 0) & (D.n_y > 0)
    D['q'] = D.q_x + D.q_y
    D['sat'] = (D.nsat_x > 0) | (D.nsat_y > 0)
    D['run'] = D.key.str.split('__').str[0].str.extract(r'(run\d+)')[0]
    D['V'] = D.key.map(lambda k: float(reg[k]['resist_V']))
    D['drift'] = D.key.map(lambda k: float(reg[k]['drift_V']))
    D['by'] = np.floor(D.by_p / BIN_MM).astype(int)      # det4's stripes run across the back uRWELL's y (1D map)
    rows = []
    for run, G in D.groupby('run'):
        ref_key = G.groupby('key').V.first().idxmax()
        if run == 'run55':                            # the drift ladder: its 243 V/cm plateau is the reference
            ref_key = 'run55_flatdrift__d700'
        R = G[G.key == ref_key]
        m = R.groupby('by').agg(n=('seen', 'size'), f=('seen', 'mean'))
        live = set(m.index[(m.n >= LIVE_N) & (m.f >= LIVE_FRAC)])
        G = G[G.by.isin(live)]
        for key, K in G.groupby('key'):
            q, s, seen = K.q.to_numpy(float), K.sat.to_numpy(bool), K.seen.to_numpy(bool)
            rows.append(dict(key=key, run=run, gas=reg[key]['gas'], V=float(reg[key]['resist_V']),
                             drift_V=float(reg[key]['drift_V']), ref=ref_key, n_live_bins=len(live), n=int(len(K)),
                             eff_live=float(seen.mean()), f_sat=float(s.mean()),
                             c50=cens_quantile(q, s, 0.5), c50_err=boot(lambda a, b: cens_quantile(a, b, 0.5), q, s),
                             c25=cens_quantile(q, s, 0.25),
                             c25_err=boot(lambda a, b: cens_quantile(a, b, 0.25), q, s),
                             med_seen=float(np.median(q[seen & ~s])) if (seen & ~s).sum() else np.nan,
                             peak_med=float(np.median(np.maximum(K.p_x, K.p_y)[seen])) if seen.any() else np.nan,
                             h_c50=cens_quantile((K.h_q_x + K.h_q_y).to_numpy(float),
                                                 ((K.h_nsat_x > 0) | (K.h_nsat_y > 0)).to_numpy(), 0.5)
                             if 'h_q_x' in K and (K.h_n_x > 0).any() else np.nan))
    return pd.DataFrame(rows).sort_values(['run', 'V', 'drift_V'])


def fit_slope(V, y, e):
    k = np.isfinite(y) & np.isfinite(e) & (y > 0) & (e > 0)
    if k.sum() < 3:
        return None
    V, ly, le = V[k], np.log(y[k]), e[k] / y[k]
    w = 1 / le ** 2
    A = np.column_stack([np.ones_like(V), V - V.mean()])
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    c = cov @ (A.T @ (w * ly))
    chi2 = float(np.sum(w * (ly - A @ c) ** 2))
    s = c[1] * 10
    return dict(slope_per10V=float(s), err=float(np.sqrt(cov[1, 1]) * 10), double_V=float(10 * np.log(2) / s)
                if s > 0 else None, chi2=chi2, dof=int(k.sum() - 2), V_min=float(V.min()), V_max=float(V.max()))


# ---------------------------------------------------------------------------------------------------------- bench
def spark_rows(spark_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(spark_dir, 'rows', '*.json'))):
        o = json.load(open(f))
        o['dt_bins'] = json.dumps(o['dt_bins'])
        out.append(o)
    return pd.DataFrame(out)


def deadtime(spark_dir, S):
    """Per chamber: the efficiency of non-discharge rays against the time since the previous discharge epoch, pooled
    over the chamber's rows and normalised to each row's own efficiency on rays more than 1 s after a discharge (or
    with none before them); and the median unsaturated charge, same normalisation."""
    edges = np.array([0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, np.inf])
    res = {}
    for det in DETS:
        keys = S[(S.det == det) & (S.n_epoch >= 20)].key.tolist()
        nb = len(edges) - 1
        k_ = np.zeros(nb); n_ = np.zeros(nb); exp_ = np.zeros(nb); var_ = np.zeros(nb)
        qr = [[] for _ in range(nb)]
        n_ep = 0
        for k in keys:
            z = np.load(os.path.join(spark_dir, 'rays', k + '.npz'))
            if 'in_discharge' not in z.files:
                continue                                 # a row of the superseded format
            ok = ~z['in_spark']
            dt, det_, q = z['dt'][ok], z['det'][ok].astype(float), z['q_unsat'][ok]
            far = dt >= 1.0
            if far.sum() < 30 or det_[far].mean() <= 0:
                continue
            e0 = det_[far].mean()
            qf = q[far & (det_ > 0) & np.isfinite(q)]
            q0 = np.median(qf) if len(qf) >= 20 else np.nan
            n_ep += len(z['epoch_t'])
            b = np.clip(np.digitize(dt, edges) - 1, 0, nb - 1)
            for i in range(nb):
                m = b == i
                # observed vs expected detections at this row's own late-time efficiency (pooled over the rows)
                k_[i] += det_[m].sum(); n_[i] += m.sum(); exp_[i] += e0 * m.sum()
                var_[i] += e0 * (1 - e0) * m.sum()
                if np.isfinite(q0):
                    qq = q[m & (det_ > 0) & np.isfinite(q)]
                    qr[i] += list(qq / q0)
        den = n_
        r = np.where(exp_ > 0, k_ / np.maximum(exp_, 1e-9), np.nan)
        e = np.where(exp_ > 0, np.sqrt(var_) / np.maximum(exp_, 1e-9), np.nan)
        # dead time per discharge: the lost fraction integrated over the bins below DEAD_MAX_S (a resistive-layer
        # recovery is a ms process; wider bins only add noise), with its one-sided 95 % upper limit
        w = np.diff(edges)
        inb = edges[1:] <= DEAD_MAX_S
        d = np.nansum(((1 - r) * w)[inb])
        de = np.sqrt(np.nansum(((e * w) ** 2)[inb]))
        res[det] = dict(rows=len(keys), epochs=int(n_ep), edges=edges.tolist(), n=den.astype(int).tolist(),
                        ratio=r.tolist(), err=e.tolist(),
                        q_ratio=[float(np.median(x)) if len(x) >= 20 else None for x in qr],
                        q_n=[len(x) for x in qr], dead_ms=float(d * 1e3), dead_err_ms=float(de * 1e3),
                        dead_ul95_ms=float(max(d, 0) * 1e3 + 1.645 * de * 1e3))
    return res


def tradeoff(S):
    E = pd.read_csv(os.path.join(REPO, 'rerun', 'effbudget', 'pass1', 'results', 'rows.csv'))
    Rr = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_res', 'results', 'rows.csv'))
    A = Rr.pivot_table(index='key', columns='view', values=['ang_s68_vt', 'pos_mid_s68_vt'], aggfunc='first')
    A.columns = [f'{a}_{b}' for a, b in A.columns]
    C = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_charge', 'results', 'rows.csv'))[['key', 'cens_25', 'f_sat']]
    T = E[['key', 'det', 'run', 'subrun', 'cls', 'resist_V', 'drift_V', 'n_ig', 'eff_ig', 'err_ig', 'loss_spark_ig',
           'loss_no_hit_ig', 'loss_far_ig']].merge(A, left_on='key', right_index=True, how='left') \
        .merge(C, on='key', how='left') \
        .merge(S[['key', 'n_trig', 'trig_rate_hz', 'veto_per_trig', 'spark_per_trig', 'burst_per_trig', 'epoch_rate_hz',
                  'f_in_spark', 'f_in_discharge', 'f_in_burst', 'eff_non_spark']], on='key', how='left')
    T['det'] = T.det.map({v: k for k, v in NAME.items()})       # the budget's short names -> mx17_N
    return T


def optimum(T):
    """Per scan series: the HV that maximises the interior_good efficiency, the plateau (within 1 point of the
    maximum) and where the discharge loss reaches 1 / 5 %."""
    out = []
    for det, run, drift in SCANS:
        s = T[(T.det == det) & (T.run == run) & (T.drift_V == drift)]
        s = s[s.subrun.str.contains('resist') | s.subrun.str.contains('hv_scan')].sort_values('resist_V')
        s = s[s.n_ig >= 100]
        if len(s) < 3:
            continue
        i = s.eff_ig.idxmax()
        plateau = s[s.eff_ig >= s.eff_ig.max() - 1.0]
        def first_over(col, thr):
            o = s[s[col] * 100 >= thr]
            return float(o.resist_V.min()) if len(o) else None
        out.append(dict(det=det, run=run, drift_V=drift, V_best=float(s.at[i, 'resist_V']),
                        eff_best=float(s.at[i, 'eff_ig']), plateau_lo=float(plateau.resist_V.min()),
                        plateau_hi=float(plateau.resist_V.max()),
                        V_discharge_1pct=first_over('f_in_discharge', 1.0),
                        V_discharge_5pct=first_over('f_in_discharge', 5.0),
                        s68_x_best=float(s.at[i, 'ang_s68_vt_x']) if 'ang_s68_vt_x' in s else None,
                        s68_y_best=float(s.at[i, 'ang_s68_vt_y']) if 'ang_s68_vt_y' in s else None))
    return pd.DataFrame(out)


def hits_gain(d):
    """The M3-free proxy (hits_gain.py): per scan series, d ln(lead p50) / dV per view on the rows with f_sat < 0.3."""
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, '*.json')))]
    H = pd.DataFrame([dict(key=r['key'], det=r['det'], run=r['run'], subrun=r['subrun'], cls=r['cls'], V=r['resist_V'],
                           drift_V=r['drift_V'], n_mip=r['n_mip'], **{f'{k}_{v}': r[v][k] for v in 'xy' for k in r[v]})
                      for r in rows])
    F = []
    for (det, run, drift), g in H.groupby(['det', 'run', 'drift_V']):
        g = g[g.subrun.str.contains('resist') | g.subrun.str.contains('hv_scan')].sort_values('V')
        if g.V.nunique() < 4:
            continue
        ent = dict(det=det, run=run, drift_V=drift, n_rows=len(g), V_min=float(g.V.min()), V_max=float(g.V.max()))
        for v in 'xy':
            k = (g[f'f_sat_{v}'] < 0.3) & np.isfinite(g[f'lead_p50_{v}']) & (g.n_mip >= 100)
            if k.sum() >= 4:
                c = np.polyfit(g.V[k], np.log(g[f'lead_p50_{v}'][k]), 1)
                ent[f'slope_{v}_per10V'] = float(c[0] * 10)
                ent[f'double_V_{v}'] = float(np.log(2) / c[0]) if c[0] > 0 else None
                ent[f'ratio_{v}'] = float(g[f'lead_p50_{v}'][k].iloc[-1] / g[f'lead_p50_{v}'][k].iloc[0])
        F.append(ent)
    return H, pd.DataFrame(F)


# ---------------------------------------------------------------------------------------------------------- figures
def fig_hits_gain(H, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, v in zip(axes, 'xy'):
        for (det, run, drift), g in H.groupby(['det', 'run', 'drift_V']):
            g = g[(g.subrun.str.contains('resist') | g.subrun.str.contains('hv_scan')) & (g.n_mip >= 100)].sort_values('V')
            if g.V.nunique() < 4:
                continue
            ls = '--' if '6-23' in run else '-'
            ax.plot(g.V, g[f'lead_p50_{v}'], ls + 'o', color=COL.get(det, MUTED), ms=4, mec='white', lw=1,
                    label=f"{NAME.get(det, det)} {run.split('_')[-1][:-3]}")
        ax.set_yscale('log')
        ax.set_xlabel('resistive-strip HV [V]', color=INK)
        ax.set_title(f'{v.upper()} view: median lead-strip amplitude, MIP-like triggers (no M3)', color=INK,
                     fontsize=9, loc='left')
        style(ax)
    axes[0].set_ylabel('ADC', color=INK)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'hits_gain.{ext}'), dpi=150)
    plt.close(fig)



def fig_hv_scan(T, figdir, paper_dir):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for det, run, drift in SCANS:
        s = T[(T.det == det) & (T.run == run) & (T.drift_V == drift)].sort_values('resist_V')
        s = s[(s.subrun.str.contains('resist') | s.subrun.str.contains('hv_scan')) & (s.n_ig >= 100)]
        if not len(s):
            continue
        lab = f"{NAME[det]} {run.split('_')[-1][:-3]}"
        mk = 'o' if 'hv_scan' in run or det in ('mx17_2', 'mx17_3') else 's'
        kw = dict(color=COL[det], mec='white', mew=0.7, ms=5, lw=1.2)
        axes[0].errorbar(s.resist_V, s.eff_ig, yerr=s.err_ig, fmt='-' + mk, label=lab, **kw)
        axes[1].plot(s.resist_V, 100 * s.f_in_discharge, '-' + mk, **kw)
        axes[1].plot(s.resist_V, 100 * s.f_in_burst, ':' + mk, color=COL[det], ms=3, lw=0.8, mfc='none')
        axes[2].plot(s.resist_V, s.epoch_rate_hz / s.trig_rate_hz, '-' + mk, **kw)
    for det in DETS:
        g = T[(T.det == det) & T.cls.isin(['golden_calib', 'golden_quoted'])]
        axes[0].plot(g.resist_V, g.eff_ig, '*', ms=11, color=COL[det], mec=INK, mew=0.5, zorder=5)
    axes[0].set_ylabel('efficiency, interior [%]', color=INK)
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc='lower left')
    axes[0].set_title('Mesh-voltage scans (stars: the golden long runs)', color=INK, fontsize=10, loc='left')
    axes[1].set_ylabel('M3 tracks lost to\na discharge [%]', color=INK)
    axes[1].text(0.01, 0.92, 'solid: discharge (a chip >= 8 strips over 3000 ADC);  dotted: >50-strip burst, no discharge',
                 transform=axes[1].transAxes, fontsize=7, color=MUTED, va='top')
    axes[2].set_ylabel('discharges per trigger', color=INK)
    axes[2].set_yscale('log')
    axes[2].set_xlabel('resistive-strip HV [V]', color=INK)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'hv_scan.{ext}'), dpi=150)
    fig.savefig(os.path.join(paper_dir, 'hv_scan.pdf'))
    plt.close(fig)


def fig_deadtime(DT, figdir, paper_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for det, d in DT.items():
        if not d['rows']:
            continue
        ed = np.array(d['edges'])
        x = np.sqrt(np.maximum(ed[:-1], 1e-3) * np.minimum(ed[1:], 10.0))
        r, e = np.array(d['ratio'], float), np.array(d['err'], float)
        k = np.array(d['n']) >= 20
        axes[0].errorbar(x[k], r[k], yerr=e[k], fmt='-o', ms=4, color=COL[det], mec='white', mew=0.7, lw=1,
                         label=f"{NAME[det]}: {d['epochs']} discharges, dead time {d['dead_ms']:.1f} ± {d['dead_err_ms']:.1f} ms")
        q = np.array([np.nan if v is None else v for v in d['q_ratio']], float)
        kq = np.isfinite(q)
        axes[1].plot(x[kq], q[kq], '-o', ms=4, color=COL[det], mec='white', mew=0.7, lw=1)
    axes[0].axhline(1, color=MUTED, lw=0.8)
    axes[0].set_xscale('log'); axes[1].set_xscale('log'); axes[1].axhline(1, color=MUTED, lw=0.8)
    axes[0].set_xlabel('time since the previous discharge [s]', color=INK)
    axes[1].set_xlabel('time since the previous discharge [s]', color=INK)
    axes[0].set_ylabel('efficiency / efficiency > 1 s', color=INK)
    axes[1].set_ylabel('median charge / median > 1 s', color=INK)
    axes[0].set_ylim(0.6, 1.2); axes[1].set_ylim(0.6, 1.4)
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower right')
    axes[0].set_title('Efficiency after a discharge', color=INK, fontsize=10, loc='left')
    axes[1].set_title('Gain after a discharge', color=INK, fontsize=10, loc='left')
    for ax in axes:
        style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'spark_deadtime.{ext}'), dpi=150)
    fig.savefig(os.path.join(paper_dir, 'spark_deadtime.pdf'))
    plt.close(fig)


def fig_det4(G, P, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for (run, gas), g in G[G.drift_V >= 700].groupby(['run', 'gas']):
        mk = 'o' if 'CF4' in gas else 's'
        ax.errorbar(g.V, g.c50, yerr=g.c50_err, fmt=mk, color=COL['mx17_4'] if 'CF4' in gas else MUTED,
                    mec='white', mew=0.7, ms=6, lw=1, label=f'{run}, {gas}: telescope denominator')
    g = P[P.key.str.startswith('run66')]
    ax.plot(g.V, g.c50_reco, 'o', mfc='none', mec=COL['mx17_4'], ms=6,
            label='run66: pass-table events only (cluster sum; threshold-biased)')
    ax.set_yscale('log')
    ax.set_xlabel('det4 resistive-strip HV [V]', color=INK)
    ax.set_ylabel('det4 charge, 50th percentile [ADC]', color=INK)
    ax.set_title('det4 at SPS H4: gain vs HV', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=7.5)
    style(ax)
    ax = axes[1]
    for (run, gas), g in G[G.drift_V >= 700].groupby(['run', 'gas']):
        ax.plot(g.V, 100 * g.eff_live, '-o', color=COL['mx17_4'] if 'CF4' in gas else MUTED, mec='white', ms=5,
                label=f'{run}')
    ax.set_xlabel('det4 resistive-strip HV [V]', color=INK)
    ax.set_ylabel('both views seen, live region [%]', color=INK)
    ax.set_title('det4 detection on telescope tracks', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'det4_gain_sps.{ext}'), dpi=150)
    plt.close(fig)


def fig_tradeoff(T, figdir):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=False)
    for ax, det in zip(axes.flat, ['mx17_2', 'mx17_3', 'mx17_6', 'mx17_7']):
        for d_, run, drift in SCANS:
            if d_ != det:
                continue
            s = T[(T.det == det) & (T.run == run) & (T.drift_V == drift)].sort_values('resist_V')
            s = s[(s.subrun.str.contains('resist') | s.subrun.str.contains('hv_scan')) & (s.n_ig >= 100)]
            if not len(s):
                continue
            ax.plot(s.resist_V, s.eff_ig, '-o', color=COL[det], ms=4, label='efficiency [%]')
            ax.plot(s.resist_V, 100 - 100 * s.f_in_discharge, '--', color=COL[det], lw=1, label='100 - discharge loss [%]')
            ax2 = ax.twinx()
            ax2.plot(s.resist_V, s.ang_s68_vt_x, '-^', color=INK, ms=3.5, lw=0.8, label='angle s68 X')
            ax2.plot(s.resist_V, s.ang_s68_vt_y, '-v', color=MUTED, ms=3.5, lw=0.8, label='angle s68 Y')
            ax2.set_ylim(0.5, 2.5)
            ax2.set_ylabel('angle s68 [deg]', color=INK, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(NAME[det], color=INK, fontsize=10, loc='left')
        ax.set_xlabel('resistive-strip HV [V]', color=INK, fontsize=8)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'tradeoff.{ext}'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------- report
def table(df, cols, fmt=None):
    fmt = fmt or {}
    h = ['<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cols) + '</tr>']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = '—' if not np.isfinite(v) else (fmt.get(c, '{:.3g}').format(v))
            cells.append(f'<td>{html.escape(str(v))}</td>')
        h.append('<tr>' + ''.join(cells) + '</tr>')
    return '\n'.join(h) + '</table>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spark', required=True); ap.add_argument('--telgain', required=True)
    ap.add_argument('--spsgain', required=True); ap.add_argument('--hitsgain', default=None)
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    fig_dir = os.path.join(a.out, 'figures'); res_dir = os.path.join(a.out, 'results')
    paper_dir = os.path.join(REPO, 'papers', 'paper1', 'figures')
    for d in (fig_dir, res_dir):
        os.makedirs(d, exist_ok=True)
    sreg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
    # det4 at SPS
    G = det4_gain(a.telgain, sreg)
    G.to_csv(os.path.join(res_dir, 'det4_gain.csv'), index=False)
    P = []
    for f in sorted(glob.glob(os.path.join(a.spsgain, 'rows', '*.json'))):
        o = json.load(open(f))
        if not (o['key'].startswith('run66') or o['key'] in ('run55_flatdrift__d700', 'run56_m70V__590V',
                                                              'run56_m70V__625V', 'run63_flat__flat700')):
            continue
        z = np.load(os.path.join(a.spsgain, 'events', o['key'] + '.npz'))
        P.append(dict(key=o['key'], run=o['run'].replace('_', ''), gas=o['gas'], V=o['resist_V'],
                      c50_reco=cens_quantile(z['cqx'] + z['cqy'], z['sat_x'] | z['sat_y'], 0.5),
                      c50_fit_reco=cens_quantile(z['qx'] + z['qy'], z['sat_x'] | z['sat_y'], 0.5), n=len(z['qx'])))
    P = pd.DataFrame(P)
    fits = {}
    series = {'run66 (Ar/CF4/iso 88/10/2, one sub-run, 580-780 V)': G.key.str.startswith('run66'),
              'run55 d700 + run56 (Ar/CO2/iso 95/3/2, 550-625 V)': G.key.isin(['run55_flatdrift__d700',
                                                                                 'run56_m70V__590V',
                                                                                 'run56_m70V__625V'])}
    for name, m in series.items():
        g = G[m].sort_values('V')
        fits[name] = dict(telescope_c50=fit_slope(g.V.to_numpy(), g.c50.to_numpy(), g.c50_err.to_numpy()),
                          telescope_c25=fit_slope(g.V.to_numpy(), g.c25.to_numpy(), g.c25_err.to_numpy()),
                          ratio_c50=float(g.c50.iloc[-1] / g.c50.iloc[0]), V=[float(g.V.iloc[0]), float(g.V.iloc[-1])],
                          eff_live=[float(g.eff_live.iloc[0]), float(g.eff_live.iloc[-1])],
                          peak_med=[float(g.peak_med.iloc[0]), float(g.peak_med.iloc[-1])])
    g = P[P.key.str.startswith('run66')].sort_values('V')
    if len(g) >= 3:
        fits['run66 (Ar/CF4/iso 88/10/2, one sub-run, 580-780 V)']['reco_only_ratio_c50'] = \
            float(g.c50_reco.iloc[-1] / g.c50_reco.iloc[0])
    fig_det4(G, P, fig_dir)
    HG = hits_gain(a.hitsgain) if a.hitsgain else None
    if HG is not None:
        HG[0].to_csv(os.path.join(res_dir, 'hits_gain_rows.csv'), index=False)
        HG[1].to_csv(os.path.join(res_dir, 'hits_gain_fits.csv'), index=False)
        fig_hits_gain(HG[0], fig_dir)
    # bench
    S = spark_rows(a.spark)
    S.to_csv(os.path.join(res_dir, 'spark_rows.csv'), index=False)
    T = tradeoff(S)
    T.to_csv(os.path.join(res_dir, 'tradeoff.csv'), index=False)
    O = optimum(T)
    DT = deadtime(a.spark, S)
    fig_hv_scan(T, fig_dir, paper_dir)
    fig_deadtime(DT, fig_dir, paper_dir)
    fig_tradeoff(T, fig_dir)
    gold = T[T.cls.isin(['golden_calib', 'golden_quoted'])][['key', 'det', 'resist_V', 'eff_ig', 'f_in_discharge',
                                                            'f_in_burst', 'eff_non_spark', 'epoch_rate_hz',
                                                            'spark_per_trig', 'ang_s68_vt_x', 'ang_s68_vt_y']]
    summary = dict(det4_sps=dict(points=G.to_dict('records'), fits=fits, reco_only=P.to_dict('records')),
                   optimum=O.to_dict('records'), deadtime=DT, golden=gold.to_dict('records'),
                   hits_gain=HG[1].to_dict('records') if HG is not None else None)
    json.dump(summary, open(os.path.join(res_dir, 'summary.json'), 'w'), indent=1, default=float)
    # html
    H = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Paper I §4: operation on pass 1</title><style>',
         'body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;color:#1f1f1e;line-height:1.45}',
         'table{border-collapse:collapse;font-size:13px;margin:0.6em 0}td,th{border:1px solid #e4e3dd;padding:3px 7px}',
         'th{background:#f4f3ef}img{max-width:100%}.cap{color:#6b6a64;font-size:13px}</style></head><body>',
         '<h1>Paper I §4 — operation: HV, gain and discharges (pass 1)</h1>',
         '<p>Generated by <code>rerun/paper1_ops/make_report.py</code>. The verdict and the open questions are in '
         '<code>README.md</code>; this page carries the numbers.</p>',
         '<h2>1. Operating point per HV scan</h2>',
         table(O, ['det', 'run', 'drift_V', 'V_best', 'eff_best', 'plateau_lo', 'plateau_hi', 'V_discharge_1pct',
                   'V_discharge_5pct', 's68_x_best', 's68_y_best']),
         '<h3>Golden rows (where the quoted numbers come from)</h3>',
         table(gold, list(gold.columns)),
         '<p><img src="figures/hv_scan.png"><br><span class="cap">The scans. Top: interior_good efficiency (R = 5 mm). '
         'Middle: M3 tracks lost because their trigger is vetoed (&gt; 50 strips): solid = a real discharge, dotted = a '
         'burst without any strip near the rail. Bottom: discharge epochs per trigger.</span></p>',
         '<p><img src="figures/tradeoff.png"><br><span class="cap">The trade-off per chamber: efficiency, the discharge '
         'ceiling and the angle resolution (s68, interior_good, v(t) arm).</span></p>',
         '<h2>2. Dead time after a discharge</h2>',
         table(pd.DataFrame([dict(det=NAME[k], rows=v['rows'], discharges=v['epochs'], dead_ms=v['dead_ms'],
                                   err_ms=v['dead_err_ms'], ul95_ms=v['dead_ul95_ms']) for k, v in DT.items()]),
               ['det', 'rows', 'discharges', 'dead_ms', 'err_ms', 'ul95_ms']),
         '<p><img src="figures/spark_deadtime.png"><br><span class="cap">Efficiency and charge of the M3 tracks against '
         'the time since the previous discharge, normalised to tracks more than 1 s after one.</span></p>',
         '<h2>3. det4 gain at SPS</h2>',
         table(G, ['key', 'gas', 'V', 'drift_V', 'n', 'eff_live', 'f_sat', 'c25', 'c50', 'c50_err', 'med_seen',
                   'h_c50']),
         '<pre>' + html.escape(json.dumps(fits, indent=1, default=float)) + '</pre>',
         '<p><img src="figures/det4_gain_sps.png"><br><span class="cap">Left: the 50th percentile of det4\'s charge on '
         'every telescope track in its live region (filled), against the same quantile on the reconstructed events '
         'only (open: the zero-suppression threshold, not the gain). Right: the detection fraction on those '
         'tracks.</span></p>']
    if HG is not None:
        H += ['<h2>4. Bench gain curves without M3 (hits proxy)</h2>',
              table(HG[1], list(HG[1].columns)),
              '<p><img src="figures/hits_gain.png"><br><span class="cap">The median lead-strip amplitude of MIP-like '
              'triggers (3-30 strips per view, no M3, no fit) against HV, per scan series; dashed = the 6-23 night. '
              'Only the slope is meaningful.</span></p>']
    H += ['</body></html>']
    open(os.path.join(a.out, 'report.html'), 'w').write('\n'.join(H))
    print(json.dumps(dict(optimum=O.to_dict('records'), deadtime={NAME[k]: (v['dead_ms'], v['dead_ul95_ms'])
                                                                    for k, v in DT.items()}, fits=fits),
                     indent=1, default=float))


if __name__ == '__main__':
    main()
