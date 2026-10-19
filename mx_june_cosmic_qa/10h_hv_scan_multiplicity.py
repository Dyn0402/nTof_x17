#!/usr/bin/env python3
"""
10h_hv_scan_multiplicity.py -- strips over threshold vs mesh voltage, with the
track angle taken out.

The raw number of strips over 5 sigma rises steeply with the mesh voltage (10e:
5 -> 12 per plane over 100 V). Two very different things can do that, and they
have opposite consequences:

  (a) the charge really only reaches threshold on part of the track at low
      gain, so the faint slices -- a strip crossed by a short piece of track,
      carrying a handful of primary ionisations -- drop out. Then the low-gain
      cluster is a BROKEN version of the high-gain one, and everything built on
      cluster shape (position, angle, dE/dx) is biased at low gain.

  (b) the cluster shape is fixed and only its overall scale moves, so the
      transverse tails -- the sharing tail, the diffused edges -- cross the
      threshold one strip at a time. Then the low-gain cluster is the same
      cluster seen through a higher relative threshold, nothing is missing from
      the middle, and the count is a threshold artefact rather than a physics
      change.

Telling them apart needs the amplitudes of the strips that did not fire, which
is what 10g recorded. Four independent handles here:

  1. **Angle normalisation.** A steeper track lights more strips at any gain,
     and the mesh ladder's sub-runs are not guaranteed to have identical angle
     distributions. The M3 reference gives the geometric footprint
     ``w_geo = gap * |tan(theta_ref)| / pitch`` (in strips) per event -- from
     the *telescope*, so it cannot move with the detector's own gain. Then
     ``n_lit = a(V) + b(V) * w_geo``: ``b`` is the fraction of the crossed
     strips that actually fire (hypothesis (a) lives here) and ``a`` is the
     footprint at normal incidence (hypothesis (b) lives here). Medians in
     narrow ``w_geo`` bands say the same thing without a fit.

  2. **Holes.** Dark strips strictly inside the lit span. Needs no angle
     normalisation at all: a hole is an internal property of one cluster. This
     is the direct signature of (a).

  3. **The per-strip profile.** Median significance at each offset from the
     peak strip, with the 5 sigma line drawn on it. If the profile only
     rescales, the count is set by where that line cuts a fixed shape.

  4. **The scaling prediction.** Take one reference voltage, scale every
     strip's signal by the measured charge ratio G(V)/G(V0), re-apply 5 sigma,
     and count. If the prediction tracks the measurement, (b) is the whole
     story and no charge is going missing. The signal is scaled and the noise
     floor is not: ``s`` is a max over 32 samples, so an empty strip sits at
     ~2 sigma, and scaling that too would invent hits. The floor is measured
     per sub-run from the outer columns (|k| = 9, 10) of the same matrix.

Gain ratios come from 10e's ``charge_angle_vs_hv.csv`` (the model-free window
sum, ``q_win``); its mtime is stamped into the meta so a stale reduction cannot
sit under this one unnoticed.

    ../.venv/bin/python 10h_hv_scan_multiplicity.py [--slides]
Output: <Analysis>/<run>/hv_scan/mx17_3/
        strip_multiplicity_vs_hv.csv   .meta.json
        strip_profile_vs_hv.csv
        multiplicity_vs_hv.png  strip_profile_vs_hv.png
        multiplicity_holes.png  multiplicity_prediction.png
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE, os.path.join(REPO, 'cosmic_bench_analysis')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reference tangents, the sub-run list and the weighted log-linear fit are
# 10e's -- imported, not copied, so the two reductions cannot drift apart
E = _load('hv_charge_angle', os.path.join(HERE, '10e_hv_scan_charge_angle.py'))

RUN, DET, BASE, ANALYSIS = E.RUN, E.DET, E.BASE, E.ANALYSIS
GAP_MM, PITCH_MM, SAT_ADC = E.GAP_MM, E.PITCH_MM, E.SAT_ADC
OUT_DIR = os.path.join(ANALYSIS, RUN, 'hv_scan', DET)
MATRIX = os.path.join(OUT_DIR, 'strip_matrix.parquet')
RAW = os.path.join(OUT_DIR, 'occupancy_raw.parquet')
LADDER = os.path.join(OUT_DIR, 'charge_angle_vs_hv.csv')
SLIDE_DIR = os.path.join(REPO, 'mpgd26', 'slides', 'assets', 'img')

SIGMA = 5.0
HALF = 10
FLOOR_K = (9, 10)          # |offset| used to measure the noise-max floor
V_REF_PRED = 465           # scaling reference: high enough to see the tails,
                           # low enough that little is clipped
W_BANDS = [(0.0, 1.0), (1.0, 2.5), (2.5, 5.0), (5.0, 9.0), (9.0, 20.0)]
NBOOT = 400
RNG = np.random.default_rng(20260828)


# ------------------------------------------------------------------ loading
def load_matrix():
    if not os.path.exists(MATRIX):
        sys.exit(f'[10h] no {MATRIX} -- run 10g first')
    return pd.read_parquet(MATRIX)


def reco_event_ids():
    """The 10d event set: M3-golden, fiducial, spark-free, reconstructed.

    The matrix (10g) is over every fiducial ray, sparks included. The charge
    ladder is quoted on the spark-free set, so this pass has to use the same
    one or the two are not comparable -- above 500 V the difference is half the
    events."""
    keep = {}
    for sub, volt, scan in E.subruns():
        p = os.path.join(ANALYSIS, RUN, sub, DET, 'wft', 'events_hvscan.parquet')
        if not os.path.exists(p):
            print(f'[10h] missing {p} -- run 10d first')
            continue
        keep[sub] = set(pd.read_parquet(p, columns=['event_id'])
                        .event_id.astype(int))
    if not keep:
        sys.exit('[10h] no 10d tables found')
    return keep


def gain_ratios():
    """G(V)/G(V_REF_PRED) per (view, scan, hv) from the model-free window sum."""
    if not os.path.exists(LADDER):
        sys.exit(f'[10h] no {LADDER} -- run 10e first')
    t = pd.read_csv(LADDER)
    col = 'q_win' if 'q_win' in t else 'q_sum'
    ref = {}
    for view in ('x', 'y'):
        s = t[(t.view == view) & (t.hv == V_REF_PRED)]
        if len(s):
            ref[view] = float(s[col].iloc[0])
    g = {}
    for _, r in t.iterrows():
        if r.view in ref and np.isfinite(r[col]):
            g[(r.view, int(r.hv))] = float(r[col]) / ref[r.view]
    meta = dict(ladder_csv=LADDER, ladder_col=col,
                ladder_mtime=time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(LADDER))),
                ladder_md5=hashlib.md5(open(LADDER, 'rb').read()).hexdigest(),
                v_ref_pred=V_REF_PRED, ref_q=ref)
    return g, t, meta


def attach_reference(df):
    """M3 reference tangent per (subrun, event) -> geometric footprint."""
    out = []
    for sub, g in df.groupby('subrun', sort=False):
        ref = E.reference_tangents(sub)
        out.append(g.merge(ref, on='event_id', how='left'))
    d = pd.concat(out, ignore_index=True)
    tan = np.where(d.view.to_numpy() == 'x', d.ref_tan_x, d.ref_tan_y)
    d['ref_tan'] = tan
    d['w_geo'] = GAP_MM * np.abs(tan) / PITCH_MM
    return d


# ------------------------------------------------------------- derived cols
def scols():
    return [f's{k:+d}' for k in range(-HALF, HALF + 1)]


def add_cluster_shape(d):
    """n_lit, lit span, internal holes and window-edge contact, per event."""
    S = d[scols()].to_numpy(float)
    lit = np.isfinite(S) & (S >= SIGMA)
    n_lit = lit.sum(axis=1)
    idx = np.arange(S.shape[1])[None, :]
    big = np.where(lit, idx, 10 ** 4)
    small = np.where(lit, idx, -1)
    lo, hi = big.min(axis=1), small.max(axis=1)
    span = np.where(n_lit > 0, hi - lo + 1, 0)
    d = d.copy()
    d['n_lit'] = n_lit
    d['span_strip'] = span
    d['n_hole'] = np.maximum(span - n_lit, 0)
    d['hole_frac'] = np.where(span > 0, d.n_hole / span, np.nan)
    d['edge_lit'] = lit[:, 0] | lit[:, -1]

    # longest contiguous lit run, and the strips detached from it. A cluster
    # broken by missing charge and a cluster whose far tail speckles above
    # threshold both produce "holes"; only the first shortens the core run.
    run = np.zeros(len(S))
    best = np.zeros(len(S))
    for j in range(S.shape[1]):
        run = np.where(lit[:, j], run + 1, 0.0)
        best = np.maximum(best, run)
    d['n_run_max'] = best
    d['n_detached'] = n_lit - best

    # what the dark strips inside the span actually contain. A hole at
    # s ~ 2 sigma is empty; a hole at s ~ 4 sigma is a strip that just missed.
    inside = (idx > lo[:, None]) & (idx < hi[:, None])
    hole_mask = inside & ~lit
    with np.errstate(invalid='ignore'):
        Sh = np.where(hole_mask, S, np.nan)
        d['hole_sig_med'] = np.nanmedian(Sh, axis=1)
        d['hole_sig_max'] = np.nanmax(np.where(np.isnan(Sh), -np.inf, Sh),
                                      axis=1)
    d.loc[~np.isfinite(d.hole_sig_max), 'hole_sig_max'] = np.nan
    # the noise-max floor: the outer columns, which are empty for all but the
    # widest tracks
    fcols = [f's{s}{k:d}' for k in FLOOR_K for s in ('-', '+')]
    d['s_outer'] = np.nanmedian(d[fcols].to_numpy(float), axis=1)
    return d


def predict_counts(d, ratios, floors):
    """n_lit predicted at every voltage from the V_REF_PRED events.

    The reference events are scaled by G(V)/G(V0) -- signal only, the noise-max
    floor held fixed -- and re-counted at 5 sigma. Predictions are made INSIDE
    each w_geo band and recombined with the target voltage's own band
    populations, so a drift in the angle mix between sub-runs cannot leak into
    the prediction."""
    S0 = {}
    for view in ('x', 'y'):
        m = d[(d.view == view) & (d.hv == V_REF_PRED)]
        if len(m):
            S0[view] = (m[scols()].to_numpy(float), m.w_geo.to_numpy(float),
                        float(floors.get((m.subrun.iloc[0], view), 2.0)))
    rows = []
    for (view, scan, hv), m in d.groupby(['view', 'scan', 'hv']):
        if view not in S0:
            continue
        f = ratios.get((view, int(hv)))
        if f is None or not np.isfinite(f):
            continue
        S, w0, fl = S0[view]
        pred_sig = fl + f * (S - fl)
        npred_ev = (np.isfinite(S) & (pred_sig >= SIGMA)).sum(axis=1)
        # band-matched recombination
        num, den = 0.0, 0.0
        for lo, hi in W_BANDS:
            tgt = ((m.w_geo >= lo) & (m.w_geo < hi)).sum()
            sel = (w0 >= lo) & (w0 < hi)
            if tgt and sel.sum() >= 20:
                num += tgt * float(np.median(npred_ev[sel]))
                den += tgt
        rows.append(dict(view=view, scan=scan, hv=int(hv),
                         gain_ratio=f,
                         n_lit_pred=float(np.median(npred_ev)),
                         n_lit_pred_bandmatched=num / den if den else np.nan,
                         n_pred_ev=int(len(npred_ev))))
    return pd.DataFrame(rows)


def angle_fit(w, n):
    """n_lit = a + b * w_geo, OLS with heteroscedasticity-robust errors."""
    w, n = np.asarray(w, float), np.asarray(n, float)
    ok = np.isfinite(w) & np.isfinite(n)
    w, n = w[ok], n[ok]
    if len(w) < 50 or np.ptp(w) < 0.5:
        return dict(a=np.nan, a_err=np.nan, b=np.nan, b_err=np.nan, n=len(w))
    X = np.column_stack([np.ones_like(w), w])
    xtx = np.linalg.inv(X.T @ X)
    beta = xtx @ (X.T @ n)
    r = n - X @ beta
    cov = xtx @ (X.T @ (X * (r ** 2)[:, None])) @ xtx        # HC0
    return dict(a=float(beta[0]), a_err=float(np.sqrt(cov[0, 0])),
                b=float(beta[1]), b_err=float(np.sqrt(cov[1, 1])), n=len(w))


def ladder(d):
    """One row per (scan, hv, view)."""
    rows = []
    for (scan, hv, view), m in d.groupby(['scan', 'hv', 'view'], sort=True):
        S = m[scols()].to_numpy(float)
        lit = np.isfinite(S) & (S >= SIGMA)
        r = dict(scan=scan, hv=int(hv), view=view, subrun=m.subrun.iloc[0],
                 n_ev=int(len(m)),
                 n_lit_med=float(np.median(m.n_lit)),
                 n_lit_mean=float(np.mean(m.n_lit)),
                 n_lit_p10=float(np.percentile(m.n_lit, 10)),
                 n_lit_p90=float(np.percentile(m.n_lit, 90)),
                 span_strip_med=float(np.median(m.span_strip)),
                 n_run_max_med=float(np.median(m.n_run_max)),
                 n_detached_mean=float(np.mean(m.n_detached)),
                 frac_detached=float(np.mean(m.n_detached > 0)),
                 hole_sig_med=float(np.nanmedian(m.hole_sig_med)),
                 hole_near_thr=float(np.nanmean(m.hole_sig_med >= 3.0)),
                 q_frac_thr=float(np.nanmedian(m.q_frac_thr))
                 if 'q_frac_thr' in m else np.nan,
                 hole_frac=float(np.nanmean(m.hole_frac)),
                 frac_with_hole=float(np.mean(m.n_hole > 0)),
                 n_hole_mean=float(np.mean(m.n_hole)),
                 frac_edge_lit=float(np.mean(m.edge_lit)),
                 s_outer_med=float(np.nanmedian(m.s_outer)),
                 peak_amp_med=float(np.median(m.peak_amp)),
                 frac_sat=float(np.mean(m.peak_amp >= SAT_ADC)),
                 noise_med=float(np.median(m.peak_noise)),
                 w_geo_med=float(np.nanmedian(m.w_geo)),
                 w_geo_p90=float(np.nanquantile(m.w_geo, 0.90)))
        r.update({f'{k}_fit': v for k, v in angle_fit(m.w_geo, m.n_lit).items()})
        # angle-matched medians. ``n_lit_meas_bandmatched`` recombines them
        # exactly as predict_counts() recombines the prediction -- a median of
        # a discrete count is not the weighted mean of its band medians, so
        # comparing the prediction against the plain median would show a fixed
        # ~0.3 strip offset that is pure estimator mismatch.
        num, den = 0.0, 0.0
        for lo, hi in W_BANDS:
            sel = m[(m.w_geo >= lo) & (m.w_geo < hi)]
            tag = f'nlit_w{lo:g}_{hi:g}'
            r[tag] = float(np.median(sel.n_lit)) if len(sel) >= 20 else np.nan
            r[tag + '_n'] = int(len(sel))
            if len(sel) >= 20:
                num += len(sel) * float(np.median(sel.n_lit))
                den += len(sel)
            r[f'hole_w{lo:g}_{hi:g}'] = (float(np.nanmean(sel.hole_frac))
                                         if len(sel) >= 20 else np.nan)
        r['n_lit_meas_bandmatched'] = num / den if den else np.nan
        # per-strip lit probability and median significance, folded on |k|
        for k in range(0, HALF + 1):
            cols = [f's{k:+d}'] if k == 0 else [f's-{k}', f's+{k}']
            v = m[cols].to_numpy(float).ravel()
            v = v[np.isfinite(v)]
            r[f'plit_k{k}'] = float(np.mean(v >= SIGMA)) if len(v) else np.nan
            r[f'sig_k{k}'] = float(np.median(v)) if len(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values(['view', 'hv']).reset_index(drop=True)


def attach_charge_fraction(d):
    """q_5s / q_win from 10f: the fraction of the collected charge that is
    over 5 sigma somewhere.

    This is the most direct form of the question "does low gain lose the small
    deposits": the denominator is threshold-free (every sample of every strip
    in the window, noise included but zero-mean), the numerator is what a
    threshold-based chain -- the DAQ, the hits, any clustering -- can ever
    see."""
    if not os.path.exists(RAW):
        d['q_frac_thr'] = np.nan
        return d
    raw = pd.read_parquet(RAW)
    key = ['subrun', 'view', 'event_id']
    d = d.merge(raw[key + ['q_5s', 'q_win']], on=key, how='left')
    d['q_frac_thr'] = np.where(d.q_win > 0, d.q_5s / d.q_win, np.nan)
    return d


def check_10f(d):
    """n_lit here must equal 10f's n_strip_5s event by event: same window, same
    threshold, two separate read passes."""
    if not os.path.exists(RAW):
        return None
    raw = pd.read_parquet(RAW)
    key = ['subrun', 'view', 'event_id']
    j = d[key + ['n_lit']].merge(raw[key + ['n_strip_5s']], on=key)
    if not len(j):
        return None
    bad = int((j.n_lit != j.n_strip_5s).sum())
    return dict(n_compared=int(len(j)), n_disagree=bad,
                max_abs_diff=int(np.abs(j.n_lit - j.n_strip_5s).max()))


# ------------------------------------------------------------------ figures
COL = {'x': 'tab:blue', 'y': 'tab:red'}


def _band(ax, t):
    """grey out the range where the peak sample is >90 % clipped."""
    s = t[t.frac_sat >= 0.90]
    if len(s):
        ax.axvspan(float(s.hv.min()) - 5, float(t.hv.max()) + 5,
                   color='0.85', zorder=0)


def fig_multiplicity(t, path, slides=False):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))
    a, b, c, dd = ax.ravel()

    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(a, s)
        a.fill_between(s.hv, s.n_lit_p10, s.n_lit_p90, color=COL[view],
                       alpha=0.15, lw=0)
        a.plot(s.hv, s.n_lit_med, 'o-', color=COL[view], label=f'{view} median')
    a.set_xlabel('mesh voltage [V]')
    a.set_ylabel('strips over 5$\\sigma$ per plane')
    a.set_title('Raw multiplicity (band: p10-p90)')
    a.legend(fontsize=8)
    a.grid(alpha=.3)

    # angle-matched
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(b, s)
        for (lo, hi), ls in zip(W_BANDS, ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]):
            col = f'nlit_w{lo:g}_{hi:g}'
            if col in s and s[col].notna().sum() > 3:
                b.plot(s.hv, s[col], linestyle=ls, marker='o', ms=3,
                       color=COL[view], lw=1.4,
                       label=f'{view}  $w_{{geo}}$ {lo:g}-{hi:g}')
    b.set_xlabel('mesh voltage [V]')
    b.set_ylabel('strips over 5$\\sigma$ (median)')
    b.set_title('Angle-matched: same footprint, every voltage')
    b.legend(fontsize=6.5, ncol=2)
    b.grid(alpha=.3)

    # a(V), b(V)
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(c, s)
        c.errorbar(s.hv, s.a_fit, yerr=s.a_err_fit, fmt='o-', color=COL[view],
                   ms=4, lw=1.2, label=f'{view}: $a$ (normal incidence)')
    c.set_xlabel('mesh voltage [V]')
    c.set_ylabel('$a$  [strips]')
    c.set_title('$n_{lit} = a(V) + b(V)\\,w_{geo}$: the offset')
    c.legend(fontsize=8)
    c.grid(alpha=.3)

    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(dd, s)
        dd.errorbar(s.hv, s.b_fit, yerr=s.b_err_fit, fmt='s-', color=COL[view],
                    ms=4, lw=1.2, label=f'{view}: $b$ (per crossed strip)')
    dd.axhline(1.0, color='k', ls=':', lw=1)
    dd.text(t.hv.min() + 2, 1.02, 'every crossed strip fires', fontsize=7)
    dd.set_xlabel('mesh voltage [V]')
    dd.set_ylabel('$b$  [strips per crossed strip]')
    dd.set_title('the slope: is the track followed?')
    dd.legend(fontsize=8)
    dd.grid(alpha=.3)

    fig.suptitle('det3 6-27 mesh ladder: strips over threshold, angle removed',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    if slides:
        fig.savefig(os.path.join(SLIDE_DIR, 'hv_multiplicity.png'), dpi=140)
    plt.close(fig)


def fig_profile(t, path, slides=False):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    volts = sorted(t.hv.unique())
    pick = [volts[0], volts[len(volts) // 4], volts[len(volts) // 2],
            volts[3 * len(volts) // 4], volts[-1]]
    cm = plt.cm.viridis(np.linspace(0, .92, len(pick)))

    for i, v in enumerate(pick):
        s = t[(t.view == 'x') & (t.hv == v)]
        if not len(s):
            continue
        y = [float(s[f'sig_k{k}'].iloc[0]) for k in range(HALF + 1)]
        ax[0].plot(range(HALF + 1), y, 'o-', color=cm[i], ms=4, label=f'{v} V')
    ax[0].axhline(SIGMA, color='k', ls='--', lw=1.2)
    ax[0].text(6.2, SIGMA * 1.12, '5$\\sigma$', fontsize=8)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('|offset| from peak strip')
    ax[0].set_ylabel('median amplitude / $\\sigma_{noise}$')
    ax[0].set_title('x plane: the profile only rescales')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3, which='both')

    for i, v in enumerate(pick):
        s = t[(t.view == 'x') & (t.hv == v)]
        if not len(s):
            continue
        y = [float(s[f'plit_k{k}'].iloc[0]) for k in range(HALF + 1)]
        ax[1].plot(range(HALF + 1), y, 'o-', color=cm[i], ms=4, label=f'{v} V')
    ax[1].set_xlabel('|offset| from peak strip')
    ax[1].set_ylabel('fraction of events with strip over 5$\\sigma$')
    ax[1].set_title('the footprint fills in from the middle out')
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3)

    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(ax[2], s)
        for k, ls in ((1, '-'), (2, '--'), (3, '-.'), (4, ':')):
            ax[2].plot(s.hv, s[f'plit_k{k}'], linestyle=ls, color=COL[view],
                       lw=1.3, label=f'{view}  |k| = {k}')
    ax[2].axhline(0.5, color='k', ls=':', lw=1)
    ax[2].set_xlabel('mesh voltage [V]')
    ax[2].set_ylabel('$P$(strip over 5$\\sigma$)')
    ax[2].set_title('per-offset turn-on curves')
    ax[2].legend(fontsize=6.5, ncol=2)
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if slides:
        fig.savefig(os.path.join(SLIDE_DIR, 'hv_strip_profile.png'), dpi=140)
    plt.close(fig)


def fig_holes(t, path, slides=False):
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.3))
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(ax[0], s)
        ax[0].plot(s.hv, 100 * s.hole_frac, 'o-', color=COL[view],
                   label=f'{view}')
        _band(ax[1], s)
        ax[1].plot(s.hv, 100 * s.frac_with_hole, 's-', color=COL[view],
                   label=f'{view}')
    ax[0].set_xlabel('mesh voltage [V]')
    ax[0].set_ylabel('dark strips inside the lit span [%]')
    ax[0].set_title('Holes: is the cluster broken at low gain?')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].set_xlabel('mesh voltage [V]')
    ax[1].set_ylabel('events with >=1 internal hole [%]')
    ax[1].set_title('...and how often')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(ax[2], s)
        for (lo, hi), ls in zip(W_BANDS, ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]):
            col = f'hole_w{lo:g}_{hi:g}'
            if col in s and s[col].notna().sum() > 3:
                ax[2].plot(s.hv, 100 * s[col], linestyle=ls, marker='o',
                           ms=3, color=COL[view], lw=1.3,
                           label=f'{view}  $w_{{geo}}$ {lo:g}-{hi:g}')
    ax[2].set_xlabel('mesh voltage [V]')
    ax[2].set_ylabel('dark strips inside lit span [%]')
    ax[2].set_title('holes by footprint width')
    ax[2].legend(fontsize=6.5, ncol=2)
    ax[2].grid(alpha=.3)

    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(ax[3], s)
        ax[3].plot(s.hv, 100 * s.q_frac_thr_norm, 'o-', color=COL[view],
                   label=f'{view}')
    ax[3].axhline(100, color='k', ls=':', lw=1)
    ax[3].set_xlabel('mesh voltage [V]')
    ax[3].set_ylabel('charge over 5$\\sigma$ [% of high-gain plateau]')
    ax[3].set_title('How much of the deposit is visible at all')
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if slides:
        fig.savefig(os.path.join(SLIDE_DIR, 'hv_multiplicity_holes.png'), dpi=140)
    plt.close(fig)


def fig_prediction(t, pred, path, slides=False):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for i, view in enumerate(('x', 'y')):
        s = t[t.view == view].sort_values('hv')
        _band(ax[i], s)
        ax[i].plot(s.hv, s.n_lit_meas_bandmatched, 'o-', color=COL[view], ms=5,
                   label='measured')
        ax[i].plot(s.hv, s.n_lit_pred_bandmatched, '^--', color='k', ms=5,
                   lw=1.2, label=f'threshold scaling from {V_REF_PRED} V')
        ax[i].axvline(V_REF_PRED, color='0.4', ls=':', lw=1)
        ax[i].set_xlabel('mesh voltage [V]')
        ax[i].set_ylabel('strips over 5$\\sigma$ (band-matched)')
        ax[i].set_title(f'{view} plane')
        ax[i].legend(fontsize=8)
        ax[i].grid(alpha=.3)
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        _band(ax[2], s)
        ax[2].plot(s.hv, s.pred_minus_meas, 'o-', color=COL[view], label=view)
    ax[2].axhline(0, color='k', lw=1)
    ax[2].axvline(V_REF_PRED, color='0.4', ls=':', lw=1)
    ax[2].set_xlabel('mesh voltage [V]')
    ax[2].set_ylabel('predicted $-$ measured [strips]')
    ax[2].set_title('what rescaling does not explain')
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    fig.suptitle('Does a fixed cluster shape + a moving threshold explain the '
                 'strip count?', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=140)
    if slides:
        fig.savefig(os.path.join(SLIDE_DIR, 'hv_multiplicity_prediction.png'),
                    dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides', action='store_true',
                    help='also drop the figures into mpgd26/slides/assets/img')
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    d = load_matrix()
    keep = reco_event_ids()
    n0 = len(d)
    d = d[[eid in keep.get(sub, set())
           for sub, eid in zip(d.subrun, d.event_id.astype(int))]].copy()
    print(f'[10h] spark-free reconstructed set: {len(d):,} of {n0:,} '
          f'event-planes ({100 * len(d) / n0:.1f} %)')

    d = add_cluster_shape(d)
    d = attach_charge_fraction(d)
    d = attach_reference(d)
    agree = check_10f(d)
    ratios, lad10e, gmeta = gain_ratios()

    t = ladder(d)
    floors = {(r.subrun, r.view): r.s_outer_med for r in t.itertuples()}
    pred = predict_counts(d, ratios, floors)
    t = t.merge(pred, on=['view', 'scan', 'hv'], how='left')
    t['pred_minus_meas'] = t.n_lit_pred_bandmatched - t.n_lit_meas_bandmatched

    # q_5s / q_win plateaus ABOVE 1 (1.03 in x, 1.07 in y). The numerator is a
    # sum of over-threshold cells, all positive; the denominator is the whole
    # window, which also contains the shaped pulse's undershoot and whatever
    # the 64-channel common-mode median took off a wide signal. Both of those
    # are signal-proportional and negative, so q_win runs low by a few per cent
    # once the signal is large -- harmless for the CHARGE SLOPE (a proportional
    # bias cancels in d ln Q / dV, which is why q_win and the deconvolved q_sum
    # agreed to 2 %), but it means the ratio is only a detected FRACTION after
    # it has been referred to its own high-gain plateau.
    t['q_frac_thr_norm'] = np.nan
    for view in ('x', 'y'):
        m = t.view == view
        pl = t.loc[m & t.hv.between(470, 490), 'q_frac_thr'].mean()
        t.loc[m, 'q_frac_thr_norm'] = t.loc[m, 'q_frac_thr'] / pl
        print(f'[10h] {view}: q_5s/q_win plateau (470-490 V) = {pl:.4f}')

    t.to_csv(os.path.join(OUT_DIR, 'strip_multiplicity_vs_hv.csv'), index=False)
    prof = t[['view', 'scan', 'hv']
             + [f'sig_k{k}' for k in range(HALF + 1)]
             + [f'plit_k{k}' for k in range(HALF + 1)]]
    prof.to_csv(os.path.join(OUT_DIR, 'strip_profile_vs_hv.csv'), index=False)

    fig_multiplicity(t, os.path.join(OUT_DIR, 'multiplicity_vs_hv.png'), a.slides)
    fig_profile(t, os.path.join(OUT_DIR, 'strip_profile_vs_hv.png'), a.slides)
    fig_holes(t, os.path.join(OUT_DIR, 'multiplicity_holes.png'), a.slides)
    fig_prediction(t, pred, os.path.join(OUT_DIR, 'multiplicity_prediction.png'),
                   a.slides)

    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      cwd=REPO, text=True).strip()
    except Exception:
        sha = 'unknown'
    meta = dict(script=os.path.basename(__file__), git=sha,
                written=time.strftime('%Y-%m-%d %H:%M:%S'),
                matrix=MATRIX, sigma=SIGMA, half=HALF,
                floor_offsets=list(FLOOR_K), w_bands=W_BANDS,
                gap_mm=GAP_MM, pitch_mm=PITCH_MM, sat_adc=SAT_ADC,
                n_event_planes=int(len(d)), n_event_planes_all=int(n0),
                cross_check_10f=agree, **gmeta)
    with open(os.path.join(OUT_DIR, 'strip_multiplicity_vs_hv.meta.json'),
              'w') as f:
        json.dump(meta, f, indent=2, default=float)

    print(f'[10h] 10f cross-check: {agree}')
    cols = ['hv', 'view', 'n_lit_med', 'a_fit', 'b_fit', 'q_frac_thr_norm',
            'hole_frac', 'frac_with_hole', 'hole_sig_med', 'n_detached_mean',
            'n_lit_meas_bandmatched', 'n_lit_pred_bandmatched',
            'pred_minus_meas', 'frac_edge_lit', 'w_geo_med']
    print(t[cols].to_string(index=False,
                            float_format=lambda v: f'{v:8.3f}'))
    print(f'[10h] wrote {OUT_DIR}')


if __name__ == '__main__':
    main()
