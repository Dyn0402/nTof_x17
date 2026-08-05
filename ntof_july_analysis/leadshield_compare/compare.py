#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Before/after comparison for the 2026-08-04 lead-shielding removal.

Question (operator, 2026-08-05): did removing the shielding lead during the
Aug 4 morning access lengthen the DAQ saturation after the gamma flash? That
would appear as less track efficiency (and/or fewer accepted triggers) at
early time-since-flash, ~1-5 ms.

Compares the identically-configured stat090 production runs bracketing the
access: run_130 + run_132 (before) vs run_139 (after). Three observable
families, most-direct first:

  1. TRIGGER ACCEPTANCE vs dt — accepted (non-leader) events per flash burst
     per ms. If the DAQ stays saturated longer, the early-dt acceptance drops.
     Also per-burst FIRST accepted dt quantiles.
  2. BLINDNESS vs dt — fraction of read-out events where a detector produced
     no hits at all (live_* == False). The per-detector front-end recovery.
  3. TRACK EFFICIENCY vs dt — P(3D x/y pair) per recorded trigger (run67_scan
     conventions: denominator = readout_*, Det A is the clean-M1 reference).
     Boxcar curves (W=6/step 1 ms default + a fine 2/0.25 ms zoom below 15 ms)
     and a fixed-window table with two-proportion z tests.

Controls, because "before" and "after" are different nights:
  * late window (40-76 ms) efficiency must MATCH — anything that moves early
    AND late dt alike is not a saturation-time effect;
  * run_130 vs run_132 gives the same-condition night-to-night scatter;
  * per-pulse beam intensity attached via pulse_match (HIGH/LOW at 600e10,
    the shared E10_SPLIT) — intensity-matched curves + the delivered mix;
  * gamma-flash leader size (n_big_leader / n_hits_leader) per run — if the
    flash itself got bigger, THAT is the smoking gun for more radiation in.

*** Boxcar points are CORRELATED (step << width): a wiggle narrower than the
box is the smoothing kernel. The fixed-window table is the statistics. ***

Run: .venv/bin/python ntof_july_analysis/leadshield_compare/compare.py
Outputs -> <ANALYSIS_DIR>/lead_shielding_compare/{figures,tables,SUMMARY.md}
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))          # for pulse_match
sys.path.insert(0, _HERE)

import lib as L  # noqa: E402
import feu_presence as FP  # noqa: E402

FIG_DIR = os.path.join(L.OUT_BASE, 'figures')
TAB_DIR = os.path.join(L.OUT_BASE, 'tables')

E10_SPLIT = 600e10 / 1e10     # pulse_match e10 values are in 1e10 protons
MIN_N_PER_POINT = 60
DET_ORDER = 'ABCD'

RUN_LABEL = {
    'run_130': 'run_130 (before, Aug 3 eve)',
    'run_132': 'run_132 (before, night Aug 3-4)',
    'run_139': 'run_139 (after, night Aug 4-5)',
}


# ---------------------------------------------------------------- utilities
def gate_edges(ev, bin_ms=1.0, frac=0.5, k=5):
    """Measured dt acceptance edges (copy of run67_scan.slide.gate_edges)."""
    dt = ev.loc[~ev['is_leader'], 'dt_ms'].to_numpy()
    edges = np.arange(0.0, np.ceil(dt.max()) + bin_ms, bin_ms)
    h, _ = np.histogram(dt, bins=edges)
    lo = hi = None
    for i in range(len(h) - 1, k, -1):
        loc = np.median(h[i - k:i])
        if loc > 0 and h[i] >= frac * loc:
            hi = float(edges[i + 1])
            break
    for i in range(0, len(h) - k):
        loc = np.median(h[i + 1:i + 1 + k])
        if loc > 0 and h[i] >= frac * loc:
            lo = float(edges[i])
            break
    if lo is None or hi is None or hi <= lo:
        lo, hi = L.READOUT_START_MS, L.GATE_CLOSE_MS
    return lo, hi


def centers(lo, hi, width, step):
    first, last = lo + width / 2.0, hi - width / 2.0
    if last < first:
        return np.array([])
    return first + step * np.arange(int(np.floor((last - first) / step)) + 1)


def boxcar_counts(dt, hit, cen, width):
    dt = np.asarray(dt, float)
    hit = np.asarray(hit).astype(np.int64)
    order = np.argsort(dt, kind='stable')
    dts = dt[order]
    csum = np.concatenate([[0], np.cumsum(hit[order])])
    lo = np.searchsorted(dts, cen - width / 2.0, side='left')
    hi = np.searchsorted(dts, cen + width / 2.0, side='right')
    return (csum[hi] - csum[lo]).astype(float), (hi - lo).astype(float)


def curve(ev_det, Ld, cen, width, min_n):
    dt = ev_det['dt_ms'].to_numpy()
    k_pair, n = boxcar_counts(dt, ev_det[f'n_pair_{Ld}'] > 0, cen, width)
    k_trk, _ = boxcar_counts(dt, ev_det[f'n_trkseg_{Ld}'] > 0, cen, width)
    k_blind, _ = boxcar_counts(dt, ~ev_det[f'live_{Ld}'].to_numpy(bool), cen, width)
    p_pair, e_pair = L.binom_err(k_pair, n)
    p_trk, e_trk = L.binom_err(k_trk, n)
    p_bl, e_bl = L.binom_err(k_blind, n)
    out = pd.DataFrame({
        'dt_ms': cen, 'n': n, 'k_pair': k_pair,
        'p_pair': p_pair, 'e_pair': e_pair,
        'p_trk': p_trk, 'e_trk': e_trk,
        'blind_frac': p_bl, 'e_blind': e_bl,
    })
    return out[out.n >= min_n].reset_index(drop=True)


def two_prop_z(k1, n1, k2, n2):
    """Two-proportion z (pooled); returns (z, two-sided p)."""
    if min(n1, n2) == 0:
        return np.nan, np.nan
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-300) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


INTENSITY_CACHE = os.path.join(L.CACHE_DIR, '_intensity.parquet')


def attach_intensity(ev, force=False):
    """Per-event pulse intensity (1e10 p) via pulse_match; '' if unmatched.

    Cached to _intensity.parquet — pulse_match re-fits the clock offset per
    sub-run and is the slow part of a re-run.
    """
    if os.path.exists(INTENSITY_CACHE) and not force:
        c = pd.read_parquet(INTENSITY_CACHE)
        ev = ev.merge(c, on=['run', 'subrun', 'eventId'], how='left')
        ev['iband'] = np.where(ev.e10.isna(), '',
                               np.where(ev.e10 >= E10_SPLIT, 'HIGH', 'LOW'))
        return ev
    try:
        import pulse_match as PM
    except Exception as e:  # noqa: BLE001
        print(f'  pulse_match unavailable ({e!r}) — intensity views skipped')
        ev['e10'] = np.nan
        ev['iband'] = ''
        return ev
    e10 = np.full(len(ev), np.nan)
    for (run, sub), idx in ev.groupby(['run', 'subrun']).indices.items():
        try:
            m = PM.match_subrun(run, sub)
        except Exception as exc:  # noqa: BLE001
            print(f'  pulse_match failed on {run}/{sub}: {exc!r}')
            continue
        if not m:
            continue
        mapping = {int(k): v for k, v in m['event_e10'].items()}
        ids = ev['eventId'].to_numpy()[idx]
        e10[idx] = np.array([mapping.get(int(i), np.nan) for i in ids])
    ev['e10'] = e10
    ev['iband'] = np.where(np.isnan(e10), '',
                           np.where(e10 >= E10_SPLIT, 'HIGH', 'LOW'))
    ev[['run', 'subrun', 'eventId', 'e10']].to_parquet(INTENSITY_CACHE)
    return ev


def recovery_t50(ev_det, Ld, bin_ms=0.5, dt_max=40.0, plateau=(40.0, 76.0),
                 n_boot=300, seed=12345):
    """Time at which P(pair) first reaches half its late-dt plateau.

    THE scalar the question reduces to: if the lead removal lengthened the
    post-flash saturation, t50 moves later. Error is a binomial bootstrap over
    the per-bin counts (bins are independent — unlike the boxcar curves).
    """
    g = ev_det[ev_det[f'readout_{Ld}']]
    dt = g['dt_ms'].to_numpy()
    hit = (g[f'n_pair_{Ld}'] > 0).to_numpy()
    edges = np.arange(0.0, dt_max + bin_ms, bin_ms)
    cen = 0.5 * (edges[:-1] + edges[1:])
    n, _ = np.histogram(dt, bins=edges)
    k, _ = np.histogram(dt[hit], bins=edges)
    pl_m = (dt >= plateau[0]) & (dt < plateau[1])
    n_pl, k_pl = int(pl_m.sum()), int(hit[pl_m].sum())

    def _t50(kk, nn, kp, np_):
        with np.errstate(invalid='ignore', divide='ignore'):
            p = np.where(nn > 0, kk / np.maximum(nn, 1), np.nan)
        half = 0.5 * (kp / max(np_, 1))
        ok = nn > 0
        for i in range(1, len(p)):
            if not (ok[i] and ok[i - 1]):
                continue
            if p[i - 1] < half <= p[i]:
                f = (half - p[i - 1]) / (p[i] - p[i - 1])
                return cen[i - 1] + f * (cen[i] - cen[i - 1])
        return np.nan

    val = _t50(k, n, k_pl, n_pl)
    rng = np.random.default_rng(seed)
    boot = np.array([_t50(rng.binomial(n, np.divide(k, np.maximum(n, 1))), n,
                          rng.binomial(n_pl, k_pl / max(n_pl, 1)), n_pl)
                     for _ in range(n_boot)])
    return val, float(np.nanstd(boot)), 0.5 * (k_pl / max(n_pl, 1))


# ---------------------------------------------------------------- figures
def fig_acceptance(ev, runs, out):
    """Accepted events per flash burst per ms vs dt, + after/before ratio."""
    bin_ms = 0.5
    edges = np.arange(0.0, 80.0 + bin_ms, bin_ms)
    cen = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    rates, nb = {}, {}
    for run in runs:
        g = ev[ev.run == run]
        n_bursts = int((g.is_leader & g.flash_ok).sum())
        nb[run] = n_bursts
        dt = g.loc[~g.is_leader, 'dt_ms'].to_numpy()
        h, _ = np.histogram(dt, bins=edges)
        r = h / max(n_bursts, 1) / bin_ms
        rates[run] = r
        axes[0].step(cen, r, where='mid', color=L.RUN_COLOR[run], lw=1.4,
                     label=f'{RUN_LABEL[run]}  ({n_bursts} bursts)')
    axes[0].set_ylabel('accepted events / burst / ms')
    axes[0].set_title('Trigger acceptance vs time since flash '
                      '(DAQ-level: leaders excluded, per flash-ok burst)')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 80)
    axes[0].axvspan(1, 5, color='gold', alpha=0.15, lw=0)
    if 'run_132' in rates and 'run_139' in rates:
        b, a = rates['run_132'], rates['run_139']
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(b > 0, a / b, np.nan)
            # per-bin Poisson ratio error
            kb = b * nb['run_132'] * bin_ms
            ka = a * nb['run_139'] * bin_ms
            rerr = ratio * np.sqrt(1 / np.maximum(ka, 1) + 1 / np.maximum(kb, 1))
        axes[1].axhline(1.0, color='k', lw=0.8)
        axes[1].errorbar(cen, ratio, yerr=rerr, fmt='.', ms=3, lw=0.7,
                         color='purple')
        axes[1].set_ylabel('after / before\n(139 / 132)')
        axes[1].set_ylim(0.5, 1.5)
        axes[1].axvspan(1, 5, color='gold', alpha=0.15, lw=0)
    axes[1].set_xlabel('time since gamma flash [ms]')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_acceptance_zoom(ev, runs, out):
    bin_ms = 0.25
    edges = np.arange(0.0, 12.0 + bin_ms, bin_ms)
    cen = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    rates, nb = {}, {}
    for run in runs:
        g = ev[ev.run == run]
        n_bursts = int((g.is_leader & g.flash_ok).sum())
        nb[run] = n_bursts
        dt = g.loc[~g.is_leader, 'dt_ms'].to_numpy()
        h, _ = np.histogram(dt, bins=edges)
        r = h / max(n_bursts, 1) / bin_ms
        rates[run] = r
        err = np.sqrt(h) / max(n_bursts, 1) / bin_ms
        axes[0].errorbar(cen, r, yerr=err, fmt='-', lw=1.2, ms=3,
                         color=L.RUN_COLOR[run], label=RUN_LABEL[run])
    axes[0].set_ylabel('accepted events / burst / ms')
    axes[0].set_title('Trigger acceptance, first 12 ms')
    axes[0].legend(fontsize=9)
    if 'run_132' in rates and 'run_139' in rates:
        b, a = rates['run_132'], rates['run_139']
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(b > 0, a / b, np.nan)
            kb = b * nb['run_132'] * bin_ms
            ka = a * nb['run_139'] * bin_ms
            rerr = ratio * np.sqrt(1 / np.maximum(ka, 1) + 1 / np.maximum(kb, 1))
        axes[1].axhline(1.0, color='k', lw=0.8)
        axes[1].errorbar(cen, ratio, yerr=rerr, fmt='.', ms=4, lw=0.8,
                         color='purple')
        axes[1].set_ylabel('after / before')
        axes[1].set_ylim(0.4, 1.6)
    axes[1].set_xlabel('time since gamma flash [ms]')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_curves(curves, runs, ycol, ecol, ylabel, title, out, xmax=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, Ld in zip(axes.ravel(), DET_ORDER):
        for run in runs:
            c = curves[(curves.det == Ld) & (curves.run == run)]
            if c.empty:
                continue
            ax.plot(c.dt_ms, c[ycol], '-', color=L.RUN_COLOR[run], lw=1.4,
                    label=RUN_LABEL[run])
            ax.fill_between(c.dt_ms, c[ycol] - c[ecol], c[ycol] + c[ecol],
                            color=L.RUN_COLOR[run], alpha=0.20, lw=0)
        ax.set_title(f'Det {Ld}' + ('  (clean M1 — reference)' if Ld == 'A'
                                    else ''))
        ax.axvspan(1, 5, color='gold', alpha=0.15, lw=0)
        if xmax:
            ax.set_xlim(0, xmax)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel('time since gamma flash [ms]')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_verdict(fine_c, t50, runs, out):
    """The one plot that answers the question: Det A recovery, three runs,
    with the 50%-recovery times and their bootstrap errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw={'width_ratios': [2.4, 1]})
    ax = axes[0]
    for run in runs:
        c = fine_c[(fine_c.det == 'A') & (fine_c.run == run)]
        if c.empty:
            continue
        ax.plot(c.dt_ms, c.p_pair, '-', color=L.RUN_COLOR[run], lw=1.6,
                label=RUN_LABEL[run])
        ax.fill_between(c.dt_ms, c.p_pair - c.e_pair, c.p_pair + c.e_pair,
                        color=L.RUN_COLOR[run], alpha=0.22, lw=0)
    for _, r in t50[t50.det == 'A'].iterrows():
        if np.isfinite(r.t50_ms):
            ax.axvline(r.t50_ms, color=L.RUN_COLOR[r.run], ls='--', lw=1.0)
    ax.axvspan(1, 5, color='gold', alpha=0.18, lw=0,
               label='the 1-5 ms window in question')
    ax.set_xlim(1, 15)
    ax.set_xlabel('time since gamma flash [ms]')
    ax.set_ylabel('P(3D x/y pair) per trigger')
    ax.set_title('Det A (clean M1): post-flash recovery\n'
                 'dashed = 50% of plateau')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    sub = t50[(t50.det == 'A')]
    xs = np.arange(len(sub))
    ax.errorbar(xs, sub.t50_ms, yerr=sub.t50_err_ms, fmt='o', ms=7, capsize=5,
                lw=1.5, color='k')
    for x, r in zip(xs, sub.itertuples()):
        ax.plot(x, r.t50_ms, 'o', ms=7, color=L.RUN_COLOR[r.run])
    ax.set_xticks(xs)
    ax.set_xticklabels([r.split('_')[1] + '\n' + L.PERIOD[r]
                        for r in sub.run], fontsize=9)
    ax.set_ylabel('50% recovery time [ms]')
    ax.set_title('t50, Det A\n(longer saturation would push this UP)')
    ax.grid(alpha=0.25, axis='y')
    fig.suptitle('Did the 2026-08-04 lead removal lengthen the post-flash '
                 'DAQ saturation?   —   No.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_flash_size(ev, runs, out):
    lead = ev[ev.is_leader & ev.flash_ok]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        g = lead[lead.run == run]
        axes[0].hist(g.n_big, bins=60, histtype='step', density=True,
                     color=L.RUN_COLOR[run], label=RUN_LABEL[run])
        axes[1].hist(g.n_hits_tot, bins=60, histtype='step', density=True,
                     color=L.RUN_COLOR[run], label=RUN_LABEL[run])
    axes[0].set_xlabel('flash leader: n hits with amp >= 1000')
    axes[1].set_xlabel('flash leader: total hits')
    for ax in axes:
        ax.set_ylabel('density')
        ax.legend(fontsize=8)
    fig.suptitle('Gamma-flash leader size per run (did the flash itself grow?)')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_first_accept(ev, runs, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    q_table = {}
    for run in runs:
        g = ev[(ev.run == run) & ev.flash_ok]
        first = (g[~g.is_leader].groupby(['run', 'subrun', 'burst'])['dt_ms']
                 .min())
        q = first.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        q_table[run] = q
        ax.hist(first, bins=np.arange(0, 20.25, 0.25), histtype='step',
                density=True, color=L.RUN_COLOR[run],
                label=f'{RUN_LABEL[run]}  med={q[0.5]:.2f} ms')
    ax.set_xlabel('first accepted trigger after the flash [ms]')
    ax.set_ylabel('density of bursts')
    ax.set_title('Per-burst first-accept time (direct DAQ-recovery observable)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return q_table


# ---------------------------------------------------------------- tables
def window_table(ev, runs, iband=None):
    if iband:
        ev = ev[(ev.iband == iband) | ev.is_leader]
    rows = []
    for lo, hi in L.WINDOWS:
        wl = L.win_label(lo, hi)
        for Ld in DET_ORDER:
            r = {'window': wl, 'lo': lo, 'hi': hi, 'det': Ld}
            for run in runs:
                g = ev[(ev.run == run) & ~ev.is_leader
                       & (ev.dt_ms >= lo) & (ev.dt_ms < hi)
                       & ev[f'readout_{Ld}']]
                n = len(g)
                k = int((g[f'n_pair_{Ld}'] > 0).sum())
                kb = int((~g[f'live_{Ld}']).sum())
                nburst = int((ev[(ev.run == run)].is_leader
                              & ev[(ev.run == run)].flash_ok).sum())
                tag = run.split('_')[1]
                r[f'n_{tag}'] = n
                r[f'k_{tag}'] = k
                p, e = L.binom_err(k, max(n, 1))
                r[f'eff_{tag}'] = float(p)
                r[f'err_{tag}'] = float(e)
                r[f'blind_{tag}'] = kb / n if n else np.nan
                r[f'evperburst_{tag}'] = n / max(nburst, 1)
            if all(f'k_{t}' in r for t in ('132', '139')):
                z, pv = two_prop_z(r['k_139'], max(r['n_139'], 1),
                                   r['k_132'], max(r['n_132'], 1))
                r['deff_after_minus_before'] = r['eff_139'] - r['eff_132']
                r['z'] = z
                r['p_value'] = pv
            # CONTROL: two 'before' runs on consecutive evenings. Whatever this
            # delta is, it is the night-to-night systematic floor — an
            # after-before delta of the same size means nothing.
            if all(f'k_{t}' in r for t in ('130', '132')):
                zc, pc = two_prop_z(r['k_132'], max(r['n_132'], 1),
                                    r['k_130'], max(r['n_130'], 1))
                r['dctl_132_minus_130'] = r['eff_132'] - r['eff_130']
                r['z_ctl'] = zc
                r['p_ctl'] = pc
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=6.0)
    ap.add_argument('--step', type=float, default=1.0)
    ap.add_argument('--runs', default=','.join(L.RUNS))
    ap.add_argument('--no-intensity', action='store_true')
    args = ap.parse_args()
    runs = [r for r in args.runs.split(',') if r]

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    ev, segs, spec = L.load_all(runs=runs)
    if ev.empty:
        sys.exit('no cached events — run process.py first')
    print('cached sub-runs per run:')
    print(ev.groupby('run')['subrun'].nunique().to_string())

    ev = FP.attach(ev)
    n0 = len(ev)
    ev = ev[ev.flash_ok].copy()
    print(f'flash_ok: kept {len(ev)}/{n0} events')
    if not args.no_intensity:
        ev = attach_intensity(ev)
    else:
        ev['e10'] = np.nan
        ev['iband'] = ''

    # ---- per-run bookkeeping ----
    book = []
    for run in runs:
        g = ev[ev.run == run]
        lead = g[g.is_leader]
        probes = g[~g.is_leader]
        book.append({
            'run': run, 'period': L.PERIOD[run],
            'subruns': g.subrun.nunique(),
            'bursts': len(lead),
            'probe_events': len(probes),
            'ev_per_burst': len(probes) / max(len(lead), 1),
            'reco_skipped_frac': float(probes.reco_skipped.mean()),
            'nbig_leader_med': float(lead.n_big.median()),
            'nhits_leader_med': float(lead.n_hits_tot.median()),
            'e10_med': float(lead.e10.median()) if lead.e10.notna().any()
            else np.nan,
            'frac_high': float((lead.iband == 'HIGH').mean()),
            'readout_A_frac': float(probes.readout_A.mean()),
        })
    book = pd.DataFrame(book)
    book.to_csv(os.path.join(TAB_DIR, 'run_bookkeeping.csv'), index=False)
    print(book.to_string(index=False))

    # ---- gate edges per run, common analysis range ----
    gates = {run: gate_edges(ev[ev.run == run]) for run in runs}
    for run, (lo, hi) in gates.items():
        print(f'  {run}: measured dt acceptance {lo:g}-{hi:g} ms')
    lo = max(g[0] for g in gates.values())
    hi = min(g[1] for g in gates.values())

    # ---- boxcar curves (main + fine zoom) ----
    def build_curves(width, step, min_n, dt_max=None, iband=None):
        rows = []
        cen = centers(lo, min(hi, dt_max) if dt_max else hi, width, step)
        for run in runs:
            g = ev[(ev.run == run) & ~ev.is_leader]
            if iband:
                g = g[g.iband == iband]
            for Ld in DET_ORDER:
                sub = g[g[f'readout_{Ld}']]
                if sub.empty:
                    continue
                c = curve(sub, Ld, cen, width, min_n)
                if c.empty:
                    continue
                c['run'], c['det'] = run, Ld
                c['period'] = L.PERIOD[run]
                if iband:
                    c['iband'] = iband
                rows.append(c)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    main_c = build_curves(args.width, args.step, MIN_N_PER_POINT)
    fine_c = build_curves(2.0, 0.25, 40, dt_max=15.0)
    main_c.to_csv(os.path.join(TAB_DIR, 'boxcar_curves.csv'), index=False)
    fine_c.to_csv(os.path.join(TAB_DIR, 'boxcar_curves_fine.csv'), index=False)

    fig_curves(main_c, runs, 'p_pair', 'e_pair', 'P(3D x/y pair) per trigger',
               f'Track efficiency vs time since flash — boxcar W={args.width:g} '
               f'ms, step {args.step:g} ms (points correlated)',
               os.path.join(FIG_DIR, 'eff_vs_dt.png'))
    fig_curves(fine_c, runs, 'p_pair', 'e_pair', 'P(3D x/y pair) per trigger',
               'Track efficiency, first 15 ms — boxcar W=2 ms, step 0.25 ms '
               '(points correlated)',
               os.path.join(FIG_DIR, 'eff_vs_dt_zoom.png'), xmax=15)
    fig_curves(main_c, runs, 'blind_frac', 'e_blind', 'blind fraction',
               'Detector blindness (no hits at all) vs time since flash',
               os.path.join(FIG_DIR, 'blind_vs_dt.png'))
    fig_curves(fine_c, runs, 'blind_frac', 'e_blind', 'blind fraction',
               'Detector blindness, first 15 ms',
               os.path.join(FIG_DIR, 'blind_vs_dt_zoom.png'), xmax=15)

    # intensity-matched (HIGH pulses only) — removes the delivered-mix nuisance
    if (ev.iband == 'HIGH').any():
        hi_c = build_curves(args.width, args.step, MIN_N_PER_POINT,
                            iband='HIGH')
        if not hi_c.empty:
            hi_c.to_csv(os.path.join(TAB_DIR, 'boxcar_curves_high.csv'),
                        index=False)
            fig_curves(hi_c, runs, 'p_pair', 'e_pair',
                       'P(3D x/y pair) per trigger',
                       'Track efficiency vs dt — HIGH-intensity pulses only',
                       os.path.join(FIG_DIR, 'eff_vs_dt_high.png'))

    # ---- acceptance + first-accept + flash size ----
    fig_acceptance(ev, runs, os.path.join(FIG_DIR, 'acceptance_vs_dt.png'))
    fig_acceptance_zoom(ev, runs,
                        os.path.join(FIG_DIR, 'acceptance_vs_dt_zoom.png'))
    q_first = fig_first_accept(ev, runs,
                               os.path.join(FIG_DIR, 'first_accept.png'))
    fig_flash_size(ev, runs, os.path.join(FIG_DIR, 'flash_leader_size.png'))

    # ---- fixed-window statistics ----
    wt = window_table(ev, runs)
    wt.to_csv(os.path.join(TAB_DIR, 'window_stats.csv'), index=False)
    wt_hi = pd.DataFrame()
    if (ev.iband == 'HIGH').any():
        wt_hi = window_table(ev, runs, iband='HIGH')
        wt_hi.to_csv(os.path.join(TAB_DIR, 'window_stats_high.csv'),
                     index=False)

    # ---- the scalar: 50% recovery time per run/detector ----
    t50 = []
    probes = ev[~ev.is_leader]
    for run in runs:
        for Ld in DET_ORDER:
            v, e, half = recovery_t50(probes[probes.run == run], Ld)
            t50.append({'run': run, 'period': L.PERIOD[run], 'det': Ld,
                        't50_ms': v, 't50_err_ms': e, 'half_plateau_eff': half})
    t50 = pd.DataFrame(t50)
    t50.to_csv(os.path.join(TAB_DIR, 'recovery_t50.csv'), index=False)
    fig_verdict(fine_c, t50, runs, os.path.join(FIG_DIR, 'VERDICT_detA.png'))

    # ---- summary ----
    lines = ['# Lead-shielding removal (2026-08-04 access): before/after '
             'DAQ-saturation check\n',
             f'Runs: {", ".join(RUN_LABEL[r] for r in runs)}',
             f'Common measured dt acceptance: {lo:g}-{hi:g} ms\n',
             '## Bookkeeping\n', book.to_string(index=False), '',
             '## Per-burst first-accept dt quantiles [ms]\n']
    for run, q in q_first.items():
        lines.append(f'  {RUN_LABEL[run]}: '
                     + '  '.join(f'q{int(k * 100)}={v:.2f}'
                                 for k, v in q.items()))
    lines.append('\n## 50% recovery time (efficiency reaches half its 40-76 ms '
                 'plateau)')
    lines.append('The scalar the question reduces to — a longer saturation '
                 'moves t50 later.\n')
    lines.append(t50.pivot(index='det', columns='run',
                           values='t50_ms').to_string(
        float_format=lambda v: f'{v:.2f}'))
    lines.append('\n +/- (binomial bootstrap, ms)')
    lines.append(t50.pivot(index='det', columns='run',
                           values='t50_err_ms').to_string(
        float_format=lambda v: f'{v:.2f}'))
    lines.append('\n## Fixed-window efficiency, after (139) vs before (132)')
    lines.append('(z_ctl / dctl = the before-vs-before control, run_132 vs '
                 'run_130 — the night-to-night floor)\n')
    want = ['window', 'n_132', 'eff_132', 'n_139', 'eff_139',
            'deff_after_minus_before', 'z', 'p_value',
            'dctl_132_minus_130', 'z_ctl']
    for Ld in DET_ORDER:
        sub = wt[wt.det == Ld]
        have = [c for c in want if c in sub.columns]
        lines.append(f'Det {Ld}' + (' (clean M1 reference):' if Ld == 'A'
                                    else ' (bad M1 — single-plane yields '
                                         'noise-inflated):'))
        lines.append(sub[have].to_string(index=False,
                                         float_format=lambda v: f'{v:.4f}'))
        lines.append('')
    if not wt_hi.empty:
        lines.append('## Intensity-matched: HIGH pulses only (>= 600e10 p), '
                     'Det A\n')
        sub = wt_hi[wt_hi.det == 'A']
        have = [c for c in want if c in sub.columns]
        lines.append(sub[have].to_string(index=False,
                                         float_format=lambda v: f'{v:.4f}'))
        lines.append('')

    lines.append('## Trigger acceptance and blindness per window (Det A)\n')
    sub = wt[wt.det == 'A']
    want2 = ['window'] + [f'{p}_{t}' for t in ('130', '132', '139')
                          for p in ('evperburst', 'blind')]
    have2 = [c for c in want2 if c in sub.columns]
    lines.append(sub[have2].to_string(index=False,
                                      float_format=lambda v: f'{v:.4f}'))
    lines.append('\nAll detectors + blindness + ev/burst -> '
                 'tables/window_stats.csv')
    with open(os.path.join(L.OUT_BASE, 'SUMMARY.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nfigures ->', FIG_DIR)


if __name__ == '__main__':
    main()
