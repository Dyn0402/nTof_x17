#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10c_hv_scan_gain.py -- relative gain vs amplification HV, for the SAME scan
that ``10b_hv_scan_efficiency.py`` measures the efficiency of.

Why
---
``10b`` says the det3 saturday scan is flat at 93-95 % from 435 V to 500 V and
then collapses.  Flat efficiency is the *absence* of a measurement: it does not
say whether the chamber is coasting on a huge gain margin or sitting one volt
above threshold, and it does not say what actually ends the plateau.  The gain
curve is the missing axis, and it turns the flat line into a statement:

  * gain rises **monotonically and exponentially** across the whole scan, by a
    factor of ~25 between 425 V and 500 V -- so the flat efficiency is not a
    saturated observable hiding a turn-on somewhere below 425 V;
  * at the LOW end the weakest 2 % of events still sit ~9 sigma over the
    pedestal, i.e. ~2x the DAQ's own 5-sigma threshold -- the detector is
    nowhere near running out of signal at 425 V;
  * what ends the plateau above 500 V is the discharge rate, not a lack of
    gain.  Gain is still climbing where the efficiency is falling.

Observable
----------
The **threshold-free peak-strip waveform maximum** on M3-selected muons, per
view -- ``mx17_sim_wft/hv_slope/extract.py``'s ``peaks.parquet``.  That file is
built from *these same sub-runs* on *this same golden M3 recipe*
(chi2 < 1.0 & NClus = 4), with no amplitude threshold anywhere, which is what
makes it a fair gain probe: a 5-sigma cut would truncate the low tail and bias
the low-voltage points *up*, faking a shallower rise.  Reading it here rather
than re-extracting keeps this script cheap; ``--check`` verifies the parquet
still describes the sub-runs on disk before anything is plotted.

This is a **relative** gain curve, normalised to the 490 V bench operating
point.  Nothing here converts ADC to electrons: the DREAM CSA range for the
June bench is not recorded, and the Garfield absolute gain is exactly the thing
the T14 campaign found to be x0.55-0.63 off, so an absolute axis would be a
number with no error bar.  d ln A / dV is the part that is measured.

Saturation
----------
The DREAM sample is 12-bit and the peak strip is on the rail above ~505 V.  The
rule, declared before the fit (and identical to ``mx17_sim_wft/hv_slope``): an
estimator is used at a voltage only while its quantile sits below 70 % of the
measured rail.  Saturated points are drawn open and never enter a fit.  Low
quantiles (p02, p10) survive higher than the median, and the +-1-strip
neighbour sum survives higher still -- the two neighbours together carry ~1.1x
the peak amplitude on these strips, but the SUM only clips once both of them
reach the rail, which buys ~15 V.  All of them agree in the overlap, which is
the check that the low quantiles are not themselves being pulled by clipping.

Usage
-----
    ../.venv/bin/python 10c_hv_scan_gain.py            # ladder + combined plot
    ../.venv/bin/python 10c_hv_scan_gain.py --check    # provenance only
    ../.venv/bin/python 10c_hv_scan_gain.py --slides   # also write deck PNGs

Output -> <cosmic_bench>/Analysis/mx17_det3_saturday_scan_6-27-26/hv_scan/
          mx17_3/gain_vs_hv.{csv,meta.json,png} and gain_and_efficiency.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

RUN = 'mx17_det3_saturday_scan_6-27-26'
DET = 'mx17_3'
BASE = '/home/dylan/x17/cosmic_bench/det3/'
ANALYSIS = os.path.join(os.path.dirname(BASE.rstrip('/')), 'Analysis', RUN)
PEAKS = os.path.expanduser('~/x17/response_sim/hv_slope/peaks.parquet')

# Fiducial region in reference-frame mm at the det3 plane (z = 702), chosen ONCE
# from the >=505 V hit map in mx17_sim_wft/hv_slope/analyse.py and applied
# unchanged at every voltage.  Kept identical here so this curve and the
# published HV-slope result are the same selection.
FID_X = (-190.0, 115.0)
FID_Y = (-190.0, 165.0)

SAT_FRAC = 0.70          # an estimator is used only below this fraction of the rail
V_REF = 490              # bench operating point -- relative gain is 1 here
SIGMA_DAQ = 5.0          # the threshold the DAQ itself loads
ESTIMATORS = [('p02', 0.02), ('p10', 0.10), ('p25', 0.25), ('p50', 0.50)]
FIT_EST = 'p10'          # the one quoted: unsaturated highest, still off the noise floor
NBOOT = 400
RNG = np.random.default_rng(20260828)

SUBRUN_RE = re.compile(r'^hv_scan(2?)_resist_(\d+)V_drift_1000V$')


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def disk_subruns():
    """(name, volt, pass) for every mesh-voltage sub-run actually on disk."""
    out = []
    for d in sorted(os.listdir(os.path.join(BASE, RUN))):
        m = SUBRUN_RE.match(d)
        if m:
            out.append((d, int(m.group(2)), 'scan2' if m.group(1) else 'scan1'))
    return sorted(out, key=lambda t: t[1])


def check_provenance(df):
    """A reduction on disk carries no basis stamp -- date it before plotting.

    Three things can silently rot: a sub-run added or re-decoded since the
    parquet was written, the parquet being older than the raw waveforms it
    claims to summarise, and the M3 recipe having moved.  All three are cheap
    to test and all three would change the curve.
    """
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    ok = True
    have = {s for s in df.subrun.unique()}
    want = {n for n, _v, _s in disk_subruns()}
    if have != want:
        ok = False
        print(f'  [FAIL] sub-run set differs: only on disk {sorted(want - have)}, '
              f'only in parquet {sorted(have - want)}')
    else:
        print(f'  [ok]   {len(want)} mesh sub-runs, parquet and disk agree')

    t_pq = os.path.getmtime(PEAKS)
    newer = [n for n in sorted(want)
             if os.path.getmtime(os.path.join(BASE, RUN, n, 'decoded_root')) > t_pq]
    if newer:
        ok = False
        print(f'  [FAIL] decoded_root re-written after the parquet: {newer}')
    else:
        print(f'  [ok]   no decoded_root touched since the parquet '
              f'({pd.Timestamp(t_pq, unit="s"):%Y-%m-%d})')

    # extract.py hardcodes the golden recipe by importing these two constants,
    # so agreeing with them today is the statement that it still holds.
    if (M3_CHI2_CUT, M3_MIN_NCLUS) != (1.0, 4):
        ok = False
        print(f'  [FAIL] M3 recipe moved to chi2<{M3_CHI2_CUT} & '
              f'NClus={M3_MIN_NCLUS}; re-run hv_slope/extract.py')
    else:
        print(f'  [ok]   M3 recipe still chi2<{M3_CHI2_CUT} & NClus={M3_MIN_NCLUS}')
    return ok


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #

def fiducial(df):
    return df[(df.ref_x > FID_X[0]) & (df.ref_x < FID_X[1]) &
              (df.ref_y > FID_Y[0]) & (df.ref_y < FID_Y[1])]


def ladder(df, rail):
    """Per (view, voltage) amplitude quantiles with bootstrap errors on ln."""
    rows = []
    for (view, volt), g in df.groupby(['view', 'volt']):
        a = g.peak_amp.values
        row = dict(view=view, hv=int(volt), scan=g.scan.iloc[0], n=len(a),
                   noise_adc=float(np.median(g.noise)),
                   fsat=float((a > 0.88 * rail).mean()),
                   nb_p50=float(np.median(g.nb_amp)),
                   n_over_p50=float(np.median(g.n_over)))
        row['nb_p50_ok'] = bool(row['nb_p50'] < SAT_FRAC * 2 * rail)
        for name, q in ESTIMATORS:
            v = float(np.quantile(a, q))
            bs = np.quantile(a[RNG.integers(0, len(a), (NBOOT, len(a)))], q, axis=1)
            row[name] = v
            row[f'{name}_lnerr'] = float(np.std(np.log(bs)))
            row[f'{name}_ok'] = bool(v < SAT_FRAC * rail)
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(['view', 'hv']).reset_index(drop=True)
    # signal-to-noise of the WEAKEST events -- this is the turn-on test
    t['snr_p02'] = t.p02 / t.noise_adc
    t['snr_p50'] = t.p50 / t.noise_adc
    return t


def loglin(v, lna, err):
    """Weighted straight-line fit of ln A on V.  Returns (slope/10 V, err, c)."""
    w = 1.0 / np.maximum(err, 1e-6) ** 2
    S, Sx = w.sum(), (w * v).sum()
    Sxx, Sy, Sxy = (w * v * v).sum(), (w * lna).sum(), (w * v * lna).sum()
    d = S * Sxx - Sx * Sx
    return ((S * Sxy - Sx * Sy) / d * 10.0, np.sqrt(S / d) * 10.0,
            (Sxx * Sy - Sx * Sxy) / d)


def fit_all(t):
    out = {}
    for view in ('x', 'y'):
        s = t[t.view == view]
        per = {}
        for name, _q in ESTIMATORS:
            m = s[f'{name}_ok'].values
            if m.sum() < 3:
                continue
            sl, se, c = loglin(s.hv[m].values, np.log(s[name][m].values),
                               s[f'{name}_lnerr'][m].values)
            per[name] = dict(slope10=sl, err10=se, intercept=c, n=int(m.sum()),
                             vmin=int(s.hv[m].min()), vmax=int(s.hv[m].max()),
                             efold_V=10.0 / sl, double_V=np.log(2) / sl * 10.0)
        m = s['nb_p50_ok'].values
        sl, se, c = loglin(s.hv[m].values, np.log(s.nb_p50[m].values),
                           np.full(int(m.sum()), 0.03))
        per['nb_p50'] = dict(slope10=sl, err10=se, intercept=c, n=int(m.sum()),
                             vmin=int(s.hv[m].min()), vmax=int(s.hv[m].max()),
                             efold_V=10.0 / sl, double_V=np.log(2) / sl * 10.0)
        out[view] = per
    return out


def relative_gain(t, est=FIT_EST):
    """Amplitude ladder renormalised to V_REF, per view.  NaN where saturated."""
    t = t.copy()
    g, ge = [], []
    for view in ('x', 'y'):
        s = t[t.view == view]
        ref = float(s[s.hv == V_REF][est].iloc[0])
        rel = np.where(s[f'{est}_ok'], s[est] / ref, np.nan)
        g.append(pd.Series(rel, index=s.index))
        ge.append(pd.Series(rel * np.hypot(s[f'{est}_lnerr'].values,
                                           float(s[s.hv == V_REF][f'{est}_lnerr'].iloc[0])),
                            index=s.index))
    t['rel_gain'] = pd.concat(g).sort_index()
    t['rel_gain_err'] = pd.concat(ge).sort_index()
    return t


def efficiency_table():
    """Both interleaved passes of 10b's efficiency scan, on one voltage axis."""
    frames = []
    for out in ('hv_scan', 'hv_scan2'):
        p = os.path.join(ANALYSIS, out, DET, 'efficiency_vs_hv.csv')
        if not os.path.exists(p):
            sys.exit(f'No {p}\nRun 10b_hv_scan_efficiency.py first.')
        d = pd.read_csv(p)
        d['pass'] = out
        frames.append(d)
    return pd.concat(frames).sort_values('hv').reset_index(drop=True)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

COL = {'p02': '#8c564b', 'p10': '#1f77b4', 'p25': '#2ca02c', 'p50': '#d62728'}
LBL = {'p02': '2nd percentile (weakest 2 % of muons)',
       'p10': '10th percentile', 'p25': '25th percentile', 'p50': 'median'}


def fig_ladder(t, fits, rail, path):
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    s = t[t.view == 'x']
    for name, _q in ESTIMATORS:
        ok = s[f'{name}_ok'].values
        ax.errorbar(s.hv[ok], s[name][ok],
                    yerr=(s[name] * s[f'{name}_lnerr'])[ok],
                    fmt='o', ms=6, color=COL[name], capsize=3, label=LBL[name])
        if (~ok).any():
            ax.plot(s.hv[~ok], s[name][~ok], 'o', ms=6, mfc='none',
                    mec=COL[name], alpha=.55)
        f = fits['x'][name]
        vv = np.linspace(s.hv.min(), s.hv.max(), 50)
        ax.plot(vv, np.exp(f['intercept'] + f['slope10'] / 10.0 * vv),
                '-', lw=1, color=COL[name], alpha=.45)
    ax.axhline(rail, color='k', lw=1.2, ls='--')
    ax.text(524, rail * 1.05, f'12-bit ADC rail ({rail:.0f})', fontsize=8,
            va='bottom', ha='right')
    n0 = float(s.noise_adc.iloc[0])
    ax.axhline(SIGMA_DAQ * n0, color='crimson', lw=1.2, ls=':')
    ax.text(427, SIGMA_DAQ * n0 * 1.08,
            f'DAQ threshold, {SIGMA_DAQ:g}$\\sigma$ = {SIGMA_DAQ * n0:.0f} ADC',
            fontsize=8, color='crimson', va='bottom')
    f = fits['x'][FIT_EST]
    ax.set_yscale('log')
    ax.set_xlabel('amplification (resistive-layer) HV [V]')
    ax.set_ylabel('peak-strip waveform maximum [ADC]')
    ax.set_title('det3 (X view) gain ladder — 27 June saturday scan\n'
                 'threshold-free peak strip on M3-selected muons '
                 f'(chi2<1, NClus=4); open = on the rail\n'
                 f'{FIT_EST}: d ln A / dV = {f["slope10"]:.3f} ± {f["err10"]:.3f} '
                 f'per 10 V  →  ×2 every {f["double_V"]:.1f} V '
                 f'({f["vmin"]}–{f["vmax"]} V)', fontsize=10)
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_combined(t, eff, fits, path, slides=False):
    """Gain and efficiency on one voltage axis -- the plateau, explained."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1],
                                                  hspace=0.08))
    s = t[t.view == 'x']
    ok = np.isfinite(s.rel_gain.values)
    f = fits['x'][FIT_EST]
    vv = np.linspace(425, 525, 60)
    ref = float(s[s.hv == V_REF][FIT_EST].iloc[0])
    hx = ax.plot(vv, np.exp(f['intercept'] + f['slope10'] / 10.0 * vv) / ref,
                 ':', lw=1.2, color='0.35',
                 label=f'exponential fit, ×2 every {f["double_V"]:.1f} V '
                       f'(dotted past the rail)')[0]
    h1 = ax.errorbar(s.hv[ok], s.rel_gain[ok], yerr=s.rel_gain_err[ok],
                     fmt='o-', ms=6, lw=1.6, color='#1f77b4', capsize=3,
                     label=f'relative gain, {LBL[FIT_EST]} (X view)')
    sy = t[t.view == 'y']
    oky = np.isfinite(sy.rel_gain.values)
    h2 = ax.plot(sy.hv[oky], sy.rel_gain[oky], 's--', ms=4, lw=1.1,
                 color='#7fb3d8', label='same, Y view')[0]
    ax.axvspan(506, 526, color='crimson', alpha=.06)
    ax.set_yscale('log')
    ax.set_ylim(0.045, 6.0)
    ax.text(516, 4.4, 'peak strip on\nthe ADC rail', ha='center', fontsize=8,
            color='crimson')
    ax.axhline(1.0, color='0.6', lw=.8)
    ax.plot([V_REF], [1.0], '*', ms=15, color='goldenrod', mec='0.3', zorder=5)
    ax.annotate('bench operating point', (V_REF, 1.0), textcoords='offset points',
                xytext=(-10, 8), ha='right', fontsize=8)
    ax.text(524, 0.055, '×25 in gain between 425 V and 500 V',
            fontsize=8.5, color='0.25', ha='right', va='bottom')
    ax.set_ylabel(f'gain relative to {V_REF} V')
    ax.grid(alpha=.3, which='both')
    ax.legend([h1, h2, hx], [h1.get_label(), h2.get_label(), hx.get_label()],
              fontsize=8, loc='upper left')

    bx.errorbar(eff.hv, eff.within_R, yerr=eff.within_R_err, fmt='o-', ms=6,
                lw=1.8, color='steelblue', capsize=3,
                label='efficiency: reconstructed within 5 mm of the reference')
    bx.axvspan(506, 526, color='crimson', alpha=.06)
    bx.set_ylim(0, 102)
    bx.set_ylabel('efficiency [%]')
    bx.set_xlabel('amplification (resistive-layer) HV [V]')
    bx.grid(alpha=.3)
    cx = bx.twinx()
    cx.plot(eff.hv, eff.spark_frac_pct, 'x:', ms=8, lw=1.5, color='crimson',
            label='discharge fraction of firing events')
    cx.set_ylabel('discharge fraction [%]', color='crimson')
    cx.tick_params(axis='y', labelcolor='crimson')
    cx.set_ylim(0, 60)
    h1, l1 = bx.get_legend_handles_labels()
    h2, l2 = cx.get_legend_handles_labels()
    bx.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower left')

    lo = s[s.hv == 425].iloc[0]
    bx.annotate(f'{lo.snr_p02:.0f}$\\sigma$ in the weakest 2 %\n'
                f'— {lo.snr_p02 / SIGMA_DAQ:.1f}× the DAQ threshold',
                (425, eff.within_R.iloc[0]), textcoords='offset points',
                xytext=(14, -34), fontsize=8, color='0.25',
                arrowprops=dict(arrowstyle='-', color='0.5', lw=.8))
    if not slides:
        ax.set_title('det3 — the plateau is a gain margin, not a saturated '
                     'observable\n'
                     'gain climbs ×%.0f over the scan while efficiency stays '
                     'flat; what ends it is the discharge rate'
                     % (np.nanmax(s.rel_gain) / np.nanmin(s.rel_gain)),
                     fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='verify the parquet still matches the disk, then stop')
    ap.add_argument('--slides', action='store_true',
                    help='also write untitled copies into mpgd26/slides/assets/img')
    ap.add_argument('--peaks', default=PEAKS)
    a = ap.parse_args()

    from qa_config import setup_paths                                # noqa: F401
    if not os.path.exists(a.peaks):
        sys.exit(f'No {a.peaks}\n'
                 f'Build it with:\n  ../.venv/bin/python '
                 f'mx17_sim_wft/hv_slope/extract.py --out {a.peaks}')

    raw = pd.read_parquet(a.peaks)
    print(f'peaks.parquet: {len(raw):,} rows, '
          f'{raw.volt.nunique()} voltages, {raw.subrun.nunique()} sub-runs')
    print('provenance:')
    ok = check_provenance(raw)
    if a.check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit('provenance check failed -- refusing to plot a stale reduction')

    df = fiducial(raw)
    print(f'fiducial x{FID_X} y{FID_Y}: keeps {len(df) / len(raw) * 100:.1f} %')

    # The effective rail: at 525 V >99 % of events clip, so the median peak
    # amplitude there IS the rail, averaged over the per-channel pedestal.
    rail = float(np.median(df[df.volt == 525].peak_amp))
    t = relative_gain(ladder(df, rail))
    fits = fit_all(t)
    eff = efficiency_table()

    od = os.path.join(ANALYSIS, 'hv_scan', DET)
    os.makedirs(od, exist_ok=True)
    t.to_csv(os.path.join(od, 'gain_vs_hv.csv'), index=False)
    json.dump(dict(run=RUN, det=DET, source=a.peaks, rail_adc=rail,
                   fiducial=dict(x=FID_X, y=FID_Y), sat_frac=SAT_FRAC,
                   v_ref=V_REF, fit_estimator=FIT_EST, n_boot=NBOOT,
                   fits=fits,
                   observable='threshold-free peak-strip waveform max, '
                              'M3-selected (chi2<1.0 & NClus=4), no amplitude cut',
                   note='RELATIVE gain only -- no ADC->electron calibration '
                        'exists for the June bench CSA range'),
              open(os.path.join(od, 'gain_vs_hv.meta.json'), 'w'), indent=1)

    print(f'\nrail = {rail:.0f} ADC   (fit on {FIT_EST}, unsaturated points only)')
    print(f'{"HV":>5} {"n":>5} {"p02":>8} {"p10":>8} {"p50":>8} {"nb_p50":>8} '
          f'{"relG":>7} {"SNR p02":>8} {"fsat%":>6}')
    for _, r in t[t.view == 'x'].iterrows():
        flag = '' if r.p10_ok else '  <- rail'
        print(f'{r.hv:>5.0f} {r.n:>5.0f} {r.p02:>8.1f} {r.p10:>8.1f} '
              f'{r.p50:>8.1f} {r.nb_p50:>8.1f} '
              f'{r.rel_gain:>7.3f} {r.snr_p02:>8.1f} {r.fsat * 100:>6.1f}{flag}')
    print()
    for view in ('x', 'y'):
        for name in ('p02', 'p10', 'p25', 'p50', 'nb_p50'):
            f = fits[view][name]
            print(f'  {view} {name:>6s}  {f["slope10"]:.4f} ± {f["err10"]:.4f} '
                  f'per 10 V   ×2 every {f["double_V"]:5.1f} V   '
                  f'({f["n"]} pts, {f["vmin"]}-{f["vmax"]} V)')

    sx = t[t.view == 'x']
    lo = sx[sx.hv == 425].iloc[0]
    for name in (FIT_EST, 'p02'):
        u = sx[sx[f'{name}_ok']]
        hi = u.iloc[-1]
        print(f'\ngain 425 -> {hi.hv:.0f} V: x{hi[name] / lo[name]:.1f} over '
              f'{hi.hv - 425:.0f} V  ({name}, both points unsaturated)')
    print(f'at 425 V the weakest 2 % of muons peak at {lo.p02:.0f} ADC = '
          f'{lo.snr_p02:.1f} sigma = {lo.snr_p02 / SIGMA_DAQ:.1f}x the DAQ '
          f'{SIGMA_DAQ:g} sigma threshold  -> no turn-on is expected anywhere '
          f'in this scan')

    fig_ladder(t, fits, rail, os.path.join(od, 'gain_vs_hv.png'))
    fig_combined(t, eff, fits, os.path.join(od, 'gain_and_efficiency.png'))
    print(f'\nwrote {od}/gain_vs_hv.{{csv,meta.json,png}} and '
          f'gain_and_efficiency.png')

    if a.slides:
        img = os.path.join(REPO, 'mpgd26', 'slides', 'assets', 'img')
        os.makedirs(img, exist_ok=True)
        fig_ladder(t, fits, rail, os.path.join(img, 'hv_gain_ladder.png'))
        fig_combined(t, eff, fits,
                     os.path.join(img, 'hv_gain_and_efficiency.png'), slides=True)
        print(f'wrote {img}/hv_gain_ladder.png, hv_gain_and_efficiency.png')


if __name__ == '__main__':
    main()
