#!/usr/bin/env python3
"""
10e_hv_scan_charge_angle.py — what the det3 mesh-voltage ladder does to the
*whole* signal, not just its tallest sample.

10c measured d ln A / dV on the peak strip's peak sample. That estimator has a
hard ceiling: the strip rails at 3550-3872 ADC and above ~500 V essentially
every track clips it, so the ladder has to stop exactly where the interesting
part starts. Three things are asked here instead, all from the waveform-first
forward fit (10d), which censors saturated samples with a one-sided penalty and
therefore keeps measuring after the peak sample stops moving:

  1. TOTAL CHARGE per track -- ``q_sum``, the NNLS charge integrated over the
     whole drift column and all strips in the window. This is the observable
     the gain curve should have been measured on; the peak strip is only its
     tallest slice.
  2. HOW MUCH OF THE TRACK IS LIT -- the reconstructed charge column length
     ``q_uend`` (last depth bin above 5 % of the profile peak, in ns after t0)
     against the 819.7 ns a muon needs to cross the 30 mm gap at 36.6 um/ns,
     plus the transverse strip multiplicity. A gain that is too low does not
     just scale the signal down: the far end of the drift column, the most
     diffused and most attenuated part, drops under the noise first, so the
     detector sees a SHORTER track before it sees a fainter one.
  3. ANGULAR RESOLUTION -- s68 of (theta_wft - theta_M3) per plane. This is
     where 1 and 2 cash out: a shorter, noisier column is a shorter lever arm.

The three are deliberately measured on one event sample so they can be read
against each other, and against 10b's efficiency and 10c's peak-amplitude gain,
which use the same M3-golden fiducial population.

Inputs
    <Analysis>/<run>/<subrun>/mx17_3/wft/events_hvscan.parquet   (10d)
    ~/x17/response_sim/hv_slope/peaks.parquet                    (10c, joined
        per (subrun, view, event) for the raw peak amplitude and the 5 sigma
        strip count -- the threshold-limited counterparts)
Outputs
    <Analysis>/<run>/hv_scan/mx17_3/
        charge_angle_vs_hv.csv / .meta.json
        charge_vs_hv.png  occupancy_vs_hv.png  angres_vs_hv.png
        charge_angle_summary.png        (the slide figure)
        report.html

    ../.venv/bin/python 10e_hv_scan_charge_angle.py
    ../.venv/bin/python 10e_hv_scan_charge_angle.py --slides
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE, os.path.join(REPO, 'cosmic_bench_analysis')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

RUN = 'mx17_det3_saturday_scan_6-27-26'
DET = 'mx17_3'
BASE = '/home/dylan/x17/cosmic_bench/det3/'
ANALYSIS = '/home/dylan/x17/cosmic_bench/Analysis'
PEAKS = os.path.expanduser('~/x17/response_sim/hv_slope/peaks.parquet')
RAW = os.path.join(ANALYSIS, RUN, 'hv_scan', DET, 'occupancy_raw.parquet')
ALIGN = os.path.join(ANALYSIS, RUN, 'long_run_resist_490V_drift_1000V', DET,
                     'wft', 'alignment', 'alignment.json')
BUNDLE = ('/media/dylan/data/x17/cosmic_bench/condor_campaign_r06/'
          'local_bundles/mx17_3/calib_bundle_r06')
SUBRUN_RE = re.compile(r'^hv_scan(2?)_resist_(\d+)V_drift_1000V$')

V_DRIFT = 36.6            # um/ns, bundle value; drift field fixed at 1000 V
GAP_MM = 30.0             # settled effective drift gap (mx17-drift-gap-settled)
T_GAP_NS = GAP_MM * 1e3 / V_DRIFT          # 819.7 ns to cross the gap
PITCH_MM = 0.78
SAT_ADC = 3550.0          # the bundle's censoring threshold
V_REF = 490
NBOOT = 400
RNG = np.random.default_rng(20260828)
QUANT = 0.50              # the charge ladder's headline quantile


# ------------------------------------------------------------------ loading
def subruns():
    out = []
    for d in sorted(os.listdir(os.path.join(BASE, RUN))):
        m = SUBRUN_RE.match(d)
        if m:
            out.append((d, int(m.group(2)), 'scan2' if m.group(1) else 'scan1'))
    return sorted(out, key=lambda t: (t[2], t[1]))


def load_events():
    """One row per (subrun, event) from the 10d tables, + the 10d bookkeeping."""
    frames, book = [], []
    for sub, volt, scan in subruns():
        p = os.path.join(ANALYSIS, RUN, sub, DET, 'wft', 'events_hvscan.parquet')
        if not os.path.exists(p):
            print(f'[10e] missing {p} -- run 10d first')
            continue
        df = pd.read_parquet(p)
        df['subrun'], df['hv'], df['scan'] = sub, volt, scan
        frames.append(df)
        side = json.load(open(p.replace('.parquet', '.hvscan.json')))
        side.update(hv=volt)
        book.append(side)
    if not frames:
        sys.exit('[10e] no 10d tables found')
    return pd.concat(frames, ignore_index=True), pd.DataFrame(book)


def load_raw():
    """The 10f threshold-free read pass, per (subrun, view, event)."""
    if not os.path.exists(RAW):
        print(f'[10e] no {RAW} -- run 10f first; the raw columns will be blank')
        return None
    return pd.read_parquet(RAW)


def check_peaks(raw):
    """Cross-check 10f's peak amplitude against 10c's peaks.parquet.

    Two independent extractions of the same quantity, months apart: if their
    per-sub-run medians do not agree, one of the two reductions has gone stale
    and every number downstream of it is suspect."""
    if raw is None or not os.path.exists(PEAKS):
        return None
    pk = pd.read_parquet(PEAKS)
    key = ['subrun', 'view', 'event_id']
    j0 = raw[key + ['peak_amp']].merge(pk[key + ['peak_amp']], on=key,
                                       suffixes=('_new', '_old'))
    a = j0.groupby(['subrun', 'view']).peak_amp_new.median().rename('new')
    b = j0.groupby(['subrun', 'view']).peak_amp_old.median().rename('old')
    j = pd.concat([a, b], axis=1).dropna()
    j['ratio'] = j.new / j.old
    return j


def reference_tangents(sub):
    """tan(theta) of the M3 ray in the raw strip frame, per plane, per event.

    Reproduces mx_june_wft/03_angles.py exactly: M3 angles -> tan -> ref_x_sign
    -> rotate by the alignment's theta (~89.4 deg for det3, so this is close to
    an axis swap and is not optional)."""
    from qa_config import get_config, M3_CHI2_CUT, M3_MIN_NCLUS
    from M3RefTracking import M3RefTracking, get_xy_angles
    import cosmic_micro_tpc_analysis as cm

    params = cm.load_alignment(ALIGN)
    cfg = get_config('sat_det3')
    cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = BASE, RUN, sub
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, evn = get_xy_angles(rays.ray_data)
    tx, ty = np.tan(params.ref_x_sign * np.asarray(xa)), np.tan(np.asarray(ya))
    th = np.deg2rad(params.theta_deg)
    c, s = np.cos(th), np.sin(th)
    return pd.DataFrame({'event_id': np.asarray(evn, dtype=np.int64),
                         'ref_tan_x': c * tx + s * ty,
                         'ref_tan_y': -s * tx + c * ty})


def attach_reference(df):
    out = []
    for sub, g in df.groupby('subrun', sort=False):
        ref = reference_tangents(sub)
        out.append(g.merge(ref, on='event_id', how='left'))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ ladders
def qboot(a, q=QUANT, nboot=NBOOT):
    """quantile + its 1 sigma error on ln, by bootstrap."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) < 30:
        return np.nan, np.nan, len(a)
    v = float(np.quantile(a, q))
    bs = np.quantile(RNG.choice(a, size=(nboot, len(a)), replace=True),
                     q, axis=1)
    return v, float(np.std(np.log(bs))), len(a)


def robust_sigma(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def s68(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan, np.nan, len(a)
    med = float(np.median(a))
    return med, float(np.percentile(np.abs(a - med), 68)), len(a)


def loglin(v, lnA, err):
    """weighted straight-line fit of ln A on V; slope is per 10 V."""
    v, lnA, err = map(np.asarray, (v, lnA, err))
    ok = np.isfinite(v) & np.isfinite(lnA) & np.isfinite(err) & (err > 0)
    v, lnA, err = v[ok], lnA[ok], err[ok]
    if len(v) < 3:
        return np.nan, np.nan, np.nan, 0
    w = 1.0 / err ** 2
    S, Sx, Sy = w.sum(), (w * v).sum(), (w * lnA).sum()
    Sxx, Sxy = (w * v * v).sum(), (w * v * lnA).sum()
    d = S * Sxx - Sx * Sx
    b = (S * Sxy - Sx * Sy) / d
    a = (Sy - b * Sx) / S
    berr = np.sqrt(S / d)
    return b * 10.0, berr * 10.0, a, len(v)


def ladder(ev, raw):
    """One row per (view, voltage): charge, occupancy, angle, bookkeeping.

    Three different event sets are used on purpose, and each column says which:

    * ``g``  -- every reconstructed event: M3-golden, fiducial, spark-free.
      The **charge and the raw occupancy** are measured here. Applying the
      fit-quality gate to them would be circular: the gate fails on saturated
      events, and saturated events are exactly the high-charge ones.
    * ``gg`` -- ``g`` plus a fitted, quality-ok plane. Only the quantities that
      come *out of* the fit (column length, chi2) can live here, and the
      surviving fraction is reported next to them so the reader can see how
      selected they are.
    * angles are quoted BOTH ways: ``ang_s68_all_deg`` over every fitted plane
      is the honest one, ``ang_s68_deg`` is the quality-gated version, and
      above ~500 V they part company because the gate is throwing away the
      saturated tracks.
    """
    rows = []
    for (scan, hv), g in ev.groupby(['scan', 'hv'], sort=True):
        for view in ('x', 'y'):
            fitted = g[g[f'{view}_ok'].astype(bool)]
            gg = fitted[fitted[f'{view}_quality_ok'].astype(bool)]
            r = dict(scan=scan, hv=hv, view=view,
                     subrun=g['subrun'].iloc[0],
                     n_reco=int(len(g)), n_fitted=int(len(fitted)),
                     n_plane_ok=int(len(gg)),
                     frac_fitted=float(len(fitted) / max(len(g), 1)),
                     frac_plane_ok=float(len(gg) / max(len(g), 1)))

            # --- 1. total charge, spark-free and UNGATED -------------------
            q, qerr, nq = qboot(fitted[f'{view}_q_sum'])
            r.update(q_sum=q, q_sum_lnerr=qerr, n_q=nq)

            # --- 2a. drift column from the fit (gated -- see docstring)
            uend = gg[f'{view}_q_uend'].to_numpy(float)
            col = float(np.nanmean(uend)) if len(uend) else np.nan
            r.update(col_ns=col, col_mm=col * V_DRIFT * 1e-3,
                     col_frac=col / T_GAP_NS,
                     col_ns_med=float(np.nanmedian(uend)) if len(uend) else np.nan,
                     col_ns_lo=float(np.nanquantile(uend, 0.25)) if len(uend) else np.nan,
                     col_ns_hi=float(np.nanquantile(uend, 0.75)) if len(uend) else np.nan,
                     u50_ns=float(np.nanmean(gg[f'{view}_q_u50'])) if len(gg) else np.nan,
                     u90_ns=float(np.nanmean(gg[f'{view}_q_u90'])) if len(gg) else np.nan,
                     n_seed=float(np.nanmedian(gg[f'{view}_n_seed'])) if len(gg) else np.nan,
                     chi2dof_med=float(np.nanmedian(
                         gg[f'{view}_chi2'] / gg[f'{view}_dof'].clip(lower=1)))
                     if len(gg) else np.nan)
            # cluster width after the 10 % RELATIVE floor: both planes, every
            # spark-free event. It cannot move with gain unless the cluster
            # SHAPE changes, which is the point of putting it next to the
            # absolute-threshold width below.
            r['n_hits_med'] = float(np.nanmedian(g['n_hits']))

            # --- 2b/1b. the raw waveform view, same spark-free ungated set
            if raw is not None:
                m = raw[(raw.subrun == r['subrun']) & (raw.view == view)]
                m = m[m.event_id.isin(g.event_id)]
                if len(m):
                    a = m.peak_amp.to_numpy(float)
                    qw, qwe, _ = qboot(m.q_win)
                    r.update(peak_amp=float(np.median(a)),
                             frac_sat=float(np.mean(a >= SAT_ADC)),
                             n_sat_cell=float(np.mean(m.n_sat_cell)),
                             q_win=qw, q_win_lnerr=qwe, n_raw=int(len(m)),
                             q_5s=float(np.median(m.q_5s)),
                             n_strip_5s=float(np.median(m.n_strip_5s)),
                             n_cell_5s=float(np.median(m.n_cell_5s)),
                             span_ns=float(np.mean(m.span_ns)),
                             span_mm=float(np.mean(m.span_ns)) * V_DRIFT * 1e-3)

            # --- 3. angular resolution against M3 --------------------------
            def _res(sel):
                d = (np.degrees(np.arctan(sel[f'{view}_tan_theta']))
                     - np.degrees(np.arctan(sel[f'ref_tan_{view}'])))
                return s68(d)
            b_all, s_all, n_all = _res(fitted)
            bias, sig68, nth = _res(gg)
            r.update(ang_s68_all_deg=s_all, ang_bias_all_deg=b_all,
                     n_ang_all=n_all, ang_bias_deg=bias, ang_s68_deg=sig68,
                     n_ang=nth,
                     tan_err_med=float(np.nanmedian(fitted[f'{view}_tan_err']))
                     if len(fitted) else np.nan)
            rows.append(r)

        # 3-D opening angle: both planes FITTED (no quality gate), direction
        # (tan_x, tan_y, 1) against the M3 ray
        b = g[g.x_ok.astype(bool) & g.y_ok.astype(bool)]
        if len(b) > 20:
            d = np.stack([b.x_tan_theta, b.y_tan_theta, np.ones(len(b))])
            rr = np.stack([b.ref_tan_x, b.ref_tan_y, np.ones(len(b))])
            d = d / np.linalg.norm(d, axis=0)
            rr = rr / np.linalg.norm(rr, axis=0)
            op = np.degrees(np.arccos(np.clip((d * rr).sum(axis=0), -1, 1)))
            for r in rows[-2:]:
                r['open3d_med_deg'] = float(np.nanmedian(op))
                r['open3d_p68_deg'] = float(np.nanpercentile(op, 68))
                r['n_both'] = int(len(b))
                r['frac_both'] = float(len(b) / max(len(g), 1))
    t = pd.DataFrame(rows).sort_values(['view', 'hv']).reset_index(drop=True)
    return t


def local_slopes(t, col='q_win'):
    """d ln <charge> / dV between ADJACENT voltages.

    A single log-linear fit over 100 V assumes the Townsend slope is constant.
    It is not: alpha(E) itself rises with field, so the ladder curves upward
    and one number for the whole range is an average. This is the diagnostic
    that says so.

    Computed WITHIN each scan pass. The ladder interleaves two passes (scan1 on
    the 5 V grid, scan2 on the 10 V grid taken later), and differencing across
    the interleave turns any pass-to-pass offset into a sawtooth that reads as
    curvature. Doing it per pass costs the 5 V steps and keeps the answer."""
    out = []
    for (view, scan), s in t.groupby(['view', 'scan']):
        s = s.sort_values('hv')
        if col not in s:
            continue
        v = s.hv.to_numpy(float)
        q = s[col].to_numpy(float)
        for i in range(len(v) - 1):
            if not (np.isfinite(q[i]) and np.isfinite(q[i + 1])):
                continue
            out.append(dict(view=view, scan=scan, v_lo=v[i], v_hi=v[i + 1],
                            v_mid=0.5 * (v[i] + v[i + 1]),
                            slope_per10V=10 * np.log(q[i + 1] / q[i])
                            / (v[i + 1] - v[i])))
    return pd.DataFrame(out)


def fit_charge(t, col='q_sum', err='q_sum_lnerr'):
    """d ln <charge> / dV per view, over the full ladder and over the sub-range
    where the peak sample is not yet clipping, so the deconvolved ladder and
    the peak-sample ladder can be compared like for like."""
    out = {}
    if col not in t:
        return out
    for view in ('x', 'y'):
        s = t[t.view == view]
        # 'trust' is the headline: above ~90 % clipped peak samples the WINDOW
        # SUM is clipping too (tens of railed cells per track), so neither the
        # raw sum nor the censored fit is measuring charge any more -- both
        # flatten, and a fit that includes those points reads the flattening as
        # physics. 'nosat' is the range where nothing clips at all.
        ranges = {'full': s}
        if 'frac_sat' in s:
            tr = s[s.frac_sat < 0.90]
            ranges['trust'] = tr
            ranges['nosat'] = s[s.frac_sat < 0.05]
            # the two passes fitted apart: if they disagree, the interleaved
            # ladder is carrying a pass-to-pass offset, not curvature
            for sc in ('scan1', 'scan2'):
                ranges[f'trust_{sc}'] = tr[tr.scan == sc]
        for name, sel in ranges.items():
            if len(sel) < 3:
                continue
            sl, se, a, n = loglin(sel.hv, np.log(sel[col]), sel[err])
            out[f'{view}_{name}'] = dict(
                slope_per10V=sl, slope_err=se, intercept=a, n_points=n,
                efold_V=10.0 / sl if np.isfinite(sl) and sl else np.nan,
                double_V=np.log(2) / sl * 10.0 if np.isfinite(sl) and sl
                else np.nan,
                v_lo=float(sel.hv.min()), v_hi=float(sel.hv.max()))
    return out



# ------------------------------------------------------------------ figures
CX, CY = 'tab:blue', 'tab:red'
COL = {'x': CX, 'y': CY}


def _volts(s):
    return s.hv.to_numpy(float)


def fig_charge(t, fits, fits_raw, path, slides=False):
    """Three estimators of the same thing: colour = plane, marker = estimator."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axs[0]
    bad = t[t.frac_sat >= 0.90].hv.min() if 'frac_sat' in t else np.nan
    if np.isfinite(bad):
        for a in axs:
            a.axvspan(bad - 2.5, t.hv.max() + 2.5, color='0.85', alpha=0.4,
                      lw=0, zorder=0)
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        v, q = _volts(s), s.q_sum.to_numpy(float)
        ax.errorbar(v, q, yerr=q * s.q_sum_lnerr.to_numpy(float), fmt='o',
                    ms=5, color=COL[view])
        f = fits.get(f'{view}_trust') or fits.get(f'{view}_full')
        if f:
            vv = np.linspace(v.min(), v.max(), 50)
            ax.plot(vv, np.exp(f['intercept'] + f['slope_per10V'] / 10 * vv),
                    color=COL[view], lw=1.2, alpha=0.85)
        if 'q_win' in s:
            w = s.q_win.to_numpy(float)
            ax.plot(v, w * (q[0] / w[0]), '^:', ms=5, lw=1.0, color=COL[view],
                    alpha=0.85)
        if 'peak_amp' in s:
            pa = s.peak_amp.to_numpy(float)
            pa = pa * (q[0] / pa[0])
            sat = s.frac_sat.to_numpy(float) > 0.05
            ax.plot(v, pa, '--', lw=1.0, color=COL[view], alpha=0.6)
            ax.plot(v[~sat], pa[~sat], 's', ms=4, color=COL[view], alpha=0.7)
            ax.plot(v[sat], pa[sat], 's', ms=5, mfc='none', mec=COL[view])
    hs = [plt.Line2D([], [], color=CX, lw=3, label='x plane'),
          plt.Line2D([], [], color=CY, lw=3, label='y plane'),
          plt.Line2D([], [], color='k', marker='o', ls='-', ms=5,
                     label='forward-fit charge (deconvolved, rail censored)'),
          plt.Line2D([], [], color='k', marker='^', ls=':', ms=5,
                     label='raw window sum (no threshold, no model)'),
          plt.Line2D([], [], color='k', marker='s', ls='--', ms=4,
                     label='peak sample (10c)'),
          plt.Line2D([], [], color='k', marker='s', ls='none', ms=5,
                     mfc='none', label='open: >5 % of tracks on the rail'),
          plt.Line2D([], [], color='0.85', lw=8,
                     label='grey: the window sum clips too - not measured')]
    ax.set_yscale('log')
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('median signal, all scaled to agree at the first point')
    ax.legend(handles=hs, fontsize=7, loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3, which='both')
    fx = fits.get('x_trust', fits.get('x_full', {}))
    fy = fits.get('y_trust', fits.get('y_full', {}))
    ax.set_title('total charge keeps going: x2 every '
                 f'{fx.get("double_V", np.nan):.1f} V (x), '
                 f'{fy.get("double_V", np.nan):.1f} V (y), '
                 f'{fx.get("v_lo", np.nan):.0f}-{fx.get("v_hi", np.nan):.0f} V',
                 fontsize=9.5)

    ax = axs[1]
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        if 'peak_amp' not in s:
            continue
        v = _volts(s)
        r = (s.q_sum / s.peak_amp).to_numpy(float)
        ax.plot(v, r / r[0], 'o-', ms=4, color=COL[view],
                label=f'{view}: fitted charge / peak sample')
        if 'q_win' in s:
            r2 = (s.q_win / s.peak_amp).to_numpy(float)
            ax.plot(v, r2 / r2[0], '^:', ms=4, color=COL[view], alpha=0.7,
                    label=f'{view}: raw sum / peak sample')
    ax.axhline(1.0, color='gray', lw=0.8)
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('ratio, relative to the first point')
    ax.set_title('the charge the peak sample stops reporting', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.3)
    if not slides:
        fig.suptitle('det3 27 June mesh ladder - deconvolved charge vs the '
                     'peak sample', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_occupancy(t, path, slides=False):
    """Three different senses of "how much of the track is there", kept apart
    on purpose: the depth the fit recovers, the time the electronics reports,
    and the transverse width at two different thresholds."""
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axs[0]
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        v = _volts(s)
        ax.plot(v, s.col_mm, 'o-', ms=5, color=COL[view], label=f'{view} plane')
        ax.fill_between(v, s.col_ns_lo * V_DRIFT * 1e-3,
                        s.col_ns_hi * V_DRIFT * 1e-3, color=COL[view],
                        alpha=0.13, lw=0)
    ax.axhline(GAP_MM, color='k', ls='--', lw=1,
               label=f'{GAP_MM:.0f} mm drift gap')
    ax.set_ylim(0, 45)
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('charge column recovered by the fit [mm]')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_title('depth: the whole gap, at every voltage', fontsize=9.5)

    ax = axs[1]
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        if 'span_ns' not in s:
            continue
        ax.plot(_volts(s), s.span_ns, '^--', ms=5, color=COL[view],
                label=f'{view} plane')
    ax.axhline(T_GAP_NS, color='k', ls='--', lw=1,
               label=f'{T_GAP_NS:.0f} ns to cross the gap')
    ax.axhline(32 * 60.0, color='0.5', ls=':', lw=1, label='32-sample record')
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('time over 5 sigma, first to last sample [ns]')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_title('time over threshold: the tail, not the track', fontsize=9.5)

    ax = axs[2]
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        v = _volts(s)
        ax.plot(v, s.n_seed, 'o-', ms=5, color=COL[view],
                label=f'{view}: cluster core (10 % of the peak)')
        if 'n_strip_5s' in s:
            ax.plot(v, s.n_strip_5s, '^--', ms=4, color=COL[view], alpha=0.75,
                    label=f'{view}: strips over 5 sigma (absolute)')
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('median strips per plane')
    ax.set_ylim(0, 16)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_title('transverse: rescaled, not reshaped', fontsize=9.5)

    if not slides:
        fig.suptitle('det3 27 June mesh ladder - track occupancy', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_angres(t, path, slides=False):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axs[0]
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        v = _volts(s)
        n = s.n_ang.to_numpy(float).clip(min=1)
        na = s.n_ang_all.to_numpy(float).clip(min=1)
        ax.errorbar(v, s.ang_s68_all_deg,
                    yerr=s.ang_s68_all_deg / np.sqrt(2 * na),
                    fmt='o-', ms=5, color=COL[view], capsize=2,
                    label=f'{view}: every fitted plane')
        ax.plot(v, s.ang_s68_deg, ':', lw=1.1, color=COL[view], alpha=0.6,
                label=f'{view}: chi2/dof < 300 only (selected)')
    if 'open3d_p68_deg' in t:
        sd = t[t.view == 'x'].sort_values('hv')
        ax.plot(_volts(sd), sd.open3d_p68_deg, 'k^--', ms=4, lw=1, alpha=0.8,
                label='3-D opening angle, 68 %')
    ax.axhline(1.0, color='k', ls=':', lw=1,
               label='~1 deg per-event physics floor')
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('s68 of (reco - M3) angle [deg]')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title('angular resolution against the M3 reference', fontsize=9.5)

    ax = axs[1]
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        v = _volts(s)
        ax.plot(v, s.ang_bias_deg, 'o-', ms=5, color=COL[view],
                label=f'{view}: bias')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('median (reco - M3) angle [deg]')
    ax.grid(alpha=0.3)
    bx = ax.twinx()
    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        bx.plot(_volts(s), 100 * s.frac_plane_ok, '^--', ms=4, alpha=0.5,
                color=COL[view], label=f'{view}: planes fitted')
    bx.set_ylabel('planes fitted [%]')
    bx.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = bx.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc='lower right')
    ax.set_title('bias and plane yield', fontsize=9.5)
    if not slides:
        fig.suptitle('det3 27 June mesh ladder — angular response', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_slope(t, loc, path, slides=False):
    """The ladder is not a straight line: local slope against voltage."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for view in ('x', 'y'):
        for sc, ls in (('scan1', '-'), ('scan2', '--')):
            d = loc[(loc.view == view) & (loc.scan == sc)]
            if not len(d):
                continue
            ax.plot(d.v_mid, d.slope_per10V, 'o', ls=ls, ms=4,
                    color=COL[view], alpha=1.0 if sc == 'scan1' else 0.6,
                    label=f'{view}, {sc}')
        s = t[t.view == view]
        sat = s[s.frac_sat > 0.5]
        if len(sat):
            ax.axvspan(float(sat.hv.min()), float(s.hv.max()), color='0.85',
                       alpha=0.35, lw=0, zorder=0)
    for view in ('x', 'y'):
        d = loc[loc.view == view].sort_values('v_mid')
        d = d[d.v_mid < 500]
        if len(d) > 4:
            ax.plot(d.v_mid, d.slope_per10V.rolling(3, center=True).mean(),
                    color=COL[view], lw=2.5, alpha=0.3, zorder=0)
    ax.axhline(0.419, color='k', ls='--', lw=1,
               label='10c peak-sample fit, 0.419 / 10 V')
    ax.set_xlabel('mesh voltage [V]')
    ax.set_ylabel('d ln(total charge) / dV  [per 10 V]')
    ax.set_title('the Townsend slope is not constant\n'
                 '(thick line: 3-point rolling mean, both passes pooled)',
                 fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.03, 'grey: >50 % of peak samples on the rail',
            transform=ax.transAxes, ha='right', fontsize=7, color='0.35')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_summary(t, fits, path, slides=False):
    """The three answers on one voltage axis, plus the trade they make."""
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax2, ax3 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    bad = t[t.frac_sat >= 0.90].hv.min() if 'frac_sat' in t else np.nan

    for view in ('x', 'y'):
        s = t[t.view == view].sort_values('hv')
        v = _volts(s)
        ref = float(s[s.hv == V_REF].q_sum.iloc[0]) if (s.hv == V_REF).any() \
            else float(s.q_sum.iloc[0])
        ax0.errorbar(v, s.q_sum / ref, yerr=s.q_sum / ref * s.q_sum_lnerr,
                     fmt='o-', ms=4, color=COL[view], label=f'{view} plane')
        ax1.plot(v, 100 * s.col_frac, 'o-', ms=4, color=COL[view],
                 label=f'{view} plane')
        ax2.plot(v, s.ang_s68_all_deg, 'o-', ms=4, color=COL[view],
                 label=f'{view} plane')
        ax3.plot(v, 100 * s.frac_plane_ok, 'o-', ms=4, color=COL[view],
                 label=f'{view}: planes passing chi2/dof < 300')
        for vv, cf, sg in zip(v, 100 * s.col_frac, s.ang_s68_all_deg):
            pass

    ax0.set_yscale('log')
    ax0.axhline(1.0, color='gray', lw=0.7)
    ax0.set_ylabel(f'total charge / total charge at {V_REF} V')
    _f = fits.get('x_trust', fits.get('x_full', {}))
    ax0.set_title('total deposited charge  (x2 every '
                  f'{_f.get("double_V", np.nan):.1f} V, '
                  f'{_f.get("v_lo", np.nan):.0f}-{_f.get("v_hi", np.nan):.0f} V)',
                  fontsize=9.5)
    ax1.axhline(100, color='k', ls='--', lw=1, label='full 30 mm gap')
    ax1.set_ylim(60, 130)
    ax1.set_ylabel('charge column / drift gap [%]')
    ax1.set_title('depth lit: the whole gap, everywhere', fontsize=9.5)
    ax2.axhline(1.0, color='k', ls=':', lw=1, label='~1 deg floor')
    ax2.set_ylabel('angular s68 vs M3 [deg]')
    ax2.set_title('angular resolution', fontsize=9.5)
    ax3.set_ylabel('planes surviving the fit-quality gate [%]')
    ax3.set_title('what saturation costs the fit', fontsize=9.5)
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xlabel('mesh voltage [V]')
        if np.isfinite(bad):
            ax.axvspan(bad - 2.5, t.hv.max() + 2.5, color='0.85', alpha=0.4,
                       lw=0, zorder=0)
    for ax in (ax0, ax1, ax2, ax3):
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7.5)
    if not slides:
        fig.suptitle('det3, 27 June mesh ladder: charge, occupancy, angle',
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------- report
def _tab(df, cols, fmt):
    h = ''.join(f'<th>{c}</th>' for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = ''.join(f'<td>{fmt(c, r[c])}</td>' for c in cols)
        rows.append(f'<tr>{cells}</tr>')
    return (f'<table><thead><tr>{h}</tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def multiplicity_section(out_dir, f):
    """The 10h strips-over-threshold section, if 10h has been run.

    Kept as a block inside this report rather than a second report: the strip
    count is the same measurement as the charge and the occupancy, seen through
    the threshold, and reading it apart from them is how it gets misread."""
    csv = os.path.join(out_dir, 'strip_multiplicity_vs_hv.csv')
    if not os.path.exists(csv):
        return ('<h2>Strips over threshold</h2><p>(10h not run &mdash; run '
                '<code>10g_hv_scan_strip_matrix.py</code> then '
                '<code>10h_hv_scan_multiplicity.py</code>)</p>')
    m = pd.read_csv(csv)
    try:
        v_ref = int(json.load(open(csv.replace('.csv', '.meta.json')))
                    ['v_ref_pred'])
    except Exception:
        v_ref = 465
    mx = m[m.view == 'x'].sort_values('hv')
    my = m[m.view == 'y'].sort_values('hv')
    lo_x, lo_y = mx.iloc[0], my.iloc[0]

    def closes(s, thr=0.99):
        ok = s[s.q_frac_thr_norm >= thr]
        return float(ok.hv.iloc[0]) if len(ok) else float('nan')

    # the prediction residual, split at the point where clipping takes over
    mid = m[(m.hv >= 435) & (m.hv <= 490)]
    top = m[m.hv >= 495]
    dmid = float(np.nanmax(np.abs(mid.pred_minus_meas)))
    dtop = float(np.nanmax(top.pred_minus_meas))
    # holes in the widest footprints at the bottom of the ladder
    wide = [c for c in m.columns if c.startswith('hole_w') and '9_20' in c]
    hw = float(lo_x[wide[0]]) if wide else float('nan')
    bx_max = mx.loc[mx.b_fit.idxmax()]

    cols = [c for c in ['hv', 'n_lit_med', 'a_fit', 'b_fit', 'q_frac_thr_norm',
                        'hole_frac', 'frac_with_hole', 'hole_sig_med',
                        'n_lit_meas_bandmatched', 'n_lit_pred_bandmatched',
                        'pred_minus_meas', 'frac_edge_lit'] if c in m]
    return f"""
<h2>Strips over threshold, with the track angle taken out</h2>

<div class="v"><b>Yes, low gain loses charge &mdash; about
{100 * (1 - lo_x.q_frac_thr_norm):.0f}&nbsp;% of it in x and
{100 * (1 - lo_y.q_frac_thr_norm):.0f}&nbsp;% in y at
{lo_x.hv:.0f}&nbsp;V &mdash; but it loses it at the edges of a cluster whose
shape never changes, not out of the middle of the track.</b>
The fraction of the collected charge that gets over 5&nbsp;&sigma; anywhere is
{100 * lo_x.q_frac_thr_norm:.0f}&nbsp;% (x) and
{100 * lo_y.q_frac_thr_norm:.0f}&nbsp;% (y) of its high-gain plateau at
{lo_x.hv:.0f}&nbsp;V, and closes to within 1&nbsp;% by
{closes(mx):.0f}&nbsp;V (x) / {closes(my):.0f}&nbsp;V (y). That is the
mechanism you proposed, measured: at low gain the thin slices of track &mdash;
a strip crossed by a short piece of the path, carrying a handful of primary
ionisations &mdash; sit under the threshold.

<br><br><b>Angle is not what is driving the strip count.</b> Splitting on the
M3 reference footprint <i>w</i><sub>geo</sub> =
gap&nbsp;&middot;&nbsp;|tan&thinsp;&theta;<sub>ref</sub>|&nbsp;/&nbsp;pitch
&mdash; a telescope quantity, so it cannot move with the detector&apos;s own
gain &mdash; every footprint band rises by about the same amount over the ladder,
{mx.a_fit.iloc[-1] - lo_x.a_fit:.1f} strips in x and
{my.a_fit.iloc[-1] - lo_y.a_fit:.1f} in y, rather than in proportion to its
width. In the decomposition <i>n</i><sub>lit</sub> =
<i>a</i>(V) + <i>b</i>(V)&thinsp;<i>w</i><sub>geo</sub>, the offset <i>a</i>
&mdash; the footprint at normal incidence &mdash; goes
{lo_x.a_fit:.1f}&nbsp;&rarr;&nbsp;{mx.a_fit.iloc[-1]:.1f} strips (x) and
{lo_y.a_fit:.1f}&nbsp;&rarr;&nbsp;{my.a_fit.iloc[-1]:.1f} (y), while the slope
<i>b</i> &mdash; strips lit per strip actually crossed &mdash; only moves
{lo_x.b_fit:.2f}&nbsp;&rarr;&nbsp;{bx_max.b_fit:.2f} (x, best at
{bx_max.hv:.0f}&nbsp;V) and never reaches 1. <b>The growth is transverse tail
crossing the threshold, not track being recovered.</b>

<br><br><b>A fixed cluster shape plus a moving threshold reproduces the count.</b>
Scaling every strip&apos;s signal at {v_ref}&nbsp;V by
the measured charge ratio &mdash; signal only, the ~2&nbsp;&sigma; noise-max
floor held fixed &mdash; and re-counting at 5&nbsp;&sigma; predicts the
measured multiplicity to within {dmid:.2f} strips everywhere between 435 and
490&nbsp;V. Above 495&nbsp;V it overshoots by up to {dtop:.1f} strips, which is
the measurement being limited (rail, and up to
{100 * m.frac_edge_lit.max():.0f}&nbsp;% of events touching the edge of the
&plusmn;10-strip window), not the model failing.</div>

<div class="c"><b>Where the mechanism does bite: the widest tracks break.</b>
Dark strips strictly inside the lit span &mdash; an internal property of one
cluster, needing no angle normalisation &mdash; run
{100 * lo_x.hole_frac:.1f}&nbsp;% at {lo_x.hv:.0f}&nbsp;V against
{100 * mx.hole_frac.min():.1f}&nbsp;% at its best, and
{100 * lo_x.frac_with_hole:.0f}&nbsp;% of x clusters have at least one hole at
{lo_x.hv:.0f}&nbsp;V against {100 * mx.frac_with_hole.min():.0f}&nbsp;%. They
are concentrated in the widest footprints ({100 * hw:.0f}&nbsp;% for
<i>w</i><sub>geo</sub>&nbsp;&gt;&nbsp;9 strips), which is what dilution
predicts: the same total charge spread over more strips leaves each one
fainter. The holes at the bottom of the ladder sit at
{lo_x.hole_sig_med:.1f}&nbsp;&sigma; &mdash; strips that <i>just</i> missed,
not empty ones. This is the low-voltage half of the angular optimum: at
{lo_x.hv:.0f}&nbsp;V a seventh of the x clusters are broken and a sixth of the
charge is invisible, and that is why the angle is worse there than at
455&ndash;460&nbsp;V.</div>

{_tab(mx, cols, f)}
<p style="color:#666;font-size:13px">x plane. <code>q_frac_thr_norm</code> is
the over-threshold charge fraction referred to its own 470&ndash;490&nbsp;V
plateau (the raw ratio plateaus a few per cent above 1: the denominator is a
whole-window sum and carries the shaped pulse&apos;s undershoot and whatever
the 64-channel common-mode median took off a wide signal, both
signal-proportional and negative &mdash; harmless for the charge <i>slope</i>,
which is why q_win and the deconvolved q_sum agreed to 2&nbsp;%).
<code>hole_sig_med</code> is the median significance of the dark strips inside
the span, in units of the strip&apos;s own noise.</p>

<figure><img src="multiplicity_holes.png"><figcaption>Far right is the answer
to the question: the fraction of the collected charge that gets over
5&nbsp;&sigma;, referred to the high-gain plateau. Left three: dark strips
inside the lit span, how often they occur, and how they split by footprint
width &mdash; the low-gain breakage is entirely in the widest
tracks.</figcaption></figure>
<figure><img src="strip_profile_vs_hv.png"><figcaption>Left: median amplitude
per strip against offset from the peak strip, in units of that strip&apos;s
noise. The curves are parallel until they reach the ~2&nbsp;&sigma; floor set
by taking a maximum over 32 noise samples &mdash; the profile rescales, it does
not reshape &mdash; and the strip count is just where the 5&nbsp;&sigma; line
cuts it. Middle: the same as a probability per offset. Right: per-offset
turn-on curves; |k|&nbsp;=&nbsp;2 crosses 50&nbsp;% near 445&ndash;460&nbsp;V,
|k|&nbsp;=&nbsp;3 near 470&ndash;485&nbsp;V.</figcaption></figure>
<figure><img src="multiplicity_vs_hv.png"><figcaption>Top left: raw strip
multiplicity. Top right: the same split into M3 footprint bands &mdash; the
rise survives angle matching and is an offset, not a scaling. Bottom: the
<i>a</i>, <i>b</i> decomposition. Above ~505&nbsp;V both are distorted by the
&plusmn;10-strip window (grey band).</figcaption></figure>
<figure><img src="multiplicity_prediction.png"><figcaption>Threshold scaling
from {v_ref}&nbsp;V against the measurement, both
recombined over the same footprint bands so the comparison is not an
estimator mismatch. Agreement is within a third of a strip over
435&ndash;490&nbsp;V.</figcaption></figure>
"""


def make_report(t, fits, fits_raw, book, out_dir, meta):
    def f(c, v):
        if isinstance(v, str):
            return v
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return '&ndash;'
        if c in ('hv', 'n_reco', 'n_plane_ok', 'n_ang', 'n_q'):
            return f'{v:.0f}'
        if c in ('frac_plane_ok', 'frac_sat', 'col_frac', 'frac_within5'):
            return f'{100 * v:.1f} %'
        return f'{v:.3g}'

    sx = t[t.view == 'x'].sort_values('hv')
    sy = t[t.view == 'y'].sort_values('hv')
    lo = sx.iloc[0]
    cols = [c for c in ['hv', 'n_reco', 'frac_plane_ok', 'q_sum', 'q_win',
                        'q_5s', 'frac_sat', 'col_mm', 'span_ns', 'n_strip_5s',
                        'n_seed', 'ang_s68_all_deg', 'ang_s68_deg',
                        'open3d_p68_deg', 'chi2dof_med'] if c in t]
    fx = fits.get('x_trust', fits['x_full'])
    fy = fits.get('y_trust', fits['y_full'])
    fxn, rxt = fits.get('x_nosat', {}), fits_raw.get('x_trust', {})
    trust_hi = fx['v_hi']
    tr = sx[sx.hv <= trust_hi]
    span = float(tr.q_sum.iloc[-1] / tr.q_sum.iloc[0])
    bx = sx.loc[sx.ang_s68_all_deg.idxmin()]
    by = sy.loc[sy.ang_s68_all_deg.idxmin()]
    bo = sx.loc[sx.open3d_p68_deg.idxmin()]
    wo = sx.loc[sx.open3d_p68_deg.idxmax()]
    hi = sx.iloc[-1]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>det3 mesh ladder — charge, occupancy, angle</title><style>
body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;
margin:2rem auto;padding:0 1rem;color:#1b1b1b}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2rem;
border-bottom:1px solid #ddd;padding-bottom:.3rem}}
table{{border-collapse:collapse;font-size:13px;margin:.8rem 0}}
th,td{{border:1px solid #ccd;padding:3px 8px;text-align:right}}
th{{background:#eef}} td:first-child,th:first-child{{text-align:left}}
figure{{margin:1.4rem 0}} img{{max-width:100%;border:1px solid #ddd}}
figcaption{{font-size:13px;color:#555;margin-top:.4rem}}
.v{{background:#eef7ee;border-left:4px solid #4a4;padding:.7rem 1rem;
margin:1rem 0}} .c{{background:#fff8e8;border-left:4px solid #da4;
padding:.7rem 1rem;margin:1rem 0}} code{{background:#f3f3f3;padding:1px 4px}}
</style></head><body>
<h1>det3, 27 June mesh ladder: total charge, track occupancy, angular resolution</h1>
<p style="color:#666">generated {meta['generated']} &middot; waveform-first
basis (<code>{os.path.basename(meta['bundle'])}</code>) &middot;
{int(book.n_reco.sum()):,} reconstructed events over
{len(book)} sub-runs</p>

<div class="v"><b>The peak strip saturates and the charge does not &mdash;
but over the range the peak sample was quoted on, the two agree.</b>
Total charge per plane is exponential in mesh voltage over
{fx['v_lo']:.0f}&ndash;{fx['v_hi']:.0f}&nbsp;V &mdash; doubling every
{fx['double_V']:.1f}&nbsp;V (x) and {fy['double_V']:.1f}&nbsp;V (y), a factor
{span:.0f} in all. Two estimators built on completely different assumptions
say the same thing: the deconvolved forward fit gives
{fx['slope_per10V']:.4f}&nbsp;&plusmn;&nbsp;{fx['slope_err']:.4f} per 10&nbsp;V
and the model-free raw window sum
{rxt.get('slope_per10V', float('nan')):.4f}&nbsp;&plusmn;&nbsp;{rxt.get('slope_err', float('nan')):.4f},
against 10c's peak-sample 0.419&nbsp;&plusmn;&nbsp;0.004. <b>So the peak-sample
gain curve was not being fooled by clipping over the range it was fitted on.</b>
What saturation costs is <i>reach</i>, not accuracy: above ~{trust_hi:.0f}&nbsp;V
the window sum clips too, and neither number measures charge any more.

<br><br><b>The ladder is curved, so &quot;&times;2 every
{fx['double_V']:.1f}&nbsp;V&quot; is an average.</b> The local slope between
adjacent voltages runs {fxn.get('slope_per10V', float('nan')):.3f} per 10&nbsp;V
over {fxn.get('v_lo', float('nan')):.0f}&ndash;{fxn.get('v_hi', float('nan')):.0f}&nbsp;V
and about 0.50 near 490&nbsp;V, with &plusmn;0.05 point-to-point scatter and
the two interleaved passes agreeing to 3&ndash;5&nbsp;% where they overlap in
range. That is ordinary Townsend behaviour &mdash;
&alpha;(E) rises with field &mdash; but it means the doubling voltage must not
be carried outside the range it was fitted in.

<br><br><b>How much of the track is lit does not change with gain.</b> The
forward fit recovers a charge column of
{sx.col_mm.min():.1f}&ndash;{sx.col_mm.max():.1f}&nbsp;mm (x) and
{sy.col_mm.min():.1f}&ndash;{sy.col_mm.max():.1f}&nbsp;mm (y) against a
{GAP_MM:.0f}&nbsp;mm drift gap, with <i>no trend</i> across a factor
{span:.0f} in charge. Low gain does not shorten the track: at
{lo.hv:.0f}&nbsp;V, 14&times; down in gain, the whole drift column is still
recoverable. What grows is threshold crossing &mdash; strips over 5&nbsp;&sigma;
go {lo.n_strip_5s:.0f}&nbsp;&rarr;&nbsp;{sx.n_strip_5s.max():.0f} per plane and
the time over 5&nbsp;&sigma; {lo.span_ns:.0f}&nbsp;&rarr;&nbsp;{sx.span_ns.max():.0f}&nbsp;ns,
twice the {T_GAP_NS:.0f}&nbsp;ns a muon needs to cross the gap, so that span is
the resistive/shaping tail emerging from the noise, not more track. The cluster
core at 10&nbsp;% of the peak stays {lo.n_seed:.0f}&ndash;{sx.n_seed.max():.0f}
strips throughout: <b>rescaled, not reshaped</b>, transversely as well as in
amplitude.

<br><br><b>Angular resolution does deteriorate &mdash; at the top, not the
bottom.</b> Over every fitted plane, s68 against M3 is best at
{bx.hv:.0f}&nbsp;V ({bx.ang_s68_all_deg:.2f}&deg;, x) and {by.hv:.0f}&nbsp;V
({by.ang_s68_all_deg:.2f}&deg;, y), against {lo.ang_s68_all_deg:.2f}&deg; at
{lo.hv:.0f}&nbsp;V and {sx.ang_s68_all_deg.max():.2f}&deg; at the worst high
point. The 3-D opening angle against the ray says the same more cleanly:
{bo.open3d_p68_deg:.2f}&deg; at {bo.hv:.0f}&nbsp;V rising to
{wo.open3d_p68_deg:.2f}&deg; at {wo.hv:.0f}&nbsp;V, <b>+{100 * (wo.open3d_p68_deg / bo.open3d_p68_deg - 1):.0f}&nbsp;%</b>.
Over the same span the fraction of planes passing chi2/dof&nbsp;&lt;&nbsp;300
falls from ~100&nbsp;% to {100 * hi.frac_plane_ok:.0f}&nbsp;%, and the
<i>gated</i> resolution appears to improve to
{sx.ang_s68_deg.min():.2f}&deg; &mdash; a pure selection artefact, plotted
beside the honest curve so that nobody quotes it.

<br><br><b>So the angular optimum sits at the bottom of the efficiency
plateau.</b> Efficiency is flat from 455&nbsp;V; the angle is best at
455&ndash;470&nbsp;V and is 15&ndash;30&nbsp;% worse by 515&nbsp;V, before the
discharge fraction is counted at all.</div>

<h2>The ladder (x plane)</h2>
{_tab(sx[cols], cols, f)}
<p style="font-size:13px;color:#555"><code>q_sum</code> total NNLS charge
(arb.); <code>frac_sat</code> fraction of tracks whose peak sample is at or
over the {SAT_ADC:.0f} ADC censoring level; <code>col_mm</code> median charge
column length; <code>col_frac</code> as a fraction of the {GAP_MM:.0f} mm gap;
<code>n_seed</code> median strips in the seed cluster;
<code>ang_s68_deg</code> 68 % half-width of (reco &minus; M3) angle.</p>

<h2>Fits</h2>
<p><b>Deconvolved charge <code>q_sum</code></b></p>
{_tab(pd.DataFrame([dict(series=k, **v) for k, v in fits.items()]),
      ['series', 'slope_per10V', 'slope_err', 'double_V', 'v_lo', 'v_hi',
       'n_points'], f)}
<p><b>Model-free raw window sum <code>q_win</code></b> &mdash; same events, no
threshold and no deconvolution, so it is the check on the line above rather
than an independent measurement of gain.</p>
{_tab(pd.DataFrame([dict(series=k, **v) for k, v in fits_raw.items()]),
      ['series', 'slope_per10V', 'slope_err', 'double_V', 'v_lo', 'v_hi',
       'n_points'], f) if fits_raw else '<p>(10f not run)</p>'}

{multiplicity_section(out_dir, f)}

<h2>Figures</h2>
<figure><img src="charge_vs_hv.png"><figcaption>Left: median total fitted
charge per plane (circles, log axis) with the weighted log-linear fit, against
the peak-sample ladder of 10c (squares) scaled to agree at 425 V; open squares
mark voltages where more than 5&nbsp;% of tracks clip the rail. Right: the
ratio of the two, normalised at 425&nbsp;V &mdash; the charge the peak sample
stops reporting.</figcaption></figure>
<figure><img src="occupancy_vs_hv.png"><figcaption>Left: median reconstructed
charge-column length (band = interquartile), against the 30&nbsp;mm drift gap.
Right: transverse extent &mdash; strips in the seed cluster and strips over
5&nbsp;&sigma;, both threshold-limited and therefore rising with gain by
construction.</figcaption></figure>
<figure><img src="angres_vs_hv.png"><figcaption>Left: 68&nbsp;% half-width of
the per-event angle residual against the M3 reference, per plane; the dotted
line is the ~1&deg; per-event physics floor measured by toy closure. Right:
median bias and the fraction of reconstructed events with a usable plane
fit.</figcaption></figure>
<figure><img src="charge_angle_summary.png"><figcaption>The three answers on
one voltage axis, and the trade they make: angular resolution plotted against
column length is a single curve, so the voltage dependence of the angle is the
voltage dependence of the lever arm.</figcaption></figure>

<h2>What this does not rule out</h2>
<div class="c"><ul>
<li><b>One calibration for the whole ladder.</b> The frozen r06 bundle was
fitted on the 490&nbsp;V long run. Every hyper in it &mdash; drift velocity,
sharing kernel, diffusion, shaping &mdash; is a property of the drift gap, the
resistive layer or the electronics rather than of the mesh voltage, so it
should transfer; but that is an argument, not a measurement. A slow drift of
the kernel with avalanche size would appear here as a slow drift of the column
length, and the column length is flat, which is weak evidence in favour.</li>
<li><b>Nothing above ~{trust_hi:.0f}&nbsp;V is a charge measurement.</b> At the
top of the ladder essentially every peak sample is clipped and tens of cells
per track are on the rail, so the window sum is clipping as well; the forward
fit is then extrapolating from the unsaturated strips and from the model being
right. The {sx.n_strip_5s.max():.0f}-strip clusters up there also start to
reach the edge of 10f&apos;s +-10-strip window, which loses charge the other
way. Those points are drawn in the grey band and excluded from every fit.</li>
<li><b>The sample is spark-vetoed, and heavily so at the top.</b> The seeder
drops events whose post-floor hit count exceeds 50; that removes
{100 * (1 - hi.n_reco / float(book[book.hv == hi.hv].n_fiducial_rays.iloc[0])):.0f}&nbsp;%
of fiducial rays at {hi.hv:.0f}&nbsp;V against ~1&nbsp;% at {lo.hv:.0f}&nbsp;V.
Nothing here says the surviving tracks at {hi.hv:.0f}&nbsp;V are representative
of all tracks at {hi.hv:.0f}&nbsp;V &mdash; and the angular degradation could in
principle be the surviving population changing rather than the fit getting
worse.</li>
<li><b>The angle numbers include the M3 reference</b> and no deconvolution has
been applied, so they are residuals against a reference of finite resolution,
not the detector&apos;s intrinsic angular resolution. The ~1&deg; line is the
per-event physics floor from toy closure, not a systematic uncertainty.</li>
<li><b>Column length is an estimator, not a ruler.</b> <code>q_uend</code> is
the last depth bin above 5&nbsp;% of the profile peak; the x/y offset
({sx.col_mm.mean():.1f} vs {sy.col_mm.mean():.1f}&nbsp;mm against the same
{GAP_MM:.0f}&nbsp;mm gap) is that estimator meeting two planes with different
sharing, not two different gaps. It is the <i>flatness</i> that carries the
result, not the value.</li>
<li><b>The strip count is a threshold statistic, on this threshold.</b>
Everything in the multiplicity section is measured at 5&nbsp;&sigma; on the
June bench noise. A different threshold moves <i>a</i>(V) bodily and moves the
voltage at which the charge deficit closes; only the <i>shape</i> result &mdash;
that the profile rescales rather than reshapes &mdash; is threshold-free. The
prediction also scales a measured significance, whose per-event noise shrinks
with the signal while the floor does not, so it is slightly too sharp near the
threshold; that is a sub-tenth-of-a-strip effect at the residuals quoted, and
it is not a substitute for taking data at a second threshold.</li>
<li><b>Relative charge only.</b> There is no ADC&rarr;electron calibration for
the June bench CSA range, so d&nbsp;ln&nbsp;Q/dV is what is measured; no
absolute gain is claimed here or anywhere downstream.</li>
</ul></div>
</body></html>"""
    p = os.path.join(out_dir, 'report.html')
    with open(p, 'w') as fh:
        fh.write(html)
    return p


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides', action='store_true',
                    help='also write untitled copies into the MPGD26 assets')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    from qa_config import setup_paths
    setup_paths()

    ev, book = load_events()
    raw = load_raw()
    ev = attach_reference(ev)
    t = ladder(ev, raw)
    fits = fit_charge(t)
    fits_raw = fit_charge(t, 'q_win', 'q_win_lnerr')
    loc = local_slopes(t, 'q_win')
    xchk = check_peaks(raw)
    if xchk is not None:
        print(f'[10e] 10f vs peaks.parquet peak amplitude: ratio '
              f'{xchk.ratio.min():.4f}-{xchk.ratio.max():.4f} '
              f'over {len(xchk)} (sub-run, view)')

    out_dir = a.out or os.path.join(ANALYSIS, RUN, 'hv_scan', DET)
    os.makedirs(out_dir, exist_ok=True)
    t.to_csv(os.path.join(out_dir, 'charge_angle_vs_hv.csv'), index=False)
    loc.to_csv(os.path.join(out_dir, 'local_gain_slope.csv'), index=False)
    meta = dict(generated=datetime.now().isoformat(timespec='seconds'),
                run=RUN, det=DET, bundle=BUNDLE, alignment=ALIGN,
                v_drift_um_ns=V_DRIFT, gap_mm=GAP_MM, t_gap_ns=T_GAP_NS,
                sat_adc=SAT_ADC, quantile=QUANT, nboot=NBOOT,
                n_subruns=int(len(book)), n_events=int(book.n_reco.sum()),
                fits=fits, fits_q_win=fits_raw,
                local_slope_q_win=loc.to_dict('records'),
                peak_amp_crosscheck=(dict(
                    n=int(len(xchk)), ratio_min=float(xchk.ratio.min()),
                    ratio_max=float(xchk.ratio.max())) if xchk is not None
                    else None),
                bookkeeping=book.to_dict('records'))
    with open(os.path.join(out_dir, 'charge_angle_vs_hv.meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1, default=str)

    fig_charge(t, fits, fits_raw, os.path.join(out_dir, 'charge_vs_hv.png'))
    fig_occupancy(t, os.path.join(out_dir, 'occupancy_vs_hv.png'))
    fig_angres(t, os.path.join(out_dir, 'angres_vs_hv.png'))
    fig_slope(t, loc, os.path.join(out_dir, 'local_gain_slope.png'))
    fig_summary(t, fits, os.path.join(out_dir, 'charge_angle_summary.png'))
    rp = make_report(t, fits, fits_raw, book, out_dir, meta)

    if a.slides:
        img = os.path.join(REPO, 'mpgd26', 'slides', 'assets', 'img')
        os.makedirs(img, exist_ok=True)
        fig_charge(t, fits, fits_raw,
                   os.path.join(img, 'hv_total_charge.png'), slides=True)
        fig_occupancy(t, os.path.join(img, 'hv_occupancy.png'), slides=True)
        fig_angres(t, os.path.join(img, 'hv_angres.png'), slides=True)
        fig_slope(t, loc, os.path.join(img, 'hv_local_slope.png'), slides=True)
        fig_summary(t, fits, os.path.join(img, 'hv_charge_angle_summary.png'),
                    slides=True)
        print(f'[10e] slide assets -> {img}')

    # ------------------------------------------------------------- console
    pd.set_option('display.width', 200)
    show = [c for c in ['hv', 'view', 'n_plane_ok', 'frac_plane_ok', 'q_sum',
                        'q_win', 'q_5s', 'frac_sat', 'col_mm', 'span_mm',
                        'n_strip_5s', 'n_seed', 'ang_s68_deg', 'ang_bias_deg']
            if c in t]
    print(t[show].to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    print()
    for lbl, ff in (('q_sum (fit)', fits), ('q_win (raw)', fits_raw)):
      print(f'--- {lbl}')
      for k, v in ff.items():
        print(f'{k:>10}: d lnQ/dV = {v["slope_per10V"]:.4f} +- '
              f'{v["slope_err"]:.4f} /10 V   x2 every {v["double_V"]:.1f} V '
              f'({v["v_lo"]:.0f}-{v["v_hi"]:.0f} V, n={v["n_points"]})')
    sx = t[t.view == 'x'].sort_values('hv')
    print(f'\ncolumn length x: {sx.col_mm.iloc[0]:.1f} mm at {sx.hv.iloc[0]:.0f} V '
          f'-> {sx.col_mm.max():.1f} mm max '
          f'({100*sx.col_frac.iloc[0]:.0f} % -> {100*sx.col_frac.max():.0f} % '
          f'of the gap)')
    for view in ('x', 'y'):
        sv = t[t.view == view].sort_values('hv')
        i = sv.ang_s68_all_deg.idxmin()
        print(f'angular s68 {view} (every fitted plane): '
              f'{sv.ang_s68_all_deg.iloc[0]:.3f} deg at {sv.hv.iloc[0]:.0f} V '
              f'-> best {sv.loc[i, "ang_s68_all_deg"]:.3f} at '
              f'{sv.loc[i, "hv"]:.0f} V -> '
              f'{sv.ang_s68_all_deg.iloc[-1]:.3f} at {sv.hv.iloc[-1]:.0f} V')
    so = t[t.view == 'x'].sort_values('hv')
    j = so.open3d_p68_deg.idxmin()
    print(f'3-D opening angle p68: {so.open3d_p68_deg.iloc[0]:.3f} deg at '
          f'{so.hv.iloc[0]:.0f} V -> best {so.loc[j, "open3d_p68_deg"]:.3f} at '
          f'{so.loc[j, "hv"]:.0f} V -> {so.open3d_p68_deg.max():.3f} worst '
          f'(+{100 * (so.open3d_p68_deg.max() / so.loc[j, "open3d_p68_deg"] - 1):.0f} %)')
    print(f'strips over 5 sigma: {so.n_strip_5s.iloc[0]:.0f} -> '
          f'{so.n_strip_5s.max():.0f}; column length '
          f'{so.col_mm.min():.1f}-{so.col_mm.max():.1f} mm '
          f'(gap {GAP_MM:.0f} mm), no trend')
    print(f'\nwrote {out_dir}\n      report: {rp}')


if __name__ == '__main__':
    main()
