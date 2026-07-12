#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
39_spark_deadtime.py

Spark DEAD TIME -> efficiency ceiling (PLAN_39, paper point 6 completion).

The spark phenomenology (Poisson in time, muon-induced, edge-dominated) is done
elsewhere (det{3,7}_spark_analysis/). What is missing is the BRIDGE from spark
RATE to an IRREDUCIBLE efficiency loss: after a full-detector discharge, is the
detector blind / gain-sagged for some recovery time, and what efficiency ceiling
does that impose at a given spark rate?

RESULT (see the dated block at the bottom of PLAN_39): there is NO measurable
post-spark dead time on either detector. The DAQ next-event gap is identical
after a spark and after a normal event; the efficiency and the gain are flat
versus time-since-spark within errors. The discharges are localised and
non-propagating -- they do not blind the pad for subsequent triggered events.
The ONLY spark-induced efficiency loss is the in-spark crossing coincidence
itself (a muon that crosses DURING a discharge is unmeasurable): 4.4% of
crossings on det3, 35.7% on det7. That coincidence is what sets the operational
ceiling; there is no additional recovery tail. This script measures the null and
its upper limit, and decomposes the efficiency accordingly.

Method
  1. Timeline. events.npz (from extract_sparks.py) gives, per eventId over the
     whole run: trigger timestamp `ts` (ns), multiplicity, spark flag (>50 strips),
     and the M3 ray projection. ts is monotonic in eventId for firing events;
     silent M3 crossings carry ts=0 and get a timestamp by interpolating the
     (eventId -> ts) curve. Spark epochs are merged within 10 ms (afterpulse
     trains) so any recovery time is not biased by train structure.
  2. DAQ-level dead time. For the recorded (firing) event stream, compare the gap
     to the NEXT firing event (in time and in eventId) after a spark vs after a
     normal event. A DAQ that goes busy after a discharge would show a larger
     next-event gap / eventId skip -- dead time the efficiency curve can never see.
  3. Efficiency recovery. Denominator = clean M3 crossings inside the active area
     (same construction as 09_efficiency_breakdown, chi2<5). A crossing is
     "efficient" if a reco X+Y exists within R mm of the ray. Non-spark crossings
     are binned by dt since the previous spark epoch (log bins); the curve is fit
     with eff(dt)=eff_inf*(1-A*exp(-dt/tau)) over the OBSERVABLE window (tau bounded
     to <=5 s -- longer taus are unconstrained because at these spark rates dt
     rarely exceeds a few seconds). The dead-time integral A*tau (detector-seconds
     lost per spark) and its 95% upper limit are the physics output; a
     model-independent transient deficit (first bin vs the flat weighted mean) is
     reported alongside so the null does not depend on the fit form.
  4. Gain recovery. Per-event max strip amplitude for low-multiplicity muon events,
     same binning: an HV sag after a discharge would appear as an amplitude dip.
  5. Efficiency decomposition / ceiling. intrinsic (non-spark) efficiency ->
     x (1 - f_inspark)   [in-spark coincidence, the measured, rate-driven loss]
     x (1 - D_postspark)  [post-spark dead time, consistent with 0, UL shown]
     = operational efficiency. The post-spark ceiling penalty at rate r is
     <= r * (A*tau)_UL, drawn as an upper-limit band; the measured operational
     points sit below it because of the in-spark coincidence, NOT a recovery tail.

Usage:  ../.venv/bin/python 39_spark_deadtime.py g_det3_wknd [--r=5] [--rebuild-amp]
        ../.venv/bin/python 39_spark_deadtime.py g_det7_long  --r=5
Output: <alignment_tpc_veto50>/spark_deadtime/spark_deadtime.png + .csv/.json (+ amp cache)
"""
import os
import sys
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

R = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--r=')), 5.0)
REBUILD_AMP = '--rebuild-amp' in sys.argv
MERGE_S = 0.010          # merge spark events within 10 ms into one discharge epoch
SPARK_THRESH = 50
HERE = os.path.dirname(os.path.abspath(__file__))

# spark-analysis dir (holds events.npz) keyed by detector name
SPARK_DIR = {'mx17_3': 'det3_spark_analysis', 'mx17_7': 'det7_spark_analysis'}


def build_amp_cache(cache_path):
    """Per-event max strip amplitude + multiplicity over the detector FEUs."""
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'amplitude'], library='pd')
    raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    g = raw.groupby('eventId')['amplitude']
    ev = g.size().index.values.astype(np.int64)
    amax = g.max().values.astype(np.float32)
    asum = g.sum().values.astype(np.float32)
    mult = raw.groupby('eventId').size().values.astype(np.int32)
    np.savez(cache_path, eventId=ev, amp_max=amax, amp_sum=asum, mult=mult)
    return dict(eventId=ev, amp_max=amax, amp_sum=asum, mult=mult)


def main():
    out_dir = CFG.out_dir('alignment_tpc_veto50', 'spark_deadtime')
    sdir = os.path.join(HERE, SPARK_DIR[CFG.DET_NAME])
    ev_npz = np.load(os.path.join(sdir, 'events.npz'), allow_pickle=True)

    # ---------- timeline ----------
    eid = ev_npz['eventId'].astype(np.int64)
    ts = ev_npz['ts'].astype(np.float64) / 1e9          # seconds
    spark = ev_npz['spark'].astype(bool)
    fire = ts > 0
    # (eventId -> ts) interpolant from firing events, monotonic in eventId
    of = np.argsort(eid[fire])
    fe = eid[fire][of]
    ft = ts[fire][of]
    t0 = ft.min()
    ft = ft - t0                                        # start timeline at 0

    def t_of_eid(e):
        return np.interp(e, fe, ft)

    ts_c = np.where(fire, ts - t0, np.nan)              # real seconds for firing events

    # spark epochs: sort spark times, merge within MERGE_S
    st = np.sort(ts_c[spark])
    epochs_t = [st[0]]
    for t in st[1:]:
        if t - epochs_t[-1] > MERGE_S:
            epochs_t.append(t)
    epochs_t = np.array(epochs_t)
    dur = ft.max() - ft.min()
    r_spark = len(epochs_t) / dur
    print(f'{CFG.KEY}: dur={dur/3600:.2f} h  firing={fire.sum()}  '
          f'spark_events={spark.sum()}  spark_epochs={len(epochs_t)}  '
          f'r_spark={r_spark:.3f} Hz')

    def dt_since_spark(t):
        """time since the most recent spark epoch strictly before t (NaN if none)."""
        i = np.searchsorted(epochs_t, t, side='right') - 1
        return np.where(i >= 0, t - epochs_t[np.clip(i, 0, len(epochs_t) - 1)], np.nan)

    # ---------- DAQ-level dead time: next firing-event gap after spark vs normal ----------
    # walk the firing stream in time order
    fdt = np.diff(ft)                                    # gap to next firing event (s)
    fde = np.diff(fe)                                    # eventId gap to next firing event
    fsp = spark[fire][of][:-1]                           # is the CURRENT event a spark?
    gap_after_spark = fdt[fsp]
    gap_after_norm = fdt[~fsp]
    eskip_after_spark = fde[fsp]
    eskip_after_norm = fde[~fsp]

    # ---------- efficiency recovery: reco vs M3 crossings (09-style denominator) ----------
    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json'))
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', 'event_results.pkl'), 'rb'))
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=5.0)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in res if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}
    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])

    spark_set = set(int(e) for e in eid[spark])
    fired_set = set(int(e) for e in eid[fire])
    det_lo, det_hi = int(fe.min()), int(fe.max())

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    evn = [int(v) for v in evn]

    c_eid, c_eff, c_t = [], [], []
    n_active = n_inspark = 0
    for e, x, y in zip(evn, px, py):
        if e < det_lo or e > det_hi:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1 and ay0 <= y <= ay1):
            continue
        n_active += 1
        if e in spark_set:                               # in-spark crossing: separate loss channel
            n_inspark += 1
            continue
        eff = 1 if (e in reco and np.hypot(x - reco[e][0], y - reco[e][1]) <= R) else 0
        c_eid.append(e)
        c_eff.append(eff)
        c_t.append(t_of_eid(e))
    c_eid = np.array(c_eid, np.int64)
    c_eff = np.array(c_eff)
    c_t = np.array(c_t)
    c_dt = dt_since_spark(c_t)

    f_inspark = n_inspark / n_active                     # in-spark coincidence fraction
    ok = np.isfinite(c_dt) & (c_dt >= 0)
    eff_nonspark = c_eff.mean()                          # intrinsic (given not-in-spark)
    eff_operational = eff_nonspark * (1 - f_inspark)     # includes in-spark coincidence loss
    print(f'  active crossings={n_active}  in-spark={n_inspark} (f={f_inspark*100:.1f}%)  '
          f'eff_nonspark={eff_nonspark*100:.1f}%  eff_operational={eff_operational*100:.1f}%')

    # log-spaced dt bins over the OBSERVABLE window
    edges = np.logspace(np.log10(0.05), np.log10(20.0), 14)
    idx = np.digitize(c_dt[ok], edges)
    ctr, effm, efferr, ns = [], [], [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() < 30:
            continue
        k = c_eff[ok][m].sum()
        n = m.sum()
        ctr.append(np.sqrt(edges[b - 1] * edges[b]))
        effm.append(k / n)
        efferr.append(max(np.sqrt((k / n) * (1 - k / n) / n), 0.5 / n))
        ns.append(n)
    ctr = np.array(ctr); effm = np.array(effm); efferr = np.array(efferr); ns = np.array(ns)

    # inverse-variance weighted mean = the flat reference (efficiency is flat vs dt)
    w = 1.0 / efferr ** 2
    eff_flat = float(np.sum(w * effm) / np.sum(w))
    eff_flat_err = float(1.0 / np.sqrt(np.sum(w)))
    chi2_flat = float(np.sum(((effm - eff_flat) / efferr) ** 2))
    ndf_flat = len(effm) - 1
    # model-independent transient deficit: FIRST bin vs the flat mean (a recovery would
    # make the just-after-spark bin sit BELOW the mean). Sign convention: deficit>0 = dip.
    deficit = eff_flat - effm[0]
    deficit_err = float(np.hypot(efferr[0], eff_flat_err))
    print(f'  eff flat mean={eff_flat*100:.2f}+/-{eff_flat_err*100:.2f}%  '
          f'chi2/ndf={chi2_flat:.1f}/{ndf_flat}  '
          f'first-bin deficit={deficit*100:+.2f}+/-{deficit_err*100:.2f}%')

    # fit eff(dt)=eff_inf*(1-A*exp(-dt/tau)); tau bounded to the observable window (<=5 s)
    # so A*tau is the dead-time integral we can actually constrain. A>=0 (a recovery, not
    # an anti-recovery). The 95% UL on the integral A*tau is the physics upper limit.
    def recov(dt, eff_inf, A, tau):
        return eff_inf * (1 - A * np.exp(-dt / tau))
    fit = None
    try:
        popt, pcov = curve_fit(recov, ctr, effm, p0=[eff_flat, 0.05, 1.0], sigma=efferr,
                               absolute_sigma=True, maxfev=40000,
                               bounds=([0.5, 0.0, 0.05], [1.0, 0.5, 5.0]))
        perr = np.sqrt(np.diag(pcov))
        integ = popt[1] * popt[2]                        # A*tau [detector-seconds per spark]
        # error on the product via first-order propagation (+ covariance term)
        integ_err = np.sqrt((popt[2] * perr[1]) ** 2 + (popt[1] * perr[2]) ** 2
                            + 2 * popt[1] * popt[2] * pcov[1, 2])
        integ_ul = max(integ + 1.645 * integ_err, 0.0)   # 95% one-sided UL
        fit = dict(eff_inf=popt[0], A=popt[1], tau=popt[2],
                   eff_inf_err=perr[0], A_err=perr[1], tau_err=perr[2],
                   integ=float(integ), integ_err=float(integ_err), integ_ul=float(integ_ul))
        print(f'  eff fit: eff_inf={popt[0]*100:.1f}%  A={popt[1]:.3f}+/-{perr[1]:.3f}  '
              f'tau={popt[2]:.2f}+/-{perr[2]:.2f} s  |  A*tau={integ:.3f}+/-{integ_err:.3f} s'
              f'  (95%% UL {integ_ul:.3f} s)')
    except Exception as ex:
        print(f'  eff fit FAILED: {ex}')

    # ---------- gain recovery: per-event max amplitude ----------
    amp_cache = os.path.join(out_dir, 'amp_cache.npz')
    if REBUILD_AMP or not os.path.exists(amp_cache):
        print('  building amplitude cache from combined_hits ...')
        amp = build_amp_cache(amp_cache)
    else:
        amp = dict(np.load(amp_cache))
    amp_ev = amp['eventId']
    amp_max = amp['amp_max']
    amp_mult = amp['mult']
    amp_t = t_of_eid(amp_ev)
    amp_dt = dt_since_spark(amp_t)
    # clean muon events only: modest multiplicity, not a spark
    muon = (amp_mult >= 4) & (amp_mult <= SPARK_THRESH) & np.isfinite(amp_dt) & (amp_dt >= 0)
    a_dt = amp_dt[muon]
    a_amp = amp_max[muon].astype(float)
    aidx = np.digitize(a_dt, edges)
    a_ctr, a_med, a_lo, a_hi, a_n = [], [], [], [], []
    for b in range(1, len(edges)):
        m = aidx == b
        if m.sum() < 30:
            continue
        vals = a_amp[m]
        a_ctr.append(np.sqrt(edges[b - 1] * edges[b]))
        a_med.append(np.median(vals))
        a_lo.append(np.percentile(vals, 16))
        a_hi.append(np.percentile(vals, 84))
        a_n.append(m.sum())
    a_ctr = np.array(a_ctr); a_med = np.array(a_med)
    a_lo = np.array(a_lo); a_hi = np.array(a_hi); a_n = np.array(a_n)
    # amplitude is flat vs dt: use the overall muon median as the reference (dt>5 s is
    # too sparse at these rates for a stable plateau). Quote the first-bin sag as the
    # model-independent gain-recovery limit.
    amp_ref = float(np.median(a_amp)) if len(a_amp) else np.nan
    amp_sag = float((amp_ref - a_med[0]) / amp_ref) if (len(a_med) and np.isfinite(amp_ref)) else np.nan
    print(f'  amp muon median={amp_ref:.0f}  first-bin sag={amp_sag*100:+.1f}%  (flat = no HV droop)')

    # ---------- post-spark dead-time upper limit ----------
    # The exp fit's A*tau integral is kept for reference, but at these spark rates dt
    # rarely exceeds a few seconds so tau (and hence A*tau) is degenerate -- for det7 it
    # rails and the integral is meaningless. The ROBUST, model-independent bound is the
    # transient deficit: the efficiency in the first observable bin (dt ~ 64 ms) relative
    # to the flat plateau. It is consistent with zero on both detectors (negative on det7
    # = a small EXCESS, the opposite of a recovery). Since the non-spark efficiency is
    # flat all the way out, this peak deficit also bounds the time-AVERAGED post-spark
    # loss. Report a one-sided 95% UL in efficiency POINTS (never below the 1.645-sigma
    # sensitivity), and its relative value.
    A_tau = fit['integ'] if fit else 0.0
    A_tau_ul = fit['integ_ul'] if fit else np.nan
    D_ul_pts = max(deficit, 0.0) * 100 + 1.645 * deficit_err * 100   # abs efficiency points
    D_ul_rel = D_ul_pts / (eff_flat * 100)                            # fraction of efficiency

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) DAQ dead time
    ax = axes[0, 0]
    bins = np.logspace(np.log10(0.02), np.log10(30), 60)
    ax.hist(gap_after_norm, bins=bins, density=True, histtype='step',
            color='steelblue', lw=1.6, label=f'after normal (med {np.median(gap_after_norm)*1e3:.0f} ms)')
    ax.hist(gap_after_spark, bins=bins, density=True, histtype='step',
            color='crimson', lw=1.6, label=f'after spark (med {np.median(gap_after_spark)*1e3:.0f} ms)')
    ax.set_xscale('log')
    ax.set_xlabel('gap to next recorded event [s]')
    ax.set_ylabel('density')
    ax.set_title('(a) DAQ dead time: next-event gap after spark vs normal')
    ax.legend(fontsize=9)
    ax.text(0.02, 0.02,
            f'median eventId skip: spark {np.median(eskip_after_spark):.0f} vs '
            f'normal {np.median(eskip_after_norm):.0f}',
            transform=ax.transAxes, fontsize=8, va='bottom')

    # (b) efficiency vs time-since-spark: flat (no recovery), with UL band
    ax = axes[0, 1]
    ax.errorbar(ctr, effm * 100, yerr=np.array(efferr) * 100, fmt='o',
                color='seagreen', capsize=2, label='efficiency (R=%g mm)' % R)
    ax.axhspan((eff_flat - eff_flat_err) * 100, (eff_flat + eff_flat_err) * 100,
               color='seagreen', alpha=0.12)
    ax.axhline(eff_flat * 100, color='seagreen', ls='--', lw=1.2,
               label=r'flat mean %.1f%% ($\chi^2$/ndf=%.1f/%d)' % (eff_flat * 100, chi2_flat, ndf_flat))
    if fit and fit['A_err'] < 0.1 and 0 < fit['integ_ul'] / fit['tau'] < 0.5:
        xx = np.logspace(np.log10(ctr.min()), np.log10(ctr.max()), 200)
        # 95%-UL recovery envelope: strongest allowed dip at fitted tau (drawn only when
        # the fit is well constrained; at high spark rate tau is degenerate -> skipped)
        A_ul = fit['integ_ul'] / fit['tau']
        ax.plot(xx, eff_flat * (1 - A_ul * np.exp(-xx / fit['tau'])) * 100,
                'k:', lw=1.3, label=r'95%% UL recovery ($A\tau\leq$%.2f s)' % fit['integ_ul'])
    ax.set_xscale('log')
    ax.set_xlabel('time since previous spark  $\\Delta t$ [s]')
    ax.set_ylabel('efficiency [%]')
    ax.set_title('(b) efficiency vs $\\Delta t$ — flat: no post-spark recovery')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=0.3)

    # (c) gain vs time-since-spark: flat (no HV sag)
    ax = axes[1, 0]
    if len(a_ctr):
        ax.fill_between(a_ctr, a_lo, a_hi, color='mediumpurple', alpha=0.2, label='16-84%')
        ax.plot(a_ctr, a_med, 'o-', color='indigo', label='median max amplitude')
        if np.isfinite(amp_ref):
            ax.axhline(amp_ref, color='gray', ls='--', lw=1,
                       label=f'muon median {amp_ref:.0f}  (sag {amp_sag*100:+.1f}%)')
    ax.set_xscale('log')
    ax.set_xlabel('time since previous spark  $\\Delta t$ [s]')
    ax.set_ylabel('per-event max strip amplitude [ADC]')
    ax.set_title('(c) gain vs $\\Delta t$ — flat: no HV droop')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=0.3)

    # (d) efficiency decomposition: in-spark coincidence is the only spark loss
    ax = axes[1, 1]
    stages = ['intrinsic\n(non-spark)', '× (1−f_inspark)\nin-spark coinc.',
              '− post-spark\n(UL, ≈0)']
    e0 = eff_nonspark * 100
    e1 = eff_operational * 100
    e2 = e1                                                # post-spark nominal = 0
    vals = [e0, e1, e2]
    cols = ['#4c72b0', '#dd8452', '#55a868']
    ax.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.015, f'{v:.1f}%', ha='center', fontsize=10)
    # annotate the in-spark coincidence loss in the gap between bars 0 and 1
    ax.annotate('', xy=(0.5, e1), xytext=(0.5, e0),
                arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.4))
    ax.text(0.5, (e0 + e1) / 2, f'−{f_inspark*100:.1f}%\n(rate-\ndriven)',
            color='crimson', fontsize=8.5, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    ax.errorbar([2], [e2], yerr=[[D_ul_pts], [0]], fmt='none', ecolor='seagreen',
                elinewidth=1.4, capsize=4)
    ax.text(2.0, e2 - max(vals) * 0.10,
            f'post-spark\ndead time\n≤ {D_ul_pts:.1f} pts (95% UL)\nconsistent with 0',
            color='seagreen', fontsize=8.5, ha='center', va='top')
    ax.set_xticks(range(3))
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel('active-area efficiency [%]')
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title(f'(d) efficiency decomposition — f_inspark={f_inspark*100:.1f}% at '
                 f'{r_spark:.2f} Hz')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'{CFG.DET_NAME} spark dead time & efficiency ceiling — '
                 f'{CFG.RUN}/{CFG.SUB_RUN}  (r_spark={r_spark:.2f} Hz, '
                 f'no post-spark recovery resolved)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(out_dir, 'spark_deadtime.png'), dpi=150, bbox_inches='tight')
    print(f'  wrote {out_dir}/spark_deadtime.png')

    # ---------- CSV / JSON of the numbers ----------
    summary = dict(
        key=CFG.KEY, det=CFG.DET_NAME, run=CFG.RUN, subrun=CFG.SUB_RUN,
        R_mm=R, duration_h=dur / 3600, n_firing=int(fire.sum()),
        n_spark_events=int(spark.sum()), n_spark_epochs=int(len(epochs_t)),
        r_spark_Hz=r_spark,
        n_active_crossings=int(n_active), n_inspark_crossings=int(n_inspark),
        f_inspark_pct=float(f_inspark * 100),
        n_nonspark_crossings=int(len(c_eid)),
        eff_nonspark_pct=float(eff_nonspark * 100),
        eff_operational_pct=float(eff_operational * 100),
        eff_flat_pct=float(eff_flat * 100), eff_flat_err_pct=float(eff_flat_err * 100),
        eff_flat_chi2=chi2_flat, eff_flat_ndf=ndf_flat,
        transient_deficit_pct=float(deficit * 100),
        transient_deficit_err_pct=float(deficit_err * 100),
        daq_gap_after_spark_ms=float(np.median(gap_after_spark) * 1e3),
        daq_gap_after_norm_ms=float(np.median(gap_after_norm) * 1e3),
        daq_eidskip_after_spark=float(np.median(eskip_after_spark)),
        daq_eidskip_after_norm=float(np.median(eskip_after_norm)),
        amp_muon_median=float(amp_ref) if np.isfinite(amp_ref) else None,
        amp_first_bin_sag_pct=float(amp_sag * 100) if np.isfinite(amp_sag) else None,
        A_tau_s=float(A_tau), A_tau_ul_s=float(A_tau_ul) if np.isfinite(A_tau_ul) else None,
        D_postspark_ul_pts=float(D_ul_pts), D_postspark_ul_rel_pct=float(D_ul_rel * 100),
    )
    if fit:
        summary.update(fit_eff_inf_pct=float(fit['eff_inf'] * 100),
                       fit_A=float(fit['A']), fit_A_err=float(fit['A_err']),
                       fit_tau_s=float(fit['tau']), fit_tau_err_s=float(fit['tau_err']))
    json.dump(summary, open(os.path.join(out_dir, 'spark_deadtime.json'), 'w'), indent=2)

    # per-bin CSV
    with open(os.path.join(out_dir, 'spark_deadtime.csv'), 'w') as f:
        f.write('dt_center_s,eff,eff_err,n_crossings,amp_median,amp_n\n')
        amap = {round(c, 6): (m, n) for c, m, n in zip(a_ctr, a_med, a_n)}
        for c, m, e, n in zip(ctr, effm, efferr, ns):
            am, an_ = amap.get(round(c, 6), ('', ''))
            f.write(f'{c:.4f},{m:.4f},{e:.4f},{n},{am},{an_}\n')
    print(f'  wrote spark_deadtime.csv + .json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
