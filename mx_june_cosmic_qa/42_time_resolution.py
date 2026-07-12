#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
42_time_resolution.py

TIME RESOLUTION in micro-TPC mode (paper topic 10).

Corrected framing (2026-07-11, after reading mm_strip_reconstruction):
the bench DOES have an absolute time reference. The acquisition is triggered by
a top+bottom scintillator-paddle coincidence (PMTs, ~5 ns each), so the readout
window start ~ the muon crossing time. The DREAM 60-ns sampling clock is
free-running w.r.t. that trigger, but the 3-bit fine timestamp `ftst` records the
trigger-to-clock phase (10 ns granularity, 100 MHz), and the reconstruction ADDS
`ftst*10 ns` to every strip time (WaveformAnalyzer.cpp:390-392). Confirmed here:
hit `time` is flat vs ftst (slope ~0); undoing the correction restores −10 ns/step.
So per-strip `time` is an absolute leading-edge time (30 %-of-peak constant-fraction
crossing, sub-sample interpolated) referenced to the phase-corrected trigger.

Three complementary numbers, all on frozen data:

  A. SINGLE-STRIP σ_t — scatter of micro-TPC strip times about the fitted line
     time(position) on inclined tracks (per-event, (N−2)-normalised). Also its
     amplitude dependence (per-strip time-walk), which is a correctable calibration.

  B. PLANE-TO-PLANE detector σ_t (clean, telescope- AND geometry-free). The X and
     Y strip layers sit under ONE drift gap and collect the SAME drifting electrons,
     so the shortest-drift charge gives the SAME leading time in both layers →
     σ(t_X−t_Y)/√2 is the single-plane detector time resolution; the median of the
     difference is the residual inter-plane time-walk (a bias). Differencing cancels
     the trigger, ftst and geometry, isolating the detector.

  C. ABSOLUTE event-time resolution — spread of the leading-edge event time relative
     to the (ftst-corrected) trigger. Upper limit (includes the event-to-event drift
     geometry). Its budget: detector (B) ⊕ scintillator (~5 ns) ⊕ ftst quant (10/√12 ≈
     2.9 ns) ⊕ geometry — detector-dominated. The ftst correction is demonstrated by
     comparing the absolute spread with/without it.

All converted to equivalent drift distance via σ_z = v_drift·σ_t (34.30 µm/ns).

Usage: ../.venv/bin/python 42_time_resolution.py sat_det3 [--veto=50]
Output: <alignment_tpc_vetoN>/time_resolution/
        time_resolution.png (overview) + figs/*.png (report figures) +
        time_resolution.{csv,json}
"""
import os
import sys
import glob
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
SAMPLE_NS = 60.0
FTST_NS = 10.0            # ns per fine-timestamp unit (100 MHz DREAM clock)
GAP_MM = getattr(cm, 'GAP_THRESHOLD_MM', 2.0)
PITCH_MM = 0.78
AMP_THR = 100.0
T_WINDOW = (0.0, 2000.0)
MIN_STRIPS = 3
CORE_FRAC = 0.30
MIN_CORE_A = 4
MIN_PTP_MM = 2.0
V_DRIFT = 34.30
CF_FRAC = 0.30           # constant fraction used in reconstruction (for the schematic)
SCINT_NS = 5.0           # scintillator-paddle single-PMT time resolution (hardware note)

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'time_resolution')
FIGS = CFG.out_dir(f'alignment_tpc{tag}', 'time_resolution', 'figs')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')


def largest_cluster(pos):
    o = np.argsort(pos)
    breaks = np.where(np.diff(pos[o]) > GAP_MM)[0]
    return max(np.split(o, breaks + 1), key=len)


def s68(a):
    q = np.percentile(a, [16, 84])
    return 0.5 * (q[1] - q[0])


def load_ftst():
    """Per-event fine timestamp from the decoded FEU-X tree (one value/event)."""
    ftst = {}
    for fn in sorted(glob.glob(os.path.join(DEC_DIR, f'*_{CFG.MX17_FEU_X:02d}.root'))):
        a = uproot.open(fn)['nt'].arrays(['eventId', 'ftst'], library='np')
        ftst.update(dict(zip(a['eventId'].tolist(), a['ftst'].tolist())))
    return ftst


# ============================================================ schematics
def fig_algorithm(consts):
    """Diagram: the absolute-timing chain (trigger → ftst phase → CF crossing)."""
    fig, (axc, axw) = plt.subplots(1, 2, figsize=(15, 5.4))

    # --- left: clock / trigger / ftst phase ---
    axc.set_title('Absolute-time chain: scintillator trigger → ftst phase → window',
                  fontsize=11)
    axc.set_xlim(0, 12); axc.set_ylim(0, 6); axc.axis('off')
    # free-running 60 ns sample grid
    for i in range(13):
        x = i * 0.9 + 0.3
        axc.plot([x, x], [4.2, 4.9], color='0.6', lw=1)
    axc.annotate('', xy=(11.9, 4.55), xytext=(0.3, 4.55),
                 arrowprops=dict(arrowstyle='-', color='0.6'))
    axc.text(6, 5.15, 'free-running DREAM 60 ns sample clock', ha='center',
             fontsize=9, color='0.4')
    # trigger (scintillator) arrow at a random phase
    xt = 3.0 * 0.9 + 0.3 + 0.42
    axc.add_patch(FancyArrowPatch((xt, 2.4), (xt, 4.2), arrowstyle='-|>',
                                  mutation_scale=16, color='crimson', lw=2))
    axc.text(xt, 2.05, 'scintillator\ncoincidence trigger\n(top+bottom PMTs, ~5 ns)',
             ha='center', va='top', fontsize=8.5, color='crimson')
    # ftst measures the phase between trigger and the next sample edge
    xs = 4 * 0.9 + 0.3
    axc.annotate('', xy=(xs, 3.9), xytext=(xt, 3.9),
                 arrowprops=dict(arrowstyle='<->', color='tab:blue', lw=1.6))
    axc.text((xt + xs) / 2, 3.55, 'ftst\n(0–7, 10 ns steps)', ha='center', va='top',
             fontsize=8.5, color='tab:blue')
    axc.text(6, 0.9, 'reco ADDS ftst·10 ns to every strip time\n'
             '→ window start re-referenced to the trigger (10 ns granularity)',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', fc='#eef4ff', ec='tab:blue'))

    # --- right: constant-fraction leading-edge timing on a pulse ---
    axw.set_title('Per-strip time: 30 % constant-fraction crossing (sub-sample)',
                  fontsize=11)
    t = np.linspace(0, 20, 400)
    # a plausible DREAM pulse (fast rise, slow tail)
    pk = 6.0
    y = np.where(t < pk, np.exp(-((t - pk) / 2.4) ** 2),
                 np.exp(-(t - pk) / 5.5))
    y = y / y.max()
    samp_t = np.arange(0, 20, SAMPLE_NS / 60.0 * 3.6)  # coarse markers for illustration
    axw.plot(t, y, color='tab:purple', lw=2, label='baseline-subtracted pulse')
    axw.plot(samp_t, np.interp(samp_t, t, y), 'o', color='0.5', ms=5,
             label='60 ns samples')
    axw.axhline(CF_FRAC, color='tab:green', ls='--', lw=1.3,
                label=f'{CF_FRAC:.0%} of peak')
    # crossing point
    rise = t < pk
    tc = np.interp(CF_FRAC, y[rise], t[rise])
    axw.plot([tc, tc], [0, CF_FRAC], color='crimson', lw=1.6)
    axw.plot(tc, CF_FRAC, 'v', color='crimson', ms=9,
             label='interpolated crossing = time')
    axw.set_xlabel('time in window [a.u.]'); axw.set_ylabel('amplitude / peak')
    axw.legend(fontsize=8.5, loc='upper right'); axw.set_ylim(0, 1.12)
    axw.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'algorithm_schematic.png'), dpi=160)
    plt.close(fig)


def fig_geometry():
    """Diagram: one drift gap, two orthogonal strip layers, same arrival times."""
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title('micro-TPC: one drift gap, two orthogonal readout layers\n'
                 'both timestamp the SAME drifting electrons', fontsize=11)
    # mesh (top) and readout (bottom)
    ax.plot([1, 9], [8.4, 8.4], color='k', lw=2)
    ax.text(9.05, 8.4, 'mesh', va='center', fontsize=9)
    ax.add_patch(Rectangle((1, 1.2), 8, 0.35, color='tab:blue', alpha=0.35))
    ax.add_patch(Rectangle((1, 0.85), 8, 0.35, color='tab:orange', alpha=0.35))
    ax.text(9.05, 1.37, 'X strips', va='center', fontsize=9, color='tab:blue')
    ax.text(9.05, 1.02, 'Y strips (⊥)', va='center', fontsize=9, color='tab:orange')
    # inclined muon track
    ax.plot([2.3, 5.2], [8.4, 1.55], color='crimson', lw=2.5)
    ax.text(2.0, 8.6, 'muon track', color='crimson', fontsize=9)
    # ionization points drifting down
    zs = np.linspace(1.7, 8.2, 7)
    xs = np.interp(zs, [1.55, 8.4], [5.2, 2.3])
    for xi, zi in zip(xs, zs):
        ax.plot(xi, zi, 'o', color='crimson', ms=4)
        ax.add_patch(FancyArrowPatch((xi, zi), (xi, 1.6), arrowstyle='-|>',
                                     mutation_scale=8, color='0.6', lw=0.8))
    # leading edge = nearest-to-readout ionization
    ax.annotate('shortest drift → leading edge\n(same time in X and Y)',
                xy=(xs[0], 1.75), xytext=(5.7, 3.4), fontsize=8.5,
                arrowprops=dict(arrowstyle='->', color='k'))
    ax.text(0.2, 5.0, 'drift', rotation=90, fontsize=9, color='0.5')
    ax.add_patch(FancyArrowPatch((0.6, 8.0), (0.6, 1.7), arrowstyle='-|>',
                                 mutation_scale=10, color='0.5', lw=1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'geometry_schematic.png'), dpi=160)
    plt.close(fig)


def main():
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)

    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    print(f'{CFG.RUN} / {CFG.SUB_RUN} — {len(fs)} combined_hits file(s)')
    df = uproot.concatenate(
        [f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'time',
                     'time_of_max'], library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)]
    hpe = df.groupby('eventId')['eventId'].transform('size')
    df = df[hpe <= VETO].copy()
    df = cm._map_strip_positions(df, det)
    df = df[(df['time'] > T_WINDOW[0]) & (df['time'] < T_WINDOW[1])
            & (df['amplitude'] > AMP_THR)].copy()
    print(f'{len(df):,} in-window hits after veto{VETO} + amp>{AMP_THR:g}')
    ftst_by = load_ftst()
    print(f'ftst loaded for {len(ftst_by):,} events '
          f'(range {min(ftst_by.values())}–{max(ftst_by.values())})')

    planes = {'x': 'x_position_mm', 'y': 'y_position_mm'}
    resid_sig = {'x': [], 'y': []}
    resid_drift = []
    walk_relamp, walk_resid = [], []   # per-strip time-walk (residual vs rel-amp)
    ev = {}
    for eid, g in df.groupby('eventId', sort=False):
        rec = {}
        for p, pcol in planes.items():
            gp = g[g[pcol].notna()]
            if len(gp) < MIN_STRIPS:
                continue
            idx = largest_cluster(gp[pcol].to_numpy())
            if len(idx) < MIN_STRIPS:
                continue
            gc = gp.iloc[idx]
            pos = gc[pcol].to_numpy(); t = gc['time'].to_numpy()
            amp = gc['amplitude'].to_numpy()
            rec[p] = dict(t_lead=float(t.min()), t_med=float(np.median(t)),
                          amp_sum=float(amp.sum()), n=len(idx))
            core = amp >= CORE_FRAC * amp.max()
            if core.sum() >= MIN_CORE_A and np.ptp(pos[core]) >= MIN_PTP_MM:
                pc, tc, ac = pos[core], t[core], amp[core]
                coef = np.polyfit(pc, tc, 1)
                r = tc - np.polyval(coef, pc)
                resid_sig[p].append(np.sqrt(np.sum(r ** 2) / (core.sum() - 2)))
                resid_drift.append(float(tc.min()))
                walk_relamp.append(ac / ac.max())
                walk_resid.append(r)
        if 'x' in rec and 'y' in rec:
            rec['ftst'] = ftst_by.get(eid, np.nan)
            ev[eid] = rec
    print(f'{len(ev):,} events with a cluster in BOTH planes')

    # ---------------- A. single-strip ----------------
    A = {p: float(np.median(resid_sig[p])) for p in 'xy'}
    A_comb = float(np.median(np.concatenate([resid_sig['x'], resid_sig['y']])))
    relamp = np.concatenate(walk_relamp); wres = np.concatenate(walk_resid)
    wa_ctr, wa_med = [], []
    for b0 in np.arange(0.30, 1.0, 0.05):
        m = (relamp >= b0) & (relamp < b0 + 0.05)
        if m.sum() > 200:
            wa_ctr.append(b0 + 0.025); wa_med.append(np.median(wres[m]))
    wa_ctr, wa_med = np.array(wa_ctr), np.array(wa_med)
    walk_perstrip = np.polyfit(wa_ctr, wa_med, 1)[0] if len(wa_ctr) > 2 else np.nan
    print(f'\n== A. single-strip σ_t = {A_comb:.1f} ns ({A_comb*V_DRIFT/1000:.2f} mm) '
          f"[x {A['x']:.1f} / y {A['y']:.1f}] ==")
    print(f'   per-strip time-walk slope = {walk_perstrip:+.0f} ns per unit '
          'relative amplitude (correctable calibration)')

    # ---------------- B. plane-to-plane ----------------
    eids = list(ev)
    dt_lead = np.array([ev[e]['x']['t_lead'] - ev[e]['y']['t_lead'] for e in eids])
    dt_med = np.array([ev[e]['x']['t_med'] - ev[e]['y']['t_med'] for e in eids])
    asym = np.array([(ev[e]['x']['amp_sum'] - ev[e]['y']['amp_sum']) /
                     (ev[e]['x']['amp_sum'] + ev[e]['y']['amp_sum']) for e in eids])
    lead_bias, lead_s68 = float(np.median(dt_lead)), float(s68(dt_lead))
    lead_single = lead_s68 / np.sqrt(2.0)
    a_edges = np.linspace(-0.6, 0.6, 13); a_ctr, a_med = [], []
    for lo, hi in zip(a_edges[:-1], a_edges[1:]):
        m = (asym >= lo) & (asym < hi)
        if m.sum() > 40:
            a_ctr.append(0.5 * (lo + hi)); a_med.append(np.median(dt_lead[m]))
    a_ctr, a_med = np.array(a_ctr), np.array(a_med)
    walk_slope = float(np.polyfit(a_ctr, a_med, 1)[0]) if len(a_ctr) > 2 else np.nan
    dt_lead_corr = dt_lead - walk_slope * asym
    lead_single_corr = float(s68(dt_lead_corr)) / np.sqrt(2.0)
    print(f'\n== B. plane-to-plane detector σ_t ==')
    print(f'   leading edge: bias {lead_bias:+.1f} ns, σ68(Δ) {lead_s68:.1f} '
          f'→ single-plane {lead_single:.1f} ns ({lead_single*V_DRIFT/1000:.2f} mm)')
    print(f'   walk-corrected → {lead_single_corr:.1f} ns '
          f'({lead_single_corr*V_DRIFT/1000:.2f} mm) [intrinsic floor]')
    print(f'   inter-plane walk vs charge asymmetry {walk_slope:+.0f} ns/asym; '
          f'median-time estimator {s68(dt_med)/np.sqrt(2):.1f} ns')

    # ---------------- C. absolute event-time ----------------
    have_ft = [e for e in eids if np.isfinite(ev[e]['ftst'])]
    t0 = np.array([min(ev[e]['x']['t_lead'], ev[e]['y']['t_lead']) for e in have_ft])
    ft = np.array([ev[e]['ftst'] for e in have_ft])
    t0_unc = t0 - ft * FTST_NS
    abs_s68 = float(s68(t0)); abs_s68_unc = float(s68(t0_unc))
    slope_corr = float(np.polyfit(ft, t0, 1)[0])
    slope_unc = float(np.polyfit(ft, t0_unc, 1)[0])
    ftst_quant = FTST_NS / np.sqrt(12)
    # detector-dominated budget: abs^2 = det^2 + scint^2 + ftst^2 + geom^2
    det_term = lead_single
    geom_term = float(np.sqrt(max(abs_s68 ** 2 - det_term ** 2
                                  - SCINT_NS ** 2 - ftst_quant ** 2, 0.0)))
    print(f'\n== C. absolute event-time (leading edge vs trigger) ==')
    print(f'   σ68(t0) corrected {abs_s68:.1f} ns (UL, incl geometry); '
          f'uncorrected {abs_s68_unc:.1f} ns')
    print(f'   ftst slope: corrected {slope_corr:+.2f} vs uncorrected '
          f'{slope_unc:+.2f} ns/step (correction removes the phase term)')
    print(f'   budget: detector {det_term:.0f} ⊕ scint {SCINT_NS:.0f} ⊕ '
          f'ftst-quant {ftst_quant:.1f} ⊕ geometry {geom_term:.0f} ns '
          '→ detector-dominated')

    # ---------------- σ_t vs drift depth ----------------
    tlx = np.array([ev[e]['x']['t_lead'] for e in eids])
    d_edges = np.arange(200, 1400, 150.0); d_ctr, d_res = [], []
    for lo, hi in zip(d_edges[:-1], d_edges[1:]):
        m = (tlx >= lo) & (tlx < hi)
        if m.sum() > 80:
            d_ctr.append(0.5 * (lo + hi)); d_res.append(s68(dt_lead[m]) / np.sqrt(2))
    d_ctr, d_res = np.array(d_ctr), np.array(d_res)

    # ---------------- summary ----------------
    summ = dict(
        run=CFG.RUN, subrun=CFG.SUB_RUN, det=CFG.DET_NAME, veto=VETO,
        n_events_dualplane=len(ev), v_drift_um_ns=V_DRIFT, sampling_ns=SAMPLE_NS,
        single_strip_sigma_ns=A_comb, single_strip_sigma_ns_x=A['x'],
        single_strip_sigma_ns_y=A['y'], single_strip_sigma_mm=A_comb*V_DRIFT/1000,
        perstrip_walk_ns_per_relamp=float(walk_perstrip),
        lead_bias_ns=lead_bias, lead_dt_sigma68_ns=lead_s68,
        lead_singleplane_ns=lead_single, lead_singleplane_mm=lead_single*V_DRIFT/1000,
        lead_singleplane_walkcorr_ns=lead_single_corr,
        interplane_walk_ns_per_asym=walk_slope,
        med_singleplane_ns=float(s68(dt_med)/np.sqrt(2)),
        abs_t0_sigma68_ns=abs_s68, abs_t0_sigma68_uncorr_ns=abs_s68_unc,
        abs_t0_mm=abs_s68*V_DRIFT/1000,
        ftst_slope_corr_ns=slope_corr, ftst_slope_uncorr_ns=slope_unc,
        ftst_quant_ns=ftst_quant, scint_ns=SCINT_NS, geometry_term_ns=geom_term,
        note=('Absolute timing available (scintillator-coincidence trigger + '
              'applied ftst phase correction). Detector-dominated budget; '
              'plane-to-plane number is the pure detector term.'))
    with open(os.path.join(OUT, 'time_resolution.json'), 'w') as fh:
        json.dump(summ, fh, indent=2)
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, 'time_resolution.csv'), index=False)

    # ================= report figures =================
    fig_algorithm(dict())
    fig_geometry()

    # ftst correction demonstration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for arr, lab, c, sl in [(t0_unc, 'uncorrected (undo ftst)', 'tab:red', slope_unc),
                            (t0, 'ftst-corrected (as reconstructed)', 'tab:blue',
                             slope_corr)]:
        cx, cy = [], []
        for k in sorted(set(ft.tolist())):
            m = ft == k
            if m.sum() > 30:
                cx.append(k); cy.append(np.median(arr[m]))
        a1.plot(cx, cy, 'o-', color=c, label=f'{lab}  ({sl:+.1f} ns/step)')
    a1.set_xlabel('fine timestamp ftst (10 ns units)')
    a1.set_ylabel('median leading-edge event time [ns]')
    a1.set_title('ftst phase correction removes the sampling-clock phase')
    a1.legend(fontsize=9); a1.grid(alpha=0.3)
    a2.hist(np.clip(t0_unc, 150, 500), bins=np.arange(150, 502, 10), histtype='step',
            lw=2, color='tab:red', label=f'uncorrected σ68={abs_s68_unc:.0f} ns')
    a2.hist(np.clip(t0, 150, 500), bins=np.arange(150, 502, 10), histtype='step',
            lw=2, color='tab:blue', label=f'corrected σ68={abs_s68:.0f} ns')
    a2.set_xlabel('absolute leading-edge event time vs trigger [ns]')
    a2.set_ylabel('events'); a2.set_title('absolute event time (both planes, earliest)')
    a2.legend(fontsize=9); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'ftst_correction.png'), dpi=160)
    plt.close(fig)

    # single-strip: residual + per-strip walk
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    allsig = np.concatenate([resid_sig['x'], resid_sig['y']])
    a1.hist(np.clip(allsig, 0, 120), bins=np.arange(0, 122, 4), color='tab:purple',
            alpha=0.75)
    a1.axvline(A_comb, color='crimson', lw=2,
               label=f'median {A_comb:.0f} ns = {A_comb*V_DRIFT/1000:.2f} mm')
    a1.set_xlabel('per-event single-strip σ_t (line-fit residual) [ns]')
    a1.set_ylabel('inclined tracks'); a1.set_title('A. single-strip time resolution')
    a1.legend(fontsize=10); a1.grid(alpha=0.3)
    a2.hexbin(relamp, np.clip(wres, -250, 250), gridsize=40,
              extent=(0.3, 1.0, -250, 250), norm=LogNorm(), cmap='viridis')
    a2.plot(wa_ctr, wa_med, 'r.-', lw=1.6, label=f'median ({walk_perstrip:+.0f} ns/rel-amp)')
    a2.axhline(0, color='w', lw=0.8)
    a2.set_xlabel('strip amplitude / cluster max'); a2.set_ylabel('time residual [ns]')
    a2.set_title('per-strip time-walk (correctable calibration)')
    a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'single_strip.png'), dpi=160)
    plt.close(fig)

    # plane-to-plane: headline + walk
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    a1.hist(np.clip(dt_lead, -200, 200), bins=np.arange(-200, 202, 8),
            color='tab:blue', alpha=0.72)
    a1.axvline(lead_bias, color='k', ls='--', lw=1, label=f'bias {lead_bias:+.1f} ns')
    a1.axvspan(lead_bias - lead_s68, lead_bias + lead_s68, color='crimson', alpha=0.15,
               label=f'σ68(Δ)={lead_s68:.0f} ns → single-plane {lead_single:.0f} ns')
    a1.set_xlabel('X-plane − Y-plane leading-edge time [ns]'); a1.set_ylabel('events')
    a1.set_title('B. plane-to-plane detector agreement (headline)')
    a1.legend(fontsize=9); a1.grid(alpha=0.3)
    a2.hist2d(np.clip(asym, -0.6, 0.6), np.clip(dt_lead, -200, 200), bins=[40, 40],
              range=[[-0.6, 0.6], [-200, 200]], norm=LogNorm(), cmap='viridis')
    a2.plot(a_ctr, a_med, 'o-', color='crimson', ms=5, label='median Δt_lead')
    a2.axhline(0, color='w', lw=0.8)
    a2.set_xlabel('plane charge asymmetry (q_X−q_Y)/(q_X+q_Y)')
    a2.set_ylabel('Δt_lead (X−Y) [ns]')
    a2.set_title(f'inter-plane time-walk ({walk_slope:+.0f} ns/asym → 33→{lead_single_corr:.0f} ns)')
    a2.legend(fontsize=9, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'plane_to_plane.png'), dpi=160)
    plt.close(fig)

    # drift-depth
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(d_ctr, d_res, 'o-', color='tab:green', ms=6)
    for xx, yy in zip(d_ctr, d_res):
        ax.annotate(f'{yy*V_DRIFT/1000:.2f} mm', (xx, yy), fontsize=8,
                    textcoords='offset points', xytext=(0, 7), ha='center')
    ax.set_xlabel('event leading-edge drift time [ns]  (short → long drift)')
    ax.set_ylabel('single-plane σ_t [ns]')
    ax.set_title('time resolution vs drift depth (diffusion + attachment S/N)')
    ax.grid(alpha=0.3); ax.set_ylim(0, max(50, d_res.max() * 1.25) if len(d_res) else 50)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'drift_depth.png'), dpi=160)
    plt.close(fig)

    # ================= overview (kept) =================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    axes[0, 0].hist(np.clip(allsig, 0, 120), bins=np.arange(0, 122, 4),
                    color='tab:purple', alpha=0.7)
    axes[0, 0].axvline(A_comb, color='crimson', lw=2, label=f'{A_comb:.0f} ns')
    axes[0, 0].set_title('A. single-strip σ_t'); axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_xlabel('σ_t [ns]')
    axes[0, 1].hist(np.clip(dt_lead, -200, 200), bins=np.arange(-200, 202, 8),
                    color='tab:blue', alpha=0.7)
    axes[0, 1].axvspan(lead_bias - lead_s68, lead_bias + lead_s68, color='crimson',
                       alpha=0.15)
    axes[0, 1].set_title(f'B. plane-to-plane → {lead_single:.0f} ns/plane')
    axes[0, 1].set_xlabel('Δt_lead (X−Y) [ns]')
    axes[0, 2].hist(np.clip(t0, 150, 500), bins=np.arange(150, 502, 10),
                    color='tab:blue', histtype='step', lw=2, label=f'σ68={abs_s68:.0f}')
    axes[0, 2].hist(np.clip(t0_unc, 150, 500), bins=np.arange(150, 502, 10),
                    color='tab:red', histtype='step', lw=2, label=f'no ftst={abs_s68_unc:.0f}')
    axes[0, 2].set_title('C. absolute event time'); axes[0, 2].legend(fontsize=9)
    axes[0, 2].set_xlabel('t0 vs trigger [ns]')
    axes[1, 0].hist2d(np.clip(asym, -0.6, 0.6), np.clip(dt_lead, -200, 200),
                      bins=[40, 40], range=[[-0.6, 0.6], [-200, 200]], norm=LogNorm(),
                      cmap='viridis')
    axes[1, 0].plot(a_ctr, a_med, 'o-', color='crimson', ms=4)
    axes[1, 0].set_title(f'inter-plane walk {walk_slope:+.0f} ns/asym')
    axes[1, 0].set_xlabel('charge asymmetry'); axes[1, 0].set_ylabel('Δt_lead [ns]')
    if len(d_ctr):
        axes[1, 1].plot(d_ctr, d_res, 'o-', color='tab:green', ms=6)
    axes[1, 1].set_title('σ_t vs drift depth'); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlabel('leading-edge drift time [ns]'); axes[1, 1].set_ylabel('σ_t [ns]')
    axes[1, 1].set_ylim(0, max(50, d_res.max()*1.25) if len(d_res) else 50)
    ax = axes[1, 2]; ax.axis('off')
    txt = [
        f"TIME RESOLUTION — {CFG.DET_NAME}",
        f"{CFG.RUN}",
        f"n = {len(ev):,} dual-plane events, v = {V_DRIFT:.1f} µm/ns",
        "",
        "ABSOLUTE timing IS available:",
        "  scintillator trigger + applied ftst phase.",
        "",
        f"single-strip  σ_t   {A_comb:5.1f} ns  ({A_comb*V_DRIFT/1000:.2f} mm)",
        f"detector      σ_t   {lead_single:5.1f} ns  ({lead_single*V_DRIFT/1000:.2f} mm)",
        f"  walk-corrected    {lead_single_corr:5.1f} ns  ({lead_single_corr*V_DRIFT/1000:.2f} mm)",
        f"absolute t0  σ68    {abs_s68:5.1f} ns  (UL, +geom)",
        f"inter-plane bias    {lead_bias:+5.1f} ns  (≈0)",
        "",
        "budget (abs):",
        f"  detector {det_term:.0f} ⊕ scint {SCINT_NS:.0f} ⊕",
        f"  ftst-q {ftst_quant:.1f} ⊕ geom {geom_term:.0f}  → det-limited",
        "",
        "≈1 mm drift precision ↔ transverse σ (topic 9)",
    ]
    ax.text(0.02, 0.98, '\n'.join(txt), transform=ax.transAxes, va='top',
            fontsize=10.5, family='monospace')
    fig.suptitle(f'{CFG.RUN} — micro-TPC time resolution (absolute + intrinsic)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'time_resolution.png'), dpi=160)
    plt.close(fig)
    print(f'\nOverview: {OUT}/time_resolution.png\nReport figs: {FIGS}/')


if __name__ == '__main__':
    main()
