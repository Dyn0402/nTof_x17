#!/usr/bin/env python3
"""
A2 — run_61 singles 2-D scan: drift velocity, effective gap, efficiency vs
drift & resist HV.

Observables (documented; multiple cross-checked estimators, distributions
plotted so the edge definitions are auditable):

  t0_daq (anode / DAQ latency):  P5 of track-segment t0_ns. ~HV-independent
    (Det A: ~80 ns == the config's 'first arrival ~sample 3'). Anchors depth.

  T_max (full-gap transit = gap / v_drift):  high percentile (default P95) of
    the per-track tspan_ns of MICRO-TPC 'track' segments (in-pair preferred for
    purity). Rationale: a track's tspan measures the drift-time spread across
    the depth extent it traverses; the steepest full-gap radial tracks span the
    whole 30 mm, so the upper tail of tspan -> T_max. On the clean Det A this is
    unambiguous and monotonic in drift HV (run_58, same estimator: 972 ns @700 V
    -> 1975 ns @200 V; run_61 spans the same 700-200 V range, so its curve is a
    direct 6-point repeat).
    The ensemble t1_ns edge is NOT used (window-truncated + noise-contaminated).
    Cross-check: charge-weighted arrival spectrum (driftspec) centroid/edge.

  v_drift = D_nom / T_max  with D_nom = 30 mm (geometry). Model-independent in
    its E-dependence; the absolute scale carries the D_nom assumption.

  Effective gap:  D_eff(E) = v_ref(E) * T_max(E). If the true gas v(E) matched a
    reference curve, D_eff would be flat at the true gap; a FALL of D_eff at low
    field flags a shrinking active depth (far-cathode charge lost / field too
    low). v_ref here is the bench Ar/iso *95/5* curve baked into
    geometry.DriftModel -> ONLY a shape reference; a 90/10 Garfield curve is
    needed for the absolute gap (flagged).

  Efficiency:  P(3D x/y pair | late probe) per trigger vs drift (and resist).
    Relative (per-trigger) efficiency; absolute needs an external track
    reference (inter-chamber pointing, follow-up). 'late' here pools all four
    recovered comb teeth (27-69 ms), see scan_lib.

run_61 grid: 6 drift points (700, 600, 500, 400, 300, 200 V) x 10 resist
(560->515 V), all 60 sub-runs complete — the drift axis is a full scan, unlike
run_58's 150 V tail which was still acquiring when its analysis was first run.

Run: .venv/bin/python ntof_july_analysis/run61_scan/analyze_drift.py
Output -> <ANALYSIS_DIR>/July_HV_Scan/run61_scan/drift/
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import scan_lib as L  # noqa: E402
import feu_presence as FP  # noqa: E402
from ntof_tracking.reco import geometry as geo  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'drift')
DET_COL = {'A': 'crimson', 'B': 'royalblue', 'C': 'seagreen', 'D': 'darkorange'}
D_NOM_MM = geo.DRIFT_GAP          # 30 mm nominal
TMAX_PCTL = 95                    # tspan percentile used for T_max
MIN_TRACKS = 40                   # min tracks in a (drift,det) cell to fit T_max

# Magboltz Ar/iC4H10 90/10 at CERN pressure (720.8 Torr) — the RUN gas & site.
# DRY sim: the real nTOF gas is humidified, so the measurement is expected to
# sit ~15-20% below this (water slows drift).
GARFIELD_9010 = os.path.join(
    _REPO, 'garfield_sim', 'results', 'drift_velocity_Ar_iC4H10_90_10_CERN.json')


def garfield_vE():
    """(E [V/cm], v [um/ns]) from the dry Garfield 90/10 table, E-sorted."""
    d = json.load(open(GARFIELD_9010))
    E = np.array([p['E_Vcm'] for p in d['points']], float)
    v = np.array([p['v_um_per_ns'] for p in d['points']], float)
    o = np.argsort(E)
    return E[o], v[o]


def load():
    ev, segs, spec = L.load_all()
    if segs.empty:
        sys.exit('no cached track segments yet — run process.py first')
    segs = L.add_probe_class(segs)
    segs = segs[segs.flash_ok].copy()
    ev = L.add_probe_class(ev)
    ev = ev[ev.flash_ok].copy()
    ev = FP.attach(ev)      # run_61 FEU-dropout flags (efficiency denominators)
    # NOTE: `segs` needs no correction — a track segment can only exist in an
    # event where that detector was read out, and the drift estimators (t0_daq,
    # T_max) are distribution shapes over segments, not per-trigger rates.
    print(f'loaded {len(segs)} track segs, {len(ev)} events, '
          f'{spec.subrun.nunique()} spec sub-runs; '
          f'drifts={sorted(segs.drift.unique())}')
    return ev, segs, spec


def binom_err(k, n):
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / np.maximum(n, 1))


def drift_summary(segs, pure=True):
    """Per (drift, det): t0_daq, T_max(=P95 tspan), v_drift, n. Late probe."""
    s = segs[segs.probe_class == 'late']
    rows = []
    for (dr, det), g in s.groupby(['drift', 'det']):
        gp = g[g.in_pair] if (pure and g.in_pair.sum() >= MIN_TRACKS) else g
        n = len(gp)
        if n < MIN_TRACKS:
            rows.append(dict(drift=dr, det=det, n=n, t0_daq=np.nan,
                             t_max=np.nan, v_drift=np.nan, used_pure=False))
            continue
        t0 = float(np.percentile(gp.t0_ns, 5))
        tmax = float(np.percentile(gp.tspan_ns, TMAX_PCTL))
        rows.append(dict(
            drift=dr, det=det, n=n, t0_daq=t0, t_max=tmax,
            v_drift=D_NOM_MM / tmax * 1000.0,     # um/ns
            e_field=dr / (D_NOM_MM / 10.0),        # V/cm at nominal gap
            used_pure=bool(pure and g.in_pair.sum() >= MIN_TRACKS)))
    return pd.DataFrame(rows).sort_values(['det', 'drift'])


def fig_tspan_dist(segs):
    """Per-track tspan distributions per drift (the T_max estimator, audited)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    drifts = sorted(segs.drift.unique())
    cmap = plt.cm.plasma
    bins = np.arange(0, 3000, 80)
    for ax, det in zip(axes.flat, 'ABCD'):
        for i, dr in enumerate(drifts):
            g = segs[(segs.det == det) & (segs.drift == dr)
                     & (segs.probe_class == 'late')]
            gp = g[g.in_pair] if g.in_pair.sum() >= MIN_TRACKS else g
            if len(gp) < MIN_TRACKS:
                continue
            c = cmap(i / max(1, len(drifts) - 1))
            ax.hist(gp.tspan_ns, bins=bins, density=True, histtype='step',
                    color=c, lw=1.5, label=f'{dr} V' if det == 'A' else None)
            ax.axvline(np.percentile(gp.tspan_ns, TMAX_PCTL), color=c,
                       ls=':', lw=1)
        ax.set_title(f'Det {det}' + (' (clean M1)' if det == 'A' else ''),
                     color=DET_COL[det], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('track-segment tspan [ns]')
    for ax in axes[:, 0]:
        ax.set_ylabel('density')
    fig.legend(loc='center right', fontsize=8, title='drift HV')
    fig.suptitle(f'run_61 — micro-TPC track tspan vs drift HV '
                 f'(dotted = P{TMAX_PCTL} = T_max estimator; late probe)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.91, 1))
    return _save(fig, 'tspan_dist.png')


def fig_drift_spectrum(spec):
    """Charge-weighted arrival spectrum per drift (visual: column stretch)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    drifts = sorted(spec.drift.unique())
    cmap = plt.cm.plasma
    for ax, det in zip(axes.flat, 'ABCD'):
        for i, dr in enumerate(drifts):
            s = (spec[(spec.det == det) & (spec.drift == dr)]
                 .groupby('t_ns')['sum_amp'].sum())
            if s.empty:
                continue
            t = s.index.to_numpy(); q = s.to_numpy(float)
            base = np.median(q[(t > -500) & (t < -200)]) if (t < -200).any() else 0
            q = np.clip(q - base, 0, None)
            q = np.convolve(q, np.ones(3) / 3, mode='same')
            if q.max() > 0:
                q = q / q.max()
            c = cmap(i / max(1, len(drifts) - 1))
            ax.plot(t, q, color=c, lw=1.3, label=f'{dr} V' if det == 'A' else None)
        ax.set_xlim(-400, 4000)
        ax.set_title(f'Det {det}', color=DET_COL[det], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('hit arrival time in readout window [ns]')
    for ax in axes[:, 0]:
        ax.set_ylabel('baseline-sub. charge (norm.)')
    fig.legend(loc='center right', fontsize=8, title='drift HV')
    fig.suptitle('run_61 — clean-hit charge arrival spectrum vs drift '
                 '(column stretches / peak shifts later as field drops)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.91, 1))
    return _save(fig, 'drift_spectrum.png')


def fig_vdrift(summ):
    """v_drift vs E-field per det; bench 95/5 curve as a shape reference."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for det in 'ABCD':
        s = summ[(summ.det == det) & summ.v_drift.notna()].sort_values('e_field')
        axes[0].plot(s.e_field, s.v_drift, color=DET_COL[det], marker='o', ms=5,
                     label=f'Det {det}')
        axes[1].plot(s.drift, s.t_max, color=DET_COL[det], marker='o', ms=5,
                     label=f'Det {det}')
    gE, gV = garfield_vE()
    m = gE <= 260
    axes[0].plot(gE[m], gV[m], 'k--', lw=1.3,
                 label='Garfield 90/10 (dry, Magboltz)')
    axes[0].set_xlabel('drift E-field [V/cm] (nominal 30 mm gap)')
    axes[0].set_ylabel('v_drift = 30 mm / T_max  [um/ns]')
    axes[1].set_xlabel('drift HV [V]')
    axes[1].set_ylabel('T_max [ns]  (P%d tspan)' % TMAX_PCTL)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('run_61 — measured drift velocity & full-gap transit time '
                 'vs field', fontsize=12)
    fig.tight_layout()
    return _save(fig, 'vdrift.png')


def fig_eff_gap(summ):
    """Effective gap D_eff(E) = v_Garfield(E) * T_max; flatness = constant gap.

    Uses the DRY Garfield 90/10 velocity: D_eff is then an UPPER bound on the
    true gap (real wet gas is slower -> true D_eff = D_eff_dry * v_wet/v_dry,
    i.e. scale down by ~0.8). A field-independent D_eff = the active drift depth
    is constant; a low-field rise flags stronger water slowing at low field."""
    gE, gV = garfield_vE()
    fig, ax = plt.subplots(figsize=(8, 5))
    for det in 'ABCD':
        s = summ[(summ.det == det) & summ.t_max.notna()].sort_values('e_field')
        vref = np.interp(s.e_field, gE, gV)     # um/ns, dry 90/10
        d_eff = vref * s.t_max / 1000.0         # mm
        ax.plot(s.e_field, d_eff, color=DET_COL[det], marker='o', ms=5,
                label=f'Det {det}')
    ax.axhline(D_NOM_MM, color='gray', ls='--', lw=1, label='nominal 30 mm')
    ax.set_xlabel('drift E-field [V/cm]')
    ax.set_ylabel('effective gap D_eff = v_Garfield(E) x T_max  [mm]')
    ax.set_title('effective drift gap vs field  (dry Garfield 90/10 -> UPPER '
                 'bound; real wet gas scales it down ~x0.8)', fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, 'eff_gap.png')


def fig_efficiency(ev):
    """P(3D pair | late) vs drift, per det, resist-pooled + per-resist family."""
    late = ev[ev.probe_class == 'late']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for det in 'ABCD':
        rows = []
        for dr, g in late.groupby('drift'):
            g = g[g[f'readout_{det}']]          # FEU-dropout correction (run_61)
            k = int((g[f'n_pair_{det}'] > 0).sum()); n = len(g)
            p, e = binom_err(k, n)
            rows.append((dr, p, e))
        rows = np.array(rows)
        axes[0].errorbar(rows[:, 0], rows[:, 1], rows[:, 2], color=DET_COL[det],
                         marker='o', ms=5, capsize=2, label=f'Det {det}')
    axes[0].set_xlabel('drift HV [V]'); axes[0].set_ylabel('P(3D pair | late)')
    axes[0].set_title('efficiency vs drift (resist-pooled)', fontsize=10)
    # Det A per-resist family (clean)
    cmap = plt.cm.viridis
    resists = sorted(late.resist.unique())
    for i, r in enumerate(resists):
        rows = []
        for dr, g in late[(late.resist == r) & late.readout_A].groupby('drift'):
            k = int((g['n_pair_A'] > 0).sum()); n = len(g)
            if n < 50:
                continue
            p, e = binom_err(k, n)
            rows.append((dr, p))
        if rows:
            rows = np.array(rows)
            axes[1].plot(rows[:, 0], rows[:, 1], color=cmap(i / max(1, len(resists) - 1)),
                         marker='o', ms=4, label=f'{r} V')
    axes[1].set_xlabel('drift HV [V]'); axes[1].set_ylabel('P(3D pair | late)')
    axes[1].set_title('Det A efficiency vs drift, per resist', fontsize=10)
    axes[1].legend(fontsize=7, title='resist')
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('run_61 — track-finding efficiency vs drift HV', fontsize=12)
    fig.tight_layout()
    return _save(fig, 'efficiency_vs_drift.png')


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, segs, spec = load()
    summ = drift_summary(segs)
    summ.to_csv(os.path.join(OUT, 'drift_summary.csv'), index=False)
    for f in (fig_tspan_dist(segs), fig_drift_spectrum(spec), fig_vdrift(summ),
              fig_eff_gap(summ), fig_efficiency(ev)):
        print('  ->', f)
    gE, gV = garfield_vE()
    print('\nDet A drift-velocity summary (clean M1) vs dry Garfield 90/10:')
    print(f'  {"drift":>6} {"E[V/cm]":>8} {"t0":>5} {"Tmax":>6} {"v_meas":>7} '
          f'{"v_garf":>7} {"ratio":>6}')
    for _, r in summ[summ.det == 'A'].iterrows():
        vg = float(np.interp(r.e_field, gE, gV))
        print(f'  {r.drift:6.0f} {r.e_field:8.0f} {r.t0_daq:5.0f} {r.t_max:6.0f} '
              f'{r.v_drift:7.1f} {vg:7.1f} {r.v_drift / vg:6.2f}')
    hi = summ[(summ.det == 'A') & (summ.e_field >= 190)]
    if not hi.empty:
        vg = np.interp(hi.e_field, gE, gV)
        print(f'  high-field (E>=190 V/cm): v_meas/v_garf = '
              f'{np.mean(hi.v_drift.to_numpy() / vg):.2f} '
              f'(deficit ~= water); saturated v_meas ~ {hi.v_drift.max():.1f} um/ns')


if __name__ == '__main__':
    main()
