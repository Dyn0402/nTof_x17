#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_passport_report.py -- figures and report.html for `track_passport`.

Generated from the tables in <calo>/passport/, so re-running the analysis and
then this updates the numbers, the figures and the text together.  Figures
are referenced as relative links (figures/x.png).

    python -m ntof_calorimetry.make_passport_report
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'sept26_prelim_analysis'))

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ntof_calorimetry import track_passport as TP  # noqa: E402

PP = TP.PP
FIG = PP / 'figures'
G4_STYLE = dict(lw=1.6, ls='-', alpha=0.9)


def _arr(s) -> np.ndarray:
    return np.asarray(json.loads(s), float)


# ------------------------------------------------------------------ figures
def fig_path() -> Path:
    PC = pd.read_csv(PP / 'path_length_cosmic.csv')
    PL = pd.read_csv(PP / 'path_length.csv')
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=False)
    rows = []
    # cosmic, MM and wall
    ax = axs[0]
    for _, r in PC.iterrows():
        x, y = _arr(r.sec_med), _arr(r.dep_med)
        mk = 'o' if r.layer == 'MM gap' else 's'
        ax.plot(x, y, mk, color=fs.DET_COLOR[r.arm], ms=6, mew=1.4, mfc='none' if r.layer == 'wall' else fs.DET_COLOR[r.arm],
                label=f'{r.arm} {r.layer}: {r.slope:.2f} ± {r.err:.2f}')
        rows += [dict(panel='cosmic', arm=r.arm, layer=r.layer, sec=a, rel=b) for a, b in zip(x, y)]
    xs = np.linspace(1.0, 1.45, 20)
    ax.plot(xs, xs, color=fs.MUTED, lw=1, ls='--', label='∝ sec (path length)')
    ax.set_xlabel('sec θ (track, from the MM)')
    ax.set_ylabel('deposit / first bin')
    ax.set_title('Cosmic muons: deposit follows the path', fontsize=10, loc='left')
    ax.legend(fontsize=7.5, frameon=False)
    # beam: MM and wall, data vs Geant4
    for ax, layers, ttl in ((axs[1], ('MM gap',), 'Beam electrons, MM gap'),
                            (axs[2], ('wall',), 'Beam electrons, wall (97 mm on)')):
        for _, r in PL[PL.layer.isin(layers)].iterrows():
            x, y = _arr(r.sec_med), _arr(r.dep_med)
            if not len(y):
                continue
            y = y / y[0]
            c = fs.DET_COLOR[r.arm]
            if r.source == 'data':
                ax.plot(x, y, 'o', color=c, ms=5.5, label=f'{r.arm} data: {r.slope:.2f} ± {r.err:.2f} (sec<1.3: {r.slope_core:.2f})')
            else:
                ax.plot(x, y, color=c, **G4_STYLE, label=f'{r.arm} Geant4: {r.slope:.2f} ± {r.err:.2f} (sec<1.3: {r.slope_core:.2f})')
            rows += [dict(panel=ttl, arm=r.arm, layer=r.layer, source=r.source, sec=a, rel=b) for a, b in zip(x, y)]
        ax.plot(xs, xs, color=fs.MUTED, lw=1, ls='--')
        ax.set_xlabel('sec θ in the gap')
        ax.set_title(ttl, fontsize=10, loc='left')
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fs.save(fig, FIG / 'path_length', data=pd.DataFrame(rows))


def fig_energy() -> Path:
    EN = pd.read_csv(PP / 'energy.csv')
    subsets = ['all', 'sec < 1.1', 'sec >= 1.1', 'one lepton', 'several', 'T < 3 MeV', 'T 3-5 MeV', 'T 5-7 MeV',
               'T >= 7 MeV']
    feats = list(TP.FEATURES)
    cols = ['#9aa5b1', '#6a7583', fs.DET_COLOR['A'], fs.ACCENT]
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    w = 0.2
    for i, f in enumerate(feats):
        e = EN[EN.features == f].set_index('subset').reindex(subsets)
        ax.bar(np.arange(len(subsets)) + (i - 1.5) * w, e.w68, w * 0.9, color=cols[i], label=f)
    ax.set_xticks(np.arange(len(subsets)), subsets, fontsize=8.5)
    ax.set_ylabel('68 % half-width of T̂ − T  [MeV]')
    ax.set_title('Kinetic energy at the gap (Geant4, triggered tracks, cross-validated kNN)', fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    return fs.save(fig, FIG / 'energy', data=EN)


def fig_two_particles() -> Path:
    """Median wall dE/dx in bins of the MM gap dE/dx, data vs Geant4, with
    the error of the median: flat means the two do not see the same thing."""
    P = pd.read_parquet(PP / 'two_particles_points.parquet')
    edges = np.array([0, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 2.6, 10])
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    rows = []
    for k, arm in enumerate(TP.ARMS):
        for src in ('data', 'Geant4'):
            p = P[(P.arm == arm) & (P.source == src)]
            b = np.digitize(p.mm, edges)
            for i in range(1, len(edges)):
                v = p.wall[b == i]
                if len(v) < 25:
                    continue
                rows.append(dict(arm=arm, source=src, mm=float(p.mm[b == i].median()), wall_med=float(v.median()),
                                 err=float(1.2533 * v.std() / np.sqrt(len(v))), n=len(v)))
    R = pd.DataFrame(rows)
    for k, arm in enumerate(TP.ARMS):
        c = fs.DET_COLOR[arm]
        for src, off, kw in (('data', 0.985, dict(fmt='o', ms=5)), ('Geant4', 1.015, dict(fmt='s', ms=4, mfc='none', mew=1.2))):
            r = R[(R.arm == arm) & (R.source == src)]
            ax.errorbar(r.mm * off * (1 + 0.02 * k), r.wall_med, r.err, color=c, lw=1, capsize=0,
                        label=f'{arm} {src}', **kw)
    ax.axhline(1, color=fs.MUTED, lw=1, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('MM gap dE/dx / median')
    ax.set_ylabel('median wall dE/dx / median')
    ax.set_title('The wall does not follow the MM deposit, in data or Geant4', fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    return fs.save(fig, FIG / 'two_particles', data=R)


def fig_multi_truth() -> Path:
    """Geant4: the MM deposit split by the truth (one lepton / several)."""
    P = pd.read_parquet(PP / 'two_particles_points.parquet')
    P = P[P.source == 'Geant4']
    bins = np.geomspace(0.3, 8, 40)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
    rows = []
    for ax, col, lab in ((axs[0], 'mm', 'MM gap dE/dx / median'), (axs[1], 'wall', 'wall dE/dx / median')):
        for m, nm, c in ((False, 'one lepton in the gap', fs.MUTED), (True, 'several particles', fs.TRACK)):
            v = P[P.multi == m][col]
            h, _ = np.histogram(v, bins)
            ax.stairs(h / h.sum(), bins, color=c, lw=1.6, label=nm)
            rows += [dict(var=col, multi=m, lo=a, hi=b, frac=f) for a, b, f in zip(bins[:-1], bins[1:], h / h.sum())]
        ax.set_xscale('log')
        ax.set_xlabel(lab)
        ax.legend(fontsize=8, frameon=False)
    axs[0].set_ylabel('fraction of tracks')
    axs[0].set_title('Geant4: the MM separates them', fontsize=10, loc='left')
    axs[1].set_title('… the wall does not', fontsize=10, loc='left')
    fig.tight_layout()
    return fs.save(fig, FIG / 'multi_truth', data=pd.DataFrame(rows))


def fig_predictions() -> Path:
    D = pd.read_parquet(PP / 'passport_data.parquet')
    S = pd.read_parquet(PP / 'passport_sim.parquet')
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.4))
    rows = []
    tb = np.arange(0, 11, 0.4)
    pb = np.linspace(0, 1, 21)
    for arm in TP.ARMS:
        c = fs.DET_COLOR[arm]
        d = D[(D.arm == arm) & D.mm_ok]
        s = S[S.armL == arm]
        for ax, col, b in ((axs[0], 'T_hat', tb), (axs[1], 'p_multi', pb)):
            hd, _ = np.histogram(d[col], b)
            hs, _ = np.histogram(s[col], b)
            x = 0.5 * (b[1:] + b[:-1])
            ax.plot(x, hd / hd.sum(), 'o', color=c, ms=4, label=f'{arm} data (n={len(d)})')
            ax.stairs(hs / hs.sum(), b, color=c, lw=1.4, label=f'{arm} Geant4')
            rows += [dict(arm=arm, var=col, x=a, data=u, g4=v) for a, u, v in zip(x, hd / hd.sum(), hs / hs.sum())]
        ht, _ = np.histogram(s['T'], tb)
        axs[0].stairs(ht / ht.sum(), tb, color=c, lw=1, ls=':', label=f'{arm} Geant4 true T')
        # neutron energy
        e = D[D.arm == arm].E_n_meV
        eb = np.geomspace(0.3, 15, 30)
        he, _ = np.histogram(e, eb)
        axs[2].stairs(he / he.sum(), eb, color=c, lw=1.4, label=arm)
    axs[0].set_xlabel('T at the gap  [MeV]')
    axs[1].set_xlabel('P(several particles in the gap)')
    axs[2].set_xlabel('E_n from the time of flight  [meV]')
    axs[2].set_xscale('log')
    axs[0].set_ylabel('fraction of tracks')
    axs[0].set_title('Estimated electron energy', fontsize=10, loc='left')
    axs[1].set_title('Multi-particle probability', fontsize=10, loc='left')
    axs[2].set_title('Neutron energy (≥ 12 ms)', fontsize=10, loc='left')
    for ax in axs:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fs.save(fig, FIG / 'predictions', data=pd.DataFrame(rows))


def card(r: pd.Series, i: int) -> Path:
    """One track's passport: side view of the stack with the track and the
    deposits, and the numbers."""
    fig = plt.figure(figsize=(9.6, 3.9))
    ax = fig.add_axes([0.05, 0.13, 0.42, 0.74])
    # side view: z along the chamber normal (mesh at 0, drift gap 0-30 mm), u across
    lay = [('MM gap', 0, 30, '#dfe6ee'), ('wall', TP.Z_WALL, TP.Z_WALL + 3, '#f2d9b5'),
           ('plastic', TP.Z_PLAS, TP.Z_PLAS + 20, '#f5c38c'), ('liquid', TP.Z_LIQ, TP.Z_LIQ + 60, '#e7d2ec')]
    for nm, z0, z1, col in lay:
        ax.axvspan(z0, z1, color=col, lw=0)
        ax.text(0.5 * (z0 + z1), 205, nm, ha='center', va='bottom', fontsize=8, color=fs.MUTED)
    z = np.linspace(0, TP.Z_LIQ + 60, 50)
    u = r.x_local + r.tanx * z
    ax.plot(z, u, color=fs.TRACK, lw=1.6)
    # MM drift profile: charge along the depth, drawn as a strip along the track
    prof = np.asarray(r.x_prof, float) + np.asarray(r.y_prof, float)
    if prof.size:
        # 60 ns per sample (20 samples, 1200 ns); 140 ns is the template peak
        # (pair_event_display: depth = (t - 140 ns - t0) v)
        t = np.arange(prof.size) * 60.0
        dz = np.clip((t - 140.0 - r.t0) * r.v_drift / 1000.0, 0, 30)
        pz = np.clip(prof / max(prof.max(), 1), 0, 1)
        ax.scatter(dz, r.x_local + r.tanx * dz, s=8 + 60 * pz, color=fs.DET_COLOR[r.arm], alpha=0.6, lw=0)
    fired = dict(wall=r.e_wall_g4, plastic=r.e_plas_g4, liquid=r.e_liq if r.liq_fired else 0)
    for (nm, z0, z1, _), e in zip(lay[1:], fired.values()):
        if e and e > 0:
            uu = r.x_local + r.tanx * 0.5 * (z0 + z1)
            ax.text(0.5 * (z0 + z1), uu + 18, f'{e:.2f} MeV', ha='center', fontsize=8, color=fs.INK)
    ax.text(15, r.x_local + 18, f'{r.E_mm:.1f} keV', ha='center', fontsize=8, color=fs.INK)
    ax.set_xlim(-10, TP.Z_LIQ + 70)
    ax.set_ylim(-230, 230)
    ax.set_xlabel('distance from the mesh along the chamber normal  [mm]')
    ax.set_ylabel('x  [mm]')
    ax.set_title(f'{r.example}: {r.arm} {r.subrun} event {int(r.event_id)}', fontsize=10, loc='left')
    # the numbers
    a2 = fig.add_axes([0.52, 0.05, 0.46, 0.9])
    a2.axis('off')
    lines = [
        ('Time', f'{r.t_since_flash_ns / 1e6:.2f} ms after the flash  →  E_n ≈ {r.E_n_meV:.2f} meV'),
        ('Direction', f'θ = {r.theta_deg:.1f}°, φ = {r.phi_deg:.0f}°  (sec θ = {r.sec:.3f})'),
        ('Position', f'x, y = {r.x_local:.0f}, {r.y_local:.0f} mm on the chamber'),
        ('Pointing', f'{r.dca_axis_mm:.0f} mm from the capsule axis'
                     f'{"  (points at the capsule)" if r.to_capsule else ""}'),
        ('MM gap', f'{r.E_mm:.1f} keV ± {100 * r.sigma_mm:.0f} %   ({r.q_rel:.2f} × median at normal incidence)'),
        ('Wall', f'{r.e_wall_g4:.2f} MeV in 3 mm × sec  (group {int(r.wgrp)})'),
        ('Plastic', f'{r.e_plas_g4:.2f} MeV, bar {int(r.bar)}{"  SATURATED" if r.p_sat else ""}'
                    f'   (MIP × sec = {3.40 * r.sec:.2f})'),
        ('Liquid', f'{r.e_liq:.2f} MeV (MIP scale): got through the plastic' if r.liq_fired else 'no signal'),
        ('Energy', f'T at the gap ≈ {r.T_hat:.1f} ± {r.T_sd:.1f} MeV'),
        ('Particles', f'P(several in the gap) = {r.p_multi:.2f};  P(pair conversion) = {r.p_conv:.2f}'),
        ('Class', r.cls),
    ]
    y = 0.95
    for k, v in lines:
        a2.text(0.0, y, k, fontsize=9, color=fs.MUTED, va='top', transform=a2.transAxes)
        a2.text(0.2, y, v, fontsize=9, color=fs.INK, va='top', transform=a2.transAxes)
        y -= 0.083
    return fs.save(fig, FIG / f'card_{i:02d}', data=r.drop(['x_prof', 'y_prof']).to_frame().T)


# ------------------------------------------------------------------- report
def _tbl(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    h = '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns) + '</tr></thead><tbody>'
    for _, r in df.iterrows():
        h += '<tr>' + ''.join(
            f'<td>{("—" if pd.isna(r[c]) else fmt[c].format(r[c]) if c in fmt else html.escape(str(r[c])))}</td>'
            for c in df.columns) + '</tr>'
    return h + '</tbody></table>'


def report() -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fs.use()
    figs = dict(path=fig_path(), energy=fig_energy(), two=fig_two_particles(), multi=fig_multi_truth(),
                pred=fig_predictions())
    EX = pd.read_pickle(PP / 'examples.pkl')
    B = pd.read_pickle(TP.EDEP / f'beam_{TP.RUN}.pkl')[['subrun', 'event_id', 'arm', 'track_id', 'v_drift_um_ns']]
    B['subrun'] = B.subrun.astype(str)
    EX = EX.merge(B, on=['subrun', 'event_id', 'arm', 'track_id'], how='left').rename(columns={'v_drift_um_ns': 'v_drift'})
    cards = [card(r, i) for i, (_, r) in enumerate(EX.iterrows())]
    S = json.loads((PP / 'summary.json').read_text())
    PC = pd.read_csv(PP / 'path_length_cosmic.csv')
    PL = pd.read_csv(PP / 'path_length.csv')
    TPt = pd.read_csv(PP / 'two_particles.csv')
    EN = pd.read_csv(PP / 'energy.csv')
    OR = pd.read_csv(PP / 'origin.csv')
    cal = S['cal']

    def en(f, s):
        x = EN[(EN.features == f) & (EN.subset == s)]
        return float(x.w68.iloc[0]) if len(x) else np.nan

    def orr(t, f, samp='triggered'):
        x = OR[(OR['sample'] == samp) & (OR.target == t) & (OR.features == f)]
        return float(x.auc.iloc[0])

    mmc = PC[PC.layer == 'MM gap']
    wc = PC[PC.layer == 'wall']
    g4t = TPt[TPt.source == 'Geant4']
    dat = TPt[TPt.source == 'data']
    head = [
        ('MM gap deposit, per track', ', '.join(f'{a}: ±{100 * cal[a]["sigma_total"]:.0f} %' for a in TP.ARMS),
         'beam electrons, after the in-situ gain map; sim chain ⊕ measured extra spread'),
        ('Path length (cosmics)', ', '.join(f'MM {r.arm} sec^{r.slope:.2f}±{r.err:.2f}' for _, r in mmc.iterrows()),
         f'Heed expects sec^{mmc.expected.iloc[0]:.2f}; wall ' +
         ', '.join(f'{r.arm} sec^{r.slope:.2f}±{r.err:.2f}' for _, r in wc.iterrows())),
        ('Energy at the gap', f'{en("plastic", "all"):.2f} → {en("plastic+wall+angle", "all"):.2f} MeV',
         f'w68, plastic alone → with the track angle; tracks above 25°: {en("plastic", "sec >= 1.1"):.2f} → '
         f'{en("plastic+wall+angle", "sec >= 1.1"):.2f} MeV'),
        ('Several particles in the gap', f'AUC {g4t.auc_multi_mm.mean():.2f} (MM) vs {g4t.auc_multi_wall.mean():.2f} (wall)',
         f'Geant4; {100 * g4t.frac_multi.mean():.0f} % of triggered tracks'),
        ('Capsule vs elsewhere', f'AUC {orr("capsule", "tracking (position, direction)", "all gap tracks"):.2f}',
         'Geant4, from the track geometry alone (all tracks through the gap)'),
    ]
    tiles = ''.join(f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
                    f'<div class="s">{html.escape(s)}</div></div>' for k, v, s in head)
    cls = pd.DataFrame(S['classes']).T.fillna(0).reset_index().rename(columns={'index': 'arm'})
    # the low-energy class, data vs Geant4 (plastic below 0.8 MIP x sec: stopped early)
    G4 = pd.read_parquet(PP / 'g4.parquet')
    G4 = G4[G4.trig]
    PD = pd.read_parquet(PP / 'passport_data.parquet')
    lowT = pd.DataFrame([dict(arm=a, data=float((PD[PD.arm == a].e_plas_g4 < 0.8 * 3.4 * PD[PD.arm == a].sec).mean()),
                              geant4=float((G4[G4.armL == a].e_plas < 0.8 * 3.4 * G4[G4.armL == a].sec).mean()),
                              liquid_fired_data=float(PD[PD.arm == a].liq_fired.mean()))
                         for a in TP.ARMS])
    cal_rows = pd.DataFrame([dict(arm=a, k_beam=c['k_beam'], sim_rsd=c['sim_rsd'], sim_bias=c['sim_bias'],
                                  data_over_sim=c['scale'], scale_err=c['scale_err'], sigma_extra=c['sigma_x'],
                                  sigma_total=c['sigma_total'], mip_keV=c['mip_keV'], n=c['n_data'])
                             for a, c in cal.items()])
    body = f"""
<h1>Track passport: everything the set-up measures about one track</h1>
<p class="sub">run_145, chambers A and C, ≥ 12 ms after the flash, single tracks on their own arm's trigger
({S['n_data']:,} tracks, {S['n_data_mm']:,} with a usable MM deposit) &middot; Geant4 neutron-beam sample
({S['n_g4']:,} gap tracks, {S['n_g4_trig']:,} triggered) &middot; generated by
<code>ntof_calorimetry/make_passport_report.py</code></p>

<div class="verdict"><b>Verdict.</b> Tracking and calorimetry describe the same particle and add to each other.
On cosmic muons, the MM gap and the wall deposits grow with the path length the track predicts.
On beam electrons, the track angle sharpens the plastic's energy for steep tracks.
The MM gap deposit flags tracks that carry several particles, which the scintillators cannot do.
The geometry alone separates capsule-born tracks from the rest.
<b>What the set-up cannot do:</b> tell e⁺ from e⁻ (no field), or measure energy above ~6 MeV.
The plastic is 20 mm thick, so faster particles get through, and the liquid behind it is threshold-limited.</div>

<div class="tiles">{tiles}</div>

<h2>1. One track, every layer</h2>
<p>Each card is one data track. The side view shows the stack along the chamber normal, with the track
extrapolated from the MM. The blue dots are the MM's drift profile: charge against depth along the track.
The panel on the right lists what each layer says about the track.
The energy and the particle probabilities come from Geant4-trained estimators (sections 4–5) applied to the
measured numbers. They are model-dependent.</p>
{''.join(f'<figure><img src="figures/{c.name}" alt="track passport card"></figure>' for c in cards)}

<h2>2. Path length: do the tracks and the counters see the same particle?</h2>
<p>The tracking predicts the path in every layer: 30 mm × sec θ in the gap, and 3 mm × sec θ in the wall
for a track that does not scatter on the way. On <b>cosmic muons</b> (run_149, MIPs, no scattering) both
deposits follow it. On <b>beam electrons</b> they do not, and Geant4 agrees.
The electrons scatter in the MM's material and in 10 cm of air before the wall, so the gap angle says
little about the angle at the wall. Low-energy and multi-particle tracks are also steeper, which pulls
the MM slope above 1.</p>
<figure><img src="figures/path_length.png" alt="deposit vs sec theta"><figcaption>Median deposit in
bins of sec θ, normalised to the first bin; the power-law slope is in the legend. Dashed: ∝ sec θ.</figcaption></figure>
{_tbl(pd.concat([PC.assign(source='cosmic data')[['arm', 'layer', 'source', 'n', 'slope', 'err', 'expected']],
                 PL[['arm', 'layer', 'source', 'n', 'slope', 'err', 'slope_core', 'err_core']]]),
      dict(slope='{:.2f}', err='{:.2f}', expected='{:.2f}', slope_core='{:.2f}', err_core='{:.2f}'))}

<h2>3. MM gap deposit on beam electrons: the calibration</h2>
<p>This closes the open item in MM_EDEP.md. The window curve and the ADC-per-keV constant k now come from
the <b>beam</b> simulation, which was generated at the beam gain, and no longer from the cosmic one.
With them the beam simulation's bias goes from −0.46 to ≈ 0. The data/sim scale is the forward fold on the
late beam data after the in-situ gain map. The median deposit at normal incidence then reads
{', '.join(f'{a}: {cal[a]["mip_keV"]:.1f} keV' for a in TP.ARMS)} of ionisation in the simulation's truth
(Heed, W = 27.5 eV).</p>
{_tbl(cal_rows, dict(k_beam='{:.0f}', sim_rsd='{:.3f}', sim_bias='{:+.3f}', data_over_sim='{:.3f}',
                     scale_err='{:.3f}', sigma_extra='{:.2f}', sigma_total='{:.3f}', mip_keV='{:.2f}'))}

<h2>4. Two particles: the MM sees them, the wall does not</h2>
<p>In Geant4, {100 * g4t.frac_multi.mean():.0f} % of triggered tracks have several particles sharing the gap
deposit (the dominant one carries &lt; 80 %): a delta ray, the partner of a Compton electron, or a converted pair.
The MM's gap deposit picks these out (AUC {g4t.auc_multi_mm.mean():.2f}). The wall's deposit does not
(AUC {g4t.auc_multi_wall.mean():.2f}): the second particle is soft and rarely reaches the wall, 97 mm
further on. So the MM and the wall dE/dx barely correlate, in the data (Spearman
{', '.join(f'{r.arm} {r.spearman:+.3f}' for _, r in dat.iterrows())}) as in Geant4
({', '.join(f'{r.arm} {r.spearman:+.3f}' for _, r in g4t.iterrows())}). The joint tail (both above 1.6 × median)
is lifted by the same factor in both
(data {', '.join(f'{r.lift:.2f}' for _, r in dat.iterrows())}, Geant4 {', '.join(f'{r.lift:.2f}' for _, r in g4t.iterrows())}).</p>
<figure><img src="figures/multi_truth.png" alt="MM and wall deposit by truth"></figure>
<figure><img src="figures/two_particles.png" alt="wall vs MM in data and Geant4"><figcaption>Points: data;
lines: Geant4 with the measured resolutions.</figcaption></figure>
{_tbl(TPt, {c: '{:.3f}' for c in TPt.columns if TPt[c].dtype.kind == 'f'})}

<h2>5. Energy: what each measurement adds</h2>
<p>This is the kinetic energy of the leading lepton at the gap, for tracks that fire the plastic (the trigger).
It uses k-nearest-neighbour regression on Geant4 with 5-fold cross-validation. The <b>track angle</b> is the
largest gain after the plastic itself, because the plastic deposit of a lepton that gets through scales with
its path. The gain is in steep tracks and in multi-particle tracks. The MM gap deposit adds nothing to the
energy at these energies: a 1–10 MeV electron is minimum-ionising in 30 mm of gas, within the relativistic rise.</p>
<figure><img src="figures/energy.png" alt="energy resolution by feature set"></figure>
{_tbl(EN, dict(w68='{:.2f}', rel_w68='{:.3f}', bias='{:+.2f}'))}

<h2>6. Origin and process</h2>
<p>These are Geant4 truths that the measurements can partly recover:
<b>capsule</b> (the lepton was made in the capsule's Al or CFRP, i.e. by a capture γ from the capsule),
<b>conv</b> (the lepton comes from a pair conversion, not a Compton scatter) and <b>multi</b> (several
particles in the gap). Each is a cross-validated kNN, and the score is the AUC.</p>
{_tbl(OR, dict(base_rate='{:.3f}', auc='{:.3f}'))}

<h2>7. The estimators on the data</h2>
<p>The Geant4-trained estimators applied to every data track, set against the same estimators on Geant4.
The neutron energy is shown for completeness. At ≥ 12 ms it is all sub-thermal: 0.4–14 meV at 19.5 m.</p>
<figure><img src="figures/predictions.png" alt="estimator outputs data vs Geant4"></figure>
<p>Track classes in the data (fraction per arm):</p>
{_tbl(cls, {c: '{:.3f}' for c in cls.columns if c != 'arm'})}
<p>Two classes can be checked against Geant4 directly from the counters. The share of tracks whose plastic
deposit is below 0.8 × MIP × sec θ (the lepton stopped early) agrees, including C's much larger share,
which comes from the geometry. The liquid fires behind a few per cent of A's tracks and almost never behind
C's: the C liquid's MIP is only ~1.3 × its ZS threshold.</p>
{_tbl(lowT, dict(data='{:.3f}', geant4='{:.3f}', liquid_fired_data='{:.4f}'))}

<h2>What this does not rule out</h2>
<ul>
<li>The energy and the particle probabilities are Geant4 estimators applied to data. They are only as good as
the Geant4 sample: thermal neutrons on the 2 cm nose, {S['n_g4_trig']:,} triggered tracks. The data's
counter scales were moved onto Geant4's (wall_plastic scale fits, including C1's +15 % offset),
so a wrong Geant4 energy scale moves the data with it.</li>
<li>"Several particles" is a Geant4 definition (dominant track &lt; 80 % of the gap deposit). Its data rate is
inferred, not measured. The data agree with Geant4 on the MM–wall joint tail, but that tail is small.</li>
<li>The Geant4 sample has no liquid. "Through the plastic" in Geant4 is T &gt; 6 MeV at the gap, an
approximation. In the data it is the liquid firing above its ZS threshold, which is threshold-limited
(MIP at 1.1–1.6 × threshold).</li>
<li>The MM deposit uses only contained-drift tracks on A (f &gt; 1.35); C is mostly an extrapolation of the
first ~80 % of the drift (MM_EDEP.md).</li>
<li>Beam path-length slopes depend on the selection (the trigger requires the plastic), not only on the
physics. They are comparisons with Geant4, not tests of a law.</li>
</ul>
"""
    css = """
:root{color-scheme:light dark;--bg:#fbfbfa;--surface:#fff;--line:#e3e2df;--ink:#14140f;--ink2:#55534c;--acc:#8a3f8f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15151a;--surface:#1c1c22;--line:#33333c;--ink:#f2f2ef;--ink2:#b6b4ab;--acc:#c58fca}}
:root[data-theme="dark"]{--bg:#15151a;--surface:#1c1c22;--line:#33333c;--ink:#f2f2ef;--ink2:#b6b4ab;--acc:#c58fca}
*{box-sizing:border-box}body{margin:0;padding:28px 16px 64px;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto}h1{font-size:1.6rem;margin:0 0 6px}h2{font-size:1.12rem;margin:38px 0 12px;
padding-bottom:6px;border-bottom:1px solid var(--line)}.sub{color:var(--ink2);font-size:.88rem}
.verdict{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--acc);padding:14px 16px;
border-radius:6px;margin:18px 0}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.tile .k{font-size:.78rem;color:var(--ink2)}.tile .v{font-size:1.05rem;font-weight:600;margin:3px 0}
.tile .s{font-size:.76rem;color:var(--ink2)}figure{margin:16px 0}img{max-width:100%;background:#fff;border-radius:4px}
figcaption{font-size:.84rem;color:var(--ink2)}table{border-collapse:collapse;font-size:.8rem;display:block;
overflow-x:auto;margin:10px 0}th,td{padding:3px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:500}code{font-size:.85em}
"""
    out = PP / 'report.html'
    out.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                   f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                   f'<title>Track passport</title><style>{css}</style></head><body><main>{body}</main></body></html>')
    print(f'-> {out}')
    return out


if __name__ == '__main__':
    report()
