#!/usr/bin/env python3
"""The board closes the loop: strip 32's charge against the chamber's, properly.

The comparison in compare_hv_current.py scaled one strip to the chamber with a
strip count (930 / 465 / 1024) chosen by assumption. This script replaces the
assumption with the measured board (gerbers + solved electrostatics, all in
~/CLionProjects/MX17_Geant) and adds an independent cross-check: the flash's own
intensity compression, read through the sheet capacitance, is a *local
densitometer* at the strip -- it measures the same local charge density as the
waveform lobe, with none of the same calibrations.

Board facts used (sources in SOURCES below):
  - the readout is a 512x512 pad grid, checkerboard-bussed: a "Y strip" is a
    comb of 256 pads on 1.56 mm pitch; X and Y live in the SAME pad plane, so a
    uniform flash splits EXACTLY 50/50 between the views (90-degree symmetry);
  - 85 % of the charge on the resistive layer images onto the pad plane (W2
    boundary; the mesh takes the rest);
  - the ESL is 550 um resistive strips on 800 um pitch along y, draining only at
    the two y-end buses, tau ~ 17 ms at the frozen 2 MOhm/sq -- so on the 9 us
    window the sheet is charge-conserving and the imon current is the whole
    per-pulse charge, delivered ~1 s later through the supply;
  - sheet-to-readout capacitance c' = 0.4985 uF/m^2, which converts a charge
    surface density into an amplification-field sag and therefore into the
    measured dedicated/parasitic compression.

Consumes results_imon.json + results_709.json (+ the npz for the cycle budget).
Writes results_board.json + figures/board_*.png.
"""
import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path('/media/dylan/data/x17/ntof_mm_flash')
FIG = HERE / 'figures'

SOURCES = {
    'checkerboard': 'MX17_Geant response/common/channel_map.py -- L5/L6 gerber stubs: '
                    'Y channel = row comb of 256 pads (col+row even), X = column comb (odd)',
    'capture': 'MX17_Geant response/solver/wpot_w2.py + design/report/V6_PAD_GAPS_2026-08-08.md: '
               'prompt channel capture 0.842-0.853 (W2, floating inter-pad channels); '
               'W1 (grounded gaps) 0.665',
    'cprime': 'MX17_Geant response/solver/wpot.py: c\' = eps0(1/g + eps_r/d_eff), '
              'g=150 um, d_eff=70.5 um (50 um kapton eps 3.5 + 18.8 um glue eps 3.2)',
    'esl': 'MX17_Geant RESPONSE_SIM_PLAN.md section 1: ESL strips 550/250 um on 800 um '
           'pitch along y, bused at both y-ends only; tau_drain = L^2/(pi^2 D) = 17 ms '
           'at 2 MOhm/sq (band 1.42-2.56 measured via T2b)',
    'diffusion': 'MX17_Geant solver: D = 1/(rho_s c\') = 1.0 m^2/s at 2 MOhm/sq',
    'geometry': 'common/mx17_active_area.py (det A = mx17_3): pitch 398.58/512 mm, '
                'Y passivation 18.0/18.7 mm -> 465 live Y channels, live 361.9 x 398.6 mm',
}

# ---- board constants ---------------------------------------------------------
CAPTURE = 0.85            # W2 0.842-0.853; the 1 % spread is irrelevant here
CAPTURE_W1 = 0.665
VIEW_SPLIT = 0.5          # exact, by checkerboard symmetry, for uniform illumination
CPRIME_F_M2 = 4.985e-7
RHO_S_MOHM = 2.0          # frozen; measured band 1.42-2.56
D_M2_S = 1.0 / (RHO_S_MOHM * 1e6 * CPRIME_F_M2)
ESL_LEN_M = 0.412
TAU_DRAIN_S = ESL_LEN_M ** 2 / (np.pi ** 2 * D_M2_S)
CABLE = 1.0092            # charge referred back to the strip (Appendix B)

PITCH_CM = 398.58 / 512 / 10
STRIP_LEN_CM = 39.858
N_Y_LIVE = 465
A_BAND_CM2 = PITCH_CM * STRIP_LEN_CM            # one Y channel's collection band
LIVE_AREA_CM2 = STRIP_LEN_CM * 36.188
EFF_SHARE = CAPTURE * VIEW_SPLIT / N_Y_LIVE     # of the chamber charge, per live Y channel

FLASH_WIN_S = 9e-6

MV_PER_COUNT = 5043.7915 / 65536
PC_PER_CT_NS = MV_PER_COUNT * 1e-3 * 1e-9 / 50.0 * 1e12

C_DED, C_PAR, C_HV, C_REF = '#2f6f9f', '#c0632c', '#2e7d4f', '#6a6a6a'


def deficit_of_x(x, rho):
    """Per-proton dedicated-vs-parasitic deficit for parasitic sheet charge x
    (in units of c'V_e), intensity ratio rho, from dQ = e^{-Q/(c'V_e)} dP."""
    x = np.asarray(x, float)
    return 1.0 - np.log1p(rho * x) / (rho * np.log1p(x))


def x_of_deficit(d, rho):
    """Invert deficit_of_x by bisection (monotonic in x)."""
    if d <= 0:
        return 0.0
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if deficit_of_x(mid, rho) < d:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ri = json.load(open(HERE / 'results_imon.json'))
    r9 = json.load(open(HERE / 'results_709.json'))
    rho = r9['intensity_ratio_from_pkup']
    # e-fold of the gain at the strip, measured on the same run; the band spans
    # the four fitted slopes (both instruments, both pulse classes)
    ve_mid, ve_lo, ve_hi = 21.7, 20.2, 24.1
    sig0 = CPRIME_F_M2 * ve_mid * 1e8               # pC/cm^2 per unit x
    mad9 = {(r['drift'], r['resist'], r['cls']): (r['charge_mad'], r['n'])
            for r in r9['scan']}

    rows = []
    for p in ri['plateaus']:
        if not p.get('wf_parasitic_pC') or not p.get('wf_dedicated_pC') or p['q_nC'] <= 0:
            continue
        q_d = p['wf_dedicated_pC'] * CABLE
        q_p = p['wf_parasitic_pC'] * CABLE
        n_d, n_p = p['wf_dedicated_n'], p['wf_parasitic_n']
        q_mix = (n_d * q_d + n_p * q_p) / (n_d + n_p)
        expected = p['q_nC'] * 1e3 * EFF_SHARE      # uniform share, pC
        r_meas = q_d / q_p
        d_meas = 1.0 - r_meas / rho
        # error on the deficit from the per-class MADs
        e = np.nan
        k_d = mad9.get((p['drift'], p['resist'], 'dedicated'))
        k_p = mad9.get((p['drift'], p['resist'], 'parasitic'))
        if k_d and k_p:
            rel = np.hypot(1.4826 * k_d[0] / np.sqrt(k_d[1]) / q_d * CABLE,
                           1.4826 * k_p[0] / np.sqrt(k_p[1]) / q_p * CABLE)
            e = r_meas * rel / rho
        # the three densitometers, parasitic class (the linear-most one)
        sig_lobe = q_p / (CAPTURE * VIEW_SPLIT * A_BAND_CM2)          # pC/cm^2
        q_par_chamber = p['q_nC'] * 1e3 * (n_d + n_p) / (n_d * r_meas + n_p)
        sig_avg = q_par_chamber / LIVE_AREA_CM2
        x_lobe = np.expm1(sig_lobe / sig0)
        x_avg = np.expm1(sig_avg / sig0)
        x_meas = x_of_deficit(d_meas, rho)
        rows.append(dict(
            drift=p['drift'], resist=p['resist'], label=p['label'],
            q_imon_nC=p['q_nC'], q_imon_err_nC=p['q_err_nC'], q_strip_mix_pC=q_mix,
            q_strip_ded_pC=q_d, q_strip_par_pC=q_p,
            expected_uniform_pC=expected, residual=q_mix / expected,
            deficit_meas=d_meas, deficit_err=None if np.isnan(e) else float(e),
            deficit_pred_lobe=float(deficit_of_x(x_lobe, rho)),
            deficit_pred_uniform=float(deficit_of_x(x_avg, rho)),
            sigma_par_lobe_pC_cm2=sig_lobe, sigma_par_avg_pC_cm2=sig_avg,
            sigma_par_compression_pC_cm2=float(sig0 * np.log1p(x_meas)),
            enhancement_lobe=sig_lobe / sig_avg,
            enhancement_compression=float(sig0 * np.log1p(x_meas) / sig_avg)
            if x_meas > 0 else None,
        ))

    resid = np.array([r['residual'] for r in rows])
    wp = next(r for r in rows if r['drift'] == 700 and r['resist'] == 540)

    # ---- cycle budget on the strip's own record ------------------------------
    d = np.load(DATA / 'mm_224709.npz')
    wall = d['wall']
    secs = wall[:, 0] // 10000 * 3600 + (wall[:, 0] // 100) % 100 * 60 + wall[:, 0] % 100
    stats = d['stats']
    ev = stats[:, 0].astype(int)
    z = d['zs']
    q_zs_per_bunch = np.zeros(len(d['bunch']))
    np.add.at(q_zs_per_bunch, z[:, 0].astype(int), z[:, 5] * PC_PER_CT_NS)
    plats = list(csv.DictReader(open(DATA / 'hv_plateaus_224709.csv')))
    p540 = next(p for p in plats
                if int(p['A_drift_V']) == 700 and int(p['A_resist_V']) == 540)
    m = (secs[ev] >= int(p540['start_s']) + 45) & (secs[ev] <= int(p540['end_s']))
    tail_pc = float(np.median(q_zs_per_bunch[ev][m]))
    p570 = next(p for p in plats
                if int(p['A_drift_V']) == 700 and int(p['A_resist_V']) == 570)
    m570 = (secs[ev] >= int(p570['start_s']) + 45) & (secs[ev] <= int(p570['end_s']))
    tail570_pc = float(np.median(q_zs_per_bunch[ev][m570]))

    # ---- drift supply during the scan ----------------------------------------
    imon_rows = list(csv.DictReader(open(DATA / 'imon_224709.csv')))
    drift_i = np.array([float(r['A_drift_imon']) for r in imon_rows])

    # ---- the resist leakage baseline: steps at every drift-HV move -----------
    def _secs(ts):
        hh, mm, ss = ts[11:13], ts[14:16], ts[17:19]
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    it = np.array([_secs(r['timestamp']) for r in imon_rows], float)
    ii = np.array([float(r['A_resist_imon']) for r in imon_rows])
    leak_seq, worst_half_shift = [], 0.0
    for p in plats:
        m = (it >= int(p['start_s']) + 45) & (it <= int(p['end_s']))
        if m.sum() < 60:
            continue
        x = ii[m]
        h = len(x) // 2
        worst_half_shift = max(worst_half_shift,
                               abs(np.median(x[:h]) - np.median(x[h:])) * 1e3)
        leak_seq.append((int(p['A_drift_V']), int(p['A_resist_V']),
                         float(np.median(x)) * 1e3))
    steps = []
    for a, b in zip(leak_seq, leak_seq[1:]):
        if a[0] != b[0]:
            steps.append(dict(drift_move=f'{a[0]} -> {b[0]} V',
                              leak_before_nA=a[2], leak_after_nA=b[2]))
    res_leakage = dict(
        steps_at_drift_moves=steps,
        branch_700_decay_nA=[leak_seq[0][2], leak_seq[13][2]],
        worst_within_plateau_median_shift_nA=float(worst_half_shift),
        note='every drift-HV move kicks the resist leakage up by 0.15-0.57 uA, '
             'which then relaxes over tens of minutes (the 700 V branch shows the '
             'same relaxation from the pre-scan ramp, 60 -> 16 nA). The direction '
             'excludes a resistive path from the cage divider (drift went DOWN and '
             'leakage went UP): it is charging/relaxation current from moving the '
             'cathode. It is subtracted per plateau by the median, its worst '
             'within-plateau movement is ~24 nA and roughly linear (mean-median '
             'cancels a linear drift to first order), and the bootstrap error bars '
             'include what remains.')

    # ---- can the imon split dedicated from parasitic on this run? ------------
    # Clock offset imon vs n_TOF bunch clock: lag scan of the detrended excess
    # against the 1 s-binned pulse train.
    ex = np.full(len(it), np.nan)
    for p in plats:
        m = (it >= int(p['start_s']) + 45) & (it <= int(p['end_s']))
        if m.sum() >= 60:
            ex[m] = ii[m] - np.median(ii[m])
    t0 = int(it.min())
    n_bins = int(it.max()) - t0 + 1
    pulse_bin = np.zeros(n_bins)
    for b in secs:
        k = int(b) - t0
        if 0 <= k < n_bins:
            pulse_bin[k] += 1
    exb = np.zeros(n_bins)
    have = np.zeros(n_bins, bool)
    okm = ~np.isnan(ex)
    for tt, e in zip(it[okm], ex[okm]):
        exb[int(tt) - t0] += e
        have[int(tt) - t0] = True
    lags = {lag: float(np.corrcoef(exb[have], np.roll(pulse_bin, lag)[have])[0, 1])
            for lag in range(-10, 11)}
    best_lag = max(lags, key=lags.get)
    # class per bunch, same recipe as analyse_709
    order = np.argsort(secs)
    bs = np.sort(secs) + best_lag
    pk_all = np.zeros(len(d['bunch']))
    pk_all[d['pkup'][:, 0].astype(int)] = d['pkup'][:, 1]
    pkb = pk_all[order]
    vb = pkb > 1000
    spl = 0.5 * (np.percentile(pkb[vb], 10) + np.percentile(pkb[vb], 90))
    cls_b = np.where(pkb > spl, 1, 0)
    cls_b[~vb] = -1
    gp = np.diff(bs, prepend=-1e9)
    gn = np.diff(bs, append=1e9)
    iso_strict = (gp >= 6) & (gn >= 4) & (cls_b >= 0)
    res_class = dict(
        clock_offset_s=int(best_lag), lag_corr=lags[best_lag],
        runner_up_corr=float(sorted(lags.values())[-2]),
        n_isolated_strict=dict(dedicated=int((cls_b[iso_strict] == 1).sum()),
                               parasitic=int((cls_b[iso_strict] == 0).sum())),
        verdict='NO on this run: with gaps wide enough that no neighbour response '
                'enters the fold window (>=6 s before, >=4 s after), the surviving '
                'isolated pulses are essentially all dedicated -- the PS supercycle '
                'places parasitic pulses 1.2-2.4 s from their neighbours, so '
                'isolation and intensity class are entangled. The least-squares '
                'split is the documented ill-conditioning trap (July handoff 8.6). '
                'The imon CAN make the split when the 1 s time base is '
                'reconstructed to ms accuracy: run_79 at the production point '
                'gives a per-proton deficit of 15 % (det A) / 4 % (det C), same '
                'sign and size as the waveform\'s 7.6 %.')

    res = dict(
        sources=SOURCES,
        board=dict(capture=CAPTURE, capture_w1=CAPTURE_W1, view_split=VIEW_SPLIT,
                   n_y_live=N_Y_LIVE, eff_share=EFF_SHARE,
                   uniform_multiplier=1.0 / EFF_SHARE,
                   cprime_uF_m2=CPRIME_F_M2 * 1e6, rho_s_MOhm_sq=RHO_S_MOHM,
                   D_m2_s=D_M2_S, tau_drain_ms=TAU_DRAIN_S * 1e3,
                   spread_in_window_mm=float(np.sqrt(2 * D_M2_S * FLASH_WIN_S) * 1e3),
                   e_fold_V=dict(mid=ve_mid, lo=ve_lo, hi=ve_hi),
                   sig0_pC_cm2=sig0, intensity_ratio=rho),
        working_point=dict(
            setpoint='drift 700 V, amplification 540 V',
            q_imon_nC=wp['q_imon_nC'],
            q_strip_mix_pC=wp['q_strip_mix_pC'],
            expected_uniform_pC=wp['expected_uniform_pC'],
            residual=wp['residual'],
            residual_w1=wp['residual'] * CAPTURE / CAPTURE_W1,
            sigma_par=dict(chamber_avg=wp['sigma_par_avg_pC_cm2'],
                           from_compression=wp['sigma_par_compression_pC_cm2'],
                           from_lobe=wp['sigma_par_lobe_pC_cm2']),
            enhancement=dict(lobe=wp['enhancement_lobe'],
                             compression=wp['enhancement_compression']),
            deficit=dict(measured=wp['deficit_meas'],
                         predicted_from_lobe=wp['deficit_pred_lobe'],
                         predicted_if_uniform=wp['deficit_pred_uniform']),
            sag_V=dict(chamber_avg=wp['sigma_par_avg_pC_cm2'] * 1e-8 / CPRIME_F_M2,
                       local_par=wp['sigma_par_compression_pC_cm2'] * 1e-8 / CPRIME_F_M2,
                       local_ded=wp['q_strip_ded_pC']
                       / (CAPTURE * VIEW_SPLIT * A_BAND_CM2) * 1e-8 / CPRIME_F_M2),
            chamber_by_class_nC=dict(
                parasitic=wp['sigma_par_avg_pC_cm2'] * LIVE_AREA_CM2 / 1e3,
                dedicated=wp['sigma_par_avg_pC_cm2'] * LIVE_AREA_CM2 / 1e3
                * wp['q_strip_ded_pC'] / wp['q_strip_par_pC'],
                note='the imon mix unfolded with the counted class mix and the '
                     'waveform-measured dedicated/parasitic ratio'),
            drain_return=dict(
                current_nA=wp['q_strip_mix_pC'] * 1e-3 / TAU_DRAIN_S,
                voltage_uV=wp['q_strip_mix_pC'] * 1e-12 / TAU_DRAIN_S * 50 * 1e6,
                note='mean image-return current while the sheet drains: '
                     'Q/tau through 50 ohm -- unmeasurably small, so the positive '
                     'lobe cleanly measures the local image'),
        ),
        residual_constancy=dict(n=len(rows), mean=float(resid.mean()),
                                sd=float(resid.std(ddof=1)),
                                spread_pct=float(100 * resid.std(ddof=1) / resid.mean())),
        cycle_budget=dict(
            tail_over_flash_540=tail_pc / wp['q_strip_mix_pC'] * CABLE and
            tail_pc / (wp['q_strip_mix_pC'] / CABLE),
            tail_pC_540=tail_pc, tail_pC_570=tail570_pc,
            note='ZS blocks >30 us summed per bunch, median over the plateau; '
                 'above-threshold only (this run ZS is 10x coarser in mV than July)'),
        drift_supply=dict(mean_minus_median_uA=float(drift_i.mean() - np.median(drift_i)),
                          distinct_values=sorted(set(np.round(drift_i, 3).tolist())),
                          note='pure divider-current quantisation tracking the drift '
                               'setting; no beam response -- the avalanche charge '
                               'does not return through the drift supply'),
        leakage=res_leakage,
        imon_class_split=res_class,
        scan=rows,
    )

    # ============================ figures ====================================
    # -- 1. the ledger: three densitometers + the residual across the scan -----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    dens = [('chamber average\n(HV current / live area)', wp['sigma_par']
             if isinstance(wp.get('sigma_par'), (int, float)) else wp['sigma_par_avg_pC_cm2'], C_HV),
            ('local at strip 32,\nfrom the compression', wp['sigma_par_compression_pC_cm2'], C_PAR),
            ('local at strip 32,\nfrom the waveform lobe', wp['sigma_par_lobe_pC_cm2'], C_DED)]
    ypos = np.arange(len(dens))
    ax[0].barh(ypos, [x[1] for x in dens], color=[x[2] for x in dens], height=.5)
    for y, (lab, val, _c) in zip(ypos, dens):
        mult = val / dens[0][1]
        ax[0].text(val + 8, y, f'{val:.0f}' + ('' if y == 0 else f'   ({mult:.1f}x)'),
                   va='center', fontsize=9)
    ax[0].set_yticks(ypos)
    ax[0].set_yticklabels([x[0] for x in dens], fontsize=9)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('flash charge surface density, parasitic pulses (pC/cm$^2$)')
    ax[0].set_title('Three densitometers, 700 / 540 V')
    ax[0].set_xlim(0, max(x[1] for x in dens) * 1.28)
    ax[0].grid(alpha=.3, axis='x')

    for drift, mk in ((700, 'o'), (600, 's'), (500, '^')):
        pts = sorted((r['resist'], r['residual']) for r in rows if r['drift'] == drift)
        if not pts:
            continue
        ax[1].plot([p[0] for p in pts], [p[1] for p in pts], mk, ms=5,
                   color=C_DED, mfc='none' if drift != 700 else C_DED,
                   label=f'drift {drift} V')
    ax[1].axhline(resid.mean(), color=C_REF, lw=1)
    ax[1].axhspan(resid.mean() - resid.std(ddof=1), resid.mean() + resid.std(ddof=1),
                  color=C_REF, alpha=.15, lw=0)
    ax[1].text(0.02, 0.93, f'{resid.mean():.1f} $\\pm$ {resid.std(ddof=1):.1f} '
               f'across {len(rows)} plateaus', transform=ax[1].transAxes, fontsize=9)
    ax[1].axhline(1.0, color='k', lw=.8, ls=':')
    ax[1].text(0.02, 0.04, 'uniform illumination', transform=ax[1].transAxes,
               fontsize=8, color=C_REF)
    ax[1].set_xlabel('detector A amplification voltage (V)')
    ax[1].set_ylabel('measured strip / uniform share')
    ax[1].set_title('The residual is one constant')
    ax[1].set_ylim(0, resid.max() * 1.25)
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8, loc='center right')
    fig.tight_layout()
    fig.savefig(FIG / 'board_ledger.png', dpi=140)
    plt.close(fig)

    # -- 2. the compression closure: deficit vs the strip's own lobe density ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sig_grid = np.geomspace(20, 4000, 200)
    for ve, ls, lw, lab in ((ve_mid, '-', 1.5, f'sheet-charging model, $V_e$={ve_mid} V'),
                            (ve_lo, '-', 0, None), (ve_hi, '-', 0, None)):
        pass
    s0m, s0l, s0h = (CPRIME_F_M2 * v * 1e8 for v in (ve_mid, ve_lo, ve_hi))
    ax.plot(sig_grid, 100 * deficit_of_x(np.expm1(sig_grid / s0m), rho), '-',
            color='k', lw=1.5, label='sheet-charging prediction (no free parameters)')
    ax.fill_between(sig_grid,
                    100 * deficit_of_x(np.expm1(sig_grid / s0h), rho),
                    100 * deficit_of_x(np.expm1(sig_grid / s0l), rho),
                    color='k', alpha=.12, lw=0,
                    label=f'gain e-fold {ve_lo}-{ve_hi} V')
    for drift, mk, filled in ((700, 'o', True), (600, 's', False), (500, '^', False)):
        pts = [(r['sigma_par_lobe_pC_cm2'], 100 * r['deficit_meas'],
                100 * (r['deficit_err'] or 0)) for r in rows if r['drift'] == drift]
        if not pts:
            continue
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es, fmt=mk, ms=5.5, color=C_DED,
                    mfc=C_DED if filled else 'none', lw=0, elinewidth=1,
                    capsize=2, label=f'measured, drift {drift} V')
    # where the same plateaus would sit if the strip carried only a uniform share
    un = [(r['sigma_par_avg_pC_cm2'], 100 * r['deficit_meas']) for r in rows]
    ax.plot([u[0] for u in un], [u[1] for u in un], 'x', ms=4.5, color=C_REF,
            label='same bunches, if the strip saw the chamber average')
    w = next(r for r in rows if r['drift'] == 700 and r['resist'] == 540)
    ax.plot(w['sigma_par_lobe_pC_cm2'], 100 * w['deficit_meas'], 'o', ms=13,
            mfc='none', mec='k', mew=1.2)
    ax.annotate('700 / 540 V\n(working point)',
                (w['sigma_par_lobe_pC_cm2'], 100 * w['deficit_meas']),
                textcoords='offset points', xytext=(-30, 16), fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('parasitic flash density at the strip (pC/cm$^2$)')
    ax.set_ylabel('per-proton deficit, dedicated vs parasitic (%)')
    ax.set_title('The compression is a local densitometer, and it agrees with the lobe')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(-18, 45)
    fig.tight_layout()
    fig.savefig(FIG / 'board_compression.png', dpi=140)
    plt.close(fig)

    # -- 3. the board, and the two current loops -------------------------------
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(12.2, 5.4),
                                 gridspec_kw=dict(width_ratios=[1.5, 1]))
    aL.set_xlim(0, 11.6)
    aL.set_ylim(0, 8.3)
    aL.axis('off')
    aL.set_title('Cross-section (not to scale) — where the flash charge goes', fontsize=10)

    def layer(y0, h, color, label, x0=0.4, w=6.9, fs=8):
        aL.add_patch(Rectangle((x0, y0), w, h, fc=color, ec='none'))
        aL.text(x0 + w + 0.15, y0 + h / 2, label, va='center', fontsize=fs)

    layer(7.3, 0.22, '#555555', 'mesh (grounded) — takes ~15 % of the image')
    layer(5.7, 1.6, '#eef3f7', 'amplification gap 150 µm')
    # ESL strip cross-sections (they run along y = into the page)
    for i in range(9):
        aL.add_patch(Rectangle((0.4 + i * 0.77, 5.42), 0.54, 0.24,
                               fc='#8a5a2c', ec='none'))
    aL.text(7.45, 5.54, 'resistive strips (ESL), along y ⊗\n550/250 µm on 800 µm pitch',
            va='center', fontsize=8)
    layer(4.55, 0.4, '#e8d9a0', 'coverlay 50 µm + glue 19 µm   (c′ = 0.50 µF/m²)')
    # pad plane: checkerboard parity alternates along x
    for i in range(9):
        c = C_DED if i % 2 == 0 else C_PAR
        aL.add_patch(Rectangle((0.42 + i * 0.77, 4.0), 0.66, 0.42, fc=c, ec='none',
                               alpha=0.8 if i != 4 else 1.0))
    aL.text(7.45, 4.21, 'pad plane, 680 µm pads:\nY combs / X combs interleaved',
            va='center', fontsize=8)
    layer(3.2, 0.6, '#dfe4e8', 'FR4 + bus layers (Y along x, X along y)')

    # avalanche down, ions up
    aL.add_patch(FancyArrowPatch((2.65, 7.2), (2.65, 5.75), arrowstyle='-|>',
                                 mutation_scale=14, color='#b3352e', lw=2))
    aL.text(2.82, 6.45, 'avalanche e⁻\n(full drift sweep ≤ ~2 µs)',
            fontsize=8, color='#b3352e', va='center')
    aL.add_patch(FancyArrowPatch((2.0, 5.75), (2.0, 7.2), arrowstyle='-|>',
                                 mutation_scale=10, color='#999999', lw=1.2))
    aL.text(1.85, 6.45, 'ions', fontsize=7.5, color='#777777',
            va='center', ha='right')
    # image arrow, in the ESL-coverlay gap
    aL.add_patch(FancyArrowPatch((3.55, 5.40), (3.55, 4.58), arrowstyle='-|>',
                                 mutation_scale=10, color='k', lw=1.2))
    aL.text(3.72, 5.08, '85 % of the image → pads, exactly 50/50 X/Y',
            fontsize=8, va='center')
    # loop 1: the measured pad -> 50 ohm
    aL.add_patch(FancyArrowPatch((3.83, 3.95), (3.83, 1.75), arrowstyle='-|>',
                                 mutation_scale=12, color=C_DED, lw=1.8))
    aL.add_patch(Rectangle((3.48, 1.2), 0.7, 0.5, fc='none', ec=C_DED, lw=1.5))
    aL.text(4.35, 1.45, '50 Ω  (n_TOF DAQ, 1 GS/s)\nsees the LOCAL image — µs timescale',
            fontsize=8, color=C_DED, va='center')
    # loop 2: drain along strip to the end bus -> imon (elbow down the left edge)
    aL.add_patch(FancyArrowPatch((1.3, 5.54), (0.22, 5.54), arrowstyle='-|>',
                                 mutation_scale=12, color=C_HV, lw=1.8))
    aL.plot([0.22, 0.22], [5.54, 3.0], color=C_HV, lw=1.4)
    aL.text(0.06, 2.75,
            'drains along y to the end buses (τ ≈ 17 ms)\n'
            '→ HV supply, imon = the WHOLE charge,\n     averaged over ~1 s',
            fontsize=8, color=C_HV, va='top')

    # right: top view, one Y comb among the checkerboard + ESL strips
    aR.set_xlim(-0.7, 12.6)
    aR.set_ylim(-1.15, 8.75)
    aR.axis('off')
    aR.set_title('Top view — what “strip 32” actually is', fontsize=10)
    for r_ in range(8):
        for c_ in range(8):
            par = (r_ + c_) % 2 == 0
            on_row = r_ == 3 and par
            col = C_DED if on_row else ('#b9cbda' if par else '#e3c4ad')
            aR.add_patch(Rectangle((c_ + 0.06, r_ + 0.06), 0.88, 0.88,
                                   fc=col, ec='none'))
    for i in range(11):
        x = -0.55 + i * 0.82
        aR.add_patch(Rectangle((x, -0.05), 0.56, 8.1, fc='#8a5a2c', alpha=.18, ec='none'))
    aR.add_patch(Rectangle((-0.62, 8.12), 9.0, 0.28, fc='#666666'))
    aR.add_patch(Rectangle((-0.62, -0.42), 9.0, 0.28, fc='#666666'))
    aR.text(3.9, 8.26, 'ESL end bus → HV', fontsize=7.5, color='w',
            ha='center', va='center')
    aR.text(3.9, -0.28, 'ESL end bus → HV', fontsize=7.5, color='w',
            ha='center', va='center')
    aR.add_patch(FancyArrowPatch((7.6, 3.5), (8.6, 3.5), arrowstyle='-|>',
                                 mutation_scale=10, color=C_DED, lw=1.5))
    aR.text(8.75, 3.5, 'one Y channel =\na comb of 256 pads\non 1.56 mm pitch\n→ 50 Ω',
            fontsize=8, color=C_DED, va='center')
    aR.text(8.75, 1.0, 'X combs\n(other parity)', fontsize=8, color=C_PAR, va='center')
    aR.text(8.75, 6.4, 'ESL strips ∥ y\n(800 µm pitch,\nbeating the 780 µm pads)',
            fontsize=8, color='#8a5a2c', va='center')
    fig.tight_layout()
    fig.savefig(FIG / 'board_stack.png', dpi=140)
    plt.close(fig)

    # -- 4. the final comparison: absolute per-strip charges, and the ladder ---
    OPERATING_BAND_V = (540, 560)   # Aug production point .. the gas's target point
    fig, (aC, aB) = plt.subplots(1, 2, figsize=(11.6, 4.8),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    r700 = sorted((r for r in rows if r['drift'] == 700), key=lambda r: r['resist'])
    vv = [r['resist'] for r in r700]
    mix = [r['q_strip_mix_pC'] for r in r700]
    aC.plot(vv, [r['q_strip_ded_pC'] for r in r700], 'o-', color=C_DED, ms=4.5,
            lw=1.3, label='waveform, strip 32, dedicated')
    aC.plot(vv, [r['q_strip_par_pC'] for r in r700], 's-', color=C_PAR, ms=4.5,
            lw=1.3, label='waveform, strip 32, parasitic')
    aC.plot(vv, mix, ':', color=C_REF, lw=1.2, label='waveform, pulse mix')
    aC.errorbar(vv, [1e3 * r['q_imon_nC'] * EFF_SHARE for r in r700],
                yerr=[1e3 * r['q_imon_err_nC'] * EFF_SHARE for r in r700],
                fmt='D-', color=C_HV, ms=4, lw=1.2, capsize=2,
                label='HV current $\\to$ uniform share per strip (mix)')
    enh = wp['enhancement_compression']
    aC.plot(vv, [1e3 * r['q_imon_nC'] * EFF_SHARE * enh for r in r700], 'D--',
            color=C_HV, ms=4, lw=1.1, mfc='none',
            label=f'HV current $\\to$ local at strip 32 (compression, $\\times${enh:.1f})')
    # the two gaps, annotated where the curves are clear of each other
    i0 = vv.index(510)
    y_share = 1e3 * r700[i0]['q_imon_nC'] * EFF_SHARE
    aC.annotate('', xy=(510, mix[i0]), xytext=(510, y_share),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1))
    aC.text(511.5, np.sqrt(mix[i0] * y_share),
            f'$\\times${wp["residual"]:.1f}', fontsize=9)
    aC.text(542, 200, 'the dashed local estimate runs\n~40 % below the pulse mix',
            fontsize=8, color='#3a3a3a')
    aC.axhline(0.6, color='#b3352e', lw=1.4)
    aC.text(544, 0.75, 'DREAM CSA full scale (600 fC)', fontsize=8, color='#b3352e')
    aC.axvspan(*OPERATING_BAND_V, color=C_REF, alpha=.14, lw=0)
    aC.text(0.5 * sum(OPERATING_BAND_V), 2.1, 'operating\nregion',
            fontsize=8, color=C_REF, ha='center')
    aC.set_yscale('log')
    aC.set_ylim(0.4, 20000)
    aC.set_xlabel('detector A amplification voltage (V)')
    aC.set_ylabel('flash charge per strip (pC)')
    aC.set_title('Both methods, absolute, drift 700 V')
    aC.grid(alpha=.3, which='both')
    aC.legend(fontsize=7.2, loc='upper left')

    items = [('DREAM CSA full scale\n(600 fC setting)', 0.6, C_REF, '0.6 pC'),
             ('chamber-average strip\n(HV current, board accounting)',
              wp['expected_uniform_pC'], C_HV,
              f"{wp['expected_uniform_pC']:.0f} pC   ({wp['expected_uniform_pC']/0.6:,.0f}$\\times$)"),
             ('strip 32, measured\n(waveform, dedicated)', wp['q_strip_ded_pC'], C_DED,
              f"{wp['q_strip_ded_pC']:.0f} pC   ({wp['q_strip_ded_pC']/0.6:,.0f}$\\times$)"),
             ('whole chamber\n(HV current)', wp['q_imon_nC'] * 1e3, C_PAR,
              f"{wp['q_imon_nC']:.0f} nC   ({wp['q_imon_nC']*1e3/0.6/1e3:,.0f}k$\\times$)")]
    ypos = np.arange(len(items))
    aB.set_axisbelow(True)
    aB.barh(ypos, [x[1] for x in items], color=[x[2] for x in items], height=.55)
    for y, (_lab, val, _c, tag) in zip(ypos, items):
        aB.text(val * 1.5, y, tag, va='center', fontsize=8.5)
    aB.axvline(0.6, color='#b3352e', lw=1.2, ls='--')
    aB.set_yticks(ypos)
    aB.set_yticklabels([x[0] for x in items], fontsize=8.5)
    aB.invert_yaxis()
    aB.set_xscale('log')
    aB.set_xlim(0.2, 3e6)
    aB.set_xlabel('charge per beam pulse (pC), 700 / 540 V')
    aB.set_title('What the front end is asked to swallow')
    aB.grid(alpha=.3, axis='x', which='both')
    fig.tight_layout()
    fig.savefig(FIG / 'compare_final.png', dpi=140)
    plt.close(fig)

    # -- 5. the argument as a cartoon, one panel per step ----------------------
    d_pct = 100 * wp['deficit_meas']
    d_uni = 100 * wp['deficit_pred_uniform']
    sag_loc = wp['sigma_par_compression_pC_cm2'] * 1e-8 / CPRIME_F_M2
    sag_avg = wp['sigma_par_avg_pC_cm2'] * 1e-8 / CPRIME_F_M2
    ratio_meas = rho * (1 - wp['deficit_meas'])
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.5))
    for a in axs:
        a.axis('off')
        a.set_xlim(0, 10)
        a.set_ylim(0, 10)

    # panel 1: the capacitor that throttles its own gain
    a = axs[0]
    a.set_title('1 — arriving charge lowers the voltage\nthat amplifies it', fontsize=10)
    a.add_patch(Rectangle((1.0, 8.0), 5.8, 0.25, fc='#555555', ec='none'))
    a.text(7.05, 8.1, 'mesh (grounded)', fontsize=8, va='center')
    for x in (2.0, 3.4, 4.8):
        a.add_patch(FancyArrowPatch((x, 9.6), (x, 6.55), arrowstyle='-|>',
                                    mutation_scale=11, color='#b3352e', lw=1.6))
    a.text(5.35, 9.3, 'flash charge', fontsize=8, color='#b3352e')
    a.text(7.05, 7.15, 'amplification gap:\ngain ×e per 22 V', fontsize=8, va='center')
    a.add_patch(Rectangle((1.0, 6.0), 5.8, 0.3, fc='#8a5a2c', ec='none'))
    a.text(3.9, 6.62, '−  −  −  −  −  −', fontsize=10,
           color='#b3352e', ha='center', va='center')
    a.text(7.05, 6.1, 'resistive layer\n(+540 V)', fontsize=8, va='center')
    # the capacitor to ground, and the blocked drain
    a.plot([5.4, 5.4], [6.0, 5.35], color='k', lw=1.2)
    a.plot([4.9, 5.9], [5.35, 5.35], color='k', lw=1.6)
    a.plot([4.9, 5.9], [5.05, 5.05], color='k', lw=1.6)
    a.plot([5.4, 5.4], [5.05, 4.5], color='k', lw=1.2)
    for w_, y_ in ((0.5, 4.5), (0.34, 4.32), (0.18, 4.14)):
        a.plot([5.4 - w_, 5.4 + w_], [y_, y_], color='k', lw=1.2)
    a.text(6.1, 4.85, "c′ = 0.50 µF/m²", fontsize=8.5, va='center')
    a.add_patch(FancyArrowPatch((1.0, 6.15), (0.25, 6.15), arrowstyle='-|>',
                                mutation_scale=10, color=C_HV, lw=1.4))
    a.text(0.1, 5.55, 'drain to the HV supply\ntakes ~17 ms —\nblocked during the flash',
           fontsize=7.5, color=C_HV, va='top')
    a.text(5.0, 2.6, 'charge piles up  →  V drops by ΔV = σ / c′\n'
           '→  charge arriving late is amplified less',
           fontsize=9, ha='center', va='center',
           bbox=dict(fc='#f4f4f5', ec='#c0c0c0', lw=.8, boxstyle='round,pad=0.5'))

    # panel 2: the x2 experiment
    a = axs[1]
    a.set_title('2 — the beam doubles the input\nevery 36 s', fontsize=10)
    y0, hu = 1.6, 2.9                       # baseline and height of one charge unit
    a.plot([0.6, 9.4], [y0, y0], color='k', lw=1)
    a.add_patch(Rectangle((1.3, y0), 1.5, hu, fc=C_PAR, ec='none'))
    a.text(2.05, y0 - 0.45, 'parasitic\n1× protons', fontsize=8, ha='center', va='top')
    a.text(2.05, y0 + hu + 0.25, '1.00', fontsize=9, ha='center')
    a.add_patch(Rectangle((4.6, y0), 1.5, rho * hu, fill=False, ec=C_REF,
                          lw=1.2, ls='--'))
    a.text(5.35, y0 + rho * hu + 0.25, f'expected {rho:.2f}', fontsize=8,
           ha='center', color=C_REF)
    a.add_patch(Rectangle((4.6, y0), 1.5, ratio_meas * hu, fc=C_DED, ec='none'))
    a.text(5.35, y0 - 0.45, 'dedicated\n2× protons', fontsize=8, ha='center', va='top')
    a.text(5.75, y0 + ratio_meas * hu - 0.35, f'{ratio_meas:.2f}', fontsize=9,
           ha='center', color='w')
    a.add_patch(Rectangle((4.6, y0 + ratio_meas * hu), 1.5,
                          (rho - ratio_meas) * hu, fc='none', ec='#b3352e',
                          lw=1.0, hatch='///'))
    a.annotate(f'{d_pct:.1f} % per proton missing\n= the sag at work',
               xy=(6.1, y0 + 0.5 * (rho + ratio_meas) * hu), xytext=(6.6, 8.3),
               fontsize=8.5, color='#b3352e',
               arrowprops=dict(arrowstyle='->', color='#b3352e', lw=1))
    a.text(5.0, 0.15, 'a ratio on one strip → 50 Ω, capture, cable all cancel',
           fontsize=8.5, ha='center', style='italic')

    # panel 3: reading the shortfall backwards
    a = axs[2]
    a.set_title('3 — the shortfall is a ruler\nfor the local density', fontsize=10)

    def box(x, y, text, ec, fs=8):
        a.add_patch(Rectangle((x, y), 2.7, 1.7, fc='#fbfbfc', ec=ec, lw=1.2))
        a.text(x + 1.35, y + 0.85, text, fontsize=fs, ha='center', va='center')

    def arrow(x0, x1, y):
        a.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
                                    mutation_scale=11, color='k', lw=1.1))

    a.text(0.15, 8.9, 'if this strip saw only the chamber average:', fontsize=8.5,
           color=C_REF)
    box(0.15, 6.9, f'{wp["sigma_par_avg_pC_cm2"]:.0f} pC/cm²\n(uniform share)', C_REF)
    arrow(2.9, 3.4, 7.75)
    box(3.45, 6.9, f'sag {sag_avg:.1f} V', C_REF)
    arrow(6.2, 6.7, 7.75)
    box(6.75, 6.9, f'shortfall {d_uni:.0f} %', C_REF)
    a.text(8.1, 6.45, f'✗ measured: {d_pct:.1f} %', fontsize=9, color='#b3352e',
           ha='center', va='top')
    a.text(0.15, 4.6, 'what the measured shortfall requires:', fontsize=8.5,
           color=C_DED)
    box(0.15, 2.6, f'shortfall {d_pct:.1f} %\n(measured)', C_DED)
    arrow(2.9, 3.4, 3.45)
    box(3.45, 2.6, f'needs sag\n≈ {sag_loc:.0f} V', C_DED)
    arrow(6.2, 6.7, 3.45)
    box(6.75, 2.6, f'{wp["sigma_par_compression_pC_cm2"]:.0f} pC/cm²\n'
        f'= {wp["enhancement_compression"]:.1f}× average ✓', '#2e7d4f')
    a.text(5.0, 0.9, f'(the waveform lobe says {wp["sigma_par_lobe_pC_cm2"]:.0f} pC/cm² '
           f'= {wp["enhancement_lobe"]:.1f}× — same answer, different ruler)',
           fontsize=8, ha='center', style='italic')
    fig.tight_layout()
    fig.savefig(FIG / 'board_cartoon.png', dpi=140)
    plt.close(fig)

    with open(HERE / 'results_board.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'scan'}, indent=1))


if __name__ == '__main__':
    main()
