#!/usr/bin/env python3
"""Two independent measurements of the gamma-flash charge, compared.

  (1) this package: the waveform on ONE strip, 1 GS/s, no charge-sensitive
      preamplifier -- run 224709, MMA = strip 32 of detector A, cable Y8;
  (2) ntof_july_analysis/flash_charge/: the resistive-layer HV supply current,
      whole chamber, Q = (mean(imon) - median(imon)) / f_pulse.

They share nothing: different electrode, different instrument, different
bandwidth by nine orders of magnitude. Everything below is det A in Ar/iso 90/10.

Writes results_compare.json + two figures.
"""
import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
HVCSV = HERE.parent.parent / 'ntof_july_analysis/flash_charge/results/flash_charge_subruns.csv'
FIG = HERE / 'figures'

N_STRIPS = 512                     # one readout plane
CSA_FULL_SCALE_FC = 600.0          # DREAM CSA largest input range (50/100/200/600)
CSA_FEEDBACK_NA = (9.0, 90.0)      # AICON feedback limit, from the saturation note
CABLE = 1.0092                     # 20 m copper RG-58, charge referred back to the strip


def load_hv():
    rows = []
    with open(HVCSV) as f:
        for r in csv.DictReader(f):
            try:
                r['resist_v'] = float(r['resist_v'])
                r['drift_v'] = float(r['drift_v'])
                r['q'] = float(r['q_per_pulse_nc'])
                r['qerr'] = float(r['q_err_nc'])
            except ValueError:
                continue
            rows.append(r)
    return rows


def efold(v, q):
    k, _ = np.polyfit(np.asarray(v, float), np.log(np.asarray(q, float)), 1)
    return 1.0 / k, float(np.exp(10 * k))


def main():
    hv = load_hv()
    wf = json.load(open(HERE / 'results_709.json'))
    chain = json.load(open(HERE / 'results_chain.json'))
    res = {'inputs': dict(waveform='run 224709, MMA = strip 32 of det A, cable Y8',
                          hv_current=str(HVCSV.relative_to(HERE.parent.parent)),
                          gas='Ar/iso 90/10 throughout', detector='A')}

    # ---------------- axis 1: gain slope, calibration-free -------------------
    slopes = []
    for det in ('A', 'B', 'C'):
        pts = sorted((r['resist_v'], r['q'], r['drift_v']) for r in hv
                     if r['run'] == 'run_57' and r['det'] == det and r['q'] > 0)
        if len(pts) < 5:
            continue
        v = [p[0] for p in pts]
        q = [p[1] for p in pts]
        e, per10 = efold(v, q)
        slopes.append(dict(method='HV supply current', run='run_57', det=det,
                           drift_V=pts[0][2], n=len(pts),
                           v_lo=min(v), v_hi=max(v), e_fold_V=float(e),
                           gain_per_10V=per10))
    for cls in ('dedicated', 'parasitic'):
        f = wf['gain_fits'][f'700_{cls}']
        slopes.append(dict(method='waveform, strip 32', run='224709', det='A',
                           drift_V=700, n=f['n_points'], v_lo=500, v_hi=570,
                           e_fold_V=f['e_fold_V'], gain_per_10V=f['gain_per_10V'],
                           pulse_class=cls))
    res['gain_slope'] = slopes
    hv_a = [s for s in slopes if s['method'].startswith('HV') and s['det'] == 'A'][0]
    wf_par = [s for s in slopes if s.get('pulse_class') == 'parasitic'][0]
    res['slope_agreement'] = dict(
        hv_det_A_e_fold_V=hv_a['e_fold_V'],
        waveform_parasitic_e_fold_V=wf_par['e_fold_V'],
        ratio=wf_par['e_fold_V'] / hv_a['e_fold_V'])

    # ---------------- axis 2: absolute charge at the shared setpoint ---------
    prod = [r for r in hv if r['run'] == 'run_158' and r['det'] == 'A']
    q_hv_nc = float(np.mean([r['q'] for r in prod]))
    q_hv_err = float(np.std([r['q'] for r in prod], ddof=1))
    # the waveform at the same (drift 700, resist 540) point, both pulse classes
    pts = {r['cls']: r for r in wf['scan'] if r['drift'] == 700 and r['resist'] == 540}
    n_d, n_p = pts['dedicated']['n'], pts['parasitic']['n']
    q_ded = pts['dedicated']['charge_pC'] * CABLE
    q_par = pts['parasitic']['charge_pC'] * CABLE
    q_mix = (n_d * q_ded + n_p * q_par) / (n_d + n_p)
    uniform_per_strip_pc = q_hv_nc * 1e3 / N_STRIPS
    res['absolute'] = dict(
        setpoint='drift 700 V, amplification 540 V, det A, Ar/iso 90/10, 2026-08-09',
        hv_run='run_158', hv_nC_per_pulse=q_hv_nc, hv_nC_rms=q_hv_err,
        hv_n_subruns=len(prod),
        waveform_run='224709',
        strip_pC_dedicated=q_ded, strip_pC_parasitic=q_par,
        strip_pC_pulse_mix=q_mix, n_dedicated=n_d, n_parasitic=n_p,
        uniform_expectation_pC_per_strip=uniform_per_strip_pc,
        concentration_factor=q_mix / uniform_per_strip_pc,
        implied_plane_total_nC=q_mix * N_STRIPS / 1e3,
        note='concentration factor = measured strip / (chamber charge spread '
             'uniformly over 512 strips of one plane). It absorbs both the beam '
             'profile at strip 32 and any X/Y charge sharing.')

    # ---------------- axis 3: intensity compression --------------------------
    # waveform: charge ratio against the pickup-measured proton ratio
    wf_ratio = q_ded / q_par
    wf_int = wf['intensity_ratio_from_pkup']
    # HV: isolated-pulse fold, per 1e10 protons, run_79 (HANDOFF section 8.5)
    fold = {'A': dict(parasitic=268.7, dedicated=228.3),
            'C': dict(parasitic=150.6, dedicated=144.4)}
    res['compression'] = dict(
        waveform=dict(charge_ratio=wf_ratio, intensity_ratio=wf_int,
                      per_proton_deficit_pct=100 * (1 - wf_ratio / wf_int)),
        hv_fold={d: dict(per_proton_deficit_pct=100 * (1 - v['dedicated'] / v['parasitic']),
                         source='HANDOFF section 8.5, run_79 isolated-pulse fold')
                 for d, v in fold.items()})

    # ---------------- what it means for DREAM --------------------------------
    strip_over_fs = q_ded * 1e3 / CSA_FULL_SCALE_FC
    per_chan_uniform_fc = uniform_per_strip_pc * 1e3
    e_fold = wf['gain_fits']['700_dedicated']['e_fold_V']
    dv_needed = e_fold * np.log(strip_over_fs)
    res['dream'] = dict(
        csa_full_scale_fC=CSA_FULL_SCALE_FC,
        csa_ranges_fC=[50, 100, 200, 600],
        strip_charge_pC=q_ded,
        strip_over_full_scale=strip_over_fs,
        chamber_average_per_strip_over_full_scale=per_chan_uniform_fc / CSA_FULL_SCALE_FC,
        volts_to_fit_in_range=dv_needed,
        voltage_that_would_be_needed=540 - dv_needed,
        drain_time_ms=[q_ded * 1e-12 / (i * 1e-9) * 1e3 for i in CSA_FEEDBACK_NA],
        feedback_nA=list(CSA_FEEDBACK_NA),
        note='drain time = Q / I_feedback, the time a pinned CSA needs to swallow '
             'the flash charge; brackets the millisecond dead time seen in DREAM.')

    # ============================ figures ====================================
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: the two gain curves, each normalised at 540 V
    hvA = sorted((r['resist_v'], r['q']) for r in hv
                 if r['run'] == 'run_57' and r['det'] == 'A' and r['q'] > 0)
    v_hv = np.array([p[0] for p in hvA])
    q_hvv = np.array([p[1] for p in hvA])
    ref_hv = np.exp(np.interp(540, v_hv, np.log(q_hvv)))
    ax[0].semilogy(v_hv, q_hvv / ref_hv, 'o', ms=3.5, color='#2e7d4f',
                   label='HV supply current, whole chamber (run_57)')
    for cls, c, mk in (('dedicated', '#2f6f9f', 'o'), ('parasitic', '#c0632c', 's')):
        p = sorted((r['resist'], r['charge_pC']) for r in wf['scan']
                   if r['drift'] == 700 and r['cls'] == cls)
        vv = np.array([x[0] for x in p])
        qq = np.array([x[1] for x in p])
        ref = np.exp(np.interp(540, vv, np.log(qq)))
        ax[0].semilogy(vv, qq / ref, mk, ms=4, color=c, ls='-', lw=1.1,
                       label=f'waveform, strip 32, {cls} (224709)')
    ax[0].set_xlabel('detector A amplification voltage (V)')
    ax[0].set_ylabel('charge, normalised at 540 V')
    ax[0].set_title('Same slope, two instruments')
    ax[0].grid(alpha=.3, which='both')
    ax[0].legend(fontsize=7.5)

    # right: the dynamic-range ladder
    items = [('DREAM CSA full scale\n(600 fC setting)', CSA_FULL_SCALE_FC * 1e-3, '#6a6a6a'),
             ('chamber average per strip\n(HV current / 512)', uniform_per_strip_pc, '#2e7d4f'),
             ('strip 32, measured\n(waveform, dedicated)', q_ded, '#2f6f9f'),
             ('whole chamber\n(HV current)', q_hv_nc * 1e3, '#c0632c')]
    ypos = np.arange(len(items))
    ax[1].barh(ypos, [i[1] for i in items], color=[i[2] for i in items], height=.55)
    ax[1].set_yticks(ypos)
    ax[1].set_yticklabels([i[0] for i in items], fontsize=8)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('charge per beam pulse (pC)')
    ax[1].set_title('What the front end is asked to swallow')
    ax[1].grid(alpha=.3, axis='x', which='both')
    for y, (lab, val, _c) in zip(ypos, items):
        mult = val / (CSA_FULL_SCALE_FC * 1e-3)
        tag = f'{val:,.3g} pC' + ('' if y == 0 else f'   ({mult:,.0f}$\\times$ full scale)')
        ax[1].text(val * 1.4, y, tag, va='center', fontsize=8)
    ax[1].set_xlim(0.3, 3e7)
    fig.tight_layout()
    fig.savefig(FIG / 'compare_hv.png', dpi=140)
    plt.close(fig)

    with open(HERE / 'results_compare.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == '__main__':
    main()
