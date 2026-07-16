"""12c_export_pss_calib.py — Export the plastic-PMT HV-scan calibration to the
DAQ repo (<daq_dir>/calibrations/pss/ + pss_trigger/), following the wal_*
export pattern.

From the 12 cache (plastic HV scan, run224466) this writes:

pss/
  gain_vs_hv_<run>.json        per-PMT gain curves: fitted power-law indices
                               (combined + per pass), the full measured curve
                               (V, median) per step, operating point, CAEN
                               channel, ADC->mV
  hv_equalization_<run>.json   the equalization: target, per-PMT suggested V
  global_gain_slide_<run>.csv  THE operational table: per-PMT voltages that
                               scale the (equalized) fleet response by a common
                               factor g, using each PMT's own index
  figures/                     gain_curves, equalization, coinc_spectra_linear

pss_trigger/
  threshold_scan_<run>.csv     common threshold (post-equalization mV) ->
                               per-PMT tagged efficiency + summed late rate
  thresholds_<run>.json        efficiencies at the 10 mV hardware floor
                               (equalized + current gains) and the HV sets
                               that recover 99%/95% tagged eff at 10 mV
  figures/                     threshold_scan

Usage: python 12c_export_pss_calib.py [run_stem] [daq_dir]
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224466'
DAQ = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / 'PycharmProjects/nTof_x17_DAQ'
BASE = Path(__file__).parent
OUT = DAQ / 'calibrations' / 'pss'

d = np.load(BASE / 'cache' / f'12_hvscan_{RUN_STEM}.npz')
CEN = np.sqrt(d['amp_edges'][:-1] * d['amp_edges'][1:])
LABELS = [str(s) for s in d['step_labels']]
VOLTS = d['step_volts']
PASS = d['step_pass']
GATING = d['step_gating']
SB = float(d['sb_scale'])
N_STEP = len(LABELS)
FAC = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_STEM}.json').read_text())['factors']

NOMINAL_V = {('A', 0): 1325, ('A', 1): 1275, ('B', 0): 1325, ('B', 1): 1300,
             ('C', 0): 1300, ('C', 1): 1300, ('D', 0): 1300, ('D', 1): 1300}
CAEN = {('A', 0): '7:0', ('A', 1): '7:1', ('B', 0): '7:2', ('B', 1): '7:3',
        ('C', 0): '7:4', ('C', 1): '7:5', ('D', 0): '7:6', ('D', 1): '7:7'}
SCAN_RANGE = (1200, 1600)
GAIN_FACTORS = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]


def coinc_median(i, ai, b):
    sub = np.clip(d['pss_mip'][i, ai, 0, b] - SB * d['pss_mip'][i, ai, 1, b], 0, None)
    c = np.cumsum(sub)
    return float(CEN[np.searchsorted(c, c[-1] / 2)]) if c[-1] > 200 else np.nan


def fit_n(ai, b, passes):
    ii = [i for i in range(N_STEP) if PASS[i] in passes]
    v = VOLTS[ii]
    m = np.array([coinc_median(i, ai, b) for i in ii])
    ok = np.isfinite(m)
    return float(np.polyfit(np.log(v[ok]), np.log(m[ok]), 1)[0])


PROVENANCE = {
    'run': RUN_STEM,
    'exported': str(date.today()),
    'source': 'nTof_x17/mx_july_beam_qa (cache 12_hvscan; scripts 12/12b/12c); '
              'HV log ~/beam_july/scint_hv_scan/2026-07-16_13-32-34_plastic_scan_1/',
    'method': 'wall-tagged coincident plastic spectra (+-8 ns calibrated window, '
              'sideband-subtracted); observable = spectrum median. Scan: all 8 PMTs '
              'at common V, 9 proton-gated steps 1200-1600 V in two interleaved '
              'passes, + pre-scan (1500 V) and post-scan (nominal) windows.',
    'hv_state': 'operating point after the scan: AL 1325, AR 1275, BL 1325, '
                'others 1300 V; SiPM flash gating ON in pass 2 and after',
    'caveats': [
        'medians are a RELATIVE gain standard, not a MIP energy scale — the '
        'plastic sits behind the wall, so the wall tag does not select plastic '
        'MIPs; absolute calibration awaits the LIQ triple coincidence',
        'power laws validated only inside the scanned 1200-1600 V range; '
        'settings outside are extrapolation',
        'pass 1 = flash gating OFF, pass 2 = ON; the two passes give consistent '
        'gain fits, so curves are usable regardless of gating state',
        'PSSC1 (C-L) has the lowest index (n~4.8) and visibly flattens above '
        '~1500 V — avoid running it high',
        're-derive after PMT/base hardware changes: 12_plastic_hv_scan.py + this export',
    ],
}

OUT.mkdir(parents=True, exist_ok=True)
i_end = LABELS.index('end_nominal')

# ---------------- gain_vs_hv: per-PMT curves + fits
gain = {'provenance': PROVENANCE, 'model': 'median(V) = median(V_ref) * (V/V_ref)^n',
        'pmts': {}}
med_now = {}
n_comb = {}
for ai, st in enumerate('ABCD'):
    for b in range(2):
        name = f'PSS{st}{b + 1}'
        med_now[(ai, b)] = coinc_median(i_end, ai, b)
        n_comb[(ai, b)] = fit_n(ai, b, (1, 2))
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        curve = []
        for i in range(N_STEP):
            m = coinc_median(i, ai, b)
            v = float(VOLTS[i]) if VOLTS[i] > 0 else NOMINAL_V[(st, b)]
            curve.append({'step': LABELS[i], 'v': v,
                          'median_adc': round(m, 1),
                          'pass': int(PASS[i]), 'flash_gating': bool(GATING[i])})
        gain['pmts'][name] = {
            'detector': f'plastic_{st}_{"LR"[b]}',
            'caen_hv_channel': CAEN[(st, b)],
            'v_operating': NOMINAL_V[(st, b)],
            'median_at_operating_adc': round(med_now[(ai, b)], 1),
            'median_at_operating_mV': round(med_now[(ai, b)] * f_mv, 2),
            'adc_to_mV': round(float(f_mv), 6),
            'n_combined': round(n_comb[(ai, b)], 2),
            'n_pass1': round(fit_n(ai, b, (1,)), 2),
            'n_pass2': round(fit_n(ai, b, (2,)), 2),
            'curve': curve,
        }
(OUT / f'gain_vs_hv_{RUN_STEM}.json').write_text(json.dumps(gain, indent=2))

# ---------------- hv_equalization
target = float(np.exp(np.mean([np.log(v) for v in med_now.values()])))
eq = {'provenance': PROVENANCE,
      'target_adc': round(target, 1),
      'target_rationale': 'geometric mean of the 8 responses at the current '
                          'operating point — equalize around where we already '
                          'run, minimal net HV movement; NOT rate-derived (no '
                          'plateau exists) and not an absolute energy target',
      'pmts': {}}
v_eq = {}
for ai, st in enumerate('ABCD'):
    for b in range(2):
        name = f'PSS{st}{b + 1}'
        v0 = NOMINAL_V[(st, b)]
        n = n_comb[(ai, b)]
        v_new = v0 * (target / med_now[(ai, b)]) ** (1 / n)
        v_eq[(ai, b)] = v_new
        eq['pmts'][name] = {
            'v_now': v0,
            'median_now_adc': round(med_now[(ai, b)], 1),
            'n': round(n, 2),
            'v_suggested': round(v_new),
            'dv': round(v_new - v0),
        }
(OUT / f'hv_equalization_{RUN_STEM}.json').write_text(json.dumps(eq, indent=2))

# ---------------- global gain slide table (the operational deliverable)
lines = ['# Per-PMT voltages [V] that scale the equalized fleet response by a '
         'common factor g,',
         f'# from the equalized baseline (g=1.0 column = hv_equalization_{RUN_STEM}.json).',
         '# V(g) = V_eq * g^(1/n) per PMT. Scanned/validated range: '
         f'{SCAN_RANGE[0]}-{SCAN_RANGE[1]} V; values outside are marked with *.',
         'pmt,caen,n,' + ','.join(f'g={g}' for g in GAIN_FACTORS)]
for ai, st in enumerate('ABCD'):
    for b in range(2):
        n = n_comb[(ai, b)]
        cells = []
        for g in GAIN_FACTORS:
            v = v_eq[(ai, b)] * g ** (1 / n)
            flag = '' if SCAN_RANGE[0] <= v <= SCAN_RANGE[1] else '*'
            cells.append(f'{v:.0f}{flag}')
        lines.append(f'PSS{st}{b + 1},{CAEN[(st, b)]},{n:.2f},' + ','.join(cells))
(OUT / f'global_gain_slide_{RUN_STEM}.csv').write_text('\n'.join(lines) + '\n')

# ---------------- figures
figs = OUT / 'figures'
figs.mkdir(exist_ok=True)
for f in ['gain_curves.png', 'equalization.png', 'coinc_spectra_linear.png']:
    src = BASE / 'figures' / '12_hv_scan' / f
    if src.exists():
        shutil.copy2(src, figs / f'{Path(f).stem}_{RUN_STEM}.png')

# ================ pss_trigger: threshold scan vs the 10 mV hardware floor
TRIG = DAQ / 'calibrations' / 'pss_trigger'
TRIG.mkdir(parents=True, exist_ok=True)
HW_MIN_MV = 10.0
n_gb = float(d['n_good_bunches'][i_end])
thr = np.geomspace(1.5, 200, 250)


def eff_rate(scale_to_target):
    """Per-PMT tagged efficiency + summed inclusive late rate vs `thr` [mV].
    scale_to_target=True evaluates on the post-equalization amplitude scale."""
    effs, rate = {}, np.zeros_like(thr)
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            f_mv = FAC[f'PSS{st}'][str(b + 1)]
            scale = (target / med_now[(ai, b)]) if scale_to_target else 1.0
            x = CEN * f_mv * scale
            sub = np.clip(d['pss_mip'][i_end, ai, 0, b]
                          - SB * d['pss_mip'][i_end, ai, 1, b], 0, None)
            inc = d['pss_amp'][i_end, ai, b]
            effs[f'PSS{st}{b + 1}'] = np.interp(
                thr, x, np.cumsum(sub[::-1])[::-1] / max(sub.sum(), 1))
            rate += np.interp(thr, x, np.cumsum(inc[::-1])[::-1] / n_gb)
    return effs, rate


eff_eq, rate_eq = eff_rate(True)
eff_cur, _ = eff_rate(False)
min_eff = np.min(list(eff_eq.values()), axis=0)
t99 = float(thr[np.searchsorted(-min_eff, -0.99)])
t95 = float(thr[np.searchsorted(-min_eff, -0.95)])

hv_recover = {}
for label, t_ref in (('eff99_at_10mV', t99), ('eff95_at_10mV', t95)):
    g = HW_MIN_MV / t_ref
    opt = {'gain_factor': round(g, 2), 'pmts': {}}
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            n = n_comb[(ai, b)]
            v = v_eq[(ai, b)] * g ** (1 / n)
            opt['pmts'][f'PSS{st}{b + 1}'] = {
                'v': round(v),
                'extrapolated': not (SCAN_RANGE[0] <= v <= SCAN_RANGE[1])}
    hv_recover[label] = opt

at10 = {name: {'eff_equalized_gain': round(float(np.interp(HW_MIN_MV, thr, e)), 3),
               'eff_current_gain': round(float(np.interp(HW_MIN_MV, thr, eff_cur[name])), 3)}
        for name, e in eff_eq.items()}
trig = {
    'provenance': PROVENANCE,
    'hardware_min_threshold_mV': HW_MIN_MV,
    'note': 'no signal/noise valley: threshold choice is efficiency-driven. '
            'Unconstrained recommendation was 4-5 mV (99% tagged eff at 4.2); '
            'the 10 mV hardware floor costs efficiency unless HV is raised. '
            'Efficiencies are for the wall-tagged (particle-like) population; '
            'rates are the relative late-TOF sample. Thresholds are '
            'digitizer-equivalent mV.',
    'tagged_eff_at_hardware_min': at10,
    'worst_pmt_eff_at_hardware_min_equalized': round(float(np.interp(HW_MIN_MV, thr, min_eff)), 3),
    'hv_to_recover': {
        'model': 'V = V_equalized * g^(1/n) per PMT; g scales the whole '
                 'spectrum so 10 mV sits at the unconstrained 99%/95% points',
        'status': 'OPTION ONLY - deferred until a true plastic MIP calibration '
                  '(LIQ triple coincidence) sets the absolute scale',
        **hv_recover},
}
(TRIG / f'thresholds_{RUN_STEM}.json').write_text(json.dumps(trig, indent=2))

csv = ['# Common threshold scan, POST-EQUALIZATION amplitude scale '
       f'(equalized voltages of pss/hv_equalization_{RUN_STEM}.json).',
       '# eff = wall-tagged (sideband-subtracted) fraction above threshold; '
       'rate = summed late hits/bunch, 8 PMTs (relative).',
       'thr_mV,' + ','.join(eff_eq) + ',min_eff,late_hits_per_bunch']
for i in range(0, len(thr), 2):
    csv.append(f'{thr[i]:.2f},' + ','.join(f'{e[i]:.4f}' for e in eff_eq.values())
               + f',{min_eff[i]:.4f},{rate_eq[i]:.0f}')
(TRIG / f'threshold_scan_{RUN_STEM}.csv').write_text('\n'.join(csv) + '\n')

tfigs = TRIG / 'figures'
tfigs.mkdir(exist_ok=True)
src = BASE / 'figures' / '12_hv_scan' / 'threshold_scan.png'
if src.exists():
    shutil.copy2(src, tfigs / f'threshold_scan_{RUN_STEM}.png')

print(f'pss calibration -> {OUT}')
print(f'  target {target:.0f} ADC; equalized voltages: ' +
      ', '.join(f'PSS{st}{b + 1}:{v_eq[(ai, b)]:.0f}'
                for ai, st in enumerate('ABCD') for b in range(2)))
print(f'pss_trigger -> {TRIG}')
print(f'  at {HW_MIN_MV:.0f} mV floor: worst tagged eff '
      f'{float(np.interp(HW_MIN_MV, thr, min_eff)):.3f} (equalized gains); '
      f'99%/95% recovery gain factors {hv_recover["eff99_at_10mV"]["gain_factor"]}'
      f'/{hv_recover["eff95_at_10mV"]["gain_factor"]}')
