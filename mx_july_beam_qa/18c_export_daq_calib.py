"""18c_export_daq_calib.py — Export wall trigger + MIP calibrations to the DAQ repo.

Writes machine-readable calibration files (JSON + CSV threshold-scan table) and
copies the operating figures into <daq_dir>/calibrations/. Everything derives
from the 18 (trigger sums) and 07 (per-channel MIP) caches of this analysis.

Usage: python 18c_export_daq_calib.py [run_stem] [daq_dir]
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224460'
DAQ = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / 'PycharmProjects/nTof_x17_DAQ'
BASE = Path(__file__).parent
CAL = DAQ / 'calibrations'
TARGET_EFF = 0.95

d18 = np.load(BASE / 'cache' / f'18_trigsum_{RUN_STEM}.npz')
edges = d18['sum_edges']
cen = 0.5 * (edges[:-1] + edges[1:])
sb18 = float(d18['sb_scale'])
nb = float(d18['n_bunches'])
kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
kern /= kern.sum()

d07 = np.load(BASE / 'cache' / f'07_mip_{RUN_STEM}.npz')
ae7 = d07['amp_edges']
cen7 = np.sqrt(ae7[:-1] * ae7[1:])
sb7 = float(d07['sb_scale'])
kern7 = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
kern7 /= kern7.sum()
fac = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_STEM}.json').read_text())['factors']

PROVENANCE = {
    'run': RUN_STEM,
    'exported': str(date.today()),
    'source': 'nTof_x17/mx_july_beam_qa (caches 18_trigsum, 07_mip)',
    'selection': 'beam-on bunches, hits >0.1 ms after gamma flash, '
                 'duplication-vetoed (same-side neighbor +-4 ns, amp ratio 1/3-3)',
    'hv_state': 'walls at July operating point (WALA ~30% low gain, WALD g4 weak); '
                'plastics after the 224404->224460 retune',
    'caveats': [
        'sum thresholds are digitizer-equivalent mV of amp(top)+amp(bottom); '
        'map to hardware discriminator units before setting',
        'rates are per-bunch for the late-TOF sample: relative, not in-spill absolute',
        'duplication short (WALA 5-7, WALD 2-4, 5-7) is NOT removable in hardware: '
        'unfixed it adds ~+17% (A) / +25% (D) trigger rate',
        're-derive after any wall HV change: 18_trigger_threshold.py + this export',
    ],
}


def group_quantities(st, g):
    cand = d18[f'{st}_cand'][g]
    raw = d18[f'{st}_cand_raw'][g]
    sub = d18[f'{st}_tag'][g, 0] - sb18 * d18[f'{st}_tag'][g, 1]
    sm = np.convolve(sub, kern, mode='same')
    m = cen > 15
    pk = float(cen[m][np.argmax(sm[m])])
    hi = (cen >= 1.3 * pk) & (cen <= 2.5 * pk)
    eps = float(sub[hi].sum() / max(cand[hi].sum(), 1))
    mip = sub / max(eps, 1e-9)
    eff = np.cumsum(sub[::-1])[::-1] / max(sub.sum(), 1)
    pur = np.minimum(np.cumsum(mip[::-1])[::-1] /
                     np.maximum(np.cumsum(cand[::-1])[::-1], 1), 1)
    rate = np.cumsum(cand[::-1])[::-1] / nb
    rate_raw = np.cumsum(raw[::-1])[::-1] / nb
    return pk, eps, eff, pur, rate, rate_raw


# ---------------- wal_trigger: JSON + CSV scan table
wt = CAL / 'wal_trigger'
wt.mkdir(parents=True, exist_ok=True)
cal = {'provenance': PROVENANCE, 'target_eff_weakest_group': TARGET_EFF, 'walls': {}}
csv_lines = ['wall,thr_mV,eff_g1,eff_g2,eff_g3,eff_g4,min_eff,'
             'pur_g1,pur_g2,pur_g3,pur_g4,pairs_per_bunch,pairs_per_bunch_dup_unfixed']
for st in 'ABCD':
    qs = [group_quantities(st, g) for g in range(4)]
    min_eff = np.min([q[2] for q in qs], axis=0)
    i_rec = np.nonzero(min_eff >= TARGET_EFF)[0]
    i_rec = int(i_rec[-1]) if len(i_rec) else 0
    cal['walls'][st] = {
        'recommended_threshold_mV': float(cen[i_rec]),
        'mip_sum_peak_mV': [q[0] for q in qs],
        'plastic_tag_eff': [q[1] for q in qs],
        'at_recommended': {
            'eff': [float(q[2][i_rec]) for q in qs],
            'purity': [float(q[3][i_rec]) for q in qs],
            'pairs_per_bunch': float(sum(q[4][i_rec] for q in qs)),
            'pairs_per_bunch_dup_unfixed': float(sum(q[5][i_rec] for q in qs)),
        },
    }
    for i in range(0, 100):
        csv_lines.append(
            f'{st},{cen[i]:.1f},' +
            ','.join(f'{q[2][i]:.4f}' for q in qs) + f',{min_eff[i]:.4f},' +
            ','.join(f'{q[3][i]:.3f}' for q in qs) +
            f',{sum(q[4][i] for q in qs):.1f},{sum(q[5][i] for q in qs):.1f}')
(wt / f'thresholds_{RUN_STEM}.json').write_text(json.dumps(cal, indent=2))
(wt / f'threshold_scan_{RUN_STEM}.csv').write_text('\n'.join(csv_lines) + '\n')

figs = wt / 'figures'
figs.mkdir(exist_ok=True)
for f in ['threshold_scan.png', 'trigsum_spectra_linear.png', 'purity_vs_eff.png']:
    src = BASE / 'figures' / RUN_STEM / '18_trigger' / f
    if src.exists():
        shutil.copy2(src, figs / f'{Path(f).stem}_{RUN_STEM}.png')

# ---------------- wal_mip: per-channel MIP constants
wm = CAL / 'wal_mip'
wm.mkdir(parents=True, exist_ok=True)
mip = {'provenance': PROVENANCE, 'channels': {}}
for st in 'ABCD':
    for c in range(8):
        sub = d07[f'{st}_wal'][0, c] - sb7 * d07[f'{st}_wal'][1, c]
        sm = np.convolve(sub, kern7, mode='same')
        m = cen7 > 300
        pk_adc = float(cen7[m][np.argmax(sm[m])])
        f_mv = fac[f'WAL{st}'][str(c + 1)]
        mip['channels'][f'WAL{st}{c + 1}'] = {
            'mip_peak_adc': round(pk_adc, 1),
            'mip_peak_mV': round(pk_adc * f_mv, 2),
            'adc_to_mV': round(f_mv, 6),
        }
(wm / f'mip_{RUN_STEM}.json').write_text(json.dumps(mip, indent=2))

print(f'wal_trigger -> {wt}')
print(f'wal_mip     -> {wm}')
