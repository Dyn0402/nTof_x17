"""23h_equalized_voltages.py — RECOMMENDED plastic PMT bias voltages to equalize
the gains on the Y-88 699 keVee Compton edge, in the CURRENT (FIFO) readout.

This is the hardware deliverable. Each PMT's 699 keVee edge is slid along its own
gain-vs-V power law to a common target; the target is chosen to keep the fleet-
mean HV unchanged (minimal perturbation, overall gain preserved). The Y-88 edge
is measured in BNC-T, so it is multiplied by the per-PMT FIFO factor first
(edge699_fifo_mv in plastic_hv_gain_absolute.json) — otherwise the FIFO would
re-un-equalize it.

Why Y-88 (vs the HV-scan coincident median): it is an absolute, known-energy line
(699 keVee) at the trigger-relevant low-energy scale, and it disagrees with the
median specifically on DL/CR (mild nonlinearity, see 23g) --- equalizing on the
edge fixes those.

CAEN channels: card 07, ch 0..7 = AL AR BL BR CL CR DL DR (run_config_beam.py).

Output: calib/plastic_hv_equalized_y88.json  (set these in hardware)
Usage: python 23h_equalized_voltages.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

BASE = Path(__file__).parent
CALIB = BASE / 'calib'
PMTS = ['PSSA1', 'PSSA2', 'PSSB1', 'PSSB2', 'PSSC1', 'PSSC2', 'PSSD1', 'PSSD2']
LABEL = {'PSSA1': 'AL', 'PSSA2': 'AR', 'PSSB1': 'BL', 'PSSB2': 'BR',
         'PSSC1': 'CL', 'PSSC2': 'CR', 'PSSD1': 'DL', 'PSSD2': 'DR'}
CAEN_CH = {'PSSA1': (7, 0), 'PSSA2': (7, 1), 'PSSB1': (7, 2), 'PSSB2': (7, 3),
           'PSSC1': (7, 4), 'PSSC2': (7, 5), 'PSSD1': (7, 6), 'PSSD2': (7, 7)}


def main():
    cal = json.loads((CALIB / 'plastic_hv_gain_absolute.json').read_text())['pmts']

    def V(pmt, T):
        r = cal[pmt]
        return r['V_equalized'] * (T / r['edge699_fifo_mv']) ** (1 / r['n_fifo'])

    cur_mean = np.mean([cal[p]['V_equalized'] for p in PMTS])
    T = brentq(lambda t: np.mean([V(p, t) for p in PMTS]) - cur_mean, 10, 120)

    out = {'note': 'Recommended plastic PMT bias to equalize gains on the Y-88 '
                   '699 keVee edge in the CURRENT (FIFO) readout. Target = '
                   f'{T:.1f} mV edge, chosen to keep the fleet-mean HV at '
                   f'{cur_mean:.0f} V (overall gain unchanged). Set on CAEN card '
                   '07, channels 0-7 = AL AR BL BR CL CR DL DR.',
           'target_699kevee_mv_fifo': round(float(T), 1),
           'fleet_mean_V': round(float(cur_mean)),
           'voltages': {}}
    print(f'RECOMMENDED plastic HV (Y-88 edge equalization, FIFO). '
          f'699 keVee -> {T:.1f} mV, fleet-mean {cur_mean:.0f} V.\n')
    print(f'{"ch":3s} {"CAEN":>6s} {"current":>7s} {"-> set":>6s} {"change":>7s}')
    for p in PMTS:
        v = int(round(V(p, T)))
        out['voltages'][p] = dict(label=LABEL[p],
                                  caen=f'{CAEN_CH[p][0]:02d}.{CAEN_CH[p][1]:03d}',
                                  current_V=cal[p]['V_equalized'],
                                  set_V=v, change_V=v - cal[p]['V_equalized'])
        print(f'{LABEL[p]:3s} {CAEN_CH[p][0]:02d}.{CAEN_CH[p][1]:03d} '
              f'{cal[p]["V_equalized"]:7d} {v:6d} {v - cal[p]["V_equalized"]:+7d}')
    (CALIB / 'plastic_hv_equalized_y88.json').write_text(json.dumps(out, indent=2))
    vs = [out['voltages'][p]['set_V'] for p in PMTS]
    print(f'\nrange {min(vs)}-{max(vs)} V (HV scans went to 1600 V OK).')
    print('-> calib/plastic_hv_equalized_y88.json')


if __name__ == '__main__':
    main()
