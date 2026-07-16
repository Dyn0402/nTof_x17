"""
10_pmt_gain.py — Plastic-PMT relative gain analysis + HV suggestions.

Per PMT (8 channels: PSS[A-D] x L/R), overlays the raw (inclusive) amplitude
spectrum with the wall-coincidence-selected one, and estimates relative gains
from spectral quantiles in mV:
  - inclusive median   (cross-arm comparable: same beam background mix)
  - coincidence median (cleanest within an arm; A/D population differs)

HV suggestion: PMT gain ~ V^ALPHA (ALPHA ~ 7 typical for 8-10 stage tubes;
CALIBRATE with an HV scan before trusting the last volt):
  V_new = V_now * (target_gain / gain)^(1/ALPHA)

Prefers the late-TOF inclusive cache (09) when present; falls back to the
full-run inclusive (01) with a printed warning.

Usage: python 10_pmt_gain.py [run_stem]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '10_pmt_gain'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
from adc_mv import mv_factors

HV_NOW = {'PSSA1': 1325, 'PSSA2': 1275, 'PSSB1': 1325, 'PSSB2': 1300,
          'PSSC1': 1300, 'PSSC2': 1300, 'PSSD1': 1300, 'PSSD2': 1300}
LR = {'1': 'L', '2': 'R'}
ALPHA = 7.0
QMIN_ADC = 60         # quantiles computed above this (clear of threshold)

d7 = np.load(BASE / 'cache' / f'07_mip_{RUN_STEM}.npz')
EDGES = d7['amp_edges']
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])
SC7 = float(d7['sb_scale'])
FAC = mv_factors()

late = BASE / 'cache' / f'09_late_{RUN_STEM}.npz'
if late.exists():
    d_inc = np.load(late)
    inc_key = lambda tree: f'{tree}_amp_late'
    inc_label = 'all hits (>0.1 ms, good bunches)'
else:
    print('WARNING: 09 cache missing -> falling back to full-run inclusive (01)')
    d_inc = np.load(BASE / 'cache' / f'01_qa_{RUN_STEM}.npz')
    inc_key = lambda tree: f'{tree}_amp'
    inc_label = 'all hits (full run)'


def quantile(h, q, xmin_adc=QMIN_ADC):
    m = CEN > xmin_adc
    c = np.cumsum(np.clip(h[m], 0, None))
    if c[-1] <= 0:
        return np.nan
    return CEN[m][np.searchsorted(c, q * c[-1])]


def main():
    wid = np.diff(EDGES)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    rows = []
    for i, st in enumerate('ABCD'):
        for pc in range(2):
            ax = axes[pc][i]
            name = f'PSS{st}{pc + 1}'
            f_mv = FAC[f'PSS{st}'][pc]
            h_inc = d_inc[inc_key(f'PSS{st}')][pc]
            h_coi = d7[f'{st}_pss'][0, pc] - SC7 * d7[f'{st}_pss'][1, pc]
            for h, c, lab in ((h_inc, '#6b7280', inc_label),
                              (h_coi, '#c2410c', 'wall-coincidence selected')):
                dens = h / (wid * f_mv)
                m = CEN > QMIN_ADC
                dens = dens / np.trapz(dens[m], CEN[m] * f_mv)
                ax.plot(CEN * f_mv, dens, lw=1.3, color=c, label=lab)
            ax.set_xlim(0, 160)
            ax.set_ylim(bottom=0)
            ax.set_title(f'{name} ({LR[str(pc + 1)]}, {HV_NOW[name]} V)')
            ax.grid(alpha=0.25)
            if i == 0 and pc == 0:
                ax.legend(fontsize=8, frameon=False)
            if pc == 1:
                ax.set_xlabel('amplitude [mV]')
            if i == 0:
                ax.set_ylabel('normalized density [1/mV]')
            rows.append({'name': name,
                         'hv': HV_NOW[name],
                         'med_inc': quantile(h_inc, 0.5) * f_mv,
                         'med_coi': quantile(h_coi, 0.5) * f_mv})
    fig.suptitle(f'{RUN_STEM}: plastic PMT spectra, raw vs wall-coincidence selected '
                 '(unit area above ~2 mV)')
    fig.tight_layout()
    fig.savefig(OUT / 'pmt_raw_vs_coinc.png', dpi=140)
    plt.close(fig)

    # gain table: reference = geometric mean of inclusive medians
    meds = np.array([r['med_inc'] for r in rows])
    ref = np.exp(np.nanmean(np.log(meds)))
    print(f'\nreference scale (geo-mean inclusive median): {ref:.1f} mV, alpha={ALPHA}')
    print(f'{"PMT":8s} {"HV[V]":>6s} {"med_inc[mV]":>12s} {"med_coi[mV]":>12s} '
          f'{"rel gain":>9s} {"V(equalize)":>12s} {"V(x2 target)":>13s}')
    for r in rows:
        g = r['med_inc'] / ref
        v_eq = r['hv'] * (1 / g) ** (1 / ALPHA)
        v_x2 = r['hv'] * (2 / g) ** (1 / ALPHA)
        print(f'{r["name"]:8s} {r["hv"]:6d} {r["med_inc"]:12.1f} {r["med_coi"]:12.1f} '
              f'{g:9.2f} {v_eq:12.0f} {v_x2:13.0f}')
    print('\nV(equalize): bring every PMT to the current average scale.')
    print('V(x2 target): equalize AND double the overall scale (if MIPs are '
          'below threshold, raising all plastics lifts them in).')
    print('NOTE: alpha=7 assumed; do a 2-point HV scan to calibrate before '
          'trusting shifts >25 V.')


if __name__ == '__main__':
    main()
