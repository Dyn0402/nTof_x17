"""
08_inclusive_vs_coinc.py — Shape comparison: inclusive amplitude spectra (all hits,
01 cache) vs true-coincidence spectra (sideband-subtracted, 07 cache), per arm,
channels summed, unit-area normalized above the threshold region (>200 ADC).

Note the samples differ beyond the coincidence requirement: inclusive = all
bunches / all tof; coincidence = post-recovery bunches, tof > 0.1 ms. The shape
change from the coincidence cut is the point of the figure.

Usage: python 08_inclusive_vs_coinc.py [run_stem]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '08_inclusive_vs_coinc'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
from adc_mv import mv_factors

late = BASE / 'cache' / f'09_late_{RUN_STEM}.npz'
INC_SUFFIX = '_amp'
if late.exists():
    d1 = np.load(late)
    INC_SUFFIX = '_amp_late'
else:
    print('WARNING: 09 cache missing -> full-run inclusive')
    d1 = np.load(BASE / 'cache' / f'01_qa_{RUN_STEM}.npz')
d7 = np.load(BASE / 'cache' / f'07_mip_{RUN_STEM}.npz')
assert np.allclose(d1['amp_edges'], d7['amp_edges'])
EDGES = d1['amp_edges']
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])
WID = np.diff(EDGES)
SC = float(d7['sb_scale'])
FAC = mv_factors()
NORM_MIN = 200  # ADC; normalize above the threshold spike

C_INC = '#6b7280'
C_COI = '#c2410c'


def curves(st, det):
    tree = ('WAL' if det == 'wal' else 'PSS') + st
    n_ch = 8 if det == 'wal' else 2
    inc = d1[f'{tree}{INC_SUFFIX}'][:n_ch].sum(axis=0)
    coi = (d7[f'{st}_{det}'][0] - SC * d7[f'{st}_{det}'][1]).sum(axis=0)
    f_mv = FAC[tree].mean()
    x = CEN * f_mv
    out = []
    for h in (inc, coi):
        dens = h / (WID * f_mv)
        m = CEN > NORM_MIN
        area = np.trapz(dens[m], x[m])
        out.append(dens / area)
    return x, out[0], out[1]


for scale in ('linear', 'log'):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex='row' if scale == 'linear' else False)
    for row, det in enumerate(('wal', 'pss')):
        for col, st in enumerate('ABCD'):
            ax = axes[row][col]
            x, inc, coi = curves(st, det)
            ax.plot(x, inc, lw=1.4, color=C_INC, label='all hits (no cuts)')
            ax.plot(x, coi, lw=1.4, color=C_COI, label='true coincidences')
            if scale == 'linear':
                ax.set_xlim(0, 120 if det == 'wal' else 160)
                ax.set_ylim(bottom=0)
            else:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(1e-7, 1)
            ax.set_title(('WAL' if det == 'wal' else 'PSS') + st)
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=9, frameon=False)
            if row == 1:
                ax.set_xlabel('amplitude [mV]')
            if col == 0:
                ax.set_ylabel('normalized density [1/mV]')
    fig.suptitle(f'{RUN_STEM}: inclusive vs coincidence amplitude spectra, channels '
                 f'summed, unit area above {NORM_MIN * 0.0305:.0f} mV ({scale})')
    fig.tight_layout()
    fig.savefig(OUT / f'inclusive_vs_coinc_{scale}.png', dpi=140)
    plt.close(fig)
print(f'Figures -> {OUT}')
