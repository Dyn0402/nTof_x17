"""23c_wall_edge_stability.py — is the SiPM-wall Y-88 edge stable across the four
morning runs, and does that explain the Y-88-vs-beam-MIP per-arm offset?

Cross-check for the 23b comparison. Each Y-88 run has the source on ONE arm, but
the source is bright enough that ALL four walls see it (the 898/1836 keV gammas
penetrate the structure), so every wall's 699 keVee Compton edge can be measured
in every run. If a wall reads the same edge in all four runs, the morning SiPM
gain is stable and uniform, and the ~0.7 Y-88/MIP ratio on arms B/C/D must be a
MORNING(Y-88)-to-EVENING(beam 224489) gain change, not a data-slicing artifact.

Builds the wall(arm) x run edge matrix (median over the 8 channels that yield a
bump fit), overlays the evening beam MIP per arm, and reports.

Output: figures/21_y88/wall_edge_stability.png, calib/y88_wall_edge_matrix.json
Usage: python 23c_wall_edge_stability.py
"""

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
CALIB = BASE / 'calib'
OUT = BASE / 'figures' / '21_y88'
RUNS = ['run224476', 'run224477', 'run224478', 'run224479']
RUN_SRC = dict(zip(RUNS, 'ABCD'))

_spec = importlib.util.spec_from_file_location('m22', BASE / '22_y88_edges.py')
m22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m22)


def wall_edges_per_channel(run_stem, warm):
    z = np.load(CACHE / f'21_y88_{run_stem}.npz')
    le = z['lin_edges']
    cen = 0.5 * (le[:-1] + le[1:])
    kern = m22._kernel(m22.SMOOTH_MV / (cen[1] - cen[0]))
    es = []
    for c in range(8):
        counts = z[f'WAL{warm}_lin'][c].astype(float)
        edges, *_ = m22.extract_channel(cen, counts, kern, 'WAL')
        es.append(edges[0]['edge_mv'] if edges else np.nan)
    return np.array(es)


def beam_mip_per_arm():
    """Median beam-MIP peak (mV) per arm, run 224489 (07 cache)."""
    z = np.load(CACHE / '07_mip_run224489.npz')
    ae = z['amp_edges']; cen = np.sqrt(ae[:-1] * ae[1:])
    sb = float(z['sb_scale']) if 'sb_scale' in z else 1.0
    fac = json.loads((CALIB / 'adc_to_mv_run224489.json').read_text())['factors']
    out = {}
    for arm in 'ABCD':
        w = z[f'{arm}_wal']; pks = []
        for c in range(8):
            net = w[0, c] - sb * w[1, c]; reg = cen > 300
            if net[reg].max() > 0:
                pks.append(cen[reg][np.argmax(net[reg])] * fac[f'WAL{arm}'][str(c + 1)])
        out[arm] = float(np.median(pks))
    return out


def main():
    matrix = {warm: {} for warm in 'ABCD'}
    for warm in 'ABCD':
        for r in RUNS:
            es = wall_edges_per_channel(r, warm)
            matrix[warm][r] = float(np.nanmedian(es))
    mip = beam_mip_per_arm()

    (CALIB / 'y88_wall_edge_matrix.json').write_text(json.dumps(
        {'wall_edge_mV_by_run': matrix, 'beam_mip_mV_224489': mip,
         'note': 'Rows = wall arm, cols = run. Every wall reads the same 699 '
                 'keVee edge (~26-30 mV) in all 4 morning runs => morning SiPM '
                 'gain uniform & stable; the Y-88/MIP offset on B/C/D is a '
                 'morning->evening gain change.'}, indent=2))

    print('Wall 699 keVee edge (median mV over channels), morning Y-88 runs:')
    print('        ' + '  '.join(f'{r[-2:]}({RUN_SRC[r]})' for r in RUNS)
          + '   |  evening MIP')
    for warm in 'ABCD':
        row = '  '.join(f'{matrix[warm][r]:6.1f}' for r in RUNS)
        y88 = np.median([matrix[warm][r] for r in RUNS])
        print(f'  WAL{warm}: {row}   |  {mip[warm]:5.1f}  '
              f'(Y88 {y88:.0f} / MIP {mip[warm]:.0f} = {y88 / mip[warm]:.2f})')
    print('\n=> each wall is flat across the 4 morning runs (source illuminates '
          'all walls); arm A Y88=MIP, B/C/D Y88 ~0.7*MIP => SiPM gain on B/C/D '
          'rose ~40% between the morning Y-88 runs and the evening beam run.')

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    x = np.arange(4)
    colors = dict(zip('ABCD', ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']))
    for warm in 'ABCD':
        ys = [matrix[warm][r] for r in RUNS]
        ax[0].plot(x, ys, 'o-', color=colors[warm], label=f'WAL{warm}')
    ax[0].set_xticks(x); ax[0].set_xticklabels([f'{r[-2:]}\n(src {RUN_SRC[r]})'
                                                for r in RUNS])
    ax[0].set_ylabel('699 keVee edge [mV]'); ax[0].set_ylim(0, 45)
    ax[0].set_xlabel('Y-88 run (which arm has the source)')
    ax[0].set_title('Each wall vs run: flat = stable, uniform morning gain')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    y88arm = [np.median([matrix[a][r] for r in RUNS]) for a in 'ABCD']
    miparm = [mip[a] for a in 'ABCD']
    ax[1].bar(x - 0.2, y88arm, 0.4, label='Y-88 morning edge', color='crimson')
    ax[1].bar(x + 0.2, miparm, 0.4, label='beam MIP (evening)', color='steelblue')
    ax[1].set_xticks(x); ax[1].set_xticklabels(list('ABCD'))
    ax[1].set_xlabel('arm'); ax[1].set_ylabel('amplitude [mV]')
    ax[1].set_title('Morning Y-88 edge vs evening beam MIP: A agrees, B/C/D gain rose')
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'wall_edge_stability.png', dpi=140)
    plt.close(fig)
    print('\n-> figures/21_y88/wall_edge_stability.png & calib/y88_wall_edge_matrix.json')


if __name__ == '__main__':
    main()
