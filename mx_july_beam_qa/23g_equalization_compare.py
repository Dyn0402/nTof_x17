"""23g_equalization_compare.py — compare the three plastic-PMT gain
equalizations:
  (1) 224466 HV scan, coincident MEDIAN   (BNC-T; the CURRENT fixed voltages)
  (2) 224489 HV scan, coincident MEDIAN   (FIFO)
  (3) Y-88 699 keVee Compton EDGE          (BNC-T; absolute-energy observable)

Each is expressed as the per-PMT bias that brings its observable to the fleet
value, sliding along that PMT's own gain-vs-V power law; all three are
normalized to the same fleet-MEAN voltage (1300 V) so only the RELATIVE pattern
is compared. The Y-88 edge equalization uses the 224466 (same BNC-T config)
power-law exponents, anchored at the voltage the Y-88 runs were taken (the
gain-equalized set). Does equalizing the MEDIAN also equalize the low-energy
EDGE? And how much does the FIFO change it?

Output: figures/21_y88/equalization_compare.png, calib/equalization_compare.json
Usage: python 23g_equalization_compare.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
CALIB = BASE / 'calib'
OUT = BASE / 'figures' / '21_y88'
PMTS = [f'PSS{s}{b + 1}' for s in 'ABCD' for b in range(2)]
LABELS = {'PSSA1': 'AL', 'PSSA2': 'AR', 'PSSB1': 'BL', 'PSSB2': 'BR',
          'PSSC1': 'CL', 'PSSC2': 'CR', 'PSSD1': 'DL', 'PSSD2': 'DR'}
# the voltages the Y-88 runs were taken at (= the 224466/BNC-T median equalization)
VGE = {'PSSA1': 1303, 'PSSA2': 1242, 'PSSB1': 1376, 'PSSB2': 1279,
       'PSSC1': 1180, 'PSSC2': 1307, 'PSSD1': 1303, 'PSSD2': 1417}
FLEET_V = 1300.0


def coinc_medians(run):
    d = np.load(CACHE / f'12_hvscan_{run}.npz')
    edges = d['amp_edges']
    c = np.sqrt(edges[:-1] * edges[1:])
    sb = float(d['sb_scale'])
    volts = d['step_volts']
    fac = json.loads((CALIB / f'adc_to_mv_{run}.json').read_text())['factors']
    med = np.full((4, 2, len(volts)), np.nan)
    for i in range(len(volts)):
        for ai, st in enumerate('ABCD'):
            for b in range(2):
                sub = np.clip(d['pss_mip'][i, ai, 0, b]
                              - sb * d['pss_mip'][i, ai, 1, b], 0, None)
                cum = np.cumsum(sub)
                if cum[-1] > 200:
                    med[ai, b, i] = c[np.searchsorted(cum, cum[-1] / 2)] \
                        * fac[f'PSS{st}'][str(b + 1)]
    return volts, med


def powerlaws(run):
    v, m = coinc_medians(run)
    out = {}
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            vv = np.array([v[i] for i in range(len(v)) if v[i] > 0])
            mm = np.array([m[ai, b, i] for i in range(len(v)) if v[i] > 0])
            ok = np.isfinite(mm)
            out[f'PSS{st}{b + 1}'] = np.polyfit(np.log(vv[ok]), np.log(mm[ok]), 1)
    return out


def normalize(volts):
    s = FLEET_V / np.mean(list(volts.values()))
    return {p: v * s for p, v in volts.items()}


def median_eq(fitd):
    def V(T):
        return {p: np.exp((np.log(T) - lnA) / n) for p, (n, lnA) in fitd.items()}
    T = brentq(lambda t: np.mean(list(V(t).values())) - FLEET_V, 5, 500)
    return normalize(V(T))


def main():
    f6 = powerlaws('run224466')
    f9 = powerlaws('run224489')
    eq6 = median_eq(f6)
    eq9 = median_eq(f9)
    # Y-88 edge equalization (BNC-T): slide the edge from VGE along the 224466 n
    edge = {p: json.loads((CALIB / 'y88_energy_calib.json').read_text())
            ['channels'][p]['edge699_mv'] for p in PMTS}

    def Vy(T):
        return {p: VGE[p] * (T / edge[p]) ** (1 / f6[p][0]) for p in PMTS}
    Ty = brentq(lambda t: np.mean(list(Vy(t).values())) - FLEET_V, 1, 200)
    eqy = normalize(Vy(Ty))

    res = {'note': 'Per-PMT equalization bias (V), all normalized to fleet-mean '
                   '1300 V. med466=BNC-T median (current fixed set), med489=FIFO '
                   'median, y88edge=699 keVee Compton edge (BNC-T).',
           'channels': {p: dict(med466=round(eq6[p], 0), med489=round(eq9[p], 0),
                                y88edge=round(eqy[p], 0)) for p in PMTS}}
    (CALIB / 'equalization_compare.json').write_text(json.dumps(res, indent=2))

    print(f'{"PMT":4s} {"466 med":>7s} {"489 med":>7s} {"Y88 edge":>8s}  '
          f'{"Y88-466":>7s} {"489-466":>7s}')
    for p in PMTS:
        print(f'{LABELS[p]:4s} {eq6[p]:7.0f} {eq9[p]:7.0f} {eqy[p]:8.0f}  '
              f'{eqy[p] - eq6[p]:+7.0f} {eq9[p] - eq6[p]:+7.0f}')
    dy = np.array([eqy[p] - eq6[p] for p in PMTS])
    d9 = np.array([eq9[p] - eq6[p] for p in PMTS])
    print(f'\nrms vs the current (466-median) set:  Y88-edge {dy.std():.0f} V, '
          f'489-FIFO-median {d9.std():.0f} V  (per-PMT range '
          f'Y88 {dy.min():+.0f}..{dy.max():+.0f}, 489 {d9.min():+.0f}..{d9.max():+.0f})')

    figure(eq6, eq9, eqy)
    print('\n-> figures/21_y88/equalization_compare.png & '
          'calib/equalization_compare.json')


def figure(eq6, eq9, eqy):
    x = np.arange(len(PMTS))
    labs = [LABELS[p] for p in PMTS]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    # (1) the three equalization voltages per PMT
    ax[0].plot(x, [eq6[p] for p in PMTS], 'o-', ms=7, label='224466 median (BNC-T, current)')
    ax[0].plot(x, [eq9[p] for p in PMTS], 's-', ms=7, label='224489 median (FIFO)')
    ax[0].plot(x, [eqy[p] for p in PMTS], '^-', ms=8, label='Y-88 699 keVee edge (BNC-T)')
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs)
    ax[0].set_ylabel('equalized bias [V]  (norm. to fleet mean 1300 V)')
    ax[0].set_title('Three plastic equalizations vs PMT')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    # (2) differences from the current (466-median) set
    w = 0.38
    ax[1].bar(x - w / 2, [eq9[p] - eq6[p] for p in PMTS], w,
              label='489 FIFO median $-$ current', color='steelblue')
    ax[1].bar(x + w / 2, [eqy[p] - eq6[p] for p in PMTS], w,
              label='Y-88 edge $-$ current', color='crimson')
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs)
    ax[1].set_ylabel('$\\Delta$ bias from current [V]')
    ax[1].set_title('Re-tune suggested by each vs the current (BNC-T median) set')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.suptitle('Plastic PMT gain equalization: HV-scan medians (BNC-T, FIFO) '
                 'vs the Y-88 Compton-edge')
    fig.tight_layout()
    fig.savefig(OUT / 'equalization_compare.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
