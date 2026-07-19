"""23d_hv_gain_absolute.py — plastic PMT gain curves (both HV scans) anchored to
the Y-88 absolute energy scale, for setting the equalized operating voltages.

Two HV scans give the plastic gain vs bias, per PMT, as the sideband-subtracted
wall-tagged coincident-median amplitude (a clean gain proxy; 12 cache):
  - run 224466 = BNC-T split  (the OLD readout; gives the old equalization)
  - run 224489 = linear FIFO  (the CURRENT readout)  <-- use this one to set HV
The FIFO recovers x1.13-1.65 per PMT over the BNC-T (19c); at matched HV the two
curves differ by exactly that factor, confirming the gain shape is the same.

The coincident median has no absolute energy meaning on its own. The Y-88 source
scan (22/23) provides the missing absolute anchor: a KNOWN 698.63 keVee Compton
edge, measured on BNC-T at the gain-equalized bias, which we convert to the FIFO
config with the per-PMT FIFO factor. That pins mV-per-keVee vs bias for every
PMT in the current readout, so we can (a) EQUALIZE the channels and (b) SCALE the
whole set to put a chosen energy at a chosen amplitude (above the ~4.9 mV trigger
threshold, below the ~2 V full scale).

IMPORTANT DISAGREEMENT (documented, not resolved): the Y-88 absolute scale says
the triples "plastic MIP" (19d, assumed 5.05 MeV in 2.5 cm PVT) actually sits at
~130 keVee — a ~40x disagreement. The Y-88 line is trustworthy (known energy);
the triples MIP peak is likely mis-identified or statistics-starved. Flagged for
a long-run recheck. See HV_GAIN_Y88_ANALYSIS.md.

Outputs:
  figures/21_y88/hv_gain_curves.png      both scans, power-law fits, FIFO ratio
  figures/21_y88/hv_absolute_scale.png   mV/keVee vs bias (FIFO) + Y-88 anchors
  calib/plastic_hv_gain_absolute.json    per-PMT n, FIFO factor, mV/keVee(Vge),
                                         and equalization voltage tables
Usage: python 23d_hv_gain_absolute.py
"""

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

PMTS = [f'PSS{s}{b + 1}' for s in 'ABCD' for b in range(2)]
# gain-equalized bias = the Y-88 run bias (run_config_beam.py; confirmed against
# the 224489 pre-scan HV log)
VGE = {'PSSA1': 1303, 'PSSA2': 1242, 'PSSB1': 1376, 'PSSB2': 1279,
       'PSSC1': 1180, 'PSSC2': 1307, 'PSSD1': 1303, 'PSSD2': 1417}
# per-PMT FIFO/BNC-T gain ratio (19c, 224489 vs 224466 matched-HV medians)
FIFO = {'PSSA1': 1.65, 'PSSA2': 1.56, 'PSSB1': 1.23, 'PSSB2': 1.60,
        'PSSC1': 1.26, 'PSSC2': 1.53, 'PSSD1': 1.13, 'PSSD2': 1.33}
E699_KEV = 698.63
THRESH_MV = 4.9            # plastic trigger threshold (README)
FULLSCALE_MV = 2000.0     # DAQ full scale (saturation)
EQ_TARGETS_MV = [30, 40, 50, 60, 70]   # candidate 699-keVee amplitudes to equalize to


def coinc_medians(run):
    """(volts, median_mV[arm,bar,step]) — sideband-subtracted wall-tagged
    coincident-median plastic amplitude per HV step (the gain proxy)."""
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


def powerlaw(volts, med_ab):
    v = np.array([volts[i] for i in range(len(volts)) if volts[i] > 0])
    m = np.array([med_ab[i] for i in range(len(volts)) if volts[i] > 0])
    ok = np.isfinite(m)
    n, lnA = np.polyfit(np.log(v[ok]), np.log(m[ok]), 1)
    return float(n), float(np.exp(lnA)), v[ok], m[ok]


def main():
    v6, m6 = coinc_medians('run224466')     # BNC-T
    v9, m9 = coinc_medians('run224489')     # FIFO (current)
    y88 = json.loads((CALIB / 'y88_energy_calib.json').read_text())['channels']

    rec = {}
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            pmt = f'PSS{st}{b + 1}'
            n9, A9, _, _ = powerlaw(v9, m9[ai, b])
            n6, A6, _, _ = powerlaw(v6, m6[ai, b])
            e699_fifo = y88[pmt]['edge699_mv'] * FIFO[pmt]     # mV at Vge, FIFO
            mv_per_kevee = e699_fifo / E699_KEV
            # equalization bias to put 699 keVee at each target amplitude (FIFO):
            eq = {str(T): round(VGE[pmt] * (T / e699_fifo) ** (1.0 / n9), 0)
                  for T in EQ_TARGETS_MV}
            rec[pmt] = dict(n_fifo=round(n9, 2), n_bnct=round(n6, 2),
                            fifo_over_bnct=FIFO[pmt], V_equalized=VGE[pmt],
                            edge699_bnct_mv=y88[pmt]['edge699_mv'],
                            edge699_fifo_mv=round(e699_fifo, 2),
                            mv_per_kevee_fifo=round(mv_per_kevee, 4),
                            eq_bias_for_699_at=eq)

    out = {'note': 'Plastic PMT gain curves anchored to the Y-88 699 keVee edge, '
                   'CURRENT (FIFO) readout. eq_bias_for_699_at[T] = per-PMT bias '
                   'that puts the 698.63 keVee Compton edge at T mV (equalized). '
                   'Any energy scales linearly: amp_mV(E_keVee, V) = '
                   '(E_keVee/698.63)*edge699_fifo_mv*(V/V_equalized)^n_fifo.',
           'threshold_mv': THRESH_MV, 'fullscale_mv': FULLSCALE_MV,
           'y88_vs_triples_mip': 'Y-88 absolute scale => triples MIP (19d, assumed '
                                 '5.05 MeV) really ~130 keVee; ~40x disagreement, '
                                 'trust Y-88, recheck triples with a long run.',
           'pmts': rec}
    (CALIB / 'plastic_hv_gain_absolute.json').write_text(json.dumps(out, indent=2))

    # ---- report ----
    print('=== Plastic gain, CURRENT (FIFO) config, Y-88-anchored ===')
    print(f'{"PMT":6s} {"n":>4s} {"FIFO":>5s} {"Vge":>5s} {"699@Vge mV":>10s} '
          f'{"mV/keVee":>8s}  equalize 699 keVee to:')
    print(f'{"":42s} ' + '  '.join(f'{T}mV' for T in EQ_TARGETS_MV))
    for pmt in PMTS:
        r = rec[pmt]
        eqs = '  '.join(f'{r["eq_bias_for_699_at"][str(T)]:.0f}'
                        for T in EQ_TARGETS_MV)
        print(f'{pmt:6s} {r["n_fifo"]:4.1f} {r["fifo_over_bnct"]:5.2f} '
              f'{r["V_equalized"]:5d} {r["edge699_fifo_mv"]:10.1f} '
              f'{r["mv_per_kevee_fifo"]:8.4f}  {eqs}')
    # recommend equalizing to the target that keeps every PMT within CAEN-sane
    # bias and the 699 edge well above threshold
    print(f'\nTrigger threshold {THRESH_MV} mV, full scale {FULLSCALE_MV:.0f} mV.')
    print('At each equalized target T, 699 keVee sits at T mV on every PMT; the '
          '~130 keVee MIP would sit at T*130/699 mV (e.g. T=50 -> ~9 mV, ~2x '
          'threshold).')

    figures(v6, m6, v9, m9, rec)
    print('\n-> calib/plastic_hv_gain_absolute.json & figures/21_y88/hv_*.png')


def figures(v6, m6, v9, m9, rec):
    arm_c = dict(zip('ABCD', ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']))
    vv = np.linspace(1150, 1650, 50)

    # (1) gain curves both scans
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            pmt = f'PSS{st}{b + 1}'
            for run, (v, m), mk in ((6, (v6, m6), 'o'), (9, (v9, m9), 's')):
                vs = [v[i] for i in range(len(v)) if v[i] > 0]
                ms = [m[ai, b, i] for i in range(len(v)) if v[i] > 0]
                ax[0].plot(vs, ms, mk, ms=3, color=arm_c[st], alpha=0.5 if run == 6 else 1)
    ax[0].set_yscale('log'); ax[0].set_xlabel('PMT bias [V]')
    ax[0].set_ylabel('coincident-median amplitude [mV]')
    ax[0].set_title('Plastic gain curves: circles=BNC-T (224466), squares=FIFO (224489)')
    ax[0].grid(alpha=0.3, which='both')

    # (2) absolute mV/keVee vs bias (FIFO), Y-88 anchors
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            pmt = f'PSS{st}{b + 1}'
            r = rec[pmt]
            g = r['mv_per_kevee_fifo'] * (vv / r['V_equalized']) ** r['n_fifo']
            ax[1].plot(vv, g * 1000, '-', color=arm_c[st], lw=1,
                       ls='-' if b == 0 else '--')
            ax[1].plot(r['V_equalized'], r['mv_per_kevee_fifo'] * 1000, 'o',
                       color=arm_c[st], ms=6)
    ax[1].set_xlabel('PMT bias [V]'); ax[1].set_ylabel('gain [mV / MeVee] (FIFO)')
    ax[1].set_title('Absolute plastic gain (Y-88-anchored); dots = Y-88 bias')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'hv_gain_curves.png', dpi=140)
    plt.close(fig)

    # (3) equalization: bias to place 699 keVee at each target
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pmt in PMTS:
        Ts = EQ_TARGETS_MV
        Vs = [rec[pmt]['eq_bias_for_699_at'][str(T)] for T in Ts]
        ax.plot(Ts, Vs, 'o-', ms=4, label=pmt)
    ax.set_xlabel('target 699 keVee amplitude [mV] (the "scale-together" knob)')
    ax.set_ylabel('required PMT bias [V]')
    ax.set_title('Equalized bias per PMT vs the common 699-keVee target amplitude')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'hv_absolute_scale.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
