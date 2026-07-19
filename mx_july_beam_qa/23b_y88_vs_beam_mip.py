"""23b_y88_vs_beam_mip.py — do the Y-88 Compton-edge calibration and the beam-MIP
calibration agree?

The SiPM walls are the clean comparison: they are biased all together at one
fixed voltage that did not change between the Y-88 source runs (224476-79,
morning of 2026-07-17) and the beam run 224489 (same day, evening), and BOTH are
now reconstructed with the same (Mucciola) PSA. So the two calibrations can be
compared DIRECTLY, with no HV transport:

  - Y-88:  the 698.63 keVee Compton-edge bump  (22 output, calib/y88_energy_calib)
  - beam:  the through-going MIP peak, sideband-subtracted wall x (top/bottom)
           coincidence  (07 cache, run 224489)

If the two agree and the response is linear, edge_mV / MIP_mV = 0.699 / E_MIP,
so the ratio measures the MIP deposit E_MIP (expected a few hundred keV in a
thin bar) — the same number for every channel if the calibrations are
consistent.

Plastics are NOT transported here: the Y-88 plastic HV was raised well above the
operating point (at the gain-equalized ~1180-1417 V setpoints the 699 keVee edge
would be ~1 mV, below threshold — yet the spectra run to ~100 mV), and the exact
raised value is not recorded, so an absolute plastic comparison is still blocked.
What IS shown for the plastics: the per-channel Y-88 699 edges are equal to
+-15%, i.e. consistent with the channels being gain-equalized.

Outputs:
  figures/21_y88/y88_vs_beam_mip.png
  calib/y88_vs_beam_mip.json
Usage: python 23b_y88_vs_beam_mip.py
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
E699 = 698.63            # keVee, the 898 keV Compton edge


def beam_wall_mip_mv(run_stem='run224489'):
    """Per-channel beam wall-MIP peak (mV): sideband-subtracted wall coincidence
    spectrum (07 cache), log-binned mode. {WALx{c}: mV}."""
    z = np.load(CACHE / f'07_mip_{run_stem}.npz')
    ae = z['amp_edges']
    cen = np.sqrt(ae[:-1] * ae[1:])
    sb = float(z['sb_scale']) if 'sb_scale' in z else 1.0
    fac = json.loads((CALIB / f'adc_to_mv_{run_stem}.json').read_text())['factors']
    out = {}
    for arm in 'ABCD':
        w = z[f'{arm}_wal']
        for c in range(8):
            net = w[0, c] - sb * w[1, c]
            reg = cen > 300
            if net[reg].max() <= 0:
                continue
            pk = cen[reg][np.argmax(net[reg])]
            out[f'WAL{arm}{c + 1}'] = pk * fac[f'WAL{arm}'][str(c + 1)]
    return out


def main():
    cal = json.loads((CALIB / 'y88_energy_calib.json').read_text())['channels']
    mip = beam_wall_mip_mv()

    rows = []
    for ch, v in cal.items():
        if v['kind'] == 'WAL' and ch in mip:
            edge = v['edge699_mv']
            rows.append((ch, edge, mip[ch], edge / mip[ch]))
    ratios = np.array([r[3] for r in rows])
    e_mip_implied = E699 / 1000 / ratios       # MeV

    result = {'comparison': 'Y-88 699 keVee wall Compton edge vs beam (224489) '
                            'wall MIP peak, same SiPM HV & same PSA — no transport.',
              'edge_over_mip_median': round(float(np.median(ratios)), 3),
              'implied_E_MIP_MeV_median': round(float(np.median(e_mip_implied)), 3),
              'channels': {r[0]: dict(y88_edge_mv=round(r[1], 2),
                                      beam_mip_mv=round(r[2], 2),
                                      edge_over_mip=round(r[3], 3))
                           for r in rows}}
    (CALIB / 'y88_vs_beam_mip.json').write_text(json.dumps(result, indent=2))

    print('=== SiPM wall: Y-88 699 keVee edge vs beam MIP (224489, same PSA/HV) ===')
    print(f'{"ch":7s} {"Y88 edge":>8s} {"beam MIP":>8s} {"edge/MIP":>8s} '
          f'{"impl E_MIP":>10s}')
    for (ch, edge, m, r), em in zip(rows, e_mip_implied):
        print(f'{ch:7s} {edge:8.1f} {m:8.1f} {r:8.2f} {em * 1000:9.0f} keV')
    print(f'\nmedian edge/MIP = {np.median(ratios):.2f}  '
          f'-> implied MIP deposit {np.median(e_mip_implied) * 1000:.0f} keV')
    print('arm A ratio ~1.0 (Y-88 and beam MIP agree); B/C/D ~0.7 (Y-88 ~30% '
          'below), i.e. a per-arm morning-vs-evening SiPM gain difference and/or '
          'the log-mode MIP bias.')

    figure(rows, ratios)
    print('\n-> calib/y88_vs_beam_mip.json  &  figures/21_y88/y88_vs_beam_mip.png')


def figure(rows, ratios):
    chs = [r[0] for r in rows]
    edges = [r[1] for r in rows]
    mips = [r[2] for r in rows]
    x = np.arange(len(chs))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))

    ax[0].bar(x - 0.2, edges, 0.4, label='Y-88 699 keVee edge', color='crimson')
    ax[0].bar(x + 0.2, mips, 0.4, label='beam MIP (224489)', color='steelblue')
    ax[0].set_xticks(x); ax[0].set_xticklabels(chs, rotation=45, fontsize=8)
    ax[0].set_ylabel('amplitude [mV]')
    ax[0].set_title('SiPM wall: Y-88 Compton edge vs beam MIP (same HV & PSA)')
    ax[0].legend(fontsize=9)

    ax[1].axhline(1.0, color='k', ls='--', lw=1, label='edge=MIP (E$_{MIP}$=699 keV)')
    ax[1].axhline(0.699 / 0.6, color='gray', ls=':', lw=1,
                  label='E$_{MIP}$=600 keV')
    ax[1].bar(x, ratios, 0.6, color='seagreen')
    ax[1].set_xticks(x); ax[1].set_xticklabels(chs, rotation=45, fontsize=8)
    ax[1].set_ylabel('Y-88 edge / beam MIP')
    ax[1].set_title(f'ratio per channel (median {np.median(ratios):.2f})')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'y88_vs_beam_mip.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
