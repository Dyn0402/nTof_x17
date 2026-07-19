"""23_y88_energy_calib.py — Y-88 energy calibration per channel (T3).

From the Compton-edge positions (22 output) build mV-per-keVee per channel for
all three organic-scintillator detectors, on the reprocessed files (new PSA with
WALL+LIQ pulse-shape fitting):
  - Plastics (PSS): two independent edges (699 & 1612 keVee) -> through-origin
    slope, plus the measured 1612/699 ratio as a linearity check (expect 2.307).
  - SiPM walls (WAL) and liquids (LIQ): the single 699 keVee bump -> one slope.
The liquid slope is the FIRST LIQ absolute energy scale.

Cross-detector check: the three detector types view the SAME source, so their
699 keVee edges are compared per arm (they are different detectors at different
gains, so this is a sanity/consistency view, not an equality test).

NOTE the run-224404 wall-MIP cross-check from the earlier (official-file) pass
is dropped here: 224404 was reconstructed with the OLD PSA while these Y-88 runs
now use the new pulse-shape-fitting PSA, so that comparison is no longer
apples-to-apples. A same-PSA wall MIP (e.g. from run 224489) would restore it.

Still needs Dylan: transporting the plastic edges to nominal HV (the plastic HV
was raised for these runs; the values are not in the data).

Outputs:
  calib/y88_energy_calib.json          per channel: mV/keVee (+ ratio for PSS)
  figures/21_y88/energy_calib.png      mV/keVee per channel + per-arm 699 compare
Usage: python 23_y88_energy_calib.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
CALIB = BASE / 'calib'
OUT = BASE / 'figures' / '21_y88'
RUNS = ['run224476', 'run224477', 'run224478', 'run224479']
E699, E1612 = 698.63, 1612.06


def main():
    calib = {'note': 'Y-88 energy calibration (mV per keVee) at the run 224476-79'
                     ' HV, from the reprocessed (new-PSA) files. Plastic HV was'
                     ' RAISED for these runs; transporting to nominal needs the'
                     ' HV values from Dylan. PSS: two independent Compton edges;'
                     ' WAL/LIQ: the single 699 keVee bump. LIQ slope is the'
                     ' first LIQ absolute energy scale.',
             'channels': {}}

    for r in RUNS:
        d = json.loads((CALIB / f'y88_edges_{r}.json').read_text())
        arm = d['source_arm']
        for ch, v in d['channels'].items():
            if not v['edges']:
                continue
            e = {edge['kevee']: edge for edge in v['edges']}
            rec = dict(kind=v['kind'], arm=arm, run=r)
            if v['kind'] == 'PSS' and E699 in e and E1612 in e:
                m699, m1612 = e[E699]['edge_mv'], e[E1612]['edge_mv']
                x = np.array([E699, E1612]); y = np.array([m699, m1612])
                slope = float((x @ y) / (x @ x))          # mV/keVee, through 0
                rec.update(mv_per_kevee=round(slope, 5),
                           mv_per_mevee=round(slope * 1000, 3),
                           edge699_mv=m699, edge1612_mv=m1612,
                           ratio_1612_over_699=v['edge_ratio_1612_over_699'],
                           nonlinearity=round(v['edge_ratio_1612_over_699']
                                              / (E1612 / E699), 3)
                           if v['edge_ratio_1612_over_699'] else None)
            else:                                          # WAL / LIQ single edge
                m699 = e[E699]['edge_mv']
                rec.update(mv_per_kevee=round(m699 / E699, 5),
                           mv_per_mevee=round(m699 / E699 * 1000, 3),
                           edge699_mv=m699)
            calib['channels'][ch] = rec

    (CALIB / 'y88_energy_calib.json').write_text(json.dumps(calib, indent=2))
    report(calib)
    figure(calib)
    print('\n-> calib/y88_energy_calib.json  &  figures/21_y88/energy_calib.png')


def report(calib):
    ch = calib['channels']
    print('=== PLASTICS (two edges) ===')
    print(f'{"ch":7s} {"699mV":>6s} {"1612mV":>7s} {"mV/MeVee":>9s} '
          f'{"ratio":>6s} {"nonlin":>6s}')
    for k, r in ch.items():
        if r['kind'] == 'PSS':
            print(f'{k:7s} {r["edge699_mv"]:6.1f} {r["edge1612_mv"]:7.1f} '
                  f'{r["mv_per_mevee"]:9.1f} {str(r["ratio_1612_over_699"]):>6s} '
                  f'{str(r["nonlinearity"]):>6s}')
    for kind, lab in (('LIQ', 'LIQUIDS (first LIQ scale)'), ('WAL', 'WALLS')):
        rows = [(k, r) for k, r in ch.items() if r['kind'] == kind]
        print(f'\n=== {lab}: 699 keVee bump ===')
        print(f'{"ch":7s} {"699mV":>6s} {"mV/MeVee":>9s}')
        for k, r in rows:
            print(f'{k:7s} {r["edge699_mv"]:6.1f} {r["mv_per_mevee"]:9.1f}')


def figure(calib):
    ch = calib['channels']
    pss = {k: v for k, v in ch.items() if v['kind'] == 'PSS'}
    liq = {k: v for k, v in ch.items() if v['kind'] == 'LIQ'}
    wal = {k: v for k, v in ch.items() if v['kind'] == 'WAL'}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (1) mV/MeVee per plastic channel
    ks = list(pss)
    ax[0].bar(ks, [pss[k]['mv_per_mevee'] for k in ks], color='crimson')
    ax[0].set_ylabel('mV / MeVee (raised HV)')
    ax[0].set_title('Plastic Y-88 gain per channel')
    ax[0].tick_params(axis='x', rotation=45, labelsize=8)

    # (2) plastic two-edge linearity (measured ratio, not imposed)
    for k in ks:
        ax[1].plot([E699, E1612], [pss[k]['edge699_mv'], pss[k]['edge1612_mv']],
                   'o-', ms=5, label=f"{k} (r={pss[k]['ratio_1612_over_699']})")
    ax[1].set_xlabel('Compton edge energy [keVee]')
    ax[1].set_ylabel('amplitude [mV]')
    ax[1].set_title('Plastic two-edge linearity (ratio measured, exp 2.31)')
    ax[1].legend(fontsize=6, ncol=2)
    ax[1].grid(alpha=0.3)

    # (3) 699 keVee edge per arm, per detector type (same source, sanity view)
    arms = 'ABCD'
    x = np.arange(4)
    pss_arm = [np.mean([v['edge699_mv'] for v in pss.values() if v['arm'] == a])
               for a in arms]
    liq_arm = [next((v['edge699_mv'] for v in liq.values() if v['arm'] == a),
                    np.nan) for a in arms]
    wal_arm = [np.median([v['edge699_mv'] for v in wal.values()
                          if v['arm'] == a] or [np.nan]) for a in arms]
    ax[2].bar(x - 0.25, pss_arm, 0.25, label='PSS (mean)', color='crimson')
    ax[2].bar(x, liq_arm, 0.25, label='LIQ', color='darkgreen')
    ax[2].bar(x + 0.25, wal_arm, 0.25, label='WAL (median)', color='steelblue')
    ax[2].set_xticks(x); ax[2].set_xticklabels(list(arms))
    ax[2].set_xlabel('arm'); ax[2].set_ylabel('699 keVee edge [mV]')
    ax[2].set_title('699 keVee edge per arm & detector (same source)')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'energy_calib.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
