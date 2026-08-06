#!/usr/bin/env python3
"""
run58_column_scan.py — drift-column length vs drift voltage, per detector.

Aggregates the per-sub-run parquets produced on lxplus/condor by
`lxplus/run58_columns.py` (see `lxplus/README.md`). run_58 sweeps the drift
voltage 700 -> 200 V with a **64-sample (3.84 us) window that contains the whole
column at every point**, so this is the one July dataset that can answer:

  **does each chamber's drift field actually respond to its supply?**

A chamber with a real, graded field must show the column lengthening as the
field is lowered (on the clean detector A the existing run_58 analysis sees
T_max 972 ns @700 V -> 1975 ns @200 V, a factor 2). A chamber whose cathode
potential is not set by the supply will not track it. Detector B draws zero
bleeder current at every voltage of this scan (its degrador divider is absent —
`HANDOFF_2026-07-30_readout_window_and_detB.md` §4.3b-c), so B is the case
under test.

    .venv/bin/python mx_july_beam_qa/run58_column_scan.py \
        [--dir mx_july_beam_qa/cache/run58_columns] [--min-span 3]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_NS = 60.0
GAP_MM = 30.0            # nominal mechanical gap; absolute v carries this


def load(d):
    files = sorted(glob.glob(os.path.join(d, 'columns_*.parquet')))
    if not files:
        raise SystemExit(f'no parquets in {d} — pull them from lxplus first:\n'
                         "  rsync -av -e 'ssh -K -o ControlPath=none' "
                         f"lxplus:x17run58/out/ {d}/")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'{len(files)} sub-runs, {len(df):,} rows')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(HERE, 'cache', 'run58_columns'))
    ap.add_argument('--min-span', type=float, default=3.0,
                    help='samples; drops isochronous point deposits')
    ap.add_argument('--ladder', type=float, default=0.7)
    ap.add_argument('--amp-min', type=float, default=300.0)
    ap.add_argument('--out', default=os.path.join(HERE, 'figures',
                                                  'run58_column_scan.png'))
    args = ap.parse_args()

    df = load(args.dir)
    live = df[df['busy'] == False].dropna(subset=['span'])          # noqa: E712
    cl = live[(live['ladder'] > args.ladder) & live['n'].between(5, 25)
              & (live['amp'] > args.amp_min) & (live['span'] > args.min_span)]
    print(f'{len(live):,} clusters -> {len(cl):,} clean gap-crossing columns\n')

    print('median column length [samples of 60 ns] vs drift voltage')
    drifts = sorted(cl['drift'].unique())
    hdr = f'{"det":5s} ' + ' '.join(f'{d:>7d}' for d in drifts)
    print(hdr)
    print('-' * len(hdr))
    table = {}
    for det in 'ABCD':
        for plane in 'xy':
            c = cl[(cl['det'] == det) & (cl['plane'] == plane)]
            row, ns = [], []
            for d in drifts:
                s = c[c['drift'] == d]['span']
                if len(s) >= 15:
                    row.append(f'{np.median(s):7.1f}')
                    ns.append(float(np.median(s)))
                else:
                    row.append(f'{"-":>7s}')
                    ns.append(np.nan)
            table[det + plane] = ns
            print(f'{det}{plane:4s} ' + ' '.join(row))

    print('\nresponse to the field: column length at 200 V / at 700 V')
    print('(a real drift field must lengthen the column as E falls; the clean')
    print(' detector A is the control)')
    for k, v in table.items():
        v = np.array(v, float)
        lo = v[0] if np.isfinite(v[0]) else np.nan          # lowest drift
        hi = v[-1] if np.isfinite(v[-1]) else np.nan        # highest drift
        if np.isfinite(lo) and np.isfinite(hi) and hi > 0:
            print(f'  {k}: {lo:.1f} / {hi:.1f} = {lo / hi:.2f}x')

    print('\nimplied v_drift [um/ns] = 30 mm / (span * 60 ns)  '
          '(RELATIVE only: a fixed amplitude cut truncates both column ends)')
    print(hdr)
    for k, v in table.items():
        print(f'{k:5s} ' + ' '.join(
            f'{GAP_MM * 1e3 / (x * SAMPLE_NS):7.1f}' if np.isfinite(x) else f'{"-":>7s}'
            for x in v))

    print('\nfraction of columns hitting the 64-sample ceiling (truncation check)')
    print(hdr)
    for det in 'ABCD':
        for plane in 'xy':
            c = cl[(cl['det'] == det) & (cl['plane'] == plane)]
            row = []
            for d in drifts:
                s = c[c['drift'] == d]
                row.append(f'{100 * s["ceil"].astype(bool).mean():6.1f}%'
                           if len(s) >= 15 else f'{"-":>7s}')
            print(f'{det}{plane:4s} ' + ' '.join(row))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for det, col in zip('ABCD', ('C0', 'C3', 'C2', 'C1')):
            for plane, ls in (('x', '-'), ('y', '--')):
                v = np.array(table[det + plane], float)
                axes[0].plot(drifts, v, ls, color=col, marker='o', ms=4,
                             label=f'{det}{plane}')
                axes[1].plot(drifts, GAP_MM * 1e3 / (v * SAMPLE_NS), ls,
                             color=col, marker='o', ms=4)
        axes[0].set_xlabel('drift voltage [V]')
        axes[0].set_ylabel('median column length [samples]')
        axes[0].set_title('run_58: drift column vs field (64-smp window)')
        axes[0].legend(fontsize=7, ncol=4)
        axes[0].grid(alpha=.3)
        axes[1].set_xlabel('drift voltage [V]')
        axes[1].set_ylabel(r'implied $v_{drift}$ [$\mu$m/ns], gap = 30 mm')
        axes[1].grid(alpha=.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, dpi=140)
        print('\nwrote', args.out)
    except Exception as e:
        print('(no figure:', e, ')')


if __name__ == '__main__':
    main()
