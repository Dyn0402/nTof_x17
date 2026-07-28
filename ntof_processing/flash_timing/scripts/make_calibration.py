#!/usr/bin/env python3
"""Build flash_timing_calibration.json from the per-channel measurements.

The calibration is a set of constants C such that, for any run and any bunch,

    t_flash(detector) = tof_PKUP(bunch) + C[detector]        [ns]

where tof_PKUP is the time of that bunch's PKUP pulse in the same acquisition
window.  PKUP is never gated and its flash finder never fails, so this works in
every run regardless of the SiPM divert state and regardless of whether the
PSA's own tflash for that tree is any good.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

OFF_RUNS = [224356, 224357, 224358, 224359, 224360, 224464, 224466]
EPOCH = {224356: '2026-07-11', 224357: '2026-07-11', 224358: '2026-07-11',
         224359: '2026-07-11', 224360: '2026-07-11',
         224464: '2026-07-16', 224466: '2026-07-16'}


def load_rows():
    rows = []
    for r in csv.DictReader(open(DATA / 'per_channel_flash_timing.csv')):
        d = {}
        for k, v in r.items():
            if k == 'tree':
                d[k] = v
            elif v == '':
                d[k] = float('nan')
            else:
                d[k] = float(v)
        rows.append(d)
    return rows


def agg(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return (float(v.mean()), float(v.std()), len(v)) if len(v) else (float('nan'), float('nan'), 0)


def main():
    rows = load_rows()
    runs = sorted({int(r['run']) for r in rows})
    off = [r for r in rows if int(r['run']) in OFF_RUNS]
    trees = sorted({r['tree'] for r in off})

    cal = {}
    for t in trees:
        chans = sorted({int(r['ch']) for r in off if r['tree'] == t})
        ch_out = {}
        for ch in chans:
            sub = [r for r in off if r['tree'] == t and int(r['ch']) == ch]
            m, s, n = agg([r['dt_mean'] for r in sub])
            sig, _, _ = agg([r['dt_sigma'] for r in sub])
            # epoch split
            e1 = agg([r['dt_mean'] for r in sub if EPOCH[int(r['run'])] == '2026-07-11'])
            e2 = agg([r['dt_mean'] for r in sub if EPOCH[int(r['run'])] == '2026-07-16'])
            core, _, _ = agg([r['frac_core'] for r in sub])
            # a channel is trustworthy if its per-bunch spread is small AND the
            # flash hit is found in most bunches
            quality = 'good' if (sig < 8 and core > 0.85) else (
                'usable' if (sig < 25 and core > 0.4) else 'bad')
            ch_out[str(ch)] = dict(
                C_ns=round(m, 2), run_to_run_std_ns=round(s, 2), n_runs=n,
                per_bunch_sigma_ns=round(sig, 2),
                frac_bunches_with_flash=round(core, 3), quality=quality,
                C_2026_07_11=round(e1[0], 2) if np.isfinite(e1[0]) else None,
                C_2026_07_16=round(e2[0], 2) if np.isfinite(e2[0]) else None,
            )
        # the tree constant uses only the trustworthy channels
        cvals = [v['C_ns'] for v in ch_out.values() if v['quality'] != 'bad']
        if not cvals:
            cvals = [v['C_ns'] for v in ch_out.values()]
        cal[t] = dict(
            C_ns=round(float(np.mean(cvals)), 2),
            excluded_channels=[k for k, v in ch_out.items() if v['quality'] == 'bad'],
            channel_spread_std_ns=round(float(np.std(cvals)), 2),
            channel_spread_range_ns=[round(min(cvals), 1), round(max(cvals), 1)],
            n_channels=len(cvals),
            channels=ch_out,
        )

    out = {
        '_schema': {
            'usage': 't_flash_at_detector(bunch) = tof_PKUP(bunch) + C_ns',
            'reference': 'PKUP pulse time: tof of the largest-amplitude PKUP hit of that '
                         'bunch (numerically equal to the PKUP tflash, whose finder never fails)',
            'sign': 'C_ns is NEGATIVE: the flash reaches the detectors ~1.7 us BEFORE the '
                    'pickup pulse appears in the digitiser window',
            'scope': 'C is an instrumental constant (flight + cable + front-end). It does not '
                     'depend on the SiPM divert state, so it applies to every run in the '
                     'campaign, not only the divert-off ones.',
            'units': 'ns',
        },
        'measured': {
            'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'runs': [r for r in runs if r in OFF_RUNS],
            'why_these_runs': 'the only runs in the 2026 EAR2 X17 campaign taken with the SiPM '
                              'wall blanking gate disabled, so the walls record the true gamma '
                              'flash instead of the gate transient (see ../FLASH_DIVERT_OFF_RUNS.md)',
            'method': 'per (bunch, channel) the largest-amplitude hit within +-300 ns of the '
                      'bunch flash anchor; its `tof` minus the same bunch PKUP `tof`; median '
                      'over bunches after a +-100 ns core cut that removes PSA mis-tags',
        },
        'constants': cal,
    }
    p = DATA / 'flash_timing_calibration.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print('wrote', p)
    for t in trees:
        c = cal[t]
        print(f"  {t}: C = {c['C_ns']:+9.2f} ns   channel spread {c['channel_spread_std_ns']:.2f} "
              f"(range {c['channel_spread_range_ns'][0]} .. {c['channel_spread_range_ns'][1]})")


if __name__ == '__main__':
    main()
