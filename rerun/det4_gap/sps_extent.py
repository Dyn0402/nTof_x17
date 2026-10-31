#!/usr/bin/env python3
"""sps_extent.py <registry key> --out DIR [--per-plateau 3000] [--half-win 32]

det4's drift gap with no timing (rerun/det4_gap/README.md): the per-strip, time-integrated charge of each event in a
+-half_win-strip window, from the pass's own cache builder (rerun/model_audit/sps_build_cache.build_cache: the same
baseline, seed rule, hot and coupled channels as the final SPS pass), so that the whole inclined column of a rotated
row fits in the window (the pass keeps +-16 strips = +-12.5 mm around the brightest strip; at 25.4 deg a 28-30 mm gap
projects to 13-14 mm, which can overrun one side).

Writes DIR/<key>.npz: per event and view the strip positions (mm), the window integral of the positive samples, the
peak sample and its index (the time-position line of an inclined track separates its strips from noise and pile-up), and
a 'bad' flag for the strips the pass censors (oscillating, hot, coupled in this event), which carry no charge
information and are left out of both charge and exposure downstream.
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis'),
                os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
import sps_build_cache as sbc                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--out', required=True)
    ap.add_argument('--registry', default=os.path.join(REPO, 'rerun', 'sps_registry.csv'))
    ap.add_argument('--per-plateau', type=int, default=3000); ap.add_argument('--half-win', type=int, default=32)
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(a.registry))}[a.key]
    mount = float(row['mount'].split()[0]) if row['mount'].split()[0] not in ('flat', '0') else 0.0
    tan = float(np.tan(np.radians(mount)))
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=a.per_plateau - len(evs),
                              half_win=a.half_win, tan=tan, only=[row['plateau']])
        for e in got.get(row['plateau'], {}).items():
            evs[(sub, e[0])] = e[1]
        if len(evs) >= a.per_plateau:
            break
    n = 2 * a.half_win + 1
    out = dict(key=a.key, mount_deg=mount, half_win=a.half_win, subrun=[], eid=[])
    for v in ('x', 'y'):
        out[f'{v}_pos'] = np.full((len(evs), n), np.nan, np.float32)
        out[f'{v}_q'] = np.zeros((len(evs), n), np.float32)
        out[f'{v}_bad'] = np.ones((len(evs), n), bool)
        out[f'{v}_pk'] = np.zeros((len(evs), n), np.float32)          # peak sample (ADC)
        out[f'{v}_tpk'] = np.full((len(evs), n), -1, np.int8)         # its sample index
        out[f'{v}_seed'] = np.zeros(len(evs), np.int16)
    bad = sbc.SEED_BAD
    for i, ((sub, eid), e) in enumerate(evs.items()):
        out['subrun'].append(sub); out['eid'].append(int(eid))
        for v in ('x', 'y'):
            W = e[v]['W']; ch = e[v]['ch'].astype(int); m = len(ch)
            q = np.clip(W, 0, None).sum(1)
            # blanked strips (oscillating / hot / coupled in this event) are all-zero in W: flag them
            b = bad[ch] | np.isin(ch, e.get(f'censor_{v}', []))
            seed_hot = np.array([c in sbc.HOT_LAST.get((row['run'], sub, row['plateau']), []) for c in ch], bool)
            b |= seed_hot & (W.max(1) <= 0)
            out[f'{v}_pos'][i, :m] = e[v]['pos']; out[f'{v}_q'][i, :m] = q; out[f'{v}_bad'][i, :m] = b
            out[f'{v}_pk'][i, :m] = W.max(1); out[f'{v}_tpk'][i, :m] = W.argmax(1)
            out[f'{v}_seed'][i] = e[f'seed_ch_{v}']
    out['subrun'] = np.array(out['subrun']); out['eid'] = np.array(out['eid'])
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, f'{a.key}.npz'), **out)
    print(f'{a.key}: {len(evs)} events -> {a.out}/{a.key}.npz', flush=True)


if __name__ == '__main__':
    main()
