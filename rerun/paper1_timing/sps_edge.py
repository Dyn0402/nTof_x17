#!/usr/bin/env python3
"""sps_edge.py <sps registry key> --out O.npz [--n N] -- the SPS strip sums S(t) for the column-start timing (Paper I §7).

Reads one flat SPS row exactly as rerun/sps_audit/footprint.py does (plateau by wall clock, the oscillating and per-plateau
hot channels out of the seed, seed >= 50 ADC, no strip minimum; sps_cluster.cluster gives the cluster) and saves, per event
and view, the summed waveform S(t) of the cluster's strips (+- 1.5 pitch of the centroid plus half the cluster, censored
strips out: footprint.leading_edge's sum), with ftst, t_wall, the cluster's charge and strip count.  The estimators
(estimators.py) are fitted offline by sps_timing.py.  Run inside the frozen code archive (WFT_REPO), on condor.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'rerun', 'sps_audit'),
                os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc   # noqa: E402
import sps_cluster              # noqa: E402

QMIN, MIN_STRIPS = 50.0, 1


def strip_sum(P, c, bad):
    W = np.asarray(P['W'], float); pos = np.asarray(P['pos'], float); ch = np.asarray(P['ch'], int)
    m = (np.abs(pos - c['pos']) <= 1.5 * sps_cluster.PITCH_MM + 0.5 * c['n_strips'] * sps_cluster.PITCH_MM) & ~np.isin(ch, list(bad))
    return W[m].sum(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('key'); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6000)
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    dead = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json')))
    orig = sbc.event_record

    def record(ch, sm, am, raw, nsmp, half_win=16, qmin=300.0, tan=0.0, seed_bad=None, min_strips=4, couple=None,
               keep_peaks=False):
        # the build_cache's own hot / coupled-channel probe (keep_peaks=True) runs on the PASS selection; the records
        # themselves drop the pass's 300 ADC / 4-strip cut (footprint.py's resolution sample)
        if keep_peaks:
            return orig(ch, sm, am, raw, nsmp, half_win, qmin, tan, seed_bad=seed_bad, min_strips=min_strips,
                        couple=couple, keep_peaks=True)
        return orig(ch, sm, am, raw, nsmp, half_win, QMIN, tan, seed_bad=seed_bad, min_strips=MIN_STRIPS, couple=couple)

    sbc.event_record = record
    evs = {}
    for sub in row['subruns'].split():
        sbc.HOT_LAST.clear()
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=a.n - len(evs), tan=0.0, only=[row['plateau']],
                              verbose=False).get(row['plateau'], {})
        hot = np.zeros(512, bool)
        for (r_, s_, L_), chs in sbc.HOT_LAST.items():
            if s_ == sub and L_ == row['plateau']:
                hot[chs] = True
        for k, e in got.items():
            e['_hot'] = hot
        evs.update(got)
        if len(evs) >= a.n:
            break
    out = {k: [] for k in ('ev', 'ftst', 't_wall', 'x_S', 'y_S', 'x_q', 'y_q', 'x_n', 'y_n', 'x_pos', 'y_pos', 'x_ok', 'y_ok')}
    ns = None
    for k, e in evs.items():
        out['ev'].append(k); out['ftst'].append(int(e.get('ftst_x', -1))); out['t_wall'].append(float(e.get('t_wall', np.nan)))
        for v in 'xy':
            bad = set(dead[v]['dead']) | set(sbc.RAW_CM_BAD) | set(np.flatnonzero(e['_hot']).tolist())
            c = sps_cluster.cluster(e[v], bad=bad)
            if ns is None:
                ns = np.asarray(e[v]['W']).shape[1]
            if c is None:
                S = np.zeros(ns); q = n = p = np.nan; ok = False
            else:
                S = strip_sum(e[v], c, bad); q, n, p, ok = c['q_sum'], c['n_strips'], c['pos'], True
            S = np.asarray(S, float)
            if len(S) != ns:
                S = np.r_[S, np.zeros(max(ns - len(S), 0))][:ns]
            out[f'{v}_S'].append(S.astype(np.float32)); out[f'{v}_q'].append(q); out[f'{v}_n'].append(n)
            out[f'{v}_pos'].append(p); out[f'{v}_ok'].append(ok)
    np.savez_compressed(a.out, key=a.key, **{k: np.asarray(v) for k, v in out.items()})
    print(f'{a.key}: {len(out["ev"])} events, {ns} samples -> {a.out}')


if __name__ == '__main__':
    main()
