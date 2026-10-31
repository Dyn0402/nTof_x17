#!/usr/bin/env python3
"""footprint.py <registry key> --out O.parquet [--n 3000]

The head-on charge footprint of one flat SPS row, without the pass's peak >= 300 ADC / >= 4 strips cut and without a fit:
the row's events are read exactly as sps_build_cache.build_cache reads them (plateau by wall clock, 20 s edges, the
oscillating and per-plateau hot channels out of the seed), each view is seeded at >= 50 ADC (5 x noise) with no strip
minimum, and sps_cluster.cluster gives the cluster (strips, lead fraction, neighbour ratios, summed peak charge, peak
time).  The fit's censored strips (the H4 dead list + RAW_CM_BAD + the hot ones) are bridged, as in the pass.
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                                     # noqa: E402
import sps_cluster                                                # noqa: E402

QMIN, MIN_STRIPS = 50.0, 1
#: absolute threshold [ADC] on the cluster's summed waveform for the column-start time (~6 x its noise for 3 strips)
T_ABS_ADC = 100.0


def leading_edge(P, c, bad):
    """(t50, t20, t_abs) [ns]: where the summed waveform of the cluster's strips (+-1.5 pitch of the cluster centroid, censored
    strips out) first crosses 50 % / 20 % of its maximum and T_ABS_ADC (the column start), linearly interpolated, on the rising edge before the maximum.
    At normal incidence the column start (the charge from next to the mesh) is the event's time; the wft fit's t0 is not,
    because with w = 0 every depth lands on the same strips and the profiled depth distribution absorbs a t0 shift."""
    if c is None:
        return np.nan, np.nan, np.nan
    W = np.asarray(P['W'], float); pos = np.asarray(P['pos'], float); ch = np.asarray(P['ch'], int)
    m = (np.abs(pos - c['pos']) <= 1.5 * sps_cluster.PITCH_MM + 0.5 * c['n_strips'] * sps_cluster.PITCH_MM) & ~np.isin(ch, list(bad))
    s = W[m].sum(0)
    if not len(s) or s.max() <= 0:
        return np.nan, np.nan, np.nan
    ip = int(np.argmax(s)); out = []
    for fr in (0.5, 0.2, None):
        th = fr * s[ip] if fr else min(T_ABS_ADC, 0.5 * s[ip]); t = np.nan
        for i in range(ip, 0, -1):
            if s[i - 1] < th <= s[i]:
                t = (i - 1 + (th - s[i - 1]) / (s[i] - s[i - 1])) * sps_cluster.SAMPLE_NS
                break
        out.append(float(t))
    return tuple(out)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument('key'); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=3000)
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    dead = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json')))
    orig = sbc.event_record
    evs = {}
    for sub in row['subruns'].split():
        sbc.HOT_LAST.clear()
        # the hot probe inside build_cache runs on the PASS selection (event_record is patched only after it)
        sbc.event_record = orig
        hot = np.zeros(512, bool)
        probe = sbc.build_cache(row['run'], sub, row['dataset'], sbc.HOT_SCAN, tan=0.0, only=[row['plateau']], verbose=False, hot_scan=0)
        for L, pe in probe.items():
            cnt = np.zeros(512)
            for e in pe.values():
                cnt[e['seed_ch_x']] += 1; cnt[e['seed_ch_y']] += 1
            hot |= sbc.hot_channels(cnt, max(len(pe), 1))
        sbc.event_record = lambda ch, sm, am, raw, nsmp, hw=16, qmin=300.0, tan=0.0, seed_bad=None, _h=hot: \
            orig(ch, sm, am, raw, nsmp, hw, QMIN, tan, seed_bad=sbc.SEED_BAD | _h, min_strips=MIN_STRIPS)
        got = sbc.build_cache(row['run'], sub, row['dataset'], a.n - len(evs), tan=0.0, only=[row['plateau']], verbose=False,
                              hot_scan=0).get(row['plateau'], {})
        for k, e in got.items():
            e['_hot'] = hot
        evs.update(got)
        if len(evs) >= a.n:
            break
    rows = []
    for k, e in evs.items():
        o = dict(key=a.key, ev=k, ftst=int(e.get('ftst_x', -1)), t_wall=float(e.get('t_wall', np.nan)))
        for v in 'xy':
            bad = set(dead[v]['dead']) | set(sbc.RAW_CM_BAD) | set(np.flatnonzero(e['_hot']).tolist())
            c = sps_cluster.cluster(e[v], bad=bad)
            o[f'{v}_t_le_ns'], o[f'{v}_t_le20_ns'], o[f'{v}_t_abs_ns'] = leading_edge(e[v], c, bad)
            o[f'{v}_seed_amp'] = float(np.asarray(e[v]['W']).max())
            for kk in ('pos', 'n_strips', 'q_sum', 'q_lead', 'lead_frac', 'nb_lo', 'nb_hi', 't_peak_ns'):
                o[f'{v}_c_{kk}'] = c[kk] if c else np.nan
        rows.append(o)
    pd.DataFrame(rows).to_parquet(a.out, index=False)
    print(f'{a.key}: {len(rows)} events -> {a.out}')


if __name__ == '__main__':
    main()
