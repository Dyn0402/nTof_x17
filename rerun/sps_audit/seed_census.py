#!/usr/bin/env python3
"""seed_census.py <registry key> --out O.json

Which events of a pass-1 SPS row the committed seed fix changes, without fitting: the row's cache is rebuilt as the
pass built it (pre-fix seed: no channel excluded; same sub-runs, plateau, 3000 events, half-win 16), the per-channel
seed counts are kept, and the fixed builder's exclusion set is evaluated on them: RAW_CM_BAD (ch 372, 510) plus the
plateau's hot channels by sps_build_cache.hot_channels on the first HOT_SCAN events.  An event changes iff its pre-fix
seed (X or Y) is in that set; those events get a new window (or are dropped) and every other event is bit-identical,
except that a hot channel is now also censored in the fit (sps_reco extra_dead), which touches windows that contain it.
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                                     # noqa: E402

_orig = sbc.event_record
NONE = np.zeros(512, bool)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('key'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
    sbc.event_record = lambda *x, **k: _orig(*x, **dict(k, seed_bad=NONE))       # the pre-fix seed
    evs = {}; probe = []
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=3000 - len(evs), tan=tan, only=[row['plateau']],
                              verbose=False, hot_scan=0).get(row['plateau'], {})
        probe.append((sub, list(got.values())[:sbc.HOT_SCAN]))
        evs.update(got)
        if len(evs) >= 3000:
            break
    bad = sbc.SEED_BAD.copy(); hot_all = []
    for sub, pe in probe:
        if not pe:
            continue
        cnt = np.zeros(512)
        for e in pe:
            cnt[e['seed_ch_x']] += 1; cnt[e['seed_ch_y']] += 1
        h = sbc.hot_channels(cnt, len(pe)); bad |= h; hot_all += [int(c) for c in np.flatnonzero(h)]
    seeds = np.zeros(512, int)
    for e in evs.values():
        seeds[e['seed_ch_x']] += 1; seeds[e['seed_ch_y']] += 1
    changed = [k for k, e in evs.items() if bad[e['seed_ch_x']] or bad[e['seed_ch_y']]]
    out = dict(key=a.key, n_events=len(evs), hot=sorted(set(hot_all)), n_changed=len(changed),
               seeds_on_excluded={int(c): int(seeds[c]) for c in np.flatnonzero(bad) if seeds[c]},
               top_seeds={int(c): int(seeds[c]) for c in np.argsort(seeds)[::-1][:6]})
    json.dump(out, open(a.out, 'w'))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
