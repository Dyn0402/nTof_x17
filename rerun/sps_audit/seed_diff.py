#!/usr/bin/env python3
"""seed_diff.py <registry key> --out O.json

Which events of a pass-1 SPS row change under the seed fix (sps_build_cache.SEED_BAD), without fitting anything: the
row's cache is rebuilt exactly as sps_reco.main does (same sub-runs, plateau, per-plateau 3000, half-win 16), gated on
the PRE-FIX record so the event list is the pass table's, and each event is flagged when the fixed builder seeds a
different strip in a view or drops the event.  An event that is flagged gets a different window and fit; the others
are bit-identical.  Writes the counts per view and the flagged event keys.
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                                     # noqa: E402

_orig = sbc.event_record
NONE = np.zeros(512, bool)


def _both(ch, sm, am, raw, nsmp, half_win=16, qmin=300.0, tan=0.0, seed_bad=None):
    old = _orig(ch, sm, am, raw, nsmp, half_win, qmin, tan, seed_bad=NONE)
    if old is None:
        return None
    new = _orig(ch, sm, am, raw, nsmp, half_win, qmin, tan)
    old['changed_x'] = new is None or new['seed_ch_x'] != old['seed_ch_x']
    old['changed_y'] = new is None or new['seed_ch_y'] != old['seed_ch_y']
    old['dropped'] = new is None
    return old


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('key'); ap.add_argument('--out', required=True)
    ap.add_argument('--per-plateau', type=int, default=3000)
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
    sbc.event_record = _both
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=a.per_plateau - len(evs), tan=tan,
                              only=[row['plateau']], verbose=False).get(row['plateau'], {})
        evs.update(got)
        if len(evs) >= a.per_plateau:
            break
    ch_x = sorted(k for k, r in evs.items() if r['changed_x']); ch_y = sorted(k for k, r in evs.items() if r['changed_y'])
    old_y = {}
    for k in ch_y:
        old_y[int(evs[k]['seed_ch_y'])] = old_y.get(int(evs[k]['seed_ch_y']), 0) + 1
    out = dict(key=a.key, n_events=len(evs), n_changed_x=len(ch_x), n_changed_y=len(ch_y),
               n_dropped=int(sum(r['dropped'] for r in evs.values())), old_seed_y_of_changed=old_y,
               changed_keys=sorted(set(ch_x) | set(ch_y)))
    json.dump(out, open(a.out, 'w'))
    print(f'{a.key}: {len(evs)} events, seed changed X {len(ch_x)}, Y {len(ch_y)} (old Y seeds {old_y}), dropped {out["n_dropped"]}')


if __name__ == '__main__':
    main()
