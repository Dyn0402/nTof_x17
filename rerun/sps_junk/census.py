#!/usr/bin/env python3
"""census.py <registry key> --out O.json [--pass-dir D]

Which SPS rows the coupled-channel rule (sps_build_cache.COUPLE_*, 2026-10-31) and the own-events hot probe change,
without fitting: the row's cache is rebuilt with the new builder exactly as sps_reco builds it (same sub-runs, plateau,
3000 events, half-win 16) and compared with the pass-1 product (events.parquet + meta):
  - the event set (event_id);
  - the hot channels (meta hot_seed_channels) -- a new one is censored in every window that holds it;
  - per event the own-centroid ref (ref_x/ref_y): it moves iff the seed or a blanked strip changed;
  - the events with a coupled channel censored (censor_x/censor_y non-empty) -- their fit changes.
A row needs its sps node re-run iff any of these differ.  Also the junk metric of rerun/sps_audit (the own centroid
ref_y within 154-160.5 mm, and within 248-253 mm around ch 510) before and after.
"""
import argparse, csv, json, os, sys
import numpy as np

REPO = os.environ['WFT_REPO']
DEV = os.environ.get('SPS_DEV')                     # a directory holding the builder under test (first on the path)
sys.path[:0] = ([DEV] if DEV else []) + [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                         # noqa: E402

PASS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps'
ZONES = {'z372': (154.0, 160.5), 'z510': (248.0, 253.0)}


def zone_frac(y):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    return {k: float(((y >= lo) & (y < hi)).mean()) if len(y) else np.nan for k, (lo, hi) in ZONES.items()}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument('key'); ap.add_argument('--out', required=True)
    ap.add_argument('--pass-dir', default=PASS)
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=3000 - len(evs), tan=tan, only=[row['plateau']],
                              verbose=False).get(row['plateau'], {})
        evs.update(got)
        if len(evs) >= 3000:
            break
    hot = sorted({c for (r_, s_, L), cs in sbc.HOT_LAST.items() if r_ == row['run'] and L == row['plateau'] for c in cs})
    coupled = {s_: {str(o): cs for o, cs in d.items()} for (r_, s_, L), d in sbc.COUPLE_LAST.items()
               if r_ == row['run'] and L == row['plateau'] and d}
    t = pd.read_parquet(os.path.join(a.pass_dir, a.key, 'events.parquet'), columns=['event_id', 'ref_x', 'ref_y'])
    meta = json.load(open(os.path.join(a.pass_dir, a.key, 'events.meta.json')))
    old = t.set_index('event_id')
    ids_new = np.array(sorted(evs), np.int64); ids_old = old.index.to_numpy(np.int64)
    common = np.intersect1d(ids_new, ids_old)
    rx = np.array([evs[i]['ref_mesh_x'] for i in common]); ry = np.array([evs[i]['ref_mesh_y'] for i in common])
    moved = (np.abs(rx - old.loc[common, 'ref_x'].to_numpy()) > 1e-3) | (np.abs(ry - old.loc[common, 'ref_y'].to_numpy()) > 1e-3)
    cens = np.array([bool(evs[i].get('censor_x') or evs[i].get('censor_y')) for i in ids_new])
    hot_old = sorted(meta.get('hot_seed_channels') or [])
    n_only_new, n_only_old = int(len(np.setdiff1d(ids_new, ids_old))), int(len(np.setdiff1d(ids_old, ids_new)))
    out = dict(key=a.key, n_new=len(ids_new), n_old=len(ids_old), only_new=n_only_new, only_old=n_only_old,
               ref_moved=int(moved.sum()), censored_events=int(cens.sum()), hot_old=hot_old, hot_new=hot,
               coupled=coupled, junk_old=zone_frac(old.ref_y), junk_new=zone_frac([evs[i]['ref_mesh_y'] for i in ids_new]))
    out['changes'] = bool(n_only_new or n_only_old or moved.any() or cens.any() or hot != hot_old)
    json.dump(out, open(a.out, 'w'))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
