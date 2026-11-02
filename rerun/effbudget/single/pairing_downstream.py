#!/usr/bin/env python3
"""pairing_downstream.py --orig R --new N --timing-rows DIR [--out results/pairing_downstream.json]
-- the Paper I products that cut on n_tracks, on the pass-1 tables (R) and on the pairing-refreshed ones (N).

  paper1_timing/matched_rows.py   per row: the M3-matched single-track sample (both views ok, n_tracks == 1) joined
                                  with timing_row.py's per-trigger times; n and the widths (sigma_xy, sigma_evt,
                                  sigma_common), no bootstrap.  timing rows: x17rerun/paper1_timing/rows/<key>.npz.
  paper1_timing/angle.py          the golden rows' angle bins (same cut), run as is on both trees.
  paper1_floor/analyze.py         its 'topo' sample cuts n_candidates <= 1 AND n_tracks <= 1 in the view; the cost mode
                                  is the gate mode whenever a view has <= 1 candidate, so that sample cannot move:
                                  checked here as "events with n_tracks changed and n_candidates <= 1 in a view" = 0.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_timing'))
import matched_rows as MR  # noqa: E402

GOLDEN = ['o22_long_det2', 'c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'sat_det3',
          'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3', 'g_det4', 'g_det6_long',
          'g_det7_long']


def matched(root, md, npz):
    ev = os.path.join(root, 'Analysis_pass1', md['run'], md['subrun'], md['det'], 'wft', 'events.parquet')
    if not (os.path.exists(npz) and os.path.exists(ev)):
        return None, ev
    z = np.load(npz); tags = list(z['acq_tags'])
    T = pd.DataFrame(dict(acq_tag=[tags[i] for i in z['acq']], event_id=z['eid'], tx=z['tx'], ty=z['ty'],
                          ex=z['ex'], ey=z['ey']))
    E = pd.read_parquet(ev, columns=['acq_tag', 'event_id', 'x_ok', 'y_ok', 'n_tracks'])
    m = T.merge(E, on=['acq_tag', 'event_id'])
    return m[m.x_ok & m.y_ok & (m.n_tracks == 1)], ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', required=True); ap.add_argument('--new', required=True)
    ap.add_argument('--timing-rows', required=True)
    ap.add_argument('--jobs', default=os.path.join(REPO, 'rerun', 'paper1_timing', 'jobs.txt'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'pairing_downstream.json'))
    a = ap.parse_args()
    rows, pairs = [], {'orig': [], 'new': []}
    for line in open(a.jobs):
        _c, _b, stem, *meta = line.split()
        md = dict(kv.split('=', 1) for kv in meta)
        npz = os.path.join(a.timing_rows, os.path.basename(stem) + '.npz')
        mo, evo = matched(a.orig, md, npz)
        mn, evn = matched(a.new, md, npz)
        if mo is None or mn is None:
            continue
        r = dict(key=md['key'], det=md['det'], n_orig=int(len(mo)), n_new=int(len(mn)),
                 n_lost=int(len(mo.merge(mn[['acq_tag', 'event_id']], how='left', indicator=True)
                               .query('_merge == "left_only"'))),
                 n_gained=int(len(mn.merge(mo[['acq_tag', 'event_id']], how='left', indicator=True)
                                 .query('_merge == "left_only"'))))
        if len(mo) >= 150:
            wo = MR.widths(*[mo[c].to_numpy(float) for c in ('tx', 'ty', 'ex', 'ey')])
            wn = MR.widths(*[mn[c].to_numpy(float) for c in ('tx', 'ty', 'ex', 'ey')])
            for k in ('sigma_xy', 'sigma_evt', 'sigma_common'):
                r[f'{k}_orig'], r[f'{k}_new'] = wo[k], wn[k]
                r[f'd_{k}'] = wn[k] - wo[k]
        rows.append(r)
        if md['key'] in GOLDEN:
            pairs['orig'].append(f"{md['key']}={npz},{evo}")
            pairs['new'].append(f"{md['key']}={npz},{evn}")
    R = pd.DataFrame(rows)
    out = dict(timing_rows=dict(n_rows=int(len(R)), rows_with_change=int(((R.n_lost + R.n_gained) > 0).sum()),
                                events_lost=int(R.n_lost.sum()), events_gained=int(R.n_gained.sum()),
                                events_total=int(R.n_orig.sum()),
                                max_abs_d_sigma_evt_ns=float(R.d_sigma_evt.abs().max()),
                                max_abs_d_sigma_xy_ns=float(R.d_sigma_xy.abs().max()),
                                changed=R[(R.n_lost + R.n_gained) > 0].to_dict('records')))
    ang = {}
    for nm in ('orig', 'new'):
        f = os.path.join(os.path.dirname(a.out), f'_angle_{nm}.json')
        subprocess.run([sys.executable, os.path.join(REPO, 'rerun', 'paper1_timing', 'angle.py'), '--pairs', *pairs[nm],
                        '--out', f], check=True, capture_output=True)
        ang[nm] = json.load(open(f)); os.remove(f)
    diffs = []

    def walk(x, y, path=''):
        if isinstance(x, dict):
            for k in x:
                walk(x[k], (y or {}).get(k), f'{path}/{k}')
        elif isinstance(x, list):
            for i, (u, v) in enumerate(zip(x, y or [])):
                walk(u, v, f'{path}[{i}]')
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not (x == y or (np.isnan(x) and np.isnan(y))):
                diffs.append(dict(path=path, orig=x, new=y))
    walk(ang['orig'], ang['new'])
    out['angle_changes'] = diffs
    # paper1_floor topo: n_tracks changes only where both views have >= 2 candidates
    bad = 0
    for dirpath, _d, files in os.walk(os.path.join(a.new, 'Analysis_pass1')):
        if 'events.parquet' not in files:
            continue
        rel = os.path.relpath(dirpath, os.path.join(a.new, 'Analysis_pass1'))
        en = pd.read_parquet(os.path.join(dirpath, 'events.parquet'))
        eo = pd.read_parquet(os.path.join(a.orig, 'Analysis_pass1', rel, 'events.parquet'))
        ch = en.n_tracks.to_numpy() != eo.n_tracks.to_numpy()
        bad += int((ch & ((eo.x_n_candidates.to_numpy() <= 1) | (eo.y_n_candidates.to_numpy() <= 1))).sum())
    out['floor_topo_events_moved'] = bad
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('angle',)}, indent=1, default=float)[:3000])


if __name__ == '__main__':
    main()
