#!/usr/bin/env python3
"""pairing_eff_check.py --root R --cost-root C -- does the 'cost' pairing move the det3 efficiency inputs?

The efficiency chain (A1, effbudget/budget.py, single_track.py, radius.py) reads no track_id, track_gated or n_tracks:
it reads the ranked candidates' p0 and tan_theta.  A re-pairing moves only the tangent of a NON-winner candidate (the
bow block re-maps a paired candidate with its partner's coordinate); p0 never moves.  This compares, between the
pass-1 candidates table (R) and the one pairing_replay.py --tables wrote for mode 'cost' (C):
  * budget.py's trigger-track rule (per view, the most vertical ranked candidate): events whose pick changes;
  * single_track.py's second-particle test (second_3d / second_paired / attached, the only place it reads tan):
    events whose verdict changes, on every event with >= 2 candidates in a view.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_track as st  # noqa: E402

ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--cost-root', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--all-rows', action='store_true', help='every row of --cost-root (the refreshed tables)')
    a = ap.parse_args()
    res = {}
    rows = dict(ROWS)
    if a.all_rows:
        base = os.path.join(a.cost_root, 'Analysis_pass1')
        rows = {os.path.relpath(d, base)[:-4]: os.path.relpath(d, base)[:-4] for d, _s, f in os.walk(base)
                if 'events.candidates.parquet' in f}
    for key, rel in sorted(rows.items()):
        det = rel.split('/')[-1].replace('mx17_', 'det')
        rp = os.path.join(a.root, 'x17rerun/paper1_eff/budget', rel, f'rays_{det}.parquet')
        if not os.path.exists(rp):
            continue
        tabs = {}
        for nm, r in (('pass1', a.root), ('cost', a.cost_root)):
            C = pd.read_parquet(os.path.join(r, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
            C = C[C['rank'] >= 0].copy()
            if 'acq_tag' in C:      # event ids repeated across acquisitions (det3_test): cannot be matched to rays
                u = C.drop_duplicates(['acq_tag', 'event_id']).event_id.value_counts()
                C = C[C.event_id.isin(u.index[u == 1])]
            tabs[nm] = C
        pick = {}
        for nm, C in tabs.items():
            C['atan'] = C.tan_theta.abs()
            pick[nm] = C.sort_values(['atan', 'rank']).groupby(['event_id', 'plane'])['rank'].first()
        moved = tabs['pass1'].set_index(['event_id', 'plane', 'rank']).tan_theta.sub(
            tabs['cost'].set_index(['event_id', 'plane', 'rank']).tan_theta).abs()
        B = pd.read_parquet(rp)
        B = B.drop_duplicates('event_id').set_index('event_id')
        # does a changed rule pick change the rule's detection (pick within 5 mm of M3 in both views)?
        near = {}
        for nm, C in tabs.items():
            P = C.set_index(['event_id', 'plane', 'rank']).p0
            pk = pick[nm].reset_index()
            pk['p0'] = P.reindex(list(zip(pk.event_id, pk.plane, pk['rank']))).to_numpy()
            pk['mesh'] = np.where(pk.plane == 'x', B.mesh_x.reindex(pk.event_id), B.mesh_y.reindex(pk.event_id))
            pk['ok'] = (pk.p0 - pk.mesh).abs() < 5.0
            near[nm] = pk.groupby('event_id').ok.all()
        rule_det_changed = int((near['pass1'] != near['cost'].reindex(near['pass1'].index)).sum())
        n = tabs['pass1'].groupby(['event_id', 'plane']).size().unstack(fill_value=0)
        ev = [e for e in n.index[(n.x >= 2) | (n.y >= 2)] if e in B.index]
        sp = {nm: st.second_particle(r, rel, B, ev)[0] for nm, r in (('pass1', a.root), ('cost', a.cost_root))}
        cols = ['second_3d', 'second_paired', 'attached', 'one_view_only']
        dif = (sp['pass1'][cols] != sp['cost'][cols].reindex(sp['pass1'].index)).any(axis=1)
        res[key] = dict(candidates=int(len(moved)), candidates_tan_moved=int((moved > 1e-12).sum()),
                        rank0_tan_moved=int((moved[moved.index.get_level_values('rank') == 0] > 1e-12).sum()),
                        rule_pick_changed=int((pick['pass1'] != pick['cost'].reindex(pick['pass1'].index)).sum()),
                        rule_views=int(len(pick['pass1'])), rule_detection_changed=rule_det_changed,
                        second_particle_events=int(len(ev)), second_particle_verdict_changed=int(dif.sum()),
                        changed_events=[int(e) for e in sp['pass1'].index[dif]][:20])
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        open(a.out, 'w').write(s)


if __name__ == '__main__':
    main()
