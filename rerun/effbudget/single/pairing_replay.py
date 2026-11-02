#!/usr/bin/env python3
"""pairing_replay.py [--root R] [--bundles B] [--out DIR] -- replay wft.reco.select_tracks on a pass's candidate table.

select_tracks reads only scalar candidate attributes (plausible, dchi2, rescued, t0, q_sum, q_u50, q_u90, n_*_strips)
that events.candidates.parquet stores, and the det3 bench bundles' xy_pairing has no 'prof'/'lqc' feature (the two
that need the depth profile).  So a pass's track selection can be re-run exactly on its own candidate table, with no
waveform.  The bow block (wft.reco.apply_bow) is replayed too: it re-maps a paired candidate's tangent with its
partner's coordinate, so a re-pairing moves the tangent of non-winner candidates (winners are re-mapped with the
winner pair last, so the events table does not move).

For each mode it reports, against the pass ('gate' must reproduce the file's track_id and tan_theta exactly):
  * the 552 M3-truth two-particle events of xy_pairing.py: correct / wrong / unpaired for the M3 particle;
  * every event of the two rows: events whose pairs change, n_tracks transitions, and the tangent shift of candidates.
Written for rerun/effbudget/single/PAIRING_FIX_2026-11-02.md.

  --root     the cosmic_bench tree holding Analysis_pass1/<rel>/wft/events.candidates.parquet
  --bundles  a tree holding Analysis_pass1/<rel>/wft/calib_bundle/ (default: --root)
  --events   xy_pairing_events.parquet (xy_pairing.py's output; default: results/ next to this script)
"""
import argparse, json, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from wft import reco  # noqa: E402
from wft.calib import CalibrationBundle  # noqa: E402

ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
ATTR = ('p0', 'w', 't0', 'tan_theta', 'q_sum', 'q_u50', 'q_u90', 'q_uend', 'n_sat_strips', 'n_flagged_strips',
        'n_band_strips')
MODES = ('gate', 'cost')


def fits_of(g):
    """{plane: [fit-like, in rank order]} from one event's candidate rows (rank >= 0)."""
    out = {}
    for p in ('x', 'y'):
        h = g[(g.plane == p) & (g['rank'] >= 0)].sort_values('rank')
        L = []
        for r in h.itertuples(index=False):
            f = SimpleNamespace(**{k: getattr(r, k) for k in ATTR})
            f._plausible, f._dchi2, f._rescued = bool(r.plausible), float(r.dchi2), bool(r.rescued)
            f._rank = int(r.rank)
            f.tan_file = float(r.tan_theta)
            L.append(f)
        out[p] = L
    return out


def ftst_diff_of(g):
    fx, fy = g[g.plane == 'x'].ftst, g[g.plane == 'y'].ftst
    if len(fx) and len(fy) and fx.iloc[0] >= 0 and fy.iloc[0] >= 0:
        return int(fx.iloc[0]) - int(fy.iloc[0])
    return None


def tracks(fits, pairs):
    """{(plane, rank): track_id}, as candidate_rows writes it."""
    d = {}
    for tid, (i, j, _g) in enumerate(pairs):
        d[('x', fits['x'][i]._rank)] = tid
        d[('y', fits['y'][j]._rank)] = tid
    return d


def bow_tans(cal, fits, pairs):
    """{(plane, rank): tan} as _worker_fit leaves it: own-view mapping, then each pair, then the winner pair."""
    for p in ('x', 'y'):
        for f in fits[p]:
            f.tan_theta = reco.tan_of(cal, p, f.w, f.p0)
    for i, j, _g in pairs:
        reco.apply_bow(cal, fits['x'][i], fits['y'][j])
    w = [fits[p][0] if fits[p] else None for p in ('x', 'y')]
    if w[0] is not None or w[1] is not None:
        reco.apply_bow(cal, w[0], w[1])
    return {(p, f._rank): float(f.tan_theta) for p in ('x', 'y') for f in fits[p]}


def classify(fits, pairs, r):
    """correct / wrong / unpaired for the M3 particle, as xy_pairing.py defines them, under the pairs given
    (r carries the ranks of the M3 X/Y candidates and of the brightest other candidate of each view)."""
    t = tracks(fits, pairs)
    xm, ym = t.get(('x', r.xm_rank), -1), t.get(('y', r.ym_rank), -1)
    xo, yo = t.get(('x', r.xo_rank), -1), t.get(('y', r.yo_rank), -1)
    if xm >= 0 and xm == ym:
        return 'correct'
    if (xm >= 0 and xm == yo) or (ym >= 0 and ym == xo):
        return 'wrong'
    return 'unpaired'



def bundle_dir(root, bundles, rel):
    """The bundle the row was reconstructed with: events.meta.json 'calibration' (an EOS or condor-scratch path whose
    tail is <rel of the bundle's row>/wft/calib_bundle), resolved under ``bundles``; else the row's own."""
    try:
        p = json.load(open(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.meta.json')))['calibration']
        for tag in ('/Analysis_pass1/', '/Analysis/'):
            if tag in p:
                p = p.split(tag, 1)[1]
                break
        d = os.path.join(bundles, 'Analysis_pass1', p)
        if os.path.exists(os.path.join(d, 'bundle.json')):
            return d
    except (OSError, KeyError, ValueError):
        pass
    return os.path.join(bundles, 'Analysis_pass1', rel, 'wft/calib_bundle')


def discover(root):
    """{rel: rel} of every Analysis_pass1/<run>/<sub>/<det> with a candidates table (sps excluded)."""
    base = os.path.join(root, 'Analysis_pass1')
    out = {}
    for run in sorted(os.listdir(base)):
        if run == 'sps' or not os.path.isdir(os.path.join(base, run)):
            continue
        for sub in sorted(os.listdir(os.path.join(base, run))):
            for det in sorted(os.listdir(os.path.join(base, run, sub))):
                rel = f'{run}/{sub}/{det}'
                if os.path.exists(os.path.join(base, rel, 'wft/events.candidates.parquet')):
                    out[rel] = rel
    return out


def replay_row(job):
    """One row: replay every event in every mode; optionally write the refreshed tables.  -> (rows, row summary)."""
    key, rel, root, bundles, modes, truth_rows, tables = job
    bdir = bundle_dir(root, bundles, rel)
    cal = CalibrationBundle.load(bdir)
    pairing = cal.xy_pairing
    reco.TC_TOL_NS = reco.tc_tol(pairing)
    C = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
    C = C[C['rank'] >= 0]
    Ek = truth_rows.set_index('event_id') if truth_rows is not None and len(truth_rows) else None
    upd, ntr, evrows = {}, {}, []
    last = modes[-1]
    # an event is (acq_tag, event_id): event ids repeat across a row's acquisitions on some rows (det3_test short_run)
    if 'acq_tag' not in C:
        C = C.assign(acq_tag='')
    dup_ids = set(C.drop_duplicates(['acq_tag', 'event_id']).event_id.value_counts().loc[lambda s: s > 1].index)
    for (acq, e), g in C.groupby(['acq_tag', 'event_id'], sort=False):
        fits = fits_of(g)
        fd = ftst_diff_of(g)
        prod = {(r.plane, int(r.rank)): int(r.track_id) for r in g.itertuples() if r.track_id >= 0}
        P = {m: reco.select_tracks(fits, fd, cal, max_tracks=3, pairing=pairing, mode=m) for m in modes}
        T = {m: bow_tans(cal, fits, P[m]) for m in modes}
        tan_file = {(p, f._rank): f.tan_file for p in ('x', 'y') for f in fits[p]}
        row = dict(row=key, rel=rel, acq_tag=acq, event_id=int(e), id_unique=int(e) not in dup_ids,
                   nx=len(fits['x']), ny=len(fits['y']),
                   reproduced=tracks(fits, P['gate']) == prod,
                   tan_reproduced=all(abs(T['gate'][k] - v) < 1e-9 for k, v in tan_file.items() if np.isfinite(v)))
        truth = Ek.loc[int(e)] if Ek is not None and int(e) in Ek.index and int(e) not in dup_ids else None
        for m in modes:
            row[f'ntr_{m}'] = int(sum(gg for *_x, gg in P[m]))
            row[f'pairs_{m}'] = str([(int(i), int(j), bool(gg)) for i, j, gg in P[m]])
            d = [abs(T[m][k] - T['gate'][k]) for k in tan_file if np.isfinite(T[m][k]) and np.isfinite(T['gate'][k])]
            row[f'dtan_max_{m}'] = float(max(d)) if d else 0.0
            row[f'dtan_winner_{m}'] = float(max([abs(T[m][(p, 0)] - T['gate'][(p, 0)]) for p in ('x', 'y')
                                                 if (p, 0) in T[m] and np.isfinite(T[m][(p, 0)])] or [0.0]))
            row[f'truth_{m}'] = classify(fits, P[m], truth) if truth is not None else None
        row['prod_file'] = truth['prod'] if truth is not None else None
        evrows.append(row)
        if tables and P[last] != P['gate']:
            tid = {}
            for t, (i, j, gg) in enumerate(P[last]):
                tid[('x', fits['x'][i]._rank)] = (t, gg)
                tid[('y', fits['y'][j]._rank)] = (t, gg)
            # the tangent: the file's value plus the shift the re-pairing makes in the replay (bow re-map with the new
            # partner), so the ~0.4 % of events whose bow replay does not reproduce the file keep the file's values
            # wherever the pairing does not move them
            for k, tn in T[last].items():
                shift = tn - T['gate'][k] if np.isfinite(tn) and np.isfinite(T['gate'][k]) else 0.0
                upd[(acq, int(e)) + k] = tid.get(k, (-1, False)) + (tan_file[k] + shift,)
            ntr[(acq, int(e))] = row[f'ntr_{last}']
    summ = dict(row=key, rel=rel, bundle=os.path.relpath(bdir, bundles), select_in_bundle=pairing.get('select'),
                pairing_features=pairing.get('features'), events=len(evrows))
    if tables:
        d = os.path.join(tables, 'Analysis_pass1', rel, 'wft')
        os.makedirs(d, exist_ok=True)
        full = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
        acqs = full.acq_tag if 'acq_tag' in full else pd.Series('', index=full.index)
        keys = list(zip(acqs, full.event_id.astype(int), full.plane, full['rank'].astype(int)))
        hit = np.array([k in upd for k in keys], dtype=bool)
        for col, n in (('track_id', 0), ('track_gated', 1), ('tan_theta', 2)):
            v = full[col].to_numpy().copy()
            if hit.any():
                v[hit] = [upd[k][n] for k, h in zip(keys, hit) if h]
            full[col] = v
        full['theta_deg'] = np.where(hit, np.degrees(np.arctan(full.tan_theta)), full.theta_deg)
        full.to_parquet(os.path.join(d, 'events.candidates.parquet'))
        ev = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.parquet'))
        if 'n_tracks' in ev:
            ea = ev.acq_tag if 'acq_tag' in ev else pd.Series('', index=ev.index)
            new = np.array([ntr.get((a_, int(e_)), n_) for a_, e_, n_ in zip(ea, ev.event_id, ev.n_tracks)])
            summ['events_table_n_tracks_changed'] = int((new != ev.n_tracks.to_numpy()).sum())
            ev['n_tracks'] = new.astype(ev['n_tracks'].dtype)
        ev.to_parquet(os.path.join(d, 'events.parquet'))
        summ['candidate_rows_changed'] = int(hit.sum())
    return evrows, summ


def main():
    from concurrent.futures import ProcessPoolExecutor
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--bundles', default=None)
    ap.add_argument('--events', default=os.path.join(here, 'results', 'xy_pairing_events.parquet'))
    ap.add_argument('--out', default=os.path.join(here, 'results'))
    ap.add_argument('--modes', default=','.join(MODES))
    ap.add_argument('--all-rows', action='store_true', help='every bench row under <root>/Analysis_pass1 (not sps), '
                                                            'each with the bundle its events.meta.json names')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--tables', default=None, help="write each row's candidates table and events table as the LAST mode "
                                                    "would have written them (track_id, track_gated, tan_theta, "
                                                    "theta_deg; n_tracks; only events whose pairs change) under "
                                                    "<tables>/Analysis_pass1/<rel>/wft/")
    a = ap.parse_args()
    modes = a.modes.split(',')
    bund = a.bundles or a.root
    E = pd.read_parquet(a.events) if a.events and os.path.exists(a.events) else None
    rows = discover(a.root) if a.all_rows else ROWS
    rel2key = {v: k for k, v in ROWS.items()}
    jobs = []
    for key, rel in rows.items():
        k = rel2key.get(rel, key)
        tr = E[E.row == k] if E is not None and k in set(E.row) else None
        jobs.append((k, rel, a.root, bund, modes, tr, a.tables))
    evrows, summs = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r, s in ex.map(replay_row, jobs):
            evrows += r
            summs.append(s)
    D = pd.DataFrame(evrows)
    res = dict(rows=summs)
    T = D[D.prod_file.notna()]
    res['replay'] = dict(rows=len(summs), events=int(len(D)), track_id_reproduced=float(D.reproduced.mean()),
                         tan_reproduced=float(D.tan_reproduced.mean()),
                         truth_gate_matches_xy_pairing=float((T.truth_gate == T.prod_file).mean()) if len(T) else None)
    for m in modes:
        k = T[f'truth_{m}']
        ch = D[f'pairs_{m}'] != D['pairs_gate']
        single = (D.nx <= 1) & (D.ny <= 1)
        moved = D[f'dtan_max_{m}'] > 1e-9
        pc = float((k == 'correct').mean()) if len(T) else float('nan')
        res[m] = dict(
            truth=dict(n=int(len(T)), **{c: int((k == c).sum()) for c in ('correct', 'wrong', 'unpaired')},
                       frac_correct=pc, frac_correct_err=float(np.sqrt(pc * (1 - pc) / len(T))) if len(T) else None),
            events_pairs_changed=int(ch.sum()), events_pairs_changed_frac=float(ch.mean()),
            changed_single_candidate_events=int((ch & single).sum()),
            ntracks_changed=int((D[f'ntr_{m}'] != D['ntr_gate']).sum()),
            ntracks_transitions={f'{o}->{n}': int(c) for (o, n), c in
                                 D[D[f'ntr_{m}'] != D['ntr_gate']].groupby(['ntr_gate', f'ntr_{m}']).size().items()},
            ntracks_eq1=dict(gate=int((D.ntr_gate == 1).sum()), new=int((D[f'ntr_{m}'] == 1).sum())),
            winner_tan_moved=int((D[f'dtan_winner_{m}'] > 1e-9).sum()),
            candidate_tan_moved_events=int(moved.sum()),
            candidate_dtan_p50_p99_max_of_moved=([float(np.percentile(D[f'dtan_max_{m}'][moved], q))
                                                  for q in (50, 99, 100)] if moved.any() else None))
    if len(T):
        res['truth_transitions'] = {m: {f'{o}->{n}': int(c) for (o, n), c in
                                        T.groupby(['truth_gate', f'truth_{m}']).size().items()}
                                    for m in modes if m != 'gate'}
    os.makedirs(a.out, exist_ok=True)
    D.to_parquet(os.path.join(a.out, 'pairing_replay_events.parquet'))
    json.dump(res, open(os.path.join(a.out, 'pairing_replay.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1, default=str))


if __name__ == '__main__':
    main()
