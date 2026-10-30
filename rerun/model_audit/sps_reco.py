#!/usr/bin/env python3
"""sps_reco.py <registry key> --out DIR [--per-plateau N] [--jobs 8] [--registry rerun/sps_registry.csv]
             [--bundle <bench det4 calib_bundle>] [--tan-mode mount|zero]

The SPS det4 reconstruction stage (RERUN_PLAN §12 B3): one registry row (dataset + plateau) -> <out>/events.parquet in
the wft table layout (event_id, x_*/y_* PlaneFit columns, n_tracks) + events.meta.json, from the decoded FEU 03 files
on EOS.  The bench det4 bundle is adapted to the row: v = the row's v_span (CF4 ladder) or the bundle v, K = the row's
K, 64/32 samples from the dataset, the channel layer measured at H4 in place of the bench one (sps_mask.py: the bench
lists are bench channel numbers, a different strip at H4; --mask bench carries them by position, none drops them),
exact-zero samples fitted as measured zeros on ZS rows (suppressed = below threshold) and censored as ``missing`` on
RAW rows (the FEU's dropped sample groups; ws_sps 2026-10-27), the wft algorithm otherwise as is (pivot start, column guard, ... are part of it since
2026-10-27).  Seeds are the chamber's own brightest cluster per view (no
uRWELL); the reference columns ref_x/ref_y carry the own charge centroid and tan_mount the mount angle.
"""
import argparse, csv, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, HERE]
#: the pass's det4 bundle (calib phase CAL4, Analysis_pass1; PASS_PLAN E4)
BUNDLE = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle'
_EV = None
_RAW = False


def _init(cache, bundle, k, v, nsamp, mask='h4', raw=False, extra_dead=None):
    global _EV, _RAW
    from wft.calib import CalibrationBundle
    from wft import model as wm
    import sps_mask
    _EV = pickle.load(open(cache, 'rb'))
    _RAW = bool(raw)
    cal = CalibrationBundle.load(bundle)
    sps_mask.apply(cal, mask)
    # the row's hot (oscillating) channels found by the cache builder's seed probe: censored like the dead ones
    for p, chs in (extra_dead or {}).items():
        cal.dead[p] = sorted(set(cal.dead.get(p, [])) | {int(c) for c in chs})
    if v:
        cal.v_drift = float(v)
    cal.t0_abs, cal.t0_prior_sigma = {}, 0.0        # the beam trigger is asynchronous to the DREAM clock
    # the bench bow block, v map and rigid tilt w0 are properties of the bench mounting (FB/FA/F9): never at H4, the
    # same rule as ntof_tracking/wft_beam.py (kw stays)
    cal.provenance = dict(cal.provenance or {}, dropped_bow=bool(getattr(cal, 'bow', None)),
                          dropped_v_map=bool(getattr(cal, 'v_map', None)), dropped_w0=dict(cal.w0 or {}))
    cal.bow, cal.v_map, cal.w0 = {}, {}, {}
    wm.use_calibration(cal); wm.set_depth_bins(k); wm.set_nsamp(nsamp)


def one(eid):
    from wft import model as wm, reco as wr
    ev = _EV[eid]; fits = {}
    for plane in 'xy':
        P = ev.get(plane)
        if P is None:
            fits[plane] = None; continue
        if _RAW:                     # an exact 0 on a RAW row is a sample the FEU dropped, not a measurement (see main)
            P = dict(P, missing=np.asarray(P['W']) == 0.0)
        try:
            fits[plane] = wr.fit_plane(P, plane, wm.CAL)
        except Exception:
            fits[plane] = None
    row = wr.row_from_fits(int(eid), fits, n_hits=0, spark=False)
    row['ref_x'], row['ref_y'] = ev.get('ref_mesh_x', np.nan), ev.get('ref_mesh_y', np.nan)
    row['tan_mount'], row['t_wall'] = ev.get('tan_x', np.nan), ev.get('t_wall', np.nan)
    # the DREAM trigger counter and the acquisition file (TODO B6/GC: join key with the run, subrun, event id)
    row['timestamp'], row['acq_tag'] = int(ev.get('timestamp', -1)), str(ev.get('file', ''))
    row['n_tracks'] = int(fits['x'] is not None and fits['y'] is not None)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--out', required=True); ap.add_argument('--registry', default=os.path.join(REPO, 'rerun', 'sps_registry.csv'))
    ap.add_argument('--bundle', default=BUNDLE); ap.add_argument('--per-plateau', type=int, default=3000); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--mask', default='h4', choices=('h4', 'h4_stripes', 'bench', 'none'), help='channel layer at H4 (sps_mask.py)')
    ap.add_argument('--half-win', type=int, default=16); ap.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(a.registry))}[a.key]
    import pandas as pd
    import sps_build_cache as sbc
    sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'analysis')); import datasets
    D = datasets.get(row['dataset']); nsamp = int(D['n_samples'])
    # mount column: 'flat', '25.64 deg' or '15.465 deg' (run_61 rot15 ladder); was tan(25.64) for every rotated row
    tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
    # an exact 0 in the cache means 'not read out' (sps_build_cache._baseline moves real zeros off 0).  On ZS rows that is
    # a suppressed sample, below threshold: fitted as a measured zero (ws_sps: closest to the RAW answer).  On the RAW
    # rows (run_71) it is the FEU's dropped sample groups (~23 % of samples): censored as missing.  A property of the data,
    # carried by each window (one()), not a switch.
    raw = bool(int(row['raw']))
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=a.per_plateau - len(evs), half_win=a.half_win, tan=tan,
                              only=[row['plateau']], verbose=False).get(row['plateau'], {})
        evs.update(got)
        print(f'[sps] {sub}: {len(got)} events in {row["plateau"]} ({len(evs)} total)', flush=True)
        if len(evs) >= a.per_plateau:
            break
    hot = sorted({c for (r_, s_, L), cs in sbc.HOT_LAST.items() if r_ == row['run'] and L == row['plateau'] for c in cs})
    from det4_sps_map import VIEW
    extra_dead = {p: [c for c in hot if VIEW[c] == p] for p in ('x', 'y')}
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.work, f'sps_cache_{a.key}.pkl'); pickle.dump(evs, open(cache, 'wb'), protocol=4)
    v = float(row['v_span_um_ns']) if row['v_span_um_ns'] else None
    # K fallback for rows without v_span: the column at v (or 12 um/ns) plus the registry's +15 margin (PASS_PLAN E4)
    k = int(row['K']) if row['K'] else int(np.ceil(28800.0 / (v or 12.0) / 60.0)) + 15
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(cache, a.bundle, k, v, nsamp, a.mask, raw, extra_dead)) as pool:
        rows = list(pool.map(one, sorted(evs), chunksize=4))
    from wft import reco as wr
    df = pd.DataFrame(rows).sort_values('event_id').reset_index(drop=True)
    df, q_med = wr.quality_flags(df)          # chi2/dof flag, CHI2DOF_REL x the table median per view (GE)
    df.to_parquet(os.path.join(a.out, 'events.parquet'), index=False)
    meta = dict(n_events=len(df), registry_row=row, quality_rule=wr.quality_rule(q_med), bundle=a.bundle, v_drift=v, n_depth_bins=k, n_samples=nsamp, zeros='missing (RAW)' if raw else 'measured (ZS)',
                channel_layer=a.mask, hot_seed_channels=hot, seeding='own brightest strip +- half_win (not RAW_CM_BAD, not the hot channels)', half_win=a.half_win,
                env={k_: v_ for k_, v_ in os.environ.items() if k_.startswith('WFT_')},
                diagnostic_overrides={k_: str(v_) for k_, v_ in __import__('wft.diag').diag.ACTIVE.items()},
                code=os.environ.get('RERUN_CODE_COMMIT', '?'))
    json.dump(meta, open(os.path.join(a.out, 'events.meta.json'), 'w'), indent=1, default=str)
    os.remove(cache)
    print(f'[sps] wrote {a.out}/events.parquet ({len(df)} events)')


if __name__ == '__main__':
    main()
