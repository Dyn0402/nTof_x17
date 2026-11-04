#!/usr/bin/env python3
"""q7lib.py -- Q7 + tan_max scan (NTOF_PASS_PLAN_2026-11-02 §6; NOTE_PAIRING_FIX_2026-11-02 §3-4): one reconstruction,
every configuration.

A configuration is (tan_max, split_corrob, select):
  tan_max       raw |tan| limit inside ``plausible`` (wft.reco._candidate_score / fit_plane_two); inf = 'colwin', i.e.
                plausible = the column-end window only (the acceptance then lives at stage 3, --tan-max-true)
  split_corrob  'count' (n_TOF today) | 'charge' (the bench, MERGED_PROJECTION_2026-11-02)
  select        'gate' (n_TOF today) | 'cost' (the bench, PAIRING_FIX_2026-11-02)

Why one pass is enough: the per-candidate one-track fit (fit_plane) does not read tan_max, the pairing mode or the split
mode.  Those enter only through the ``_plausible`` flag, the per-plane ranking (plausible, dchi2), the joint two-track
stage (triggers, guards, revert) and select_tracks.  So :func:`fit_multi` fits every candidate window once, then, per
configuration, re-derives ``_plausible`` and the ranking exactly as fit_plane_candidates would, re-runs the two-track
stage with the configuration's globals, and the selection.  ``check`` verifies this against wft.reco._worker_fit on
bundle copies that carry the configuration (bit-identical candidate tables).

Nothing in wft/ is edited: the configuration is applied through the bundle object the worker holds (``cal.tan_max``)
and the documented A/B globals (``SPLIT_CORROB_MODE``; select_tracks' ``mode`` argument).
"""
import copy
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)

INF = 1.0e9
TMS = (0.5, 0.6, 0.7, 0.8, 1.0, INF)
SPLITS = ('count', 'charge')
SELECTS = ('gate', 'cost')
DATA = os.environ.get('Q7_DATA', os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'ntof_pass1', 'q7'))
RUNS = os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'ntof_pass1', 'runs')
BUNDLES = os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'ntof_pass1', 'bundles')
HITS_DIR = 'combined_hits_root_rr1_cm0'     # Q4 decision: rr1 hits from the fa80c06 decodes
MIN_STRIPS = 3                              # ntof_reco.MIN_STRIPS
FEUS = dict(A=(3, 4), B=(5, 6), C=(7, 8), D=(1, 2))


def tm_label(tm):
    return 'colwin' if tm >= INF else f'tm{tm:g}'


def cfg_name(tm, split, sel):
    return f'{tm_label(tm)}_{split}_{sel}'


def all_configs(tms=TMS, splits=SPLITS, selects=SELECTS):
    return [(tm, sp, se) for tm in tms for sp in splits for se in selects]


BASELINE = cfg_name(0.6, 'count', 'gate')


# --------------------------------------------------------------------------- the worker
_OPTS = {}


_FP2 = {'orig': None, 'cache': {}}


def _fit_plane_two_memo(P, plane, cal, parent, hyper=None, t0_hints=(), f_thresh=None, probe=None, t0_mode=None):
    """wft.reco.fit_plane_two, memoised within one event across configurations.  The joint fit itself does not read
    tan_max or the threshold; only the children's plausibility, the both_plausible guard, guards_ok, accepted and
    threshold do, and those are re-derived here exactly as fit_plane_two derives them."""
    from wft import reco
    key = (plane, id(P), float(parent.p0), float(parent.w), float(parent.t0),
           tuple(float(h) for h in (t0_hints or ())), t0_mode, None if hyper is None else id(hyper))
    if key not in _FP2['cache']:
        _FP2['cache'][key] = copy.deepcopy(_FP2['orig'](P, plane, cal, parent, hyper=hyper, t0_hints=t0_hints,
                                                         f_thresh=f_thresh, probe=probe, t0_mode=t0_mode))
    r = copy.deepcopy(_FP2['cache'][key])
    if r is None:
        return None
    u_lo, u_hi = reco.u_window(cal)
    fa, fb = r['children']
    plaus = [bool(np.isfinite(f.q_uend) and u_lo <= f.q_uend <= u_hi and abs(f.tan_theta) < reco.tan_max())
             for f in (fa, fb)]
    thr = reco.TWO_TRACK_F if f_thresh is None else f_thresh
    r['both_plausible'] = bool(plaus[0] and plaus[1])
    r['guards_ok'] = bool(r['distinguishable'] and r['column_shared'] and r['both_plausible'])
    r['accepted'] = bool(r['fstat'] >= thr and r['guards_ok'])
    r['threshold'] = float(thr)
    for i, f in enumerate((fa, fb)):
        f._plausible = plaus[i]
    r['a_plausible'], r['b_plausible'] = plaus
    return r


def worker_init(bundle_path, configs, memo=True):
    from wft import reco
    reco._worker_init(bundle_path)
    _OPTS['configs'] = list(configs)
    if memo and _FP2['orig'] is None:
        _FP2['orig'] = reco.fit_plane_two
        reco.fit_plane_two = _fit_plane_two_memo


def _rank(fits_win_order, tm):
    """fit_plane_candidates' ranking for limit ``tm``: plausible = column window and |tan| < tm, then (plausible, dchi2)
    descending, ties in window order (its stable reverse sort over the window-ordered list)."""
    for f in fits_win_order:
        f._plausible = bool(f._q7_colwin and abs(f._q7_tan1) < tm)
    r = sorted(fits_win_order, key=lambda f: (1 if f._plausible else 0, f._dchi2), reverse=True)
    for f in r:
        f.n_candidates = len(r)
    return r


def fit_multi(payload):
    """wft.reco._worker_fit for every configuration in _OPTS['configs'] -> {cfg name: row (with '_cand', '_splits')}."""
    from wft import reco
    eid, wins, seeds, n_hits, spark, ftst = payload
    ftst = ftst if isinstance(ftst, dict) else {}
    _FP2['cache'].clear()
    cal = reco._CAL
    saved_tm, saved_sc = cal.tan_max, reco.SPLIT_CORROB_MODE
    base = {}
    cal.tan_max = INF
    try:
        for plane in ('x', 'y'):
            cand = wins.get(plane)
            if not cand:
                base[plane] = []
                continue
            try:
                _best, ranked = reco.fit_plane_candidates(
                    cand, plane, cal, seeds=seeds.get(plane), return_all=True,
                    t0_prior=reco.t0_prior_for(cal, plane, ftst.get(plane)))
            except Exception:
                ranked = []
            for f in ranked:
                f._q7_colwin = bool(f._plausible)       # tan_max = inf: the column window alone
                f._q7_tan1 = float(f.tan_theta)         # the single-view tangent _candidate_score tested
            base[plane] = sorted(ranked, key=lambda f: f._win)
    finally:
        cal.tan_max = saved_tm
    # beam payloads carry no '_rescue_<plane>' window, so the head-on rescue never runs here (as in _worker_fit)
    if any(wins.get(f'_rescue_{p}') is not None for p in ('x', 'y')):
        raise NotImplementedError('head-on rescue windows: not reproduced by fit_multi')
    ftst_diff = (ftst['x'] - ftst['y'] if ftst.get('x') is not None and ftst.get('y') is not None else None)
    out = {}
    stages = {}
    for tm, split, sel in _OPTS['configs']:
        key = (tm, split)
        if key not in stages:
            stages[key] = _two_track_stage(base, wins, tm, split, ftst_diff)
        all_fits, fits, splits, replaced = copy.deepcopy(stages[key])
        cal.tan_max = tm
        try:
            try:
                pairs = reco.select_tracks(all_fits, ftst_diff, cal, max_tracks=reco.MAX_TRACKS,
                                           pairing=reco._PAIRING, mode=sel)
            except Exception:
                pairs = []
            if getattr(cal, 'bow', None) or (getattr(cal, 'v_map', None) or {}).get('xy'):
                for ix, iy, _g in pairs:
                    try:
                        reco.apply_bow(cal, all_fits['x'][ix], all_fits['y'][iy])
                    except Exception:
                        pass
                if fits.get('x') is not None or fits.get('y') is not None:
                    reco.apply_bow(cal, fits.get('x'), fits.get('y'))
            row = reco.row_from_fits(eid, fits, n_hits, spark)
            row['n_tracks'] = int(sum(1 for _i, _j, g in pairs if g))
            hb = [getattr(fits.get(p), '_bow_h', None) for p in ('x', 'y')]
            row['bow_h'] = float(next((h for h in hb if h is not None), np.nan))
            for p, h in zip('xy', hb):
                row[f'{p}_bow_h'] = float(h) if h is not None else np.nan
            for plane in ('x', 'y'):
                f = ftst.get(plane)
                row[f'{plane}_ftst'] = int(f) if f is not None else -1
            if reco.TWO_TRACK:
                row['n_splits'] = int(sum(1 for r in splits if r['accepted']))
            row['_cand'] = reco.candidate_rows(eid, all_fits, pairs, ftst, replaced)
            row['_splits'] = [dict(event_id=int(eid), **r) for r in splits]
            out[cfg_name(tm, split, sel)] = row
        finally:
            cal.tan_max = saved_tm
    reco.SPLIT_CORROB_MODE = saved_sc
    return eid, out


def _two_track_stage(base, wins, tm, split, ftst_diff):
    """Ranking + joint two-track stage + lost-track revert for (tm, split): _worker_fit's TWO_TRACK block."""
    from wft import reco
    cal = reco._CAL
    b = copy.deepcopy(base)
    all_fits = {p: _rank(b[p], tm) for p in ('x', 'y')}
    fits = {p: (all_fits[p][0] if all_fits[p] else None) for p in ('x', 'y')}
    splits, replaced = [], {}
    saved = cal.tan_max
    cal.tan_max = tm
    reco.SPLIT_CORROB_MODE = split
    try:
        if reco.TWO_TRACK:
            try:
                kept = all_fits
                all_fits, splits, replaced = reco.resolve_two_tracks(all_fits, wins, cal, ftst_diff, attempts=None)
                if any(replaced.values()) and reco._n_gated(kept, ftst_diff) > reco._n_gated(all_fits, ftst_diff):
                    for f in [g for v in replaced.values() for g in v]:
                        f._split_replaced = False
                    for r in splits:
                        r['accepted'] = False
                        r['reverted_lost_track'] = True
                    all_fits, replaced = kept, {'x': [], 'y': []}
                for plane in ('x', 'y'):
                    if replaced.get(plane) and all_fits.get(plane):
                        fits[plane] = all_fits[plane][0]
            except Exception:
                splits, replaced = [], {}
    finally:
        cal.tan_max = saved
    return all_fits, fits, splits, replaced


def run_payloads(bundle, payloads, configs, jobs, chunksize=2):
    """-> {cfg: (event rows, candidate rows, split rows)}"""
    from concurrent.futures import ProcessPoolExecutor
    res = {cfg_name(*c): ([], [], []) for c in configs}
    with ProcessPoolExecutor(max_workers=jobs, initializer=worker_init, initargs=(bundle, configs)) as pool:
        for eid, out in pool.map(_safe_fit, payloads, chunksize=chunksize):
            for name, row in out.items():
                E, C, S = res[name]
                C.extend(row.pop('_cand', []))
                S.extend(row.pop('_splits', []))
                E.append(row)
    return res


def _safe_fit(payload):
    try:
        return fit_multi(payload)
    except Exception:
        traceback.print_exc()
        raise


# --------------------------------------------------------------------------- inputs
def beam_cfg(run, subrun, arm, base=RUNS):
    from ntof_tracking import wft_beam as wb
    fx, fy = FEUS[arm]
    return wb.BeamConfig(KEY=f'run_{run}_{subrun}_mx17_{arm}', RUN=f'run_{run}', SUB_RUN=subrun,
                         DET_NAME=f'mx17_{arm}', MX17_FEU_X=fx, MX17_FEU_Y=fy, BASE_PATH=base.rstrip('/') + '/',
                         OUT_BASE='/tmp', HITS_DIR=HITS_DIR,
                         conditions=dict(gas='Ar/Iso 90/10', sample_ns=60.0, n_samples=20))


def tag_payloads(cfg, tag, cal, events=None, limit=None, pad=3):
    """ntof_reco's seeding + windowing for one acquisition (wft_beam.reconstruct_subrun, the hits path)."""
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    pos = wio.strip_position_map(cfg)
    hp = wb.hits_file_for_tag(cfg, tag)
    hits = wb.read_hits_tag(hp, (cfg.MX17_FEU_X, cfg.MX17_FEU_Y))
    seeds = wb.seeds_from_hits_beam(hits, pos, cfg.MX17_FEU_X, cfg.MX17_FEU_Y, hot=cal.hot,
                                    min_strips=MIN_STRIPS, dead=cal.dead)
    wanted = set(seeds)
    if events is not None:
        wanted &= set(events)
    if limit:
        wanted = set(sorted(wanted)[:limit])
    return wb._windows_for_tag(cfg, tag, pos, seeds, wanted, pad, dead=cal.dead), len(wanted)


def subrun_tags(cfg):
    from ntof_tracking import wft_beam as wb
    return wb.subrun_tags(cfg)


def write_tables(res, out_dir, stem, extra_cols=None):
    """One parquet per kind with a 'cfg' column: <stem>.events / .candidates / .splits.parquet"""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    for k, idx in (('events', 0), ('candidates', 1), ('splits', 2)):
        frames = []
        for name, t in res.items():
            if t[idx]:
                frames.append(pd.DataFrame(t[idx]).assign(cfg=name, **(extra_cols or {})))
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(os.path.join(out_dir, f'{stem}.{k}.parquet'), index=False)


def ac_seeded_events(run, subrun, tags=None):
    """{tag: event ids} seeded in both views of both A and C (the A-C line candidates; seeding only, no fit)."""
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft.calib import CalibrationBundle
    out = {}
    cfgs = {arm: beam_cfg(run, subrun, arm) for arm in 'AC'}
    cals = {arm: CalibrationBundle.load(os.path.join(BUNDLES, f'v1k_{arm}')) for arm in 'AC'}
    for tag in tags or subrun_tags(cfgs['A']):
        ids = None
        for arm in 'AC':
            cfg, cal = cfgs[arm], cals[arm]
            pos = wio.strip_position_map(cfg)
            hits = wb.read_hits_tag(wb.hits_file_for_tag(cfg, tag), (cfg.MX17_FEU_X, cfg.MX17_FEU_Y))
            s = wb.seeds_from_hits_beam(hits, pos, cfg.MX17_FEU_X, cfg.MX17_FEU_Y, hot=cal.hot,
                                        min_strips=MIN_STRIPS, dead=cal.dead)
            e = {k for k, v in s.items() if v['x'] and v['y']}
            ids = e if ids is None else ids & e
        out[tag] = ids
    return out
