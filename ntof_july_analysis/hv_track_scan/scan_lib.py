#!/usr/bin/env python3
"""
Library for the run_53/run_55 doubles-trigger recursive HV scan analysis
(90/10 gas, resist 520-560 V x cycles, drift 800 / A 600).

Event-timing model (established 2026-07-18, tt_check*.py):
  Each beam pulse opens a 30 ms gate; the DAQ records EXACTLY 17 events per
  burst in a rigid service pattern: 1 flash leader (t=0, physical stamp),
  ~9 'early' accepts stamped 0.02-0.1 ms, 3 'mid' accepts stamped 8-12 ms,
  3 'late' accepts stamped 17-23 ms. Mid/late stamps carry real ms-scale
  timing (MM liveness evolves between batches at high HV); the 0.1-8 and
  12-17 ms regions are never sampled. The external N1081B TT stream is not
  usable as per-event truth in-window (edge storm + early dropouts).

Per sub-run this module builds and caches two tables:
  events.parquet — one row per event: burst/dt/probe-class + per-det MM
                   summaries (hits, clean strips, track segments, 3D pairs)
  segs.parquet   — one row per track-class segment or 3D pair component,
                   with charges for gain curves.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from ntof_tracking.reco import io, noise, segments as segmod, pairing  # noqa: E402
from ntof_tracking.reco import geometry as geo  # noqa: E402

SUBRUN_RE = re.compile(
    r'scintd_r(?P<resist>\d+)_dr(?P<drift>\d+)dA(?P<driftA>\d+)'
    r'_c(?P<cycle>\d+)_(?P<seq>\d+)$')

OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'hv_track_scan')
CACHE_DIR = os.path.join(OUT_BASE, 'cache')

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
DLET = {d: d[-1] for d in DETS}

BURST_GAP_S = 0.10
FLASH_AMP = 1000
FLASH_NBIG = 150
# probe classes from the accept-stamp comb
CLASS_EDGES_MS = {'early': (0.0, 1.0), 'mid': (5.0, 14.0), 'late': (14.0, 28.0)}
BUSY_CLEAN_STRIPS = 120     # search.py convention: busy det = discharge/pile-up


def parse_subrun(name):
    m = SUBRUN_RE.match(name)
    if not m:
        return None
    d = {k: int(v) for k, v in m.groupdict().items()}
    d['name'] = name
    return d


def list_subruns(run):
    """All complete (combined_hits present) scan sub-runs of a run, parsed."""
    out = []
    run_dir = os.path.join(io.BASE_PATH, run)
    for name in sorted(os.listdir(run_dir)):
        d = parse_subrun(name)
        if not d:
            continue
        comb = glob.glob(os.path.join(run_dir, name, 'combined_hits_root',
                                      '*_datrun_*.root'))
        if not comb:
            continue
        d['run'] = run
        out.append(d)
    return out


def probe_class(dt_ms, is_leader):
    cls = np.full(len(dt_ms), '', dtype=object)
    cls[is_leader] = 'leader'
    for name, (lo, hi) in CLASS_EDGES_MS.items():
        m = ~is_leader & (dt_ms >= lo) & (dt_ms < hi)
        cls[m] = name
    return cls


def _cache_paths(run, subrun):
    d = os.path.join(CACHE_DIR, run)
    os.makedirs(d, exist_ok=True)
    return (os.path.join(d, f'{subrun}_events.parquet'),
            os.path.join(d, f'{subrun}_segs.parquet'))


def build_subrun_tables(run, subrun, lut=None, cfg=None, force=False):
    """Build (or load cached) per-event and per-segment tables for a sub-run."""
    ev_path, seg_path = _cache_paths(run, subrun)
    if not force and os.path.exists(ev_path) and os.path.exists(seg_path):
        return pd.read_parquet(ev_path), pd.read_parquet(seg_path)

    if cfg is None:
        cfg = io.load_run_config(run)
    if lut is None:
        lut = io.build_channel_lut(cfg)
    hits = io.load_subrun_hits(run, subrun, lut,
                               columns=io.HIT_COLUMNS + ['trigger_timestamp_ns'])
    if hits is None or hits.empty:
        return None, None
    drift = geo.DriftModel.from_drift_hv(io.parse_drift_hv(subrun) or 800.0)

    # ---- per-event base table (vectorized) ----
    hits = hits.sort_values('eventId', kind='stable')
    ev_ids = hits['eventId'].to_numpy()
    uev, first = np.unique(ev_ids, return_index=True)
    bounds = np.append(first, len(ev_ids))
    tns = hits['trigger_timestamp_ns'].to_numpy(np.float64)[first]
    amp = hits['amplitude'].to_numpy()
    big = (amp >= FLASH_AMP).astype(np.int64)
    n_big = np.add.reduceat(big, first)
    n_hits_tot = np.diff(bounds)

    order = np.argsort(tns, kind='stable')
    uev, tns, n_big, n_hits_tot = (a[order] for a in (uev, tns, n_big, n_hits_tot))

    t_s = (tns - tns[0]) / 1e9
    new_b = np.append(True, np.diff(t_s) > BURST_GAP_S)
    bid = np.cumsum(new_b) - 1
    lead_idx = np.flatnonzero(new_b)
    flash_ok = (n_big[lead_idx] > FLASH_NBIG)[bid]
    dt_ms = (t_s - t_s[lead_idx][bid]) * 1e3
    cls = probe_class(dt_ms, new_b)

    ev = pd.DataFrame({
        'eventId': uev, 'tns': tns, 'burst': bid, 'is_leader': new_b,
        'flash_ok': flash_ok, 'dt_ms': dt_ms, 'probe_class': cls,
        'n_hits_tot': n_hits_tot, 'n_big': n_big,
        'n_big_leader': n_big[lead_idx][bid],
        'n_hits_leader': n_hits_tot[lead_idx][bid],
    })

    # per-det raw/clean hit counts for every event (vectorized)
    for det in DETS:
        L = DLET[det]
        sub = hits[hits['det'] == det]
        cnt = sub.groupby('eventId').size()
        ev[f'n_hits_{L}'] = ev['eventId'].map(cnt).fillna(0).astype(int)

    # ---- reco on probe events only (leaders/flash-like skipped) ----
    do_reco = (~ev['is_leader'] & (ev['n_big'] <= FLASH_NBIG)).to_numpy()
    reco_ids = set(ev['eventId'][do_reco].tolist())

    per_ev_rows = {}
    seg_rows = []
    for evid, g in hits.groupby('eventId', sort=False):
        if evid not in reco_ids:
            continue
        g = noise.flag_noise(g)
        row = {}
        # clean-strip counts per det (busy/discharge veto material)
        cl = g[g['clean']]
        for det in DETS:
            L = DLET[det]
            gd = cl[cl['det'] == det]
            row[f'n_clean_{L}'] = len(gd)
            row[f'n_clean_strips_{L}'] = gd['channel'].nunique()
        segs = segmod.segments_for_event(g)
        pairs = pairing.pair_xy_3d(segs, drift)
        paired = {(p['det'],): p for p in pairs}
        for det in DETS:
            L = DLET[det]
            tsegs = [s for s in segs if s['det'] == det and s['cls'] == 'track']
            row[f'n_trkseg_{L}'] = len(tsegs)
            row[f'n_trkseg_x_{L}'] = sum(s['plane'] == 'x' for s in tsegs)
            row[f'n_trkseg_y_{L}'] = sum(s['plane'] == 'y' for s in tsegs)
            best = max(tsegs, key=lambda s: s['q_sum'], default=None)
            row[f'seg_q_{L}'] = best['q_sum'] if best else np.nan
            row[f'seg_amax_{L}'] = best['a_max'] if best else np.nan
            dp = [p for p in pairs if p['det'] == det]
            row[f'n_pair_{L}'] = len(dp)
            bp = max(dp, key=lambda p: p['q_x'] + p['q_y'], default=None)
            row[f'pair_q_{L}'] = (bp['q_x'] + bp['q_y']) if bp else np.nan
            row[f'pair_iou_{L}'] = bp['iou'] if bp else np.nan
        per_ev_rows[evid] = row
        for s in segs:
            if s['cls'] != 'track':
                continue
            seg_rows.append({
                'eventId': evid, 'det': DLET[s['det']], 'plane': s['plane'],
                'n_strips': s['n_strips'], 'n_hits': s['n_hits'],
                'pspan_mm': s['pspan_mm'], 'tspan_ns': s['tspan_ns'],
                'q_sum': s['q_sum'], 'a_max': s['a_max'],
                'r2': s.get('r2', np.nan),
                'in_pair': (s['det'],) in paired,
            })

    extra = pd.DataFrame.from_dict(per_ev_rows, orient='index')
    extra.index.name = 'eventId'
    ev = ev.merge(extra.reset_index(), on='eventId', how='left')

    segs_df = pd.DataFrame(seg_rows)
    if not segs_df.empty:
        segs_df = segs_df.merge(
            ev[['eventId', 'dt_ms', 'probe_class', 'flash_ok', 'burst']],
            on='eventId', how='left')

    meta = parse_subrun(subrun) or {}
    for df in (ev, segs_df):
        if df is not None and not df.empty:
            df['run'] = run
            df['subrun'] = subrun
            df['resist'] = meta.get('resist', -1)
            df['cycle'] = meta.get('cycle', -1)

    ev.to_parquet(ev_path)
    (segs_df if not segs_df.empty else pd.DataFrame()).to_parquet(seg_path)
    return ev, segs_df


def load_all(runs=('run_53', 'run_55'), require_cache=True):
    """Concatenate cached tables across runs; skips uncached sub-runs."""
    evs, segs = [], []
    for run in runs:
        for d in list_subruns(run):
            ev_path, seg_path = _cache_paths(run, d['name'])
            if not os.path.exists(ev_path):
                if require_cache:
                    continue
                build_subrun_tables(run, d['name'])
            evs.append(pd.read_parquet(ev_path))
            s = pd.read_parquet(seg_path)
            if not s.empty:
                segs.append(s)
    ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
    sg = pd.concat(segs, ignore_index=True) if segs else pd.DataFrame()
    return ev, sg
