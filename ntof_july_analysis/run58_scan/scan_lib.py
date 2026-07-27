#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the run_58 SINGLES+PS 2-D drift x resist scan analysis
(2026-07-19->, Ar/Iso 90/10, 3He target, neutrons, no Pb; full readout
64 smp x 60 ns = 3.84 us, IPD 90, latency 33).

Scan grid (subruns 'sngPS_dr{drift}_r{resist}_{seq}'):
  drift OUTER (all 4 dets)   700 -> 150 V
  resist INNER A/B/C         580 -> 540 V  (-5 V, 9 pts); det D held 10 V lower

Singles timing model (probed 2026-07-20): each beam pulse opens a 30 ms N93B
gate; ~8-9 singles are accepted per gate in a deadtime comb. The burst leader
(first accept) is ~the gamma flash (confirmed by n_big saturated hits). dt_ms
= time since leader = time since flash = the post-flash recovery/saturation
axis. Continuous (no rigid probe comb like the doubles run_53/55), so bin
dt_ms finely downstream.

Per sub-run this module builds and caches three tables:
  events.parquet    — one row per event: burst/dt/flash + per-det MM summaries
                      (raw & clean hits, clean strips, track segments, 3D pairs,
                      best-segment charge, busy flag)
  segs.parquet      — one row per track-class segment (charges, spans, tspan for
                      the drift measurement, dt_ms of its event)
  driftspec.parquet — per-det clean-hit time spectrum (count + amplitude-weighted)
                      in fixed 20 ns bins, aggregated over all reco'd events;
                      the drift-column leading (t0/anode) and trailing (cathode)
                      edges live here -> v_drift & effective gap (analyze_drift).

Reco chain reused verbatim from ntof_tracking.reco (noise -> segments ->
pairing -> geometry), exactly as the doubles hv_track_scan.
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
    r'sngPS_dr(?P<drift>\d+)_r(?P<resist>\d+)_(?P<seq>\d+)$')

RUN = 'run_58'
OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'run58_scan')
CACHE_DIR = os.path.join(OUT_BASE, 'cache')

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
DLET = {d: d[-1] for d in DETS}
DET_D_RESIST_OFFSET = -10.0     # det D resist held 10 V below the A/B/C setpoint

BURST_GAP_S = 0.10              # >>0.03 s gate, <<supercycle -> clean burst split
FLASH_AMP = 1000               # amplitude counted as a 'big' (saturated-ish) hit
FLASH_NBIG = 150               # leader with > this many big hits = confirmed flash
BUSY_CLEAN_STRIPS = 120        # search.py convention: busy det = discharge/pile-up

# deadtime-comb accept batches (singles, run_58): each 30 ms gate yields ~3
# batches at fixed dt-since-flash (probed 2026-07-20: ~0.02, ~13, ~28 ms; the
# 0.2-8 and 15-22 ms regions are never sampled). Computed at ANALYSIS time from
# the cached dt_ms -> no re-caching needed. 'late' = fully post-flash recovered.
CLASS_EDGES_MS = {'early': (0.0, 1.0), 'mid': (8.0, 18.0), 'late': (20.0, 33.0)}

# drift-column time spectrum: fixed bins spanning the 64 smp x 60 ns window
# (plus a little pre-window margin for the baseline / negative-time tail)
SPEC_LO_NS, SPEC_HI_NS, SPEC_BIN_NS = -600.0, 4200.0, 20.0
SPEC_EDGES = np.arange(SPEC_LO_NS, SPEC_HI_NS + SPEC_BIN_NS, SPEC_BIN_NS)
SPEC_CENTERS = 0.5 * (SPEC_EDGES[:-1] + SPEC_EDGES[1:])


def parse_subrun(name):
    m = SUBRUN_RE.match(name)
    if not m:
        return None
    d = {k: int(v) for k, v in m.groupdict().items()}
    d['name'] = name
    return d


def list_subruns(run=RUN):
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


def _cache_paths(run, subrun):
    d = os.path.join(CACHE_DIR, run)
    os.makedirs(d, exist_ok=True)
    return (os.path.join(d, f'{subrun}_events.parquet'),
            os.path.join(d, f'{subrun}_segs.parquet'),
            os.path.join(d, f'{subrun}_driftspec.parquet'))


def build_subrun_tables(run, subrun, lut=None, cfg=None, force=False):
    """Build (or load cached) per-event, per-segment and drift-spectrum tables."""
    ev_path, seg_path, spec_path = _cache_paths(run, subrun)
    if (not force and os.path.exists(ev_path) and os.path.exists(seg_path)
            and os.path.exists(spec_path)):
        return (pd.read_parquet(ev_path), pd.read_parquet(seg_path),
                pd.read_parquet(spec_path))

    if cfg is None:
        cfg = io.load_run_config(run)
    if lut is None:
        lut = io.build_channel_lut(cfg)
    hits = io.load_subrun_hits(run, subrun, lut,
                               columns=io.HIT_COLUMNS + ['trigger_timestamp_ns'])
    if hits is None or hits.empty:
        return None, None, None
    drift_hv = io.parse_drift_hv(subrun) or 0.0
    drift = geo.DriftModel.from_drift_hv(drift_hv or 800.0)

    # ---- per-event base table (vectorized burst / dt / flash model) ----
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

    ev = pd.DataFrame({
        'eventId': uev, 'tns': tns, 'burst': bid, 'is_leader': new_b,
        'flash_ok': flash_ok, 'dt_ms': dt_ms,
        'n_hits_tot': n_hits_tot, 'n_big': n_big,
        'n_big_leader': n_big[lead_idx][bid],
        'n_hits_leader': n_hits_tot[lead_idx][bid],
    })

    # per-det raw hit counts for every event (vectorized)
    for det in DETS:
        L = DLET[det]
        cnt = hits[hits['det'] == det].groupby('eventId').size()
        ev[f'n_hits_{L}'] = ev['eventId'].map(cnt).fillna(0).astype(int)

    # ---- reco on probe events only (leaders / flash-like skipped) ----
    do_reco = (~ev['is_leader'] & (ev['n_big'] <= FLASH_NBIG)).to_numpy()
    reco_ids = set(ev['eventId'][do_reco].tolist())

    # drift-column spectra: per det (count + amplitude-weighted) accumulators
    spec_n = {det: np.zeros(len(SPEC_CENTERS)) for det in DETS}
    spec_q = {det: np.zeros(len(SPEC_CENTERS)) for det in DETS}

    per_ev_rows = {}
    seg_rows = []
    for evid, g in hits.groupby('eventId', sort=False):
        if evid not in reco_ids:
            continue
        g = noise.flag_noise(g)
        row = {}
        cl = g[g['clean']]
        # accumulate the clean-hit drift-time spectrum per det
        for det in DETS:
            gd = cl[cl['det'] == det]
            L = DLET[det]
            row[f'n_clean_{L}'] = len(gd)
            row[f'n_clean_strips_{L}'] = gd['channel'].nunique()
            if len(gd):
                t = gd['time'].to_numpy()
                a = gd['amplitude'].to_numpy(float)
                spec_n[det] += np.histogram(t, bins=SPEC_EDGES)[0]
                spec_q[det] += np.histogram(t, bins=SPEC_EDGES, weights=a)[0]
        segs = segmod.segments_for_event(g)
        pairs = pairing.pair_xy_3d(segs, drift)
        paired = {(p['det'],) for p in pairs}
        for det in DETS:
            L = DLET[det]
            tsegs = [s for s in segs if s['det'] == det and s['cls'] == 'track']
            row[f'n_trkseg_{L}'] = len(tsegs)
            row[f'n_trkseg_x_{L}'] = sum(s['plane'] == 'x' for s in tsegs)
            row[f'n_trkseg_y_{L}'] = sum(s['plane'] == 'y' for s in tsegs)
            best = max(tsegs, key=lambda s: s['q_sum'], default=None)
            row[f'seg_q_{L}'] = best['q_sum'] if best else np.nan
            row[f'seg_amax_{L}'] = best['a_max'] if best else np.nan
            row[f'seg_tspan_{L}'] = best['tspan_ns'] if best else np.nan
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
                't0_ns': s['t0_ns'], 't1_ns': s['t1_ns'],
                'q_sum': s['q_sum'], 'a_max': s['a_max'],
                'r2': s.get('r2', np.nan),
                'duration_ns': s.get('duration_ns', np.nan),
                'in_pair': (s['det'],) in paired,
            })

    extra = pd.DataFrame.from_dict(per_ev_rows, orient='index')
    extra.index.name = 'eventId'
    ev = ev.merge(extra.reset_index(), on='eventId', how='left')

    segs_df = pd.DataFrame(seg_rows)
    if not segs_df.empty:
        segs_df = segs_df.merge(ev[['eventId', 'dt_ms', 'flash_ok', 'burst']],
                                on='eventId', how='left')

    # drift-spectrum long table
    spec_rows = []
    for det in DETS:
        L = DLET[det]
        for c, n, q in zip(SPEC_CENTERS, spec_n[det], spec_q[det]):
            spec_rows.append({'det': L, 't_ns': c, 'n_clean': n, 'sum_amp': q})
    spec_df = pd.DataFrame(spec_rows)

    meta = parse_subrun(subrun) or {}
    for df in (ev, segs_df, spec_df):
        if df is not None and not df.empty:
            df['run'] = run
            df['subrun'] = subrun
            df['drift'] = meta.get('drift', -1)
            df['resist'] = meta.get('resist', -1)

    ev.to_parquet(ev_path)
    (segs_df if segs_df is not None and not segs_df.empty
     else pd.DataFrame()).to_parquet(seg_path)
    spec_df.to_parquet(spec_path)
    return ev, segs_df, spec_df


def load_all(run=RUN, require_cache=True):
    """Concatenate cached tables; skips uncached sub-runs (or builds them)."""
    evs, segs, specs = [], [], []
    for d in list_subruns(run):
        ev_p, seg_p, spec_p = _cache_paths(run, d['name'])
        if not os.path.exists(ev_p):
            if require_cache:
                continue
            build_subrun_tables(run, d['name'])
        evs.append(pd.read_parquet(ev_p))
        s = pd.read_parquet(seg_p)
        if not s.empty:
            segs.append(s)
        specs.append(pd.read_parquet(spec_p))
    ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
    sg = pd.concat(segs, ignore_index=True) if segs else pd.DataFrame()
    sp = pd.concat(specs, ignore_index=True) if specs else pd.DataFrame()
    return ev, sg, sp


def resist_for_det(resist_abc, det_letter):
    """Effective resist HV of a detector given the A/B/C setpoint (D is -10 V)."""
    return resist_abc + (DET_D_RESIST_OFFSET if det_letter == 'D' else 0.0)


def probe_class(dt_ms):
    """Map dt-since-flash [ms] to the deadtime-comb batch (early/mid/late/'').

    Vectorized: accepts a scalar, array, or pandas Series.
    """
    dt = np.asarray(dt_ms, float)
    cls = np.full(dt.shape, '', dtype=object)
    for name, (lo, hi) in CLASS_EDGES_MS.items():
        cls[(dt >= lo) & (dt < hi)] = name
    return cls if cls.ndim else cls.item()


def add_probe_class(ev):
    """Return a copy of an events table with a 'probe_class' column added."""
    ev = ev.copy()
    ev['probe_class'] = probe_class(ev['dt_ms'].to_numpy())
    return ev
