#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the run_61 SINGLES+PS 2-D drift x resist scan analysis
(2026-07-20 17:56 -> 2026-07-21 04:32, finished normally; Ar/Iso 90/10, 3He
target, neutrons, no Pb; full readout 64 smp x 60 ns = 3.84 us, IPD 90,
latency 33).

Sibling of ntof_july_analysis/run58_scan — same trigger recipe (scint --singles
--ps-pickup), same reco chain, same deliverables. Kept as a separate package so
run_58's published numbers stay frozen. Two things genuinely differ and both are
handled below: the HV grid, and the deadtime-comb timing.

Scan grid (subruns 'sngPS_dr{drift}_r{resist}_{seq}'), 60 sub-runs, ALL COMPLETE:
  drift OUTER (all 4 dets)   700, 600, 500, 400, 300, 200 V   (6 pts)
  resist INNER A/B/C         560 -> 515 V (-5 V, 10 pts, taken as two
                             interleaved passes: 560->520 by -10, then
                             555->515 by -10); det D held 10 V lower.
Note the resist window sits 20 V BELOW run_58's (580->540): the two scans
overlap only on 560..540 V, and run_61 extends the curve down to 515 V.

Timing model — MEASURED on run_61 itself (sngPS_dr700_r560_000 and
sngPS_dr200_r515_059 agree), because run_58's hardcoded windows do NOT fit:

  idx 0   dt = 0        1 event   n_big ~85k   <- the gamma flash trigger
  idx 1-4 dt ~ 4.1 ms   4 events  ~6 raw hits  <- front-end BLIND, 14% still
                                                  saturated
  idx 5-6 dt ~ 13.5 ms  2 events  ~96 hits     <- partially recovered
  idx 7+  dt ~ 27.2, 41.0, 55.3, 69.1 ms, 2 events each, ~500 hits <- recovered
  => ~14.9 events/spill, dead cycle ~13.6 ms.

run_58's CLASS_EDGES_MS were (0,1)/(8,18)/(20,33) ms. Applied to run_61 those
would put NOTHING in 'early' (its 0-1 ms slot holds only the flash trigger,
which is excluded from reco), silently drop the 4.1 ms blind batch, and keep
only the 27 ms pair out of the four recovered teeth. The edges below are
re-derived from the comb above; the early/mid/late SEMANTICS are unchanged
(first reconstructable batch / partially recovered / fully recovered), so the
late-probe efficiency stays comparable with run_58 cell-for-cell.

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

RUN = 'run_61'
OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'run61_scan')
CACHE_DIR = os.path.join(OUT_BASE, 'cache')

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
DLET = {d: d[-1] for d in DETS}
DET_D_RESIST_OFFSET = -10.0     # det D resist held 10 V below the A/B/C setpoint

BURST_GAP_S = 0.10              # spill lasts ~77 ms, inter-spill >=1.16 s -> any
                                # threshold in 0.09-0.5 s gives the same bursts
FLASH_AMP = 1000               # amplitude counted as a 'big' (saturated-ish) hit
FLASH_NBIG = 150               # leader with > this many big hits = confirmed flash
                               # (run_61 leaders: n_big ~85k, 100% confirmed at
                               # drift 700, 99.4% at drift 200)
BUSY_CLEAN_STRIPS = 120        # search.py convention: busy det = discharge/pile-up
RECO_MAX_HITS = 60000          # above this an event is zero-filled + busy-flagged
                               # instead of reco'd (pathological pile-up). Still
                               # counted in the efficiency denominator.

# deadtime-comb accept batches, MEASURED on run_61 (see module docstring):
# flash at 0, then 4 events at ~4.1 ms, 2 at ~13.5, and 2 each at ~27.2 / 41.0 /
# 55.3 / 69.1 ms. Computed at ANALYSIS time from the cached dt_ms -> no
# re-caching needed if these are retuned.
#   early = the 4.1 ms batch  — first reconstructable events, front-end blind
#   mid   = the 13.5 ms pair  — partially recovered
#   late  = 27 ms and beyond  — fully recovered (4 teeth, ~8 events/spill)
# 'late' is 4x the statistics run_58's (20,33) window could reach.
CLASS_EDGES_MS = {'early': (1.0, 8.0), 'mid': (8.0, 20.0), 'late': (20.0, 95.0)}
DEAD_CYCLE_MS = 13.6

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
        # position within the spill: 0 = flash, 1-4 = the 4.1 ms blind batch,
        # 5-6 = 13.5 ms, 7+ = the recovered teeth. Handy cross-check on the
        # dt-based probe classes (the two agree to <1% on run_61).
        'idx_in_burst': np.arange(len(tns)) - lead_idx[bid],
        'n_hits_tot': n_hits_tot, 'n_big': n_big,
        'n_big_leader': n_big[lead_idx][bid],
        'n_hits_leader': n_hits_tot[lead_idx][bid],
    })

    # per-det raw hit counts for every event (vectorized)
    for det in DETS:
        L = DLET[det]
        cnt = hits[hits['det'] == det].groupby('eventId').size()
        ev[f'n_hits_{L}'] = ev['eventId'].map(cnt).fillna(0).astype(int)

    # ---- reco on probe events only (the flash leader is not a physics trigger) ----
    # CHANGED vs run_58, which ALSO dropped every n_big>FLASH_NBIG event. In
    # run_61 that would remove ~14% of the 4.1 ms 'early' batch — i.e. exactly
    # the flash-saturated events whose loss IS the post-flash inefficiency we
    # are measuring — and would bias the early-probe yield upward. Here they are
    # reconstructed and kept; only pathological pile-up (> RECO_MAX_HITS) is
    # zero-filled and flagged reco_skipped, staying in the denominator.
    # Note: this makes run_61's 'early' number not directly comparable with
    # run_58's; 'mid'/'late' (~0% saturated) are unaffected.
    do_reco = (~ev['is_leader'] & (ev['n_hits_tot'] <= RECO_MAX_HITS)).to_numpy()
    ev['reco_skipped'] = ~do_reco & ~ev['is_leader']
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
    # zero-fill the skipped (pathological pile-up) events: no tracks found
    for Ld in DLET.values():
        for c in (f'n_trkseg_{Ld}', f'n_trkseg_x_{Ld}', f'n_trkseg_y_{Ld}',
                  f'n_pair_{Ld}'):
            ev.loc[ev['reco_skipped'], c] = 0

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
