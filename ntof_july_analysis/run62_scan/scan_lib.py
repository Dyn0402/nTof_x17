#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the run_62 SINGLES-ONLY (no PS) 2-D drift x resist scan analysis
(2026-07-21, Ar/Iso 90/10, 3He target, neutrons, no Pb; full readout
64 smp x 60 ns = 3.84 us, IPD 90, latency 33).

Sibling of ntof_july_analysis/run58_scan (same reco chain, same deliverables);
the differences are all in the trigger and therefore in the TIME AXIS — see
below. Kept as a separate package so run_58's published numbers stay frozen.

Scan grid (subruns 'sng_dr{drift}_r{resist}_{seq}'), truncated at 3 h:
  drift 700 V: resist A/B/C 560 -> 520 V (-5 V, 9 pts)   [complete]
  drift 300 V: resist A/B/C 560 -> 545 V (-5 V, 4 pts)   [last one 1.5 min only]
  det D resist held 10 V below the A/B/C setpoint throughout.
Note the resist window is 560->520, i.e. 20 V BELOW run_58's 580->540; the two
scans overlap only on 560..540.

Timing model — the run_58 model does NOT carry over unchanged.  run_62 has NO
PS/gamma-flash trigger leg, so no event is triggered BY the flash. What the
beam still provides is the n_TOF proton pulse: each pulse opens the 30 ms N93B
gate and the DAQ accepts scint singles in a deadtime comb of ~6 teeth spaced by
the ~13.6 ms SCA readout cycle (measured; see flash_comb/run62_spill_comb.py):

  tooth 0 : 5 events, all within ~0.15 ms of the spill start  <- AT the flash
  tooth 1+: ~2 events each at 13.6, 27.2, 40.8, 54.4, 68 ms

So the burst leader is a PHYSICS single, not the flash, and dt_ms = time since
the leader = time since the spill start ~= time since the flash (the flash is
prompt and tooth 0 opens within 150 us of it). The recovery axis therefore
survives, and is in fact better sampled than in run_58: run_58 spent tooth 0 on
the flash trigger itself, run_62 hands those 5 slots to physics.

Consequences for this library (each flagged CHANGED vs run58_scan/scan_lib.py):
  * the burst-quality gate is `spill_ok` (a full 5-event tooth 0 = the spill
    start was captured), NOT run_58's `flash_ok` (leader saturated by the flash).
  * leaders ARE reconstructed — they are physics events here.
  * saturated / pathological events are NOT dropped from the table; they are
    zero-filled and flagged `busy`, so they stay in the efficiency DENOMINATOR.
    (Post-flash blindness is a real inefficiency and tooth 0 is where it lives:
    ~90% of tooth-0 events are front-end-blind with ~4 raw hits, ~10% are flash
    giants with >2000 saturated hits.)
  * probe class is defined by comb TOOTH index, not by hand-placed ms windows.

Per sub-run this module builds and caches three tables:
  events.parquet    — one row per event: burst/dt/tooth + per-det MM summaries
                      (raw & clean hits, clean strips, track segments, 3D pairs,
                      best-segment charge, busy flag)
  segs.parquet      — one row per track-class segment (charges, spans, tspan for
                      the drift measurement, dt_ms of its event)
  driftspec.parquet — per-det clean-hit time spectrum (count + amplitude-weighted)
                      in fixed 20 ns bins, aggregated over all reco'd events;
                      the drift-column leading (t0/anode) and trailing (cathode)
                      edges live here -> v_drift & effective gap (analyze_drift).

Reco chain reused verbatim from ntof_tracking.reco (noise -> segments ->
pairing -> geometry), exactly as run_58 / the doubles hv_track_scan.
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
    r'sng_dr(?P<drift>\d+)_r(?P<resist>\d+)_(?P<seq>\d+)$')

RUN = 'run_62'
OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'run62_scan')
CACHE_DIR = os.path.join(OUT_BASE, 'cache')

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
DLET = {d: d[-1] for d in DETS}
DET_D_RESIST_OFFSET = -10.0     # det D resist held 10 V below the A/B/C setpoint

BURST_GAP_S = 0.10              # spill lasts ~80 ms, inter-spill >=1.16 s -> any
                                # threshold in 0.09-0.5 s gives the same 192
                                # bursts/subrun (checked on sng_dr700_r560_000)
FLASH_AMP = 1000               # amplitude counted as a 'big' (saturated-ish) hit
FLASH_NBIG = 150               # > this many big hits = flash-saturated / discharge
BUSY_CLEAN_STRIPS = 120        # search.py convention: busy det = discharge/pile-up
RECO_MAX_HITS = 60000          # above this an event is zero-filled + busy-flagged
                               # instead of reco'd (pathological pile-up; ~1% of
                               # events, all in the last tooth). Still counted in
                               # the efficiency denominator.

# comb structure (measured, flash_comb/run62_*_spillcomb.json): teeth at
# 0, 13.6, 27.2, 40.8, 54.4, 68 ms. Tooth index is the natural probe axis;
# the ms edges below are only the fallback / plotting labels.
DEAD_CYCLE_MS = 13.6
N_TEETH = 6
FIRST_TOOTH_N = 5              # rested-SCA buffer depth at n_samples=64
SPILL_OK_MIN_TOOTH0 = 4        # burst accepted as a full spill if tooth 0 has
                               # >= this many events (98% of spills have 5;
                               # short ones are run/file-boundary truncations)

# probe classes, defined on the tooth index (run_58 used hand-placed ms windows
# because its comb only had 3 batches):
#   early = tooth 0   (at the flash, front-end blind/saturated)
#   mid   = tooth 1   (~13.6 ms, partially recovered)
#   late  = tooth >=2 (>=27 ms, fully recovered)  <- the efficiency probe
CLASS_TEETH = {'early': (0, 0), 'mid': (1, 1), 'late': (2, N_TEETH + 2)}
CLASS_EDGES_MS = {'early': (0.0, 3.0), 'mid': (8.0, 20.0), 'late': (20.0, 95.0)}

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
    dt_ms = (t_s - t_s[lead_idx][bid]) * 1e3
    # CHANGED vs run_58: no flash trigger -> the comb tooth replaces the
    # flash-leader tag, and a spill is 'ok' when its first tooth is complete.
    tooth = np.rint(dt_ms / DEAD_CYCLE_MS).astype(int)
    n_tooth0 = np.bincount(bid[tooth == 0], minlength=bid[-1] + 1)
    spill_ok = (n_tooth0 >= SPILL_OK_MIN_TOOTH0)[bid]

    ev = pd.DataFrame({
        'eventId': uev, 'tns': tns, 'burst': bid, 'is_leader': new_b,
        'spill_ok': spill_ok, 'dt_ms': dt_ms, 'tooth': tooth,
        'n_tooth0': n_tooth0[bid],
        'n_hits_tot': n_hits_tot, 'n_big': n_big,
        'n_big_leader': n_big[lead_idx][bid],
        'n_hits_leader': n_hits_tot[lead_idx][bid],
    })

    # per-det raw hit counts for every event (vectorized)
    for det in DETS:
        L = DLET[det]
        cnt = hits[hits['det'] == det].groupby('eventId').size()
        ev[f'n_hits_{L}'] = ev['eventId'].map(cnt).fillna(0).astype(int)

    # ---- reco on every event ----
    # CHANGED vs run_58, which skipped burst leaders (there = the flash trigger)
    # and every n_big>150 event. Here the leader is a physics single, and the
    # saturated events are the post-flash blindness we are trying to MEASURE, so
    # both are kept. Only pathological pile-up (hit count above RECO_MAX_HITS,
    # ~1% of events, essentially all in the last comb tooth) is skipped for
    # runtime; those rows are zero-filled and flagged reco_skipped, so they stay
    # in the efficiency denominator as genuine inefficiency.
    do_reco = (ev['n_hits_tot'] <= RECO_MAX_HITS).to_numpy()
    ev['reco_skipped'] = ~do_reco
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
        segs_df = segs_df.merge(ev[['eventId', 'dt_ms', 'tooth', 'spill_ok',
                                    'burst']],
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


def probe_class(tooth):
    """Map comb-tooth index to the probe class (early/mid/late/'').

    Vectorized: accepts a scalar, array, or pandas Series. run_58 keyed this off
    dt_ms because its comb had only three batches; run_62's 6-tooth comb makes
    the tooth index the cleaner handle (and the teeth drift by ~1 ms across a
    spill, which fixed ms windows clip).
    """
    t = np.asarray(tooth, int)
    cls = np.full(t.shape, '', dtype=object)
    for name, (lo, hi) in CLASS_TEETH.items():
        cls[(t >= lo) & (t <= hi)] = name
    return cls if cls.ndim else cls.item()


def add_probe_class(ev):
    """Return a copy of an events table with a 'probe_class' column added."""
    ev = ev.copy()
    ev['probe_class'] = probe_class(ev['tooth'].to_numpy())
    return ev
