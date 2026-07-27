#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the run_67 PLASTIC-THRESHOLD x drift x resist HV scan
(started 2026-07-22 ~22:45; Ar/Iso 90/10, 3He target, neutrons, no Pb;
32 smp x 60 ns = 1.92 us, latency 35, IPD 5, RAW / ZS off, on 10 GbE).

Sibling of run64_scan / run61_scan — SAME trigger recipe
(scint --singles --ps-pickup), SAME reco chain, SAME per-event/segment/drift
deliverables. Separate package so the other runs' numbers stay frozen.

WHAT DIFFERS FROM run_64 (read this before touching the analysis):

1. **A PLASTIC-THRESHOLD axis.** The M2 plastic-scintillator discriminator
   threshold is stepped per sub-run over the GEANT ladder 1.41 / 1.13 / 0.90 MIP
   (tags m141 / m113 / m090). This is the NEW third scan axis on top of
   drift x resist. Sub-run names: 'm{mip}On_dr{drift}_r{resist}_{seq}'.
   Every table carries a `mip` column (141 / 113 / 90, integer x100 MIP).

2. **The deadtime comb is GONE — time-since-flash is BROADLY spread, not
   quantized.** run_58/61/64 used a doubles/PS deadtime comb that accepted in a
   rigid ladder (~5 / 13.5 / 27 / 41 ... ms), so events fell into discrete teeth
   and the analysis binned by comb tooth (early/mid/late). run_67 is a SINGLES
   trigger inside the N93B acceptance gate (~1 -> 81 ms after the flash), plus
   the FEU watermark was dropped to Hwm 2 so the post-flash burst is spread
   evenly across the window. The result is a CONTINUOUS time-since-flash
   distribution. There is no comb to bin by. **Windows are defined by hand at
   analysis time** (see WINDOW_SETS) — the operator wants to try several
   binnings, so windows are NOT baked into the cache; the cached tables store
   raw `dt_ms` per event and every window set is applied downstream.

3. **det D resist = A/B/C setpoint (NO -10 V offset).** run_64 held D 10 V low;
   run_67 does not (DET_D_RESIST_OFFSET = 0). resist_for_det therefore returns
   the setpoint unchanged for all four detectors.

MESH (unchanged from run_64): injection is cabled to A and C for the WHOLE run
and never switched (M6.B drives the mesh switches AND holds up the SiPM wall
bias, so it CANNOT be switched off without collapsing the walls — see
docs/HANDOFF_2026-07-22_m6_enable_layers.md). B and D are uncabled and act as
the simultaneous, same-beam, same-HV, full-gain **no-mesh control** inside every
sub-run. There is NO same-detector mesh on/off contrast within this run.

FLASH ANCHOR: PS + singles are co-framed in the 32-sample window (M4.D in0 G&D
delay 1800 ns pulls the flash from sample 43 to ~13, beside the singles MM at
~11). The per-burst leader is the saturating gamma-flash trigger (n_big ~1e5);
dt_ms is measured from it, exactly as run_64. flash_ok gates on n_big > 150.

Grid (COMPLETE blocks as of 2026-07-23): drift {600, 700, 500} x resist
{550..520 in 5 V steps} x mip {141, 113, 090} = 63 sub-runs; a drift-400 block
was only fragmentary (2 sub-runs) — check per-cell n, do not assume a full grid
at drift 400.

Per sub-run this module builds and caches three tables (identical schema to
run64_scan):
  events.parquet    — one row per event: burst/dt/flash + per-det MM summaries
  segs.parquet      — one row per track-class segment (for the drift measurement)
  driftspec.parquet — per-det clean-hit drift-time spectrum, 20 ns bins

Reco chain reused verbatim from ntof_tracking.reco.
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

# m{mip}On_dr{drift}_r{resist}_{seq} — mesh is always 'On' this run, but keep the
# token so a stray 'Off' sub-run (voided, see run67-hv-mesh-thresh-scan memory)
# would parse and be visible rather than silently dropped.
SUBRUN_RE = re.compile(
    r'm(?P<mip>\d+)(?P<mesh>On|Off)_dr(?P<drift>\d+)_r(?P<resist>\d+)_(?P<seq>\d+)$')

RUN = 'run_67'
OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'July_HV_Scan', 'run67_scan')
CACHE_DIR = os.path.join(OUT_BASE, 'cache')

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
DLET = {d: d[-1] for d in DETS}
DET_D_RESIST_OFFSET = 0.0       # run_67: det D resist = A/B/C setpoint (no offset)

# mesh charge-injection is cabled to A and C for the whole run; B and D are the
# uncabled, same-beam, full-gain no-mesh control. Never modulated.
MESH_DETS = ('A', 'C')
CONTROL_DETS = ('B', 'D')
MESH_DELAY_NS = 1260.0

# plastic-threshold ladder, as the integer tag (MIP x100) -> label
MIP_LEVELS = [141, 113, 90]
MIP_LABEL = {141: '1.41 MIP', 113: '1.13 MIP', 90: '0.90 MIP'}
# non-blue on purpose: the IPC production curve owns blue in the flash-timing
# overlays, so the recorded-event histograms must be visually distinct from it.
MIP_COLOR = {141: '#d95f0e', 113: '#238b45', 90: '#7d1f8f'}

BURST_GAP_S = 0.10             # spill lasts ~77 ms, inter-spill >=1.16 s
FLASH_AMP = 1000               # amplitude counted as a 'big' (saturated-ish) hit
FLASH_NBIG = 150               # leader with > this many big hits = confirmed flash
BUSY_CLEAN_STRIPS = 120        # busy det = discharge / pile-up
RECO_MAX_HITS = 60000          # above this an event is zero-filled + busy-flagged
                               # instead of reco'd; still in the denominator

# ---- time-since-flash windows, defined BY HAND (run_67 has no comb) ----
# The N93B gate accepts ~1 -> 81 ms after the flash; readout opens at 1 ms.
# The operator asked for two binnings; both are applied downstream from the same
# cache. Edges in ms, [lo, hi).
READOUT_START_MS = 1.0
GATE_CLOSE_MS = 81.0
WINDOW_SETS = {
    'broad': [(1.0, 10.0), (10.0, 30.0), (30.0, 80.0)],
    'fine':  [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 30.0),
              (30.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 80.0)],
}


def win_label(lo, hi):
    return f'{lo:g}-{hi:g} ms'


# drift-column time spectrum bins (32 smp x 60 ns = 1.92 us window + margin)
SPEC_LO_NS, SPEC_HI_NS, SPEC_BIN_NS = -600.0, 2100.0, 20.0
SPEC_EDGES = np.arange(SPEC_LO_NS, SPEC_HI_NS + SPEC_BIN_NS, SPEC_BIN_NS)
SPEC_CENTERS = 0.5 * (SPEC_EDGES[:-1] + SPEC_EDGES[1:])


def parse_subrun(name):
    m = SUBRUN_RE.match(name)
    if not m:
        return None
    d = {'name': name, 'mesh': m.group('mesh')}
    for k in ('mip', 'drift', 'resist', 'seq'):
        d[k] = int(m.group(k))
    return d


def list_subruns(run=RUN, mesh='On'):
    """Complete (combined_hits present) scan sub-runs of a run, parsed.

    mesh='On' (default) skips the voided mesh-OFF sub-runs; mesh=None keeps all.
    """
    out = []
    run_dir = os.path.join(io.BASE_PATH, run)
    for name in sorted(os.listdir(run_dir)):
        d = parse_subrun(name)
        if not d:
            continue
        if mesh is not None and d['mesh'] != mesh:
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
        # position within the spill; with singles the comb is gone, so this is a
        # crude ordinal (0 = flash leader, 1.. = singles), not a fixed-dt tooth.
        'idx_in_burst': np.arange(len(tns)) - lead_idx[bid],
        'n_hits_tot': n_hits_tot, 'n_big': n_big,
        'n_big_leader': n_big[lead_idx][bid],
        'n_hits_leader': n_hits_tot[lead_idx][bid],
    })

    for det in DETS:
        L = DLET[det]
        cnt = hits[hits['det'] == det].groupby('eventId').size()
        ev[f'n_hits_{L}'] = ev['eventId'].map(cnt).fillna(0).astype(int)

    # ---- reco on probe events only (the flash leader is not a physics trigger) ----
    do_reco = (~ev['is_leader'] & (ev['n_hits_tot'] <= RECO_MAX_HITS)).to_numpy()
    ev['reco_skipped'] = ~do_reco & ~ev['is_leader']
    reco_ids = set(ev['eventId'][do_reco].tolist())

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
    for Ld in DLET.values():
        for c in (f'n_trkseg_{Ld}', f'n_trkseg_x_{Ld}', f'n_trkseg_y_{Ld}',
                  f'n_pair_{Ld}'):
            ev.loc[ev['reco_skipped'], c] = 0

    segs_df = pd.DataFrame(seg_rows)
    if not segs_df.empty:
        segs_df = segs_df.merge(ev[['eventId', 'dt_ms', 'flash_ok', 'burst']],
                                on='eventId', how='left')

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
            df['mip'] = meta.get('mip', -1)
            df['drift'] = meta.get('drift', -1)
            df['resist'] = meta.get('resist', -1)

    ev.to_parquet(ev_path)
    (segs_df if segs_df is not None and not segs_df.empty
     else pd.DataFrame()).to_parquet(seg_path)
    spec_df.to_parquet(spec_path)
    return ev, segs_df, spec_df


def load_all(run=RUN, require_cache=True, mesh='On'):
    """Concatenate cached tables; skips uncached sub-runs (or builds them)."""
    evs, segs, specs = [], [], []
    for d in list_subruns(run, mesh=mesh):
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
    """Effective resist HV of a detector given the A/B/C setpoint.

    run_67: det D = A/B/C (no offset), so this is the identity for all dets.
    """
    return resist_abc + (DET_D_RESIST_OFFSET if det_letter == 'D' else 0.0)


def assign_window(dt_ms, windows):
    """Map dt-since-flash [ms] to a window label given [(lo,hi), ...] edges.

    Returns an object array of 'lo-hi ms' labels ('' outside every window).
    Vectorized: accepts a scalar, array, or pandas Series.
    """
    dt = np.asarray(dt_ms, float)
    out = np.full(dt.shape, '', dtype=object)
    for lo, hi in windows:
        out[(dt >= lo) & (dt < hi)] = win_label(lo, hi)
    return out if out.ndim else out.item()


def add_window(ev, windows):
    """Return a copy of an events table with a 'window' column for one set."""
    ev = ev.copy()
    ev['window'] = assign_window(ev['dt_ms'].to_numpy(), windows)
    return ev
