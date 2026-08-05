#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the LEAD-SHIELDING before/after comparison (access of 2026-08-04).

On the morning of 2026-08-04 a large amount of shielding lead was removed from
the setup during an access (run_132 was operator-killed at 08:44 for it). The
worry: a bigger gamma flash reaching the detectors/DAQ lengthens the post-flash
saturation, which would show up as LESS track efficiency (and/or fewer accepted
triggers) at early time-since-flash, ~1-5 ms.

The comparison uses the identically-configured PRODUCTION stat090 runs that
bracket the access — same trigger recipe, same HV, same readout, same gas:

    run_130  before  2026-08-03 18:08 -> 22:08   (5 sub-runs, night-to-night check)
    run_132  before  2026-08-03 22:33 -> 08:45   (11 sub-runs, killed for access)
    run_139  after   2026-08-04 22:01 -> 08:58   (11 sub-runs)

All three: PS + SINGLES at the run_67 optimum, drift 700 V, resist A540/B540/
C525/D520, plastic 0.90 MIP, RAW 20 smp x 60 ns, latency 27, Hwm 1/Lwm 0,
Ar/Iso 90/10, no Pb beam filter. Sub-run names 'stat090_{seq}'.

Sibling of run67_scan/scan_lib.py — SAME reco chain (ntof_tracking.reco), SAME
per-event/segment/driftspec cache schema, SAME efficiency conventions
(P(3D x/y pair) per recorded trigger; denominator = readout_*; blind_frac an
observable, never a cut; Det A the clean-M1 reference). Copied rather than
imported so run_67's numbers stay frozen; the only structural changes:

  * sub-run pattern 'stat090_{seq}' — no HV axes; the cell key is `run`.
  * drift HV is not in the sub-run name: fixed DRIFT_HV = 700 V from the
    run configs (identical in all three runs).
  * meta columns are run/subrun/seq/period instead of mip/drift/resist.

Outputs -> <ANALYSIS_DIR>/lead_shielding_compare/  (per the operator's ask:
a new directory in ~/beam_july/analysis).
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

SUBRUN_RE = re.compile(r'stat090_(?P<seq>\d+)$')

# run -> before/after the 2026-08-04 morning access (the lead removal)
RUNS = ['run_130', 'run_132', 'run_139']
PERIOD = {'run_130': 'before', 'run_132': 'before', 'run_139': 'after'}
PERIOD_COLOR = {'before': '#1f77b4', 'after': '#d62728'}
RUN_COLOR = {'run_130': '#7fb3d5', 'run_132': '#1f77b4', 'run_139': '#d62728'}

DRIFT_HV = 700.0               # identical in all three run_configs

OUT_BASE = os.path.join(io.ANALYSIS_DIR, 'lead_shielding_compare')
CACHE_DIR = os.path.join(OUT_BASE, 'cache')

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']
DLET = {d: d[-1] for d in DETS}

BURST_GAP_S = 0.10             # spill ~77 ms, inter-spill >= 1.16 s (as run_67)
FLASH_AMP = 1000               # amplitude counted as a 'big' (saturated-ish) hit
FLASH_NBIG = 150               # leader with > this many big hits = confirmed flash
RECO_MAX_HITS = 60000          # above this an event is zero-filled + flagged

# nominal acceptance gate (N93B, as run_67); the real edges are MEASURED by
# slide-style gate finding in the comparison script.
READOUT_START_MS = 1.0
GATE_CLOSE_MS = 81.0

# hand windows for the summary table. The early ones are fine on purpose —
# the operator's question is 1-5 ms.
WINDOWS = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 10.0),
           (10.0, 20.0), (20.0, 40.0), (40.0, 76.0)]

# drift-column time spectrum bins (20 smp x 60 ns = 1.2 us window + margin)
SPEC_LO_NS, SPEC_HI_NS, SPEC_BIN_NS = -600.0, 2100.0, 20.0
SPEC_EDGES = np.arange(SPEC_LO_NS, SPEC_HI_NS + SPEC_BIN_NS, SPEC_BIN_NS)
SPEC_CENTERS = 0.5 * (SPEC_EDGES[:-1] + SPEC_EDGES[1:])


def win_label(lo, hi):
    return f'{lo:g}-{hi:g} ms'


def parse_subrun(name):
    m = SUBRUN_RE.match(name)
    if not m:
        return None
    return {'name': name, 'seq': int(m.group('seq'))}


def list_subruns(run):
    """Complete (combined_hits present) stat090 sub-runs of a run, parsed."""
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
    """Build (or load cached) per-event, per-segment and drift-spectrum tables.

    Verbatim run67_scan.scan_lib.build_subrun_tables apart from the meta
    columns and the fixed drift HV (stat090 names carry no HV token).
    """
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
    drift = geo.DriftModel.from_drift_hv(DRIFT_HV)

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
            df['seq'] = meta.get('seq', -1)
            df['period'] = PERIOD.get(run, '?')

    ev.to_parquet(ev_path)
    (segs_df if segs_df is not None and not segs_df.empty
     else pd.DataFrame()).to_parquet(seg_path)
    spec_df.to_parquet(spec_path)
    return ev, segs_df, spec_df


def load_all(runs=None, require_cache=True):
    """Concatenate cached tables across runs; skips uncached sub-runs."""
    runs = runs or RUNS
    evs, segs, specs = [], [], []
    for run in runs:
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


def binom_err(k, n):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / np.maximum(n, 1))
