#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trackcache.py — loader for the 27_run55 per-strip candidate cache.

Concatenates all subrun npz into one candidate-cluster table (pandas) plus a
flat strip store, with helpers to slice the per-strip arrays of any cluster and
to enrich clusters with derived micro-TPC quantities (drift-time occupancy,
robust anchored slope, head-on flag).  Shared by 27b (gate) and 27c (align).
"""
import glob
import os
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', '27_run55')
SAMPLE_NS = 60.0
DET_NAMES = ['A', 'B', 'C', 'D']

# drift geometry (run_config det_center_coords: radius = |center|, transverse
# offset perpendicular to the radial/drift axis).  v_drift @ each det's drift HV
# from garfield Ar/iso 90/10; A=600 V, B/C/D=800 V over 30 mm.
DRIFT = {
    'A': dict(R=234.6, v_um_ns=40.46, drift_v=600),
    'B': dict(R=234.1, v_um_ns=44.06, drift_v=800),
    'C': dict(R=234.6, v_um_ns=44.06, drift_v=800),
    'D': dict(R=234.1, v_um_ns=44.06, drift_v=800),
}
STRIP_KEYS = ('pos', 'amp', 'smp', 'tmx', 'integ', 'tot', 'lsm', 'rsm',
              'sat', 'feu', 'ch')


def load_all(cache_dir=CACHE_DIR, subruns=None):
    """Return (clusters_df, strips_dict, events_df).

    clusters_df rows are candidate clusters with a global strip offset `goff`
    into strips_dict[key] arrays (length `n`).  events_df is per-trigger context
    (t_ms, burst, flash_ok, resist_v, nhit per det/plane) keyed by (sub, eid).
    """
    files = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
    if subruns:
        files = [f for f in files
                 if os.path.basename(f)[:-4] in subruns]
    cl_rows = []
    strips = {k: [] for k in STRIP_KEYS}
    ev_rows = []
    goff = 0
    for si, f in enumerate(files):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        rv = int(d['resist_v']); cyc = int(d['cycle'])
        nc = len(d['c_eid'])
        # append strips, remap offsets to global
        for k in STRIP_KEYS:
            strips[k].append(d['s_' + k])
        cl = pd.DataFrame(dict(
            sub=name, subi=si, resist_v=rv, cycle=cyc,
            eid=d['c_eid'], det=d['c_det'], pln=d['c_pln'], n=d['c_n'],
            qsum=d['c_qsum'], extent=d['c_extent'], cen=d['c_cen'],
            smplo=d['c_smplo'], smphi=d['c_smphi'], dur=d['c_dur'],
            mono=d['c_mono'], nsat=d['c_nsat'], wfmax=d['c_wfmax'],
            slope=d['c_slope'],
            goff=d['c_off'].astype(np.int64) + goff, len=d['c_len']))
        cl_rows.append(cl)
        goff += int(d['s_pos'].shape[0])
        # events
        eid = d['eventId']; nev = len(eid)
        nhit = d['nhit']  # (nev,4,2)
        ev = pd.DataFrame(dict(
            sub=name, subi=si, resist_v=rv, eid=eid.astype(np.int64),
            t_ms=d['t_ms'], burst=d['burst'], flash_ok=d['flash_ok'],
            n_raw=d['n_raw_hits']))
        for di in range(4):
            ev[f'nhit_{DET_NAMES[di]}x'] = nhit[:, di, 0]
            ev[f'nhit_{DET_NAMES[di]}y'] = nhit[:, di, 1]
        ev_rows.append(ev)
    clusters = pd.concat(cl_rows, ignore_index=True)
    strips = {k: np.concatenate(v) for k, v in strips.items()}
    events = pd.concat(ev_rows, ignore_index=True)
    clusters['detn'] = [DET_NAMES[i] for i in clusters['det']]
    clusters['plnn'] = np.where(clusters['pln'].values == 0, 'x', 'y')
    return clusters, strips, events


def cluster_strips(row, strips):
    """Return dict of per-strip arrays for one cluster row."""
    o = int(row['goff']); n = int(row['len'])
    return {k: strips[k][o:o + n] for k in STRIP_KEYS}


def add_derived(clusters, strips):
    """Vectorized derived per-cluster quantities:
      t_occ    drift-time occupancy = max(rsm)-min(lsm) [samples] (full-gap
               measure valid for inclined AND head-on)
      t_lo,t_hi absolute pulse-extent window [samples]
      wmean_pos amplitude-weighted centroid (== cen; kept for parity)
      headon    n<=3 and wfmax large
    """
    goff = clusters['goff'].values.astype(np.int64)
    order = np.argsort(goff, kind='stable')
    goff_s = goff[order]
    # segments are contiguous [goff, goff+len); reduceat over segment starts
    t_lo_s = np.minimum.reduceat(strips['lsm'], goff_s)
    t_hi_s = np.maximum.reduceat(strips['rsm'], goff_s)
    amp_max_s = np.maximum.reduceat(strips['amp'], goff_s)
    amp_sum_s = np.add.reduceat(strips['amp'], goff_s)
    q_hi_s = amp_max_s / np.maximum(amp_sum_s, 1e-9)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    t_lo = t_lo_s[inv]; t_hi = t_hi_s[inv]; q_hi = q_hi_s[inv]
    clusters = clusters.copy()
    clusters['t_lo'] = t_lo
    clusters['t_hi'] = t_hi
    clusters['t_occ'] = t_hi - t_lo
    clusters['qfrac_lead'] = q_hi
    return clusters


# ---------------------------------------------------------------------------
# realism gate (shared by 27b / 27c)
# ---------------------------------------------------------------------------
GATE = dict(
    extent_max=25.0,      # compact position extent [mm]
    n_lo=4, n_hi=20,      # inclined strip count
    dur_min=6.0,          # inclined: PEAK-TIME (max_sample) trail span [samples]
    mono_min=0.8,         # |trail monotonicity| corr(pos, max_sample)
    ho_n_max=3,           # head-on strip count
    ho_wfmax_min=9.0,     # head-on: min single-strip pulse width [samples]
    ho_dur_max=3.0,       # head-on: peaks bunched (small max_sample span)
    ho_t_occ_min=9.0,     # head-on: long pulse envelope [samples]
    iou_min=0.30,         # X/Y drift-window overlap
    dur_match_max=6.0,    # |t_occ_x - t_occ_y| [samples]
    fbal_pull_max=3.0,    # charge-balance pull (|f-f_med|/f_s68)
    f_med=0.49, f_s68=0.10,
)


def tag_tracklike(cl, gate=GATE):
    """Add boolean columns inclined / headon / tracklike to a clusters frame
    that already has add_derived() columns.  Full-gap traversal required.

    inclined: an ordered micro-TPC trail whose PEAK times (max_sample) span a
      good fraction of the gap (dur = smphi-smplo).  NB: use the peak-time span,
      NOT the pulse-envelope t_occ, which is inflated by wide/saturated single
      pulses at high gain and would admit prompt-but-wide blobs (tan~0).
    headon: few strips, peaks bunched (small dur) but a LONG saturated pulse
      envelope (wfmax / t_occ) = normal-incidence track drifting the full gap
      onto one strip.  (Purified against long-pulse junk by 27w waveforms.)"""
    inclined = ((cl['n'] >= gate['n_lo']) & (cl['n'] <= gate['n_hi']) &
                (cl['extent'] <= gate['extent_max']) &
                (cl['dur'] >= gate['dur_min']) &
                (cl['mono'].abs() >= gate['mono_min']))
    headon = ((cl['n'] <= gate['ho_n_max']) &
              (cl['wfmax'] >= gate['ho_wfmax_min']) &
              (cl['dur'] <= gate['ho_dur_max']) &
              (cl['t_occ'] >= gate['ho_t_occ_min']))
    cl = cl.copy()
    cl['inclined'] = inclined.fillna(False)
    cl['headon'] = headon.fillna(False)
    cl['tracklike'] = cl['inclined'] | cl['headon']
    return cl


def _iou(a0, a1, b0, b1):
    lo = max(a0, b0); hi = min(a1, b1)
    inter = max(0.0, hi - lo)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_xy(tl, gate=GATE, shuffle_y=False, seed=0):
    """Greedy best X<->Y pairing of track-like clusters within each
    (sub, eid, det).  Returns a DataFrame of accepted 3D segments.
    shuffle_y=True forms the accidental null by pairing each event's X
    clusters with another (random) event's Y clusters in the same det/sub."""
    rows = []
    rng = np.random.default_rng(seed)
    for (sub, det), g in tl.groupby(['sub', 'det']):
        xs_by = {e: sub_g for e, sub_g in g[g['pln'] == 0].groupby('eid')}
        ys_by = {e: sub_g for e, sub_g in g[g['pln'] == 1].groupby('eid')}
        xe = list(xs_by); ye = list(ys_by)
        if not xe or not ye:
            continue
        for e in xe:
            if shuffle_y:
                ee = ye[rng.integers(len(ye))]
            else:
                ee = e
                if ee not in ys_by:
                    continue
            xs = xs_by[e]; ys = ys_by[ee]
            best = None
            for _, rx in xs.iterrows():
                for _, ry in ys.iterrows():
                    io = _iou(rx['t_lo'], rx['t_hi'], ry['t_lo'], ry['t_hi'])
                    if io < gate['iou_min']:
                        continue
                    durd = abs(rx['t_occ'] - ry['t_occ'])
                    if durd > gate['dur_match_max']:
                        continue
                    f = rx['qsum'] / (rx['qsum'] + ry['qsum'])
                    pull = abs(f - gate['f_med']) / gate['f_s68']
                    if pull > gate['fbal_pull_max']:
                        continue
                    sc = io - 0.05 * durd - 0.1 * pull
                    cand = dict(sub=sub, det=det, detn=DET_NAMES[det], eid=int(e),
                                iou=io, durdiff=durd, fbal=f, pull=pull,
                                xcen=float(rx['cen']), ycen=float(ry['cen']),
                                xslope=float(rx['slope']), yslope=float(ry['slope']),
                                xn=int(rx['n']), yn=int(ry['n']),
                                qx=float(rx['qsum']), qy=float(ry['qsum']),
                                t_lo=min(rx['t_lo'], ry['t_lo']),
                                t_hi=max(rx['t_hi'], ry['t_hi']),
                                x_headon=bool(rx['headon']), y_headon=bool(ry['headon']),
                                resist_v=int(rx['resist_v']), score=sc)
                    if best is None or sc > best['score']:
                        best = cand
            if best:
                rows.append(best)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    cl, ss, ev = load_all()
    print('clusters', len(cl), 'strips', len(ss['pos']), 'events', len(ev))
    print(cl.groupby('detn')['pln'].count())
    cl = add_derived(cl, ss)
    print(cl[['n', 'extent', 'dur', 't_occ', 'mono', 'wfmax']].describe())
