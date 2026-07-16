#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m3_cut_tradeoff.py -- efficiency, spatial resolution AND temporal resolution vs
the M3 reference-track chi2 cut, plus the statistics cost, on ONE figure.

Motivation (M3_CUT_AND_ACTIVE_AREA_NOTE.md): the detector's apparent performance
is limited by the M3 REFERENCE tracker, not the chamber -- tightening the M3 chi2
keeps sharpening the position residual with no plateau inside chi2<5. This tool
turns that observation into a continuous trade-off curve for all three headline
paper numbers at once, so an operating point (the chi2<1.0 & NClus=4 recipe) can be
read off against its statistics cost.

Design (clean + "fine-adjust"-friendly):
  * ONE heavy pass per run-key builds a per-event table keyed by eventId and caches
    it to  m3_cut_tradeoff_data_<KEY>.npz  (M3 chi2x/y, NClusx/y, theta, crossing
    category, spatial residual r, and the plane-to-plane leading-edge dt for timing).
  * EVERY cut scan after that is a pure in-memory re-cut of the cache -- instant --
    so a finer chi2 grid costs nothing. Rebuild only with --rebuild (or when the
    upstream alignment / event_results cache changes).

Metrics vs the cut  T = max(Chi2X, Chi2Y) <= T  (both planes), NClus = min(NClusX,
NClusY) >= N:
  * EFFICIENCY (<=R mm) = reco_near / (all in-box active-area crossings passing the
    cut)  -- same denominator (sparks included) as 09_efficiency_breakdown.py.
  * SPATIAL core sigma = robust std of |r| (|r|<15 mm) over reco crossings passing
    the cut  -- same estimator as m3_cut_scan.py / 09.
  * TEMPORAL single-plane sigma_t = s68(dt_lead)/sqrt(2), walk-corrected, over
    dual-plane cluster events whose M3 track passes the cut  -- script 42's headline
    detector time resolution, here joined to the reference quality by eventId.
  * STATISTICS: % of crossings retained + absolute counts (denominator / reco /
    timing) vs the cut.

Usage:
  ../../.venv/bin/python m3_cut_tradeoff.py                     # fleet default
  ../../.venv/bin/python m3_cut_tradeoff.py g_det3_wknd --fine  # dense grid, 1 det
  ../../.venv/bin/python m3_cut_tradeoff.py sat_det3 --grid 5,2,1,0.5,0.3
  ../../.venv/bin/python m3_cut_tradeoff.py g_det4 --rebuild    # force re-extract
Flags: --fine (0.2..5 in 0.1 steps) | --grid a,b,c | --nclus N,M (families, def 3,4)
       | --R=5 | --veto=50 | --rebuild
"""
import os
import sys
import json
import pickle
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_config import get_config, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions
import awkward as ak
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = {'reco_near': 0, 'reco_far': 1, 'spark': 2, 'hit_no_reco': 3, 'no_hit': 4}

# --- fleet default: one golden key per physical detector (see qa_config) ----------
FLEET = ['g_det3_wknd', 'o22_long_det2', 'g_det4', 'g_det6_long', 'g_det7_long']
LABELS = {'g_det3_wknd': 'det3 (A)', 'sat_det3': 'det3 (A, sat)',
          'o22_long_det2': 'det2 (B)', 'g_det4': 'det4 (E)',
          'g_det6_long': 'det6 (C)', 'g_det7_long': 'det7 (D)'}
COLORS = {'g_det3_wknd': '#1f4e8c', 'sat_det3': '#5b8fd6', 'o22_long_det2': '#2e8b57',
          'g_det4': '#8e44ad', 'g_det6_long': '#e67e22', 'g_det7_long': '#c0392b'}

# --- timing (matches 42_time_resolution.py) --------------------------------------
V_DRIFT = 34.30          # um/ns
AMP_THR = 100.0
T_WINDOW = (0.0, 2000.0)
MIN_STRIPS = 3
GAP_MM = getattr(cm, 'GAP_THRESHOLD_MM', 12.0)


def rstd(v, ns=3, it=5):
    """Iterated 3-sigma-clipped std -- the 09 / m3_cut_scan core-sigma estimator."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size < 3:
        return float('nan')
    for _ in range(it):
        m, s = np.median(v), np.std(v); k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10:
            break
        v = v[k]
    return float(np.std(v))


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size < 3:
        return float('nan')
    q = np.percentile(a, [16, 84])
    return 0.5 * (q[1] - q[0])


def _largest_cluster(pos):
    o = np.argsort(pos)
    breaks = np.where(np.diff(pos[o]) > GAP_MM)[0]
    return max(np.split(o, breaks + 1), key=len)


# ============================================================ build (heavy, cached)
def build(KEY, R, VETO):
    """One pass over M3 rays + detector hits -> per-event trade-off table (cached)."""
    CFG = get_config(KEY)
    print(f'[build {KEY}] DET={CFG.DET_NAME}  RUN={CFG.RUN}/{CFG.SUB_RUN}')

    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, 'alignment_tpc_veto50',
                                            'alignment.json'))
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', 'event_results.pkl'), 'rb'))
    # chi2_cut=5.0 is the SCAN CEILING (records the loose sample so we can re-cut
    # tighter post-hoc); NClus stays at the class default so the cache holds every
    # track down to NClus>=3 and the scan can apply NClus>=N itself.
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=5.0)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)

    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in res
            if r.has_both and np.isfinite(r.det_x_aligned_mm)
            and np.isfinite(r.det_y_aligned_mm)}

    # per-event M3 reference-track quality (ray_data flat: 1 track/event)
    m3_evn = ak.to_numpy(rays.ray_data['evn']).astype(int)
    m3_cx = ak.to_numpy(rays.ray_data['Chi2X']).astype(float)
    m3_cy = ak.to_numpy(rays.ray_data['Chi2Y']).astype(float)
    m3_nx = ak.to_numpy(rays.ray_data['NClusX']).astype(float)
    m3_ny = ak.to_numpy(rays.ray_data['NClusY']).astype(float)
    xang, yang, _ = get_xy_angles(rays.ray_data)
    tth = np.hypot(np.tan(np.asarray(xang)), np.tan(np.asarray(yang)))
    m3_th = np.degrees(np.arctan(tth))
    q_by_id = {int(e): (m3_cx[i], m3_cy[i], m3_nx[i], m3_ny[i], m3_th[i])
               for i, e in enumerate(m3_evn)}

    # ------- detector hits: multiplicity (spark), fired-set, AND plane t_lead -------
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                            expressions=['eventId', 'feu', 'channel', 'amplitude',
                                         'time'], library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)].copy()
    mult_by_ev = df.groupby('eventId').size().to_dict()   # raw strips (spark tag / veto)
    det_hit = set(int(e) for e in df['eventId'].unique())
    det_lo, det_hi = int(df['eventId'].min()), int(df['eventId'].max())

    # plane-to-plane leading-edge dt on veto'd, in-window, above-threshold clusters
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    hpe = df.groupby('eventId')['eventId'].transform('size')
    dft = df[hpe <= VETO].copy()
    dft = cm._map_strip_positions(dft, det)
    dft = dft[(dft['time'] > T_WINDOW[0]) & (dft['time'] < T_WINDOW[1])
              & (dft['amplitude'] > AMP_THR)].copy()
    t_eid, t_dt, t_asym = [], [], []
    planes = {'x': 'x_position_mm', 'y': 'y_position_mm'}
    for eid, g in dft.groupby('eventId', sort=False):
        rec = {}
        for p, pcol in planes.items():
            gp = g[g[pcol].notna()]
            if len(gp) < MIN_STRIPS:
                continue
            idx = _largest_cluster(gp[pcol].to_numpy())
            if len(idx) < MIN_STRIPS:
                continue
            gc = gp.iloc[idx]
            rec[p] = (float(gc['time'].to_numpy().min()),
                      float(gc['amplitude'].to_numpy().sum()))
        if 'x' in rec and 'y' in rec:
            qx, qy = rec['x'][1], rec['y'][1]
            t_eid.append(int(eid)); t_dt.append(rec['x'][0] - rec['y'][0])
            t_asym.append((qx - qy) / (qx + qy) if (qx + qy) else np.nan)

    # ------- per in-box active-area crossing (any category) with M3 quality + r -----
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr); py = np.array(yr)
    evn = [int(v) for v in evn]

    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])
    box = [float(ax0), float(ax1), float(ay0), float(ay1)]

    C = {k: [] for k in ('eid', 'chi2x', 'chi2y', 'nclx', 'ncly', 'theta', 'cat', 'r')}
    for e, x, y in zip(evn, px, py):
        if e < det_lo or e > det_hi:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1 and ay0 <= y <= ay1):
            continue
        q = q_by_id.get(e, (np.nan,) * 5)
        r = np.nan
        if mult_by_ev.get(e, 0) > VETO:
            cat = 'spark'
        elif e in reco:
            dxp, dyp = reco[e]
            r = float(np.hypot(x - dxp, y - dyp))
            cat = 'reco_far' if r > R else 'reco_near'
        elif e in det_hit:
            cat = 'hit_no_reco'
        else:
            cat = 'no_hit'
        C['eid'].append(e); C['chi2x'].append(q[0]); C['chi2y'].append(q[1])
        C['nclx'].append(q[2]); C['ncly'].append(q[3]); C['theta'].append(q[4])
        C['cat'].append(CAT[cat]); C['r'].append(r)

    NPZ = os.path.join(HERE, f'm3_cut_tradeoff_data_{KEY}.npz')
    np.savez(NPZ, box=np.array(box), R=R, VETO=VETO, det_lo=det_lo, det_hi=det_hi,
             cat_codes=json.dumps(CAT),
             t_eid=np.array(t_eid, int), t_dt=np.array(t_dt, float),
             t_asym=np.array(t_asym, float),
             **{k: np.array(v, float if k != 'eid' and k != 'cat' else int)
                for k, v in C.items()})
    print(f'[build {KEY}] {len(C["eid"]):,} in-box crossings, '
          f'{len(t_eid):,} dual-plane timing events -> {os.path.basename(NPZ)}')
    return NPZ


def load(KEY, R, VETO, rebuild):
    NPZ = os.path.join(HERE, f'm3_cut_tradeoff_data_{KEY}.npz')
    if rebuild or not os.path.exists(NPZ):
        build(KEY, R, VETO)
    return np.load(NPZ, allow_pickle=True)


# ============================================================ scan (instant re-cut)
def scan(d, cuts, nclus):
    """Return dict of per-cut metric arrays for one NClus>=nclus family."""
    cx, cy = d['chi2x'], d['chi2y']
    ncl = np.minimum(d['nclx'], d['ncly'])
    cmax = np.maximum(cx, cy)
    cat, r = d['cat'], d['r']
    spark = CAT['spark']; near = CAT['reco_near']
    is_reco = (cat == near) | (cat == CAT['reco_far'])

    # timing: build eid->(dt,asym); one global walk correction (like script 42)
    t_eid, t_dt, t_asym = d['t_eid'], d['t_dt'], d['t_asym']
    # walk slope from the full timing sample, binned in charge asymmetry
    ae = np.linspace(-0.6, 0.6, 13); ac, am = [], []
    for lo, hi in zip(ae[:-1], ae[1:]):
        m = (t_asym >= lo) & (t_asym < hi)
        if m.sum() > 40:
            ac.append(0.5 * (lo + hi)); am.append(np.median(t_dt[m]))
    walk = float(np.polyfit(ac, am, 1)[0]) if len(ac) > 2 else 0.0
    t_dt_corr = t_dt - walk * np.where(np.isfinite(t_asym), t_asym, 0.0)
    # map eventId -> (dt_corr) for the join; and per-eid chi2/nclus already on crossings
    q_by_eid = {int(e): (cmax[i], ncl[i]) for i, e in enumerate(d['eid'])}

    out = dict(cut=[], eff=[], eff_err=[], sig=[], sigt_ns=[], sigt_mm=[],
               n_denom=[], n_reco=[], n_time=[], frac=[], walk=walk)
    n_total = int(len(cat))
    for T in cuts:
        keep = (cmax <= T) & (ncl >= nclus)
        ndenom = int(keep.sum())
        rn = int((keep & (cat == near)).sum())
        eff = rn / ndenom if ndenom else np.nan
        # binomial error on the ratio
        err = np.sqrt(eff * (1 - eff) / ndenom) if ndenom else np.nan
        rr = r[keep & is_reco]
        sig = rstd(rr[rr < 15])
        n_reco = int(np.isfinite(rr).sum())
        # timing subset: events whose M3 track passes the same cut
        tk = np.array([q_by_eid.get(int(e), (np.inf, -1))[0] <= T
                       and q_by_eid.get(int(e), (np.inf, -1))[1] >= nclus
                       for e in t_eid]) if len(t_eid) else np.array([], bool)
        st = s68(t_dt_corr[tk]) / np.sqrt(2.0) if tk.any() else np.nan
        out['cut'].append(float(T)); out['eff'].append(100 * eff)
        out['eff_err'].append(100 * err); out['sig'].append(sig)
        out['sigt_ns'].append(st); out['sigt_mm'].append(st * V_DRIFT / 1000.0)
        out['n_denom'].append(ndenom); out['n_reco'].append(n_reco)
        out['n_time'].append(int(tk.sum())); out['frac'].append(100 * ndenom / n_total)
    for k in out:
        if k != 'walk':
            out[k] = np.array(out[k], float)
    return out


# ============================================================ plot
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    keys = args if args else FLEET
    rebuild = '--rebuild' in sys.argv
    R = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--R=')), 5.0)
    VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
    if '--grid' in sys.argv:
        cuts = [float(x) for x in sys.argv[sys.argv.index('--grid') + 1].split(',')]
    elif '--fine' in sys.argv:
        cuts = list(np.round(np.arange(5.0, 0.19, -0.1), 2))
    else:
        cuts = [5, 3, 2, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
    cuts = sorted(cuts, reverse=True)
    nclus_arg = next((a.split('=')[1] for a in sys.argv if a.startswith('--nclus=')), '3,4')
    nclus_fams = sorted(int(x) for x in nclus_arg.split(','))

    tag = '_'.join(keys) if len(keys) <= 2 else 'fleet'
    print(f'keys={keys}  cuts={cuts}  nclus_families={nclus_fams}  R={R}')

    results = {}
    for KEY in keys:
        d = load(KEY, R, VETO, rebuild)
        results[KEY] = {N: scan(d, cuts, N) for N in nclus_fams}

    # ---- figure: 2x2 (efficiency / spatial sigma / temporal sigma / statistics) ----
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10.2))
    (a_eff, a_sig), (a_time, a_stat) = ax
    N_primary = max(nclus_fams)                 # solid = strict (recipe) family
    ls_by_n = {N: ('-' if N == N_primary else '--') for N in nclus_fams}
    mk_by_n = {N: ('o' if N == N_primary else 's') for N in nclus_fams}

    for KEY in keys:
        col = COLORS.get(KEY, None); lab = LABELS.get(KEY, KEY)
        for N in nclus_fams:
            s = results[KEY][N]
            nl = f'{lab}' + ('' if N == N_primary else f' (NClus≥{N})')
            a_eff.errorbar(s['cut'], s['eff'], yerr=s['eff_err'], color=col,
                           ls=ls_by_n[N], marker=mk_by_n[N], ms=4, lw=1.5,
                           capsize=2, label=nl if N == N_primary else None)
            a_sig.plot(s['cut'], s['sig'], color=col, ls=ls_by_n[N],
                       marker=mk_by_n[N], ms=4, lw=1.5,
                       label=nl if N == N_primary else None)
            a_time.plot(s['cut'], s['sigt_ns'], color=col, ls=ls_by_n[N],
                        marker=mk_by_n[N], ms=4, lw=1.5,
                        label=nl if N == N_primary else None)
            a_stat.plot(s['cut'], s['frac'], color=col, ls=ls_by_n[N],
                        marker=mk_by_n[N], ms=4, lw=1.5,
                        label=nl if N == N_primary else None)

    for a in (a_eff, a_sig, a_time, a_stat):
        a.axvline(1.0, color='k', ls=':', lw=0.9)          # recipe operating point
        a.set_xlabel('M3 reference cut  max(Chi2X, Chi2Y) ≤ T')
        a.grid(alpha=0.3); a.invert_xaxis()                # stricter cut to the right
    a_eff.set_ylabel('efficiency (≤%g mm) [%%]' % R)
    a_eff.set_title('(a) efficiency vs M3 reference cut')
    a_sig.set_ylabel('spatial core σ(|r|<15) [mm]')
    a_sig.set_title('(b) spatial resolution — keeps sharpening (reference-limited)')
    # temporal: 0-anchored so the ~flat, detector-limited trend reads honestly;
    # twin axis gives the drift-equivalent σ_z. Set ylim BEFORE building the twin.
    a_time.set_ylabel('temporal σ_t single-plane, walk-corr [ns]')
    a_time.set_title('(c) temporal resolution — detector-limited (≈flat vs cut)')
    _thi = np.nanmax([np.nanmax(results[K][N]['sigt_ns'])
                      for K in keys for N in nclus_fams])
    a_time.set_ylim(0, _thi * 1.18)
    a_t2 = a_time.twinx()
    a_t2.set_ylim(0, _thi * 1.18 * V_DRIFT / 1000.0)
    a_t2.set_ylabel('drift-equivalent σ_z [mm]')
    a_stat.set_ylabel('statistics retained [% of χ²<5 crossings]')
    a_stat.set_title('(d) statistics cost of the cut')
    a_stat.axhline(43, color='grey', ls=':', lw=0.8)
    a_stat.text(a_stat.get_xlim()[0], 44, '~43% at recipe (det3)', fontsize=7,
                color='grey', va='bottom', ha='left')

    # linestyle key (NClus family) — proxy handles, second legend on panel (a)
    from matplotlib.lines import Line2D
    style_handles = [Line2D([0], [0], color='0.3', ls=ls_by_n[N], marker=mk_by_n[N],
                            ms=4, label=('NClus≥%d (recipe)' % N if N == N_primary
                                         else 'NClus≥%d' % N)) for N in nclus_fams]
    leg2 = a_eff.legend(handles=style_handles, fontsize=7.5, loc='lower left',
                        title='line style', title_fontsize=7.5)
    a_eff.add_artist(leg2)
    a_eff.legend(fontsize=8, ncol=2, loc='upper left')

    fig.suptitle('M3 reference-cut trade-off: efficiency / spatial σ / temporal σ / '
                 'statistics   (vertical line = χ²≤1.0 operating point;  '
                 'solid = NClus≥%d, dashed = NClus≥%d)'
                 % (N_primary, min(nclus_fams)), fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT = os.path.join(HERE, f'm3_cut_tradeoff_{tag}.png')
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'wrote {OUT}')

    # ---- json dump (machine-readable, for tables / re-quoting) ----
    dump = {KEY: {f'nclus{N}': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                for k, v in results[KEY][N].items()}
                  for N in nclus_fams} for KEY in keys}
    JOUT = os.path.join(HERE, f'm3_cut_tradeoff_{tag}.json')
    json.dump(dump, open(JOUT, 'w'), indent=2)
    print(f'wrote {JOUT}')

    # ---- console summary at the recipe point (chi2<=1.0, NClus>=primary) ----
    print(f'\n== at recipe point χ²≤1.0 & NClus≥{N_primary} ==')
    ci = int(np.argmin(np.abs(np.array(cuts) - 1.0)))
    print(f'{"det":<14}{"eff%":>8}{"σ_x[mm]":>9}{"σt[ns]":>8}{"σz[mm]":>8}'
          f'{"kept%":>7}{"N_den":>8}{"N_time":>8}')
    for KEY in keys:
        s = results[KEY][N_primary]
        print(f'{LABELS.get(KEY, KEY):<14}{s["eff"][ci]:>8.1f}{s["sig"][ci]:>9.2f}'
              f'{s["sigt_ns"][ci]:>8.1f}{s["sigt_mm"][ci]:>8.2f}{s["frac"][ci]:>7.1f}'
              f'{int(s["n_denom"][ci]):>8d}{int(s["n_time"][ci]):>8d}')


if __name__ == '__main__':
    main()
