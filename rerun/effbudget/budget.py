#!/usr/bin/env python3
"""budget.py <run_key> <det> --table events.parquet --alignment alignment.json --out DIR [--r 5 3]

The efficiency loss budget (Dylan 10-25: "if 95 % efficient, exactly where did the other 5 % go?"): every M3 ray that
crosses the chamber, followed to the one place it ends up. Same footing as mx_june_wft/02_efficiency.py (M3 recipe
chi2 < 1 and NClus = 4, detection from hits, the waveform fit for the point, incomplete events out of the
denominator), with three changes:
  - the area is the MEASURED active area (common/mx17_active_area.TRUE_ACTIVE_BY_DET) in the raw strip frame, not a
    percentile box of the reconstructed points, and every ray carries its region: interior (>= 30 mm from the edge of
    an unpassivated coordinate, >= 8 mm from a passivated one), edge band, or outside (acceptance, reported apart);
  - one good M3 ray per event (as 03_angles): events with several recipe rays are counted apart;
  - every loss is split by cause, first match wins (below); the far ones carry the C22 class (classify.py) when the
    event was in that study.

Per M3 ray inside the active area, X+Y (|r| <= R in the aligned frame), first match:
  detected        X and Y fitted (no cluster-quality cut), |r| <= R, n_dropped <= 2 in both views
  rejected_cut    as detected, but a view has n_dropped > 2 (competing clusters): found, removed by the quality cut
  spark           > SPARK_VETO_HITS strips above the significance floor (a discharge, not a muon)
  no_hit          no strip fired in either view
  hw_known        the ray sits on known hardware: det7 X card 1 (x < 199.3 mm), det4 dead / low-gain band
  one_view_silent one view has no strip at all
  charge_elsewhere a view has strips but none within 5 mm of the M3 footprint (M3 not on what MX17 saw, or an
                  accidental with the muon below threshold)
  not_seeded      charge at the footprint in both views, but the event / a view is not in the fit table
  fit_failed      in the table, a view's fit not ok
  far             fitted in both views, |r| > R  (split by C22 class: MULTI, DELTA, EDGE, POS_TAIL, FIT, FIT_SAT, ...)
Per view the same ladder with the 1D residual |res_v| <= R and that view's own hits.
Writes <out>/rays_<det>.parquet (one row per ray: region, category per R, per-view status) and budget_<det>.json.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import uproot  # noqa: E402
import cosmic_micro_tpc_analysis as cm  # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft import compat  # noqa: E402
from wft.io import strip_position_map  # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS  # noqa: E402
from common.mx17_fiducial import ray_regions, good_x, DET7_X_CARD1_MM  # noqa: E402

NEAR = 5.0
OUT = os.path.expanduser('~/x17/cosmic_bench/outliers')
LADDER = ['detected', 'rejected_cut', 'spark', 'no_hit', 'hw_known', 'one_view_silent', 'charge_elsewhere',
          'not_seeded', 'fit_failed', 'far']


def rays_table(cfg, params):
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xs, _, ex = get_xy_positions(rays.ray_data, params.z_x)
    _, ys, ey = get_xy_positions(rays.ray_data, params.z_y)
    xa, ya, ea = get_xy_angles(rays.ray_data)
    X = pd.DataFrame(dict(event_id=np.asarray(ex, int), ref_x=params.ref_x_sign * np.asarray(xs, float)))
    Y = pd.DataFrame(dict(event_id=np.asarray(ey, int), ref_y=np.asarray(ys, float)))
    A = pd.DataFrame(dict(event_id=np.asarray(ea, int), tx=np.tan(params.ref_x_sign * np.asarray(xa, float)),
                          ty=np.tan(np.asarray(ya, float))))
    n_multi = int(X.event_id.duplicated(keep=False).sum())
    for T in (X, Y, A):
        T.drop_duplicates('event_id', keep=False, inplace=True)
    R = X.merge(Y, on='event_id').merge(A, on='event_id')
    # raw strip frame (inverse of the alignment transform, as attach_reference_positions / _rotate_ref_tangents)
    th = np.deg2rad(params.theta_deg); c, s = np.cos(th), np.sin(th); cx, cy = params.centre_x, params.centre_y
    u = R.ref_x - cx - params.x_offset; v = R.ref_y - cy - params.y_offset
    R['mesh_x'] = cx + c * u + s * v; R['mesh_y'] = cy - s * u + c * v
    R['tan_x'] = c * R.tx + s * R.ty; R['tan_y'] = -s * R.tx + c * R.ty
    return R, n_multi


def regions(R, det):
    """Fiducial region of every ray (common.mx17_fiducial: the one definition 03_angles / 04_maps use too)."""
    F = ray_regions(R.mesh_x.to_numpy(), R.mesh_y.to_numpy(), R.tan_x.to_numpy(), R.tan_y.to_numpy(), det)
    for k in ('lo_x', 'hi_x', 'lo_y', 'hi_y', 'region_x', 'region_y', 'region', 'dist_x', 'dist_y'):
        R[k] = F[k]
    R['good_x'] = good_x(det, F['lo_x'])
    return R


def hits_per_view(cfg, R):
    """Per event and view: hit count, distance of the nearest hit cluster to the footprint [lo, hi], and the spark
    multiplicity (significance floor as 02_efficiency). Hits are candidate finding / QA only."""
    pm = strip_position_map(cfg)
    pitch = {f: float(np.median(np.diff(np.sort(p[np.isfinite(p)])))) for f, p in pm.items()}
    fs = sorted(glob.glob(os.path.join(cfg.combined_hits_dir, '*.root')))
    fs = [f for f in fs if '_datrun_' in f]
    D = uproot.concatenate([f'{f}:hits' for f in fs], expressions=['eventId', 'feu', 'channel', 'amplitude',
                                                                    'significance'], library='pd')
    D = D[D.feu.isin(cfg.MX17_FEUS)]
    lo_e, hi_e = int(D.eventId.min()), int(D.eventId.max())
    fl = cm.apply_significance_floor(D, rel=SIG_REL_FLOOR)
    mult = fl.groupby('eventId').size()
    want = set(R.event_id)
    D = D[D.eventId.isin(want)]
    fp = R.set_index('event_id')[['lo_x', 'hi_x', 'lo_y', 'hi_y']]
    rows = {}
    for (eid, feu), g in D.groupby(['eventId', 'feu']):
        v = 'x' if feu == cfg.MX17_FEU_X else 'y'
        pos = np.sort(pm[feu][g.channel.to_numpy()]); pos = pos[np.isfinite(pos)]
        r = rows.setdefault(int(eid), {})
        r[f'h{v}_n'] = len(pos)
        if not len(pos):
            continue
        cut = np.flatnonzero(np.diff(pos) > 2.5 * pitch[feu]) + 1
        lo, hi = fp.at[eid, f'lo_{v}'], fp.at[eid, f'hi_{v}']
        dmin = min(0.0 if (c.max() >= lo - 1e-9 and c.min() <= hi + 1e-9) else min(abs(c.min() - hi), abs(c.max() - lo))
                   for c in np.split(pos, cut))
        r[f'h{v}_dfp'] = dmin
    H = pd.DataFrame([dict(event_id=k, **v) for k, v in rows.items()])
    return H, mult, (lo_e, hi_e)


def not_measured(cfg):
    p = os.path.join(os.path.dirname(cfg.combined_hits_dir.rstrip('/')), 'incomplete_events.json')
    s = set()
    if os.path.exists(p):
        for per in json.load(open(p)).values():
            for feu, ids in per.items():
                if int(feu) in cfg.MX17_FEUS:
                    s.update(int(i) for i in ids)
    return s, os.path.exists(p)


def hw_mask(R, short):
    hw = np.zeros(len(R), bool); hwv = {'x': np.zeros(len(R), bool), 'y': np.zeros(len(R), bool)}
    if short == 'det7':
        hwv['x'] = (R.lo_x < DET7_X_CARD1_MM).to_numpy(); hw = hwv['x'].copy()
    if short == 'det4':
        # dead / low-gain bands along x from the C22 per-event charge ratio, mapped onto position (2 mm bins)
        J = pd.read_parquet(os.path.join(OUT, 'joined_det4.parquet'))[['event_id', 'ref_mesh_x']]
        L = pd.read_parquet(os.path.join(OUT, 'an', 'live_det4.parquet')).merge(J, on='event_id').dropna()
        b = np.arange(0, 402, 2.0)
        med = L.groupby(np.digitize(L.ref_mesh_x, b)).live.median()
        dead = set(med.index[med < 0.3])
        hw = np.isin(np.digitize(R.mesh_x, b), list(dead)); hwv['x'] = hwv['y'] = hw
    return hw, hwv


HW_SOURCE = {'mx17_7': f'X card 1: footprint x < {DET7_X_CARD1_MM} mm (ch 0-255)',
             'det7': f'X card 1: footprint x < {DET7_X_CARD1_MM} mm (ch 0-255)',
             'det4': 'dead / low-gain bands, 2 mm bins with C22 live ratio < 0.3 (OUT/an/live_det4.parquet, pre-pass)'}


def cand_arms(R, path, radii):
    """Per ray, from the fit's candidates sidecar (rank >= 0: every candidate production ranked), three ways to pick one candidate per view:
      prod   rank 0 (what the table holds)
      rule   the trigger-track rule (Dylan 10-28: a downstream selection): the most vertical candidate, smallest
             |tan_theta|, ties to production's rank. Parameter-free, uses only det's own fit; the cosmic trigger's
             muon is steeper-limited than the knock-on electrons it makes in the material above (multiparticle/
             choose.py: 'most_vertical'). Not for beam data.
      near   the candidate nearest the M3 footprint (uses the reference: multi-track CREDIT, never a selection)
      posonly rank 0 WITHOUT the fit-sanity cut (|tan| <= TAN_SANE, p0 on the plate; compat.apply_fit_sanity): a
             position-only detection for rows whose angle scale is broken (skeptical det3 6-23: folded v 1.7-3.5 um/ns
             puts 5-12 % of right-position fits at |tan| > 3, which the table then counts as fit_failed)
    Every pick but posonly is made among SANE candidates (the same cut the table's ok carries), so prod reproduces the
    budget's 'detected' exactly (prod_closure_mismatch = 0).
    det_<arm>_R: both views' picks within R (raw-frame hypot) with n_dropped <= MAX_DROPPED on the picked candidate
    (not for 'any'); n_cand_v: sane ranked candidates in view v."""
    C = pd.read_parquet(path)
    # every ranked candidate (rank >= 0): the 'plausible' flag is the beam-tuned gate and is False on ~16 % of the
    # bench's production winners (multitrack.py: it misses ~35 % of clean bench singles), so it is not used here
    C = C[C['rank'] >= 0]
    C = C[C.event_id.isin(set(R.event_id))].drop_duplicates(['event_id', 'plane', 'rank'])
    M = R.set_index('event_id')[['mesh_x', 'mesh_y']]
    C = C.join(M, on='event_id')
    C['res'] = np.where(C.plane == 'x', C.p0 - C.mesh_x, C.p0 - C.mesh_y)
    C['ares'] = C.res.abs(); C['atan'] = C.tan_theta.abs()
    raw0 = C[C['rank'] == 0]
    C = C[np.isfinite(C.tan_theta) & (C.atan <= compat.TAN_SANE) & np.isfinite(C.p0) & C.p0.between(*compat.P0_SANE_MM)]
    picks = {'rule': C.sort_values(['event_id', 'plane', 'atan', 'rank']).groupby(['event_id', 'plane']).head(1),
             'near': C.sort_values(['event_id', 'plane', 'ares', 'rank']).groupby(['event_id', 'plane']).head(1),
             'prod': C[C['rank'] == 0], 'posonly': raw0}
    n = C.groupby(['event_id', 'plane']).size().unstack().reindex(columns=['x', 'y'])
    for v in 'xy':
        R[f'n_cand_{v}'] = R.event_id.map(n[v]).fillna(0).astype(int)
    for arm, P in picks.items():
        P = P.set_index(['event_id', 'plane'])
        for v in 'xy':
            Q = P.xs(v, level='plane')
            R[f'{arm}_res_{v}'] = R.event_id.map(Q.res)
            R[f'{arm}_nd_{v}'] = R.event_id.map(Q.n_dropped)
            R[f'{arm}_rank_{v}'] = R.event_id.map(Q['rank'])
        r = np.hypot(R[f'{arm}_res_x'], R[f'{arm}_res_y']).to_numpy()
        nd = ((R[f'{arm}_nd_x'].fillna(99) <= compat.MAX_DROPPED) & (R[f'{arm}_nd_y'].fillna(99) <= compat.MAX_DROPPED)).to_numpy()
        for Rmm in radii:
            key = 'any' if arm == 'near' else arm
            R[f'det_{key}_{Rmm:g}'] = np.nan_to_num(r, nan=1e9) <= Rmm
            if arm != 'near':
                R[f'det_{key}_{Rmm:g}'] &= nd
    return True


def cand_summary(U, radii):
    """How often each pick is the M3 particle on the views with >= 2 particle candidates where the M3 particle is a
    candidate (its nearest candidate within R in that view), and the event-level detection with each pick."""
    out = {}
    for Rmm in radii:
        e = dict(n=int(len(U)))
        for v in 'xy':
            m = (U[f'n_cand_{v}'] >= 2) & (U[f'near_res_{v}'].abs() <= Rmm)
            e[f'multi_views_{v}'] = int(m.sum())
            e[f'prod_right_{v}'] = int((m & (U[f'near_rank_{v}'] == 0)).sum())
            e[f'rule_right_{v}'] = int((m & (U[f'rule_rank_{v}'] == U[f'near_rank_{v}'])).sum())
        e['frac_events_multi'] = float(((U.n_cand_x >= 2) | (U.n_cand_y >= 2)).mean()) if len(U) else np.nan
        for arm in ('rule', 'any', 'prod', 'posonly'):
            e[f'detected_{arm}'] = float(100 * U[f'det_{arm}_{Rmm:g}'].mean()) if len(U) else np.nan
        # closure: rank 0 of the sidecar through the same rule must reproduce the budget's 'detected'
        e['prod_closure_mismatch'] = int((U[f'det_prod_{Rmm:g}'] != (U[f'cat_xy_{Rmm:g}'] == 'detected')).sum())
        out[f'{Rmm:g}'] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key'); ap.add_argument('det', help='short name: det2 det3 det4 det6 det7')
    ap.add_argument('--table', required=True); ap.add_argument('--alignment', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--r', type=float, nargs='+', default=[5.0, 3.0])
    ap.add_argument('--classified', default=None, help='C22 classified_<det>.parquet made on this table (far classes)')
    ap.add_argument('--cand', default=None, help='the table\'s events.candidates.parquet: multi-candidate accounting, '
                                                  'the trigger-track rule and the multi-track credit (cand_arms)')
    a = ap.parse_args()
    cfg = get_config(a.run_key); det = cfg.DET_NAME
    params = cm.load_alignment(a.alignment)
    R, n_multi = rays_table(cfg, params)
    R = regions(R, det)
    H, mult, (lo_e, hi_e) = hits_per_view(cfg, R)
    nm, have_nm = not_measured(cfg)
    n_all = len(R)
    R = R[(R.event_id >= lo_e) & (R.event_id <= hi_e)]
    n_range = len(R)
    R['not_measured'] = R.event_id.isin(nm)
    R = R.merge(H, on='event_id', how='left')
    for v in 'xy':
        R[f'h{v}_n'] = R[f'h{v}_n'].fillna(0).astype(int)
    R['mult'] = R.event_id.map(mult).fillna(0).astype(int)

    df = compat.load_table(a.table, max_dropped=None)
    dup = df.event_id.duplicated(keep=False)
    df = df[~dup]
    # finalize tables are the first --limit matched events (make_pass_dag: 8000): rays past the table's last event
    # were never offered to the fit, so they leave the denominator (n_beyond_table)
    n_beyond = int((R.event_id > df.event_id.max()).sum())
    R = R[R.event_id <= df.event_id.max()]
    res = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    cm.attach_reference_positions(res, rays, params, params.ref_x_sign * np.array(xa), an)
    P = pd.DataFrame([dict(event_id=r.event_id, det_xa=r.det_x_aligned_mm, det_ya=r.det_y_aligned_mm) for r in res])
    T = df[['event_id', 'x_ok', 'y_ok', 'x_n_dropped', 'y_n_dropped', 'x_p0', 'y_p0']].merge(P, on='event_id')
    R = R.merge(T, on='event_id', how='left')
    R['in_table'] = R.x_ok.notna()
    for v in 'xy':
        R[f'{v}_ok'] = R[f'{v}_ok'].astype('boolean').fillna(False).astype(bool)
        # per view in the RAW strip frame (olib.rres): on a chamber rotated ~90 deg to M3 the aligned x is the
        # detector's Y view, so an aligned-frame per-view residual mixes the two planes
        R[f'res_{v}'] = R[f'{v}_p0'] - R[f'mesh_{v}']
    R['r_al'] = np.hypot(R.det_xa - R.ref_x, R.det_ya - R.ref_y)   # 02_efficiency's |r| (aligned frame)
    R['r'] = np.hypot(R.res_x, R.res_y)                             # the same distance, raw frame
    R['hw'], hwv = hw_mask(R, a.det)
    for v in 'xy':
        R[f'hw_{v}'] = hwv[v]

    # C22 far classes: only from a classification made on THIS table (--classified); a file from another table would
    # put that table's classes on these rays (before 2026-10-31 it was picked up implicitly from OUT)
    C = a.classified
    if C and os.path.exists(C):
        K = pd.read_parquet(C)[['event_id', 'cls', 'outlier']]
        R = R.merge(K[K.outlier][['event_id', 'cls']], on='event_id', how='left')
    else:
        R['cls'] = np.nan

    def ladder(R, Rmm, views):
        cat = np.full(len(R), '', dtype=object)
        nd_ok = np.ones(len(R), bool); fit_ok = np.ones(len(R), bool); silent = np.zeros(len(R), bool)
        elsewhere = np.zeros(len(R), bool); hw = np.zeros(len(R), bool)
        for v in views:
            nd_ok &= (R[f'{v}_n_dropped'].fillna(99) <= compat.MAX_DROPPED).to_numpy()
            fit_ok &= R[f'{v}_ok'].to_numpy()
            silent |= (R[f'h{v}_n'] == 0).to_numpy()
            elsewhere |= ((R[f'h{v}_n'] > 0) & ~(R[f'h{v}_dfp'] <= NEAR)).to_numpy()
            hw |= R[f'hw_{v}'].to_numpy()
        dist = R.r.to_numpy() if len(views) == 2 else np.abs(R[f'res_{views[0]}'].to_numpy())
        near = fit_ok & (dist <= Rmm)
        nohit = np.all([(R[f'h{v}_n'] == 0).to_numpy() for v in views], axis=0)
        rules = [('detected', near & nd_ok), ('rejected_cut', near), ('spark', (R.mult > SPARK_VETO_HITS).to_numpy()),
                 ('no_hit', nohit), ('hw_known', hw), ('one_view_silent', silent), ('charge_elsewhere', elsewhere),
                 ('not_seeded', ~R.in_table.to_numpy()), ('fit_failed', ~fit_ok), ('far', np.ones(len(R), bool))]
        for name, m in rules:
            cat[(cat == '') & m] = name
        return cat

    for Rmm in a.r:
        R[f'cat_xy_{Rmm:g}'] = ladder(R, Rmm, 'xy')
        for v in 'xy':
            R[f'cat_{v}_{Rmm:g}'] = ladder(R, Rmm, v)

    S_cand = cand_arms(R, a.cand, a.r) if a.cand else None

    os.makedirs(a.out, exist_ok=True)
    R.drop(columns=[c for c in R.columns if R[c].dtype == object and c.startswith('h') and c.endswith('_cl')],
           errors='ignore').to_parquet(os.path.join(a.out, f'rays_{a.det}.parquet'))
    # summary: denominators per region group, every category in % of the in-area, measured rays
    M = R[~R.not_measured]
    groups = {'active': M.region != 'outside', 'interior': M.region == 'interior',
              'edge': ~M.region.isin(['interior', 'outside']),
              'interior_live': (M.region == 'interior') & ~M.hw,
              'interior_good': (M.region == 'interior') & M.good_x}
    S = dict(run_key=a.run_key, det=a.det, table=a.table, alignment=a.alignment, classified=C, cand=a.cand,
             hw_map=HW_SOURCE.get(a.det), n_rays_recipe=n_all,
             n_events_multi_ray=n_multi, n_in_event_range=n_range, n_beyond_table=n_beyond, n_not_measured=int(R.not_measured.sum()),
             incomplete_list=have_nm, n_outside=int((M.region == 'outside').sum()), radii=a.r, budget={})
    for g, m in groups.items():
        U = M[m]
        if not len(U):
            continue
        for Rmm in a.r:
            for v in ('xy', 'x', 'y'):
                col = f'cat_{v}_{Rmm:g}'
                vc = U[col].value_counts()
                ent = {k: float(100 * vc.get(k, 0) / len(U)) for k in LADDER}
                far = U[U[col] == 'far'].cls.fillna('unclassified').value_counts()
                ent['far_by_cls'] = {k: float(100 * n / len(U)) for k, n in far.items()}
                ent['n'] = int(len(U))
                if v == 'xy' and S_cand is not None:
                    for arm in ('rule', 'any', 'posonly'):
                        ent[f'detected_{arm}'] = float(100 * U[f'det_{arm}_{Rmm:g}'].sum() / len(U))
                S['budget'][f'{g}|{v}|{Rmm:g}'] = ent
    if S_cand is not None:
        S['cand'] = {g: cand_summary(M[m], a.r) for g, m in groups.items() if m.any()}
    json.dump(S, open(os.path.join(a.out, f'budget_{a.det}.json'), 'w'), indent=1)
    b = S['budget']
    for g in groups:
        if f'{g}|xy|5' not in b:
            continue
        e = b[f'{g}|xy|5']
        print(f"{a.det} {g:9s} n {e['n']:6d}  " + '  '.join(f'{k} {e[k]:.2f}' for k in LADDER if e[k] > 0))


if __name__ == '__main__':
    main()
