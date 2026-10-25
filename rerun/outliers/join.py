#!/usr/bin/env python3
"""join.py <run_key> --table events.parquet --alignment alignment.json --out joined.parquet

One row per event of a finalize table (wft, every plane, NO cluster-quality cut so the rejected fits are kept and
flagged), joined with
  ref_*   the M3 reference exactly as 03_angles.py builds it (chi2 < 1, NClus = 4, one good ray per event):
          tangents rotated into the raw strip frame, the position at the mesh in raw strip coordinates
          (ref_mesh_*), and the aligned-frame residual of the fitted position (res_*)
  m3_*    every ray the tracker wrote for that evn (no cut): number of rays, number passing the recipe, the best
          ray's chi2 / NClus per view, and the spread of the alternative rays at the detector
  h*_     the combined hits of the plane (candidate finding / QA only, never geometry): hit-strip clusters
          (gap > 2 strips splits), the largest and second cluster's summed amplitude, whether the reference
          and the fitted p0 sit on a cluster, the saturated-hit count
Angles: d_<v> = atan(tan_fit) - atan(tan_ref) [deg]; the ladder-scaled residual is recomputed in classify.py.
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
import awkward as ak  # noqa: E402
import cosmic_micro_tpc_analysis as cm  # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles  # noqa: E402
from wft import compat  # noqa: E402
from wft.io import strip_position_map  # noqa: E402


def all_rays(ray_dir, z):
    """evn -> per-event summary of every ray (no cut)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ray_dir, '*_rays.root'))):
        a = uproot.open(f)['T'].arrays(['evn', 'Z_Up', 'Z_Down', 'X_Up', 'X_Down', 'Y_Up', 'Y_Down',
                                        'Chi2X', 'Chi2Y', 'NClusX', 'NClusY'], library='ak')
        for e in a:
            n = len(e['Chi2X'])
            if n == 0:
                out[int(e['evn'])] = dict(m3_nray=0, m3_ngood=0)
                continue
            cx, cy = np.asarray(e['Chi2X']), np.asarray(e['Chi2Y'])
            nx, ny = np.asarray(e['NClusX']), np.asarray(e['NClusY'])
            t = (z - e['Z_Up']) / (e['Z_Down'] - e['Z_Up'])
            xs = np.asarray(e['X_Up']) + t * (np.asarray(e['X_Down']) - np.asarray(e['X_Up']))
            ys = np.asarray(e['Y_Up']) + t * (np.asarray(e['Y_Down']) - np.asarray(e['Y_Up']))
            good = (cx < M3_CHI2_CUT) & (cy < M3_CHI2_CUT) & (nx >= M3_MIN_NCLUS) & (ny >= M3_MIN_NCLUS)
            b = int(np.argmin(cx + cy + 1e3 * ~good))
            sep = np.hypot(xs - xs[b], ys - ys[b]); sep[b] = np.nan
            out[int(e['evn'])] = dict(m3_nray=n, m3_ngood=int(good.sum()), m3_chi2x=float(cx[b]),
                                      m3_chi2y=float(cy[b]), m3_nclx=int(nx[b]), m3_ncly=int(ny[b]),
                                      m3_alt_sep=float(np.nanmin(sep)) if n > 1 else np.nan)
    return out


def hit_features(cfg, eids):
    """Per event and plane: clusters of hit strips (QA / candidate finding only)."""
    pmap = strip_position_map(cfg)
    pitch = {}
    for feu, p in pmap.items():
        q = np.sort(p[np.isfinite(p)]); pitch[feu] = float(np.median(np.diff(q)))
    want = set(int(e) for e in eids); rows = {}
    for f in sorted(glob.glob(os.path.join(cfg.combined_hits_dir, '*.root'))):
        a = uproot.open(f)['hits'].arrays(['eventId', 'channel', 'amplitude', 'feu', 'saturated'], library='pd')
        a = a[a.eventId.isin(want)]
        for eid, g in a.groupby('eventId'):
            r = rows.setdefault(int(eid), {})
            for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
                h = g[g.feu == feu]
                pos = pmap[feu][h.channel.to_numpy()]
                ok = np.isfinite(pos); pos = pos[ok]; amp = h.amplitude.to_numpy()[ok]
                r[f'h{v}_n'] = len(pos); r[f'h{v}_nsat'] = int(h.saturated.to_numpy()[ok].sum())
                if len(pos) == 0:
                    continue
                o = np.argsort(pos); pos, amp = pos[o], amp[o]
                cut = np.flatnonzero(np.diff(pos) > 2.5 * pitch[feu]) + 1
                cl = [(pos[s].min(), pos[s].max(), amp[s].sum(), len(s))
                      for s in np.split(np.arange(len(pos)), cut)]
                cl.sort(key=lambda c: -c[2])
                r[f'h{v}_ncl'] = len(cl)
                r[f'h{v}_ncl2'] = sum(c[3] >= 2 for c in cl)
                r[f'h{v}_q1'] = cl[0][2]; r[f'h{v}_q2'] = cl[1][2] if len(cl) > 1 else 0.0
                r[f'h{v}_w1'] = cl[0][3]
                r[f'h{v}_lo1'], r[f'h{v}_hi1'] = cl[0][0], cl[0][1]
                r[f'h{v}_cl'] = json.dumps([[round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 1), int(c[3])] for c in cl[:8]])
    return rows, pitch


def dist_to_clusters(p, cl_json):
    if not isinstance(cl_json, str) or not np.isfinite(p):
        return np.nan, np.nan
    cl = json.loads(cl_json); best, bq = np.inf, np.nan
    for lo, hi, q, n in cl:
        d = 0.0 if lo <= p <= hi else min(abs(p - lo), abs(p - hi))
        if d < best:
            best, bq = d, q
    return best, bq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key'); ap.add_argument('--table', required=True); ap.add_argument('--alignment', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = get_config(a.run_key)
    params = cm.load_alignment(a.alignment)
    df = compat.load_table(a.table, max_dropped=None)
    df = df[~df.event_id.duplicated(keep=False)].reset_index(drop=True)
    results = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    R = []
    for r in results:
        d = dict(event_id=r.event_id, ref_x=r.ref_x_mm, ref_y=r.ref_y_mm, det_xa=r.det_x_aligned_mm,
                 det_ya=r.det_y_aligned_mm, ref_mesh_x=r.ref_mesh_x_mm, ref_mesh_y=r.ref_mesh_y_mm)
        if np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_mesh_x_mm):
            d['ref_tan_x'], d['ref_tan_y'] = cm._rotate_ref_tangents(r, params)
        R.append(d)
    R = pd.DataFrame(R)
    out = df.merge(R, on='event_id', how='left')
    out['res_x'] = out.det_xa - out.ref_x; out['res_y'] = out.det_ya - out.ref_y
    for v in 'xy':
        out[f'd_{v}'] = np.degrees(np.arctan(out[f'{v}_tan_theta']) - np.arctan(out[f'ref_tan_{v}']))
    m3 = all_rays(cfg.m3_tracking_dir, params.z_mean)
    M = pd.DataFrame([dict(event_id=k, **v) for k, v in m3.items()])
    out = out.merge(M, on='event_id', how='left')
    hits, pitch = hit_features(cfg, out.event_id.to_numpy())
    H = pd.DataFrame([dict(event_id=k, **v) for k, v in hits.items()])
    out = out.merge(H, on='event_id', how='left')
    for v in 'xy':
        dd = [dist_to_clusters(p, c) for p, c in zip(out[f'ref_mesh_{v}'], out.get(f'h{v}_cl'))]
        out[f'h{v}_dref'] = [x[0] for x in dd]; out[f'h{v}_qref'] = [x[1] for x in dd]
        dd = [dist_to_clusters(p, c) for p, c in zip(out[f'{v}_p0'], out.get(f'h{v}_cl'))]
        out[f'h{v}_dfit'] = [x[0] for x in dd]
    out.attrs = {}
    out.to_parquet(a.out)
    json.dump(dict(run_key=a.run_key, table=a.table, alignment=a.alignment, pitch={str(k): v for k, v in pitch.items()},
                   m3_recipe=f'chi2<{M3_CHI2_CUT} NClus>={M3_MIN_NCLUS}'),
              open(a.out.replace('.parquet', '.meta.json'), 'w'), indent=1)
    n = out.ref_tan_x.notna().sum()
    print(f'{a.run_key}: {len(out)} events, {n} with a reference -> {a.out}')


if __name__ == '__main__':
    main()
