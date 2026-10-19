#!/usr/bin/env python3
"""
attachment_geometric.py — is charge lost with drift depth? A time-free test.

On an inclined track every strip collects the ionisation of one depth slice of
the same thickness (pitch / |tan|), so with no attachment the charge per strip
is flat along the track. Attachment makes it fall as exp(-z / lambda) toward
the cathode.

Here the depth of each strip comes from GEOMETRY ONLY: the M3 reference line
(aligned into the raw strip frame exactly as build_cache.py does) gives

    u = (strip position - reference position) / tan_ref        [mm along z]

so a contained track is a plateau in u, between the mesh and the cathode edge.
No drift time, drift velocity, sharing kernel or fitted track enters the depth.
Strip charge is the raw waveform (pedestal + common-mode subtracted, wft.io):
both the peak amplitude and the positive-sample integral are recorded.
Sharing moves charge only +-2 strips, which smooths the plateau edges and
leaves its interior untouched.

Per event the strip charges are normalised by that event's own interior mean,
which removes Landau fluctuations and the 1/|tan| slice-thickness factor; the
stack is then the median normalised charge per u bin. Peak timing is used for
one thing only: telling which plateau edge is the mesh (early charge).

The fit is a straight line in log(charge) over the plateau interior, which
gives 1/lambda directly. The hits-era claim to test is lambda = 15-18 mm
(PAPER_STATUS.md sec. 7, DET3_WEEKEND_ANALYSIS.md sec. 4) — a factor 0.3 over
20 mm.

    ../../.venv/bin/python mx_june_wft/drift_consistency/attachment_geometric.py sat_det3
    ../../.venv/bin/python mx_june_wft/drift_consistency/attachment_geometric.py g_det4
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

import qa_config                                         # noqa: E402
from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS          # noqa: E402
qa_config.setup_paths()
import cosmic_micro_tpc_analysis as cm                   # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft import reco as wr                               # noqa: E402
from wft import seed as wseed                            # noqa: E402
from wft import io as wio                                # noqa: E402

DATA = '/media/dylan/data/x17/cosmic_bench'
OUT = os.path.join(DATA, 'Analysis', 'drift_consistency', 'attachment')
# run key -> (data subdir on the data disk, alignment dir under OUT_BASE)
RUNS = {'sat_det3': ('det3', 'wft/alignment_lp'),
        'g_det4': ('det4_day', 'wft/alignment')}
TAN_LO, TAN_HI = 0.20, 0.70
U_BINS = np.arange(-45.0, 45.01, 1.0)


def config(key):
    c = qa_config.get_config(key)
    sub, al = RUNS[key]
    cfg = qa_config._Config(key, c.RUN, c.SUB_RUN, c.MX17_FEUS, c.DET_PLANE_Z,
                            base_path=os.path.join(DATA, sub) + '/',
                            det_name=c.DET_NAME)
    return cfg, os.path.join(cfg.OUT_BASE, al, 'alignment.json')


def m3_truth(cfg, align_path):
    """Same transform as mx_june_wft/bench/build_cache.py."""
    params = cm.load_alignment(align_path)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    ya = np.array(ya)
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    th = np.deg2rad(params.theta_deg)
    ct, st = np.cos(th), np.sin(th)
    u = px - params.centre_x - params.x_offset
    v = py - params.centre_y - params.y_offset
    rx = params.centre_x + ct * u + st * v
    ry = params.centre_y - st * u + ct * v
    tan = {int(e): (ct * np.tan(tx) + st * np.tan(ty), -st * np.tan(tx) + ct * np.tan(ty))
           for e, tx, ty in zip(an, xa, ya)}
    return {int(e): dict(x=float(a), y=float(b), tx=tan.get(int(e), (np.nan,) * 2)[0],
                         ty=tan.get(int(e), (np.nan,) * 2)[1])
            for e, a, b in zip(evn, rx, ry)}


def collect(cfg, truth, limit):
    pos_maps = wio.strip_position_map(cfg)
    hits = wr._load_hits(cfg)
    seeds = wseed.seeds_from_hits(hits, pos_maps, cfg.MX17_FEU_X, cfg.MX17_FEU_Y)
    del hits
    wanted = {e for e in set(seeds) & set(truth)
              if not seeds[e]['spark'] and (seeds[e]['x'] or seeds[e]['y'])}
    wanted = {e for e in wanted
              if TAN_LO <= abs(truth[e]['tx']) <= TAN_HI
              or TAN_LO <= abs(truth[e]['ty']) <= TAN_HI}
    if limit:
        wanted = set(sorted(wanted)[:limit])
    print(f'{len(wanted):,} inclined M3-matched events to read')
    rows = {'x': [], 'y': []}
    for payloads in wr._stream_windows(cfg, pos_maps, seeds, wanted, pad_strips=4):
        for eid, wins, _sd, _nh, _sp, _ft in payloads:
            t = truth[eid]
            for plane, ref, tan in (('x', t['x'], t['tx']), ('y', t['y'], t['ty'])):
                if plane not in wins or not (TAN_LO <= abs(tan) <= TAN_HI):
                    continue
                w = max(wins[plane], key=lambda d: float(d['W'].max()))
                W, pos, noise = w['W'], w['pos'].astype(float), w['noise']
                peak = W.max(axis=1)
                integ = np.clip(W, 0, None).sum(axis=1)
                tpk = W.argmax(axis=1).astype(float)
                rows[plane].append(dict(eid=eid, tan=tan, u=(pos - ref) / tan,
                                        peak=peak, integ=integ, tpk=tpk,
                                        sig=peak / noise))
    return rows


def orientation(rows):
    """+1 if depth z = u increases toward the cathode: per event, the peak
    sample of the significant strips must rise with u (charge from deeper
    arrives later). Per-event slopes, so trigger jitter between events cannot
    dilute it. Used only for the sign, never as a velocity."""
    sl = []
    for r in rows:
        m = r['sig'] > 8
        if m.sum() >= 5:
            sl.append(np.polyfit(r['u'][m], r['tpk'][m], 1)[0])
    sl = np.array(sl)
    return (1.0 if np.median(sl) > 0 else -1.0), float(np.median(sl)), float(np.mean(sl > 0))


def profile(rows, qkey, sgn):
    """Stacked, per-event-normalised charge vs depth z = sgn * u."""
    zc = 0.5 * (U_BINS[1:] + U_BINS[:-1])

    def stack(lo, hi):
        stacks = [[] for _ in zc]
        for r in rows:
            z = sgn * r['u']
            q = r[qkey]
            core = (z > lo) & (z < hi)
            if core.sum() < 3 or q[core].mean() <= 0:
                continue
            qn = q / q[core].mean()
            idx = np.digitize(z, U_BINS) - 1
            for i, val in zip(idx, qn):
                if 0 <= i < len(zc):
                    stacks[i].append(val)
        med = np.array([np.median(s) if len(s) > 30 else np.nan for s in stacks])
        err = np.array([1.2533 * np.std(s) / np.sqrt(len(s)) if len(s) > 30 else np.nan
                        for s in stacks])
        return med, err, np.array([len(s) for s in stacks])

    # pass 1 with a generous core, pass 2 normalised on the measured plateau
    med, _, _ = stack(5.0, 20.0)
    lo, hi = half_max_edges(zc, med)
    med, err, n = stack(lo + 4, hi - 4)
    lo, hi = half_max_edges(zc, med)
    return dict(z=zc, med=med, err=err, n=n, z50_lo=lo, z50_hi=hi,
                column_fwhm=hi - lo)


def half_max_edges(z, med):
    ok = np.isfinite(med)
    z, m = z[ok], med[ok]
    plateau = np.median(m[(m > 0.5 * np.nanmax(m))])
    half = 0.5 * plateau
    above = np.where(m > half)[0]
    i0, i1 = above[0], above[-1]
    lo = np.interp(half, [m[i0 - 1], m[i0]], [z[i0 - 1], z[i0]])
    hi = np.interp(half, [m[i1 + 1], m[i1]], [z[i1 + 1], z[i1]])
    return float(lo), float(hi)


def fit_lambda(p, margin=4.0):
    sel = ((p['z'] > p['z50_lo'] + margin) & (p['z'] < p['z50_hi'] - margin)
           & np.isfinite(p['med']))
    z, y, e = p['z'][sel], np.log(p['med'][sel]), p['err'][sel] / p['med'][sel]
    A = np.vstack([np.ones_like(z), z]).T
    Wt = 1 / e ** 2
    cov = np.linalg.inv(A.T @ (A * Wt[:, None]))
    b = cov @ (A.T @ (Wt * y))
    chi2 = float(np.sum(Wt * (y - A @ b) ** 2))
    # inflate the error when the straight line is not a perfect description
    s = max(1.0, np.sqrt(chi2 / max(len(z) - 2, 1)))
    k, dk = b[1], np.sqrt(cov[1, 1]) * s
    return dict(inv_lambda=float(-k), inv_lambda_err=float(dk),
                z_range=[float(z.min()), float(z.max())], chi2=chi2, ndf=len(z) - 2,
                ratio_over_20mm=float(np.exp(k * 20)),
                lambda_95cl_lower=float(1 / max(-k + 1.645 * dk, 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', choices=sorted(RUNS))
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    cfg, align = config(args.run_key)
    os.makedirs(OUT, exist_ok=True)
    truth = m3_truth(cfg, align)
    print(f'{len(truth):,} M3 tracks')
    rows = collect(cfg, truth, args.limit)
    res = dict(run_key=args.run_key, alignment=align)
    for plane in ('x', 'y'):
        sgn, tslope, fpos = orientation(rows[plane])
        res[plane] = {'n_tracks': len(rows[plane]), 'sign': sgn,
                      'peak_sample_per_mm': tslope, 'frac_positive': fpos}
        for qkey in ('peak', 'integ'):
            p = profile(rows[plane], qkey, sgn)
            f = fit_lambda(p)
            res[plane][qkey] = dict(fit=f, **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                              for k, v in p.items()})
            print(f"{plane} {qkey:5s} n={len(rows[plane])} column {p['z50_lo']:.1f}..{p['z50_hi']:.1f} mm (FWHM {p['column_fwhm']:.1f})  "
                  f"1/lambda = {f['inv_lambda']*1e3:+.2f} +- {f['inv_lambda_err']*1e3:.2f} /m  "
                  f"q(20mm)/q(0) = {f['ratio_over_20mm']:.3f}  lambda > {f['lambda_95cl_lower']:.0f} mm (95% CL)")
    out = os.path.join(OUT, f'attachment_{args.run_key}.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
