#!/usr/bin/env python3
"""edge_study.py --colsum-root R --bundle-root B [--out results/edge_study.json] [--figs figures]
-- what the edge zone (< 30 mm from the active-area box) is, from the pass-1 waveform products (Paper I §5, D4).

The S4 re-check found the head-on stack T differs in the edge zone by -86 / +40 / +60 ns (det2/3/4) and left the cause open.
The head-on stack has a few hundred edge events; FA's column duration (colsum.py: the strip-sum box fit, every trigger, no
M3, no track model, no angle) has ~10x more and measures the same thing, T = local gap / local v.  Per golden row:

  1. T (mean of the two views; clean = both views >= 5 pulse strips, one cluster, unsaturated) against the distance d of
     the trigger's centroid to the nearest side of the active-area box, per side (x-low, x-high, y-low, y-high);
  2. the bow's prediction: the mesh-height surface h(U, W) of the golden bundle (wft.reco.bow_h, M3 z up: h > 0 = mesh
     nearer the cathode) gives a local gap g0 - h, so T_pred = T_c (1 - h / g0).  A one-parameter fit T = T_c (1 - c h / g0)
     over the whole plate: c = 1 if the whole dish is in the gap (cathode flat), c = 0 if the cathode follows the mesh;
  3. the residual after the fitted bow against d: whatever is left near the edge is not the dish (fringe field, cage);
  4. the same with the charge (summed charge q) and the pulse-strip count, to separate a gain or a footprint edge from a
     drift edge.
Trimmed means (20 %) with bootstrap errors.  Read-only.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
from common.mx17_active_area import TRUE_ACTIVE_BY_DET, STRIP_MAX_MM  # noqa: E402

GOLD = {  # golden row per chamber: run/subrun under the colsum and bundle roots
    'mx17_2': 'mx17_det2_det3_overnight_6-22-26/long_run',
    'mx17_3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V',
    'mx17_4': 'mx17_det4_day_6-24-26/long_run',
    'mx17_6': 'mx17_det6_det7_overnight_6-26-26/long_run',
    'mx17_7': 'mx17_det6_det7_overnight_6-26-26/long_run',
}
SHORT = {k: 'det' + k[-1] for k in GOLD}
G0 = 30.0            # nominal drift gap [mm] (T14 effective gap; only the ratio h / g0 enters)
NSIG_MIN = 5
TRIM = 0.2
D_EDGES = np.array([0, 5, 10, 15, 20, 30, 45, 60, 90, 130, 200])
# --phys (Paper I edge_turnon, 2026-11-03): profiles against the distance to the PHYSICAL edge of the strip plane, the
# box distance plus the measured passivation on the y sides, from 0 mm; centroids up to PHYS_SLACK outside the box are kept.
D_EDGES_PHYS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 130, 200])
PHYS_SLACK = 20.0
SIDES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
NB = 50


def tmean(a):
    a = np.sort(np.asarray(a, float)[np.isfinite(a)])
    k = int(TRIM * len(a))
    return float(np.mean(a[k:len(a) - k])) if len(a) > 2 * k + 5 else np.nan


def tmean_err(a, rng):
    a = np.asarray(a, float)[np.isfinite(a)]
    if len(a) < 20:
        return np.nan
    return float(np.std([tmean(a[rng.integers(0, len(a), len(a))]) for _ in range(NB)]))


def poly(c, U, W):
    return (c.get('gu', 0) * U + c.get('gw', 0) * W + c.get('kuu', 0) * U * U / 2 + c.get('kww', 0) * W * W / 2
            + c.get('kuw', 0) * U * W)


def table(path, det, bundle, dmin=-2.0):
    z = np.load(path, allow_pickle=True)
    one = lambda p: z[f'{p}_has'] & (z[f'{p}_nsig'] >= NSIG_MIN) & (z[f'{p}_ncl'] == 1) & np.isfinite(z[f'{p}_T'])
    ok = one('x') & one('y') & ~z['x_sat'] & ~z['y_sat']
    xc, yc = z['x_cen'].astype(float), z['y_cen'].astype(float)
    b = TRUE_ACTIVE_BY_DET[det]
    dd = np.stack([xc - b['x'][0], b['x'][1] - xc, yc - b['y'][0], b['y'][1] - yc])
    side = np.argmin(dd, axis=0); d = dd.min(axis=0)
    ok &= d >= dmin                                           # centroid inside the box (2 mm slack)
    U, W = (xc - 200.0) / 100.0, (yc - 200.0) / 100.0
    h = poly(bundle['bow']['h'], U, W)
    shift = np.array([0.0, 0.0, b['y'][0], STRIP_MAX_MM - b['y'][1]])      # physical edge = box edge - passivation
    dp = d + shift[side]
    return dict(dp=dp[ok], T=0.5 * (z['x_T'] + z['y_T'])[ok], Tx=z['x_T'][ok], Ty=z['y_T'][ok], q=0.5 * (z['x_q'] + z['y_q'])[ok],
                ns=0.5 * (z['x_nsig'] + z['y_nsig'])[ok].astype(float), d=d[ok], side=side[ok], h=h[ok], xc=xc[ok],
                yc=yc[ok], n_all=int(len(ok)))


def fit_bow(T, h, sel):
    """T = Tc (1 - c h / G0) by least squares on the trimmed core (|T - median| < 3 MAD), over `sel`."""
    t, hh = T[sel], h[sel]
    med = np.median(t); mad = 1.4826 * np.median(np.abs(t - med))
    k = np.abs(t - med) < 3 * mad
    A = np.c_[np.ones(k.sum()), -hh[k] / G0]
    (Tc, cTc), res, *_ = np.linalg.lstsq(A, t[k], rcond=None)
    cov = np.linalg.inv(A.T @ A) * (np.sum((t[k] - A @ [Tc, cTc]) ** 2) / max(k.sum() - 2, 1))
    c = cTc / Tc
    return dict(Tc=float(Tc), c=float(c), c_err=float(np.sqrt(cov[1, 1]) / Tc), n=int(k.sum()))


def profile(v, d, side, rng, sides=None, edges=None):
    edges = D_EDGES if edges is None else edges
    out = []
    for i in range(len(edges) - 1):
        s = (d >= edges[i]) & (d < edges[i + 1])
        if sides is not None:
            s &= np.isin(side, sides)
        out.append(dict(d_lo=float(edges[i]), d_hi=float(edges[i + 1]), n=int(s.sum()), m=tmean(v[s]),
                        e=tmean_err(v[s], rng)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--colsum-root', required=True); ap.add_argument('--bundle-root', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'edge_study.json'))
    ap.add_argument('--phys', action='store_true', help='distance to the physical edge, from 0 mm (writes the *_phys json)')
    ap.add_argument('--dets', nargs='*', default=None, help='golden-row keys to run (default all)')
    ap.add_argument('--figs', default=os.path.join(HERE, 'figures'))
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    res = {}
    for det, row in GOLD.items():
        if a.dets and det not in a.dets:
            continue
        cs = glob.glob(os.path.join(a.colsum_root, '**', row, det, 'fa', 'colsum.npz'), recursive=True)
        bj = glob.glob(os.path.join(a.bundle_root, '**', row, det, 'wft', 'calib_bundle', 'bundle.json'), recursive=True)
        if not cs or not bj:
            print(det, 'missing', cs, bj); continue
        B = json.load(open(bj[0]))
        C = table(cs[0], det, B, dmin=-PHYS_SLACK if a.phys else -2.0)
        Cw = C
        if a.phys:                          # every old field is computed on the old selection (centroid >= -2 mm); only prof_phys uses Cw
            k = C['d'] >= -2.0
            C = {kk: (v[k] if isinstance(v, np.ndarray) else v) for kk, v in Cw.items()}
        inn = C['d'] >= 30
        bow = fit_bow(C['T'], C['h'], np.ones(len(C['T']), bool))
        bow_in = fit_bow(C['T'], C['h'], inn)
        Tpred1 = bow['Tc'] * (1 - C['h'] / G0)                     # c = 1 (whole dish in the gap)
        resid = C['T'] - bow_in['Tc'] * (1 - bow_in['c'] * C['h'] / G0)
        r = dict(row=row, colsum=cs[0], n_clean=int(len(C['T'])), n_all=C['n_all'], v_bundle=B.get('v_drift'),
                 bow_fit_all=bow, bow_fit_interior=bow_in,
                 T_interior=tmean(C['T'][inn]), T_interior_err=tmean_err(C['T'][inn], rng),
                 T_edge=tmean(C['T'][~inn]), T_edge_err=tmean_err(C['T'][~inn], rng),
                 dT_edge=tmean(C['T'][~inn]) - tmean(C['T'][inn]),
                 dT_edge_pred_c1=tmean(Tpred1[~inn]) - tmean(Tpred1[inn]),
                 h_edge_minus_interior=float(np.mean(C['h'][~inn]) - np.mean(C['h'][inn])),
                 prof_T=profile(C['T'], C['d'], C['side'], rng),
                 prof_T_side={SIDES[s]: profile(C['T'], C['d'], C['side'], rng, [s]) for s in range(4)},
                 prof_resid=profile(resid, C['d'], C['side'], rng),
                 prof_resid_side={SIDES[s]: profile(resid, C['d'], C['side'], rng, [s]) for s in range(4)},
                 prof_Tx=profile(C['Tx'], C['d'], C['side'], rng), prof_Ty=profile(C['Ty'], C['d'], C['side'], rng),
                 prof_q=profile(C['q'], C['d'], C['side'], rng), prof_ns=profile(C['ns'], C['d'], C['side'], rng),
                 prof_h=profile(C['h'], C['d'], C['side'], rng))
        if a.phys:
            r['prof_T_side_phys'] = {SIDES[s]: profile(Cw['T'], Cw['dp'], Cw['side'], rng, [s], D_EDGES_PHYS) for s in range(4)}
            r['n_clean_phys'] = int(len(Cw['T']))
        # T map, 25 mm cells
        e = np.arange(0, 400.1, 25.0)
        ix, iy = np.digitize(C['xc'], e) - 1, np.digitize(C['yc'], e) - 1
        M = np.full((len(e) - 1, len(e) - 1), np.nan); Hm = np.full_like(M, np.nan); N = np.zeros_like(M)
        for i in range(len(e) - 1):
            for j in range(len(e) - 1):
                s = (ix == i) & (iy == j)
                N[i, j] = s.sum()
                if s.sum() >= 30:
                    M[i, j] = tmean(C['T'][s]); Hm[i, j] = np.mean(C['h'][s])
        r['map'] = dict(edges=e.tolist(), T=np.where(np.isfinite(M), M, -1).tolist(), h=np.where(np.isfinite(Hm), Hm, -99).tolist(),
                        n=N.tolist())
        res[SHORT[det]] = r
        print(f"{SHORT[det]}: n {r['n_clean']}  T_in {r['T_interior']:.1f}  dT_edge {r['dT_edge']:+.1f} ns  "
              f"bow c=1 predicts {r['dT_edge_pred_c1']:+.1f}  fitted c {bow_in['c']:.2f}+-{bow_in['c_err']:.2f} (interior) "
              f"{bow['c']:.2f}+-{bow['c_err']:.2f} (all)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    figures(res, a.figs)


def figures(res, outdir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    dets = list(res)
    fig, axs = plt.subplots(2, len(dets), figsize=(3.6 * len(dets), 6.4), squeeze=False)
    for k, d in enumerate(dets):
        r = res[d]
        for row, key, lab in ((0, 'prof_T_side', 'T [ns]'), (1, 'prof_resid_side', 'T - bow fit [ns]')):
            ax = axs[row, k]
            for s, P in r[key].items():
                c = [(p['d_lo'] + p['d_hi']) / 2 for p in P]; m = [p['m'] for p in P]; e = [p['e'] for p in P]
                ax.errorbar(c, m, yerr=e, fmt='.-', ms=3, lw=0.7, label=s.replace('_', ' '))
            if row == 0:
                P = r['prof_T']; c = [(p['d_lo'] + p['d_hi']) / 2 for p in P]
                pred = [r['bow_fit_interior']['Tc'] * (1 - r['bow_fit_interior']['c'] * p['m'] / G0) for p in r['prof_h']]
                ax.plot(c, pred, 'k--', lw=1, label='bow (fitted c)')
            else:
                ax.axhline(0, color='k', lw=0.6)
            ax.axvline(30, color='grey', ls=':', lw=0.8)
            ax.set_xscale('symlog', linthresh=10); ax.set_xlabel('distance to the active-area edge [mm]')
            ax.set_ylabel(lab); ax.grid(alpha=0.3)
            ax.set_title(f"{d}: c = {r['bow_fit_interior']['c']:.2f}" if row == 0 else f'{d}: after the bow', fontsize=9)
            if k == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'edge_T_profiles.png'), dpi=110); fig.savefig(os.path.join(outdir, 'edge_T_profiles.pdf')); plt.close(fig)
    fig, axs = plt.subplots(2, len(dets), figsize=(3.6 * len(dets), 6.6), squeeze=False)
    for k, d in enumerate(dets):
        m = res[d]['map']; e = m['edges']; T = np.array(m['T'], float); T[T < 0] = np.nan
        H = np.array(m['h'], float); H[H < -90] = np.nan
        Tc, c = res[d]['bow_fit_interior']['Tc'], res[d]['bow_fit_interior']['c']
        im = axs[0, k].imshow(T.T, origin='lower', extent=[e[0], e[-1], e[0], e[-1]], cmap='viridis')
        plt.colorbar(im, ax=axs[0, k], label='T [ns]'); axs[0, k].set_title(f'{d}: column duration', fontsize=9)
        im = axs[1, k].imshow((T - Tc * (1 - c * H / G0)).T, origin='lower', extent=[e[0], e[-1], e[0], e[-1]], cmap='RdBu_r', vmin=-40, vmax=40)
        plt.colorbar(im, ax=axs[1, k], label='T - bow [ns]'); axs[1, k].set_title(f'{d}: after the bow', fontsize=9)
        for ax in axs[:, k]:
            ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'edge_T_maps.png'), dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
