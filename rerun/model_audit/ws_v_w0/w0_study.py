#!/usr/bin/env python3
"""w0_study.py -- what the zero-angle offset w0 is, and where it comes from (F9 of the B4 model audit).

The forward model puts each 60 ns drift slice k at transverse position p0 + w * u_k (u_k = time since t0), so w is
the lateral speed of the landing point [um/ns] and the track angle is tan = (w - w0) / (kw * v).  w0 is therefore
the lateral speed the fit reports for a track that the M3 telescope says is perpendicular: an angle offset
dtheta = w0 / v.

Three classes of cause predict different signatures:

  A  geometric: the chamber's mesh plane is tilted against the telescope by slope s = dz/dp.  The fit sees
     tan_fit = tan_ref + s  ->  w0 = s * v (a constant ANGLE at every drift field), AND the mesh height varies
     across the chamber, so the position residual at the mesh grows with tan * p:  r_p = tan_p * (dz0 + s_p p + s_q q).
     The second effect is measured from the positions alone, independent of w.
  B  field-line tilt (non-parallel cathode, fringe fields): electrons drift along a tilted field, w0/v constant in a
     linear geometry, but the mesh is level, so no tan*p term in the residuals.
  C  time-domain model mismatch (an unmodelled one-sided late neighbour signal, a skewed kernel): a fixed time shift
     towards one neighbour is a fixed w0 in um/ns, so the ANGLE grows as 1/v when the field is lowered.

Part 1: the offset per row with an estimator that does not mix the scale in.  set_w0kw.py takes
        w0 = median(w - v_bundle * tan_ref); when kw != 1 (a 500 V row read with the 1000 V v) that picks up
        (kw - 1) * v * <tan_ref>, so here the offset is the intercept of a robust line w = a + b * tan_ref.
Part 2: chamber tilt from the position residuals (golden rows): r_p = a + b tan_p + c tan_p (p - pbar) + d tan_p (q - qbar).
        c is the mesh slope along p, the tilt that enters view p's angle; d is the slope along q (enters view q).
Outputs: <root>/ws_v_w0/w0_study.json, figures in rerun/model_audit/ws_v_w0/figures/.
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import theilslopes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
WS = os.path.join(ROOT, 'ws_v_w0')
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
SLOT_Z = {'det2': 702, 'det3': 702, 'det4': 702, 'det6': 232, 'det7': 702}
GAP_MM = 30.0
RNG = np.random.default_rng(9)

# det3 rows: (label, per_event.npz, drift voltage)
DET3_ROWS = [('sat 1000 V (golden)', f'{ROOT}/det3/final_physmask/angles/per_event.npz', 1000),
             ('sat 700 V', f'{WS}/dryrun/d700/vrefit/angles/per_event.npz', 700),
             ('sat 500 V', f'{WS}/dryrun/d500/vrefit/angles/per_event.npz', 500),
             ('6-25 500 V', f'{WS}/dryrun/day625/vrefit/angles/per_event.npz', 500),
             ('sat 1000 V, mesh 460 V', f'{WS}/dryrun/h460/angles/per_event.npz', 1000),
             ('sat 1000 V, mesh 480 V', f'{WS}/dryrun/h480/angles/per_event.npz', 1000)]


def huber_line(x, y, k=1.345, it=30):
    """Huber IRLS straight line y = a + b x (robust to the far tail, ~Theil-Sen at a fraction of the cost)."""
    X = np.column_stack([np.ones_like(x), x]); wt = np.ones_like(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        u = np.abs(r) / (k * s); wt = np.where(u <= 1, 1.0, 1.0 / u)
        nb = np.linalg.lstsq(X * np.sqrt(wt)[:, None], y * np.sqrt(wt), rcond=None)[0]
        if np.allclose(nb, beta, atol=1e-7):
            break
        beta = nb
    return beta


def line(w, tr, hi=0.30, nboot=200):
    """robust line w = a + b tan over |tan| < hi; returns a, b, offset angle a/b [deg] with bootstrap errors."""
    m = np.isfinite(w) & np.isfinite(tr) & (np.abs(tr) < hi)
    w, tr = w[m], tr[m]
    def fit(i):
        return tuple(huber_line(tr[i], w[i]))
    ts = theilslopes(w[:3000], tr[:3000])                    # cross-check of the Huber line (first 3000 events)
    a, b = fit(np.arange(len(w)))
    bs = np.array([fit(RNG.integers(0, len(w), len(w))) for _ in range(nboot)])
    ang = np.degrees(np.arctan(bs[:, 0] / bs[:, 1]))
    return dict(n=int(len(w)), a=float(a), a_err=float(bs[:, 0].std()), b=float(b), b_err=float(bs[:, 1].std()),
                off_deg=float(np.degrees(np.arctan(a / b))), off_deg_err=float(ang.std()),
                a_theilsen_3000=float(ts[1]), b_theilsen_3000=float(ts[0]))


def setw0(w, tr, v):
    s = np.isfinite(w) & np.isfinite(tr) & (np.abs(tr) < 0.30)
    return float(np.median(w[s] - v * tr[s]))


def part1():
    out = {}
    for det in DETS:
        z = np.load(f'{ROOT}/{det}/final_physmask/angles/per_event.npz')
        v = json.load(open(f'{ROOT}/{det}/calib_bundle/bundle.json'))['v_drift']
        out[det] = {p: dict(line(z[f'{p}_w'] * 1e3, z[f'{p}_tan_ref']), w0_setw0=setw0(z[f'{p}_w'] * 1e3, z[f'{p}_tan_ref'], v))
                    for p in 'xy'}
    rows = []
    for lab, f, vd in DET3_ROWS:
        z = np.load(f)
        meta = glob.glob(os.path.join(os.path.dirname(os.path.dirname(f)), 'events*.meta.json'))
        vb = json.load(open(meta[0]))['bundle']['v_drift'] if meta else 35.8
        r = dict(label=lab, drift_V=vd, E_Vcm=vd / (GAP_MM / 10), v_bundle=vb)
        for p in 'xy':
            r[p] = dict(line(z[f'{p}_w'] * 1e3, z[f'{p}_tan_ref']), w0_setw0=setw0(z[f'{p}_w'] * 1e3, z[f'{p}_tan_ref'], vb))
        rows.append(r)
        print(f"{lab:24s} n {r['x']['n']:5d}  v_line x {r['x']['b']:.1f} y {r['y']['b']:.1f}  "
              f"intercept x {r['x']['a']:+.3f}±{r['x']['a_err']:.3f} y {r['y']['a']:+.3f}±{r['y']['a_err']:.3f} um/ns  "
              f"angle x {r['x']['off_deg']:+.2f}±{r['x']['off_deg_err']:.2f} y {r['y']['off_deg']:+.2f}±{r['y']['off_deg_err']:.2f} deg  "
              f"[set_w0 x {r['x']['w0_setw0']:+.3f} y {r['y']['w0_setw0']:+.3f}]")
    return out, rows


def robust_lsq(X, y, clip=3.0, it=6):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    keep = m.copy()
    for _ in range(it):
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = m & (np.abs(r) < clip * s)
    r = y[keep] - X[keep] @ beta
    cov = np.linalg.inv(X[keep].T @ X[keep]) * np.var(r)
    return beta, np.sqrt(np.diag(cov)), int(keep.sum())


EXTRA = {'det3_d700': 'det3, sat 700 V', 'det3_d500': 'det3, sat 500 V', 'det3_day625': 'det3, 6-25 (bottom slot)'}


def part2():
    out = {}
    for det in DETS + list(EXTRA):
        t = pd.read_parquet(f'{WS}/{det}.parquet')
        R = {}
        for p in 'xy':                       # the angle offset of the same events, for the comparison
            ok = t[f'{p}_ok'].astype(bool).to_numpy()
            R[f'offset_{p}'] = line(t[f'{p}_w'].to_numpy()[ok] * 1e3, t[f'tan_ref_{p}'].to_numpy()[ok], nboot=100)
        for p, q in (('x', 'y'), ('y', 'x')):
            ok = t[f'{p}_ok'].astype(bool) & (np.abs(t[f'{p}_p0'] - t[f'ref_mesh_{p}']) < 5)
            s = t[ok]
            r = (s[f'{p}_p0'] - s[f'ref_mesh_{p}']).to_numpy()
            tp = s[f'tan_ref_{p}'].to_numpy()
            P = s[f'ref_mesh_{p}'].to_numpy(); Q = s[f'ref_mesh_{q}'].to_numpy()
            P = P - np.median(P); Q = Q - np.median(Q)
            X = np.column_stack([np.ones_like(r), tp, tp * P, tp * Q, P, Q])
            beta, err, n = robust_lsq(X, r)
            R[p] = dict(n=n, dz0_mm=float(beta[1]), dz0_err=float(err[1]),
                        slope_along_self=float(beta[2]), slope_along_self_err=float(err[2]),
                        slope_along_other=float(beta[3]), slope_along_other_err=float(err[3]))
        # each mesh slope is measured twice: s_x from view x (tan_x * x) and from view y (tan_y * x)
        for p, q in (('x', 'y'), ('y', 'x')):
            a, ea = R[p]['slope_along_self'], R[p]['slope_along_self_err']
            b, eb = R[q]['slope_along_other'], R[q]['slope_along_other_err']
            wgt = np.array([1 / ea ** 2, 1 / eb ** 2])
            R[f'mesh_slope_along_{p}'] = dict(from_view_self=a, from_view_other=b,
                                              combined=float((a * wgt[0] + b * wgt[1]) / wgt.sum()),
                                              combined_err=float(1 / np.sqrt(wgt.sum())),
                                              deg=float(np.degrees((a * wgt[0] + b * wgt[1]) / wgt.sum())))
        out[det] = R
        print(det, ' '.join(f"s_{p} {R[f'mesh_slope_along_{p}']['deg']:+.3f}±{np.degrees(R[f'mesh_slope_along_{p}']['combined_err']):.3f} deg "
                            f"(self {np.degrees(R[f'mesh_slope_along_{p}']['from_view_self']):+.3f}, other "
                            f"{np.degrees(R[f'mesh_slope_along_{p}']['from_view_other']):+.3f})" for p in 'xy'),
              f"dz0 x {R['x']['dz0_mm']:+.1f} y {R['y']['dz0_mm']:+.1f} mm  | angle offset "
              f"x {R['offset_x']['off_deg']:+.3f}±{R['offset_x']['off_deg_err']:.3f} y {R['offset_y']['off_deg']:+.3f}±{R['offset_y']['off_deg_err']:.3f}")
    xs = np.array([out[d][f'mesh_slope_along_{p}']['deg'] for d in out for p in 'xy'])
    ys = np.array([out[d][f'offset_{p}']['off_deg'] for d in out for p in 'xy'])
    print(f'tilt vs offset over {len(xs)} chamber-views: corr {np.corrcoef(xs, ys)[0, 1]:.3f}, '
          f'median |offset - tilt| {np.median(np.abs(ys - xs)):.3f} deg, rms offset {np.sqrt(np.mean(ys ** 2)):.3f} -> '
          f'rms after subtracting the tilt {np.sqrt(np.mean((ys - xs) ** 2)):.3f} deg')
    return out


def figures(p1, rows, p2):
    os.makedirs(FIG, exist_ok=True)
    # 1: what w0 is -- the zero-angle offset on det3 Y
    z = np.load(f'{ROOT}/det3/final_physmask/angles/per_event.npz')
    w, tr = z['y_w'] * 1e3, z['y_tan_ref']
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bins = np.linspace(-0.06, 0.06, 13); c = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(w[(tr >= a) & (tr < b)]) for a, b in zip(bins[:-1], bins[1:])]
    err = [1.2533 * 1.4826 * np.median(np.abs(w[(tr >= a) & (tr < b)] - m)) / np.sqrt(((tr >= a) & (tr < b)).sum())
           for (a, b), m in zip(zip(bins[:-1], bins[1:]), med)]
    L = p1['det3']['y']
    ax[0].errorbar(c, med, yerr=err, fmt='o', color='k', label='median fitted w (det3 Y)')
    xx = np.linspace(-0.06, 0.06, 2)
    ax[0].plot(xx, L['b'] * xx, '--', color='0.5', label='no offset: w = v·tan')
    ax[0].plot(xx, L['a'] + L['b'] * xx, '-', color='#C44E52', label=f"fit: w = {L['a']:+.2f} + {L['b']:.1f}·tan")
    ax[0].axhline(0, color='0.8', lw=.8); ax[0].axvline(0, color='0.8', lw=.8)
    ax[0].set_xlabel('M3 reference tan θ (Y view)'); ax[0].set_ylabel('fitted lateral speed w  [µm/ns]')
    ax[0].set_title('w0 = the fitted w of a track M3 calls perpendicular'); ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_xlim(-0.062, 0.062)
    # panel b: per chamber, angle offsets
    a = ax[1]
    for k, det in enumerate(DETS):
        for j, (p, col) in enumerate((('x', '#4C72B0'), ('y', '#C44E52'))):
            L = p1[det][p]
            a.errorbar(k + (j - .5) * .25, L['off_deg'], yerr=L['off_deg_err'], fmt='o', color=col, capsize=3,
                       label=f'{p.upper()} view' if k == 0 else None)
    a.axhline(0, color='0.5', lw=1); a.set_xticks(range(5)); a.set_xticklabels([f'{d}\n(z={SLOT_Z[d]})' for d in DETS])
    a.set_ylabel('angle offset w0 / v  [deg]'); a.set_title('Zero-angle offset per chamber'); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'w0_what_it_is.png'), dpi=140); plt.close(fig)

    # 2: field dependence on det3 -- geometric (constant angle) vs time-domain (constant w0)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for j, (p, col) in enumerate((('x', '#4C72B0'), ('y', '#C44E52'))):
        for r in rows:
            mk = 's' if 'mesh' in r['label'] else ('D' if '6-25' in r['label'] else 'o')
            ax[0].errorbar(r[p]['b'], r[p]['a'], yerr=r[p]['a_err'], fmt=mk, color=col, capsize=3)
            ax[1].errorbar(r[p]['b'], r[p]['off_deg'], yerr=r[p]['off_deg_err'], fmt=mk, color=col, capsize=3)
        ax[0].plot([], [], 'o', color=col, label=f'{p.upper()} view')
    for a in ax:
        a.axhline(0, color='0.5', lw=1); a.set_xlabel('ladder v of the row (slope of w vs tan)  [µm/ns]')
    ax[0].set_ylabel('intercept w0  [µm/ns]'); ax[0].set_title('det3: w0 in µm/ns vs drift speed')
    ax[1].set_ylabel('angle offset w0/v  [deg]'); ax[1].set_title('det3: as an angle')
    ax[0].plot([], [], 'o', color='k', label='sat 6-27 drift scan'); ax[0].plot([], [], 'D', color='k', label='6-25, 500 V')
    ax[0].plot([], [], 's', color='k', label='mesh-HV rows, 1000 V')
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'w0_vs_field_det3.png'), dpi=140); plt.close(fig)

    # 3: geometric tilt from positions vs angle offset
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for det in DETS + list(EXTRA):
        for p, mk in (('x', 'o'), ('y', 's')):
            s = p2[det][f'mesh_slope_along_{p}']; o = p2[det][f'offset_{p}']
            ax.errorbar(s['deg'], o['off_deg'], xerr=np.degrees(s['combined_err']), yerr=o['off_deg_err'],
                        fmt=mk, capsize=3, mfc='none' if det in EXTRA else None,
                        label=f"{EXTRA.get(det, det)} {p.upper()}")
    lim = 1.0
    ax.plot([-lim, lim * 0.5], [-lim, lim * 0.5], 'k--', lw=1, label='tilt explains it all')
    ax.axhline(0, color='0.8', lw=.8); ax.axvline(0, color='0.8', lw=.8)
    ax.set_xlabel('mesh tilt from POSITION residuals  [deg]'); ax.set_ylabel('angle offset w0/v from ANGLES  [deg]')
    ax.set_title('Is w0 a mechanical tilt of the chamber?'); ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'w0_vs_tilt.png'), dpi=140); plt.close(fig)


def main():
    p1, rows = part1()
    for det in DETS:
        print(det, ' '.join(f"{p}: a {p1[det][p]['a']:+.3f}±{p1[det][p]['a_err']:.3f} b {p1[det][p]['b']:.2f} "
                            f"angle {p1[det][p]['off_deg']:+.3f}±{p1[det][p]['off_deg_err']:.3f} (set_w0 {p1[det][p]['w0_setw0']:+.3f})"
                            for p in 'xy'))
    p2 = part2()
    json.dump(dict(per_chamber=p1, det3_rows=rows, tilt=p2), open(f'{WS}/w0_study.json', 'w'), indent=1)
    figures(p1, rows, p2)


if __name__ == '__main__':
    main()
