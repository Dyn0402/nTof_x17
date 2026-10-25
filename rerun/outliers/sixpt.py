#!/usr/bin/env python3
"""sixpt.py -- which point is wrong? Six-point lines on the 6-26 night (det6 at z ~ 232, det7 at z ~ 702 + 4 M3 layers).

Per event and M3 view, the four M3 measured cluster positions (rebuilt rays, b71ffc9 branches) and the two MX17
fitted positions at the mesh (p0 of the raw view that this M3 view maps to, transformed into the M3 frame with
each chamber's own alignment) are fitted by one straight line (weights: M3 0.25 mm, MX17 0.5 mm). Leave-one-out:
the point whose removal lowers chi2 most, and its pull against the line through the other five.
Selection: both chambers fit both views (production selection), det7 X on card 2 (x >= 199.3), both footprints
>= 20 mm inside the active area. Output: ~/x17/cosmic_bench/outliers/sixpt.parquet (one row per event and view).
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO]
from olib import BASE, load  # noqa: E402
from m3feat import RUNS, ALIGN, ZL  # noqa: E402

SIG_M3, SIG_MX = 0.25, 0.5
NAMES = ['L0', 'L1', 'L2', 'L3', 'det7', 'det6']


def rays(det):
    import uproot, awkward as ak
    out = []
    for f in sorted(glob.glob(os.path.join(BASE, 'm3clus', f'{RUNS[det]}__*_rays.root'))):
        r = uproot.open(f)['T'].arrays(['evn'] + [f'm{c}{l}' for c in 'XY' for l in range(4)], library='ak')
        d = {'event_id': ak.to_numpy(r.evn)}
        for k in r.fields[1:]:
            d[k] = ak.to_numpy(ak.fill_none(ak.firsts(r[k]), np.nan))
        out.append(pd.DataFrame(d))
    return pd.concat(out).drop_duplicates('event_id', keep=False)


def to_m3(det, d):
    """MX17 mesh point (raw p0_x, p0_y) -> M3 frame (x, y) and the z of each M3 view."""
    al = json.load(open(os.path.join(BASE, 'final', ALIGN[det], 'wft/final_phys/alignment/alignment.json')))
    th = np.deg2rad(al['theta_deg']); ct, st = np.cos(th), np.sin(th)
    dx, dy = d.x_p0 - al['centre_x'], d.y_p0 - al['centre_y']
    ax = ct * dx - st * dy + al['centre_x'] + al['x_offset']
    ay = st * dx + ct * dy + al['centre_y'] + al['y_offset']
    return al['ref_x_sign'] * ax, ay, al['z_x'], al['z_y']


def fit(z, x, s):
    w = 1 / s ** 2; A = np.c_[z, np.ones_like(z)]
    W = np.diag(w); c = np.linalg.solve(A.T @ W @ A, A.T @ W @ x)
    r = x - A @ c; return c, float((w * r * r).sum())


def main():
    sel = {}
    for det in ('det6', 'det7'):
        d = load(det)
        k = d.use_x & d.use_y
        for v in 'xy':
            lo = np.minimum(d[f'ref_mesh_{v}'], d[f'ref_mesh_{v}'] + 30 * d[f'ref_tan_{v}'])
            hi = np.maximum(d[f'ref_mesh_{v}'], d[f'ref_mesh_{v}'] + 30 * d[f'ref_tan_{v}'])
            k &= (lo > 20) & (hi < 378)
        if det == 'det7':
            k &= np.minimum(d.ref_mesh_x, d.ref_mesh_x + 30 * d.ref_tan_x) >= 199.3
        e = d[k].copy(); mx, my, zx, zy = to_m3(det, e)
        sel[det] = pd.DataFrame({'event_id': e.event_id, f'{det}_X': mx, f'{det}_Y': my,
                                 f'{det}_zx': zx, f'{det}_zy': zy})
    J = sel['det7'].merge(sel['det6'], on='event_id').merge(rays('det7'), on='event_id')
    C = []
    for det in ('det6', 'det7'):
        c = pd.read_parquet(os.path.join(BASE, f'classified_{det}.parquet'), columns=['event_id', 'cls', 'why'])
        J = J.merge(c.rename(columns={'cls': f'cls_{det}', 'why': f'why_{det}'}), on='event_id', how='left')
    rows = []
    for r in J.to_dict('records'):
        for V in 'XY':
            z = np.array(ZL + [r[f'det7_z{V.lower()}'], r[f'det6_z{V.lower()}']])
            x = np.array([r[f'm{V}{l}'] for l in range(4)] + [r[f'det7_{V}'], r[f'det6_{V}']])
            s = np.array([SIG_M3] * 4 + [SIG_MX] * 2)
            if not np.all(np.isfinite(x)):
                continue
            _, c2 = fit(z, x, s)
            best = (np.inf, -1, 0.0)
            for i in range(6):
                m = np.arange(6) != i
                c, c2i = fit(z[m], x[m], s[m])
                pred = c[0] * z[i] + c[1]
                if c2i < best[0]:
                    best = (c2i, i, x[i] - pred)
            c4, _ = fit(z[:4], x[:4], s[:4])
            rows.append(dict(event_id=r['event_id'], view=V, chi2_6=c2, chi2_5=best[0], worst=NAMES[best[1]],
                             worst_shift=best[2], d7_vs_m3=x[4] - (c4[0] * z[4] + c4[1]),
                             d6_vs_m3=x[5] - (c4[0] * z[5] + c4[1]),
                             cls_det7=r['cls_det7'], cls_det6=r['cls_det6']))
    S = pd.DataFrame(rows); S.to_parquet(os.path.join(BASE, 'sixpt.parquet'))
    print(len(J), 'events in both chambers;', len(S), 'event-views')
    big = S[(S.chi2_6 > 50) & (S.chi2_5 < 10)]
    print('one-point-explained failures (chi2_6 > 50, chi2_5 < 10):', len(big), f'({100 * len(big) / len(J):.2f} % of events)')
    print(big.worst.value_counts().to_string())
    print('det7 POS_TAIL events:'); print(S[S.cls_det7 == 'POS_TAIL'][['view', 'chi2_6', 'chi2_5', 'worst', 'worst_shift', 'd7_vs_m3', 'd6_vs_m3']].round(2).to_string())


if __name__ == '__main__':
    main()
