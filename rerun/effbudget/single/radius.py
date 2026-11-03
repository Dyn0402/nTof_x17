#!/usr/bin/env python3
"""radius.py [--root R] [--out DIR] -- Paper I: det3 efficiency against the matching radius, and a reason for R.

"Detected" = both views fitted, <= 2 competing strips per view, and the fitted track within R of the M3 ray at the
mesh (raw frame, |r| = hypot(res_x, res_y)). The radius sets what the efficiency means: too small and resolution
tails (multiple scattering of soft muons, bent M3 tracks, the chamber's own tail) count as losses; too large and an
unrelated deposit can be counted as the muon.

Measured here:
  eff(R)       production pick, and 'any candidate' (credit: some candidate within R in each view), R = 0.25..40 mm;
  acc(R)       the accidental rate: det3's fit of one event against the M3 ray of ANOTHER event (event mixing, 20
               shifts), i.e. how often an unrelated track lands inside R. This is an upper bound on fakes, because
               in the real event the M3 particle is also there.
  the residual |r| distribution of detected tracks (core sigma per view, tail fractions).
Result (2026-11-02): eff(R) has two regimes, the resolution core (steep up to ~4 mm; core sigma 0.55 mm per view,
M3 pointing + scattering dominated) and a slow linear rise of ~0.1-0.2 pt per mm out to 40 mm (mis-measured tracks,
which A1 classes as delta / saturated blob / bent M3, plus accidentals growing as R^2). The choice is the knee:
R = 5 mm = 9 core sigma, where the Gaussian core is exhausted and acc(R) is 0.07 % (below the 0.15 % statistical
error and ~70x below the inefficiency). R in [4, 8] mm moves the efficiency by -0.5 / +0.6 pt (quoted systematic).
Writes <out>/radius.json.
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
RADII = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0]
MAX_DROPPED = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1)
    res = {}
    for key, rel in ROWS.items():
        B = pd.read_parquet(os.path.join(a.root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet'))
        B = B.drop_duplicates('event_id')
        B = B[(~B.not_measured) & (B.region == 'interior') & B.good_x].reset_index(drop=True)
        ok = (B.x_ok & B.y_ok & (B.x_n_dropped.fillna(99) <= MAX_DROPPED) & (B.y_n_dropped.fillna(99) <= MAX_DROPPED)).to_numpy()
        r = B.r.to_numpy()
        nx, ny = B.near_res_x.abs().to_numpy(), B.near_res_y.abs().to_numpy()
        eff, cred, acc = [], [], []
        # event mixing: det3 p0 of event j against the M3 mesh point of event i
        px, py = B.x_p0.to_numpy(), B.y_p0.to_numpy(); mx, my = B.mesh_x.to_numpy(), B.mesh_y.to_numpy()
        shifts = rng.integers(1, len(B) - 1, 20)
        rmix = np.concatenate([np.hypot(np.roll(px, s) - mx, np.roll(py, s) - my)[np.roll(ok, s)] for s in shifts])
        for R in RADII:
            eff.append(100 * float((ok & (r <= R)).mean()))
            cred.append(100 * float(((ok & (r <= R)) | (np.hypot(nx, ny) <= R)).mean()))   # 2026-11-03: circle, as budget det_any_R (was a square, +0.15 pt)
            acc.append(100 * float((rmix <= R).mean()) * float(ok.mean()))
        d = ok & (r <= 40)
        rx, ry = B.res_x.to_numpy()[d], B.res_y.to_numpy()[d]
        core = {v: float(1.4826 * np.median(np.abs(u - np.median(u)))) for v, u in (('x', rx), ('y', ry))}
        tails = {f'>{t:g} mm': 100 * float((r[d] > t).mean()) for t in (2, 3, 5, 8, 10)}
        # full distributions (figure 'residuals'): per-view residual of the production fit (both views fitted,
        # n_dropped ok, no radius cut), radial |r|, and the event-mixed |r| scaled to the same number of fits
        be = np.linspace(-40, 40, 321); br = np.r_[0, np.geomspace(0.05, 200, 120)]
        hx = np.histogram(B.res_x.to_numpy()[ok], be)[0]; hy = np.histogram(B.res_y.to_numpy()[ok], be)[0]
        hr = np.histogram(r[ok], br)[0]; hm = np.histogram(rmix, br)[0] * ok.sum() / len(rmix)
        res[key] = dict(n=int(len(B)), radii=RADII, eff=eff, credit=cred, accidental=acc, core_sigma_mm=core,
                        tail_pct_of_matched_within_40=tails, n_fitted=int(ok.sum()),
                        hist=dict(edges_view=be.tolist(), res_x=hx.tolist(), res_y=hy.tolist(), edges_r=br.tolist(),
                                  r=hr.tolist(), r_mixed=hm.tolist()))
        print(f'== {key} n={len(B)} core sigma x/y {core["x"]:.2f}/{core["y"]:.2f} mm, tails {tails}')
        for R, e, c, f in zip(RADII, eff, cred, acc):
            print(f'  R={R:5.2f}  eff {e:6.2f}  credit {c:6.2f}  accidental {f:7.4f}')
    json.dump(res, open(os.path.join(a.out, 'radius.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
