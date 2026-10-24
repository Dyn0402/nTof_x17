#!/usr/bin/env python3
"""make_phys_bundle.py <analysis_root> [--it it1] [--dry] -- the physical RC bundles for the five golden rows.

<row>/wft/w12/<it>/timeD/bundle  ->  <row>/wft/w12/<it>/phys/bundle  (finalize with --bundle-sub phys/bundle)

The recipe (rerun/det4, 2026-10-25): the RC kernel with its inputs taken from physics, not from the calibration
fit, where the fit is known to absorb other things:
  sigma_p0_x/y   0.238 / 0.343 mm: det3's per-event prompt footprint (FINDINGS §38/§42: strip box + induction
                 ~0.2 mm); the calibration fits ran 0.03 (det6, at a bound) to 0.53 (det7)
  Dp             the unsaturated head-on stack (cloud_basics unsat_width, sqrt(2 Dd)), which agrees with Magboltz
                 on det3/det4: det2 0.025, det3 0.023, det4 0.020, det7 0.0225; det6 keeps 0.018 (its stack, 0.014
                 +- 0.004, is window-limited)
  rc_D_y         det4 from the H4 beam (2.2e-4, decided 2026-10-12); others keep the calibration value
  v_drift        det6 rescaled to the adopted 28 mm gap (x 28/30); others keep the timing v
  sigma_p0_prof  [0.42] (x), [0.50] (y): per-event profile for bright/saturated clusters (wft.reco, opt-in key)
Everything else (templates, t0_abs, c-keys, gains) is the timeD bundle's.  w0/kw are re-measured by finalize.
"""
import argparse, json, os, shutil

ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
        'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
DP = {'det2': 0.025, 'det3': 0.023, 'det4': 0.020, 'det6': 0.018, 'det7': 0.0225}
COMMON = dict(sigma_p0_x=0.238, sigma_p0_y=0.343, sigma_p0_prof_x=[0.42], sigma_p0_prof_y=[0.50])
EXTRA = {'det4': dict(rc_D_y=2.2e-4)}
V_SCALE = {'det6': 28.0 / 30.0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--it', default='it1')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    for d, rel in ROWS.items():
        src = os.path.join(a.root, rel, 'wft', 'w12', a.it, 'timeD', 'bundle')
        dst = os.path.join(a.root, rel, 'wft', 'w12', a.it, 'phys', 'bundle')
        b = json.load(open(os.path.join(src, 'bundle.json')))
        old = {k: b['hyper'].get(k) for k in ('sigma_p0_x', 'sigma_p0_y', 'Dp', 'rc_D_y')}
        new = dict(COMMON, Dp=DP[d], **EXTRA.get(d, {}))
        v0 = b['v_drift']
        b['hyper'].update(new)
        b['v_drift'] = v0 * V_SCALE.get(d, 1.0)
        b.setdefault('provenance', {})['phys_recipe'] = dict(
            tool='rerun/w12/make_phys_bundle.py', source=src, replaced=old, v_was=v0, set=new,
            evidence='rerun/det4 (2026-10-25): compare.py on 1320 held-out events per chamber')
        print(d, 'v', round(v0, 2), '->', round(b['v_drift'], 2), '|', old, '->', new)
        if a.dry:
            continue
        if os.path.exists(dst):
            raise SystemExit(f'{dst} exists: not overwriting')
        shutil.copytree(src, dst)
        json.dump(b, open(os.path.join(dst, 'bundle.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
