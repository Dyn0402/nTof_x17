#!/usr/bin/env python3
"""sps_summary.py --dirs DIR [DIR ...] [--json out.json] -- the SPS known-angle table from heldout_arm.py outputs on the
run_63 25.64° caches: per plateau and arm, in each view the median fitted w, |w|/tan(25.64°) against the span-route
v(E), the angle at that v, its per-event robust width, the far fraction (> 10° from the mount angle in the inclined
view) and the median χ²/dof.  Markdown + JSON.
"""
import argparse, glob, json, os, re
import numpy as np

TANM = float(np.tan(np.radians(25.64)))
VS = {'d675': (235, 14.0), 'd625': (217, 13.2), 'd575': (200, 12.4), 'd525': (182, 11.8), 'd475': (165, 11.6), 'd425': (148, 11.6), 'd325': (113, 11.5)}


def rs(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dirs', nargs='+', required=True); ap.add_argument('--json', default=None)
    a = ap.parse_args(); out = {}
    L = ['| plateau (V/cm) | arm | n | Y: w med [µm/ns] | |w|/tan vs v_span | θ_Y(v_span) med ± rσ | far>10° | χ²/dof Y | X: θ med ± rσ | χ²/dof X |',
         '|' + '---|' * 10]
    rows = []
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, 'run63_rot25_operating_00_*.npz'))):
            m = re.match(r'run63_rot25_operating_00_(d\d+)_(\w+)\.npz', os.path.basename(f))
            if not m:
                continue
            pl, arm = m.groups(); E, v = VS[pl]; P = np.load(f)['per']
            r = dict(plateau=pl, field=E, v_span=v, arm=arm, dir=os.path.basename(d))
            for vi, view in enumerate('xy'):
                R = P[P[:, 1] == vi]
                if not len(R):
                    continue
                w = R[:, 3]; th = np.degrees(np.arctan(w / v)); chi = R[:, 4] / np.maximum(R[:, 5], 1)
                r[view] = dict(n=int(len(R)), w_med=float(np.median(w)), w_rsig=rs(w), ratio=float(abs(np.median(w)) / TANM / v),
                               theta_med=float(np.median(th)), theta_rsig=rs(th), far10=float(np.mean(np.abs(np.abs(th) - 25.64) > 10)) if view == 'y' else float(np.mean(np.abs(th) > 10)),
                               chi2dof=float(np.median(chi)), uend_p50=float(np.nanmedian(R[:, 14])))
            rows.append(r)
    rows.sort(key=lambda r: (-r['field'], r['dir'], r['arm']))
    for r in rows:
        y, x = r.get('y', {}), r.get('x', {})
        L.append(f"| {r['plateau']} ({r['field']}) | {r['arm']} ({r['dir']}) | {y.get('n')} | {y.get('w_med', np.nan):+.2f} | {y.get('ratio', np.nan):.3f} | "
                 f"{y.get('theta_med', np.nan):+.1f} ± {y.get('theta_rsig', np.nan):.1f} | {100 * y.get('far10', np.nan):.0f} % | {y.get('chi2dof', np.nan):.0f} | "
                 f"{x.get('theta_med', np.nan):+.1f} ± {x.get('theta_rsig', np.nan):.1f} | {x.get('chi2dof', np.nan):.0f} |")
        out[f"{r['plateau']}|{r['arm']}|{r['dir']}"] = r
    md = '\n'.join(L) + '\n'; print(md)
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1); open(a.json.replace('.json', '.md'), 'w').write(md)


if __name__ == '__main__':
    main()
