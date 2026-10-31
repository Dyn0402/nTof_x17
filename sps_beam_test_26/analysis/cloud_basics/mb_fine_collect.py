#!/usr/bin/env python3
"""mb_fine_collect.py <dir with the fine jobs' magboltz_*_c30.json> [--hs DIR] -- file the fine Magboltz grids
(MB_FINE_2026-10-31.md) and check the SPS grids at their midpoints.

  results/air_fine    bench: the complete fine cube (make_fine_jobs.BENCH_*), with the existing air_hs points
                      that sit on its nodes reused (same recipe, 3e8 collisions)
  results/air_mid     the SPS midpoint validation points (never a fit grid: they would leave holes)
  results/air_merged  the source to fit with from now on: bench = air_fine (it supersedes the coarse bench
                      cube; a union would leave holes), beam and co2 = air_hs as extended for the gas fit
  results/mb_mid_check.json  each midpoint against GasGrid(air_hs) by trilinear interpolation, as v / eta
                      differences and as the water / O2 shift a fit would make

--hs: the directory holding every air_hs point incl. the gas-fit extension (lxplus ~/cloud_basics_condor/att);
default results/air_hs.  Files are copied, not linked, so the repo dirs stand alone.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gasmodel as M            # noqa: E402
import make_fine_jobs as J      # noqa: E402

RES = os.path.join(HERE, 'results')


def fill(dst, files):
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fine'); ap.add_argument('--hs', default=os.path.join(RES, 'air_hs'))
    a = ap.parse_args()
    fine = {os.path.basename(f): f for f in glob.glob(os.path.join(a.fine, 'magboltz_*_c30.json'))}
    hs = {os.path.basename(f): f for f in glob.glob(os.path.join(a.hs, 'magboltz_*_c30.json'))}
    want = {f'magboltz_{t}_E{e}_c{J.NCOLL}.json' for t, e in J.jobs('bench')}
    have = {n: fine.get(n) or hs.get(n) for n in want}
    missing = sorted(n for n, p in have.items() if p is None)
    print(f'bench fine cube: {len(want) - len(missing)}/{len(want)} points ({len(missing)} missing)')
    fill(os.path.join(RES, 'air_fine'), [p for p in have.values() if p])
    mid = {f'magboltz_{t}_E{e}_c{J.NCOLL}.json' for t, e in J.jobs('mid')}
    fill(os.path.join(RES, 'air_mid'), [fine[n] for n in mid if n in fine])
    merged = [p for p in have.values() if p] + [p for n, p in hs.items() if not n.startswith('magboltz_bench_')]
    shutil.rmtree(os.path.join(RES, 'air_merged'), ignore_errors=True)
    fill(os.path.join(RES, 'air_merged'), merged)

    # midpoint check against the grid the SPS gas fit uses
    out = []
    for gas in ('beam', 'co2'):
        G = M.GasGrid(gas, 'air_merged')
        for n in sorted(mid):
            if not n.startswith(f'magboltz_{gas}_') or n not in fine:
                continue
            m = re.search(r'_w([0-9p]+)_a([0-9p]+)_E', n)
            w, x = (float(s.replace('p', '.')) for s in m.groups())
            pt = json.load(open(fine[n]))['points'][0]; E = float(pt['E_Vcm'])
            p = G(w, x, E)
            dvdw = (G(w + 0.05, x, E)['v'] - G(w - 0.05, x, E)['v']) / 0.1
            deda = (G(w, x + 0.01, E)['eta'] - G(w, max(0, x - 0.01), E)['eta']) / (x + 0.01 - max(0, x - 0.01))
            dv = p['v'] - pt['v_true_um_ns']; de = p['eta'] - pt['eta_per_cm']
            out.append(dict(gas=gas, water=w, air=x, E=E, v_mc=pt['v_true_um_ns'], v_grid=p['v'],
                            dv_rel=dv / pt['v_true_um_ns'], eta_mc=pt['eta_per_cm'], eta_grid=p['eta'],
                            deta_rel=de / pt['eta_per_cm'] if pt['eta_per_cm'] > 0 else None,
                            water_shift=dv / dvdw if dvdw else None,
                            o2_shift_ppm=de / deda * 2095 if deda > 0 else None))
    json.dump(dict(missing=missing, mid=out), open(os.path.join(RES, 'mb_mid_check.json'), 'w'), indent=1)
    for r in out:
        print(f"{r['gas']:4s} w{r['water']:5.3f} a{r['air']:5.3f} E{r['E']:4.0f}: dv {r['dv_rel'] * 100:+.2f} %  "
              f"deta {(r['deta_rel'] or 0) * 100:+.1f} %  -> water {r['water_shift']:+.4f} %  O2 {r['o2_shift_ppm'] or 0:+.1f} ppm")


if __name__ == '__main__':
    main()
