#!/usr/bin/env python3
"""cabling.py -- n_TOF vs bench channel -> strip-position maps (P2 of rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md).

    cabling.py [--configs DIR] [--out cabling_check.json]

The bench dead lists (rerun/channel_layer/<det>_<key>.json) are FEU channel numbers (0-511 per view) of the bench
golden row.  They carry to n_TOF BY NUMBER only if, for each chamber and view, the same channel number reads the same
strip.  This computes, through common.Mx17StripMap.Detector.map_hit (the map every bench/n_TOF tool uses, with each
run_config's dream_feus and dream_feu_orientation), the position of every channel:
  * of the bench golden row of each chamber (its run_config.json under the bench data root), and
  * of every n_TOF run in the registry (run_configs fetched from EOS into --configs, one per run),
and reports, per (chamber, view, n_TOF run), the number of channels whose position differs (> 0.01 mm) and the FEU
numbers.  Also importable: ``pos_maps(arm, cfg)`` -> {'x': (512,), 'y': (512,)} and ``POS`` (n_TOF reference maps).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
from common.Mx17StripMap import Mx17StripMap, Detector  # noqa: E402

MAP_CSV = os.path.join(REPO, 'mx17_m1_map.csv')
WORK = os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'ntof_pass1', 'channel_layer')
BENCH_ROOT = '/media/dylan/data/x17/cosmic_bench'
#: public arm -> (n_TOF run_config name, bench alias, bench run dir under BENCH_ROOT, bench census json)
ARMS = {
    'A': ('mx17_A', 'mx17_3', 'det3/mx17_det3_saturday_scan_6-27-26', 'det3_sat_det3.json'),
    'B': ('mx17_B', 'mx17_2', 'det2_det3/mx17_det2_det3_overnight_6-22-26', 'det2_o22_long_det2.json'),
    'C': ('mx17_C', 'mx17_6', 'det6_det7/mx17_det6_det7_overnight_6-26-26', 'det6_g_det6_long.json'),
    'D': ('mx17_D', 'mx17_7', 'det6_det7/mx17_det6_det7_overnight_6-26-26', 'det7_g_det7_long.json'),
}
_SMAP = None


def _smap():
    global _SMAP
    if _SMAP is None:
        _SMAP = Mx17StripMap(MAP_CSV)
    return _SMAP


def pos_maps(det_cfg):
    """{'x': (512,), 'y': (512,), 'feu': {'x': id, 'y': id}} strip positions [mm, strip-map frame] per FEU channel."""
    det = Detector(name=det_cfg['name'], det_cfg=det_cfg, strip_map=_smap())
    out, feu = {}, {}
    for view in ('x', 'y'):
        ids = sorted({f for k, (f, _c) in det_cfg['dream_feus'].items() if k[0] == view})
        assert len(ids) == 1, (det_cfg['name'], view, ids)
        feu[view] = ids[0]
        p = np.full(512, np.nan)
        for ch in range(512):
            r = det.map_hit(ids[0], ch)
            if r is not None:
                v = r[0] if view == 'x' else r[1]
                p[ch] = np.nan if v is None else v
        out[view] = p
    out['feu'] = feu
    return out


def det_cfg_of(cfg, name):
    for d in cfg['detectors']:
        if d.get('name') == name:
            return d
    raise KeyError(name)


def bench_maps(arm):
    _n, alias, rdir, _j = ARMS[arm]
    cfg = json.load(open(os.path.join(BENCH_ROOT, rdir, 'run_config.json')))
    return pos_maps(det_cfg_of(cfg, alias))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', default=os.path.join(WORK, 'run_configs'))
    ap.add_argument('--out', default=os.path.join(HERE, 'cabling_check.json'))
    a = ap.parse_args()
    res = dict(source='rerun/ntof/channel_layer/cabling.py', map_csv='mx17_m1_map.csv', arms={})
    for arm, (name, alias, rdir, _j) in ARMS.items():
        b = bench_maps(arm)
        rec = dict(bench_run=rdir, bench_alias=alias, bench_feu=b['feu'], runs={}, n_runs=0, n_runs_identical=0,
                   ntof_feu=set())
        for f in sorted(glob.glob(os.path.join(a.configs, 'run_*.json')), key=lambda s: int(s.split('_')[-1][:-5])):
            run = os.path.basename(f)[:-5]
            d = det_cfg_of(json.load(open(f)), name)
            m = pos_maps(d)
            diff = {v: int(np.sum(~np.isclose(m[v], b[v], atol=0.01, equal_nan=True))) for v in ('x', 'y')}
            rec['runs'][run] = dict(feu=m['feu'], n_diff=diff, orientation=d.get('dream_feu_orientation', {}))
            rec['n_runs'] += 1
            rec['n_runs_identical'] += int(diff['x'] == 0 and diff['y'] == 0)
            rec['ntof_feu'].add((m['feu']['x'], m['feu']['y']))
        rec['ntof_feu'] = sorted(rec['ntof_feu'])
        rec['n_cabled'] = {v: int(np.isfinite(b[v]).sum()) for v in ('x', 'y')}
        res['arms'][arm] = rec
        print(f"{arm} ({alias}): bench FEU {b['feu']}, n_TOF FEU {rec['ntof_feu']}; "
              f"{rec['n_runs_identical']}/{rec['n_runs']} n_TOF run_configs give identical channel->strip maps "
              f"(cabled x/y {rec['n_cabled']})")
    json.dump(res, open(a.out, 'w'), indent=1)
    print('->', a.out)


if __name__ == '__main__':
    main()
