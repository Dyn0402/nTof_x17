#!/usr/bin/env python3
"""merge_maps.py -- one map file for map_view.py: A, C from angle_map.py fit (A-C cosmic lines) and D from bd_map.py
(B-D cosmic lines), in the layout of rerun/ntof/stage3/angle_map/angle_map.json.

    python -m rerun.ntof.improve.merge_maps --ac <out>/angle_map.json --d <out>/bd_map.json --out <file>
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ac', required=True)
    ap.add_argument('--d', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ac, d = json.load(open(a.ac)), json.load(open(a.d))
    maps = {k: v for k, v in ac['maps'].items() if k[0] in 'AC'}
    maps.update({k: v for k, v in d['maps'].items() if k[0] == 'D'})
    if sorted(maps) != ['A_x', 'A_y', 'C_x', 'C_y', 'D_x', 'D_y']:
        raise SystemExit(f'incomplete maps: {sorted(maps)}')
    meta = dict(ac['meta'])
    meta['D'] = dict(d['meta'], source='rerun/ntof/improve/bd_map.py (B-D cosmic lines, censored-window cut, 100 mm cells)')
    meta['frames'] = {'A': 'x view -> d_x/d_z, y view -> d_y/d_z', 'C': 'same as A',
                      'D': 'x view -> d_z/d_x, y view -> d_y/d_x'}
    meta['merged_by'] = 'rerun/ntof/improve/merge_maps.py'
    json.dump(dict(meta=meta, maps=maps), open(a.out, 'w'), indent=1)
    print('wrote', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
