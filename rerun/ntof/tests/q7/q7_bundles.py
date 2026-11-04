#!/usr/bin/env python3
"""q7_bundles.py -- copies of the v1k n_TOF bundles carrying one Q7 configuration (tan_max, split_corrob, select).

    q7_bundles.py [--arms A C] [--configs colwin_count_gate tm0.6_charge_cost ...]

A copy is the v1k bundle (bundles/v1k_<arm>: timing v, in-situ kw, t0 free) with only these fields changed, and the
change recorded in provenance['q7']:  bundle.tan_max (None never: the value is written), xy_pairing['select'],
xy_pairing['split_corrob'].  This is the route NOTE_PAIRING_FIX §4 names ("set xy_pairing['select'] in a copy of a
bundle"); ntof_bundle.py itself is not changed.  Written to $Q7_DATA/bundles/<cfg>_<arm>.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402


def parse(name):
    tml, split, sel = name.split('_')
    tm = L.INF if tml == 'colwin' else float(tml[2:])
    return tm, split, sel


def make(arm, name, src=None, out_root=None):
    tm, split, sel = parse(name)
    src = src or os.path.join(L.BUNDLES, f'v1k_{arm}')
    out = os.path.join(out_root or os.path.join(L.DATA, 'bundles'), f'{name}_{arm}')
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(src, out)
    p = os.path.join(out, 'bundle.json')
    d = json.load(open(p))
    old = dict(tan_max=d.get('tan_max'), select=d['xy_pairing'].get('select'),
               split_corrob=d['xy_pairing'].get('split_corrob'))
    d['tan_max'] = float(tm)
    d['xy_pairing']['select'] = sel
    d['xy_pairing']['split_corrob'] = split
    d.setdefault('provenance', {})['q7'] = dict(
        source=src, config=name, changed=dict(tan_max=[old['tan_max'], float(tm)], select=[old['select'], sel],
                                              split_corrob=[old['split_corrob'], split]),
        note='rerun/ntof/tests/q7 study copy; every other field is the source bundle')
    json.dump(d, open(p, 'w'), indent=1)
    from wft.calib import CalibrationBundle
    c = CalibrationBundle.load(out)
    assert c.tan_max == tm and c.xy_pairing['select'] == sel and c.xy_pairing['split_corrob'] == split
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=['A', 'C'])
    ap.add_argument('--configs', nargs='+', required=True)
    a = ap.parse_args()
    for arm in a.arms:
        for n in a.configs:
            print(make(arm, n))


if __name__ == '__main__':
    main()
