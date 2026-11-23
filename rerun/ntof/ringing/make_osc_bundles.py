#!/usr/bin/env python3
"""make_osc_bundles.py -- roadmap 4.9: chamber D's pass-1 bundles with its permanently oscillating channels added to the
hot list (the 'common' unmasked channels of d_osc.py: oscillating in > 20 % of >= 30 ms events in all three test runs).

    python rerun/ntof/ringing/make_osc_bundles.py [--src $WORK_DATA/ntof_pass1/bundles/pf] [--dst .../bundles/pf_osc]

Writes <dst>/<name> for every D_* bundle of <src>; the provenance gains 'hot_added_osc' (channels per view, source).
A hot channel is down-weighted in the fit and never seeds a cluster by itself (wft.seed.seed_candidates).
"""
import argparse, json, os, shutil
from pathlib import Path
WD = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1'
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', default=str(WD / 'bundles' / 'pf'))
ap.add_argument('--dst', default=str(WD / 'bundles' / 'pf_osc'))
ap.add_argument('--osc', default=str(WD / 'ana' / 'ringing' / 'd_osc_channels.json'))
a = ap.parse_args()
osc = json.load(open(a.osc))['common']
for b in sorted(Path(a.src).glob('D_*')):
    if not (b / 'bundle.json').exists():
        continue
    out = Path(a.dst) / b.name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(b, out)
    j = json.load(open(out / 'bundle.json'))
    add = {}
    for p in 'xy':
        old = set(j['hot'][p])
        new = [c for c in osc[p]['unmasked'] if c not in old and c not in set(j['dead'][p])]
        j['hot'][p] = sorted(old | set(new))
        add[p] = new
    j['provenance']['hot_added_osc'] = dict(channels=add, source=a.osc, roadmap='4.9')
    json.dump(j, open(out / 'bundle.json', 'w'), indent=1)
    print(b.name, {p: len(v) for p, v in add.items()}, '->', out)
