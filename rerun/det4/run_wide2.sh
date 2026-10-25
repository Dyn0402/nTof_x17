#!/bin/bash
# run_wide2.sh: a third sigma_p0 profile point (x 0.60, y 0.70) on det4 and det7 for the head-on far tail
V=~/x17/cosmic_bench/vladder; H=$(dirname $0); PY=$H/../../.venv/bin/python; O=~/x17/cosmic_bench/det4dig
for d in det4 det7; do
  rm -rf $V/$d/bundle_physwide2; cp -r $V/$d/bundle_phys $V/$d/bundle_physwide2
  python3 -c "import json;p='$V/$d/bundle_physwide2/bundle.json';b=json.load(open(p));b['hyper'].update(sigma_p0_x=0.60,sigma_p0_y=0.70);json.dump(b,open(p,'w'),indent=1)"
  $PY $H/resid_map.py --cache $V/$d/cache.pkl --bundle $V/$d/bundle_physwide2 --start 180 --n 1320 --out $O/rmF_${d}_physwide2.npz
done
