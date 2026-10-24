#!/bin/bash
# run_wide.sh: the physical model with a wide prompt footprint (det4 bundle's 0.457/0.550, det3 r06-like 0.42/0.50),
# for a per-event two-point sigma_p0 profile against run_final.sh's phys arms (same events [180, 1500))
V=~/x17/cosmic_bench/vladder; H=$(dirname $0); PY=$H/../../.venv/bin/python; O=~/x17/cosmic_bench/det4dig
mk() { rm -rf $V/$1/$3; cp -r $V/$1/$2 $V/$1/$3; python3 -c "import json;p='$V/$1/$3/bundle.json';b=json.load(open(p));b['hyper'].update(sigma_p0_x=$4,sigma_p0_y=$5);json.dump(b,open(p,'w'),indent=1)"; }
mk det4 bundle_phys bundle_physwide 0.457 0.550
mk det3 bundle_phys bundle_physwide 0.42 0.50
$PY $H/resid_map.py --cache $V/det4/cache.pkl --bundle $V/det4/bundle_physwide --start 180 --n 1320 --out $O/rmF_det4_physwide.npz
$PY $H/resid_map.py --cache $V/det3/cache.pkl --bundle $V/det3/bundle_physwide --start 180 --n 1320 --out $O/rmF_det3_physwide.npz
