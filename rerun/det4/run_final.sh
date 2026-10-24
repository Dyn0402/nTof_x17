#!/bin/bash
# run_final.sh: the physical RC model vs the current bundles and the lp kernel, on all 1320 held-out events [180, 1500)
# phys = RC kernel, sigma_p0 from det3 per-event (x 0.238 / y 0.343), Dp from Magboltz + unsat stack (det3 0.023, det4 0.020),
#        det4 D_rc,y from the H4 beam (2.2e-4), unit gains
V=~/x17/cosmic_bench/vladder; H=$(dirname $0); PY=$H/../../.venv/bin/python; O=~/x17/cosmic_bench/det4dig
mk() { rm -rf $V/$1/$3; cp -r $V/$1/$2 $V/$1/$3; python3 -c "import json;p='$V/$1/$3/bundle.json';b=json.load(open(p));b['hyper']['Dp']=$4;json.dump(b,open(p,'w'),indent=1)"; }
mk det3 bundle bundle_phys 0.023
mk det4 bundle_s3beam bundle_phys 0.020
R() { $PY $H/resid_map.py --cache $V/$1/cache.pkl --bundle $2 --start 180 --n 1320 --out $O/rmF_$3.npz; }
R det3 $V/det3/bundle det3_rc
R det3 $V/det3/bundle_phys det3_phys
R det4 $V/det4/bundle det4_rc
R det4 $V/det4/bundle_phys det4_phys
R det4 ~/x17/cosmic_bench/condor_campaign_r06/local_bundles/mx17_4/calib_bundle_lp_t0p det4_lp
