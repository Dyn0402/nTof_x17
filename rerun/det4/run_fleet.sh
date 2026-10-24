#!/bin/bash
# run_fleet.sh: the physical RC recipe on det2/6/7 (laptop vladder caches), arms rc (current bundle) / phys / physwide
# phys: sigma_p0 0.238/0.343 (det3 per-event), Dp from the unsaturated head-on stack (det2 0.025, det7 0.0225; det6
# stack 0.014 +- 0.004 is window-limited -> keep 0.018), det6 v rescaled to the adopted 28 mm gap (25.9 * 28/30)
V=~/x17/cosmic_bench/vladder; H=$(dirname $0); PY=$H/../../.venv/bin/python; O=~/x17/cosmic_bench/det4dig
until [ -s $O/rmF_det3_physwide.npz ]; do sleep 30; done
mk() { rm -rf $V/$1/$3; cp -r $V/$1/$2 $V/$1/$3; python3 -c "
import json;p='$V/$1/$3/bundle.json';b=json.load(open(p));b['hyper'].update($4)
v=$5
if v: b['v_drift']=v
json.dump(b,open(p,'w'),indent=1)"; }
mk det2 bundle bundle_phys "dict(sigma_p0_x=0.238,sigma_p0_y=0.343,Dp=0.025)" None
mk det7 bundle bundle_phys "dict(sigma_p0_x=0.238,sigma_p0_y=0.343,Dp=0.0225)" None
mk det6 bundle bundle_phys "dict(sigma_p0_x=0.238,sigma_p0_y=0.343,Dp=0.018)" 24.17
for d in det2 det7 det6; do
  mk $d bundle_phys bundle_physwide "dict(sigma_p0_x=0.42,sigma_p0_y=0.50)" None
  n=$($PY -c "import pickle;print(min(1320,len(pickle.load(open('$V/$d/cache.pkl','rb')))-180))")
  for arm in bundle bundle_phys bundle_physwide; do
    case $arm in bundle) t=${d}_rc;; bundle_phys) t=${d}_phys;; bundle_physwide) t=${d}_physwide;; esac
    $PY $H/resid_map.py --cache $V/$d/cache.pkl --bundle $V/$d/$arm --start 180 --n $n --out $O/rmF_$t.npz
  done
done
