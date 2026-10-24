#!/bin/bash
# run_dp.sh: Dp scan at the physical footprint (det3 sigma_p0) on det3 (own gains) and det4 (det3 footprint, beam D_rc, own gains)
V=~/x17/cosmic_bench/vladder; H=$(dirname $0)
until [ -s ~/x17/cosmic_bench/det4dig/rm_det3_rc_g.npz ]; do sleep 20; done
for dp in 0.018 0.022 0.025; do
  for pair in det3:bundle_g det4:bundle_s3beam_g; do
    d=${pair%%:*}; src=${pair##*:}; arm=${src}_dp$dp
    rm -rf $V/$d/$arm; cp -r $V/$d/$src $V/$d/$arm
    python3 -c "import json,sys;p='$V/$d/$arm/bundle.json';b=json.load(open(p));b['hyper']['Dp']=$dp;json.dump(b,open(p,'w'),indent=1)"
    $H/run_resid.sh ${d}_${src#bundle_}_dp$dp $V/$d/cache.pkl $V/$d/$arm
  done
done
