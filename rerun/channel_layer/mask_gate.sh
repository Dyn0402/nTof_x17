#!/bin/bash
# mask_gate.sh [dets...]: W3 gate (Dylan 10-25: channel masks into the pass bundles if they do not hurt).
# Mask = census dead + shorts (rerun/channel_layer/<det>_<key>.json, channel_census.py) as bundle `dead` (censored
# samples); noisy NOT masked (C20: as hot or censored it raised det7's affected-region far rate 46.5 -> 57-58 %),
# weak not masked, unit gains.  Arms on the laptop vladder caches, held-out events [180, 1500) as C21: the phys
# (narrow) and physwide bundles with and without the mask, then the per-event sigma_p0 profile (profile_arms.py,
# lowest chi2, no reference) and compare.py prof -> prof_mask.  Writes results/mask_gate.json.
V=~/x17/cosmic_bench/vladder; H=$(cd $(dirname $0) && pwd); R=$H/..; PY=$R/../.venv/bin/python; O=~/x17/cosmic_bench/det4dig
declare -A KEY=([det2]=o22_long_det2 [det3]=sat_det3 [det4]=g_det4 [det6]=g_det6_long [det7]=g_det7_long)
DETS=${@:-det3 det7 det4 det6 det2}
for d in $DETS; do
  for b in bundle_phys bundle_physwide; do
    rm -rf $V/$d/${b}_mask; cp -r $V/$d/$b $V/$d/${b}_mask
    $PY - <<EOF
import json
c = json.load(open('$H/${d}_${KEY[$d]}.json'))
p = '$V/$d/${b}_mask/bundle.json'; b = json.load(open(p))
b['dead'] = {v: sorted(set(int(x) for x in c[v]['dead']) | set(int(x) for x in c[v]['shorts'])) for v in 'xy'}
b.setdefault('provenance', {})['channel_mask'] = 'census dead + shorts, rerun/channel_layer/${d}_${KEY[$d]}.json (mask_gate.sh)'
json.dump(b, open(p, 'w'), indent=1)
print('$d $b dead', {v: len(x) for v, x in b['dead'].items()})
EOF
  done
  n=$($PY -c "import pickle;print(min(1320,len(pickle.load(open('$V/$d/cache.pkl','rb')))-180))")
  for b in bundle_phys bundle_physwide bundle_phys_mask bundle_physwide_mask; do
    t=G${d}_${b#bundle_}
    [ -s $O/rmF_$t.npz ] || $PY $R/det4/resid_map.py --cache $V/$d/cache.pkl --bundle $V/$d/$b --start 180 --n $n --out $O/rmF_$t.npz
  done
  (cd $R/det4 && $PY profile_arms.py G${d}_profU G${d}_phys G${d}_physwide && $PY profile_arms.py G${d}_profM G${d}_phys_mask G${d}_physwide_mask)
done
cd $R/det4 && $PY compare.py $(for d in $DETS; do echo -n "G${d}_profU:G${d}_profM "; done) --json $H/results_mask_gate.json
