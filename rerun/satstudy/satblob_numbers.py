#!/usr/bin/env python3
"""satblob_numbers.py satblob.json out.json -- the three numbers Paper I quotes from the saturated-blob test
(A1 table row and the loss_satblob caption), read from the satblob.py score output.

  agree_prod   % of the saturated views (group sat) whose production fit agrees with the reference
  agree_best   the best agreement over every other arm (censoring at the rail / bottom rail, multi-start track-only,
               track + compact blob; the 5 % model-error arm included, as the pre-pass 73.7 was the max over arms)
  pinned_worse % of the still-failing views where the model with the track pinned to the reference fits WORSE than the
               free fit (pinned|railbot|fail ref_worse_TB)
"""
import json, sys
J = json.load(open(sys.argv[1]))
sat = {k.split('|', 1)[1]: v for k, v in J.items() if k.startswith('sat|')}
best = max((v['agree'], k) for k, v in sat.items() if k != 'prod')
pin = J['pinned|railbot|fail']
out = dict(n_views=sat['prod']['n'], agree_prod=100 * sat['prod']['agree'], agree_best=100 * best[0], best_arm=best[1],
           pinned_worse=100 * pin['ref_worse_TB'], pinned_n=pin['n'],
           arms={k: 100 * v['agree'] for k, v in sorted(sat.items())})
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
