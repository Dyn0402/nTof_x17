"""qE_dilution.py -- chamber E's low gap-term amplitude b (0.17 on 6-23, 0.09 on 6-24; A, B, D give 0.37-0.48).

Hypothesis: a small-sample bias. b is the slope of the tilt residual on the gradient of the drift-time field, and the
field is a P-spline fitted to the same events (K = 8 below 4,000 events). With E's 1,400-1,800 events the field's
gradient is mostly noise, and a noisy predictor pulls a regression slope toward zero (regression dilution).

Test: subsample the large mountings (A golden, B, D) to E's size, rebuild the field and refit b exactly as R3 / q2 do
(r3_lib.gap_field, attach_field, fit_b), 30 random subsamples each. A second arm builds the field from the full
mounting and fits b on the subsample (no dilution from the field), which separates the field noise from the b noise.
Also E itself with the field from the other E mounting (the two E gap maps agree, r = 0.97), which doubles the field's
statistics without touching the events b is fitted on.

-> $WORK/audit/q16/qE_dilution.json
"""
import json
import numpy as np
from r3_lib import load_mounting, gap_field, attach_field, fit_b, R1

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
rng = np.random.default_rng(7)
NREP = 30
res = {}

for name in ('A628', 'B', 'D', 'E623', 'E624'):
    D, info, bundle = load_mounting(name)
    spl, c, ok = gap_field(D)
    attach_field(D, bundle, spl, c)
    full = fit_b(D)
    r = dict(N=int(D.n), K=int(spl.K) if hasattr(spl, 'K') else None, b_full=full)
    print(name, 'N', D.n, 'b (own field)', {k: round(v, 3) for k, v in full.items()}, flush=True)
    if not name.startswith('E'):
        for nsub in (1500, 3000):
            if D.n < 2 * nsub:
                continue
            own, fixed = [], []
            for _ in range(NREP):
                i = np.sort(rng.choice(D.n, nsub, replace=False))
                S = D.sub(i)
                s2, c2, _ = gap_field(S)            # field from the subsample itself (as for E)
                attach_field(S, bundle, s2, c2)
                own.append(fit_b(S)['xy'])
                S2 = D.sub(i)                        # field from the full mounting, b on the subsample
                attach_field(S2, bundle, spl, c)
                fixed.append(fit_b(S2)['xy'])
            r[f'sub{nsub}'] = dict(own_field=[float(np.mean(own)), float(np.std(own))],
                                   full_field=[float(np.mean(fixed)), float(np.std(fixed))])
            print(f'  subsample {nsub}: b own field {np.mean(own):.3f} +- {np.std(own):.3f};'
                  f' b with full field {np.mean(fixed):.3f} +- {np.std(fixed):.3f}', flush=True)
    res[name] = r

# E with the field of both E mountings together (same cathode, r = 0.97): more statistics for the field only
DE = {n: load_mounting(n) for n in ('E623', 'E624')}
from r3_lib import Data
cat = {k: np.concatenate([getattr(DE[n][0], k) for n in DE]) for k in ('U', 'W', 'lnq')}
J = Data(U=cat['U'], W=cat['W'], lnq=cat['lnq'], n=len(cat['U']))
sJ, cJ, _ = gap_field(J)
for n, (D, info, bundle) in DE.items():
    attach_field(D, bundle, sJ, cJ)
    b = fit_b(D)
    res[n]['b_joint_E_field'] = b
    print(n, 'b with the joint E field', {k: round(v, 3) for k, v in b.items()}, flush=True)
json.dump(res, open(f'{OUT}/qE_dilution.json', 'w'), indent=1, default=float)
