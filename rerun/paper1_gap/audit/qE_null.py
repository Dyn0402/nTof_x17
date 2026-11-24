"""qE_null.py -- the uncertainty of the gap-term amplitude b from chance alignment.

fit_b's error is an event bootstrap with the field fixed. It does not include the chance alignment of the reproducible
part of the tilt residual that is not the gap (fine mesh relief, strip effects) with the gradient of the drift-time
field. Null: evaluate the field (and its gradient) at cyclically shifted positions (U + du, W + dw wrapped on the
interior domain), refit b, 48 shifts with |du| >= 0.6 or |dw| >= 0.6 (60 mm). The spread of the null b is that
uncertainty. Controls: A golden, B, D. -> $WORK/audit/q16/qE_null.json
"""
import json
import numpy as np
from r3_lib import load_mounting, gap_field, attach_field, fit_b, R1
from a2_lib import UD, WD

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
wrap = lambda x, lo, hi: lo + np.mod(x - lo, hi - lo)
shifts = [(du, dw) for du in np.linspace(-1.4, 1.4, 8) for dw in np.linspace(-1.2, 1.2, 7)
          if abs(du) >= 0.6 or abs(dw) >= 0.6]
res = {}
for name in ('A628', 'B', 'D', 'E623', 'E624'):
    D, info, bundle = load_mounting(name)
    spl, c, ok = gap_field(D)
    attach_field(D, bundle, spl, c)
    b0 = fit_b(D)
    U0, W0 = D.U.copy(), D.W.copy()
    null = []
    for du, dw in shifts:
        Us, Ws = wrap(U0 + du, *UD), wrap(W0 + dw, *WD)
        D.glx = (spl.basis(Us, Ws, 1, 0) @ c) / 100
        D.gly = (spl.basis(Us, Ws, 0, 1) @ c) / 100
        null.append([fit_b(D)[k] for k in ('x', 'y', 'xy')])
    null = np.array(null)
    res[name] = dict(N=int(D.n), b=b0, null_mean=dict(zip('x y xy'.split(), null.mean(0).tolist())),
                     null_sd=dict(zip('x y xy'.split(), null.std(0).tolist())), n_shifts=len(shifts))
    print(name, 'N', D.n, ' b xy %.3f (x %.3f, y %.3f)' % (b0['xy'], b0['x'], b0['y']),
          ' null mean xy %+.3f sd xy %.3f (x %.3f, y %.3f)' % (null[:, 2].mean(), null[:, 2].std(), null[:, 0].std(), null[:, 1].std()), flush=True)
json.dump(res, open(f'{OUT}/qE_null.json', 'w'), indent=1, default=float)
