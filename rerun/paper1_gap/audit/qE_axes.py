"""qE_axes.py -- chamber E: does the tilt follow the drift-time field along another axis or with another sign?

Event-level regression, per view, of the tilt residual (w - tan_ref - mesh gradient of the bow block) on BOTH gradient
components of the drift-time field, scaled as in fit_b (G0 / (1 + 0.68)):
    r_x = c + b_xx dlf/dx + b_xy dlf/dy,     r_y = c + b_yx dlf/dx + b_yy dlf/dy
A correct frame gives b_xx = b_yy (about 0.45 on A, B, D) and b_xy = b_yx = 0. A frame turned by 90 degrees puts the
amplitude in the cross terms; a mirrored axis flips the sign of its diagonal term.
Field: from the mounting itself (as R3), and for E also from the other E mounting (same cathode, r = 0.97), so no event
enters both sides. Errors: bootstrap over events (100).
-> $WORK/audit/q16/qE_axes.json
"""
import json
import numpy as np
from r3_lib import load_mounting, gap_field, attach_field, G0, BETA0
import c_lib as CL

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
fac = G0 / (1 + BETA0)
rng = np.random.default_rng(11)


def coefs(D):
    out = {}
    for p in 'xy':
        w = D.wx if p == 'x' else D.wy; tr = D.trx if p == 'x' else D.trY; gh = D.ghx if p == 'x' else D.ghy
        y = w - tr - gh
        X = np.vstack([np.ones(D.n), fac * D.glx, fac * D.gly]).T
        out[p] = CL.lsq_clip(X, y, 3.5)[0][1:]
    return np.r_[out['x'], out['y']]          # b_xx, b_xy, b_yx, b_yy


def run(D, nb=100):
    c0 = coefs(D)
    B = np.array([coefs(D.sub(rng.integers(0, D.n, D.n))) for _ in range(nb)])
    return dict(zip(['b_xx', 'b_xy', 'b_yx', 'b_yy'], [[float(a), float(e)] for a, e in zip(c0, B.std(0))]))


res = {}
L = {n: load_mounting(n) for n in ('A628', 'B', 'E623', 'E624')}
F = {n: gap_field(L[n][0]) for n in L}
for n, (D, info, bundle) in L.items():
    spl, c, _ = F[n]
    attach_field(D, bundle, spl, c)
    res[n] = dict(own=run(D))
    if n.startswith('E'):
        o = 'E624' if n == 'E623' else 'E623'
        attach_field(D, bundle, F[o][0], F[o][1])
        res[n]['other_E_field'] = run(D)
    for k, v in res[n].items():
        print(n, k, '  '.join(f'{a} {x:+.3f}+-{e:.3f}' for a, (x, e) in v.items()), flush=True)
    # mean gradient magnitudes, to see how much lever arm each mounting has
    res[n]['rms_grad'] = dict(x=float(np.std(fac * D.glx)), y=float(np.std(fac * D.gly)))
    print(n, 'rms of fac*grad lf: x %.4f y %.4f' % (res[n]['rms_grad']['x'], res[n]['rms_grad']['y']))
json.dump(res, open(f'{OUT}/qE_axes.json', 'w'), indent=1)
