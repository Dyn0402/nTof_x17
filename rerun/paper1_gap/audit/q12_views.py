"""q12_views.py -- item 12, third test: is the fine charge-drift-time link common to both views (a gap, a field, a path
length) or view-specific (channel gains, strip effects)?  Fine 8x8 cell maps of ln q_sum and ln q_u90 per view, unsaturated
tracks, cross-run slopes d ln q_a / d ln T_b for a, b in {x, y}.  -> $WORK/audit/q16/q12_views.json"""
import json
import numpy as np
import q12_gain as Q    # reuses the loaded golden mounting, cells and helpers (runs its own tests first)
import d_lib as DL
import r3_lib

D = Q.D
qv, tv = {}, {}
for p in 'xy':
    qv[p] = np.concatenate([np.log(DL.load(k)[f'{p}_q_sum'].values) for k in r3_lib.MOUNT['A628']['rows']])
    tv[p] = np.concatenate([np.log(DL.load(k)[f'{p}_q_u90'].values) for k in r3_lib.MOUNT['A628']['rows']])
sel = Q.sat == 0


def maps(arr, wts):
    out = {}
    for r in (0, 1):
        m = sel & Q.inside & (D.row == r) & np.isfinite(arr)
        out[r] = Q.fine(Q.cmean(np.nan_to_num(arr), m, wts))
    return out


def slope(A, B):   # d A / d B, cross-run, noise-free denominator
    return (Q.cov(A[0], B[1]) + Q.cov(A[1], B[0])) / (2 * Q.cov(B[0], B[1]))


def all_slopes(wts):
    M = {f'q{p}': maps(qv[p], wts) for p in 'xy'}
    M.update({f'T{p}': maps(tv[p], wts) for p in 'xy'})
    return {f'{a}_on_{b}': slope(M[a], M[b]) for a in ('qx', 'qy') for b in ('Tx', 'Ty')} | \
           {'Tx_on_Ty': slope(M['Tx'], M['Ty']), 'qx_on_qy': slope(M['qx'], M['qy'])}


k0 = all_slopes(np.ones(D.n))
B = [all_slopes(Q.rng.poisson(1.0, D.n).astype(float)) for _ in range(150)]
res = {k: (float(v), float(np.nanstd([b[k] for b in B]))) for k, v in k0.items()}
print(json.dumps(res, indent=0))
json.dump(res, open(f'{Q.OUT}/q12_views.json', 'w'), indent=1)
