"""qE_gas.py -- chamber E: is the drift-time gradient along y a gap or a gas (velocity) gradient?

Per y slice (and per x slice as a control): the angle-scale velocity (slope of the per-row normalised w on tan_ref, per
view, robust) and the drift time (median walk-corrected ln q_u90). A gap change gives d ln v / d ln T = -beta/(1+beta)
(-0.46 at E's beta 0.84); a change of the gas at a fixed gap gives -1. Chamber A golden as the control (the paper's
smooth Y-velocity value is -0.45). Errors: bootstrap over events (100). -> $WORK/audit/q16/qE_gas.json
"""
import json
import numpy as np
from r3_lib import load_mounting
import c_lib as CL

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
rng = np.random.default_rng(5)


def slice_vals(D, m):
    v = []
    for w, t in ((D.wx, D.trx), (D.wy, D.trY)):
        s = m & (np.abs(t) <= 0.6) & np.isfinite(w)
        X = np.vstack([np.ones(s.sum()), t[s]]).T
        v.append(CL.lsq_clip(X, w[s], 3.5)[0][1])
    lq = D.lnq[m & np.isfinite(D.lnq)]
    return np.log(np.mean(v)), np.log(v[0]), np.log(v[1]), np.median(lq)


def run(D, axis, nsl=4):
    c = D.W if axis == 'y' else D.U
    e = np.quantile(c, np.linspace(0, 1, nsl + 1))
    idx = [(c >= e[i]) & (c <= e[i + 1]) for i in range(nsl)]
    def slope(sel_idx):
        P = np.array([slice_vals(D, s) for s in sel_idx])
        lv, lT = P[:, 0], P[:, 3]
        return np.polyfit(lT - lT.mean(), lv - lv.mean(), 1)[0], P
    k0, P = slope(idx)
    B = []
    for _ in range(100):
        i = rng.integers(0, D.n, D.n)
        bb = np.zeros(D.n, bool); np.add.at(bb, i, True)          # approximate: resample membership
        B.append(slope([s & bb for s in idx])[0])
    return dict(k=float(k0), err=float(np.std(B)), lnv=P[:, 0].tolist(), lnT=P[:, 3].tolist(),
                dlnT_range=float(P[:, 3].max() - P[:, 3].min()), dlnv_range=float(P[:, 0].max() - P[:, 0].min()))


res = {}
for name in ('A628', 'E623', 'E624'):
    D, info, bundle = load_mounting(name)
    res[name] = {ax: run(D, ax) for ax in ('y', 'x')}
    for ax, r in res[name].items():
        print(f'{name} along {ax}: d ln v / d ln T = {r["k"]:+.2f} +- {r["err"]:.2f}   (range ln T {r["dlnT_range"]:.3f}, ln v {r["dlnv_range"]:.3f})', flush=True)
json.dump(res, open(f'{OUT}/qE_gas.json', 'w'), indent=1)
