"""r4_06_check_template: (1) events-table selection vs the gate-table (telescope matched) selection on golden run 1;
(2) golden template from run 2 (spline of covariate-corrected ln q_u90), its quadratic and 'fine' (ridge) parts, and the
amplitude of the template in every sub-run (quadratic free)."""
import sys, glob, os, json, pickle
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import numpy as np, pandas as pd
import r4_lib as L
import r1_lib as R1
import gapmap as GM
K, LAM = 12, 0.1
GRID_U = np.linspace(-1.70, 1.68, 40); GRID_W = np.linspace(-1.50, 1.50, 36)
UU, WW = np.meshgrid(GRID_U, GRID_W, indexing='ij')

def corrected(d, f=None):
    f = f or L.fit_shape(d, nboot=0 + 1)
    c = f['c']
    A = L.design(d, ('walk', 'tan'))
    return d.lq.values - A[:, 6:] @ c[6:9], c

def spline_map(d, nb=0, seed=2):
    y, c = corrected(d)
    spl = R1.Spl(K); B = spl.basis(d.U.values, d.W.values)
    coef, keep = spl.fit(d.U.values, d.W.values, y - y.mean(), LAM, B=B)
    Bg = spl.basis(UU.ravel(), WW.ravel())
    m = (Bg @ coef).reshape(UU.shape)
    bm = []
    rng = np.random.default_rng(seed)
    for _ in range(nb):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
        cb, _ = spl.fit(d.U.values, d.W.values, y - y.mean(), LAM, wts=w, B=B)
        bm.append((Bg @ cb).reshape(UU.shape))
    return m, bm

def quad_basis(U, W):
    return np.vstack([np.ones(U.size), U, W, U * U / 2, W * W / 2, U * W]).T

def split(m):
    Q = quad_basis(UU.ravel(), WW.ravel())
    c = np.linalg.lstsq(Q, m.ravel(), rcond=None)[0]
    quad = (Q @ c).reshape(m.shape)
    return quad, m - quad, c

if __name__ == '__main__':
    out = {}
    # ---------- (1) selection check on golden run 1
    ev = L.load_events(L.GOLD1)
    gate = GM.load_table(L.WORK, L.REL if hasattr(L, 'REL') else GM.RUNS['run1']['rel'])
    gate = gate.copy()
    gate['xm_c'] = gate.x_p0 + 14 * gate.x_tan_theta; gate['ym_c'] = gate.y_p0 + 14 * gate.y_tan_theta
    gate['q'] = 0.5 * (gate.x_q_u90 + gate.y_q_u90); gate['lq'] = np.log(gate.q)
    gate['lQ'] = np.log(0.5 * (gate.x_q_sum + gate.y_q_sum))
    gate['U'] = (gate.xm_c - 200) / 100; gate['W'] = (gate.ym_c - 200) / 100
    gate['atx'] = gate.x_tan_theta.abs(); gate['aty'] = gate.y_tan_theta.abs()
    gate = gate[(gate.q > 0)].reset_index(drop=True)
    fe = L.fit_shape(ev, nboot=200); fg = L.fit_shape(gate, nboot=200)
    # intersection via (x_p0, y_p0) key
    ke = set(zip(ev.x_p0.round(4), ev.y_p0.round(4))); kg = set(zip(gate.x_p0.round(4), gate.y_p0.round(4)))
    print('events', len(ev), 'gate', len(gate), 'common', len(ke & kg))
    me, _ = spline_map(ev); mg, _ = spline_map(gate)
    qe, fe_, _ = split(me); qg, fg_, _ = split(mg)
    r_tot = np.corrcoef(me.ravel(), mg.ravel())[0, 1]; r_fine = np.corrcoef(fe_.ravel(), fg_.ravel())[0, 1]
    print('shape events', np.round(fe['c'][1:6], 4), np.round(fe['err'][1:6], 4))
    print('shape gate  ', np.round(fg['c'][1:6], 4), np.round(fg['err'][1:6], 4))
    print('map r total %.3f fine %.3f; rms diff (ln) %.4f, rms events map %.4f gate map %.4f' % (
        r_tot, r_fine, np.std(me - mg), np.std(me), np.std(mg)))
    out['selcheck'] = dict(N_events=len(ev), N_gate=len(gate), common=len(ke & kg), c_events=fe['c'][:6].tolist(),
                           e_events=fe['err'][:6].tolist(), c_gate=fg['c'][:6].tolist(), e_gate=fg['err'][:6].tolist(),
                           r_total=r_tot, r_fine=r_fine, rms_diff=float(np.std(me - mg)), rms_ev=float(np.std(me)),
                           rms_gate=float(np.std(mg)), rms_fine_ev=float(np.std(fe_)), rms_fine_gate=float(np.std(fg_)))
    # ---------- (2) template from run 2 (independent events from the 6-27 sub-runs and run 1)
    d2 = L.load_events(L.GOLD2)
    m2, bm2 = spline_map(d2, nb=40)
    q2, f2, cq2 = split(m2)
    me1_q, me1_f, _ = split(me)
    r12 = np.corrcoef(f2.ravel(), me1_f.ravel())[0, 1]
    sl12 = np.polyfit(f2.ravel(), me1_f.ravel(), 1)[0]
    print('template fine run2 rms %.4f; run1 fine vs run2 fine r %.3f slope %.3f' % (np.std(f2), r12, sl12))
    out['template'] = dict(rms_total=float(np.std(m2)), rms_fine=float(np.std(f2)), r_run1_run2_fine=float(r12), slope=float(sl12),
                           quad_coef=cq2.tolist())
    pickle.dump(dict(m2=m2, q2=q2, f2=f2, bm2=bm2, GRID=(GRID_U, GRID_W), m1=me), open(f'{L.OUT}/template.pkl', 'wb'))
    json.dump(out, open(f'{L.OUT}/selcheck_template.json', 'w'))
