"""r1_09_resid.py: residual maps (measured - predicted), curl test of the residual, reproducibility, hp-quadratic mesh variant. -> part4_resid.json, resid.pkl"""
import sys
from r1_lib2 import *
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
rng = np.random.default_rng(5)
J = {}; SAVE = {}
N = 8
SIZE = (340.0, 302.0)

def hp_quad():
    """per-view quadratic of the position-route mesh height from the pooled golden events, n = 6 cells (mm, U/W in 100 mm)"""
    out = {}
    for p in 'xy':
        ex, ey = np.linspace(*CL.UL, 7), np.linspace(*CL.WL, 7)
        rows = []
        for r in ('run1', 'run2'):
            d = D[r]['tab']
            rows.append(d)
        d = pd.concat(rows, ignore_index=True)
        i, j = np.searchsorted(ex, d.U.values) - 1, np.searchsorted(ey, d.W.values) - 1
        Uc, Wc, h, he = [], [], [], []
        for a in range(6):
            for b in range(6):
                m = (i == a) & (j == b); dc = d[m]
                _, hh, _, e, nn = CO.line(dc[f'tan_ref_{p}'].values, dc[f'{p}_p0'].values - dc[f'ref_mesh_{p}'].values)
                if np.isfinite(hh):
                    Uc.append(0.5 * (ex[a] + ex[a + 1])); Wc.append(0.5 * (ey[b] + ey[b + 1])); h.append(hh); he.append(e)
        Uc, Wc, h, he = map(np.array, (Uc, Wc, h, he))
        X = np.vstack([np.ones(len(h)), Uc, Wc, Uc ** 2 / 2, Wc ** 2 / 2, Uc * Wc]).T
        w = 1 / he ** 2
        c = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ h)
        out[p] = dict(gu=c[1], gw=c[2], kuu=c[3], kww=c[4], kuw=c[5])
    return out
HPQ = hp_quad(); J['hp_quad'] = {p: {k: float(x) for k, x in v.items()} for p, v in HPQ.items()}
print('hp quad', J['hp_quad'], flush=True)
def hgrad_hp(U, W, p):
    h = HPQ[p]
    return ((h['gu'] + h['kuu'] * U + h['kuw'] * W) / 100) if p == 'x' else ((h['gw'] + h['kww'] * W + h['kuw'] * U) / 100)

def meas(r, p, n):
    v = V[(r, p)]; vg = CO.line(v['tr'], v['w'])[1]
    M = CL.cellmap(v['U'], v['W'], v['w'], v['tr'], n)
    return M, M['w0'] / vg, M['w0e'] / vg

def fine(M, y, e):
    ok = np.isfinite(y) & np.isfinite(e)
    fl = 0.5 * np.median(e[ok]); w = 1 / (e[ok] ** 2 + fl ** 2)
    X = np.vstack([np.ones(ok.sum()), M['Uc'][ok], M['Wc'][ok]]).T
    c = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ y[ok])
    out = np.full_like(y, np.nan); out[ok] = y[ok] - X @ c
    return out

def grad(A, axis, size):
    d = size / A.shape[0]; out = np.full_like(A, np.nan)
    if axis == 0: out[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * d)
    else: out[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2 * d)
    return out

def cov(a, b):
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    return float(np.mean((a - a.mean()) * (b - b.mean())))

def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[ok], b[ok])[0, 1])

def predmap(r, p, field, hw, b=0.5, a=1.0):
    v = V[(r, p)]; i = 0 if p == 'x' else 1
    if hw == 'hp':
        gh = hgrad_hp(v['U'], v['W'], p)
    else:
        gh = hgrad(v['U'], v['W'], hw, BUNDLE625 if r == 'r625' else BUNDLE)[i]
    gg = field.grad(v['U'], v['W'])[i]
    return cellmean(v['U'], v['W'], a * gh + b * gg, N)

for kind in ('T', 'q'):
    for hw in ('perview', 'h', 'hp'):
        R = {}
        for dst, src in (('run1', 'run2'), ('run2', 'run1')):
            for p in 'xy':
                M, A, Ae = meas(dst, p, N)
                P = predmap(dst, p, Fld(f'{kind}_{src}'), hw)
                ok = np.isfinite(A) & np.isfinite(P)
                A0 = A - np.nanmean(A[ok]); P0 = P - np.nanmean(P[ok])
                R[(dst, p)] = dict(A=A0, P=P0, res=A0 - P0, Af=fine(M, A, Ae), Pf=fine(M, P, Ae), M=M, Ae=Ae)
                R[(dst, p)]['resf'] = fine(M, A0 - P0, Ae)
        key = f'{kind}:{hw}'
        out = {}
        for p in 'xy':
            a1, a2 = R[('run1', p)], R[('run2', p)]
            out[p] = dict(
                rms_meas=float(np.nanstd(np.concatenate([a1['A'], a2['A']]))), rms_res=float(np.nanstd(np.concatenate([a1['res'], a2['res']]))),
                rms_meas_fine=float(np.nanstd(np.concatenate([a1['Af'], a2['Af']]))), rms_res_fine=float(np.nanstd(np.concatenate([a1['resf'], a2['resf']]))),
                sig_meas_fine=float(np.sqrt(max(cov(a1['Af'], a2['Af']), 0))), cov_res_fine=cov(a1['resf'], a2['resf']),
                r_res_fine=pearson(a1['resf'], a2['resf']), r_meas_fine=pearson(a1['Af'], a2['Af']),
                var_frac_explained_full=float(1 - np.nanvar(np.concatenate([a1['res'], a2['res']])) / np.nanvar(np.concatenate([a1['A'], a2['A']]))),
                var_frac_explained_fine=float(1 - np.nanvar(np.concatenate([a1['resf'], a2['resf']])) / np.nanvar(np.concatenate([a1['Af'], a2['Af']]))),
                mean_err=float(np.nanmean(np.concatenate([a1['Ae'], a2['Ae']]))))
            # reproducible part of the residual relative to the reproducible part of the measured fine map
            out[p]['repro_resid_over_meas'] = float(cov(a1['resf'], a2['resf']) / cov(a1['Af'], a2['Af']))
        # curl test: measured and residual, cross-run Cx (view x, run a) against Cy (view y, run b != a)
        for what in ('A', 'res', 'Af', 'resf'):
            vals = []
            for a_, b_ in (('run1', 'run2'), ('run2', 'run1')):
                X_ = R[(a_, 'x')][what]; Y_ = R[(b_, 'y')][what]
                Cx = grad(fine(R[(a_, 'x')]['M'], X_, R[(a_, 'x')]['Ae']), 1, SIZE[1]); Cy = grad(fine(R[(b_, 'y')]['M'], Y_, R[(b_, 'y')]['Ae']), 0, SIZE[0])
                ok = np.isfinite(Cx) & np.isfinite(Cy)
                vals.append((Cx[ok], Cy[ok]))
            cx = np.concatenate([x for x, y in vals]); cy = np.concatenate([y for x, y in vals])
            r0 = float(np.corrcoef(cx, cy)[0, 1]); s0 = float(np.sum(cx * cy) / np.sum(cx * cx))
            bs = []
            for _ in range(1000):
                ii = rng.integers(0, len(cx), len(cx)); bs.append((np.corrcoef(cx[ii], cy[ii])[0, 1], np.sum(cx[ii] * cy[ii]) / np.sum(cx[ii] ** 2)))
            bs = np.array(bs)
            out[f'curl_{what}'] = dict(r=r0, r_err=float(bs[:, 0].std()), slope=s0, slope_err=float(bs[:, 1].std()), rms_Cx=float(cx.std()), rms_Cy=float(cy.std()), N=len(cx))
        J[key] = out
        SAVE[key] = {k: {kk: vv for kk, vv in v.items()} for k, v in R.items()}
        print(key, {p: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in out[p].items() if not k.startswith('curl')} for p in 'xy'}, flush=True)
        print('  curl', {k: (round(v['r'], 2), round(v['slope'], 2)) for k, v in out.items() if k.startswith('curl')}, flush=True)
# 6-25 own (q field), residual and in-row halves for the curl test
for hw in ('perview', 'h', 'hp'):
    R = {}
    for p in 'xy':
        M, A, Ae = meas('r625', p, N)
        P = predmap('r625', p, Fld('q_r625'), hw)
        ok = np.isfinite(A) & np.isfinite(P)
        A0 = A - np.nanmean(A[ok]); P0 = P - np.nanmean(P[ok])
        R[p] = dict(A=A0, P=P0, res=A0 - P0, Af=fine(M, A, Ae), Pf=fine(M, P, Ae), M=M, Ae=Ae); R[p]['resf'] = fine(M, A0 - P0, Ae)
    out = {}
    for p in 'xy':
        out[p] = dict(rms_meas=float(np.nanstd(R[p]['A'])), rms_res=float(np.nanstd(R[p]['res'])), rms_meas_fine=float(np.nanstd(R[p]['Af'])),
                      rms_res_fine=float(np.nanstd(R[p]['resf'])), mean_err=float(np.nanmean(R[p]['Ae'])),
                      var_frac_explained_full=float(1 - np.nanvar(R[p]['res']) / np.nanvar(R[p]['A'])),
                      var_frac_explained_fine=float(1 - np.nanvar(R[p]['resf']) / np.nanvar(R[p]['Af'])),
                      r_fine_pred_meas=pearson(R[p]['Af'], R[p]['Pf']))
    # curl within the row: cell maps from two random halves of the events (noise independent)
    for what in ('Af', 'resf'):
        pass
    J[f'r625:{hw}'] = out; SAVE[f'r625:{hw}'] = R
    print('r625', hw, out, flush=True)
# 6-25 split-half reproducibility of the residual (halves of events; the field is built from all events, which is fine since it uses no tilt)
def halfmaps(p, k):
    v = V[('r625', p)]; n = len(v['w']); h = np.random.default_rng(300 + k).random(n) < 0.5
    out = []
    for m in (h, ~h):
        vv = {q: v[q][m] for q in ('U', 'W', 'w', 'tr')}
        vg = CO.line(vv['tr'], vv['w'])[1]
        M = CL.cellmap(vv['U'], vv['W'], vv['w'], vv['tr'], 6)
        i = 0 if p == 'x' else 1
        gh = hgrad(vv['U'], vv['W'], 'perview', BUNDLE625)[i]; gg = Fld('q_r625').grad(vv['U'], vv['W'])[i]
        P = cellmean(vv['U'], vv['W'], gh + 0.5 * gg, 6)
        A = M['w0'] / vg; Ae = M['w0e'] / vg
        out.append((M, A - np.nanmean(A), P - np.nanmean(P[np.isfinite(A)]), Ae))
    return out
H = {}
for p in 'xy':
    r_, c_, s_ = [], [], []
    for k in range(5):
        (M1, A1, P1, e1), (M2, A2, P2, e2) = halfmaps(p, k)
        rf1 = fine(M1, A1 - P1, e1); rf2 = fine(M2, A2 - P2, e2); af1 = fine(M1, A1, e1); af2 = fine(M2, A2, e2)
        r_.append(pearson(rf1, rf2)); c_.append(cov(rf1, rf2) / cov(af1, af2)); s_.append(np.sqrt(max(cov(af1, af2), 0)))
    H[p] = dict(r_res_halves=float(np.mean(r_)), repro_resid_over_meas=float(np.mean(c_)), sig_meas_fine_halves=float(np.mean(s_)))
J['r625_halves_n6'] = H; print('halves', H)
json.dump(J, open(f'{OUT}/part4_resid.json', 'w'), indent=1, default=float)
pickle.dump(SAVE, open(f'{OUT}/resid.pkl', 'wb'))
