"""r4_08_summary: beta_eff(V), coefficients against V^2 and V, comparisons, mesh scan, time stability, pressure
estimate -> summary.json, tables.md; figures fig1..fig5 in $WORK/audit/r4."""
import sys, json, pickle
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import r4_lib as L
A = json.load(open(f'{L.OUT}/analysis.json')); U = A['units']
G0 = L.G0
SC = L.SC
def u(sub, run=SC): return U[f'{run}/{sub}']
drift = {V: u(f'drift_scan_resist_490V_drift_{V}V') for V in (300, 500, 700, 900, 1100)}
g1 = U[L.GOLD1]; g2 = U[L.GOLD2]; r625 = U[L.R625]
r622 = U['mx17_det2_det3_overnight_6-22-26/long_run']
NAMES = ['c0', 'gu', 'gw', 'guu', 'gww', 'guw']
def co(r, n): return r['c'][NAMES.index(n)], r['err'][NAMES.index(n)]
S = {}
# ---------------- beta_eff(V) from mean q_u90 (T ~ 1/v ~ E^-beta at fixed gap)
pts = [(500, drift[500]['mean_q']), (700, drift[700]['mean_q']), (900, drift[900]['mean_q']), (1000, g1['mean_q']), (1100, drift[1100]['mean_q'])]
def beta_fit(pts):
    V = np.array([p[0] for p in pts], float); q = np.array([p[1] for p in pts], float)
    x = np.log(V / 1000); c = np.polyfit(x, np.log(q), 2)
    return lambda v: -(c[1] + 2 * c[0] * np.log(v / 1000)), c
bfit, bc = beta_fit(pts); bfit4, bc4 = beta_fit(pts[1:])
S['beta_eff'] = {str(v): float(bfit(v)) for v in (500, 600, 700, 800, 900, 1000, 1100)}
S['beta_eff_no500'] = {str(v): float(bfit4(v)) for v in (500, 700, 900, 1000, 1100)}
S['mean_q'] = {str(p[0]): p[1] for p in pts}
print('beta_eff', S['beta_eff'], '\n no500', S['beta_eff_no500'])
def mm(val, err, V, beta=None):
    b = bfit(V) if beta is None else beta
    f = G0 / (1 + b)
    return val * f, err * f
# ---------------- main table
rows = []
def addrow(label, r, V, mesh, beta_ref=True):
    c, e = r['c'], r['err']
    row = dict(label=label, mesh=mesh, drift=V, N=r['N'], mean_q=r['mean_q'], c=c, e=e, amp=r['amp_fine'])
    rows.append(row)
for V in (300, 500, 700, 900, 1100):
    addrow(f'6-27 drift scan {V} V', drift[V], V, 490)
addrow('6-27 long run (golden run 1)', g1, 1000, 490)
addrow('6-28 p2 overnight (golden run 2)', g2, 1000, 490)
addrow('6-25 day (bottom slot)', r625, 500, 495)
addrow('6-22 overnight long run', r622, 1000, '?')
hv = [(k, v) for k, v in U.items() if 'hv_scan' in k]
for k, v in sorted(hv, key=lambda kv: (int(kv[0].split('resist_')[1].split('V')[0]))):
    addrow('6-27 ' + k.split('/')[1], v, 1000, int(k.split('resist_')[1].split('V')[0]))
S['rows'] = rows
# ---------------- coefficient against V^2 and V
def wfit(x, y, e):
    x, y, e = map(np.array, (x, y, e)); w = 1 / e ** 2
    X = np.vstack([np.ones_like(x), x]).T
    C = np.linalg.inv(X.T @ (X * w[:, None])); b = C @ (X.T @ (w * y))
    chi2 = float((((y - X @ b) / e) ** 2).sum())
    return b, C, chi2
fits = {}
Vs = [700, 900, 1000, 1100]
units = [drift[700], drift[900], g1, drift[1100]]
for name in ('guu', 'gww', 'guw', 'gu', 'gw'):
    for conv in ('ln', 'mm'):
        yv, ev = [], []
        for V, r in zip(Vs, units):
            val, er = co(r, name)
            if conv == 'mm': val, er = mm(val, er, V)
            yv.append(val); ev.append(er)
        v500, e500 = co(drift[500], name)
        if conv == 'mm': v500, e500 = mm(v500, e500, 500)
        for xf, xl in ((lambda V: (V / 1000) ** 2, 'V2'), (lambda V: V / 1000, 'V')):
            b, C, chi2 = wfit([xf(V) for V in Vs], yv, ev)
            pred = b[0] + b[1] * xf(500); perr = float(np.sqrt(C[0, 0] + 2 * C[0, 1] * xf(500) + C[1, 1] * xf(500) ** 2))
            fits[f'{name}_{conv}_{xl}'] = dict(intercept=float(b[0]), slope=float(b[1]), slope_err=float(np.sqrt(C[1, 1])), chi2=chi2,
                                                dof=2, pred500=float(pred), pred500_err=perr, obs500=float(v500), obs500_err=float(e500),
                                                dev_sigma=float((v500 - pred) / np.hypot(perr, e500)))
S['vfits_700_1100'] = fits
# V^2 line through golden (1000 V) and 6-25 (500 V): prediction at 700, 900, 1100
for name in ('gww', 'guw', 'guu'):
    v1, e1 = co(g1, name); v6, e6 = co(r625, name)
    sl = (v1 - v6) / (1.0 - 0.25)
    pr = {}
    for V in (700, 900, 1100):
        pred = v6 + sl * ((V / 1000) ** 2 - 0.25)
        ob, eo = co(drift[V], name)
        # error of prediction from the two end points
        a = (((V / 1000) ** 2 - 0.25) / 0.75)
        pe = np.sqrt((1 - a) ** 2 * e6 ** 2 + a ** 2 * e1 ** 2)
        pr[V] = dict(pred=float(pred), obs=float(ob), err_obs=float(eo), err_pred=float(pe), dev_sigma=float((ob - pred) / np.hypot(eo, pe)))
    S[f'V2_line_6-25_to_golden_{name}'] = pr
# equal-pressure-step argument
dp = lambda V: 0.5 * 8.854e-12 * (V / 5e-3) ** 2
S['pressure_Pa'] = {str(V): dp(V) for V in (300, 500, 700, 900, 1000, 1100)}
S['dp_700_1100'] = dp(1100) - dp(700); S['dp_500_1000'] = dp(1000) - dp(500)
for conv in ('ln', 'mm'):
    ch = {}
    for name in ('gww', 'guw', 'guu'):
        def g(r, V):
            v, e = co(r, name)
            return mm(v, e, V) if conv == 'mm' else (v, e)
        a, ea = g(drift[700], 700); b, eb = g(drift[1100], 1100); c, ec = g(drift[500], 500); d_, ed = g(g1, 1000)
        S[f'step_{name}_{conv}'] = dict(change_700_to_1100=[b - a, float(np.hypot(ea, eb))], change_500_to_1000=[d_ - c, float(np.hypot(ec, ed))])
# fixed gap shape expectation at 500 V
gw1 = co(g1, 'gww'); b5, b10 = bfit(500), bfit(1000)
S['gww_expected_500_fixed_shape'] = [gw1[0] * (1 + b5) / (1 + b10), gw1[1] * (1 + b5) / (1 + b10)]
S['gww_obs_500'] = list(co(drift[500], 'gww'))
S['amp_fine_expected_500'] = (1 + b5) / (1 + b10)
# direct 500 V (6-27) vs 6-25
cmp = {}
for n in ('gu', 'gw', 'guu', 'gww', 'guw'):
    a, ea = co(drift[500], n); b, eb = co(r625, n)
    cmp[n] = dict(v627_500=[a, ea], v625=[b, eb], diff=b - a, sigma=float((b - a) / np.hypot(ea, eb)))
cmp['amp_fine'] = dict(v627_500=drift[500]['amp_fine'], v625=r625['amp_fine'], diff=r625['amp_fine'][0] - drift[500]['amp_fine'][0],
                       sigma=float((r625['amp_fine'][0] - drift[500]['amp_fine'][0]) / np.hypot(*(drift[500]['amp_fine'][1], r625['amp_fine'][1]))))
S['cmp_500_vs_625'] = cmp
cmp2 = {}
for n in ('gu', 'gw', 'guu', 'gww', 'guw'):
    a, ea = co(g1, n); b, eb = co(r625, n)
    cmp2[n] = dict(golden=[a, ea], v625=[b, eb], diff=b - a, sigma=float((b - a) / np.hypot(ea, eb)))
S['cmp_golden_vs_625'] = cmp2
cmp3 = {}
for n in ('gu', 'gw', 'guu', 'gww', 'guw'):
    a, ea = co(g1, n); b, eb = co(r622, n)
    cmp3[n] = dict(golden=[a, ea], v622=[b, eb], diff=b - a, sigma=float((b - a) / np.hypot(ea, eb)))
S['cmp_golden_vs_622'] = cmp3
cmp4 = {}
for n in ('gu', 'gw', 'guu', 'gww', 'guw'):
    a, ea = co(g1, n); b, eb = co(drift[500], n)
    cmp4[n] = dict(golden=[a, ea], v500=[b, eb], diff=b - a, sigma=float((b - a) / np.hypot(ea, eb)))
S['cmp_golden_vs_500'] = cmp4
# ---------------- mesh scan
meshrows = [r for r in rows if r['label'].startswith('6-27 hv_scan')]
mv = np.array([r['mesh'] for r in meshrows], float)
ms = {}
for n, j in (('guu', 3), ('gww', 4), ('guw', 5), ('gu', 1), ('gw', 2)):
    y = np.array([r['c'][j] for r in meshrows]); e = np.array([r['e'][j] for r in meshrows])
    b, C, chi2 = wfit((mv - 475) / 100, y, e)
    mean = (y / e ** 2).sum() / (1 / e ** 2).sum(); mchi = (((y - mean) / e) ** 2).sum()
    ms[n] = dict(mean=float(mean), err=float((1 / e ** 2).sum() ** -0.5), chi2_const=float(mchi), dof=len(y) - 1, slope_per100V=float(b[1]),
                 slope_err=float(np.sqrt(C[1, 1])))
y = np.array([r['amp'][0] for r in meshrows]); e = np.array([r['amp'][1] for r in meshrows])
b, C, chi2 = wfit((mv - 475) / 100, y, e); mean = (y / e ** 2).sum() / (1 / e ** 2).sum()
ms['amp_fine'] = dict(mean=float(mean), err=float((1 / e ** 2).sum() ** -0.5), chi2_const=float((((y - mean) / e) ** 2).sum()), dof=len(y) - 1,
                      slope_per100V=float(b[1]), slope_err=float(np.sqrt(C[1, 1])))
ms['mean_q_range'] = [min(r['mean_q'] for r in meshrows), max(r['mean_q'] for r in meshrows)]
S['mesh_scan'] = ms
# ---------------- time blocks
tb = {}
for key, bl in A['blocks'].items():
    d = {}
    for n, j in (('guu', 3), ('gww', 4), ('guw', 5)):
        y = np.array([b['c'][j] for b in bl]); e = np.array([b['err'][j] for b in bl])
        m = (y / e ** 2).sum() / (1 / e ** 2).sum()
        d[n] = dict(mean=float(m), chi2=float((((y - m) / e) ** 2).sum()), dof=len(y) - 1, rms=float(y.std()), values=y.tolist(), errs=e.tolist())
    tb[key] = d
S['blocks'] = tb
for a_, b_ in (('run6-27', 'run6-28'),):
    S['run1_vs_run2'] = {n: dict(r1=co(g1, n), r2=co(g2, n), diff=co(g2, n)[0] - co(g1, n)[0], sigma=float((co(g2, n)[0] - co(g1, n)[0]) / np.hypot(co(g1, n)[1], co(g2, n)[1])))
                         for n in ('gu', 'gw', 'guu', 'gww', 'guw')}
# ---------------- foil pressure estimate
rho_k, t_k = 1420.0, 50e-6
w_k = rho_k * t_k * 9.81
pr = {'weight_kapton_Pa': w_k}
for tc in (0.1e-6, 9e-6):
    pr[f'weight_total_cu{tc*1e6:.1f}um_Pa'] = w_k + 8960 * tc * 9.81
pr['electrostatic_Pa'] = {str(V): dp(V) for V in (300, 500, 700, 900, 1000, 1100)}
pr['ratio_1000V_to_weight'] = [dp(1000) / pr['weight_total_cu9.0um_Pa'], dp(1000) / pr['weight_total_cu0.1um_Pa']]
pr['ratio_step_500_1000_to_weight'] = [S['dp_500_1000'] / pr['weight_total_cu9.0um_Pa'], S['dp_500_1000'] / pr['weight_total_cu0.1um_Pa']]
S['foil'] = pr
json.dump(S, open(f'{L.OUT}/summary.json', 'w'), indent=1, default=float)
print(json.dumps({k: S[k] for k in ('gww_expected_500_fixed_shape', 'gww_obs_500', 'amp_fine_expected_500', 'cmp_500_vs_625', 'step_gww_ln', 'step_gww_mm', 'foil', 'mesh_scan', 'run1_vs_run2')}, indent=1, default=float)[:6000])
for k, v in fits.items():
    if k.startswith('gww') or k.startswith('guw'): print(k, {a: round(b, 4) for a, b in v.items()})
for k in ('gww', 'guw', 'guu'): print(k, S[f'V2_line_6-25_to_golden_{k}'])
print({k: {n: (round(v['mean'], 4), round(v['chi2'], 1), v['dof']) for n, v in d.items()} for k, d in tb.items()})
