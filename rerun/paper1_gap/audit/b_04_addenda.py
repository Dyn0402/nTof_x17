#!/usr/bin/env python3
"""b_04_addenda.py: (6) gain confound of the T map, (7) time blocks within each run."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gapmap as GM, cellobs as CO
from b_lib import *
from b_01_maps import XE, YE
LOG = []; J = {}
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)
G0 = 28.2
data = {}
for run, R in GM.RUNS.items():
    tab = GM.load_table(WORK, R['rel'])
    m, info = GM.colsum_T(WORK, run, tab)
    m = m.copy()
    m['lnq'] = np.log(0.5 * (m.xq + m.yq))
    data[run] = (tab, m, info)
    P(run, 'event-level amplitude slope T per ln q:', round(info['T_slope_per_lnq'], 2), 'ns; median q', float(np.median(0.5 * (m.xq + m.yq))))
    # per-view event-level slopes
    for v in 'xy':
        lq = np.log(m[v + 'q']); A = np.vstack([np.ones(len(m)), lq - lq.median()]).T
        c = np.linalg.lstsq(A, m[v + 'T'].values, rcond=None)[0]
        P(f'   view {v}: event-level slope of {v}T on ln {v}q = {c[1]:.2f} ns')
        m[f'{v}Tc'] = m[v + 'T'] - c[1] * (lq - lq.median())
        J[f'slope_{run}_{v}'] = float(c[1])
    m['T_pv'] = 0.5 * (m.xTc + m.yTc)

def cellmap(m, col, n, f=GM.tmean, err=True):
    ex, ey = np.linspace(*XE, n + 1), np.linspace(*YE, n + 1)
    im, jm = GM.cell_of(m.xm.values, m.ym.values, ex, ey)
    A = np.full((n, n), np.nan); E = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            x = m[col].values[(im == a) & (jm == b)]
            if len(x) > 20:
                A[a, b] = f(x); E[a, b] = GM.boot(f, x, nb=80) if err else 0
    return A, E

P('\n##### (6a) gain map and T_amp correlation across cells (cross-run)')
out6 = {}
for n in (4, 6, 8):
    maps = {}
    for run in RUNS:
        m = data[run][1]
        lq, lqe = cellmap(m, 'lnq', n, f=lambda x: float(np.mean(x)))
        lqe = np.where(np.isfinite(lqe), np.maximum(lqe, 1e-3), np.nan)
        Tr, Tre = cellmap(m, 'T', n); Ta, Tae = cellmap(m, 'T_amp', n); Tp, Tpe = cellmap(m, 'T_pv', n)
        maps[run] = dict(lq=lq, lqe=lqe, Tr=Tr, Ta=Ta, Tae=Tae, Tp=Tp, Tpe=Tpe, Tre=Tre)
    out6[n] = maps
    for lab, key in (('raw T', 'Tr'), ('T_amp (global slope removed)', 'Ta'), ('T per-view slope', 'Tp')):
        a = [np.log(maps[r][key]).ravel() for r in RUNS]; q = [maps[r]['lq'].ravel() for r in RUNS]
        # slope d ln T / d ln q cross-run, and in ns per ln q
        s = boot_cells(lambda t1, t2, q1, q2: iv_slope(t1, t2, q1, q2), (a[0], a[1], q[0], q[1]), nb=2000)
        sn = boot_cells(lambda t1, t2, q1, q2: iv_slope(t1, t2, q1, q2), (maps['run1'][key].ravel(), maps['run2'][key].ravel(), q[0], q[1]), nb=2000)
        rr = 0.5 * (pear(maps['run1'][key].ravel(), q[1]) + pear(maps['run2'][key].ravel(), q[0]))
        P(f'n={n} {lab:30s}: cross-run slope of T on cell-mean ln q = {sn[0]:+.1f} ± {sn[1]:.1f} ns per ln q [{sn[2]:+.1f},{sn[3]:+.1f}]  (event-level removed: {J["slope_run1_x"]:.1f}/{J["slope_run2_x"]:.1f} (x), {J["slope_run1_y"]:.1f}/{J["slope_run2_y"]:.1f} (y), global {data["run1"][2]["T_slope_per_lnq"]:.1f}/{data["run2"][2]["T_slope_per_lnq"]:.1f}); raw cross-run r {rr:+.2f}')
        J[f'p6a_{n}_{key}'] = dict(slope_ns=sn[0], err=sn[1], lo=sn[2], hi=sn[3], r=rr)
    # gain map itself: quadratic residual, similarity to gap map and tilt maps
    fits = [wfit(maps[r]['lq'], maps[r]['lqe'], n, 2) for r in RUNS]
    lqs = [maps[r]['lq'].ravel() for r in RUNS]
    lqres = [f[1].ravel() for f in fits]
    P(f'n={n} gain map: rms of ln q over cells {np.nanstd(lqs[0]):.3f}/{np.nanstd(lqs[1]):.3f}; run1 x run2 r(full) {pear(lqs[0], lqs[1]):+.2f}; quadratic-residual rms {np.nanstd(lqres[0]):.3f}/{np.nanstd(lqres[1]):.3f}, r(resid) {pear(lqres[0], lqres[1]):+.2f}, signal rms {np.sqrt(max(cov(lqres[0], lqres[1]), 0)):.3f}')
    lqfit = [f[2] for f in fits]
    P(f'      gain quadratic coefs (1,U,W,U2,W2,UW): run1 {np.round(lqfit[0], 3)}  run2 {np.round(lqfit[1], 3)}')
    Fm = [MAPS[r][n] for r in RUNS]
    for lab, getter in (('ln T map', lambda r, M: np.log(M['T'])), ('w0_x map (x tilt)', lambda r, M: M['w0x']), ('w0_y map (y tilt)', lambda r, M: M['w0y']),
                        ('ln v_x map', lambda r, M: np.log(M['vx'])), ('ln v_y map', lambda r, M: np.log(M['vy']))):
        x1, x2 = getter('run1', Fm[0]).ravel(), getter('run2', Fm[1]).ravel()
        rf = 0.5 * (pear(lqs[0], x2) + pear(lqs[1], x1))
        # fine
        deg = 1 if 'w0' in lab else 2
        arr = {}
        for ri, (r, M) in enumerate(zip(RUNS, Fm)):
            if 'w0' in lab:
                k = 'w0x' if 'w0_x' in lab else 'w0y'; y, e = M[k], M[k + '_err']
            elif 'ln T' in lab: y, e = np.log(M['T']), M['T_err'] / M['T']
            else:
                k = 'vx' if 'v_x' in lab else 'vy'; y, e = np.log(M[k]), M[k + '_err'] / M[k]
            arr[ri] = wfit(y, e, n, deg)[1].ravel()
        rres = 0.5 * (pear(lqres[0], arr[1]) + pear(lqres[1], arr[0]))
        s = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (arr[0], arr[1], lqres[0], lqres[1]), nb=2000)
        P(f'      cross-run r(gain, {lab:18s}) full {rf:+.2f}   fine {rres:+.2f};  fine slope of {lab} on gain residual {s[0]:+.3f} ± {s[1]:.3f}')
        J[f'p6gain_{n}_{lab}'] = dict(r_full=rf, r_fine=rres, slope=s[0], err=s[1])

P('\n##### (6b) gap surface coefficients of ln T with different gain corrections (mm of gap: G_u = d gap/dU per 100 mm; G_ww = d2 gap / dW2; mean of the two runs)')
def surf(n, key, beta):
    out = []
    for run in RUNS:
        mp = out6[n][run]
        y = np.log(mp[key]); e = (mp['Tae'] if key == 'Ta' else mp['Tpe'] if key == 'Tp' else mp['Tre']) / mp[key]
        _, _, c = wfit(y, e, n, 2)
        out.append(np.array([c[1], c[2], 2 * c[3], 2 * c[4], c[5]]) * G0 / (1 + beta))
    return np.array(out)     # per run: G_u, G_w, G_uu, G_ww, G_uw
# cell-level IV correction: T_corr = T_raw * exp(-s_ln * (lq - mean lq)) where s_ln = cross-run slope d lnT/ d ln q on raw cell maps
for n in (4, 6, 8):
    maps = out6[n]
    # slope from cross-run IV on the log T (raw, no gain correction) vs ln q, using quadratic-residual (fine) maps so that real structure at long wavelength is kept
    sl = {}
    for variant in ('fine', 'full'):
        if variant == 'fine':
            t = [wfit(np.log(maps[r]['Tr']), maps[r]['Tre'] / maps[r]['Tr'], n, 2)[1].ravel() for r in RUNS]
            q = [wfit(maps[r]['lq'], maps[r]['lqe'], n, 2)[1].ravel() for r in RUNS]
        else:
            t = [np.log(maps[r]['Tr']).ravel() for r in RUNS]; q = [maps[r]['lq'].ravel() for r in RUNS]
        sl[variant] = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (t[0], t[1], q[0], q[1]), nb=1000)
    P(f'n={n} cell-level slope d lnT/d ln q on the RAW T: fine {sl["fine"][0]:+.4f} ± {sl["fine"][1]:.4f}, full {sl["full"][0]:+.4f} ± {sl["full"][1]:.4f}; event-level (global) d lnT/d ln q = {data["run1"][2]["T_slope_per_lnq"] / np.nanmean(maps["run1"]["Tr"]):+.4f} / {data["run2"][2]["T_slope_per_lnq"] / np.nanmean(maps["run2"]["Tr"]):+.4f}')
    for variant in ('fine', 'full'):
        for r in RUNS:
            mp = maps[r]
            mp['Tc_' + variant] = mp['Tr'] * np.exp(-sl[variant][0] * (mp['lq'] - np.nanmean(mp['lq'])))
            mp['Tc_' + variant + '_e'] = mp['Tre']
    for beta in (0.68, 1.0):
        rows = [('raw (no correction)', 'Tr', 'Tre'), ('global event slope (T_amp)', 'Ta', 'Tae'), ('per-view event slopes', 'Tp', 'Tpe')]
        for lab, key, ek in rows:
            s = np.array([wfit(np.log(out6[n][r][key]), out6[n][r][ek] / out6[n][r][key], n, 2)[2] for r in RUNS])
            cf = s[:, [1, 2, 3, 4]] * np.array([1, 1, 2, 2]) * G0 / (1 + beta)
            P(f'n={n} beta={beta} {lab:28s} G_u {cf[0,0]:+.2f}/{cf[1,0]:+.2f}  G_w {cf[0,1]:+.2f}/{cf[1,1]:+.2f}  G_uu {cf[0,2]:+.2f}/{cf[1,2]:+.2f}  G_ww {cf[0,3]:+.2f}/{cf[1,3]:+.2f}   (run1/run2)')
            J[f'p6b_{n}_{beta}_{key}'] = cf.tolist()
        for variant in ('fine', 'full'):
            key = 'Tc_' + variant
            s = np.array([wfit(np.log(out6[n][r][key]), out6[n][r]['Tre'] / out6[n][r][key], n, 2)[2] for r in RUNS])
            cf = s[:, [1, 2, 3, 4]] * np.array([1, 1, 2, 2]) * G0 / (1 + beta)
            P(f'n={n} beta={beta} cell-slope ({variant:4s}) corrected      G_u {cf[0,0]:+.2f}/{cf[1,0]:+.2f}  G_w {cf[0,1]:+.2f}/{cf[1,1]:+.2f}  G_uu {cf[0,2]:+.2f}/{cf[1,2]:+.2f}  G_ww {cf[0,3]:+.2f}/{cf[1,3]:+.2f}')
            J[f'p6b_{n}_{beta}_{key}'] = cf.tolist()
    # fine-structure reproducibility and slope of v on T after the alternative corrections (n=8 only)
    if n == 8:
        for key in ('Tr', 'Ta', 'Tp', 'Tc_fine'):
            res = [wfit(np.log(out6[n][r][key]), out6[n][r]['Tre'] / out6[n][r][key], n, 2)[1].ravel() for r in RUNS]
            vres = {v: [wfit(np.log(MAPS[r][n][v]), MAPS[r][n][v + '_err'] / MAPS[r][n][v], n, 2)[1].ravel() for r in RUNS] for v in ('vx', 'vy')}
            P(f'n=8 {key}: fine T signal rms {np.sqrt(max(cov(res[0], res[1]), 0)) * 100:.2f} %, r {pear(res[0], res[1]):+.2f};  fine d ln v / d ln T: ' +
              ', '.join(f'{v} {boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (vres[v][0], vres[v][1], res[0], res[1]), nb=1500)[0]:+.2f}' for v in ('vx', 'vy')))

P('\n##### (6c) ladder velocity v against ln q')
for n in (4, 6, 8):
    maps = out6[n]
    for v in ('vx', 'vy'):
        t = [np.log(MAPS[r][n][v]).ravel() for r in RUNS]; q = [maps[r]['lq'].ravel() for r in RUNS]
        s = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (t[0], t[1], q[0], q[1]), nb=2000)
        tf = [wfit(np.log(MAPS[r][n][v]), MAPS[r][n][v + '_err'] / MAPS[r][n][v], n, 2)[1].ravel() for r in RUNS]
        qf = [wfit(maps[r]['lq'], maps[r]['lqe'], n, 2)[1].ravel() for r in RUNS]
        sf = boot_cells(lambda a, b, c, d: iv_slope(a, b, c, d), (tf[0], tf[1], qf[0], qf[1]), nb=2000)
        P(f'n={n} d ln {v} / d ln q (cells, cross-run): full {s[0]:+.3f} ± {s[1]:.3f} [{s[2]:+.2f},{s[3]:+.2f}]; fine {sf[0]:+.3f} ± {sf[1]:.3f} [{sf[2]:+.2f},{sf[3]:+.2f}]')
        J[f'p6c_{n}_{v}'] = dict(full=s[0], full_e=s[1], fine=sf[0], fine_e=sf[1])
# event-level: ladder for low/high charge halves within each cell (n=4), per view
P('event-level: chamber-wide ladder velocity of the low-charge and high-charge halves of each 4x4 cell (view charge x_q_sum / y_q_sum)')
for run in RUNS:
    tab = data[run][0]
    ex, ey = np.linspace(*XE, 5), np.linspace(*YE, 5)
    it, jt = GM.cell_of(tab.xm.values, tab.ym.values, ex, ey)
    for p in 'xy':
        q = np.log(tab[f'{p}_q_sum'].values.clip(1e-3)); hi = np.zeros(len(tab), bool)
        for a in range(4):
            for b in range(4):
                s = (it == a) & (jt == b)
                hi[s] = q[s] > np.median(q[s])
        out = {}
        for lab, sel in (('low', ~hi), ('high', hi)):
            d = tab[sel]; tr = d[f'tan_ref_{p}'].values; w = d[f'{p}_w'].values * 1e3
            ok = np.abs(tr) <= 0.6
            _, v, _, ve, _ = CO.line(tr[ok], w[ok]); out[lab] = (v, ve, np.mean(q[sel]))
        dv = out['high'][0] / out['low'][0] - 1; dq = out['high'][2] - out['low'][2]
        ev = np.hypot(out['high'][1] / out['high'][0], out['low'][1] / out['low'][0])
        P(f'   {run} {p}: v low {out["low"][0]:.3f}±{out["low"][1]:.3f}, v high {out["high"][0]:.3f}±{out["high"][1]:.3f}; d ln v / d ln q = {np.log(out["high"][0] / out["low"][0]) / dq:+.4f} ± {ev / dq:.4f}')

# ============================================================ (7)
P('\n##### (7) time blocks within each run')
J7 = {}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ri, run in enumerate(RUNS):
    tab, m, info = data[run]
    tab = tab.sort_values(['acq_tag', 'timestamp']).reset_index(drop=True)
    m = m.sort_values(['acq_tag', 'timestamp']).reset_index(drop=True)
    nb = 10 if run == 'run2' else 8
    tab['blk'] = np.minimum((np.arange(len(tab)) * nb) // len(tab), nb - 1)
    # blocks of the matched table follow the same event order: assign by time rank in tab
    key = tab[['acq_tag', 'timestamp']].copy(); key['blk'] = tab['blk']
    m = m.merge(key.drop_duplicates(['acq_tag', 'timestamp']), on=['acq_tag', 'timestamp'], how='left')
    rows = []
    for b in range(nb):
        d = tab[tab.blk == b]; mm = m[m.blk == b]
        vx, vxe, *_ = CO.line(d.tan_ref_x.values[np.abs(d.tan_ref_x) <= .6], d.x_w.values[np.abs(d.tan_ref_x) <= .6] * 1e3)[1:4] if False else (None, None)
        sx = np.abs(d.tan_ref_x.values) <= 0.6; sy = np.abs(d.tan_ref_y.values) <= 0.6
        _, vx, _, vxe, _ = CO.line(d.tan_ref_x.values[sx], d.x_w.values[sx] * 1e3)
        _, vy, _, vye, _ = CO.line(d.tan_ref_y.values[sy], d.y_w.values[sy] * 1e3)
        T = GM.tmean(mm.T_amp.values); Te = GM.boot(GM.tmean, mm.T_amp.values, nb=100)
        rows.append((b, len(d), len(mm), vx, vxe, vy, vye, T, Te, float(np.mean(mm.lnq)), d.acq_tag.iloc[0], d.acq_tag.iloc[-1]))
        P(f'   {run} block {b}: n_tab {len(d)} n_T {len(mm)}  v_x {vx:.3f}±{vxe:.3f}  v_y {vy:.3f}±{vye:.3f}  T {T:.1f}±{Te:.1f}  <ln q> {rows[-1][9]:.3f}  acq {rows[-1][10]}..{rows[-1][11]}')
    R = np.array([r[:10] for r in rows], float)
    lnT, elT = np.log(R[:, 7]), R[:, 8] / R[:, 7]
    for vi, (vn, col) in enumerate((('v_x', 3), ('v_y', 5))):
        lnv, elv = np.log(R[:, col]), R[:, col + 1] / R[:, col]
        # weighted least squares with errors in both: orthogonal/Deming with known error ratio; also naive OLS and a bootstrap over events is expensive: use parametric MC
        w = 1 / (elv ** 2 + 0 * elT)
        A = np.vstack([np.ones(nb), lnT - lnT.mean()]).T
        c = np.linalg.lstsq(A * np.sqrt(w)[:, None], lnv * np.sqrt(w), rcond=None)[0]
        # Deming (delta = var_v / var_T of measurement errors, per block average)
        sxx, syy, sxy = np.var(lnT, ddof=1), np.var(lnv, ddof=1), np.cov(lnT, lnv)[0, 1]
        d_ = np.mean(elv ** 2) / np.mean(elT ** 2)
        dem = (syy - d_ * sxx + np.sqrt((syy - d_ * sxx) ** 2 + 4 * d_ * sxy ** 2)) / (2 * sxy)
        rng = np.random.default_rng(3); bs = []
        for _ in range(3000):
            i = rng.integers(0, nb, nb)
            if np.var(lnT[i]) == 0: continue
            sx_, sy_, sxy_ = np.var(lnT[i], ddof=1), np.var(lnv[i], ddof=1), np.cov(lnT[i], lnv[i])[0, 1]
            bs.append(np.polyfit(lnT[i], lnv[i], 1)[0])
        sig_T = np.std(lnT, ddof=1); mean_eT = np.sqrt(np.mean(elT ** 2))
        P(f'   {run} {vn}: scatter of ln T over blocks {sig_T*100:.2f} % (mean stat err {mean_eT*100:.2f} %), of ln v {np.std(lnv, ddof=1)*100:.2f} % (err {np.sqrt(np.mean(elv**2))*100:.2f} %); '
          f'OLS slope {np.polyfit(lnT, lnv, 1)[0]:+.2f} ± {np.std(bs):.2f} (bootstrap over blocks); Deming {dem:+.2f}; corr {np.corrcoef(lnT, lnv)[0,1]:+.2f}')
        J7[f'{run}_{vn}'] = dict(ols=float(np.polyfit(lnT, lnv, 1)[0]), boot_err=float(np.std(bs)), deming=float(dem), sigT=float(sig_T), eT=float(mean_eT))
        axs[vi].errorbar(100 * (lnT - lnT.mean()), 100 * (lnv - lnv.mean()), xerr=100 * elT, yerr=100 * elv, fmt='o', ms=3, label=run)
    # chamber-wide numbers of the full run for the run1 x run2 difference
J7['glob'] = {}
for run in RUNS:
    tab, m, info = data[run]
    T = GM.tmean(m.T_amp.values); Te = GM.boot(GM.tmean, m.T_amp.values, nb=100)
    gl = MAPS[run]['glob']
    J7['glob'][run] = dict(T=T, Te=Te, vx=gl['x_t06'][0], vxe=gl['x_t06'][1], vy=gl['y_t06'][0], vye=gl['y_t06'][1], Traw=GM.tmean(m['T'].values), lnq=float(m.lnq.mean()))
g1, g2 = J7['glob']['run1'], J7['glob']['run2']
dlT = np.log(g2['T'] / g1['T']); eT = np.hypot(g1['Te'] / g1['T'], g2['Te'] / g2['T'])
P(f'run2 - run1 chamber-wide: T {g1["T"]:.1f} -> {g2["T"]:.1f} ns, d ln T = {dlT*100:+.2f} ± {eT*100:.2f} %  (raw T {g1["Traw"]:.1f} -> {g2["Traw"]:.1f}); <ln q> {g1["lnq"]:.3f} -> {g2["lnq"]:.3f}')
for v in ('vx', 'vy'):
    dlv = np.log(g2[v] / g1[v]); ev = np.hypot(g1[v + 'e'] / g1[v], g2[v + 'e'] / g2[v])
    P(f'   d ln {v} = {dlv*100:+.2f} ± {ev*100:.2f} %  -> ratio d ln v / d ln T = {dlv/dlT:+.2f} ± {abs(dlv/dlT)*np.hypot(ev/abs(dlv), eT/abs(dlT)):.2f}')
    J7[f'run_diff_{v}'] = dict(dlnv=dlv, e=ev, dlnT=dlT, eT=eT)
for a in axs:
    a.set_xlabel('ln T block - mean (%)'); a.legend(fontsize=7)
axs[0].set_ylabel('ln v_x block - mean (%)'); axs[1].set_ylabel('ln v_y block - mean (%)')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_timeblocks.png', dpi=130); plt.close()
J['p7'] = J7
json.dump(J, open(f'{OUT}/addenda.json', 'w'), indent=1, default=float)
open(f'{OUT}/addenda.log', 'w').write('\n'.join(LOG))
