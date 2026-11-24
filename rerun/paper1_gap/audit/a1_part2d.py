"""a1_part2d.py -- Phi map (n = 8, 12), 50 mm periodicity scan, and the figures of Parts 1-2.  -> phi.json, per.json, png"""
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a1_lib import *  # noqa
import a1_part2 as P2

plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
D = pickle.load(open(f'{OUT}/maps.pkl', 'rb'))
maps = D['maps']
RUNS = ('run1', 'run2')

# ------------------------------------------------------------------ Phi
phi_out = {}
PH = {}
for n in (8, 12):
    mp = maps[n]
    sols = {}
    for tag, runs in (('both', RUNS), ('run1', ('run1',)), ('run2', ('run2',))):
        A, y, ids = helmholtz_system(n, [P2.resid_maps(mp[r], 'none') for r in runs])
        phi, res, H, rank, cov = helmholtz_solve(A, y)
        used = np.abs(A).sum(0) > 0
        sols[tag] = (phi, cov, used)
    N1 = n + 1
    eu, ew = np.linspace(*UL, N1) * 100, np.linspace(*WL, N1) * 100      # mm from (200, 200)
    UU, WW = np.meshgrid(eu, ew, indexing='ij')
    U, W = UU.ravel(), WW.ravel()

    def nonquad(phi, cov, used):
        X = np.vstack([np.ones(used.sum()), U[used], W[used], U[used] ** 2, U[used] * W[used], W[used] ** 2]).T
        Pm = np.eye(used.sum()) - X @ np.linalg.pinv(X)
        p = Pm @ phi[used]
        c = Pm @ cov[np.ix_(used, used)] @ Pm.T
        q = np.full(len(phi), np.nan); q[used] = p
        e = np.full(len(phi), np.nan); e[used] = np.sqrt(np.clip(np.diag(c), 0, None))
        return q, e

    q, e = nonquad(*sols['both'])
    q1, e1 = nonquad(*sols['run1']); q2, e2 = nonquad(*sols['run2'])
    ok = np.isfinite(q1) & np.isfinite(q2)
    cc = float(np.mean((q1[ok] - q1[ok].mean()) * (q2[ok] - q2[ok].mean())))
    # also a linear-and-quadratic-free, cubic-free? report the quadratic surface itself
    X = np.vstack([np.ones(len(U)), U, W, U ** 2, U * W, W ** 2]).T
    used = sols['both'][2]
    cq = np.linalg.lstsq(X[used], sols['both'][0][used], rcond=None)[0]
    phi_out[n] = dict(rms=float(np.nanstd(q) ), ptp=float(np.nanmax(q) - np.nanmin(q)),
                      node_err_mean=float(np.nanmean(e)), node_err_max=float(np.nanmax(e)),
                      nnode=int(np.isfinite(q).sum()), cross_run_signal_rms=float(np.sqrt(max(cc, 0))), cross_cov=cc,
                      rms_run1=float(np.nanstd(q1)), rms_run2=float(np.nanstd(q2)),
                      quad_coef_mm=dict(zip(['c', 'U', 'W', 'UU', 'UW', 'WW'], cq.tolist())),
                      r_run1_run2=float(np.corrcoef(q1[ok], q2[ok])[0, 1]))
    PH[n] = (eu, ew, q.reshape(N1, N1), e.reshape(N1, N1), q1.reshape(N1, N1), q2.reshape(N1, N1))
    print(n, phi_out[n], flush=True)
json.dump(phi_out, open(f'{OUT}/phi.json', 'w'), indent=1)
pickle.dump(PH, open(f'{OUT}/phi.pkl', 'wb'))

fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
for r, n in enumerate((8, 12)):
    eu, ew, q, e, q1, q2 = PH[n]
    for c, (z, tt) in enumerate(((q1, 'run 1'), (q2, 'run 2'), (q, 'runs combined'))):
        im = ax[r, c].pcolormesh(eu, ew, 1e3 * z.T, cmap='RdBu_r', vmin=-1000, vmax=1000, shading='nearest')
        ax[r, c].set_title(f'n = {n} ({LU / n:.0f} mm cells), {tt}')
        ax[r, c].set_xlabel('x - 200 (mm)'); ax[r, c].set_ylabel('y - 200 (mm)')
        ax[r, c].set_aspect('equal')
    fig.colorbar(im, ax=ax[r, :], label='non-quadratic part of Phi (um)', shrink=0.8)
fig.savefig(f'{OUT}/fig_phi.png'); plt.close(fig)

# ------------------------------------------------------------------ figure: residual tau maps n = 8
fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
for r_, p in enumerate('xy'):
    for c, run in enumerate(RUNS):
        m = maps[8][run][p]
        R = residual(m, 'tau', 'etau', 'plane')[0]
        eu, ew = np.linspace(*UL, 9) * 100, np.linspace(*WL, 9) * 100
        im = ax[r_, c].pcolormesh(eu, ew, 1e3 * R.T, cmap='RdBu_r', vmin=-15, vmax=15)
        ax[r_, c].set_title(f'tau_{p} minus plane, {run}, 8x8'); ax[r_, c].set_aspect('equal')
        ax[r_, c].set_xlabel('x - 200 (mm)'); ax[r_, c].set_ylabel('y - 200 (mm)')
    fig.colorbar(im, ax=ax[r_, :], label='residual tilt (1e-3 tan)', shrink=0.8)
fig.savefig(f'{OUT}/fig_tau_resid_n8.png'); plt.close(fig)

# ------------------------------------------------------------------ figure: Cx against Cy
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
for a, n in zip(ax, (8, 10, 12)):
    Rx = {r: residual(maps[n][r]['x'], 'tau', 'etau', 'plane')[0] for r in RUNS}
    Ry = {r: residual(maps[n][r]['y'], 'tau', 'etau', 'plane')[0] for r in RUNS}
    ex = {r: np.where(np.isfinite(Rx[r]), maps[n][r]['x']['etau'], np.nan) for r in RUNS}
    ey = {r: np.where(np.isfinite(Ry[r]), maps[n][r]['y']['etau'], np.nan) for r in RUNS}
    C = {r: (fd_corners(Rx[r], ex[r], n)[0], fd_corners(Ry[r], ey[r], n)[1], fd_corners(Rx[r], ex[r], n)[2],
             fd_corners(Ry[r], ey[r], n)[3]) for r in RUNS}
    ok = np.ones(C['run1'][0].shape, bool)
    for r in RUNS:
        ok &= np.isfinite(C[r][0]) & np.isfinite(C[r][1])
    for (ra, rb), col in ((('run2', 'run1'), 'C0'), (('run1', 'run2'), 'C1')):
        a.errorbar(C[ra][0][ok], C[rb][1][ok], xerr=C[ra][2][ok], yerr=C[rb][3][ok], fmt='o', ms=3, color=col, alpha=.7,
                   lw=.6, label=f'Cx {ra} vs Cy {rb}')
    lim = np.nanmax(np.abs([C[r][k][ok] for r in RUNS for k in (0, 1)])) * 1.1
    a.plot([-lim, lim], [-lim, lim], 'k--', lw=.8)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
    a.set_xlabel('Cx = d tau_x / dW (tan per 100 mm)'); a.set_ylabel('Cy = d tau_y / dU (tan per 100 mm)')
    a.set_title(f'n = {n}: r(Cx, Cy other run) = {pear(C["run2"][0][ok], C["run1"][1][ok]):.2f}')
    if n == 8:
        a.legend(fontsize=7, loc='upper left')
fig.savefig(f'{OUT}/fig_curl_scatter.png'); plt.close(fig)

# ------------------------------------------------------------------ 1-D periodicity scan
def scan(run, p, coord, periods, deg=4):
    U, W, t, w = view_arrays(run, p)
    d = U if coord == 'U' else W
    o = W if coord == 'U' else U
    keep = np.ones(len(t), bool)
    v0 = rline(t, w, nmin=40)[1]
    # base model: w = a + v t + polynomial(d, deg) + plane(o)   (the smooth part, removed before the sinusoid)
    cols = [np.ones_like(t), t] + [(d / 1.7) ** k for k in range(1, deg + 1)] + [o / 1.5, (o / 1.5) ** 2]
    X0 = np.vstack(cols).T
    for _ in range(3):
        c = np.linalg.lstsq(X0[keep], w[keep], rcond=None)[0]
        r = w - X0 @ c; rs = 1.4826 * np.median(np.abs(r[keep])); keep = np.abs(r) < 3.5 * rs
    X0k, wk, dk = X0[keep], w[keep], d[keep] * 100
    c0 = np.linalg.lstsq(X0k, wk, rcond=None)[0]
    rss0 = np.sum((wk - X0k @ c0) ** 2); s2 = rss0 / (len(wk) - X0k.shape[1])
    v = c0[1]
    amp, ph, dchi = [], [], []
    for P_ in periods:
        arg = 2 * np.pi * dk / P_
        X = np.hstack([X0k, np.cos(arg)[:, None], np.sin(arg)[:, None]])
        c = np.linalg.lstsq(X, wk, rcond=None)[0]
        rss = np.sum((wk - X @ c) ** 2)
        amp.append(np.hypot(c[-2], c[-1]) / v); ph.append(np.arctan2(c[-1], c[-2])); dchi.append((rss0 - rss) / s2)
    return np.array(amp), np.array(ph), np.array(dchi), float(np.sqrt(s2) / v / np.sqrt(len(wk)) * np.sqrt(2))


periods = np.linspace(15, 150, 271)
per = {}
fig, ax = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
for r_, p in enumerate('xy'):
    for c_, coord in enumerate(('U', 'W')):
        a = ax[r_, c_]
        S = {run: scan(run, p, coord, periods) for run in RUNS}
        for run in RUNS:
            a.plot(periods, 1e3 * S[run][0], label=f'{run}')
        # coherent cross-amplitude: Re(A1 conj A2): positive only if same amplitude and phase in both runs
        cross = S['run1'][0] * S['run2'][0] * np.cos(S['run1'][1] - S['run2'][1])
        a.plot(periods, 1e3 * np.sign(cross) * np.sqrt(np.abs(cross)), 'k', lw=1, label='coherent sqrt(A1 A2 cos dphi)')
        a.axvline(49.9, color='r', ls=':', lw=1)
        a.set_xlabel('period along ' + ('x' if coord == 'U' else 'y') + ' (mm)'); a.set_ylabel('amplitude of tau_' + p + ' (1e-3)')
        a.set_title(f'tau_{p} against {"x" if coord == "U" else "y"}' + (' (own strip axis)' if (p == 'x') == (coord == 'U') else ''))
        a.set_ylim(bottom=0)
        if r_ == 0 and c_ == 0:
            a.legend(fontsize=7)
        i50 = np.argmin(np.abs(periods - 49.9))
        win = (periods > 40) & (periods < 62)
        # the typical amplitude of the scan away from the structure: median over 15-150 mm
        per[f'{p}_vs_{coord}'] = dict(
            amp_run1_50=float(S['run1'][0][i50]), amp_run2_50=float(S['run2'][0][i50]),
            dphase50=float(np.angle(np.exp(1j * (S['run1'][1][i50] - S['run2'][1][i50])))),
            cross_50=float(cross[i50]), amp_cross_50=float(np.sign(cross[i50]) * np.sqrt(abs(cross[i50]))), noise_amp_run1=S['run1'][3], noise_amp_run2=S['run2'][3],
            dchi2_run1_50=float(S['run1'][2][i50]), dchi2_run2_50=float(S['run2'][2][i50]),
            median_amp_run1=float(np.median(S['run1'][0])), median_amp_run2=float(np.median(S['run2'][0])),
            pct50=float(np.mean(cross[(periods > 20) & (periods < 100)] < cross[i50])),
            max_cross_period=float(periods[np.argmax(cross)]), max_cross=float(np.max(cross)),
            max_cross_win_40_62=float(np.max(cross[win])), period_at_max_win=float(periods[win][np.argmax(cross[win])]),
            top_cross_periods=[float(x) for x in periods[np.argsort(cross)[::-1][:8]]])
fig.savefig(f'{OUT}/fig_period_scan.png'); plt.close(fig)
json.dump(per, open(f'{OUT}/per.json', 'w'), indent=1)
for k, v in per.items():
    print(k, {a: (float('%.3g' % b) if isinstance(b, float) else b) for a, b in v.items()})

# fold at the connector pitch: residual of tau_p about its smooth fit against (own coordinate mod 49.9 mm), 10 phase bins
fold = {}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
for a, p in zip(ax, 'xy'):
    for run in RUNS:
        U, W, t, w = view_arrays(run, p)
        d = (U if p == 'x' else W) * 100
        o = W if p == 'x' else U
        X0 = np.vstack([np.ones_like(t), t] + [(d / 170) ** k for k in range(1, 5)] + [o / 1.5, (o / 1.5) ** 2]).T
        keep = np.ones(len(t), bool)
        for _ in range(3):
            c = np.linalg.lstsq(X0[keep], w[keep], rcond=None)[0]
            r = w - X0 @ c; rs = 1.4826 * np.median(np.abs(r[keep])); keep = np.abs(r) < 3.5 * rs
        v = c[1]
        res = (w - X0 @ c + 0) / v          # residual in tan units, includes noise
        ph = ((d + 200) % 49.9) / 49.9      # phase of the strip coordinate
        b = np.minimum((ph * 10).astype(int), 9)
        m = np.array([res[keep & (b == k)].mean() for k in range(10)])
        e = np.array([res[keep & (b == k)].std() / np.sqrt((keep & (b == k)).sum()) for k in range(10)])
        a.errorbar((np.arange(10) + .5) * 4.99, 1e3 * m, 1e3 * e, fmt='o-', ms=3, label=run)
        fold[f'{p}_{run}'] = dict(mean=m.tolist(), err=e.tolist(), chi2_flat=float(np.sum(((m - np.average(m, weights=1/e**2)) / e) ** 2)))
    a.set_xlabel(f'{p} mod 49.9 mm'); a.set_ylabel('residual tilt (1e-3 tan)'); a.set_title(f'tau_{p}, folded at 49.9 mm'); a.legend(fontsize=8)
fig.savefig(f'{OUT}/fig_fold50.png'); plt.close(fig)
json.dump(fold, open(f'{OUT}/fold.json', 'w'), indent=1)
print({k: round(v['chi2_flat'], 1) for k, v in fold.items()})
