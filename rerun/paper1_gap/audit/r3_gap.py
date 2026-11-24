"""r3_gap.py -- gap maps (delta ln q_u90 -> delta g for beta 0.68), saturation checks, row consistency, alignment.
-> $OUT/gapstats.json, $OUT/figs/gap_<chamber>.png"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from r3_lib import *

os.makedirs(f'{OUT}/figs', exist_ok=True)
UG, WG = np.linspace(-1.7, 1.7, 69), np.linspace(-1.51, 1.51, 61)
GU, GW = np.meshgrid(UG, WG, indexing='ij')
FAC = G0 / (1 + BETA0)


def coverage(D):
    H, _, _ = np.histogram2d(D.U, D.W, bins=[np.linspace(-1.75, 1.75, 36), np.linspace(-1.55, 1.55, 32)])
    ui = np.clip(((GU + 1.75) / 0.1).astype(int), 0, 34); wi = np.clip(((GW + 1.55) / 0.1).astype(int), 0, 30)
    return H[ui, wi] >= 4


def evalgrid(spl, c):
    return (spl.basis(GU.ravel(), GW.ravel()) @ c).reshape(GU.shape) * FAC


def quad(Z, cov):
    X = np.vstack([np.ones(cov.sum()), GU[cov], GW[cov], GU[cov] ** 2, GW[cov] ** 2, GU[cov] * GW[cov]]).T
    c = np.linalg.lstsq(X, Z[cov], rcond=None)[0]
    Q = (c[0] + c[1] * GU + c[2] * GW + c[3] * GU ** 2 + c[4] * GW ** 2 + c[5] * GU * GW)
    return Q, dict(gu=float(c[1]), gw=float(c[2]), guu=float(2 * c[3]), gww=float(2 * c[4]), guw=float(c[5]))


def corr(A, B, cov):
    a, b = A[cov] - A[cov].mean(), B[cov] - B[cov].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b))), float((a @ b) / (a @ a))


def describe(q, fine_rms, p2p):
    t = []
    def curv(g, ax):
        if abs(g) < 0.3: return f'flat along {ax}'
        return ('bowl' if g < 0 else 'dome') + f' along {ax} ({g:+.1f})'
    t.append(curv(q['guu'], 'x')); t.append(curv(q['gww'], 'y'))
    if abs(q['guw']) >= 0.3: t.append(f'twist ({q["guw"]:+.1f})')
    tilt = []
    if abs(q['gu']) >= 0.2: tilt.append(f'x {q["gu"]:+.2f}')
    if abs(q['gw']) >= 0.2: tilt.append(f'y {q["gw"]:+.2f}')
    if tilt: t.append('tilt ' + ', '.join(tilt) + ' mm/100 mm')
    t.append(f'folds rms {fine_rms:.2f} mm')
    return '; '.join(t) + f'; p2p {p2p:.1f} mm'


S = {}
FIG = {}
for name in MOUNT:
    D, info, bundle = load_mounting(name)
    spl, c, ok = gap_field(D)
    cov = coverage(D)
    Z = evalgrid(spl, c)
    Q, q = quad(Z, cov)
    F = Z - Q
    r = dict(N=D.n, K=spl.K, rows=info, quad=q)
    p1, p99 = np.percentile(Z[cov], [1, 99]); r['p2p_full'] = float(p99 - p1)
    r['p2p_fine'] = float(np.diff(np.percentile(F[cov], [1, 99]))[0]); r['rms_full'] = float(Z[cov].std()); r['rms_fine'] = float(F[cov].std())
    r['text'] = describe(q, r['rms_fine'], r['p2p_full'])
    # row consistency
    if len(info) > 1:
        fr = {}
        for i in range(len(info)):
            m = D.row == i
            if m.sum() < 500:
                continue
            Di = D.sub(m); s_i, c_i, _ = gap_field(Di, K=8)
            Zi = evalgrid(s_i, c_i); Qi, _ = quad(Zi, cov); fr[i] = (Zi, Zi - Qi)
        pairs = {}
        for a in fr:
            for b in fr:
                if a < b:
                    pairs[f'{a}-{b}'] = dict(full=corr(fr[a][0], fr[b][0], cov), fine=corr(fr[a][1], fr[b][1], cov))
        r['row_pairs'] = pairs
    # saturation: dependence of ln q on saturated strips
    sat = (D.satx > 0) | (D.saty > 0)
    resid = D.lnq - (spl.basis(D.U, D.W) @ c)
    ok = np.isfinite(resid)
    r['sat_frac_rows'] = [float(sat[D.row == i].mean()) for i in range(len(info))]
    sh = []
    for i in list(range(len(info))) + [None]:
        m = ok if i is None else ok & (D.row == i)
        a, b = resid[m & sat], resid[m & ~sat]
        if len(a) >= 100 and len(b) >= 100:
            sh.append(dict(row=i, n_sat=int(len(a)), n_unsat=int(len(b)), shift=float(a.mean() - b.mean()),
                           err=float(np.sqrt(a.var() / len(a) + b.var() / len(b))), shift_mm=float(FAC * (a.mean() - b.mean()))))
    r['sat_shift'] = sh
    # map of the saturated fraction against the q map, and the field from unsaturated events only
    if sat.sum() >= 200:
        s8 = R1.Spl(8)
        cs, _ = s8.fit(D.U, D.W, sat.astype(float) - sat.mean(), 0.1)
        Sm = (s8.basis(GU.ravel(), GW.ravel()) @ cs).reshape(GU.shape)
        r['sat_map_corr_with_q'] = corr(Sm, Z, cov)[0]
        r['sat_map_range'] = float(np.diff(np.percentile(Sm[cov], [1, 99]))[0])
    if (~sat).sum() >= 800 and sat.sum() >= 200:
        s2, c2, _ = gap_field(D, only=~sat)
        Z2 = evalgrid(s2, c2); Q2, _ = quad(Z2, cov)
        r['nosat_field'] = dict(N=int((~sat & ok).sum()), full=corr(Z, Z2, cov), fine=corr(Z - Q, Z2 - Q2, cov), p2p=float(np.diff(np.percentile(Z2[cov], [1, 99]))[0]))
    S[name] = r
    FIG[name] = (Z, F, cov, r['text'])
    print(name, r['text'], flush=True)
json.dump(S, open(f'{OUT}/gapstats.json', 'w'), indent=1, default=float)

CH = {'A': ['A625', 'A628'], 'B': ['B'], 'C': ['C'], 'D': ['D', 'Dx200'], 'E': ['E623', 'E624']}
for ch, names in CH.items():
    fig, axs = plt.subplots(len(names), 2, figsize=(7.2, 3.1 * len(names)), squeeze=False)
    for i, nm in enumerate(names):
        Z, F, cov, txt = FIG[nm]
        lim = max(0.3, np.percentile(np.abs(Z[cov]), 99))
        for j, (A, ttl) in enumerate(((Z, 'full map'), (F, 'quadratic removed'))):
            ax = axs[i, j]
            Am = np.where(cov, A, np.nan)
            im = ax.imshow(Am.T, origin='lower', extent=[-1.7 * 100 + 200, 1.7 * 100 + 200, -1.51 * 100 + 200, 1.51 * 100 + 200],
                           cmap='RdBu_r', vmin=-lim if j == 0 else -lim / 2, vmax=lim if j == 0 else lim / 2, aspect='equal')
            ax.set_title(f'{nm}: {ttl}', fontsize=10); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
            cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label(r'$\delta g$ (mm), $\beta$=0.68')
    fig.tight_layout(); fig.savefig(f'{OUT}/figs/gap_{ch}.png', dpi=130); plt.close(fig)
print('done')
