#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m3_cut_scan.py -- find the best (stricter) M3 reference-track chi2 selection.

The v2 recipe (chi2<5, NClus>=3) leaves the measured det3 position residual
LIMITED BY REFERENCE-TRACKER ERROR: tightening the M3 chi2 keeps improving the
detector's apparent core sigma with no plateau inside chi2<5. This scans the cut
(applied as max(Chi2X,Chi2Y) <= T, both planes) using the already-extracted v2
sample (which is chi2<5, so we can only scan STRICTER), and reports the
resolution / miss-rate / statistics trade-off + a detector-intrinsic estimate.

Run:  ../../.venv/bin/python m3_cut_scan.py            (uses both det3 runs)
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def rstd(v, ns=3, it=5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v); k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10: break
        v = v[k]
    return float(np.std(v))


CUTS = [5, 3, 2, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
out = {}
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
colors = {'g_det3_wknd': '#1f4e8c', 'sat_det3': '#c0392b'}

for KEY in ['g_det3_wknd', 'sat_det3']:
    d = np.load(os.path.join(HERE, f'edge_chi2_data_{KEY}.npz'), allow_pickle=True)
    r = d['r']; c = np.maximum(d['m3_chi2x'], d['m3_chi2y'])
    ncl = np.minimum(d['m3_nclusx'], d['m3_nclusy'])

    # detector-intrinsic estimate: differential sigma vs mean-chi2 in thin bins,
    # fit sigma^2 = sig_det^2 + b*<chi2>, extrapolate chi2->0
    e = np.linspace(0, 5, 21); mc, sg = [], []
    for i in range(len(e) - 1):
        m = (c >= e[i]) & (c < e[i + 1]); rr = r[m]
        if m.sum() > 300:
            mc.append(float(np.mean(c[m]))); sg.append(rstd(rr[rr < 15]))
    mc, sg = np.array(mc), np.array(sg)
    A = np.vstack([np.ones_like(mc), mc]).T
    coef, *_ = np.linalg.lstsq(A, sg ** 2, rcond=None)
    sig_det = float(np.sqrt(max(coef[0], 1e-6)))
    sig_lowbin = float(sg[0])  # cleanest measured thin bin

    rows = []
    for T in CUTS:
        m = c <= T; rr = r[m]
        rows.append(dict(cut=T, nfrac=100 * float(m.mean()), sigma=rstd(rr[rr < 15]),
                         median=float(np.median(rr)), w2=100 * float((rr <= 2).mean()),
                         w5=100 * float((rr <= 5).mean()), gt5=100 * float((rr > 5).mean())))
    # + NClus==4 variants
    ncl4 = {}
    for T in [5.0, 1.0, 0.5]:
        m = (c <= T) & (ncl >= 4); rr = r[m]
        ncl4[f'chi2<{T:g}'] = dict(nfrac=100 * float(m.mean()), sigma=rstd(rr[rr < 15]),
                                   median=float(np.median(rr)),
                                   w2=100 * float((rr <= 2).mean()), w5=100 * float((rr <= 5).mean()),
                                   gt5=100 * float((rr > 5).mean()))
    out[KEY] = dict(sig_det_extrap=sig_det, sig_cleanest_bin=sig_lowbin,
                    slope=float(coef[1]), rows=rows, nclus4=ncl4)

    nf = [x['nfrac'] for x in rows]; sig = [x['sigma'] for x in rows]
    g5 = [x['gt5'] for x in rows]; cc = [x['cut'] for x in rows]
    ax[0].plot(cc, sig, 'o-', color=colors[KEY], label=KEY)
    ax[1].plot(nf, sig, 'o-', color=colors[KEY], label=KEY)
    ax[2].plot(cc, g5, 'o-', color=colors[KEY], label=KEY)
    if KEY == 'g_det3_wknd':
        ax[0].axhline(sig_det, color='grey', ls='--', lw=1,
                      label='extrap floor %.2f mm (model-dep.)' % sig_det)
        ax[0].axhline(sig_lowbin, color='green', ls=':', lw=1,
                      label='cleanest bin (%.2f mm)' % sig_lowbin)

ax[0].set_xlabel('M3 max(Chi2X,Chi2Y) cut'); ax[0].set_ylabel('detector core sigma(|r|<15) [mm]')
ax[0].axvline(1.0, color='k', ls=':', lw=0.8); ax[0].legend(fontsize=8)
ax[0].set_title('(a) resolution keeps improving with a stricter\nreference cut -- NO plateau inside chi2<5')
ax[1].set_xlabel('statistics retained [%]'); ax[1].set_ylabel('detector core sigma [mm]')
ax[1].invert_xaxis(); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].set_title('(b) resolution vs statistics trade-off\n(right = stricter cut)')
ax[2].set_xlabel('M3 max(Chi2X,Chi2Y) cut'); ax[2].set_ylabel('reco_far (|r|>5mm) rate [%]')
ax[2].axvline(1.0, color='k', ls=':', lw=0.8); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
ax[2].set_title('(c) reco_far tail shrinks with a stricter cut\n6.8% (chi2<5) -> 4.2% (chi2<1)')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig4_m3_cut_scan.png'), dpi=140, bbox_inches='tight')
print('wrote fig4_m3_cut_scan.png')
json.dump(out, open(os.path.join(HERE, 'm3_cut_scan.json'), 'w'), indent=2)
print(json.dumps({k: {'sig_det_extrap': round(v['sig_det_extrap'], 3),
                      'sig_cleanest_bin': round(v['sig_cleanest_bin'], 3),
                      'chi2<1': {kk: round(vv, 2) for kk, vv in v['rows'][5].items()},
                      'chi2<0.5': {kk: round(vv, 2) for kk, vv in v['rows'][8].items()},
                      'chi2<1+NClus4': {kk: round(vv, 2) for kk, vv in v['nclus4']['chi2<1'].items()},
                      'chi2<0.5+NClus4': {kk: round(vv, 2) for kk, vv in v['nclus4']['chi2<0.5'].items()}}
                  for k, v in out.items()}, indent=2))
