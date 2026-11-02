#!/usr/bin/env python3
"""position_ms_fig.py -- Paper I §8 figure `bench_position_res`: the bench intrinsic position resolution, with the
extrapolation and the subtraction shown (Dylan 2026-11-02: the headline position result).

    position_ms_fig.py bins   -> results/position_ms_bins.json   (needs the floor frames; FLOOR_FRAMES=<dir>)
    position_ms_fig.py fig    -> figures/bench_position_res.{pdf,png,csv}

The numbers are position_ms.py's (results/position_ms.json): per headline row and view, the mesh residual core sigma
of near-normal tracks (|theta_M3| < 5 deg) in quintiles of the M3 chi2 of the matching coordinate, sigma^2 linear in
chi2 -> the chi2 -> 0 intercept sigma_0 (bootstrap), then chamber = sqrt(sigma_0^2 - P^2) with P the M3 pointing.
position_ms.json stores the quintile bins only for all angles, so `bins` re-runs position_ms's own selection and fit
for the near-normal subset (same functions, same seed) and checks that the intercept reproduces the stored value.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
BINS = os.path.join(HERE, 'results', 'position_ms_bins.json')
MS = os.path.join(HERE, 'results', 'position_ms.json')
SPS_R = os.path.join(REPO, 'rerun', 'sps_region', 'results', 'region.json')


def bins():
    import floorlib as C
    from xasym_sys import core_sigma, realign
    from position_ms import intercept
    MSJ = json.load(open(MS))
    out = {}
    rng = np.random.default_rng(11)
    for det, key in C.HEADLINE.items():
        F = C.load(key)
        for p in ('x', 'y'):
            st = MSJ[f'{det}|headline'][p]
            g = F[(F.view == p) & F.ig & np.isfinite(F.p0) & np.isfinite(F.r_mesh_x) & np.isfinite(F.r_mesh_y)].copy()
            mx, my = g.r_mesh_x.to_numpy(), g.r_mesh_y.to_numpy()
            res = realign(g.p0.to_numpy() - (mx if p == 'x' else my), mx, my)
            th = np.abs(np.degrees(np.arctan(g.tan_ref.to_numpy())))
            c = g[st['matched']].to_numpy()
            nn = th < 5
            r, cc = res[nn], c[nn]
            ed = np.nanpercentile(cc, [0, 20, 40, 60, 80, 100]); ed[-1] += 1e-9
            B = []
            for lo, hi in zip(ed[:-1], ed[1:]):
                m = (cc >= lo) & (cc < hi) & (np.abs(r) < 15)
                s, _ = core_sigma(r[m])
                bs = [core_sigma(r[m][rng.integers(0, m.sum(), m.sum())])[0] for _ in range(60)]
                B.append(dict(chi2=float(np.mean(cc[m])), sigma_um=1e3 * s, err_um=1e3 * float(np.nanstd(bs)), n=int(m.sum())))
            sl, ic = np.polyfit([b['chi2'] for b in B], [(b['sigma_um'] / 1e3) ** 2 for b in B], 1)
            i5, e5 = intercept(r, cc)
            ok = abs(1e3 * i5 - st['chi2_0_near_normal_um']) < 0.5
            out[f'{det}|{p}'] = dict(key=key, matched=st['matched'], n=int(nn.sum()), bins=B, slope_mm2=float(sl),
                                     ic_mm2=float(ic), sigma0_um=1e3 * i5, sigma0_err_um=1e3 * e5,
                                     reproduces_position_ms=bool(ok), P_um=st['P_um'],
                                     chamber_um=st['chamber_near_normal_um'], chamber_err_um=st['chamber_near_normal_err_um'])
            print(f'{det} {p}: sigma0 {1e3 * i5:.1f} +- {1e3 * e5:.1f} (stored {st["chi2_0_near_normal_um"]:.1f}) '
                  f'P {st["P_um"]:.0f} chamber {st["chamber_near_normal_um"]:.0f} +- {st["chamber_near_normal_err_um"]:.0f}'
                  f'  {"OK" if ok else "MISMATCH"}')
    json.dump(out, open(BINS, 'w'), indent=1)
    print('wrote', BINS)


def fig():
    sys.path.insert(0, os.path.join(REPO, 'papers'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import paper_style as ps
    ps.use()
    D = json.load(open(BINS))
    S = json.load(open(SPS_R))['resolution']['run63']
    sps = dict(x=S['R|p_mid']['X'], y=S['R|p0']['Y'])
    fig, (a1, a2) = ps.fig_for('full', height=2.6, nrows=1, ncols=2)
    rows = []
    c3 = ps.color('det3')

    # (a) det3: residual core vs M3 chi2 quintile, extrapolated to chi2 -> 0; the pointing P it is compared with
    for p in ('x', 'y'):
        d = D[f'det3|{p}']
        kw = ps.det('det3', view=p, line=False)
        x = [b['chi2'] for b in d['bins']]; y = [b['sigma_um'] for b in d['bins']]; e = [b['err_um'] for b in d['bins']]
        a1.errorbar(x, y, yerr=e, fmt=kw['marker'], ms=4, color=c3, mfc=kw.get('mfc', c3), mec=c3, zorder=5,
                    label=f'det3 {p.upper()}')
        cx = np.linspace(0, max(x) * 1.05, 100)
        a1.plot(cx, 1e3 * np.sqrt(np.maximum(d['ic_mm2'] + d['slope_mm2'] * cx, 0)), ls=ps.VIEW_LS[p], color=c3, lw=1.0,
                zorder=4)
        a1.errorbar([0], [d['sigma0_um']], yerr=[d['sigma0_err_um']], fmt='*', ms=8, color=c3,
                    mfc=c3 if p == 'x' else 'white', mec=c3, zorder=6, clip_on=False)
        a1.axhline(d['P_um'], color=ps.MUTED, lw=0.9, ls=ps.VIEW_LS[p], zorder=2)
        a1.text(0.74, d['P_um'] + (6 if p == 'x' else -6), f'M3 pointing $P_{p.upper()}$',
                fontsize=ps.BASE_PT * 0.8, color=ps.MUTED, va='bottom' if p == 'x' else 'top', ha='right')
        for b in d['bins']:
            rows.append(dict(panel='a', det='det3', view=p, chi2_m3=b['chi2'], sigma_core_um=b['sigma_um'],
                             err_um=b['err_um'], n=b['n']))
        rows.append(dict(panel='a', det='det3', view=p, chi2_m3=0.0, sigma_core_um=d['sigma0_um'],
                         err_um=d['sigma0_err_um'], n=d['n']))
    a1.set_xlim(-0.015, 0.75); a1.set_ylim(0, 900)
    a1.set_xlabel(r'M3 $\chi^2$ of the matching coordinate')
    a1.set_ylabel(r'residual core $\sigma$ [µm]')

    # (b) the subtraction: chamber = sqrt(sigma_0^2 - (s P)^2) against the scale s of the pointing; +-10 % shaded
    s = np.linspace(0.8, 1.2, 161)
    a2.axvspan(0.9, 1.1, color=ps.GRID, lw=0, zorder=0)
    for det in ('det2', 'det3'):
        for p in ('x', 'y'):
            d = D[f'{det}|{p}']
            s0, e0, P = d['sigma0_um'], d['sigma0_err_um'], d['P_um']
            ch = np.sqrt(np.maximum(s0 ** 2 - (s * P) ** 2, 0))
            kw = ps.det(det, view=p)
            if det == 'det3':
                lo = np.sqrt(np.maximum((s0 - e0) ** 2 - (s * P) ** 2, 0))
                hi = np.sqrt(np.maximum((s0 + e0) ** 2 - (s * P) ** 2, 0))
                a2.fill_between(s, lo, hi, color=c3, alpha=0.10 if p == 'x' else 0.07, lw=0, zorder=1)
            a2.plot(s, ch, ls=kw['ls'], color=kw['color'], lw=kw['lw'], zorder=kw['zorder'],
                    label=f"{det} {p.upper()}: {d['chamber_um']:.0f} ± {d['chamber_err_um']:.0f}")
            a2.plot([1.0], [d['chamber_um']], marker=kw['marker'], ms=4.5, color=kw['color'],
                    mfc=kw.get('mfc', kw['color']), mec=kw['color'], ls='', zorder=kw['zorder'] + 1)
            for si, ci in zip(s[::20], ch[::20]):
                rows.append(dict(panel='b', det=det, view=p, P_scale=si, chamber_um=ci, sigma0_um=s0, P_um=P))
            if det == 'det3':
                f = lambda k: float(np.sqrt(max(s0 ** 2 - (k * P) ** 2, 0)))
                d['p10'] = [f(1.1), f(0.9)]
    sx, sy = sps['x'], sps['y']
    a2.axhspan(min(sy['sigma_det4'], sx['sigma_det4']) * 1e3, max(sy['sigma_det4'], sx['sigma_det4']) * 1e3,
               color=ps.color('det4'), alpha=0.25, lw=0, zorder=0)
    a2.text(0.81, sy['sigma_det4'] * 1e3 - 3, 'det4 at SPS', fontsize=ps.BASE_PT * 0.8, color=ps.MUTED, va='top')
    a2.set_xlim(0.8, 1.2); a2.set_ylim(0, 330)
    a2.set_xlabel(r'assumed M3 pointing $P\,/\,P_{\rm nominal}$')
    a2.set_ylabel(r'chamber $\sqrt{\sigma_0^2 - P^2}$ [µm]')
    a2.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ps.panel_labels([a1, a2])
    from matplotlib.lines import Line2D
    hl = ps.det_handles(('det3', 'det2')) + ps.view_handles() + [
        Line2D([], [], ls='none', marker='*', ms=8, color=ps.INK, label=r'$\sigma_0$ ($\chi^2 \to 0$)')]
    ps.legend_above(fig, hl, ncol=5)
    ps.save(fig, os.path.join(HERE, 'figures', 'bench_position_res'), data=pd.DataFrame(rows))
    print('det3 +-10 % P:', {k: [round(v, 0) for v in D[k].get('p10', [])] for k in ('det3|x', 'det3|y')})
    print('wrote', os.path.join(HERE, 'figures', 'bench_position_res.pdf'))


if __name__ == '__main__':
    bins() if len(sys.argv) > 1 and sys.argv[1] == 'bins' else fig()
