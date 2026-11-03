#!/usr/bin/env python3
"""attach_charge.py -- the Paper I 'attachment_charge' figure: the head-on strip-sum signal against time, chamber E at SPS
vs chamber A on the bench (Dylan 2026-11-03: "the key plot showing charge vs time at SPS vs the cosmic bench").

A head-on track deposits charge uniformly over the gap; electrons from depth z arrive at t = z / v, so the strip-sum
signal against time is the charge collected from each depth.  Without attachment it is a flat-top box (diffusion and the
shaper round its ends); electron loss exp(-t / tau) tilts the top down with drift time.  The stacks are the waveform-fit
era products (pass-1 DREAM waveforms, trigger-placed strip-offset stacks of rerun/sps_gasfit/gas_stacks.py; no hit times).

Inputs (no fit is redone except the two fits below, which gasfit.fit_stack does exactly as the gas fit does):
  * SPS stacks: EOS x17rerun/sps_gasfit/stacks/<key>.npz  (fetched to --stacks, default $WORK_DATA/claude-jobs/77eb854f/gas/stacks)
  * composition and shaper: rerun/sps_gasfit/results/verdict.json (water, air at the quoted gap 27.9 mm) and gasfit.json
    (shaper at 27.9 mm).  Model = predict.current_field on the Magboltz (water, air) grid x the fixed shaper, t0 and
    amplitude free per stack; 'no O2' = the same water, air = 0, t0 and amplitude refitted (what the fit would be if the
    gas had no electronegative impurity).
  * bench: rerun/paper1_gas/results/twov.json, chamber A 6-27 drift scan at 700 V drift (248 V/cm): data and the
    attachment-free box model (s4f_fit at its stored sigma_0, T, dt).
Time zero = the half-height of the model's leading edge, so the three curves are aligned on the mesh end of the column
(SPS stacks are placed by trigger sample, bench stacks by the leading edge).  Each curve is normalised to its mean over
250-450 ns after that point.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'papers'), os.path.join(REPO, 'rerun', 'sps_gasfit')]
GAP = 27.9
ROWS = [('cf4', 'run71_raw__raw700', 'cf4_run71_raw', 'E at SPS, Ar/CF$_4$ (run 71)'),
        ('co2', 'run55_flatdrift__d700', 'co2_run55', 'E at SPS, Ar/CO$_2$ (run 55)')]
PLAT = (250.0, 450.0)


def t_half(t, m):
    mx = np.nanmax(m); i = int(np.argmax(m >= 0.5 * mx))
    return float(np.interp(0.5 * mx, [m[i - 1], m[i]], [t[i - 1], t[i]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stacks', default=os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'claude-jobs', '77eb854f', 'gas', 'stacks'))
    a = ap.parse_args()
    import pandas as pd
    import matplotlib.pyplot as plt
    import paper_style as ps
    import gasfit as G
    ps.use()
    gf = json.load(open(os.path.join(REPO, 'rerun', 'sps_gasfit', 'results', 'gasfit.json')))['shaper'][str(GAP)]
    V = {r['group']: r for r in json.load(open(os.path.join(REPO, 'rerun', 'sps_gasfit', 'results', 'verdict.json')))['table']}
    ns = argparse.Namespace(stacks=a.stacks, gap=GAP, k=0.0, zs_scale=1.0, sat=False, hx=2, hy=8, raw_eta=False,
                            source='air_merged', zs_win_hi=None, bench_template=False)
    G.ARGS = ns; ztab = G.zs_table(a.stacks, ns)
    Mo = G.Model('beam', 'air_merged', GAP, 0.0)
    Mc = G.Model('co2', 'air_merged', GAP, 0.0)
    curves, rows = [], []
    B = json.load(open(os.path.join(REPO, 'rerun', 'paper1_gas', 'results', 'twov.json')))
    S = B['stacks']['700']; tb = np.array(B['grid_ns'])
    yb, eb, mb = (np.array(S[k], float) for k in ('data', 'err', 'model'))
    curves.append(dict(name='A', label='A, bench, 248 V/cm', color=ps.color('det3'), t=tb, y=yb, e=eb, m=mb, m0=None,
                       E=B['fields']['700']['E']))
    for gas, key, grp, lab in ROWS:
        Mod = Mo if gas == 'cf4' else Mc
        d = [x for x in G.group_data(a.stacks, [key], ns, ztab) if x['view'] == 'x'][0]
        w, ai = V[grp]['water'], V[grp]['air']
        I = Mod.current((w, ai), Mod.E(d)); I0 = Mod.current((w, 0.0), Mod.E(d))
        c, n, m, q = G.fit_stack(d['t'], d['y'], d['e'], I, gf, d['zsd'])
        c0, n, m0, q0 = G.fit_stack(d['t'], d['y'], d['e'], I0, gf, d['zsd'])
        curves.append(dict(name='E ' + ('CF4' if gas == 'cf4' else 'CO2'), label=lab, color=ps.GAS['CF4' if gas == 'cf4' else 'CO2'],
                           t=d['t'], y=d['y'], e=d['e'], m=m, m0=m0, E=d['E'], water=w, o2=ai * 2095, chi2=c, chi2_noair=c0, n=n))
    for ci, cv in enumerate(curves):
        th = t_half(cv['t'], cv['m']); t = cv['t'] - th
        nm = (t >= PLAT[0]) & (t <= PLAT[1])
        f = np.nanmean(cv['y'][nm]); cv['tt'], cv['yn'], cv['en'] = t, cv['y'] / f, cv['e'] / f
        cv['mn'] = cv['m'] / np.nanmean(cv['m'][nm])
        cv['m0n'] = None if cv['m0'] is None else cv['m0'] / np.nanmean(cv['m0'][nm])
        for i in range(len(t)):
            if np.isfinite(cv['y'][i]):
                rows.append(dict(curve=cv['name'], t_ns=t[i], signal=cv['yn'][i], err=cv['en'][i], model=cv['mn'][i],
                                 model_no_O2=np.nan if cv['m0n'] is None else cv['m0n'][i], E_Vcm=cv['E']))
    fig, axs = ps.fig_for('full', height=2.45, ncols=3, gridspec_kw=dict(width_ratios=[1.25, 1, 1]), sharey=True)
    XL = (-250, 3300)

    def pts(ax, cv, ms=2.4, st=1, z=3):
        k = np.isfinite(cv['yn']) & (cv['tt'] > XL[0]) & (cv['tt'] < XL[1])
        ax.errorbar(cv['tt'][k][::st], cv['yn'][k][::st], yerr=cv['en'][k][::st], fmt='o', ms=ms, lw=0.6, color=cv['color'], zorder=z)
    for cv in curves:
        pts(axs[0], cv, st=2 if cv['name'] == 'A' else 1, z=5 if cv['name'] == 'A' else 3)
    for ax, cv in zip(axs[1:], curves[1:]):
        pts(ax, cv)
        ax.plot(cv['tt'], cv['mn'], '-', color=cv['color'], lw=1.3, zorder=4, label=f"gas fit, O$_2$ {cv['o2']:.0f} ppm")
        ax.plot(cv['tt'], cv['m0n'], '--', color=ps.INK, lw=1.1, zorder=4, label='same water, no O$_2$')
        ax.legend(loc='upper right', handlelength=1.4, fontsize=ps.BASE_PT * 0.78)
    for ax in axs:
        ax.set_xlim(*XL); ax.set_ylim(-0.1, 1.45); ax.set_xlabel('time after column start (ns)')
    axs[0].set_ylabel('strip-sum signal (plateau = 1)')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker='o', ls='', color=cv['color'], ms=3.5, label=cv['label']) for cv in curves]
    ps.legend_above(fig, hs, ncol=3)
    ps.panel_labels(axs)
    out = os.path.join(HERE, 'figures', 'attachment_charge')
    ps.save(fig, out, data=pd.DataFrame(rows))
    info = {cv['name']: {k: cv[k] for k in ('E', 'water', 'o2', 'chi2', 'chi2_noair', 'n') if k in cv} for cv in curves}
    print(json.dumps(info, indent=1))
    return info


if __name__ == '__main__':
    main()
