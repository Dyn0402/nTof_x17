#!/usr/bin/env python3
"""make_hv_scan_v2.py -- Paper I v2: the hv_scan figure (chambers A-D, interior efficiency and discharges per trigger
against the resist HV) with two changes only:
  * the same x range in all four panels (400-530 V), so each chamber's plateau is seen to move;
  * the plateau band is drawn from half-way to the scan point below the first plateau point to half-way to the scan
    point above the last one (the old band was the plateau points +-2 V, a hairline for chamber C, whose plateau is
    one point).  The plateau definition (within 1 point of the maximum) is unchanged.
Everything else is rerun/paper1_ops/make_report.py fig_hv_scan, on the same table (rerun/paper1_ops/results/tradeoff.csv,
the two-track efficiency rows already merged into it).

    .venv/bin/python papers/paper1_v2/figs/make_hv_scan_v2.py
Writes out/hv_scan_v2.{pdf,png,csv} and out/hv_scan_v2.json (plateaus, the long runs against their scans).
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OPS = os.path.join(REPO, 'rerun', 'paper1_ops', 'make_report.py')
spec = importlib.util.spec_from_file_location('ops_report', OPS)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)                       # module-level code only defines things (main() is guarded)
ps = R.ps
OUT = os.path.join(HERE, 'out')


def band(s, o):
    V = np.sort(s.resist_V.unique())
    lo, hi = o.plateau_lo, o.plateau_hi
    below, above = V[V < lo], V[V > hi]
    a = (lo + below.max()) / 2 if len(below) else lo - 2.5
    b = (hi + above.min()) / 2 if len(above) else hi + 2.5
    return float(a), float(b)


def main():
    import matplotlib.ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    T = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_ops', 'results', 'tradeoff.csv'))
    O = R.optimum(T)
    ps.use()
    order = sorted(R.SCANS, key=lambda r: R.ORDER.index(r[0]))
    fig, axes = ps.fig_for('full', height=2.35, ncols=len(order), sharey=True)
    data, tw, info = [], [], {}
    for ax, (det, run, drift) in zip(axes, order):
        s = R.scan_rows(T, det, run, drift)
        c = R.COL[det]
        o = O[(O.det == det) & (O.run == run)].iloc[0]
        a, b = band(s, o)
        ax.axvspan(a, b, color=c, alpha=0.16, lw=0)
        ax.errorbar(s.resist_V, s.eff_ig, yerr=s.err_ig, elinewidth=0.8,
                    **R._guide(ps.det(det, ms=3.4, mec='white', mew=0.4)))
        g = T[(T.det == det) & T.cls.isin(['golden_calib', 'golden_quoted'])]
        if len(g):
            ax.plot(g.resist_V, g.eff_ig, '*', ms=8, color=c, mec=R.INK, mew=0.5, zorder=6, ls='none')
            data.append(g.assign(golden=True)[['det', 'run', 'resist_V', 'drift_V', 'n_ig', 'eff_ig', 'err_ig', 'golden']])
        t = ax.twinx(); t.grid(False); t.spines['right'].set_visible(True)
        r = (s.epoch_rate_hz / s.trig_rate_hz).where(lambda v: v > 0)
        t.plot(s.resist_V, r, color=c, marker='o', ms=3.2, mfc='white', mew=0.9, ls='--', lw=0.9)
        tw.append(t)
        ax.set_title(f'({chr(97 + len(tw) - 1)}) ' + R.NAME[det], loc='left', color=c, fontweight='bold',
                     fontsize=ps.BASE_PT)
        ax.set_ylim(30, 101); ax.axhline(100, color=R.MUTED, lw=0.6, zorder=0)
        ax.set_xlim(400, 530)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
        data.append(s.assign(discharge_per_trig=s.epoch_rate_hz / s.trig_rate_hz)[
            ['det', 'run', 'resist_V', 'drift_V', 'n_ig', 'eff_ig', 'err_ig', 'discharge_per_trig']])
        info[ps.NAME[R.NAME[det]]] = dict(scan=run, drift_V=drift, V_best=float(o.V_best), eff_best=float(o.eff_best),
                                 plateau=[float(o.plateau_lo), float(o.plateau_hi)], band_drawn=[a, b],
                                 long_runs=[dict(run=gg.run, subrun=gg.subrun, V=float(gg.resist_V), eff=float(gg.eff_ig),
                                                 err=float(gg.err_ig), n=int(gg.n_ig),
                                                 discharge_loss_pct=float(100 * gg.f_in_discharge),
                                                 f_sat=float(gg.f_sat))
                                            for gg in g.itertuples()])
        sv = s[s.resist_V.isin(g.resist_V) & ~s.key.isin(g.key)]
        info[ps.NAME[R.NAME[det]]]["scan_at_long_run_V"] = [dict(V=float(x.resist_V), eff=float(x.eff_ig), err=float(x.err_ig),
                                                         discharge_loss_pct=float(100 * x.f_in_discharge))
                                                    for x in sv.itertuples()]
    for t in tw[1:]:
        t.sharey(tw[0])
    tw[0].set_ylim(0, 0.5)
    for t in tw[:-1]:
        t.tick_params(axis='y', right=False, labelright=False)
    tw[-1].set_ylabel('discharges per trigger')
    fig.supxlabel('resistive-strip HV [V]', fontsize=ps.BASE_PT)
    axes[0].set_ylabel('interior efficiency [%]')
    h = [Line2D([], [], color=R.INK, marker='o', ms=3.6, lw=0.9, label='interior efficiency (left)'),
         Line2D([], [], color=R.INK, marker='o', ms=3.4, lw=0.9, ls='--', mfc='white', label='discharges per trigger (right)'),
         Line2D([], [], ls='none', marker='*', ms=8, color='white', mec=R.INK, mew=0.6, label='long run'),
         Patch(color=R.INK, alpha=0.16, lw=0, label='efficiency plateau')]
    ps.legend_above(fig, handles=h, ncol=4)
    os.makedirs(OUT, exist_ok=True)
    ps.save(fig, os.path.join(OUT, 'hv_scan_v2'), data=pd.concat(data, ignore_index=True))
    json.dump(info, open(os.path.join(OUT, 'hv_scan_v2.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
