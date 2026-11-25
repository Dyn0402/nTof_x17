#!/usr/bin/env python3
"""make_report.py -- report.html and figures/ of the window-bias study, from results/{inventory,toy,datacut,attboot_summary}.csv."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, 'results'), os.path.join(HERE, 'figures')
COL = {'A': '#C44E52', 'B': '#4C72B0', 'C': '#55A868', 'D': '#CCB974', 'E': '#8172B2'}


def short(r):
    s = r.subrun.replace('resist_det6_', '').replace('_drift_700V', '').replace('drift_scan_resist_490V_drift_', 'drift ')
    return f'{r.chamber} {r.run[-7:-3] if "6-2" in r.run else ""} {s}'.strip()


def main():
    os.makedirs(FIG, exist_ok=True)
    inv = pd.read_csv(os.path.join(RES, 'inventory.csv'))
    toy = pd.read_csv(os.path.join(RES, 'toy.csv'))
    cut = pd.read_csv(os.path.join(RES, 'datacut.csv'))
    att = pd.read_csv(os.path.join(RES, 'attboot_summary.csv'))

    # 1. margins
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    d = inv.sort_values('margin').reset_index(drop=True)
    ax.barh(range(len(d)), d.margin, color=[COL[c] for c in d.chamber])
    ax.set_yticks(range(len(d))); ax.set_yticklabels([short(r) for r in d.itertuples()], fontsize=6)
    ax.axvline(0, color='k', lw=0.8); ax.axvline(500, color='k', lw=0.6, ls=':')
    ax.set_xlabel('record end - column end (ns)   [dotted: length of the electronics response, 99 % of its area]')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'margins.png'), dpi=140); plt.close(fig)

    # 2. toy
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
    for ax, tau in zip(axs, (np.inf, 4000.0, 1500.0)):
        for row, g in toy[(toy.tau == tau) & toy.noise].groupby('row'):
            s = g.groupby('margin').bias.agg(['mean', 'std'])
            ax.errorbar(s.index, s['mean'], s['std'], fmt='o-', ms=3, lw=0.8, capsize=2, label=row)
        ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6, ls=':')
        ax.set_ylim(-150, 150); ax.set_xlabel('record end - column end (ns)')
        ax.set_title('no attachment' if not np.isfinite(tau) else f'attachment tau = {tau / 1e3:.1f} us', fontsize=9)
    axs[0].set_ylabel('T fit - T true (ns), uniform-column fit'); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'toy.png'), dpi=140); plt.close(fig)

    # 3. data cut
    c = cut[(cut.boot == 0) & (cut.cut_margin != 'full')].copy()
    c['m'] = c.cut_margin.astype(float)
    sd = cut[(cut.boot > 0) & (cut.cut_margin != 'full')].groupby(['key', 'cut_margin']).dT.std()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for key, g in c.groupby('key'):
        g = g.sort_values('m')
        e = np.array([sd.get((key, cm), np.nan) for cm in g.cut_margin])
        ok = np.abs(g.dT) < 200
        ax.errorbar(g.m[ok], g.dT[ok], e[ok], fmt='o-', ms=2.5, lw=0.7, capsize=1.5, color=COL[g.chamber.iloc[0]], alpha=0.8)
    for ch, cc in COL.items():
        ax.plot([], [], 'o-', color=cc, label=ch)
    ax.axvline(116, color='k', lw=0.8, ls='--'); ax.text(120, -140, "E 6-24's margin", fontsize=8)
    ax.axhline(0, color='k', lw=0.6); ax.set_ylim(-150, 60)
    ax.set_xlabel('record end - column end after the cut (ns)'); ax.set_ylabel('T(cut) - T(full record) (ns)')
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'datacut.png'), dpi=140); plt.close(fig)

    # 4. attachment
    a = att.copy().reset_index(drop=True)
    a['lab'] = [short(r) for r in a.itertuples()]
    a = a.sort_values(['chamber', 'lab']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(a.dT, range(len(a)), xerr=a.dT_sd, fmt='o', ms=3, ecolor='gray', capsize=2)
    for i, r in a.iterrows():
        ax.plot(r.dT, i, 'o', color=COL[r.chamber], ms=4)
    ax.set_yticks(range(len(a))); ax.set_yticklabels(a.lab, fontsize=6)
    ax.axvline(0, color='k', lw=0.6); ax.set_xlim(-60, 220)
    ax.set_xlabel('T (attachment free) - T (uniform column) (ns), event bootstrap')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'attachment.png'), dpi=140); plt.close(fig)

    tb = a[['chamber', 'run', 'subrun', 'n', 'margin', 'T_uni', 'T_att', 'dT', 'dT_sd', 'tau_us', 'k_sig']].copy()
    tb['tau_us'] = tb.tau_us.where(tb.tau_us < 100, np.inf)
    ct = c.pivot_table(index=['chamber', 'subrun'], columns='m', values='dT')[[0.0, 50.0, 116.0, 200.0, 300.0, 500.0]]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift gap: record end and attachment</title>
<style>body{{font-family:sans-serif;max-width:1050px;margin:auto;padding:16px;line-height:1.45}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:2px 6px}}img{{max-width:100%}}</style></head><body>
<h1>Drift gap: does the end of the record bias the drift time?</h1>
<p><b>Verdict.</b> A record that ends after the drift column does not bias the fitted drift time T. The fit recovers T
within 5 ns in toys, and cutting the real stacks of chambers A and B (and D's long run) at 120 ns after the column moves T by
1-7 ns. Chamber E's 6-24 long run, the quoted row with the shortest record (116 ns after the column), is therefore fine:
its gap moves by less than 0.2 mm, and a fit with attachment free finds none (tau 22 us, T +2 ns). Only a record that ends
before the column end biases T; that is the 177 V/cm point of chamber A, already a lower limit.</p>
<p><b>What does bias T is attachment</b>, and a short record makes it worse. The uniform-column fit returns T too short when
the column loses charge with drift time. Chamber C's 26 June rows prefer an attachment time of about 2 us (1.5-2.4 sigma per
row): with the loss fitted, C's gap is 30.1 instead of 29.4 mm (+0.7 mm). D's 26 June HV-scan rows lean the same way
(+0.25 mm). This is stated as a limit in Paper I; the choice is Dylan's.</p>
<p><b>Found on the way: chamber D's long run had no fit.</b> Its interior stack was empty (masked X channels below 270 left
the +-2 strip offset under the 60 % rule), and x_width.fit_x returned its starting point (T = 820 ns, rms NaN) as a result.
D's water (0.48 %) and its long-run gap came from that number. Refitted with the S4 det7 recipe (X peak on ch 256-511, all
events): T = 798.6 +- 17.2 ns (the S4 re-check had 797 +- 17). D's water is now 0.44 %, its gap 29.4 mm (chi2 over its rows
12.6 -> 4.4 / 6), and D's drying 0.67 -> 0.33 %. fit_x now raises on an empty stack.</p>

<h2>Clock</h2><p>The fit's dt is on predict_x's clock (index 0 = -300 ns) with a template whose grid starts at -360 ns, so the
last charge of the column reaches the strips at stack time dt + 60 + T. The response then lasts ~500 ns (99 % of its
area); the falling edge of the stack is the column end convolved with it. Paper I's Fig. two_velocities (a) draws its
triangles at dt and dt + T, 60 ns early (cosmetic; T is unaffected).</p>

<h2>Record end against column end, every row</h2><img src="figures/margins.png">
<p>Record end = the last time where more than 60 % of the stacked events still have a sample (rc_diffusion.stack's rule).
Chamber E's records are short because its leading edges sit late in the readout window for many events (10 % of events end
720-920 ns after their leading edge).</p>

<h2>Toy: the fit's own model, cut and refitted</h2><img src="figures/toy.png">
<p>Truth = x_width.predict_x at each template row's fitted parameters (sigma_0, Dd, dt, T), with an exp(-u/tau) loss of
charge in the attachment panels; white noise of the row's fit rms; six noisy replicas per point. Without attachment, T is
unbiased for every margin >= 0 and unusable below. With attachment, the uniform fit is short already with the full
record (-7 to -9 ns at 4 us, -26 to -48 ns at 1.5 us) and shorter still when the record is cut.</p>

<h2>Real stacks cut earlier</h2><img src="figures/datacut.png">
<p>Shift of T when the real stack of a row with 400-800 ns of margin is cut at a given distance after its own column end
(same events at every cut; error bars: event bootstrap, 12 resamples). Values at selected cuts (ns):</p>
{ct.round(0).to_html()}

<h2>Fit with attachment free</h2><img src="figures/attachment.png">
<p>The same model with the charge of the slice drifting for u weighted by exp(-u/tau), 1/tau >= 0 free; closure on toys
recovers T within 5 ns and tau within ~10 % (results/attfit_closure.csv). k_sig = (1/tau) / its bootstrap sd.</p>
{tb.round(2).to_html(index=False)}

<h2>What this does not rule out</h2>
<ul><li>The attachment model is one shape (exponential loss along the column). Anything else that makes the stack fall
with drift time (a gain or a coupling that changes with depth) looks the same to the fit; the toys show the size of the
effect, not its cause.</li>
<li>Chamber C's attachment is at 1.5-2.4 sigma per row; with more rows the trend is clearer (tau grows through the
26 June HV scan), but there is no independent attachment measurement for C.</li>
<li>The toys use white noise; real residuals are correlated. The cut test on real stacks covers that for the
chambers that have margin to spare, not for E itself.</li></ul>
<p>Scripts: inventory.py, toy.py, datacut.py, attfit.py, attboot.py, make_report.py (this directory). Inputs: EOS
x17rerun/paper1_gas (fit_v1, head-on caches) and the arms' template bundles, copied to $WORK_DATA/claude-jobs/a21e7df1/gapwin;
the corrected fits are x17rerun/paper1_gas/fit_v2 (D replaced, failed arms dropped).</p>
</body></html>"""
    open(os.path.join(HERE, 'report.html'), 'w').write(html)
    print('report.html')


if __name__ == '__main__':
    main()
