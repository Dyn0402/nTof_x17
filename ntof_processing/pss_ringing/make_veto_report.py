#!/usr/bin/env python3
"""Build report_veto.html -- how to reject plastic after-pulses in the slim.

Numbers come from veto_scan.json, so re-running the scan and then this script
keeps the recommendation and the tables consistent.

    python make_veto_report.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
CSS = (HERE / 'make_report.py').read_text().split('CSS = """')[1].split('"""')[0]


def pct(x):
    return f'{x * 100:.1f}&nbsp;%'


def main():
    r = json.loads((HERE / 'veto_scan.json').read_text())
    base = r['base']
    amp = {a['amp_floor']: a for a in r['amp_scan']}
    comb = {(c['amp_floor'], c['t_hold'], c['ratio']): c for c in r['combined']}
    shadow = {(s['t_hold'], s['ratio']): s for s in r['scan']}
    prim = r['primary']

    A0, T0, R0 = 250.0, 1000.0, 0.05
    rec = shadow[(T0, R0)]          # the recommendation: the shadow flag alone
    amp_only = amp[A0]
    comb150 = comb[(150.0, T0, R0)]

    amp_rows = ''.join(
        f"<tr><td>{a['amp_floor']:.0f}</td><td>{pct(a['core_kept'])}</td>"
        f"<td>{pct(a['mid_removed'])}</td><td>{pct(a['late_removed'])}</td></tr>"
        for a in r['amp_scan'])
    sh_rows = ''.join(
        f"<tr><td>{s['t_hold']:.0f}</td><td>{s['ratio']:.2f}</td>"
        f"<td>{pct(s['core_kept'])}</td><td>{pct(s['mid_removed'])}</td>"
        f"<td>{pct(s['late_removed'])}</td><td>{pct(s['control_vetoed'])}</td></tr>"
        for s in r['scan'] if s['ratio'] in (0.02, 0.05, 0.10))
    cb_rows = ''.join(
        f"<tr><td>{c['amp_floor']:.0f}</td><td>{c['t_hold']:.0f}</td>"
        f"<td>{c['ratio']:.2f}</td><td>{pct(c['core_kept'])}</td>"
        f"<td>{pct(c['mid_removed'])}</td><td>{pct(c['late_removed'])}</td></tr>"
        for c in r['combined'] if c['t_hold'] in (300.0, 1000.0))
    pr_rows = ''.join(
        f"<tr><td>{k}</td><td>{v['n']:,}</td><td>{pct(v['within_core'])}</td>"
        f"<td>{v['median']:+.1f}</td></tr>" for k, v in prim.items())

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rejecting plastic after-pulses in the slim</title><style>{CSS}</style></head>
<body><main>

<h1>Rejecting plastic after-pulses in the slim</h1>
<p class="sub">Companion to <a href="report.html">&ldquo;Do the n_TOF plastics
ring?&rdquo;</a> &middot; measured on the reference pair
run_79/stat090_0000 &times; 224572, 400 bunches, slimmed at &plusmn;3&nbsp;&micro;s</p>

<div class="verdict">
<p><strong>One per-hit flag, and one per-trigger metric.</strong></p>
<p><strong>The flag &mdash; is this hit in the shadow of a bigger one?</strong>
<code>amp_0 &lt; {R0:.2f} &times; max(amp_0 on the same channel in the previous
{T0:.0f} ns)</code>. It removes <strong>{pct(rec['late_removed'])}</strong> of the
150&ndash;1000&nbsp;ns excess and <strong>{pct(rec['mid_removed'])}</strong> of
the 25&ndash;150&nbsp;ns excess, for <strong>{pct(1 - rec['core_kept'])}</strong>
of the core &mdash; and the core it costs is entirely the small-amplitude end.</p>
<p><strong>The metric &mdash; is the main peak where it should be?</strong> Per
(DREAM trigger, arm) take the <strong>largest-amplitude</strong> plastic hit, not
the earliest, and cut on its residual. On the arm the trigger was assigned to,
<strong>{pct(prim['largest, trigger arm']['within_core'])}</strong> land within
&plusmn;25&nbsp;ns at a median of
{prim['largest, trigger arm']['median']:+.1f}&nbsp;ns. Choosing the earliest hit
instead gives a median of {prim['earliest hit']['median']:+.0f}&nbsp;ns, because
in a microsecond-wide window the earliest hit is usually an unrelated single.</p>
<p><strong>If you cannot add a branch:</strong> <code>amp_0 &gt; {A0:.0f}</code>
alone, computable from the slim exactly as it stands today, already removes
{pct(amp_only['late_removed'])} of the late tail for the same core cost. It is
worse only in the 25&ndash;150&nbsp;ns band
({pct(amp_only['mid_removed'])} against {pct(rec['mid_removed'])}).</p>
</div>

<h2>The metric, in the form to implement</h2>
<p>Per plastic hit, computed on the <strong>full n_TOF stream</strong> with a full
{T0:.0f}&nbsp;ns of lookback &mdash; not on the slim, because an after-pulse whose
parent sits just outside the slim window is exactly the case a slim-only
recomputation gets wrong:</p>
<pre><code>shadow  = amp_0 / max(amp_0 of same-channel hits in the previous T ns)   # 0 if none
dt_prev = ns since that hit

after-pulse candidate  <=>  shadow &lt; R          (R = {R0:.2f}, T = {T0:.0f} ns)
usable plastic hit     <=>  not an after-pulse candidate</code></pre>
<p><strong>Store <code>shadow</code> and <code>dt_prev</code>, do not store the
boolean.</strong> Two float32 per hit is ~8&nbsp;B, about 18&nbsp;MB on a 74&nbsp;MB
segment, and it lets an analysis re-tune R and T without re-slimming 21&nbsp;TB.
A boolean freezes a choice that this study has only tuned on one segment.</p>

<h2>The amplitude floor: the simpler alternative</h2>
<p>Worth measuring first, because it needs no lookback and no new branch. It
works at all because the background-subtracted <code>amp_0</code> spectra of the
core and of the tail barely overlap &mdash; the core is a MIP peak, the tail is
piled up near threshold:</p>
<div class="scroll"><table>
<thead><tr><th>amp_0 &gt;</th><th>core kept</th><th>25&ndash;150 ns removed</th>
<th>150&ndash;1000 ns removed</th></tr></thead>
<tbody>{amp_rows}</tbody></table></div>
<p class="sub">Core = background-subtracted plastic excess at |dt| &lt; 25&nbsp;ns
({base['core']:,} hits); the tails are {base['mid']:,} and {base['late']:,}.</p>

<h2>The shadow flag on its own</h2>
<div class="scroll"><table>
<thead><tr><th>T [ns]</th><th>R</th><th>core kept</th>
<th>25&ndash;150 removed</th><th>150&ndash;1000 removed</th>
<th>control-window hits vetoed</th></tr></thead>
<tbody>{sh_rows}</tbody></table></div>
<p class="sub">The last column is the flag firing on the +100&nbsp;&micro;s
accidental control, i.e. on hits that <em>cannot</em> be after-pulses of anything
in the window. It runs far above the core loss because control hits are small
random singles while core hits are MIPs &mdash; which is the same fact that makes
the amplitude floor work. <strong>Lookback, not ratio, is what buys tail
removal</strong>: at R = 0.05, T = 100&nbsp;ns removes
{pct(shadow[(100.0, 0.05)]['late_removed'])} of the late tail and
T = 1000&nbsp;ns removes {pct(shadow[(1000.0, 0.05)]['late_removed'])}.</p>

<h2>Both together &mdash; and why the floor adds nothing on top</h2>
<p>At equal core cost the shadow flag beats the amplitude floor
({pct(rec['late_removed'])} of the late tail against
{pct(amp_only['late_removed'])}, {pct(rec['mid_removed'])} of the middle band
against {pct(amp_only['mid_removed'])}), and combining them is not better than
the flag alone &mdash; a floor of 150&nbsp;ADC on top gives
{pct(comb150['late_removed'])} / {pct(comb150['mid_removed'])} at
{pct(1 - comb150['core_kept'])} core cost. The two cuts are removing the same
hits, because a hit small enough to be in the shadow of a MIP is a small hit.</p>
<div class="scroll"><table>
<thead><tr><th>amp_0 &gt;</th><th>T [ns]</th><th>R</th><th>core kept</th>
<th>25&ndash;150 removed</th><th>150&ndash;1000 removed</th></tr></thead>
<tbody>{cb_rows}</tbody></table></div>

<figure>
  <img src="figures/veto_dt.png" alt="DREAM residual before and after the veto">
  <figcaption><strong>Left:</strong> the background-subtracted plastic residual
  against the corrected DREAM prediction, before and after the veto. The core
  survives; the late shoulder does not. <strong>Right:</strong> the same data at
  1&nbsp;ns binning. The late side is <em>not</em> featureless: there is a bump at
  70&ndash;90&nbsp;ns, where the 81&nbsp;ns echo lands once the parent&rsquo;s own
  residual smears it. That refines
  <code>slim_pipeline/config.py</code>&rsquo;s note that the tail &ldquo;falls
  smoothly and monotonically with no discrete echoes&rdquo; &mdash; smooth at
  5&nbsp;ns binning, not featureless.</figcaption>
</figure>

<figure>
  <img src="figures/veto_roc.png" alt="Tail removed against genuine hits vetoed">
  <figcaption><strong>What the shadow flag buys against what it costs.</strong>
  Each curve is one lookback, each marker one ratio.</figcaption>
</figure>

<h2>The per-trigger metric</h2>
<p>&ldquo;Is the main peak where it should be?&rdquo; &mdash; per (DREAM trigger,
arm), pick one plastic hit and cut on its residual:</p>
<div class="scroll"><table>
<thead><tr><th>how the primary is chosen</th><th>(trigger, arm) pairs</th>
<th>within &plusmn;25 ns</th><th>median residual [ns]</th></tr></thead>
<tbody>{pr_rows}</tbody></table></div>
<p><strong>Choose the primary by amplitude, not by time.</strong> In a window
several microseconds wide the earliest hit is almost always an unrelated single,
which is why &ldquo;earliest&rdquo; sits at a median of
{prim['earliest hit']['median']:+.0f}&nbsp;ns. Restricting to the arm the trigger
was actually assigned to is what turns this into a usable efficiency.</p>

<h2>What this does not settle</h2>
<ul>
<li><strong>One segment, one run.</strong> Tuned on run_79/stat090_0000 &times;
224572 over 400 bunches. The plastics&rsquo; singles rate varies across the
campaign and the shadow flag&rsquo;s cost scales with it, so R and T should be
re-checked on a high-rate and a low-rate segment before a campaign-wide number is
quoted.</li>
<li><strong>The {pct(1 - rec['core_kept'])} core loss is not all loss.</strong>
Some of the small-amplitude core excess is genuinely correlated with the trigger
and some of it may itself be after-pulsing that happens to land inside
&plusmn;25&nbsp;ns (the effect turns on at 18&nbsp;ns). This study does not
separate those.</li>
<li><strong>No wall-side requirement is included.</strong> The metric above is
plastic-only; combining it with the wall coincidence is the analysis&rsquo;s
choice and is not measured here.</li>
<li><strong>The amplitude floor is in ADC, not in MIPs.</strong> Per-channel gains
differ, so {A0:.0f} ADC is not the same physical threshold on all four arms. It
should be set per channel off the MIP peak before it is used for a physics
yield.</li>
</ul>

</main></body></html>
"""
    out = HERE / 'report_veto.html'
    out.write_text(html)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
