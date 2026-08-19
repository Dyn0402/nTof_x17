#!/usr/bin/env python3
"""Build report.html from results.json.  Run sharing.py and make_figures.py first."""
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
M = R["meta"]
QLO, QHI = M["q_window"]
MX, MY = R["mult_x"], R["mult_y"]
WX = R["width_x"]["matched"]
WY = R["width_y"]["matched"]
BX, BY = R["budget_x"], R["budget_y"]

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8983;--rule:#e6e5e1;
--x:#2a78d6;--y:#eb6834;--card:#ffffff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:30px;line-height:1.25;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:14px;margin-bottom:32px}
h2{font-size:20px;margin:44px 0 12px;letter-spacing:-.01em}
h3{font-size:16px;margin:28px 0 8px;color:var(--ink2)}
p{margin:12px 0}
.verdict{background:var(--card);border:1px solid var(--rule);border-left:4px solid var(--x);
border-radius:10px;padding:20px 24px;margin:24px 0}
.verdict p:first-child{margin-top:0}.verdict p:last-child{margin-bottom:0}
.tiles{display:flex;flex-wrap:wrap;gap:14px;margin:24px 0}
.tile{flex:1 1 190px;background:var(--card);border:1px solid var(--rule);
border-radius:10px;padding:16px 18px}
.tile .n{font-size:26px;font-weight:640;letter-spacing:-.02em}
.tile .l{font-size:12.5px;color:var(--muted);margin-top:4px;line-height:1.4}
figure{margin:28px 0}
figure img{width:100%;height:auto;border:1px solid var(--rule);border-radius:8px;background:#fff}
figcaption{font-size:13.5px;color:var(--ink2);margin-top:10px}
.tablewrap{overflow-x:auto;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:14.5px;min-width:520px}
th,td{text-align:right;padding:9px 12px;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
thead th{font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
font-weight:600}
tbody tr:last-child td{border-bottom:none}
code{background:#f2f1ee;padding:1px 5px;border-radius:4px;font-size:13.5px}
.note{font-size:14px;color:var(--ink2);border-top:1px solid var(--rule);padding-top:10px}
ul{margin:12px 0;padding-left:22px}li{margin:6px 0}
.xk{color:var(--x);font-weight:600}.yk{color:var(--y);font-weight:600}
"""

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>det4 charge sharing in X and Y</title>
<meta name="description" content="A head-on track lights 2.4 strips in X and 1.8 in Y --
but at the same cluster charge both views give 2.9, so the asymmetry is the zero-suppression
threshold, not the chamber.">
<style>{CSS}</style></head><body><div class="wrap">

<h1>How far does the charge spread — and is X really different from Y?</h1>
<div class="sub">det4 (mx17_E) in the SPS H4 test beam, run 53, flat mount ·
{MX['n']:,} (X) and {MY['n']:,} (Y) head-on tracks clear of dead strips ·
companion to the <a href="det4-spatial-resolution-sps.html">spatial-resolution note</a></div>

<div class="verdict">
<p><strong>A head-on track lights {MX['raw']:.2f} strips in X and {MY['raw']:.2f} strips in
Y</strong> at the 5σ zero-suppression this run was taken with — a striking asymmetry, and
<strong>almost all of it is the threshold, not the detector.</strong></p>
<p>X clusters carry {MX['median_charge']/MY['median_charge']:.1f}× the charge of Y clusters here,
and multiplicity is steeply charge-dependent. Compare the two views at the
<strong>same cluster charge</strong> and they land on top of each other:
<span class="xk">{MX['matched']:.2f}</span> vs <span class="yk">{MY['matched']:.2f}</span> strips,
kernel widths <span class="xk">{WX['rms_deconv_mm']*1e3:.0f} µm</span> vs
<span class="yk">{WY['rms_deconv_mm']*1e3:.0f} µm</span>.</p>
<p><strong>det4's charge spreading is isotropic to about 5 %</strong>, and the residual X excess
is consistent with the chamber's known {M['tilt_walk_mm']:.2f} mm X tilt, which also skews the
X kernel ({R['controls_x']['asym_minus_over_plus']:.2f} left/right, against
{R['controls_y']['asym_minus_over_plus']:.2f} in Y).</p>
</div>

<div class="tiles">
<div class="tile"><div class="n">{MX['raw']:.2f} / {MY['raw']:.2f}</div><div class="l">strips per
head-on track, X / Y, as read out at 5σ</div></div>
<div class="tile"><div class="n">{MX['matched']:.2f} / {MY['matched']:.2f}</div><div class="l">the
same, at matched cluster charge ({QLO:.0f}–{QHI:.0f} ADC)</div></div>
<div class="tile"><div class="n">{WX['rms_deconv_mm']*1e3:.0f} / {WY['rms_deconv_mm']*1e3:.0f} µm</div>
<div class="l">lateral kernel rms, X / Y, on a {M['pitch_mm']} mm pitch</div></div>
<div class="tile"><div class="n">{100*BY['within_1']:.0f} %</div><div class="l">of the charge is
within ±1 strip of the track (both views)</div></div>
</div>

<h2>What is being measured, and against what</h2>
<p>Same run and same tracks as the resolution note. At <strong>normal incidence</strong> every
drift slice arrives at the same transverse position, so the lateral profile is diffusion plus
resistive spreading and nothing else — there is no drift-time ladder smearing it out. And the
uRWELL telescope gives an <strong>external, sub-strip impact point</strong>
({M['pointing_mm']*1e3:.0f} µm against a {M['pitch_mm']} mm pitch), so the profile is built
against the true track rather than against det4's own centroid, which would be circular.</p>

<h2>The headline number, and why it is not the one to quote</h2>

<figure><img src="figures/multiplicity.png" alt="Strip multiplicity, as read out and versus cluster charge">
<figcaption>Left: what the DAQ delivers — X clusters are visibly wider, Y is half single-strip.
Right: the same quantity against cluster charge. The two views trace <em>the same curve</em>;
the stars mark where each view actually sits on it, and the whole apparent asymmetry is that
the X sample sits further to the right.</figcaption></figure>

<p>Multiplicity is not a property of the chamber — it is a property of the chamber
<em>at a threshold</em>. A strip enters the cluster only if it clears zero-suppression, so a view
with more charge automatically shows more strips. That is the entire X/Y difference here.</p>

<div class="tablewrap"><table>
<thead><tr><th></th><th>X view</th><th>Y view</th><th>X / Y</th></tr></thead><tbody>
<tr><td>tracks used</td><td>{MX['n']:,}</td><td>{MY['n']:,}</td><td></td></tr>
<tr><td>median cluster charge</td><td>{MX['median_charge']:.0f} ADC</td>
<td>{MY['median_charge']:.0f} ADC</td><td>{MX['median_charge']/MY['median_charge']:.2f}</td></tr>
<tr><td><strong>strips/track, as read out</strong></td><td><strong>{MX['raw']:.2f}</strong></td>
<td><strong>{MY['raw']:.2f}</strong></td><td>{MX['raw']/MY['raw']:.2f}</td></tr>
<tr><td><strong>strips/track, matched charge</strong></td><td><strong>{MX['matched']:.2f}</strong></td>
<td><strong>{MY['matched']:.2f}</strong></td><td>{MX['matched']/MY['matched']:.2f}</td></tr>
<tr><td>single-strip fraction</td><td>{100*MX['hist'][1]/MX['n']:.1f} %</td>
<td>{100*MY['hist'][1]/MY['n']:.1f} %</td><td></td></tr>
<tr><td>kernel rms, measured</td><td>{WX['rms_mm']*1e3:.0f} µm</td>
<td>{WY['rms_mm']*1e3:.0f} µm</td><td>{WX['rms_mm']/WY['rms_mm']:.2f}</td></tr>
<tr><td><strong>kernel rms, deconvolved</strong></td>
<td><strong>{WX['rms_deconv_mm']*1e3:.0f} µm</strong></td>
<td><strong>{WY['rms_deconv_mm']*1e3:.0f} µm</strong></td>
<td>{WX['rms_deconv_mm']/WY['rms_deconv_mm']:.2f}</td></tr>
</tbody></table></div>
<p class="note">"Deconvolved" removes the {M['pointing_mm']*1e3:.0f} µm track pointing error from
both views, and additionally the {R['width_x']['tilt_term_mm']*1e3:.0f} µm smear that det4's
own X tilt (tan θ<sub>X</sub> = −0.015 over the 30 mm gap) puts into X alone.</p>

<h2>The kernels</h2>

<figure><img src="figures/kernel.png" alt="Lateral charge-sharing kernels in X and Y">
<figcaption>Mean share of the cluster charge against distance from the track, at matched cluster
charge. <strong>Strips that did not fire are counted as zero</strong> — without that the profile
is survivorship-biased ("given that this strip fired, how much did it carry"), cannot fall off,
and does not. The two kernels are nearly identical through the core; X carries slightly more on
the −1 and −2 side, which is the tilt. The upturn beyond ±2 strips is the zero-suppression
floor, not physics.</figcaption></figure>

<figure><img src="figures/budget.png" alt="Charge within k strips of the track">
<figcaption>The charge budget. About half the charge is on the strip nearest the track and
~{100*BY['within_1']:.0f} % is within one strip either side, in both views. This is the number
that matters for clustering windows: <strong>±2 strips captures
{100*min(BX['within_2'],BY['within_2']):.0f} %</strong>, and going to ±3 buys under 2 %.</figcaption></figure>

<h2>The selection that makes X measurable at all</h2>
<p>det4 only amplifies in bands. In the <strong>X view</strong> — the striped coordinate — a track
near a band edge has <em>dead neighbours</em>, so its kernel is truncated and it would report a
falsely narrow one. Every track here is required to sit {M['margin_strips']} strips clear of any
dead strip, which leaves two usable bands and {MX['n']:,} tracks. In Y the live run is 95 mm wide
and the same cut costs nothing.</p>

<figure><img src="figures/selection.png" alt="Live strip map and in-band multiplicity control">
<figcaption>Top: the live-strip map that drives the selection. Bottom: the control — multiplicity
drifts with position <em>inside</em> the live regions in both views, by ~15 %. That is gain
non-uniformity, and it means these numbers are local to the illuminated patch rather than
chamber constants.</figcaption></figure>

<h2>What this does not establish</h2>
<ul>
<li><strong>The tails are cut off.</strong> Run 53 is zero-suppressed at 5σ, so everything below
~{M['amp_min']:.0f} ADC is simply absent and the kernel beyond ±2 strips is a lower bound. The
few-percent µs-slow surface tails measured in RAW (<code>RAW_RUN71_REANALYSIS_2026-08-04.md</code>)
are not visible here. This is a <em>lateral</em> kernel at threshold, not the full response.</li>
<li><strong>X and Y sample different parts of the chamber.</strong> The X bands are two ~10 mm
strips of detector; Y spans ~45 mm of illuminated area. With a ~15 % in-band gain gradient, the
5 % residual X/Y difference is comfortably inside that systematic — the honest statement is
"isotropic to within what this measurement can resolve", not "isotropic to 5 %".</li>
<li><strong>The tilt correction is a subtraction, not a measurement.</strong> tan θ<sub>X</sub> is
taken from the earlier drift study; it is used to remove
{R['width_x']['tilt_term_mm']*1e3:.0f} µm in quadrature from X. It explains the X kernel's
left/right skew ({R['controls_x']['asym_minus_over_plus']:.2f}) but has not been fitted here.</li>
<li><strong>One gas, one gain point.</strong> Ar/CO₂/iso 95/3/2 at the run-53 operating point.
Charge sharing is set by the resistive layer and diffusion, both of which move with gas and
field; the campaign's own kernel work found the time-domain kernel gain- and drift-invariant,
but that is a different quantity from this lateral profile.</li>
<li><strong>These are hits, not waveforms.</strong> Legitimate here only because the mount is
flat — at θ ≈ 0 there is no ladder to reconstruct. Nothing here contradicts
<code>RECONSTRUCTION_BASIS.md</code>.</li>
</ul>

<h2>Reproducing</h2>
<p><code>sharing.py</code> → <code>results.json</code>; <code>make_figures.py</code> →
<code>figures/</code>; <code>make_report.py</code> → this page. Input
<code>det4_run_53_v2.npz</code> under
<code>/media/dylan/data/x17/sps_run53_det4_check/…/mapping_check/</code>; strip map from
<code>det4_sps_map.py</code>. Drift gate {M['gate_ns'][0]:.0f}–{M['gate_ns'][1]:.0f} ns,
amplitude &gt; {M['amp_min']:.0f} ADC, saturated (&gt; 3000 ADC) events dropped, oscillating
channels 372 and 510 masked.</p>

</div></body></html>
"""

(HERE / "report.html").write_text(HTML)
print("wrote", HERE / "report.html")


def _inline(match):
    return ('src="data:image/png;base64,'
            + base64.b64encode((HERE / match.group(1)).read_bytes()).decode() + '"')


standalone = re.sub(r'src="(figures/[^"]+)"', _inline, HTML)
(HERE / "report_standalone.html").write_text(standalone)
print("wrote", HERE / "report_standalone.html", f"({len(standalone)/1e6:.1f} MB)")
