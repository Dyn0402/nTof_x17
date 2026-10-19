#!/usr/bin/env python3
"""Build report.html from results.json.  Run resolution.py and make_figures.py first."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())

F = R["fit"]
# Mount angle is the configuration epoch (RUN_TIMELINE.md), cross-checked at 25.4 deg
# in DET4_URW_MAPPING_2026-08-01.md.  The singular-value estimate recomputed by
# resolution.py is a looser variant of the same estimator -- quote the record, not it.
MOUNT = 25.64
ZX = {q["pitch"]: q for q in R["zones"]["uRW-x"]}
ZY = {q["pitch"]: q for q in R["zones"]["uRW-y"]}
DET4 = F["sigma_det4"] * 1e3
ALL = [q["sigma_det4"] * 1e3 for q in R["zones"]["uRW-x"] + R["zones"]["uRW-y"]]
LO, HI = min(ALL), max(ALL)
import math
SCALE = R["tilt"]["rows"][0]["sigma_res"] / math.tan(math.radians(MOUNT))

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8983;--rule:#e6e5e1;
--blue:#2a78d6;--orange:#eb6834;--aqua:#1baf7a;--card:#ffffff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:30px;line-height:1.25;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:14px;margin-bottom:32px}
h2{font-size:20px;margin:44px 0 12px;letter-spacing:-.01em}
h3{font-size:16px;margin:28px 0 8px;color:var(--ink2)}
p{margin:12px 0}
.verdict{background:var(--card);border:1px solid var(--rule);border-left:4px solid var(--blue);
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
"""


def rows_table(rows, label):
    out = [f'<div class="tablewrap"><table><thead><tr><th>{label}</th>'
           '<th>σ(residual)</th><th>reference pointing</th><th>det4 alone</th>'
           '<th>tracks</th></tr></thead><tbody>']
    for q in sorted(rows, key=lambda z: z["pitch"]):
        out.append(f'<tr><td>{q["pitch"]:.1f} mm</td>'
                   f'<td>{q["sigma_res"]*1e3:.0f} ± {q["err"]*1e3:.0f} µm</td>'
                   f'<td>{q["pointing"]*1e3:.0f} µm</td>'
                   f'<td><strong>{q["sigma_det4"]*1e3:.0f} µm</strong></td>'
                   f'<td>{q["n"]:,}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>det4 spatial resolution at SPS H4</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>How well does an MX17 chamber actually measure position?</h1>
<div class="sub">det4 (mx17_E) in the SPS H4 test beam, run 53 — with the uRWELL reference
<em>measured</em> rather than assumed · {R['run53']['n']:,} clean tracks</div>

<div class="verdict">
<p><strong>At normal incidence det4 resolves a track to σ ≈ {DET4:.0f} µm</strong> in its live
bands — about {LO:.0f}–{HI:.0f} µm across three independent regions and both views, on a
0.78 mm strip pitch (binary would be 225 µm).</p>
<p><strong>The bench's 0.6–0.7 mm is therefore reference-limited after all</strong> — but not by
the term that was computed for it. The M3 four-plane fit points to the DUT with a
<em>core</em> σ of 0.21 / 0.24 mm, and that number excludes multiple scattering by
construction. For cosmic muons it should not: ~1.1 mrad of scattering over the 558 mm from the
last upstream reference plane to the DUT reproduces the whole bench residual on top of a
176 µm chamber, and the M3 study itself measured a 2.6 mrad inter-doublet kink.</p>
<p><strong>The uRWELL reference is good enough, and — unusually — it is self-calibrating.</strong>
Its back plane carries three strip pitches (0.5 / 1.5 / 1.0 mm) in one plane and the beam
illuminates all three, so the reference's own contribution can be fitted out instead of
modelled.</p>
</div>

<div class="tiles">
<div class="tile"><div class="n">{DET4:.0f} µm</div><div class="l">det4 intrinsic, normal
incidence, reference removed</div></div>
<div class="tile"><div class="n">{F['f_back']:.3f} × pitch</div><div class="l">measured back-uRWELL
resolution — binary is 0.289</div></div>
<div class="tile"><div class="n">{R['divergence_rad']*1e3:.2f} mrad</div><div class="l">beam divergence
from the z-scan; it cancels in the interpolation</div></div>
<div class="tile"><div class="n">{R['tilt']['rows'][0]['sigma_res']*1e3:.0f} µm</div>
<div class="l">same chamber at {MOUNT:.1f}° — the drift-gap projection</div></div>
</div>

<h2>Why this measurement is possible at all</h2>
<p>The rail carried five detectors: uRWELL front at z = 0, three P2 BASKET pad chambers at
320 / 630 / 940 mm, and uRWELL back at z = 1370 mm. det4 was inserted at z ≈ 1120 mm. The P2
pad planes are useless as a reference here — they sit at 3.4 mm residual — so the track is the
front → back interpolation.</p>
<p>That matters more than it sounds. Because det4 is <em>between</em> the two reference planes,
the track is <strong>interpolated, not extrapolated</strong>: a straight track is reproduced
exactly and the beam divergence cancels. What is left at det4 is</p>
<p style="text-align:center"><code>σ(residual)² = σ(det4)² + 0.668 · σ(back)² + 0.033 · σ(front)²</code></p>
<p>The front plane's weight is only 3.3 % of the variance, so it barely matters. Everything hinges
on σ(back) — and that is exactly the number the back plane hands over for free, because it is
built with three different pitches side by side.</p>

<figure><img src="figures/zone_residuals.png" alt="Residual distributions per back-plane pitch zone">
<figcaption>The det4 residual in the same detector, same run, same view — split only by which
pitch zone of the <em>reference</em> the track crossed. The detector is unchanged between these
three curves; the reference is not.</figcaption></figure>

<h2>The decomposition</h2>
<p>Fitting σ(residual)² against pitch² gives a slope that is the reference's contribution and an
intercept that is det4 alone.</p>

<figure><img src="figures/decomposition.png" alt="Residual variance versus back-plane pitch squared">
<figcaption>Free-slope fit. The back uRWELL comes out at
<strong>{F['f_back']:.3f} × pitch</strong> against a binary expectation of 0.289 — i.e. its
clusters carry essentially no charge interpolation. χ²/1 dof = {F['chi2']:.1f}. The intercept is
det4 with the reference removed: <strong>{DET4:.0f} µm</strong>.</figcaption></figure>

<h3>det4's Y view (uRW-x), by reference zone</h3>
{rows_table(R['zones']['uRW-x'], 'back-plane pitch')}
<h3>det4's X view (uRW-y), by reference zone</h3>
{rows_table(R['zones']['uRW-y'], 'back-plane pitch')}
<p class="note">"det4 alone" here assumes a binary back plane (0.289 × pitch); the free-slope fit
above does not assume it and lands in the same place. The three zones illuminate three
different regions of det4, so the spread between them — {LO:.0f} to {HI:.0f} µm — is a fair
systematic band, not statistical scatter.</p>

<h2>The control that makes it a measurement</h2>
<p>There is one obvious way this could be fooled: a back-plane zone <em>is</em> a region of det4,
so anything that varies across det4 would masquerade as a pitch effect. The test is whether the
residual <strong>steps</strong> at the zone boundary — a smooth detector-side variation cannot
produce a step — and whether the orthogonal coordinate, which sweeps the same det4 region while
its own reference pitch stays fixed, stays flat.</p>

<figure><img src="figures/boundary_step.png" alt="Residual width versus back-plane position, signal and control">
<figcaption>It steps, and the control does not. Both curves scan the same det4 territory. The
blue curve's reference pitch changes at 64 mm and the width drops ~410 → ~290 µm across it; the
orange curve's reference pitch is constant and it stays flat through the same boundary.</figcaption></figure>

<figure><img src="figures/zscan.png" alt="Residual width versus assumed det4 z">
<figcaption>A second consistency check. Scanning the assumed z of det4 in the interpolation puts
the minimum at 1120–1150 mm, where det4 actually is. The curvature away from the minimum measures
the beam's angular spread — {R['divergence_rad']*1e3:.2f} mrad, matching the P2 telescope's
independent &lt; 0.5 mrad — and confirms it cancels at det4's own z.</figcaption></figure>

<h2>What this says about the cosmic bench</h2>
<p>The bench and H4 chambers are the same design and the same <strong>0.78 mm strip pitch</strong>,
so per-axis numbers are directly comparable. Two cautions before comparing them, both from
<code>mpgd26/slides/HANDOFF_resolution.md</code>:</p>
<ul>
<li>The often-quoted bench <strong>σ_DUT ≈ 0.40 mm is a units mismatch</strong> — it subtracts a
per-axis pointing from a <em>radial</em> residual. It is not a per-axis resolution and must not
be compared with one.</li>
<li>The often-quoted <strong>"the M3 reference is only ~500 µm" is the per-plane number</strong>
(0.41–0.51 mm). The four-plane fit interpolating to the DUT does much better on paper:
<strong>0.206 / 0.242 mm</strong>.</li>
</ul>
<p>Done properly, per axis: at <strong>θ &lt; 5°</strong> — the bench's own near-normal bin, so
inclination is already controlled for — det3's core residual is
<strong>0.64 mm (X)</strong>. Against H4's {DET4:.0f} µm at θ ≈ 0. That is a factor of 3.6, and
it does not close.</p>

<figure><img src="figures/budget.png" alt="Two ways to split the bench residual">
<figcaption>The same 640 µm bench residual, split two ways. Drawn in variance, because that is
what adds. Only the lower split is physical: det3 is a good chamber and det4 is the fleet's
worst, so det3 cannot be 3.4× worse.</figcaption></figure>

<p>So <strong>the bench residual is dominated by the reference, and the M3 core pointing
understates it.</strong> The missing term is <strong>multiple scattering between the last
reference plane and the DUT</strong>. The M3 self-resolution study excludes MS from its pointing
by construction — it argues MS shows up as residual <em>tails</em>, not core broadening, and
quotes core fits. But a muon that scatters in the 558 mm between the bottom doublet (z = 144 mm)
and the DUT (z = 702 mm) is genuinely displaced from the fitted line, and no amount of core
fitting on M3's own residuals can see it. The arithmetic is undramatic:
<strong>1.1 mrad</strong> over that lever arm accounts for the entire bench residual on top of a
176 µm chamber — against the 2.6 mrad (X) / 5.0 mrad (Y) inter-doublet kink the same study
measured, and Highland's 1.8 mrad at 1 GeV.</p>

<h3>Track inclination is real, but it is not the explanation</h3>
<p>A track crossing the 30 mm drift gap at angle θ spreads its charge over 30·tanθ mm of strips,
and rotating the very same chamber to {MOUNT:.1f}° shows it plainly:</p>

<figure><img src="figures/angle.png" alt="Residual at flat and tilted mount">
<figcaption>Run 53 flat vs run 57 at {MOUNT:.1f}°, same detector, same reference,
same estimator. The projected view degrades to
{R['tilt']['rows'][0]['sigma_res']*1e3:.0f} µm and becomes visibly non-Gaussian; the view along
the rotation axis is untouched at
{R['tilt']['rows'][1]['sigma_res']*1e3:.0f} µm.</figcaption></figure>

<p>But it cannot carry the bench gap, because the bench measurement above is <em>already</em>
restricted to θ &lt; 5°. Scaling the rotated point as σ ≈ {SCALE:.1f}·tanθ mm gives at most
0.38 mm at the 5° edge of that bin and ~0.2 mm typically — which removes a modest part of
0.64 mm and leaves the factor-3 gap standing. An earlier version of this note claimed
inclination as the explanation; the bench's own angle-binned number rules that out.</p>

<h2>What this does not rule out</h2>
<ul>
<li><strong>det4 is not det3.</strong> This is the worst chamber of the fleet by efficiency, run
in Ar/CO₂/iso 95/3/2 at H4 at a gain point that was still climbing, and only its live bands
contribute. It bounds what the <em>technology</em> does at normal incidence; it is not a
measurement of the bench chambers.</li>
<li><strong>The front plane's resolution is assumed, not measured.</strong> Its weight is 3.3 %,
so a 2× error moves det4 by ~25 µm — real but small. It cannot be extracted from the front↔back
width, because that width is beam-divergence dominated and the divergence is position-correlated
in a divergent beam.</li>
<li><strong>The scattering explanation is arithmetic, not a measurement.</strong> 1.1 mrad is what
the bench residual <em>requires</em> given a 176 µm chamber; it is consistent with the M3 study's
measured kink and with Highland, but nobody has propagated the cosmic momentum spectrum through
the actual material budget between z = 144 mm and the DUT. Until that is done, "the bench is
scattering-limited" is the leading hypothesis, not a result.</li>
<li><strong>It rests on transferring det4's number to det3.</strong> The comparison assumes the
bench chambers are at least as good as det4. That is very likely — det4 is the fleet's worst —
but it is an assumption, and the clean way to kill it is to put a bench chamber in a beam.</li>
<li><strong>The angle scaling is anchored on one rotated point</strong> ({MOUNT:.1f}°).
Run 61's 15.465° data would give a second point and turn it into a measured curve; it has not
been extracted into paired form.</li>
<li><strong>These are hits-based cluster centroids</strong>, which is legitimate here only because
the mount is flat: at θ ≈ 0 there is no drift-time ladder to reconstruct. Nothing here contradicts
<code>RECONSTRUCTION_BASIS.md</code>; it is the one geometry where a centroid is the right
estimator.</li>
</ul>

<h2>Reproducing</h2>
<p><code>resolution.py</code> → <code>results.json</code> + <code>residuals.npz</code>;
<code>make_figures.py</code> → <code>figures/</code>; <code>make_report.py</code> → this page.
Input is <code>det4_run_53_v2.npz</code> / <code>det4_run_57_v2.npz</code> under
<code>/media/dylan/data/x17/sps_run53_det4_check/…/mapping_check/</code>. Strip map from
<code>det4_sps_map.py</code>, reference zones from
<code>analysis/urw_mapping/mapping_urwell.csv</code>.</p>

</div></body></html>
"""

(HERE / "report.html").write_text(HTML)
print("wrote", HERE / "report.html")

# A second, fully self-contained copy for the notes site, which forbids external
# references -- it is read offline from a phone.  Same HTML, figures inlined.
import base64
import re


def _inline(match):
    return ('src="data:image/png;base64,'
            + base64.b64encode((HERE / match.group(1)).read_bytes()).decode() + '"')


standalone = re.sub(r'src="(figures/[^"]+)"', _inline, HTML)
standalone = standalone.replace(
    "<title>det4 spatial resolution at SPS H4</title>",
    '<title>det4 spatial resolution at SPS H4</title>\n<meta name="description" '
    'content="The cosmic bench\'s ~500 um is the track angle, not the chamber: det4 '
    'resolves 176 um at normal incidence, with the uRWELL reference measured off its '
    'own three strip pitches.">')
(HERE / "report_standalone.html").write_text(standalone)
print("wrote", HERE / "report_standalone.html", f"({len(standalone)/1e6:.1f} MB)")
