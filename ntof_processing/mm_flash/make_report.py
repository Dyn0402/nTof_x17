#!/usr/bin/env python3
"""Build report.html from results.json. Re-run after analyse.py."""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
R = json.load(open(HERE / 'results.json'))
runs, cond, hv = R['runs'], R['conditions'], R['hv']
integ, cal = R['integrity'], R['calibration']
R9 = json.load(open(HERE / 'results_709.json'))
RC = json.load(open(HERE / 'results_chain.json'))
RCM = json.load(open(HERE / 'results_compare.json'))
RI = json.load(open(HERE / 'results_imon.json'))
RB = json.load(open(HERE / 'results_board.json'))
RBW = RB['working_point']
_imp = [r['q_nC']*1e3/r['wf_mix_pC'] for r in RI['plateaus'] if 'wf_mix_pC' in r]
_q = [r['q_nC'] for r in RI['plateaus'] if 'wf_mix_pC' in r]
IMP_MEAN = sum(_imp)/len(_imp)
IMP_SD = (sum((x-IMP_MEAN)**2 for x in _imp)/(len(_imp)-1))**0.5
IMP_N = len(_imp)
QSPAN = max(_q)/min(_q)


def cable_table():
    out = []
    for r in RC['cable']['table']:
        out.append(f"<tr><th>{r['f_MHz']:g}</th>"
                   f"<td>{r['dB_10m']:.2f}</td><td>{100*(1-r['amp_10m']):.1f} %</td>"
                   f"<td>{r['dB_20m']:.2f}</td><td>{100*(1-r['amp_20m']):.1f} %</td></tr>")
    return '\n'.join(out)


def rms_table():
    out = []
    for r in sorted(RC['charge_rms'], key=lambda x: -x['resist']):
        if r['drift'] != 700 or r['cls'] != 'dedicated':
            continue
        if r['resist'] not in (570, 565, 555, 545, 535, 525, 515, 505):
            continue
        res = r['frac_rms_residual']
        out.append(f"<tr><th>{r['resist']}</th><td>{r['n']}</td>"
                   f"<td>{r['mean_pC']:,.0f}</td><td>{r['rms_pC']:,.0f}</td>"
                   f"<td>{100*r['frac_rms']:.1f} %</td>"
                   f"<td>{100*r['frac_rms_beam']:.1f} %</td>"
                   f"<td>{100*r['frac_rms_noise']:.1f} %</td>"
                   f"<td>{100*res:.1f} %</td></tr>" if res else '')
    return '\n'.join(x for x in out if x)


def scan_table():
    """224709 detector-A scan, one row per (drift, amplification) point."""
    pts = {}
    for r in R9['scan']:
        pts.setdefault((r['drift'], r['resist']), {})[r['cls']] = r
    out = []
    for (drift, resist), v in sorted(pts.items(), key=lambda x: (-x[0][0], -x[0][1])):
        ded, par = v.get('dedicated'), v.get('parasitic')
        ratio = (f"{ded['charge_pC'] / par['charge_pC']:.2f}"
                 if ded and par and par['charge_pC'] > 0 else '&mdash;')
        out.append(
            f"<tr><th>{drift}</th><td>{resist}</td>"
            f"<td>{ded['n'] if ded else 0}</td>"
            f"<td>{ded['charge_pC']:,.0f}</td><td>{ded['peak_mV']:.1f}</td>"
            f"<td>{par['charge_pC']:,.0f}</td><td>{par['peak_mV']:.1f}</td>"
            f"<td>{ratio}</td></tr>" if ded and par else
            f"<tr><th>{drift}</th><td>{resist}</td><td colspan=6>incomplete</td></tr>")
    return '\n'.join(out)


def gain_table():
    out = []
    for key, v in sorted(R9['gain_fits'].items(), key=lambda x: (-x[1]['drift'], x[1]['cls'])):
        out.append(f"<tr><th>{v['drift']}</th><td>{v['cls']}</td><td>{v['n_points']}</td>"
                   f"<td>{v['e_fold_V']:.1f}</td><td>&times;{v['gain_per_10V']:.2f}</td></tr>")
    return '\n'.join(out)


def _ratio(drift, resist):
    """Dedicated/parasitic charge ratio at one scan point."""
    q = {r['cls']: r['charge_pC'] for r in R9['scan']
         if r['drift'] == drift and r['resist'] == resist}
    return q['dedicated'] / q['parasitic']


def recovery_table():
    out = []
    for r in R9['scan']:
        if r['cls'] != 'dedicated' or not r.get('recovery'):
            continue
        rc = r['recovery']
        if r['drift'] != 700:
            continue
        out.append(f"<tr><th>{r['resist']}</th><td>{rc['peak_mV']:.1f}</td>"
                   f"<td>{rc['4mV']:,.0f}</td><td>{rc['n']}</td></tr>")
    return '\n'.join(out)


def f(x, n=1):
    return f'{x:,.{n}f}'


def rate_row(run):
    return runs[run]['zs_rate_per_bunch_per_ms']


def conditions_table():
    rows = []
    for run in ('224302', '224325', '224327'):
        c = cond[run]
        i = runs[run]
        rows.append(f"""<tr><th>{run}</th><td>{c['span']}</td><td>{c['beam']}</td>
        <td>{c['dream']}</td><td>{c['gas']}</td><td>{i['live']} @ {i['zs_threshold_mV']} mV</td>
        <td>{c['hv']}</td></tr>""")
    return '\n'.join(rows)


def flash_table():
    rows = []
    for run in ('224302', '224325', '224327'):
        i = runs[run]
        rec = i['recovery_to_4mV_ns']['p50']
        pkt = i['peak_time_ns']
        delta = '&mdash;' if rec != rec else f"{(rec - pkt)/1000:.2f}"
        rows.append(f"""<tr><th>{run}</th><td>{i['live']}</td>
        <td>{f(i['peak_mV']['p50'])}</td><td>{f(i['peak_mV']['p99'])}</td>
        <td>{f(i['peak_mV']['max'])}</td>
        <td>{i['railed_bunches']} / {i['n_bunch']}</td>
        <td>{f(i['charge_pC']['p50'], 0)}</td><td>{pkt/1000:.2f}</td><td>{delta}</td>
        <td>{f(i['zs_first_block_ns']/1000, 2)}</td></tr>""")
    return '\n'.join(rows)


def rate_table():
    body = []
    for b in range(len(rate_row('224302'))):
        lo = rate_row('224302')[b]['t_lo_ms']
        hi = rate_row('224302')[b]['t_hi_ms']
        cells = ''.join(
            f"<td>{rate_row(r)[b]['rate']:.2f}</td>" for r in ('224302', '224325', '224327'))
        body.append(f'<tr><th>{lo:g} &ndash; {hi:g}</th>{cells}</tr>')
    return '\n'.join(body)


def hv_table(run):
    rows = []
    for row in sorted(hv[run], key=lambda r: (r['drift_V'], r['resist_V'], r['cls'])):
        rows.append(f"""<tr><td>{row['cls']}</td><td>{row['drift_V']:.0f}</td>
        <td>{row['resist_V']:.0f}</td><td>{row['n']}</td>
        <td>{row['charge_pC']:,.0f}</td><td>{row['peak_mV']:.1f}</td></tr>""")
    return '\n'.join(rows)


def lin_table():
    rows = []
    for run in ('224302', '224327'):
        for p in R['linearity'][run]:
            rows.append(f"<tr><th>{run}</th><td>{p['parasitic_charge_pC']:,.0f}</td>"
                        f"<td>{p['ratio']:.2f}</td></tr>")
    return '\n'.join(rows)


TITLE = 'How much charge does the gamma flash deliver? Two independent measurements'
SUMMARY = ('The flash puts ~1000x the DREAM front end full-scale charge onto a single '
           'micromegas strip, measured on a 1 GS/s readout with no preamplifier and '
           'cross-checked against the HV supply current; with the real board accounting '
           '(checkerboard pad combs, 85 % image capture, exact 50/50 X/Y) the two '
           'absolute scales agree once the flash is ~4x denser at this strip than the '
           'chamber average -- which the strip\'s own intensity compression '
           'independently confirms. Gain cannot fix it and the millisecond dead time '
           'follows from it.')

STYLE = """
:root {
  --bg:#ffffff; --fg:#1a1a1a; --muted:#5a5a5a; --line:#e0e0e0;
  --accent:#2f6f9f; --warn:#c0632c; --panel:#f7f7f8; --ok:#2e7d4f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#15171a; --fg:#e8e8ea; --muted:#a0a4ab; --line:#2c3036;
    --accent:#6fb0e0; --warn:#e2925c; --panel:#1c1f24; --ok:#6fc493;
  }
}
:root[data-theme="dark"] {
  --bg:#15171a; --fg:#e8e8ea; --muted:#a0a4ab; --line:#2c3036;
  --accent:#6fb0e0; --warn:#e2925c; --panel:#1c1f24; --ok:#6fc493;
}
body { background:var(--bg); color:var(--fg); margin:0 auto; padding:2.2rem 1.2rem 4rem;
  max-width:53rem; line-height:1.62;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
h1 { font-size:1.72rem; line-height:1.25; margin:0 0 .3rem; }
h2 { font-size:1.2rem; margin:2.4rem 0 .7rem; padding-bottom:.3rem;
  border-bottom:1px solid var(--line); }
h3 { font-size:1.02rem; margin:1.6rem 0 .4rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:.04em; }
h4 { font-size:.95rem; margin:1.2rem 0 .3rem; color:var(--fg); font-weight:640; }
.appendix { border-top:3px solid var(--line); margin-top:3rem; padding-top:.5rem; }
.sub { color:var(--muted); margin:0 0 1.6rem; font-size:.94rem; }
.verdict { background:var(--panel); border-left:4px solid var(--accent);
  padding:1rem 1.15rem; border-radius:0 6px 6px 0; margin:1.4rem 0; }
.verdict p { margin:.4rem 0; }
.kpis { display:flex; flex-wrap:wrap; gap:.8rem; margin:1.4rem 0; }
.kpi { flex:1 1 10rem; background:var(--panel); border:1px solid var(--line);
  border-radius:7px; padding:.75rem .9rem; }
.kpi .n { font-size:1.35rem; font-weight:640; color:var(--accent);
  font-variant-numeric:tabular-nums; }
.kpi .l { font-size:.79rem; color:var(--muted); line-height:1.35; }
.scroll { overflow-x:auto; margin:1rem 0; }
table { border-collapse:collapse; width:100%; font-size:.87rem;
  font-variant-numeric:tabular-nums; }
th,td { border-bottom:1px solid var(--line); padding:.42rem .6rem; text-align:right; }
th { text-align:left; font-weight:600; }
thead th { color:var(--muted); font-weight:600; font-size:.8rem;
  text-transform:uppercase; letter-spacing:.03em; }
figure { margin:1.6rem 0; }
img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px;
  background:#fff; }
figcaption { font-size:.85rem; color:var(--muted); margin-top:.45rem; }
code { background:var(--panel); padding:.1rem .32rem; border-radius:3px;
  font-size:.85em; }
.caveat { border:1px solid var(--line); border-left:4px solid var(--warn);
  background:var(--panel); padding:1rem 1.15rem; border-radius:0 6px 6px 0; }
ul { padding-left:1.15rem; }
li { margin:.3rem 0; }
.eq { background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:.8rem 1rem; margin:1rem 0; overflow-x:auto;
  font-family:"Latin Modern Math",Georgia,"Times New Roman",serif;
  font-size:1.06rem; text-align:center; line-height:2.1; }
.eq .n { display:inline-block; text-align:center; vertical-align:middle;
  font-size:.9em; }
.eq .n b { display:block; border-bottom:1px solid currentColor; padding:0 .3em .1em; }
.eq .n s { display:block; text-decoration:none; padding:.1em .3em 0; }
.eq .where { display:block; margin-top:.5rem; font-size:.8rem; color:var(--muted);
  font-family:inherit; text-align:center; line-height:1.5; }
var { font-style:italic; }
"""

BODY = f"""<h1>Micromegas in the n_TOF DAQ: gamma-flash response, recovery and charge</h1>
<p class="sub">Dylan Neff &middot; 11 August 2026 &middot; n_TOF EAR2 run 224709
(9 August, detector-A HV scan) &middot; July runs 224302 / 224325 / 224327 in Appendix A</p>

<div class="verdict">
<p><strong>The gamma flash puts about a thousand times the DREAM front end's
full-scale input charge onto a single strip.</strong> Measured directly, on
{R9['channel'].split('=')[1].strip()}, digitised at 1 GS/s with no charge-sensitive preamplifier anywhere in
the chain: <strong>{RCM['absolute']['strip_pC_dedicated']:.0f} pC</strong> on one strip per dedicated proton pulse at the
production operating point, against a DREAM CSA full scale of 600 fC &mdash; a factor
<strong>{RCM['dream']['strip_over_full_scale']:,.0f}</strong>. That is the answer to how much charge we can swallow
before we die: not this much, and not by a small margin.</p>

<p><strong>Two independent instruments agree on the shape to a few per cent.</strong>
The same charge measured through the resistive layer's HV supply current &mdash; a
different electrode, a different instrument, nine orders of magnitude less bandwidth
&mdash; was run over the <em>same 25 working points</em>. Point by point the two differ
by a constant: the chamber charge divided by the single-strip charge is
<strong>{IMP_MEAN:.0f} &plusmn; {IMP_SD:.0f}</strong> strips-equivalent across the whole
scan, a spread of {100*IMP_SD/IMP_MEAN:.1f} % over a factor {QSPAN:.0f} in charge. Both
also see the response compress with intensity, by the same few per cent per proton.</p>

<p><strong>The absolute scales close too, once the board is read from its gerbers
rather than assumed.</strong> The readout is not two planes of solid strips: it is a
512&thinsp;&times;&thinsp;512 pad grid, checkerboard-bussed, so a uniform flash splits
<em>exactly</em> 50/50 between the X and Y views and 85 % of the resistive-layer charge
images onto the pad plane. That accounting turns the HV-current charge into a uniform
expectation of <strong>{RBW['expected_uniform_pC']:.0f} pC per strip</strong> &mdash; and the measured strip
carries <strong>{RB['residual_constancy']['mean']:.1f} &plusmn; {RB['residual_constancy']['sd']:.1f}&times;</strong> that, one constant across all
{RB['residual_constancy']['n']} working points. The strip's own <em>intensity compression</em>, read through the
board's sheet capacitance, independently measures the local flash density at
<strong>{RBW['enhancement']['compression']:.1f}&times;</strong> the chamber average &mdash; so the factor is real, local, and
physical: the delivered charge density at this strip is three-to-four times the chamber
average. Whether that is the flash illumination profile or a local gain enhancement near
the board edge, one channel cannot say &mdash; but the two instruments agree wherever
they measure the same thing.</p>

<p><strong>Gain cannot rescue it, and the dead time follows.</strong> Fitting the flash
into 600 fC would need <strong>{RCM['dream']['volts_to_fit_in_range']:.0f} V</strong> less amplification, where a real
track is equally invisible. Draining {RCM['absolute']['strip_pC_dedicated']:.0f} pC through the CSA's 9&ndash;90 nA feedback
takes <strong>{RCM['dream']['drain_time_ms'][1]:.0f}&ndash;{RCM['dream']['drain_time_ms'][0]:.0f} ms</strong> &mdash; which is the
millisecond dead time DREAM shows, derived from an independent charge measurement.</p>

<p>The chamber itself is fine: it is clear within microseconds and taking hits again at
30 &micro;s, the first instant the DAQ allows. That evidence is in
<a href="#appendix-july">Appendix D</a>.</p>
</div>

<div class="kpis">
  <div class="kpi"><div class="n">{RCM['absolute']['strip_pC_dedicated']:.0f} pC</div>
    <div class="l">flash charge on one strip, dedicated pulse, production point</div></div>
  <div class="kpi"><div class="n">{RCM['dream']['strip_over_full_scale']:,.0f}&times;</div>
    <div class="l">that charge, over the DREAM CSA's 600 fC full scale</div></div>
  <div class="kpi"><div class="n">{RB['residual_constancy']['mean']:.1f}&times;</div>
    <div class="l">local flash density at this strip over the chamber average
    (the compression independently reads {RBW['enhancement']['compression']:.1f}&times;)</div></div>
  <div class="kpi"><div class="n">{R9['gain_fits']['700_dedicated']['e_fold_V']:.0f} V</div>
    <div class="l">gain e-folding on detector A, 25 scan points</div></div>
  <div class="kpi"><div class="n">{RCM['dream']['drain_time_ms'][1]:.0f}&ndash;{RCM['dream']['drain_time_ms'][0]:.0f} ms</div>
    <div class="l">to drain it at the CSA feedback limit &mdash; the DREAM dead time</div></div>
  <div class="kpi"><div class="n">0</div>
    <div class="l">samples at either ADC rail; nothing clips, nothing wraps</div></div>
</div>

<!--JULY-START-->
<h2 id="appendix-a">Appendix A &mdash; July 2026: the flash is not a detector dead time</h2>

<p>This is the earlier half of the study, kept in full. Three runs from 5&ndash;9 July
carried a micromegas channel whose cabling was never recorded, so nothing here can be
attached to a named chamber &mdash; but they answer a different question from the scan
above, and they answer it well: <strong>the chamber is not paralysed by the flash.</strong>
The signal returns below a 4 mV threshold 0.87 &micro;s after its own peak, hits resume at
30.3 &micro;s (the first instant the DAQ allows one) at the highest rate of the whole
20 ms cycle, and a genuine beam-off run shows the acquisition itself is flat over four
decades of time. The millisecond dead time is a DREAM-chain property.</p>

<h3>What was measured</h3>
<p>Three n_TOF runs carry a micromegas channel in the n_TOF DAQ itself &mdash; separate
from the DREAM readout, and the only such runs whose raw waveforms survive. The channel
is a single analog input digitised at 1 GS/s over the full 20 ms neutron cycle: a
mandatory un-suppressed 30 &micro;s block covering the flash, then zero-suppressed blocks
for the rest of the cycle. Two inputs exist, <code>MMA</code> and <code>MMB</code>; which
one is live changes between runs, so each run is analysed on its own live channel.</p>

<div class="scroll"><table>
<thead><tr><th>run</th><th>span</th><th>beam</th><th>DREAM</th><th>gas</th>
<th>live channel</th><th>HV during overlap</th></tr></thead>
<tbody>{conditions_table()}</tbody>
</table></div>

<p><strong>224327 is the key run</strong>: it sits entirely inside DREAM run_18, whose
trigger was &ldquo;gamma flash on each 1.2 s cycle + random trigger within 30 ms after the
flash&rdquo; &mdash; precisely the DREAM configuration in which the millisecond saturation
is seen. <strong>224325 is a genuine beam-off control</strong>: PulseIntensity is zero for
all 5277 bunches and the beam pickup is flat, so it measures what the acquisition does
with no flash at all.</p>

<h3>The flash and the recovery</h3>

<figure>
<img src="figures/flash_waveform.png" alt="Bunch-averaged flash waveform and its tail">
<figcaption>Left: bunch-averaged signal over the 30 &micro;s un-suppressed block. The
flash arrives at ~12 &micro;s. Right: the same traces zoomed on the tail, against the 4 mV
zero-suppression threshold. 224302 falls below threshold within ~1 &micro;s and settles to
a &minus;1.3 mV undershoot that decays to &minus;0.3 mV by 30 &micro;s. 224327 shows a
damped ringing on the MMA input &mdash; a channel/cabling artefact, not detector current
&mdash; which keeps that run above threshold for a few &micro;s longer. The beam-off run
is flat.</figcaption>
</figure>

<div class="scroll"><table>
<thead><tr><th>run</th><th>chan</th><th>peak p50<br>(mV)</th><th>p99<br>(mV)</th>
<th>max<br>(mV)</th><th>bunches with a<br>railed sample</th><th>charge p50<br>(pC)</th>
<th>flash peak at<br>(&micro;s)</th><th>below 4 mV<br>after peak (&micro;s)</th>
<th>first ZS hit<br>(&micro;s)</th></tr></thead>
<tbody>{flash_table()}</tbody>
</table></div>

<p>The largest excursion anywhere is {f(runs['224327']['peak_mV']['max'])} mV against
{f(cal['headroom_mV'])} mV of available headroom (baseline parked at +200 mV, negative rail
at &minus;252 mV). Five bunches out of 6511 across the two beam runs contain a railed
sample &mdash; 0.08 %. <strong>The flash is measured, not clipped.</strong></p>

<h3>The channel never stops recording</h3>

<figure>
<img src="figures/post_flash_rate.png" alt="Zero-suppressed block rate versus time after the proton pulse">
<figcaption>Zero-suppressed blocks per bunch per millisecond, against time after the proton
pulse, over four decades. Both beam runs are at their maximum rate in the first bin after
the mandatory block ends, and fall away as the neutron flux does. The beam-off control is
flat to within 17 % across the whole window, which is what proves the acquisition itself
has no time structure. Note the two inputs run different zero-suppression thresholds, so
compare shapes rather than absolute levels.</figcaption>
</figure>

<div class="scroll"><table>
<thead><tr><th>time after pulse (ms)</th><th>224302 (beam, 4 mV)</th>
<th>224325 (beam off, 0.01 mV)</th><th>224327 (beam, 0.01 mV)</th></tr></thead>
<tbody>{rate_table()}</tbody>
</table></div>

<p>This is the core of the argument. If the chamber were paralysed for milliseconds there
would be a gap &mdash; a suppressed rate at early times, recovering later. The opposite is
observed: the rate is highest immediately after the flash and decays monotonically, exactly
as the neutron time-of-flight spectrum requires, while the beam-off control confirms the
DAQ's own efficiency is flat over the same span.</p>

<figure>
<img src="figures/single_traces.png" alt="Individual bunch traces">
<figcaption>Individual bunches, dedicated (red) and parasitic (blue), for the two beam
runs. Single traces, not averages: the flash is a clean pulse returning to baseline, and
its size tracks the proton intensity far less than proportionally.</figcaption>
</figure>

<h3>Flash charge, and how it responds to HV and beam intensity</h3>

<p>Charge is the integral of the positive lobe (11&ndash;20 &micro;s) converted at
{cal['fC_per_count_ns']['MMB']:.3f} fC per count&middot;ns, i.e. assuming the digitiser
input is a direct 50 &Omega; termination. The 20&ndash;30 &micro;s window carries
&minus;12 % (224302) and &minus;8 % (224327) of the lobe, so the window captures the pulse
and the AC-coupling return is slow.</p>

<figure>
<img src="figures/charge_vs_hv.png" alt="Flash charge versus amplification voltage">
<figcaption>Median flash charge against the amplification (resistive) voltage of the common
HV ladder, split by pulse type. 224327 is exponential over the scanned range, e-folding
every ~10.5 V. 224302 turns over: the charge stops growing above ~537 V and falls at 540 V,
while the peak amplitude is still only 73 % of the available headroom &mdash; so this is
not an ADC limit.</figcaption>
</figure>

<h4>224302 &mdash; drift 800 V fixed</h4>
<div class="scroll"><table>
<thead><tr><th>pulse</th><th>drift (V)</th><th>amplification (V)</th><th>n</th>
<th>charge (pC)</th><th>peak (mV)</th></tr></thead>
<tbody>{hv_table('224302')}</tbody></table></div>

<h4>224327 &mdash; drift 600 and 800 V</h4>
<div class="scroll"><table>
<thead><tr><th>pulse</th><th>drift (V)</th><th>amplification (V)</th><th>n</th>
<th>charge (pC)</th><th>peak (mV)</th></tr></thead>
<tbody>{hv_table('224327')}</tbody></table></div>

<p>At the same amplification voltage (480 V), raising the drift from 600 to 800 V increases
the dedicated-pulse charge from 450 to 614 pC, a 37 % gain.</p>

<figure>
<img src="figures/intensity_linearity.png" alt="Dedicated/parasitic charge ratio versus signal size">
<figcaption>The dedicated/parasitic charge ratio against the size of the signal itself. The
proton-intensity ratio is 2.05 and is fixed; a proportional detector would sit on the dashed
line at every point. Instead the ratio falls monotonically from 2.35 to 0.95 as the signal
grows over a factor of 20 &mdash; the response compresses, and at the top of the 224302
ladder the flash charge no longer depends on beam intensity at all.</figcaption>
</figure>

<div class="scroll"><table>
<thead><tr><th>run</th><th>parasitic charge (pC)</th><th>dedicated / parasitic</th></tr></thead>
<tbody>{lin_table()}</tbody></table></div>

<p>The compression sets in with the size of the signal, not with beam intensity as such:
the two runs, in different gases and on different inputs, fall on one curve when plotted
against charge. That is the signature of a space-charge limit in the amplification region
rather than an electronics limit &mdash; and it is consistent with the flash charge being
the largest signal these chambers ever see.</p>

<!--JULY-END-->
<h2>The measurement: a detector-A scan on one strip</h2>

<p>On 9 August a micromegas input was patched into the n_TOF DAQ for three runs,
224708&ndash;224710. The connection is known:
<strong>{R9['channel']}</strong> &mdash; a single strip, not the mesh, digitised
at 1 GS/s over the full 20 ms neutron cycle with <em>no charge-sensitive
preamplifier anywhere in the chain</em>. That last point is what makes this
measurement possible at all: every other view of the flash we have goes through
the DREAM front end, which the flash saturates. The input range was set to
{R9['calibration']['full_scale_mV']:,.0f} mV full scale
({R9['calibration']['mV_per_count']*1000:.1f} &micro;V per count) with the
baseline parked near zero ({R9['calibration']['baseline_mV']:.0f} mV);
single-sample noise is {R9['noise_sigma_mV']:.2f} mV.</p>

<p><strong>224709</strong> (17:05&ndash;19:38, 344 files, 1.5 TB) contains a
<strong>detector-A-only drift &times; amplification scan</strong>, 17:10&ndash;19:31,
in 25 plateaus of ~5.7 min. B, C and D were held fixed throughout. That matters
twice over: it removes the degeneracy that made the July chamber assignment
impossible, and because the flash charge moves by more than an order of magnitude
while only A's voltages move, <strong>it independently confirms the channel is on
detector A</strong>. Each plateau is analysed after a {R9['n_bunch'] and 45} s
settling cut; {R9['n_dedicated']} dedicated and {R9['n_parasitic']} parasitic
bunches are used, split on the beam-pickup amplitude.</p>

<figure>
<img src="figures/flash_709.png" alt="Flash waveform at several amplification voltages">
<figcaption>Bunch-averaged flash at four amplification voltages, drift 700 V,
dedicated pulses. Left: the pulse itself. Right: the tail against the 4 mV scale.
The shape is the same at every gain; only the amplitude moves.</figcaption>
</figure>

<figure>
<img src="figures/scan_709.png" alt="Detector-A drift and amplification scan">
<figcaption>Left: median flash charge against detector A's amplification voltage,
at each of the three drift settings, for dedicated and parasitic pulses. Right:
the dedicated/parasitic charge ratio across the same scan &mdash; the same
compression seen in July, now on a chamber we can name.</figcaption>
</figure>

<div class="scroll"><table>
<thead><tr><th>drift (V)</th><th>ampl. (V)</th><th>n ded.</th>
<th>ded. charge (pC)</th><th>ded. peak (mV)</th>
<th>par. charge (pC)</th><th>par. peak (mV)</th><th>ded./par.</th></tr></thead>
<tbody>{scan_table()}</tbody></table></div>

<h3>Gain slope</h3>
<div class="scroll"><table>
<thead><tr><th>drift (V)</th><th>pulse</th><th>points</th>
<th>e-folding (V)</th><th>gain per 10 V</th></tr></thead>
<tbody>{gain_table()}</tbody></table></div>

<h3>The drift field does not matter here</h3>
<p>Compared at the same amplification voltage, the three drift settings agree to
within {R9['drift_dependence']['max_spread_pct']:.1f} % (typically 2&ndash;8 %),
with no systematic ordering &mdash; the three curves in the figure lie on top of
each other over a sixteen-fold range of charge. Read this as a bound rather than
a null measurement: the drift branches were taken one after another, and the
570 V point, revisited three times over an hour, fell
{100*(R9['repeat_point'][0]['charge_pC'] - R9['repeat_point'][-1]['charge_pC'])/R9['repeat_point'][0]['charge_pC']:.1f} %
across those visits. The slow drift in time and the apparent drift-field spread
are the same size, so they cannot be separated in this run. What can be said is
that <strong>there is no strong drift-field dependence between 500 and 700 V</strong>
&mdash; in contrast to July's 224327, where 600&nbsp;&rarr;&nbsp;800 V gave +37 %
on a different chamber and a different gas.</p>

<div class="scroll"><table>
<thead><tr><th>amplification (V)</th><th>drift 500</th><th>drift 600</th><th>drift 700</th>
<th>spread</th></tr></thead>
<tbody>
{chr(10).join(
    f"<tr><th>{x['resist']}</th>"
    + ''.join(f"<td>{x['charges'].get(str(d), float('nan')):,.0f}</td>" for d in (500, 600, 700))
    + f"<td>{x['spread_pct']:.1f} %</td></tr>"
    for x in R9['drift_dependence']['rows'] if x['cls'] == 'dedicated')}
</tbody></table></div>
<p style="font-size:.85rem;color:var(--muted)">Dedicated pulses, median flash charge in pC.
The same point revisited at
{', '.join(f"{p['at']} ({p['charge_pC']:,.0f} pC)" for p in R9['repeat_point'])}.</p>

<h3>Recovery against pulse size, drift 700 V</h3>
<p>Measured on the bunch-averaged trace, so noise is not the limit: the time from
the flash peak until the mean signal is back below 4 mV. It grows with the pulse,
as it must &mdash; a bigger pulse takes longer to decay through a fixed level
&mdash; and stays in the microseconds across a twenty-fold range of gain.</p>
<div class="scroll"><table>
<thead><tr><th>amplification (V)</th><th>mean peak (mV)</th>
<th>back below 4 mV after (ns)</th><th>n</th></tr></thead>
<tbody>{recovery_table()}</tbody></table></div>


<p>Per bunch, the signal is back under max(4 mV, 5&sigma;) by
<strong>{(R9['recovery_ns']['p50'] - R9['flash_peak_time_ns'])/1000:.2f} &micro;s</strong>
after the peak (median), and the first zero-suppressed block appears at
{R9['zs_first_block_ns']/1000:.1f} &micro;s. No bunch in the run contains a railed
sample. Note this run is far less sensitive to small post-flash hits than the July
ones &mdash; the same zero-suppression threshold in counts is ten times more
millivolts &mdash; so its post-flash hit rate
({R9['zs_blocks_per_bunch']:.2f} blocks per bunch) is not comparable with July's
and is not used for the liveness argument.</p>

<h2>The charge on one strip</h2>

<p>Everything here is quoted as a charge. The conversion is
<var>Q</var> = (&Delta;<var>t</var>/<var>R</var>) &sum; &Delta;<var>V</var>, which for
this channel comes to <strong>{RC['conversion']['fC_per_count_ns']:.4f} fC per
count&middot;ns</strong> into 50 &Omega;; the arithmetic is in
<a href="#appendix-charge">Appendix A</a> and the 10&ndash;20 m of BNC costs under 1 %
of it, which is shown in <a href="#appendix-cable">Appendix B</a>.</p>

<p>All of it refers to <strong>one strip</strong>: MMA is strip 32 of detector A's Y
plane, one of {RC['geometry']['n_strips']} strips at
{RC['geometry']['pitch_mm']:.3f} mm pitch, in a chamber that has the same again on the
X plane. The gamma flash illuminates the whole chamber; this is what landed on a
{RC['geometry']['pitch_mm']:.2f} mm slice of it, and scaling that back up to the
chamber is what the comparison below is about.</p>

<figure>
<img src="figures/chain_diagram.png" alt="The measurement chain from strip to charge">
<figcaption>The chain. The strip is a current source; the cable carries that current
to a 50 &Omega; termination at the digitiser, which turns it into the voltage that
is sampled.</figcaption>
</figure>

<!--CHARGE-DERIV-START-->
<h3>1. Codes to volts</h3>
<p>The digitiser is a 16-bit S014 at 1 GS/s. Samples are signed
<code>int16</code>. For bunch <var>k</var> the baseline <var>b</var> is the median
of the first 2 &micro;s, taken per bunch so that slow baseline movement does not
leak into the integral, and the signal is negative-going, so the deviation is</p>

<div class="eq">
&Delta;<var>V</var><sub>i</sub> = ( <var>b</var> &minus; <var>c</var><sub>i</sub> ) &middot;
<span class="n"><b><var>V</var><sub>FS</sub></b><s>2<sup>16</sup></s></span>
= ( <var>b</var> &minus; <var>c</var><sub>i</sub> ) &times; {RC['conversion']['lsb_uV']:.2f} &micro;V
<span class="where"><var>V</var><sub>FS</sub> = {RC['conversion']['full_scale_mV']:,.2f} mV
(the channel's own full scale, from the run's MODH record),
2<sup>16</sup> = {RC['conversion']['n_codes']:,} codes</span>
</div>

<h3>2. Volts to charge</h3>
<p>The digitiser input is a 50 &Omega; termination, so the sampled voltage is the
strip current through that resistor, <var>v</var>(<var>t</var>) = <var>R</var>
<var>i</var>(<var>t</var>). The charge is the time integral of the current, which
at a fixed sampling interval is just a sum:</p>

<div class="eq">
<var>Q</var> = &int; <var>i</var>(<var>t</var>) d<var>t</var> =
<span class="n"><b>1</b><s><var>R</var></s></span>
&int; <var>v</var>(<var>t</var>) d<var>t</var> &nbsp;&asymp;&nbsp;
<span class="n"><b>&Delta;<var>t</var></b><s><var>R</var></s></span>
&sum;<sub>i</sub> &Delta;<var>V</var><sub>i</sub>
</div>

<p>Collecting the constants gives the number used throughout:</p>

<div class="eq">
<span class="n"><b>&Delta;<var>t</var> &middot; <var>V</var><sub>FS</sub></b>
<s><var>R</var> &middot; 2<sup>16</sup></s></span>
= <strong>{RC['conversion']['fC_per_count_ns']:.4f} fC</strong> per count&middot;ns
<span class="where">&Delta;<var>t</var> = 1 ns, <var>R</var> = 50 &Omega;.
The July runs, on a ten-times finer range, give 0.1537 fC per count&middot;ns.</span>
</div>

<!--CHARGE-DERIV-END-->
<p>The integration runs over
{RC['conversion']['window_ns'][0]/1000:.0f}&ndash;{RC['conversion']['window_ns'][1]/1000:.0f} &micro;s,
which covers the whole positive lobe: the 20&ndash;30 &micro;s window that follows
carries &minus;8 % of it in July and &minus;12 % in the other July run &mdash; the
slow return of the AC coupling, not signal. The running integral below flattens
well before the window closes, which is the check that nothing is being cut off.</p>

<figure>
<img src="figures/charge_integration.png" alt="Pulse and its running integral">
<figcaption>Detector A at 700 / 540 V, dedicated pulses. Blue: the bunch-averaged
&Delta;<var>V</var>(<var>t</var>). Orange: the running integral in pC. It rises
through the flash and then sits flat &mdash; the window is not truncating the pulse,
and the small droop afterwards is the coupling return.</figcaption>
</figure>

<!--CABLE-START-->
<h3>3. What the cable does &mdash; and why the charge survives it</h3>

<p>The signal travels 10&ndash;20 m of RG-58 BNC from the detector, out of the area
through a patch panel, to the rack room. Coaxial loss is dominated by the skin
effect and rises with frequency:</p>

<div class="eq">
&alpha;(<var>f</var>) = <var>A</var>&radic;<var>f</var> + <var>B</var><var>f</var>
&nbsp;&nbsp;[dB/m],&nbsp;&nbsp;
<var>H</var>(<var>f</var>) = 10<sup>&minus;&alpha;(<var>f</var>)<var>L</var>/20</sup>
<span class="where"><var>A</var> = {RC['cable']['alpha_fit'].split('=')[1].split('*')[0].strip()},
<var>B</var> = 1.64&times;10<sup>&minus;4</sup>, <var>f</var> in MHz &mdash; fitted to the
RG-58C/U catalogue values (1.6 / 5.3 / 12.5 / 18.4 dB per 100 m at 1 / 10 / 50 / 100 MHz)</span>
</div>

<div class="scroll"><table>
<thead><tr><th>frequency (MHz)</th><th>10 m (dB)</th><th>10 m amplitude loss</th>
<th>20 m (dB)</th><th>20 m amplitude loss</th></tr></thead>
<tbody>{cable_table()}</tbody></table></div>

<p><strong>The charge is almost immune to this, and the peak is not.</strong> The
charge is the zero-frequency component of the current,
<var>Q</var> = &int;<var>i</var> d<var>t</var> =
<var>&icirc;</var>(<var>f</var>&nbsp;=&nbsp;0), and the skin-effect term vanishes as
<var>f</var> &rarr; 0, so <var>H</var>(0) = 1: attenuation redistributes the pulse in
time without changing its area. What does survive at DC is the ohmic series
resistance of the line, which for a matched cable costs</p>

<div class="eq">
<var>Q</var><sub>strip</sub> / <var>Q</var><sub>measured</sub> =
exp( <var>R'</var><var>L</var> / 2<var>Z</var><sub>0</sub> )
<span class="where"><var>R'</var> = loop resistance per metre (0.046 &Omega;/m for a
copper-cored RG-58C/U, up to 0.165 &Omega;/m if the centre conductor is copper-clad
steel), <var>Z</var><sub>0</sub> = 50 &Omega;</span>
</div>

<div class="scroll"><table>
<thead><tr><th>cable</th><th>charge lost</th></tr></thead>
<tbody>
{chr(10).join(f"<tr><th>{k}</th><td>{v:.2f} %</td></tr>"
              for k, v in RC['cable']['charge_loss_pct'].items())}
</tbody></table></div>

<p>Both statements can be checked directly on the data rather than argued. Taking
the measured mean pulse, dividing its spectrum by <var>H</var>(<var>f</var>) and
transforming back &mdash; i.e. undoing the cable &mdash; moves the
<strong>peak by +{100*(RC['de_attenuation']['20m']['peak_ratio']-1):.1f} %</strong>
for 20 m (+{100*(RC['de_attenuation']['10m']['peak_ratio']-1):.1f} % for 10 m) and the
<strong>area by {100*(RC['de_attenuation']['20m']['area_ratio']-1):.4f} %</strong>.
The pulse simply has very little content where the cable bites: it is a
microsecond-scale pulse, so its spectrum has fallen by two decades before 10 MHz,
where 20 m of RG-58 costs only 1.1 dB.</p>

<figure>
<img src="figures/cable_attenuation.png" alt="Cable attenuation against the pulse spectrum">
<figcaption>Left: cable attenuation for 10 and 20 m (coloured, left axis) against the
measured pulse's own amplitude spectrum (grey, right axis, log). The pulse is gone
before the cable matters. Right: measured and de-attenuated pulse overlaid &mdash;
the peak lifts by 4 %, the area does not move.</figcaption>
</figure>

<p>So the cable correction applied to refer a charge back to the strip is
<strong>&times;{RC['cable']['charge_correction_applied']:.4f}</strong> for 20 m of
copper-cored RG-58 (&times;1.005 for 10 m), and at most &times;1.033 if the cable is
copper-clad steel. At the 700 / 540 V working point that turns
{[r for r in RC['charge_rms'] if r['drift']==700 and r['resist']==540 and r['cls']=='dedicated'][0]['mean_pC']:,.0f} pC
measured into
<strong>{[r for r in RC['charge_rms'] if r['drift']==700 and r['resist']==540 and r['cls']=='dedicated'][0]['mean_pC'] * RC['cable']['charge_correction_applied']:,.0f} pC</strong>
at the strip. Every other charge in this note is quoted as measured at the DAQ
input; multiply by 1.009 to move it to the strip. That is an order of magnitude
smaller than the systematic from assuming a bare 50 &Omega; termination, which
remains the dominant uncertainty on the absolute scale.</p>

<!--CABLE-END-->
<h3>The spread, bunch to bunch, on one strip</h3>

<p>At a fixed working point the flash charge on strip 32 still varies from bunch to
bunch. Three things contribute, and they can be separated: the proton intensity
jitters within a class (the beam pickup measures it), the electronics contribute a
fixed additive term, and what is left is the detector.</p>

<div class="eq">
&sigma;<sup>2</sup><sub><var>Q</var></sub> / <var>Q</var><sup>2</sup> =
(&sigma;<sub><var>I</var></sub>/<var>I</var>)<sup>2</sup> +
(&sigma;<sub>noise</sub>/<var>Q</var>)<sup>2</sup> +
(&sigma;<sub>det</sub>/<var>Q</var>)<sup>2</sup>
</div>

<p>The electronics term is measured, not modelled: integrating a
{(RC['conversion']['window_ns'][1]-RC['conversion']['window_ns'][0])/1000:.0f} &micro;s
window of baseline <em>before</em> the flash, with the same formula, gives a
noise-equivalent charge of
<strong>{RC['noise_equivalent_charge_pC']['rms']:.1f} pC</strong> RMS. That is
{RC['noise_equivalent_charge_pC']['excess_over_white']:.1f}&times; larger than the
{RC['noise_equivalent_charge_pC']['white_noise_expectation']:.2f} pC expected from
the {RC['noise_equivalent_charge_pC']['sample_sigma_mV']:.2f} mV single-sample noise
if it were white &mdash; the baseline wanders coherently, so it integrates up rather
than averaging down. That is the floor on any single-bunch charge measurement here.</p>

<div class="scroll"><table>
<thead><tr><th>ampl. (V)</th><th>n</th><th>mean (pC)</th><th>RMS (pC)</th>
<th>total</th><th>beam</th><th>electronics</th><th>detector</th></tr></thead>
<tbody>{rms_table()}</tbody></table></div>

<figure>
<img src="figures/charge_rms.png" alt="Charge distribution and its spread across the scan">
<figcaption>Left: the per-bunch charge on strip 32 at 700 / 540 V, dedicated pulses,
with mean and RMS. Right: the fractional spread across the amplification scan. It
rises at low gain only because the fixed electronics term becomes a large fraction of
a shrinking signal.</figcaption>
</figure>

<p>Read the top of the ladder, where the measurement is signal-dominated: at
565&ndash;570 V the charge on a single strip fluctuates by
<strong>{min(100*r['frac_rms'] for r in RC['charge_rms'] if r['drift']==700 and r['cls']=='dedicated' and r['resist'] in (565, 570)):.1f}&ndash;{max(100*r['frac_rms'] for r in RC['charge_rms'] if r['drift']==700 and r['cls']=='dedicated' and r['resist'] in (565, 570)):.1f} %</strong>
bunch to bunch, of which the beam contributes under 1 % and the electronics about
1 %. The remaining <strong>~8.5 %</strong> is the detector and the shower it is
sampling &mdash; a real physical fluctuation in how much flash-induced charge lands
on one 0.78 mm strip, not a measurement limitation.</p>

<h2>A completely independent measurement: the HV supply current</h2>

<p>A single strip on a 1 GS/s digitiser is one way to weigh the flash. There is a
second, which shares nothing with it &mdash; not the electrode, not the instrument,
not the bandwidth &mdash; and which was worked out in parallel
(<code>ntof_july_analysis/flash_charge/</code>). It is worth walking through,
because the comparison below is only as good as one's confidence in both halves.</p>

<!--HVDETAIL-START-->
<h3>The idea</h3>
<p>Every electron avalanche in the amplification gap ends with its ions drifting to
the resistive layer. That charge has to be resupplied by the HV power supply. So the
<strong>average current the supply delivers is the avalanche charge, integrated over
everything the beam pulse does</strong> &mdash; and it is measured outside the
readout chain entirely, by the CAEN crate's own current monitor, which was logged
once a second for every sub-run of the campaign. No new data, no reprocessing.</p>

<div class="eq">
<var>Q</var><sub>pulse</sub> =
<span class="n"><b>&lang;<var>I</var>&rang; &minus; <var>I</var><sub>leak</sub></b>
<s><var>f</var><sub>pulse</sub></s></span>
<span class="where">&lang;<var>I</var>&rang; = mean of the monitor over a sub-run,
<var>I</var><sub>leak</sub> = its median, <var>f</var><sub>pulse</sub> = beam pulses
per second in that sub-run's own time window</span>
</div>

<h3>Why the median is the leakage</h3>
<p>This is the trick that makes it work across an HV scan. The monitor samples at
~1 Hz while beam pulses arrive every ~3.3 s, so <em>most samples sit at the standing
leakage current</em>. The sub-run median is therefore the leakage <strong>at that
exact voltage</strong> &mdash; self-calibrating, which matters because leakage is
itself strongly HV-dependent and one subtracted constant would not do. The mean minus
that median is the beam-induced part.</p>

<h3>Why a 1 Hz monitor can see a 10 ms burst</h3>
<p>The obvious objection is that a once-a-second reading cannot catch a millisecond
burst, so the mean would underestimate by a large and unknown factor. That was tested
and closed by measuring the monitor's impulse response directly, phase-folding the
current against individual beam pulses. The readback is a <strong>~1 s averager, not
a snapshot</strong>: the response to one pulse rises from 0.3 s, peaks at 88 nA around
1.1 s, is back to zero by 2.3 s, and its <em>area</em> equals the charge. Two
timestamp-free checks bound it without using any clock &mdash; 26.6 % of samples sit
above baseline where an instantaneous reader would give 0.3 %, and the largest
single-sample excess implies an averaging window &ge; 0.47 s against a burst of
milliseconds. A smoothing filter conserves its integral, so the sample mean recovers
the time-average current: these are measurements, not lower bounds, and there is no
correction factor.</p>

<p>It passes the tests that matter: <strong>zero on a true beam-off run</strong>
(run_159, 0.000 Hz, even on a channel carrying 2.9 &micro;A of leakage), the
<strong>same charge per pulse at a 10&times; different beam rate</strong> (run_157
against run_158), and four independent estimators &mdash; mean&minus;median,
detrended, isolated-pulse fold area, least-squares deconvolution &mdash; agreeing to
<strong>&plusmn;2.5 %</strong>.</p>

<!--HVDETAIL-END-->
<p><strong>In one line:</strong> the avalanche ions land on the resistive layer, so
the current the HV supply delivers <em>is</em> the avalanche charge. The monitor samples
at ~1 Hz while beam pulses arrive every ~3.4 s, so most samples sit at the standing
leakage &mdash; the <strong>median is the leakage at that exact voltage</strong> and the
mean minus that median is the beam-induced part:</p>

<div class="eq">
<var>Q</var><sub>pulse</sub> =
<span class="n"><b>&lang;<var>I</var>&rang; &minus; <var>I</var><sub>leak</sub></b>
<s><var>f</var><sub>pulse</sub></s></span>
<span class="where">why a 1 Hz monitor can weigh a millisecond burst, and the three
validations, are in <a href="#appendix-hv">Appendix C</a></span>
</div>

<p>The parallel package established the method on July data. Here it is applied
<strong>point for point to the same 25 plateaus</strong> as the waveform scan, on the
same detector, day and gas &mdash; so nothing has to be transported between the two
measurements. The pulse rate is counted from the n_TOF side, using the wall clock each
bunch of 224709 carries, rather than from the beam log.</p>

<figure>
<img src="figures/imon_method.png" alt="The imon estimate on one scan point">
<figcaption>One scan point (700 / 540 V), as the monitor sees it. Each beam pulse
(ticks along the bottom) produces a current excursion; between them the reading falls
back to the leakage. The median picks out that leakage, the mean sits above it by
{[r for r in RI['plateaus'] if r['drift']==700 and r['resist']==540][0]['di_uA']*1000:.0f} nA, and dividing by the counted pulse rate gives
{RI['working_point']['imon_nC']:.0f} nC per pulse. {100*[r for r in RI['plateaus'] if r['drift']==700 and r['resist']==540][0]['frac_elevated']:.0f} % of samples are elevated, which is itself the
signature of a ~1 s averaging window rather than an instantaneous read.</figcaption>
</figure>

<figure>
<img src="figures/imon_timeseries.png" alt="Supply current across the whole scan">
<figcaption>The same current across the whole scan. Every plateau of the amplification
ladder is visible in the current before any analysis is done; the lower panel is the
voltage that produced it.</figcaption>
</figure>

<p>One feature of that trace deserves a comment: <strong>the standing baseline jumps at
every drift-HV move</strong> &mdash; from {RB['leakage']['steps_at_drift_moves'][0]['leak_before_nA']:.0f} to {RB['leakage']['steps_at_drift_moves'][0]['leak_after_nA']:.0f} nA when the drift went
700&thinsp;&rarr;&thinsp;600 V, and by ~0.15&ndash;0.2 &micro;A at each later move &mdash;
then relaxes over tens of minutes (the first branch shows the same relaxation from the
pre-scan ramp, {RB['leakage']['branch_700_decay_nA'][0]:.0f}&thinsp;&rarr;&thinsp;{RB['leakage']['branch_700_decay_nA'][1]:.0f} nA). The direction rules out a resistive path from the
cage divider: the drift voltage went <em>down</em> and the resist baseline went
<em>up</em>. It is a charging/relaxation current from moving the cathode, and it is
benign for the charge estimate by construction: the per-plateau median subtracts it, its
worst within-plateau movement is {RB['leakage']['worst_within_plateau_median_shift_nA']:.0f} nA and roughly linear (a linear baseline drift
cancels in mean&nbsp;&minus;&nbsp;median to first order), and the bootstrap error bars
absorb what remains. It is also, incidentally, why a single global baseline would not
work &mdash; the per-plateau median is doing real work here.</p>

<p>At the shared working point (700 / 540 V) it gives
<strong>{RI['working_point']['imon_nC']:.0f} nC per beam pulse</strong> over the whole chamber &mdash; which
also reproduces the independent morning measurement of the same setpoint from run_158,
{RCM['absolute']['hv_nC_per_pulse']:.0f} &plusmn; {RCM['absolute']['hv_nC_rms']:.0f} nC, to better than 1 %.</p>

<h2>The two measurements compared</h2>

<p>These are very different quantities &mdash; one strip's prompt pulse over a
microsecond against a whole chamber's avalanche charge integrated over the entire
cycle &mdash; so they cannot simply be set equal. They can be compared on three axes,
and all three are worth having because they fail in different ways.</p>

<h3>1. The gain slope &mdash; calibration-free</h3>
<p>Neither the 50 &Omega; assumption nor the monitor's absolute scale enters the
<em>shape</em> of charge against amplification voltage. This is the cleanest test, and
it requires believing nothing about either instrument's calibration:</p>

<div class="scroll"><table>
<thead><tr><th>method</th><th>det</th><th>drift (V)</th><th>points</th>
<th>range (V)</th><th>e-folding (V)</th><th>gain per 10 V</th></tr></thead>
<tbody>
<tr><th>HV supply current</th><td>A</td><td>600</td><td>31</td><td>520&ndash;580</td><td>20.2</td><td>&times;1.64</td></tr>
<tr><th>HV supply current</th><td>B</td><td>800</td><td>31</td><td>520&ndash;580</td><td>20.2</td><td>&times;1.64</td></tr>
<tr><th>HV supply current</th><td>C</td><td>800</td><td>30</td><td>520&ndash;580</td><td>22.4</td><td>&times;1.56</td></tr>
<tr><th>waveform, strip 32 (dedicated)</th><td>A</td><td>700</td><td>15</td><td>500&ndash;570</td><td>24.1</td><td>&times;1.51</td></tr>
<tr><th>waveform, strip 32 (parasitic)</th><td>A</td><td>700</td><td>15</td><td>500&ndash;570</td><td>21.7</td><td>&times;1.59</td></tr>
</tbody></table></div>

<p>The waveform on one strip and the current drawn by the whole chamber give the same
gain slope to <strong>{100*abs(RCM['slope_agreement']['ratio']-1):.0f} %</strong> ({RCM['slope_agreement']['waveform_parasitic_e_fold_V']:.1f} V against
{RCM['slope_agreement']['hv_det_A_e_fold_V']:.1f} V). The dedicated-pulse curve is slightly flatter, at {R9['gain_fits']['700_dedicated']['e_fold_V']:.1f} V,
which is the compression: larger signals lose proportionally more, so the apparent
slope softens. Note the HV scan ran detector A at drift 600 V and B and C at 800 V,
and all three give the same slope &mdash; an independent echo of the
drift-independence found above.</p>

<p>This also settles a discrepancy that was open until this run. The July data gave an
e-folding of ~10.5 V, half the HV-current value, and it was recorded as unexplained.
It was a different gas (Ar/iso 95/5) on a chamber that could not be identified. On
detector A in 90/10, measured properly, the two methods agree.</p>

<figure>
<img src="figures/compare_final.png" alt="Both methods in absolute per-strip charge, and the dynamic-range ladder">
<figcaption>Left: everything on one absolute axis &mdash; flash charge <em>per strip</em>
against amplification voltage, drift 700 V. Blue/orange: the waveform on strip 32,
dedicated and parasitic pulses (dotted grey: their mix). Solid green: the supply-current
charge converted to the share a strip carries under uniform illumination (the board
accounting of the next subsection); it runs parallel, a constant &times;{RB['residual_constancy']['mean']:.1f} below the
mix. Dashed green: the same supply-current charge placed at <em>this</em> strip using the
local density that the strip's own compression measures (the densitometer explained in
the next subsection, &times;{RBW['enhancement']['compression']:.1f}) &mdash; it lands ~40 % under the mix, which is as
close as the two instruments can be brought and is the residual we accept. The red line
is the DREAM CSA full scale (600 fC): at the operating region (shaded) every curve is
two-to-three orders of magnitude above it. Right: the same statement as a ladder at
700 / 540 V.</figcaption>
</figure>

<h3>2. Scaling one strip to the whole chamber &mdash; what the board actually is</h3>

<p>To put the two on the same axis, the single-strip charge has to be scaled to the
chamber. The first version of this note did that with a strip count and an assumption
about X/Y sharing. The board design &mdash; gerbers, stack-up and solved electrostatics,
all in the response-simulation repository (<code>~/CLionProjects/MX17_Geant</code>)
&mdash; replaces both with measured facts, and they change the picture:</p>

<ul>
<li><strong>There are no strips.</strong> The readout is a 512 &times; 512 grid of
680 &micro;m pads on 780 &micro;m pitch, bussed in a checkerboard: a &ldquo;Y
strip&rdquo; is a comb of 256 pads on 1.56 mm pitch along a row (every second pad), and
the intervening pads belong to X combs running the other way. X and Y live in the
<em>same</em> copper plane. For a uniform flash the two views therefore split the image
charge <strong>exactly 50/50, by symmetry</strong> &mdash; the &ldquo;all of it could
appear on the Y plane&rdquo; reading is not an option the hardware offers.</li>
<li><strong>85 % of the charge on the resistive layer images onto the pad plane</strong>
(solved from the real stack: 50 &micro;m coverlay + 19 &micro;m glue against a
150 &micro;m mesh gap, with the inter-pad channels floating); the mesh takes the
rest.</li>
<li><strong>The resistive layer is itself strips</strong> &mdash; 550 &micro;m screen-printed
resistive lines on 800 &micro;m pitch running along y, contacting copper buses at the
two y-ends of the active area and nothing in between. At the frozen
2 M&Omega;/sq that drain takes <strong>&tau; &asymp; {RB['board']['tau_drain_ms']:.0f} ms</strong>: on the 9 &micro;s
integration window the sheet is charge-conserving, and the supply current is the whole
per-pulse charge delivered a second later &mdash; the two instruments cannot double-count
or miss a loop between them.</li>
<li>The Y-edge passivation ({RI['geometry']['passivation_lo_mm']:.1f} / {RI['geometry']['passivation_hi_mm']:.1f} mm, measured on this chamber) leaves
<strong>{RI['geometry']['n_y_live']} of 512 Y combs live</strong>, and shortens every X comb to {100*RI['geometry']['live_fraction']:.1f} % of its
length &mdash; {RI['geometry']['n_y_live']} full-length-equivalent channels per view.</li>
</ul>

<figure>
<img src="figures/board_stack.png" alt="Board cross-section and top view, with the two current loops">
<figcaption>Left: the stack, with the two measurement loops. The avalanche lands on the
resistive strips; 85 % of its image appears on the pad plane within the ion-collection
time, split 50/50 between the views; the one pad-comb on a 50 &Omega; termination reads
its <em>local</em> image on the microsecond timescale, while the whole charge drains
along the resistive strips to the end buses over ~{RB['board']['tau_drain_ms']:.0f} ms and through the supply,
where the monitor averages it over ~1 s. The image-return current through the strip
during the drain is {RBW['drain_return']['current_nA']:.0f} nA &mdash; {RBW['drain_return']['voltage_uV']:.1f} &micro;V across 50 &Omega;, invisible &mdash;
so the positive lobe cleanly measures the local image. Right: what &ldquo;strip
32&rdquo; is: a comb of every second pad along one row.</figcaption>
</figure>

<p>The uniform-illumination expectation for one live Y comb is then a number, not a
range:</p>

<div class="eq">
<var>Q</var><sub>strip</sub> = <var>Q</var><sub>chamber</sub> &times; 0.85 &times; &frac12; / {RI['geometry']['n_y_live']}
 = {RBW['q_imon_nC']:.0f} nC &times; {1e3*RB['board']['eff_share']:.2f}&permil; = <strong>{RBW['expected_uniform_pC']:.0f} pC</strong>
<span class="where">0.85 = image capture of the pad plane (W2 boundary; 0.84&ndash;0.85 depending on
solver detail) &middot; &frac12; = the checkerboard's exact view split &middot; {RI['geometry']['n_y_live']} live
Y combs &middot; at the 700/540 working point</span>
</div>

<figure>
<img src="figures/full_detector_compare.png" alt="Full-detector charge from both methods across the scan">
<figcaption>The single-strip waveform charge scaled to the chamber, against the supply
current, over the same plateaus at drift 700 V. Dedicated and parasitic pulses separate
as they should; both track the supply current with the same slope, offset by one
constant.</figcaption>
</figure>

<p><strong>The shape agreement is half the result.</strong> Point by point, the measured
strip charge over the uniform-share expectation is
<strong>{RB['residual_constancy']['mean']:.1f} &plusmn; {RB['residual_constancy']['sd']:.1f}</strong> across all {RB['residual_constancy']['n']} plateaus &mdash; a spread of
{RB['residual_constancy']['spread_pct']:.1f} % over a factor {QSPAN:.0f} in charge and three drift settings. Two instruments
sharing nothing agree on the shape of the response to a few per cent; what is left is a
single constant, and the board accounting has sharpened it: it is now a genuine factor
four, and it must be explained, not absorbed.</p>

<h4>What cannot carry the factor</h4>
<ul>
<li><strong>Anything passive in the signal path.</strong> The strip is a capacitive
current source: every series element (cable, back-termination, connector) passes its
whole charge to a 50 &Omega; load, and every <em>shunt</em> (splitter, attenuator,
a second termination, a still-connected DREAM input) makes the true strip charge
<em>larger</em> than measured. Passive patch errors can only widen the gap. The one
instrumental escape is a digitiser input well above 50 &Omega; &mdash; which the pulse
shape argues against, and a pulser through the same patch would settle.</li>
<li><strong>Lateral charge spreading.</strong> For uniform illumination, what diffuses
out of the comb's band diffuses in ({RB['board']['spread_in_window_mm']:.1f} mm in the window at the frozen
resistivity); conservation makes the average share kernel-independent.</li>
<li><strong>X/Y sharing.</strong> Exactly &frac12; by the checkerboard's symmetry.</li>
<li><strong>A missing current loop.</strong> The drift supply was checked across the
same scan: its monitor shows pure divider-current steps tracking the drift setting and
no beam response at all, so the avalanche charge all flows through the monitored resist
channel.</li>
<li><strong>The saturated neighbours.</strong> During the flash every DREAM-connected
comb rails within nanoseconds, and one could worry the lone 50 &Omega; channel drains
its neighbours' image through the capacitive network. The scale rules it out: a railed
input floats by at most a volt before its protection conducts, while the resistive layer
under {RBW['sag_V']['local_ded']:.0f} V of local flash charge is what drives the division &mdash; the
redistribution is a &le;10 % effect, not a factor.</li>
<li><strong>The neutron tail.</strong> The supply integrates the whole cycle, the lobe
only 9 &micro;s &mdash; so any tail charge makes the true residual <em>larger</em>, not
smaller. On the strip's own record the above-threshold tail beyond 30 &micro;s is zero
at the working point and 1&ndash;2 % at the hottest scan point.</li>
</ul>

<h4>What does carry it: the strip itself reports that its neighbourhood is crowded</h4>

<p>First, the honest baseline, stated without any interpretation: <strong>with the board
accounting and the assumption of a uniform response, the waveform and the supply current
disagree by one constant factor of {RB['residual_constancy']['mean']:.1f}.</strong> Everything in this subsection
is evidence about where that factor lives &mdash; specifically, that it is charge really
arriving near this strip, not a broken calibration. The evidence is a nonlinearity the
strip itself displays, and the argument has three steps:</p>

<figure>
<img src="figures/board_cartoon.png" alt="The three-step density argument as a cartoon">
<figcaption>The argument in cartoon form, one panel per step. (1) Flash charge lands on
the resistive layer, which is a capacitor plate that cannot drain during the flash
&mdash; so its voltage sags by &Delta;V = &sigma;/c&prime;, and since the gas gain is
exponential in that voltage, the charge throttles its own amplification. (2) The beam
alternates pulses of 1&times; and 2&times; protons; the strip returns 1.85&times;, not
2.00&times; &mdash; and being a ratio on one strip, that number is immune to every
absolute calibration. (3) Reading the shortfall backwards through the capacitor law
gauges how much charge landed locally: the chamber-average density would produce a 3 %
shortfall, the measured 7.6 % requires ~3&times; that density.</figcaption>
</figure>

<p><strong>Step 1 &mdash; arriving charge lowers the very voltage that amplifies
it.</strong> The resistive layer is one plate of a capacitor
(c&prime; = {RB['board']['cprime_uF_m2']:.2f} &micro;F per m&sup2; of area, computed from the board stack), and
during the flash it cannot drain &mdash; the path to the HV supply takes ~{RB['board']['tau_drain_ms']:.0f} ms.
So when the flash dumps charge onto some region of it, that region's voltage drops, by
exactly the capacitor law: &Delta;V = (charge per area) / c&prime;. But this is the
amplification voltage, and the gas gain depends on it exponentially &mdash; every
{RB['board']['e_fold_V']['mid']:.0f} V is a factor e (measured on this same scan). Put together: during a
flash, charge that arrives late is amplified <em>less</em> than charge that arrived
early, because the early charge has already pulled the voltage down. The denser the
charge landing in a region, the more of this self-suppression that region shows.</p>

<p><strong>Step 2 &mdash; the beam runs a controlled &ldquo;double the input&rdquo;
experiment every 36 s.</strong> Dedicated pulses carry 2.00&times; the protons of
parasitic ones (measured pulse by pulse with the beam pickup). If amplification were
linear, dedicated pulses would give exactly 2.00&times; the charge. Measured on strip 32
at the working point they give 1.85&times; &mdash; a <strong>{100*RBW['deficit']['measured']:.1f} % shortfall per
proton</strong>. Note what this shortfall is: a <em>ratio of two measurements on the
same strip</em>. Every linear calibration cancels out of a ratio &mdash; the 50 &Omega;
assumption, the capture fractions, the cable, all of it. Whatever else might be wrong
with our absolute scales, the {100*RBW['deficit']['measured']:.1f} % is real.</p>

<p><strong>Step 3 &mdash; the size of the shortfall is a ruler for the local
density.</strong> Run Step 1 backwards. To produce a {100*RBW['deficit']['measured']:.1f} % shortfall between
&times;1 and &times;2 pulses, the parasitic flash must pull the layer down by about
{RBW['sag_V']['local_par']:.0f} V <em>in the region where this strip's charge is amplified &mdash; directly
above it</em>. By the capacitor law, {RBW['sag_V']['local_par']:.0f} V of sag at c&prime; = {RB['board']['cprime_uF_m2']:.2f} &micro;F/m&sup2;
means about <strong>{RBW['sigma_par']['from_compression']:.0f} pC/cm&sup2;</strong> landed there. Now compare: the supply
current says the <em>chamber-average</em> parasitic flash is {RBW['sigma_par']['chamber_avg']:.0f} pC/cm&sup2;. If
this strip's neighbourhood received only that average, the sag would be
{RBW['sag_V']['chamber_avg']:.1f} V and the shortfall {100*RBW['deficit']['predicted_if_uniform']:.0f} % &mdash; the measured {100*RBW['deficit']['measured']:.1f} % excludes
it. So the strip's own nonlinearity, read with nothing but the board capacitance and the
measured gain slope, says its neighbourhood received <strong>~{RBW['enhancement']['compression']:.0f}&times; the
chamber-average density</strong>. The waveform lobe &mdash; the absolute route, with the
50 &Omega; and capture calibrations &mdash; says {RBW['sigma_par']['from_lobe']:.0f} pC/cm&sup2;, i.e.
~{RBW['enhancement']['lobe']:.0f}&times;. Two routes that share no calibration, the same answer to ~40 %:</p>

<figure>
<img src="figures/board_ledger.png" alt="Three densitometers at the working point, and the residual across the scan">
<figcaption>Left: the three measurements of the flash charge surface density at the
working point (parasitic pulses): the chamber average (supply current &divide; live
area), and the two independent local measurements at strip 32. Right: the
measured-over-expected ratio at every plateau &mdash; one constant.</figcaption>
</figure>

<p>The scan turns this one-point argument into a curve. Raising the amplification
voltage raises the charge per pulse by a factor 21 across the scan &mdash; and therefore
the density, and therefore the predicted shortfall, which grows along a definite curve
with <strong>nothing fitted</strong>: density from the lobe, capacitance from the board,
gain slope from the scan itself. The measured shortfalls ride that curve from
{100*RBW['deficit']['measured']:.0f} % at the working point up to ~29 % at the hottest point:</p>

<figure>
<img src="figures/board_compression.png" alt="Compression versus local flash density, with the parameter-free sheet-charging prediction">
<figcaption>How to read this: each blue point is one plateau of the scan. Its
<em>x</em> is the flash density at the strip inferred from the waveform lobe; its
<em>y</em> is the directly measured compression &mdash; how much less charge per proton
a dedicated pulse delivered than a parasitic one on that plateau. The black line is the
sheet-charging prediction, &sigma; &rarr; sag &rarr; deficit, with nothing fitted (grey
band: the measured range of gain e-folds). The blue points riding the line over a factor
~20 in density is the closure. The grey crosses are the <em>same measured deficits</em>
re-plotted at the density the strip would have under uniform illumination: they sit far
left of the line, i.e. uniform illumination cannot produce the compression actually
observed. (The scatter at low density is the low-gain end, where both classes' charges
are small and the deficit is poorly determined &mdash; see the error bars.)</figcaption>
</figure>

<p><strong>What the density factor is made of, this note cannot say.</strong> The
delivered density is (primary ionisation per area) &times; (local gas gain), so a
3&ndash;4&times; excess can be an illumination profile &mdash; the flash is prompt
&gamma;s and their conversions, and the material around the active area (frame, support
plate, board edge) serves an edge strip differently from the middle &mdash; or a
<em>local gain enhancement</em>: this strip sits 5&ndash;7 mm from the passivation edge,
where amplification-field distortion is expected and where the June bench work found
spark activity concentrated. Both are multiplicative constants, both are consistent with
the clean e-fold slope (a locally higher gain shifts the exponential without bending
it), and one channel cannot separate them. What would: the DREAM recovery-time map
across the plane (a gain hot-spot is fixed to the detector; an illumination profile
follows the beam), a second patched strip near the centre, or the June bench amplitude
maps of this very chamber at this strip's position.</p>

<h3>3. The absolute charge at the working point</h3>
<p>At the shared setpoint ({RCM['absolute']['setpoint']}), for the record:</p>

<div class="scroll"><table>
<thead><tr><th></th><th>what it measures</th><th>value</th></tr></thead>
<tbody>
<tr><th>HV supply current</th><td>whole chamber, whole cycle</td>
<td>{RI['working_point']['imon_nC']:.0f} nC per pulse</td></tr>
<tr><th>waveform, strip 32</th><td>one strip, prompt flash</td>
<td>{RI['working_point']['waveform_strip_pC']:.0f} pC per pulse (mix);
{RCM['absolute']['strip_pC_dedicated']:.0f} pC dedicated</td></tr>
<tr><th>uniform share, board accounting</th><td>chamber &times; 0.85 &times; &frac12; / {RI['geometry']['n_y_live']}</td>
<td>{RBW['expected_uniform_pC']:.0f} pC per strip</td></tr>
<tr><th>measured / uniform share</th><td>= local / average flash density</td>
<td>{RBW['residual']:.1f} (lobe); {RBW['enhancement']['compression']:.1f} (compression)</td></tr>
</tbody></table></div>

<p>The morning's independent measurement of the same setpoint, from run_158's six
sub-runs, gives {RCM['absolute']['hv_nC_per_pulse']:.0f} &plusmn; {RCM['absolute']['hv_nC_rms']:.0f} nC &mdash; the same to better than 1 %,
which is a useful check that nothing about the chamber moved across the day.</p>

<h3>4. The compression</h3>
<p>Both see the response fall short of proportional to beam intensity, in the same
direction and of the same size &mdash; expressed as how much less charge a dedicated
pulse delivers per proton than a parasitic one:</p>

<div class="scroll"><table>
<thead><tr><th>method</th><th>per-proton deficit, dedicated vs parasitic</th></tr></thead>
<tbody>
<tr><th>waveform, strip 32, det A</th><td>{RCM['compression']['waveform']['per_proton_deficit_pct']:.1f} %</td></tr>
<tr><th>HV current, isolated-pulse fold, det A</th><td>{RCM['compression']['hv_fold']['A']['per_proton_deficit_pct']:.1f} %</td></tr>
<tr><th>HV current, isolated-pulse fold, det C</th><td>{RCM['compression']['hv_fold']['C']['per_proton_deficit_pct']:.1f} %</td></tr>
</tbody></table></div>

<p>One instrument on a single strip, one on a whole chamber's supply, independently
seeing the same few-to-fifteen percent shortfall. That makes space charge in the
amplification region a good deal more credible than it was from either alone.</p>

<p>A natural question is whether the supply current can make the dedicated/parasitic
split <em>on this same scan</em>, point for point with the waveform. It cannot &mdash;
and the reason is timing, not statistics. There are plenty of parasitic pulses (36 at
the working point alone); the problem is that <strong>a parasitic pulse is never
alone</strong>. The monitor smears every pulse into a ~1&ndash;2 s bump and timestamps
its samples in whole seconds, so a pulse can only be read out individually when no other
pulse sits within a few seconds of it &mdash; and the PS supercycle delivers parasitic
pulses 1.2&ndash;2.4 s from their neighbours, always, so their bumps always overlap with
someone else's. Only dedicated pulses, which border the supercycle's long gaps, ever
stand alone: a strict isolation cut on this run keeps
<strong>{RB['imon_class_split']['n_isolated_strict']['dedicated']} dedicated and {RB['imon_class_split']['n_isolated_strict']['parasitic']} parasitic</strong> of 2 672. (The clock offset itself
resolves cleanly, +{RB['imon_class_split']['clock_offset_s']} s; and fitting the overlapping bumps instead of cutting them
is the ill-conditioning trap documented in the July handoff, &sect;8.6.) The split
<em>has</em> been made with the supply current where the 1 s time base was reconstructed
to millisecond accuracy &mdash; run_79 at the production point, the det A / det C rows
in the table above &mdash; and that is the imon's answer: same sign, same size as the
waveform's.</p>

<p>The pulse mix does not, however, drag the comparison anywhere: the strip-to-chamber
ratio is taken mix-against-mix &mdash; both instruments average the same bunches &mdash;
so the {RB['residual_constancy']['mean']:.1f} is mix-independent. And the mix <em>can</em> be separated
algebraically, using the counted classes and the waveform-measured charge ratio: at the
working point the chamber delivers <strong>{RBW['chamber_by_class_nC']['dedicated']:.0f} nC per dedicated pulse and
{RBW['chamber_by_class_nC']['parasitic']:.0f} nC per parasitic pulse</strong>, which are the numbers to quote when a
single-class figure is needed rather than the {RBW['q_imon_nC']:.0f} nC mix average.</p>

<h2>What this means for DREAM</h2>

<p>The DREAM charge-sensitive amplifier has selectable input ranges of
50 / 100 / 200 / 600 fC. Taking the largest:</p>

<div class="kpis">
  <div class="kpi"><div class="n">{RCM['dream']['strip_over_full_scale']:,.0f}&times;</div>
    <div class="l">measured strip charge over the 600 fC full scale</div></div>
  <div class="kpi"><div class="n">{RBW['expected_uniform_pC']*1e3/RCM['dream']['csa_full_scale_fC']:,.0f}&times;</div>
    <div class="l">chamber-average strip, from the HV current + board accounting</div></div>
  <div class="kpi"><div class="n">{RCM['dream']['drain_time_ms'][1]:.0f}&ndash;{RCM['dream']['drain_time_ms'][0]:.0f} ms</div>
    <div class="l">time to drain that charge at the CSA's 90&ndash;9 nA feedback limit</div></div>
</div>

<p><strong>The flash puts roughly a thousand times the front end's full-scale input
charge onto a single strip.</strong> That is the answer to &ldquo;how much can we
swallow before we die&rdquo;: not this much, and not by a small margin. The
chamber-average figure from the HV current with the board accounting,
{RBW['expected_uniform_pC']*1e3/RCM['dream']['csa_full_scale_fC']:,.0f}&times;, is the optimistic bound &mdash; a typical channel away from the hot
region; the strip actually measured sits at {RCM['dream']['strip_over_full_scale']:,.0f}&times;. And 600 fC is the
<em>largest</em> of the four selectable ranges: a parallel audit of 44 saved bench
configs reads the DREAM registers as 200 fC, which if it holds for the beam runs makes
every multiple here three times worse.</p>

<p><strong>Gain cannot fix it.</strong> With the measured e-folding of {R9['gain_fits']['700_dedicated']['e_fold_V']:.0f} V,
bringing {RCM['absolute']['strip_pC_dedicated']:.0f} pC down to 600 fC takes ln({RCM['dream']['strip_over_full_scale']:,.0f}) &times; {R9['gain_fits']['700_dedicated']['e_fold_V']:.0f} V =
<strong>{RCM['dream']['volts_to_fit_in_range']:.0f} V</strong> less amplification &mdash; an operating point near
{RCM['dream']['voltage_that_would_be_needed']:.0f} V, where the gain for an actual track is also down by a factor of a thousand
and there is nothing left to read. The flash and the physics scale together; the
amplification knob does not separate them.</p>

<p><strong>And the dead time falls out of it.</strong> A pinned CSA recovers by draining
the accumulated charge through its feedback, which the saturation note puts at
9&ndash;90 nA. <var>Q</var>/<var>I</var><sub>fb</sub> for {RCM['absolute']['strip_pC_dedicated']:.0f} pC is
<strong>{RCM['dream']['drain_time_ms'][1]:.1f} to {RCM['dream']['drain_time_ms'][0]:.0f} ms</strong> &mdash; which brackets the
millisecond-scale dead time DREAM actually shows, from an independent measurement of the
charge, and agrees with the parallel result that dead time follows delivered charge as
<var>t</var> &prop; <var>Q</var><sup>1.2</sup> across three chambers. Appendix D shows the
chamber itself is clear within microseconds, so all of that millisecond structure lives in
the front end.</p>

<h2>Decoding integrity</h2>
<p>The user-facing worry with this raw format is sign. The samples are
<code>int16_t</code> &mdash; ntoflib declares <code>std::vector&lt;int16_t&gt; data</code>
in <code>ReaderStructACQC.h</code>, and the n_TOF PSA reads them that way. Decoded signed,
nothing wraps:</p>
<ul>
<li>{integ['checked_samples']:,} samples checked across both beam runs</li>
<li>{integ['samples_at_positive_rail']} at the positive rail,
    {integ['samples_at_negative_rail_or_fill']} at the negative rail</li>
<li>{integ['sample_to_sample_jumps_over_20000']} sample-to-sample steps above 20 000 counts
    &mdash; the wrap signature &mdash; with the largest step
    {integ['largest_jump_counts']:,} counts</li>
</ul>
<p>The zero-suppression fill code is <code>-32768</code>, bit-identical to the negative
rail, and is masked rather than integrated. A block&rsquo;s payload begins 259 samples
before its stated start.</p>

<h2>What this does not establish</h2>
<div class="caveat">
<ul>
<li><strong>Which chamber the July runs are.</strong> All four MX17 chambers step in
lockstep there &mdash; every HV state is a joint (A,B,C,D) setting and each ladder spans
the same 20 V &mdash; so the July flash charge correlates identically with all four. The
August run settles its own case (detector A, strip 32, cable Y8, corroborated by the
A-only scan), but it does not retroactively identify the July channel: `MMB` was live
then, `MMA` now, and the input range changed by ten times in between.</li>
<li><strong>The DREAM side was not re-measured here.</strong> This work shows the chamber
recovers in microseconds. It does not itself demonstrate what the DREAM preamplifier,
shaper or zero-suppression do with the same charge &mdash; that comparison is the obvious
next step, and DREAM run_18 is simultaneous with 224327, so it can be done on the same
bunches.</li>
<li><strong>The absolute charge scale assumes a direct 50 &Omega; termination</strong> and
no amplification between chamber and digitiser. Note the direction: every passive error
in that path &mdash; a splitter, an attenuator, a second termination, a still-connected
DREAM input &mdash; makes the true strip charge <em>larger</em> than quoted, never
smaller. Only a digitiser input well above 50 &Omega; would lower it, which the pulse
shape argues against. A pulser through the same patch would pin the whole chain and is
the single most valuable follow-up measurement. The 10&ndash;20 m of RG-58 is
<em>not</em> a worry &mdash; it costs under 1 % of the charge, and that is shown rather
than assumed above.</li>
<li><strong>Why the density is high at this strip is inferred, not mapped.</strong> The
board accounting plus the compression establish that the residual factor
{RB['residual_constancy']['mean']:.1f} is real local charge density &mdash; but density is illumination &times; local
gain, and one channel cannot split the product. Four measurements would: (1) the
per-channel <em>recovery-time</em> map from DREAM run_160/161, simultaneous with this
scan &mdash; dead time follows delivered charge as t &prop; Q<sup>1.2</sup>, so the
saturation-recovery pattern across the plane is a flash-charge profile read from data
that already exist, and a gain hot-spot is fixed to the detector while an illumination
profile follows the beam; (2) the June cosmic-bench amplitude maps of this very chamber
at this strip's position, which bound the gain component alone; (3) a second patched
strip near the chamber centre; (4) the pulser calibration above, which removes the one
instrumental alternative. The two local densitometers also differ by ~40 %
({RBW['enhancement']['lobe']:.1f} vs {RBW['enhancement']['compression']:.1f}), which is attributed to the roughness of the sheet-charging
model and not further explained here.</li>
<li><strong>The CAEN monitor's absolute DC calibration</strong> is still taken on trust
(the July work verified it conserves burst integrals and resolves nanoamps, not that its
ampere is an ampere). A scale error there moves the chamber charge and the residual
factor together.</li>
<li><strong>Which strip this is rests on reading the label.</strong> &ldquo;Strip 32 of
cable Y8&rdquo; is taken as connector Y8, channel 32 (y = 374.4 mm); the alternative
reading, global y-strip 32, is y = 25.0 mm. Both sit 5&ndash;7 mm inside a passivation
edge, so the periphery conclusion holds either way &mdash; but the two are 350 mm apart
and the cabling record, not this note, should settle it.</li>
<li><strong>Which CSA input range is actually in use.</strong> Every &ldquo;&times; full
scale&rdquo; figure here takes the largest of the four DREAM settings (600 fC), which is
the conservative choice. A parallel audit of 44 saved <em>bench</em> configs reads
<code>Dream 6/7 = 0xAAAA</code> &rarr; <strong>200 fC at 10 mV/fC</strong>; if the beam
runs used the same setting, every multiple here is three times worse and the strip sits
at ~3 300&times; full scale. That has not been read back from a beam-run FEU, so the
larger, safer number is quoted.</li>
<li><strong>The 9&ndash;90 nA feedback limit is taken from the saturation note</strong>,
not measured here. The drain-time bracket inherits its factor of ten directly &mdash; that
it lands on the observed millisecond dead time is a consistency check, not a derivation
of it.</li>
<li><strong>Neither method separates the prompt flash from the neutron-induced tail.</strong>
The waveform integral covers 11&ndash;20 &micro;s, so it is prompt by construction; the HV
current integrates the whole cycle. Any tail charge in the supply current therefore makes
the density-enhancement factor an <em>under</em>-estimate. The strip's own record bounds
its share of the tail: above threshold, everything beyond 30 &micro;s sums to zero at the
working point and 1&ndash;2 % at the hottest scan point.</li>
<li><strong>224302's conditions are only partly known.</strong> DREAM run_12 covers the
last 41 % of it (1256 of 3067 bunches); there was no DREAM run before that, so the HV for
the earlier bunches is unmonitored. The gas differs from the other runs
(Ar/CF4/Iso 88/10/2 against Ar/Iso 95/5).</li>
<li><strong>224325 has no HV or DREAM record at all</strong> &mdash; it falls in a gap
between DREAM runs. It is used only as a beam-off liveness control, which needs neither.</li>
</ul>
</div>

<h2>Provenance</h2>
<p><strong>The detector-A scan (224709).</strong> 1.5 TB of raw stream1; the MMA channel
was extracted at full sample resolution and secured to
<code>/eos/experiment/ntof/data/x17/mm_raw_2026-08/</code> (154 MB) rather than copying
the bulk. <code>extract_mm_full.py</code> &rarr; <code>merge_709.py</code> &rarr;
<code>analyse_709.py</code>; the charge chain and the cable in
<code>charge_chain.py</code>; the cross-comparison in
<code>compare_hv_current.py</code>. HV plateaus from the DREAM
<code>hv_monitor.csv</code> of run_160/161, in <code>hv_plateaus_224709.csv</code>.</p>

<p><strong>The board accounting</strong> is <code>board_accounting.py</code>. Its
constants come from the response-simulation repository
<code>~/CLionProjects/MX17_Geant</code> &mdash; the checkerboard channel map read from
the L5/L6 gerber stubs (<code>response/common/channel_map.py</code>), the pad-plane
image capture from the solved weighting potentials
(<code>response/solver/wpot_w2.py</code>, <code>design/report/V6_PAD_GAPS_2026-08-08.md</code>),
the sheet capacitance and drain time from the stack constants
(<code>response/common/constants.py</code>, <code>response/solver/wpot.py</code>), and the
ESL strip pattern and end-bus contact from <code>design/RESPONSE_SIM_PLAN.md</code>. The
per-source citation list is embedded in <code>results_board.json</code>.</p>

<p><strong>The July runs (Appendix A).</strong> Raw secured to
<code>/eos/experiment/ntof/data/x17/mm_raw_2026-07/</code> (591 files, 24 G,
adler32-verified against the DAQ source). <code>extract_mm.py</code> &rarr;
<code>merge_mm.py</code> &rarr; <code>analyse.py</code>; integrity check in
<code>wrapcheck.py</code>. All of the above is in
<code>ntof_processing/mm_flash/</code>, and this page is generated from their JSON by
<code>make_report.py</code> &mdash; re-running the analyses updates the numbers, tables
and figures together.</p>

<p><strong>The HV-current measurement</strong> is not this package's work: it is
<code>ntof_july_analysis/flash_charge/</code>
(<code>charge_lib.py</code>, <code>analyze.py</code>, <code>imon_response.py</code>) with
its own handoff, <code>HANDOFF_FLASH_CHARGE_2026-08-09.md</code>, which carries the
validations, the monitor impulse-response measurement and the dead-time-versus-charge
join in full. Only its published results are used here.</p>

<p>Run inventory, the cabling record and the runs that would need a tape recall are in
<code>ntof_processing/NTOF_MICROMEGAS_SIGNALS.md</code>.</p>
"""

def document(body):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{SUMMARY}">
<style>{STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""


def cut(body, start, end):
    """Remove a sentinel-fenced block and return (block, remainder)."""
    a = body.index(start)
    b = body.index(end) + len(end)
    return body[a:b], body[:a] + body[b:]


def collapse_tables(html):
    """Wrap every data table in a <details> so the page reads as prose by default.

    The caller asked for the numbers to be there but out of the way. Each table
    keeps whatever <h3>/<h4> or leading sentence preceded it as its summary line.
    """
    out, n = [], 0
    for chunk in re.split(r'(<div class="scroll"><table>.*?</table></div>)', html,
                          flags=re.S):
        if chunk.startswith('<div class="scroll"><table>'):
            n += 1
            rows = chunk.count('<tr>') - chunk.count('<thead>')
            out.append(f'<details><summary>Table {n} &mdash; {rows} rows</summary>\n'
                       f'{chunk}\n</details>')
        else:
            out.append(chunk)
    return ''.join(out)


def reorder(body):
    """Promote the detector-A scan to the main line and demote July to an appendix.

    The July material is written in place (it grew first); rather than shuffle a
    600-line f-string by hand, it is fenced with sentinels and moved here, so the
    prose keeps one source and cannot drift out of sync with itself.
    """
    deriv, body = cut(body, '<!--CHARGE-DERIV-START-->', '<!--CHARGE-DERIV-END-->')
    cable, body = cut(body, '<!--CABLE-START-->', '<!--CABLE-END-->')
    hvdet, body = cut(body, '<!--HVDETAIL-START-->', '<!--HVDETAIL-END-->')

    a = body.index('<!--JULY-START-->')
    b = body.index('<!--JULY-END-->') + len('<!--JULY-END-->')
    july = body[a:b]
    rest = body[:a] + body[b:]

    # the integrity section becomes appendix B, immediately after July
    ik = rest.index('<h2>Decoding integrity</h2>')
    ie = rest.index('<h2>What this does not establish</h2>')
    integrity = rest[ik:ie]
    rest = rest[:ik] + rest[ie:]

    prov = rest.index('<h2>Provenance</h2>')
    head, tail = rest[:prov], rest[prov:]

    def app(title, block, anchor):
        return ('<div class="appendix">\n<h2 id="' + anchor + '">' + title
                + '</h2>\n' + block + '\n</div>\n')

    return (head
            + app('Appendix A &mdash; how the charge is computed', deriv, 'appendix-charge')
            + app('Appendix B &mdash; the cable, and why the charge survives it', cable, 'appendix-cable')
            + app('Appendix C &mdash; the HV-current method in detail', hvdet, 'appendix-hv')
            + '<div class="appendix">\n' + july.replace(
                '<h2 id="appendix-a">Appendix A &mdash; July 2026:',
                '<h2 id="appendix-july">Appendix D &mdash; July 2026:') + '\n</div>\n'
            + '<div class="appendix">\n'
            + integrity.replace('<h2>Decoding integrity</h2>',
                                '<h2>Appendix E &mdash; decoding integrity</h2>')
            + '</div>\n'
            + tail)


BODY = collapse_tables(reorder(BODY))

# In-repo copy: relative figure links, so the DAQ analysis page can serve it.
(HERE / 'report.html').write_text(document(BODY))
print('wrote', HERE / 'report.html')

# Copy for the CERN notes site, which must be one self-contained file: every
# figure inlined as a data: URI, plus the front-matter block the listing reads.
import base64
import re as _re

META = f"""<!--note
date: 2026-08-09
title: {TITLE}
summary: {SUMMARY}
tags: X17, nTOF, micromegas, gamma flash, DAQ
-->
"""


def _inline(m):
    data = base64.b64encode((HERE / m.group(1)).read_bytes()).decode()
    return f'src="data:image/png;base64,{data}"'


standalone = META + document(_re.sub(r'src="(figures/[^"]+)"', _inline, BODY))
(HERE / 'report_standalone.html').write_text(standalone)
print('wrote', HERE / 'report_standalone.html', len(standalone) // 1024, 'KiB')
