#!/usr/bin/env python3
"""Build report.html for the plastic-scintillator after-pulse study.

Numbers come from the analysis artifacts, so re-running the analysis and then
this script keeps the tables, the headline and the verdict consistent.

    python make_report.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PSS = ['PSSA', 'PSSB', 'PSSC', 'PSSD']
CTRL = ['WALA', 'WALB', 'LIQA']

CSS = """
:root { --ink:#20242b; --muted:#6b7280; --rule:#e3e5e8; --surface:#fcfcfb;
        --accent:#0072B2; --warn:#D55E00; }
* { box-sizing:border-box; }
body { margin:0; padding:2.2rem 1.4rem 4rem; background:var(--surface);
       color:var(--ink); font:16px/1.62 -apple-system,BlinkMacSystemFont,
       "Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:60rem; margin:0 auto; }
h1 { font-size:1.72rem; line-height:1.25; margin:0 0 .3rem; }
h2 { font-size:1.16rem; margin:2.6rem 0 .7rem; padding-bottom:.3rem;
     border-bottom:1px solid var(--rule); }
h3 { font-size:1rem; margin:1.7rem 0 .5rem; }
.sub { color:var(--muted); margin:0 0 1.6rem; font-size:.94rem; }
.verdict { border-left:4px solid var(--accent); background:#f2f7fb;
           padding:1rem 1.2rem; margin:1.4rem 0 1.8rem; border-radius:0 6px 6px 0; }
.verdict p { margin:.45rem 0; }
.nums { display:flex; flex-wrap:wrap; gap:.8rem; margin:1.3rem 0 1.6rem; }
.num { flex:1 1 12rem; border:1px solid var(--rule); border-radius:8px;
       padding:.75rem .9rem; background:#fff; }
.num .v { font-size:1.42rem; font-weight:650; color:var(--accent);
          font-variant-numeric:tabular-nums; }
.num .k { font-size:.79rem; color:var(--muted); margin-top:.15rem; }
table { border-collapse:collapse; width:100%; margin:.9rem 0 1.2rem;
        font-size:.88rem; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:.42rem .6rem; border-bottom:1px solid var(--rule); }
th:first-child,td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; font-size:.8rem; }
tbody tr:hover { background:#f5f7f9; }
figure { margin:1.6rem 0; }
figure img { width:100%; border:1px solid var(--rule); border-radius:6px; }
figcaption { color:var(--muted); font-size:.86rem; margin-top:.5rem; }
code { background:#f0f2f4; padding:.1rem .3rem; border-radius:3px;
       font-size:.87em; }
.scroll { overflow-x:auto; }
ul { padding-left:1.2rem; } li { margin:.3rem 0; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e6e8ea; --muted:#9aa3ad; --rule:#333940; --surface:#16191d; }
  .num, table { background:#1c2025; }
  .verdict { background:#182430; }
  code { background:#252a30; }
  tbody tr:hover { background:#22272d; }
  figure img { background:#fff; }
}
"""


def band(res, det, lo, hi):
    d = res['dets'][det]
    e = np.array(res['edges'])
    c = 0.5 * (e[:-1] + e[1:])
    o = np.array(d['counts']) / d['n_leaders']
    m = np.array(d['mixed']) * d['mix_scale'] / d['n_leaders']
    s = (c >= lo) & (c < hi)
    return float((o - m)[s].sum())


def main():
    res = json.loads((HERE / 'afterpulse.json').read_text())
    veto = json.loads((HERE / 'veto_scan.json').read_text())
    core_base, late_base = veto['base']['core'], veto['base']['late']
    late_cut = 100 * next(x['late_removed'] for x in veto['scan']
                          if x['t_hold'] == 1000.0 and x['ratio'] == 0.05)
    fwd = json.loads((HERE / 'fwd.json').read_text())
    rev = json.loads((HERE / 'rev.json').read_text())
    cond = np.load(HERE / 'echo_cond_PSSB.npz')

    pss_tail = [band(res, d, 18, 1000) for d in PSS]
    pss_echo = [band(res, d, 79, 85) for d in PSS]
    wal_tail = [band(res, d, 18, 1000) for d in ('WALA', 'WALB')]

    rows = []
    for det in PSS + CTRL:
        d = res['dets'][det]
        rows.append(
            f"<tr><td>{det}</td><td>{d['n_leaders']:,}</td>"
            f"<td>{d['lead_amp_median']:,.0f}</td>"
            f"<td>{band(res, det, 18, 100):+.3f}</td>"
            f"<td>{band(res, det, 100, 1000):+.3f}</td>"
            f"<td>{band(res, det, 18, 1000):+.3f}</td>"
            f"<td>{band(res, det, 79, 85):+.4f}</td>"
            f"<td>{band(res, det, 2000, 20000):+.3f}</td></tr>")

    amp_rows = []
    for det in PSS:
        for b in res['dets'][det]['bands']:
            if (b['lo'], b['hi']) != (50, 100) or 'ratio_median' not in b:
                continue
            amp_rows.append(
                f"<tr><td>{det}</td><td>{b['n']:,}</td>"
                f"<td>{b['foll_amp_median']:,.0f}</td>"
                f"<td>{b['ratio_median'] * 100:.1f} %</td>"
                f"<td>{b['corr']:+.2f}</td></tr>")

    f_tail = band(fwd, 'PSSB', 18, 800)
    r_tail = band(rev, 'PSSB', 18, 800)
    f_echo = band(fwd, 'PSSB', 79, 85)
    r_echo = band(rev, 'PSSB', 79, 85)

    pre = int(cond['pre'])
    diff = cond['A_mean'] - cond['B_mean']
    bump = float(diff[pre + 78:pre + 88].max())

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Do the n_TOF plastics ring?</title><style>{CSS}</style></head>
<body><main>

<h1>Do the n_TOF plastic scintillators ring?</h1>
<p class="sub">Run 224572, v12 reprocessing &middot; raw stream1 chunks
segments 8, 20, 40 &middot; 2026-08-09</p>

<div class="verdict">
<p><strong>Yes &mdash; and the Pulse Shape Analysis is right to report it.</strong>
Every large plastic pulse is followed by a train of <em>real</em> secondary
pulses in the raw trace, from ~20&nbsp;ns out to about 1&nbsp;&micro;s. The PSA
turns them into <strong>{np.mean(pss_tail):.1f} extra hits per large pulse</strong>,
against <strong>{np.mean(wal_tail):.3f}</strong> on the SiPM walls in the same
beam &mdash; a factor of ~{np.mean(pss_tail) / max(np.mean(wal_tail), 1e-9):,.0f}.
And it accounts for essentially all of the long tail in the DREAM/PSS match:
on the reference pair the plastic excess at 150&ndash;1000&nbsp;ns is
{late_base:,} hits against a core of {core_base:,}, and a flag that knows only
&ldquo;this hit is under 5&nbsp;% of a bigger hit on the same channel in the
previous microsecond&rdquo; removes <strong>{late_cut:.1f}&nbsp;%</strong> of it
&mdash; see <a href="report_veto.html">the companion note</a>.</p>
<p>It is not a resonant ring. Two components sit on top of each other: a broad,
sporadic after-pulse population peaking at 32&ndash;40&nbsp;ns and decaying
through a microsecond, and a razor-sharp <strong>2&nbsp;ns-wide echo at
81&ndash;82&nbsp;ns</strong> that appears on all four plastics at the same delay
&mdash; the signature of a cable reflection (~8&nbsp;m of cable, one way).</p>
</div>

<div class="nums">
  <div class="num"><div class="v">{np.mean(pss_tail):.1f}</div>
    <div class="k">excess PSA hits per large plastic pulse, 18&ndash;1000 ns</div></div>
  <div class="num"><div class="v">{np.mean(wal_tail):.3f}</div>
    <div class="k">the same number on the SiPM walls</div></div>
  <div class="num"><div class="v">81&ndash;82 ns</div>
    <div class="k">fixed echo delay, identical on all four plastics</div></div>
  <div class="num"><div class="v">{np.mean(pss_echo):.2f}</div>
    <div class="k">echo hits per pulse (the sharp spike alone)</div></div>
  <div class="num"><div class="v">{bump * 100:.1f} %</div>
    <div class="k">of the primary&rsquo;s height, the echo in the raw trace</div></div>
</div>

<p><strong>What to do about it:</strong>
<a href="report_veto.html">Rejecting plastic after-pulses in the slim</a> &mdash;
a per-hit flag, a per-trigger metric, and what each costs.</p>

<h2>What was measured, and against what</h2>
<p>Take every <strong>isolated large hit</strong> on one plastic channel at
physics times &mdash; <code>amp_0 &gt; 3000</code> with nothing on the same
channel for 5&nbsp;&micro;s before it &mdash; and histogram the delay to every
PSA hit that follows it on that channel. Because the plastics run at ~720&nbsp;kHz
singles, the accidental level is <strong>measured, not modelled</strong>: the
identical construction is repeated with each leader&rsquo;s time transplanted
into a different bunch of the same channel, which carries the same rate profile
and the same dead time but no correlation.</p>
<p>The <strong>SiPM walls are the control that matters</strong>: same beam, same
digitiser, same PSA, same run &mdash; and a pulse three times wider, so if this
were the PSA mis-fitting a long tail the walls would show it worst.</p>

<figure>
  <img src="figures/deltat_spectrum.png"
       alt="Excess follower hits per leader against delay, for the four plastics, a wall and a liquid">
  <figcaption><strong>The correlated tail, and the echo inside it.</strong> Left:
  excess over the event-mixed accidental level, log&ndash;log. The four plastics
  lie on top of each other and run three decades above the walls. Right: the same
  data linearly below 160&nbsp;ns. The excess turns on at the PSA&rsquo;s
  two-pulse resolution (~18&nbsp;ns), peaks at 32&ndash;40&nbsp;ns, and carries a
  2&nbsp;ns-wide spike at 81&ndash;82&nbsp;ns that is 6&ndash;8&times; the local
  level on every channel.</figcaption>
</figure>

<div class="scroll"><table>
<thead><tr><th>channel</th><th>leaders</th><th>median amp_0</th>
<th>18&ndash;100 ns</th><th>100&ndash;1000 ns</th><th>18&ndash;1000 ns</th>
<th>echo 79&ndash;85 ns</th><th>2&ndash;20 &micro;s</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="sub">Excess follower hits per leader, accidental level subtracted.
The slightly negative 2&ndash;20&nbsp;&micro;s column is a known bias of the
control at long delay: a leader is <em>required</em> to sit in a quiet stretch,
so its own neighbourhood is quieter than the mixed sample it is compared with.
It does not touch the short-delay result, where the control is essentially zero.</p>

<h2>Four things that say this is real and detector-side</h2>

<h3>1. It is strictly forward in time</h3>
<p>Running the same analysis with the clock reversed &mdash; asking what
<em>precedes</em> a large pulse instead of what follows it, with the isolation
requirement dropped so the two directions are symmetric &mdash; gives
<strong>{f_tail:.2f}</strong> excess hits per leader forward against
<strong>{r_tail:.2f}</strong> backward over 18&ndash;800&nbsp;ns on PSSB. The echo
is sharper still: <strong>{f_echo:.3f}</strong> forward against
<strong>{r_echo:.3f}</strong> backward, a factor of
{f_echo / max(r_echo, 1e-9):.0f}. Real physics and accidentals are symmetric in
time; this is not.</p>

<h3>2. The walls, with a wider pulse, show none of it</h3>
<p>{wal_tail[0]:+.3f} and {wal_tail[1]:+.3f} excess hits per leader on WALA and
WALB against ~{np.mean(pss_tail):.1f} on the plastics. The wall pulse is
~72&nbsp;ns FWHM against the plastics&rsquo; ~5&ndash;18&nbsp;ns and its tail is
an order of magnitude fatter (next figure), so a fit artifact driven by pulse
tails would have to be worse there, not absent.</p>

<figure>
  <img src="figures/pulse_tails.png" alt="Median amplitude-normalised trace after a pulse">
  <figcaption><strong>The median tail carries no oscillation.</strong>
  Amplitude-normalised median trace over thousands of large isolated pulses. The
  plastics decay smoothly through 2&nbsp;% at 50&nbsp;ns to 0.2&nbsp;% at
  200&nbsp;ns, with no overshoot, no undershoot and no periodic structure &mdash;
  and the walls sit an order of magnitude <em>above</em> them throughout. A
  median is blind to a feature that only occurs on a minority of pulses, which is
  precisely what the after-pulses turn out to be.</figcaption>
</figure>

<h3>3. The follower amplitude does not scale with the primary</h3>
<div class="scroll"><table>
<thead><tr><th>channel</th><th>pairs at 50&ndash;100 ns</th>
<th>median follower amp_0</th><th>as a fraction of the leader</th>
<th>corr(leader, follower)</th></tr></thead>
<tbody>{''.join(amp_rows)}</tbody></table></div>
<p class="sub">A pure reflection would be proportional to what it reflects. These
are ~120&nbsp;ADC almost independently of a leader that ranges over an order of
magnitude, so the broad component is not one bounce of the primary.</p>

<h3>4. The 81 ns hits sit on a real bump in the raw samples</h3>
<p>Split the leaders by whether the PSA gave them a hit at 79&ndash;85&nbsp;ns,
pull the raw stream1 trace behind each, and average. The ones that got the hit
show a bump peaking at 82&nbsp;ns worth <strong>{bump * 100:.1f}&nbsp;%</strong> of
the primary; the ones that did not are smooth there. The PSA is not inventing
this.</p>

<figure>
  <img src="figures/echo_conditional.png" alt="Mean raw trace with and without the 81 ns hit">
  <figcaption><strong>The echo, conditioned on the hit.</strong> Mean
  amplitude-normalised PSSB trace for leaders with a PSA hit at 79&ndash;85&nbsp;ns
  (n={int(cond['A_n'])}) against leaders with nothing within 120&nbsp;ns
  (n={int(cond['B_n'])}). Below, the difference. The bump is localised at
  82&nbsp;ns; the smooth offset between the two classes is the selection
  (pulses that get a follower have larger tails generally).</figcaption>
</figure>

<h2>What one pulse actually looks like</h2>
<figure>
  <img src="figures/event_display.png" alt="Raw traces of single plastic pulses with PSA hits marked">
  <figcaption><strong>This is the whole result in one picture.</strong> Four
  zero-suppressed PSSB blocks, each triggered by one large isolated pulse, with
  every PSA hit drawn as a vertical line. The secondary pulses are unmistakable
  and discrete &mdash; each has its own rise and decay, the trace returns to
  baseline between them, and they run out to 500&nbsp;ns and beyond at
  1&ndash;5&nbsp;% of the primary. The PSA is reporting what is there.</figcaption>
</figure>

<figure>
  <img src="figures/same_block.png" alt="Followers split by whether they are in the leader's own recorded block">
  <figcaption><strong>Where the followers live in the readout.</strong> The
  zero-suppressed record is a guaranteed 769 samples (259 before the trigger,
  510 after) and extends while the signal keeps crossing threshold. The entire
  correlated tail falls inside the primary&rsquo;s own record, which is why the
  raw samples above could be checked at all; past ~1&nbsp;&micro;s the followers
  are separate records and sit at the accidental level.</figcaption>
</figure>

<h2>What this does <em>not</em> establish</h2>
<ul>
<li><strong>The mechanism of the broad component.</strong> Delays of
20&ndash;500&nbsp;ns with no amplitude proportionality fit PMT late pulses
(photoelectrons elastically backscattered off the first dynode, which arrive at
roughly twice the cathode&ndash;dynode transit time) or multiple reflections on a
mismatched line. Ion-feedback afterpulsing, which lives at 0.1&ndash;5&nbsp;&micro;s,
cannot explain a peak at 32&ndash;40&nbsp;ns. Nothing here distinguishes those,
and it would take a bench pulse-injection test to do so.</li>
<li><strong>The 8&nbsp;m cable length.</strong> That is arithmetic from
81.5&nbsp;ns of round trip at 0.66&nbsp;c, not a measurement. The delay is solid;
the cable it implies is a hypothesis to check against the EAR2 cabling.</li>
<li><strong>What the remaining core is.</strong> The after-pulse flag costs
~10&nbsp;% of the background-subtracted core at |dt| &lt; 25&nbsp;ns, all of it
small-amplitude. Whether those are genuine low-light coincidences or after-pulses
that happen to land inside the accept window &mdash; the effect turns on at
18&nbsp;ns &mdash; is not separated here.</li>
<li><strong>Generality across the campaign.</strong> One run (224572), ten
segments of hits, three raw chunks. All four plastics agree to within 30&nbsp;%
on every number, which argues the effect is structural rather than a channel
fault, but no other run has been checked.</li>
<li><strong>Whether the liquids share a mechanism.</strong> LIQA shows a much
larger correlated excess ({band(res, 'LIQA', 18, 1000):.1f} per leader) with its
own time structure. That is a separate question, already partly covered by the
v12 liquid pile-up work.</li>
</ul>

<h2>Reproducing</h2>
<p>All of it runs from local data
(<code>/media/dylan/data/x17/ntof_reproc/v12_liqpileup</code> and
<code>/media/dylan/data/x17/ntof_raw_224572</code>) in a few minutes:</p>
<pre><code>python afterpulse_spectrum.py --parts 1 -o afterpulse.json
python afterpulse_spectrum.py --parts 1 --dets PSSB --quiet 0 --max-dt 2000 -o fwd.json
python afterpulse_spectrum.py --parts 1 --dets PSSB --quiet 0 --max-dt 2000 --reverse -o rev.json
python raw_pss_blocks.py  &lt;head_8.bin&gt; --dets PSSA..LIQA --stack stack_head8.npz
python echo_conditional.py &lt;head_8 20 40&gt; --det PSSB -o echo_cond_PSSB.npz
python same_block.py       &lt;head_8 20 40&gt; -o same_block.json
python event_display.py    &lt;head_8.bin&gt; --det PSSB -n 4
python make_figures.py &amp;&amp; python make_report.py</code></pre>

</main></body></html>
"""
    out = HERE / 'report.html'
    out.write_text(html)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
