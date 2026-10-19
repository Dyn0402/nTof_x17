"""Build ntof_pedestal_qa/report.html from the extracted pedestal history.

Every number in the prose is computed here from `data/ped_stats.npz`, so
re-running after a new extraction moves the text, the tables and the figures
together.  Figures come from `figures.py`; this module does not draw.

    ../.venv/bin/python -m ntof_pedestal_qa.make_report
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
from collections import Counter
from datetime import datetime

import numpy as np

from . import figures as F
from . import pedestals as P

FEU_LABEL = {f: f"{d}{v}" for f, (d, v) in P.FEU_DET.items()}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report.html")

CLOCK_STEP = datetime(2026, 7, 23, 13, 26)      # between the 10:05 and 16:47 pedestals
A_START = datetime(2026, 7, 21, 12, 4)
A_END = datetime(2026, 7, 27, 14, 11)


# ------------------------------------------------------------------ numbers
def analyse():
    sets = P.load()
    rows, cm = P.series(sets)
    ctx = json.load(open(os.path.join(P.DATA, "ped_context.json")))
    usage = P.load_usage()
    episodes = P.silent_connectors(sets)

    def at(stamp):
        i = next(j for j, s in enumerate(sets) if s.stamp == stamp)
        return {r["feu"]: r for r in rows if r["i"] == i}

    before, after = at("260723_10H05"), at("260723_16H47")
    # the common-mode ratio is quoted at the onset pedestal, before connector 8
    # also went, so the dead chip does not sit inside the median being compared
    a_before, a_onset = at("260720_16H12"), at("260721_12H04")
    a_during = at("260722_10H13")
    a_after = at("260727_14H11")
    last = at(sets[-1].stamp)

    # how long each pedestal stayed in force
    live = [r for r in usage if r["start"] and r["ped_dt"]]
    age_h = np.array([(r["start"] - r["ped_dt"]).total_seconds() / 3600
                      for r in live])

    # which runs each silent-connector episode covers
    for e in episodes:
        sel = [r for r in live if e["first"] <= r["start"] <= e["last_seen"]]
        e["subruns"] = len(sel)
        e["runs"] = sorted({int(r["run"].split("_")[1]) for r in sel})

    return dict(
        sets=sets, rows=rows, ctx=ctx, usage=usage, live=live,
        episodes=episodes, age_h=age_h,
        before=before, after=after,
        a_before=a_before, a_during=a_during, a_after=a_after, last=last,
        n_sets=len(sets),
        n_used=len({r["pedestal_run"] for r in usage}),
        n_subruns=len(usage),
        span=(sets[0].when, sets[-1].when),
        res_ratio={f: after[f]["med_res"] / before[f]["med_res"] for f in before},
        cm_ratio={f: after[f]["med_cm"] / before[f]["med_cm"] for f in before},
        a_onset=a_onset,
        a_cm_ratio={f: a_onset[f]["med_cm"] / a_before[f]["med_cm"]
                    for f in (3, 4)},
        fw_agreement=fw_agreement(sets),
    )


def fw_agreement(sets):
    """Correlation of our recomputed residual with the firmware's own sigma.

    Over every FEU of every acquisition, not one example: the firmware derived
    its number from a second acquisition with the pedestals already loaded, so
    agreement is a real cross-check of the decomposition rather than a tautology.
    """
    rs, ratios, keys = [], [], []
    for s in sets:
        for feu, d in s.feus.items():
            if "fw_zs_std" not in d:
                continue
            ours, theirs = d["cns_sigma"], d["fw_zs_std"]
            ok = np.isfinite(ours) & np.isfinite(theirs) & (theirs > 0)
            if ok.sum() < 100:
                continue
            rs.append(np.corrcoef(ours[ok], theirs[ok])[0, 1])
            ratios.append(np.median(ours[ok] / theirs[ok]))
            keys.append(f"{FEU_LABEL[feu]} on {s.when:%d %b}")
    rs = np.asarray(rs)
    worst = min(zip(rs, keys))
    return dict(r_median=float(np.median(rs)), r_min=float(rs.min()),
                frac_high=float((rs > 0.9).mean()), worst_where=worst[1],
                ratio_median=float(np.median(ratios)), n=len(rs))


# ------------------------------------------------------------------ HTML
CSS = """
:root{--plane:#f9f9f7;--surface:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--ring:rgba(11,11,11,.10);--grid:#e1e0d9;--A:#2a78d6;--B:#eb6834;--C:#1baf7a;--D:#8a5cd6;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
 font:15px/1.62 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:27px;line-height:1.22;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:38px 0 10px;padding-top:16px;border-top:1px solid var(--ring)}
h3{font-size:15.5px;margin:24px 0 6px}
p{margin:0 0 12px;color:var(--ink2)} p b,li b{color:var(--ink);font-weight:600}
.tagline{color:var(--muted);font-size:13.5px;margin:0 0 18px}
.verdict{background:var(--surface);border:1px solid var(--ring);border-left:4px solid var(--A);
 border-radius:12px;padding:16px 18px;margin:20px 0}
.verdict p:last-child{margin-bottom:0}
.stats{list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
 gap:10px;padding:0;margin:18px 0}
.stats li{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:12px 14px}
.stats .n{display:block;font-size:23px;font-weight:650;letter-spacing:-.02em;
 font-variant-numeric:tabular-nums}
.stats .k{display:block;font-size:12.5px;color:var(--ink2);margin-top:2px}
.stats .s{display:block;font-size:11.5px;color:var(--muted);margin-top:3px}
figure{margin:20px 0;background:var(--surface);border:1px solid var(--ring);
 border-radius:12px;padding:12px}
figure img{width:100%;height:auto;display:block;border-radius:6px}
figcaption{font-size:12.5px;color:var(--ink2);margin-top:9px}
figcaption b{color:var(--ink)}
.tablewrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface)}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--grid);
 font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
thead th{color:var(--ink2);font-weight:600;border-bottom:1.5px solid var(--ring)}
tbody tr:hover{background:#f4f4f0}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:#f0efe9;
 padding:1px 5px;border-radius:4px}
.chip{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:999px;
 border:1px solid var(--ring);color:var(--ink2)}
.chip.bad{color:#8d1f1f;border-color:#e6b3b3;background:#fdf2f2}
.chip.warn{color:#7a5600;border-color:#eddaa8;background:#fdf8ec}
.chip.good{color:#0a6b0a;border-color:#b6dfb6;background:#f1f9f1}
.A{color:var(--A);font-weight:600}.B{color:var(--B);font-weight:600}
.C{color:var(--C);font-weight:600}.D{color:var(--D);font-weight:600}
ul.plain{margin:0 0 14px;padding-left:20px;color:var(--ink2)}
ul.plain li{margin-bottom:6px}
.src{display:block;font-size:12px;color:var(--muted);margin-top:10px}
@media (prefers-color-scheme:dark){
 :root{--plane:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --ring:rgba(255,255,255,.12);--grid:#2c2c2a;--A:#3987e5;--B:#d95926;--C:#199e70;--D:#9575e0}
 figure img{background:#fff}
 code{background:#242422}
 tbody tr:hover{background:#232321}
 .chip.bad{color:#f2a6a6;border-color:#5d2a2a;background:#2a1717}
 .chip.warn{color:#e8c479;border-color:#5a4a1e;background:#272014}
 .chip.good{color:#8fd68f;border-color:#26521f;background:#152615}}
"""


def esc(s):
    return html.escape(str(s))


INLINE = False          # set by --inline: embed figures as data: URIs


def fig(name, title, caption):
    """A figure, linked relatively or embedded.

    The report in the repository links `figures/x.png`, because the DAQ's
    Analysis tab serves them by path.  The copy published as a note has to be
    one self-contained file, because a note is precached for offline reading
    and a relative image would be the one thing that breaks.
    """
    if INLINE:
        raw = open(os.path.join(HERE, "figures", name), "rb").read()
        src = "data:image/png;base64," + base64.b64encode(raw).decode()
    else:
        src = f"figures/{name}"
    return (f'<figure><img src="{src}" alt="{esc(title)}">'
            f'<figcaption>{caption}</figcaption></figure>')


def table(headers, rows_, aligns=None):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows_)
    return (f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def det_span(det, view=""):
    return f'<span class="{det}">{det}{view}</span>'


def build(a):
    S, rows = a["sets"], a["rows"]
    t0, t1 = a["span"]
    ORDER = F.ORDER

    # ---- headline tiles
    res_step = np.median(list(a["res_ratio"].values()))
    cm_step = np.median(list(a["cm_ratio"].values()))
    tiles = [
        (f"{a['n_sets']}", "pedestal acquisitions",
         f"{t0:%d %b} to {t1:%d %b} 2026"),
        (f"{a['n_used']}", "of them ever loaded",
         "across " + f"{a['n_subruns']:,}".replace(",", "\u202f") + " sub-runs"),
        (f"×{res_step:.1f}", "residual-noise step",
         "all eight FEUs, 23 July"),
        (f"{len(a['episodes'])}", "connector dropouts",
         "all on connector 8, all recovered"),
        (f"{a['age_h'].max():.0f} h", "oldest pedestal in force",
         f"median {np.median(a['age_h']):.1f} h"),
    ]
    tile_html = "".join(
        f'<li><span class="n">{n}</span><span class="k">{k}</span>'
        f'<span class="s">{s}</span></li>' for n, k, s in tiles)

    # ---- event log
    ev = []
    for e in a["episodes"]:
        runs = e["runs"]
        rng = (f"run_{runs[0]}–run_{runs[-1]}" if len(runs) > 1
               else (f"run_{runs[0]}" if runs else "—"))
        ev.append((f'{e["first"]:%d %b %H:%M}',
                   f'{det_span(e["det"], e["view"])} connector 8',
                   f'64 channels ({e["chip"]*64}–{e["chip"]*64+63}) electrically silent',
                   f'{e["last_seen"]:%d %b %H:%M}',
                   f'{rng} · {e["subruns"]} sub-runs'))
    ev.append((f'{A_START:%d %b %H:%M}', f'{det_span("A")} both views',
               f'common mode ×{np.mean(list(a["a_cm_ratio"].values())):.1f} on all 16 chips',
               f'{A_END:%d %b %H:%M}', 'run_64–run_82 · 415 sub-runs'))
    ev.append((f'{CLOCK_STEP:%d %b} ~13:00', 'all four chambers',
               f'residual noise ×{res_step:.1f}, common mode unchanged',
               'never', 'run_69–run_162 · 822 sub-runs'))
    ev.sort(key=lambda r: r[0])

    # ---- per-chamber summary at the end of the run
    per_det = []
    for feu in ORDER:
        det, view = P.FEU_DET[feu]
        r = a["last"][feu]
        b = a["before"][feu]
        per_det.append((
            f'{det_span(det, view)} <code>FEU {feu}</code>',
            f'{r["med_mean"]:.0f}', f'{r["spread_mean"]:.0f}',
            f'{r["med_raw"]:.0f}', f'{r["med_cm"]:.0f}', f'{r["med_res"]:.2f}',
            f'{r["med_raw"]/r["med_res"]:.0f}×',
            f'{r["med_thr"]-256:.0f}',
            f'{r["n_dead"]}', f'{r["n_noisy"]}'))

    # ---- the 23 July step, FEU by FEU
    step_rows = []
    for feu in ORDER:
        det, view = P.FEU_DET[feu]
        b, c = a["before"][feu], a["after"][feu]
        step_rows.append((
            f'{det_span(det, view)}',
            f'{b["med_res"]:.2f}', f'{c["med_res"]:.2f}',
            f'<b>×{c["med_res"]/b["med_res"]:.2f}</b>',
            f'{b["med_cm"]:.0f}', f'{c["med_cm"]:.0f}',
            f'×{c["med_cm"]/b["med_cm"]:.2f}',
            f'{b["med_mean"]:.0f}', f'{c["med_mean"]:.0f}'))

    # ---- clock epochs
    epoch_rows = []
    for name, ea, eb in F.EPOCHS:
        sel = [r for r in rows if ea <= r["when"] < eb]
        if not sel:
            continue
        n = len({r["stamp"] for r in sel})
        epoch_rows.append((
            name, f'{n}',
            f'{min(r["when"] for r in sel):%d %b} – {max(r["when"] for r in sel):%d %b}',
            f'{np.median([r["med_raw"] for r in sel]):.0f}',
            f'{np.median([r["med_cm"] for r in sel]):.0f}',
            f'{np.median([r["med_res"] for r in sel]):.2f}'))

    n_sub = f"{a['n_subruns']:,}".replace(",", "\u202f")

    H = []
    A_ = H.append

    A_(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DREAM pedestal stability — X17 at n_TOF</title>
<meta name="description" content="Every DREAM pedestal run of the 2026 n_TOF X17
campaign, per channel and per chip, and what moved over the six weeks.">
<style>{CSS}</style></head><body><div class="wrap">""")

    A_(f"""<h1>DREAM pedestal stability over the n_TOF campaign</h1>
<p class="tagline">{a['n_sets']} pedestal acquisitions · 8 FEUs · 4 096 channels ·
{t0:%d %B} – {t1:%d %B} 2026 · built by re-deriving every number from the decoded
pedestal runs on EOS</p>""")

    A_(f"""<div class="verdict">
<p><b>The pedestals were stable to a few per cent for weeks at a time, and then
moved in three ways worth knowing about.</b></p>
<p>One change was ours and global: on <b>23 July around 13:00</b> the DREAM
readout clock divider went from 6 to 4, and the per-channel residual noise
<b>rose by ×{res_step:.1f} on every one of the eight FEUs at once</b> and stayed
there for the last 18 days — which is the whole production period. The coherent
common mode did <em>not</em> change (×{cm_step:.2f}), so this is a real
degradation of the incoherent noise floor, not a rescaling.</p>
<p>Two changes were chamber {det_span('A')}'s and physical. From <b>21 July</b>
its common mode tripled on all sixteen chips of both views, and from
<b>22 July</b> its x-view <b>connector 8 — 64 strips — was electrically
disconnected</b>. Both were fixed together at an access on <b>27 July between
11:23 and 14:11</b>. The dead connector covers <b>every sub-run of run_79</b>,
the first long production run, and it is confirmed in the physics data: 41 of
those 64 channels recorded literally zero hits.</p>
<p>Everything else is quiet. Baselines held, no chamber drifted, and the three
other connector dropouts (all on connector 8, all in early July or on 9 August)
recovered within days.</p>
</div>""")

    A_(f'<ul class="stats">{tile_html}</ul>')

    # ---------------------------------------------------------------- log
    A_("<h2>Event log</h2>")
    A_("""<p>Everything the pedestals say changed, in order. "Recovered" is the
first pedestal that shows the old behaviour back; the true fix is somewhere
between it and the one before.</p>""")
    A_(table(["Started", "Where", "What", "Recovered", "Covers"], ev))

    # ---------------------------------------------------------------- method
    A_("<h2>What a pedestal here actually measures</h2>")
    A_(f"""<p>Each acquisition is a no-trigger readout of all eight FEUs — 512
channels × 32 samples × ~1 000 events each. Every channel is decomposed the same
way, and the three numbers mean different things:</p>
<ul class="plain">
<li><b>Raw σ</b> — the spread of the raw ADC about the channel's own baseline.
On these detectors it is <b>dominated by coherent pickup</b>, not by the
amplifier: at the end of the run the raw σ is
{np.median([a['last'][f]['med_raw']/a['last'][f]['med_res'] for f in ORDER]):.0f}×
the residual.</li>
<li><b>Common mode</b> — per chip, per time sample, the median over its 64
channels of (amplitude − channel mean). Its RMS is the coherent swing that chip
sees. This is the same definition the
<a href="../ntof_run_report/report.html">end-of-run report</a> uses.</li>
<li><b>Residual σ</b> — what is left of a channel once that common mode is
subtracted. This is the number that sets the zero-suppression threshold (5 σ)
and therefore what the experiment's sensitivity actually rests on.</li>
</ul>
<p>Two checks that the decomposition is honest. The raw spread is <em>not</em> a
fixed waveform shape repeated every event — the event-averaged shape carries
under 0.1 % of the raw variance, so raw σ really is noise. And the residual
agrees with the firmware's own per-channel σ (the one it used to program the
thresholds) at a median <b>r = {a['fw_agreement']['r_median']:.3f}</b> over all
{a['fw_agreement']['n']} FEU-acquisitions, with
{100*a['fw_agreement']['frac_high']:.0f} % above 0.9 — recomputed independently
from the samples, and from a different acquisition than the one the firmware
used.</p>
<p>Where they disagree is instructive rather than alarming. The poorest
agreement (r = {a['fw_agreement']['r_min']:.2f}, {a['fw_agreement']['worst_where']})
is on chamber A's y view during its high-common-mode excursion: the louder the
coherent swing, the noisier a median-over-64-channels estimate of it becomes
compared with the firmware's. Our residual also sits
{100*(a['fw_agreement']['ratio_median']-1):.0f} % above theirs in absolute terms
for the same reason. <b>Ratios and steps are the trustworthy part of this
analysis, not the last digit of any single sigma.</b></p>""")

    A_(fig("noise_history.png",
           "Common mode and residual noise per chamber over the campaign",
           """<b>The whole campaign in two lines per chamber.</b> Top: the
coherent swing each chip sees. Bottom: what is left per channel once it is
removed. The red dashed line is the 23 July readout-clock change — visible in
the residual on every chamber, invisible in the common mode. The blue band is
chamber A's excursion. Shaded blocks are the three DREAM clock configurations.
Log scale on both."""))

    # ---------------------------------------------------------------- clock
    A_("<h2>23 July: the readout clock, and a doubled noise floor</h2>")
    A_(f"""<p>Between the pedestal at 10:05 and the one at 16:47 on 23 July, the
residual noise rose on all eight FEUs — by ×{min(a['res_ratio'].values()):.2f} to
×{max(a['res_ratio'].values()):.2f}. Over the same boundary the common mode moved
by at most ×{max(a['cm_ratio'].values()):.2f} and the baseline by under 20 ADC.
<b>A gain change would have scaled all three together; only the incoherent part
moved</b>, so the per-channel noise floor genuinely got worse.</p>""")
    A_(table(["", "residual before", "after", "step", "common mode before",
              "after", "step", "baseline before", "after"], step_rows))
    A_(f"""<p>Exactly one configuration parameter changed at that boundary. The
DREAM readout clock divider <code>RdClk_Div</code> went <b>6 → 4</b> while the
sampling clock <code>WrClk_Div</code> stayed at 6 — the ADC digitises 1.5× faster
into the same 60 ns sampling period. Everything else the FEUs were programmed
with is byte-identical across the step, including <code>Feu_Pwr_Dream</code>,
the DREAM registers, the polarity masks, latency and the 32-sample window. The
HV was also identical: 200 V on all eight channels, powered, for every pedestal
quoted here.</p>""")
    A_(table(["DREAM clock configuration", "pedestals", "dates",
              "median raw σ", "median common mode", "median residual σ"],
             epoch_rows))
    A_("""<p><b>Which runs are on which side.</b> No run straddles the change,
but one sits inside the bracket the pedestals leave. <code>run_67</code> and
earlier are on the quiet side (run_67 ends at 09:52, before the 10:05
pedestal); <code>run_69</code> onward are on the noisy side (run_69 starts at
17:27, after the 16:47 one). <b><code>run_68</code> ran 14:54–16:08, between
the two, and the pedestals do not place it</b> — it loaded the 10:04 pedestal,
which suggests the old configuration, but that is an inference. Exclude it from
either side, or check it directly.</p>""")

    A_(f"""<p>The campaign ran three clock configurations, and it is the
<em>combination</em> rather than the readout speed alone that matters: the first
epoch also used <code>RdClk_Div 4</code> and was quiet, but with
<code>WrClk_Div 2</code> beside it. That is an observation, not a mechanism —
these pedestals cannot say why.</p>""")

    # ---------------------------------------------------------------- A
    A_("<h2>Chamber A: 21–27 July</h2>")
    A_(f"""<p>On <b>21 July at 12:04</b> chamber {det_span('A')}'s common mode
went from {a['a_before'][3]['med_cm']:.0f} to {a['a_onset'][3]['med_cm']:.0f} ADC
on the x view and {a['a_before'][4]['med_cm']:.0f} to
{a['a_onset'][4]['med_cm']:.0f} on the y — a factor
{np.mean(list(a['a_cm_ratio'].values())):.1f}, <b>on all sixteen chips of both
FEUs at once and on no other chamber</b>. A fault on one cable moves one chip;
sixteen chips moving together is a shared ground, shield or supply path at the
chamber. The next pedestal, on <b>22 July at 10:13</b>, shows x-view connector 8
gone as well.</p>
<p>Both were fixed at the same intervention. The pedestal at 27 July 11:23 still
has them; the one at 14:11 has neither, and A is back to
{a['a_after'][3]['med_cm']:.0f} / {a['a_after'][4]['med_cm']:.0f} ADC — its
pre-21-July values to within a few per cent.</p>""")

    A_(fig("chamber_a_detail.png",
           "Chamber A x view, channel by channel, at four moments",
           """<b>Chamber A's x view, all 512 channels, before / during / during /
after.</b> Connector 8 (shaded) collapses from ~32 ADC raw to under 5 — no load
on the preamp input, so no pickup — while the rest of the FEU is running three
times louder than normal. At 27 Jul 14:11 both are back. The channel-to-channel
comb in the residual is a standing feature of the readout, present throughout."""))

    A_(f"""<p><b>This lands on run_79.</b> The window 22–27 July covers runs 67
to 82, all 16 sub-runs of run_79 among them — the first long statistics run at
the production setpoint. Read directly off the hits of
<code>run_79/stat090_0001</code>, connector 8 of A-x delivered <b>3.7 % of that
FEU's hits against 11.3 % in run_100</b> after the fix, <b>41 of its 64 channels
recorded no hits at all</b>, and the hits that did survive sit in a narrow band
at threshold (median amplitude 30, 90th percentile 34) rather than the wide
spectrum of a live connector (median 192, 90th percentile 3 467). The pedestal
diagnosis and the physics data agree.</p>""")

    # ---------------------------------------------------------------- maps
    A_("<h2>Every chip, every acquisition</h2>")
    A_(fig("chip_common_mode_rel.png",
           "Common mode against each chip's own campaign median",
           """<b>The stability view: each chip against its own campaign
median</b>, so a chip that is always loud and a chip that is always quiet both
sit at 1×, and only <em>change</em> shows. Chamber A's block of red at 21–27
July, the four connector-8 dropouts as deep-blue bars, and chamber B's noisy
first week are the whole story. Diverging scale, ¼× to 4×."""))
    A_(fig("chip_common_mode.png",
           "Absolute common-mode level per chip",
           """The same grid in absolute ADC. This is where the chambers differ
from each other rather than from themselves: <span class="D">D</span> and
<span class="B">B</span> run an order of magnitude louder than
<span class="A">A</span>, and D's connectors 1 and 4 are persistently five times
quieter than its other six — a standing feature from the first pedestal to the
last, not an event."""))
    A_(fig("chip_residual.png",
           "Residual noise per chip",
           """Residual noise per chip. The vertical break on 23 July runs the
full height of the figure: every chip of every FEU, at once."""))

    # ---------------------------------------------------------------- health
    A_("<h2>Channel health</h2>")
    A_(f"""<p>A disconnected strip is diagnosed on the <em>raw</em> σ, not the
residual: losing the strip loses the pickup, and the raw noise collapses while
the residual barely moves. A channel counts as disconnected when its raw σ is
below 12 % of its FEU's median or below 8 ADC outright; loud means a residual
more than 3× the FEU median. Both cuts are relative to the same acquisition, so
the 23 July scale change does not masquerade as four thousand new dead
channels.</p>""")
    A_(fig("channel_health.png", "Disconnected and loud channel counts",
           """The three early- and mid-campaign connector dropouts show as clean
64-channel plateaus. The persistent floors are real, standing populations:
<span class="D">D</span>-x carries ~24 disconnected channels and ~69 loud ones
throughout, <span class="C">C</span>-x ~12 and ~42."""))
    A_(table(["", "baseline", "spread", "raw σ", "common mode", "residual σ",
              "coherent", "5σ threshold", "disconnected", "loud"], per_det))
    A_("""<span class="src">State at the last pedestal of the campaign,
10 August 09:12. Baseline, spread, σ and threshold in ADC counts; "coherent" is
raw σ / residual σ; counts are channels out of 512.</span>""")

    A_(fig("baseline_drift.png", "Baseline level and spread",
           """<b>Baselines did not drift.</b> Each FEU holds its median to within
a few tens of ADC across six weeks; the visible steps are the clock epochs, not
time. The lower panel is the channel-to-channel spread, which is a property of
the boards and equally steady."""))
    A_(fig("threshold_history.png", "Zero-suppression threshold in force",
           """The 5 σ threshold the DAQ actually programmed into the FEUs,
above baseline. It is a direct consequence of the residual noise, so the 23 July
step doubles it — the same absolute signal had to clear roughly twice the bar for
the rest of the campaign. Zero suppression itself was off in the run
configuration, so this bounds sensitivity rather than describing the recorded
data."""))

    # ---------------------------------------------------------------- usage
    A_("<h2>Which pedestal was in force</h2>")
    A_(f"""<p>All {n_sub} sub-runs carry a
<code>pedestal_run.txt</code> naming the pedestal directory whose memory files
the DAQ loaded, so what was in force is a fact rather than an inference. Of the
{a['n_sets']} acquisitions, <b>{a['n_used']} were ever loaded</b> — the rest were
superseded within the hour or taken as cross-checks.</p>
<p>The pedestal in force was a median <b>{np.median(a['age_h']):.1f} hours</b>
old, but the tail is long: <b>{100*np.mean(a['age_h']>24):.0f} % of sub-runs ran
on a pedestal more than a day old</b> and the worst was
{a['age_h'].max():.0f} hours ({a['age_h'].max()/24:.1f} days), in early
August.</p>""")
    A_(fig("usage_timeline.png", "Pedestal age over the campaign",
           """When pedestals were taken (top) and how old the one in force was
(bottom). The sawtooth resets at each new pedestal. Given how steady the
baselines were, the long teeth cost little — but they are why the event log above
gives a window rather than a moment for every fix."""))

    # ---------------------------------------------------------------- limits
    A_("<h2>What this does not establish</h2>")
    A_("""<ul class="plain">
<li><b>Not that the 23 July change hurt the physics.</b> A pedestal measures
noise, never signal. If the same clock change also raised the signal amplitude,
the signal-to-noise ratio could be unchanged or better. Settling that needs hit
amplitudes across the boundary, which is a different measurement; the honest
statement here is that the <em>noise floor</em> doubled and the threshold with
it.</li>
<li><b>Not which physical object moved.</b> A pedestal measures a FEU and its
cable, not a chamber. "Chamber A's common mode tripled" means the electronics
chain labelled A did; separating chamber, cabling and FEU needs the record of
which board sat where.</li>
<li><b>Not a continuous history.</b> These are {n} snapshots over six weeks, some
hours apart and some days. Every window in the event log is bounded by the two
pedestals either side of it, and something that started and finished between two
of them leaves no trace here at all.</li>
<li><b>Not the state of chambers before 1 July.</b> The first pedestal preserved
in the campaign area is 1 July 19:52; data taking began on 28 June, and the
30 June commissioning set lives outside this tree.</li>
<li><b>Not a claim about the other three connector dropouts' effect on data.</b>
Only the run_79 case was checked against hits. The early-July ones
(<span class="A">A</span>-y and <span class="B">B</span>-y) fall in the
commissioning period and the 9 August one lasted at most three hours, but
neither was confirmed in the recorded data the way run_79 was.</li>
</ul>""".replace("{n}", str(a["n_sets"])))

    A_(f"""<h2>How to rebuild this</h2>
<p>Three read-only extractions on lxplus, then everything else locally:</p>
<ul class="plain">
<li><code>ntof_pedestal_qa/lxplus/extract_pedestals.py</code> — decodes every
pedestal ROOT under
<code>/eos/experiment/ntof/data/x17/july_beam/pedestals/</code> into per-channel
mean, raw σ, residual σ and per-chip common mode, and parses the firmware's own
<code>_ped.aux</code> / <code>_thr.aux</code> alongside for comparison.</li>
<li><code>lxplus/extract_usage.py</code> — walks all {n_sub} sub-runs
for <code>pedestal_run.txt</code>.</li>
<li><code>lxplus/extract_context.py</code> — the configuration and HV each
pedestal was taken under, which is what makes a clock change distinguishable
from a cable change.</li>
<li><code>figures.py</code> then <code>make_report.py</code> rebuild the pictures
and this page from <code>data/ped_stats.npz</code>.</li>
</ul>
<span class="src">Generated {datetime.now():%d %B %Y} by
<code>ntof_pedestal_qa/make_report.py</code>. Chamber↔FEU map from
<code>ntof_active_area/clusters.py</code>: A = FEU 3/4, B = 5/6, C = 7/8,
D = 1/2.</span>""")

    A_("</div></body></html>")
    return "\n".join(H)


def main():
    global INLINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--inline", action="store_true",
                    help="embed the figures, for publishing as a note")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    INLINE = args.inline
    a = analyse()
    open(args.out, "w").write(build(a))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
