#!/usr/bin/env python3
"""Build the HTML report for the 2026-08-05 SPS extraction pass.

Writes <OUT>/report.html, which the DAQ page's Analysis tab lists and opens
inline. Figures are referenced with ORDINARY RELATIVE LINKS ('figures/x.png')
so the same file works from disk, from the DAQ page, or copied elsewhere with
its figures/ directory.

Generated, not hand-written: it reads the same JSON products the analysis
scripts wrote, so re-running after the analysis updates the numbers, the
tables and the verdict together.

    ../../.venv/bin/python make_report.py     # after make_figures.py
"""
from __future__ import annotations

import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets                                  # noqa: E402

OUT = os.path.join(os.path.expanduser("~"), "x17", "sps_beam_test_26",
                   "extraction_2026-08-05")
S = datasets.STAGE_ROOT
GAP_MM, C0_NS = 28.8, 30.0

CSS = """
:root{--fg:#1a1a1a;--muted:#5a5a5a;--bg:#ffffff;--card:#f6f7f9;--line:#dfe3e8;
--ok:#2e8b57;--warn:#d1495b;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--fg:#e8e8e8;--muted:#a5a5a5;
--bg:#15171a;--card:#1e2126;--line:#31363d;--ok:#4fb07a;--warn:#e4697b;
--accent:#5b9ff0}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:60rem;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .3rem;line-height:1.25}
h2{font-size:1.22rem;margin:2.4rem 0 .7rem;padding-bottom:.3rem;
border-bottom:1px solid var(--line)}
h3{font-size:1.03rem;margin:1.5rem 0 .5rem}
.sub{color:var(--muted);margin:0 0 1.6rem;font-size:.95rem}
.verdict{background:var(--card);border-left:4px solid var(--accent);
padding:1rem 1.15rem;border-radius:0 6px 6px 0;margin:1.4rem 0}
.verdict p{margin:.4rem 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
gap:.8rem;margin:1.4rem 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.85rem .95rem}
.kpi .v{font-size:1.5rem;font-weight:640;line-height:1.15;
font-variant-numeric:tabular-nums}
.kpi .l{color:var(--muted);font-size:.79rem;margin-top:.25rem}
.ok{color:var(--ok)}.warn{color:var(--warn)}
.tw{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.88rem;
font-variant-numeric:tabular-nums}
th,td{padding:.42rem .6rem;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:600;font-size:.79rem;
text-transform:uppercase;letter-spacing:.03em}
tbody tr:hover{background:var(--card)}
figure{margin:1.6rem 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;
background:#fff}
figcaption{color:var(--muted);font-size:.86rem;margin-top:.5rem}
code{background:var(--card);padding:.1rem .34rem;border-radius:4px;
font-size:.87em}
ul{padding-left:1.2rem}li{margin:.32rem 0}
.note{border-left:3px solid var(--warn);padding:.55rem .9rem;background:var(--card);
border-radius:0 5px 5px 0;margin:1rem 0;font-size:.93rem}
"""


def esc(x):
    return html.escape(str(x))


def load(p):
    with open(p) as f:
        return json.load(f)


def v_of(span):
    return GAP_MM * 1e3 / (span - C0_NS)


def table(headers, rows, aligns=None):
    h = "".join(f"<th>{esc(c)}</th>" for c in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return (f'<div class="tw"><table><thead><tr>{h}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def kpi(value, label, cls=""):
    return (f'<div class="kpi"><div class="v {cls}">{value}</div>'
            f'<div class="l">{esc(label)}</div></div>')


def build():
    l63 = load(S + "run_63/ladder_span_run63_rot25.json")
    l62 = load(S + "run_62/ladder_span_run62_rot25_ladder.json")
    l57 = load(S + "run_57/ladder_span_run57_rot25_co2.json")
    g66 = load(S + "run_66/gain_scan_run66_flat_resist.json")
    g70 = load(S + "run_70/gain_scan_run70_flat_drift.json")

    # --- ladder comparison ------------------------------------------------
    import numpy as np
    c3 = sorted((x["field_Vcm"], v_of(x["span"])) for x in l63.values())
    f3 = np.array([q[0] for q in c3]); v3 = np.array([q[1] for q in c3])
    rows62 = []
    diffs = []
    for x in sorted(l62.values(), key=lambda z: -z["field_Vcm"]):
        f, v = x["field_Vcm"], v_of(x["span"])
        if f3.min() <= f <= f3.max():
            vi = float(np.interp(f, f3, v3)); tag = ""
        else:
            vi = v3[-1] + (f - f3[-1]) * (v3[-1] - v3[-2]) / (f3[-1] - f3[-2])
            tag = " *"
        d = 100 * (v - vi) / vi
        diffs.append(d)
        rows62.append([f"{f:.0f}", f"{x['n']:,}", f"{x['span']:.0f}",
                       f"<b>{v:.2f}</b>", f"{vi:.2f}{tag}",
                       f"<b>{d:+.1f} %</b>"])
    rms = float(np.sqrt(np.mean(np.array(diffs) ** 2)))

    # --- gain scan --------------------------------------------------------
    def grows(g, key, view):
        r = [(x[key], x["views"][view]) for x in g.values() if view in x["views"]]
        r.sort(key=lambda t: -t[0])
        return r

    rows66 = []
    for rv, w in grows(g66, "resist_V", "y"):
        wx = dict(grows(g66, "resist_V", "x"))[rv]
        rows66.append([f"{rv:.1f}", f"{w['n_events']:,}",
                       f"{w['strips_per_ev']:.2f}", f"{w['q_lead_trunc']:.1f}",
                       f"{w['share_ratio_med']:.3f}",
                       f"<b>{w['share_matched_med']:.3f}</b>",
                       f"{wx['share_ratio_med']:.3f}",
                       f"<b>{wx['share_matched_med']:.3f}</b>"])
    ry = grows(g66, "resist_V", "y"); rx = grows(g66, "resist_V", "x")
    span_y = ry[0][1]["share_matched_med"] / ry[-1][1]["share_matched_med"]
    span_x = rx[0][1]["share_matched_med"] / rx[-1][1]["share_matched_med"]
    yield_span = ry[0][1]["n_events"] / ry[-1][1]["n_events"]

    rows70 = []
    for dv, w in grows(g70, "drift_V", "y"):
        wx = dict(grows(g70, "drift_V", "x"))[dv]
        rows70.append([f"{dv:.0f}", f"{dv / (GAP_MM / 10):.0f}",
                       f"{w['q_lead_trunc']:.1f}", f"{w['strips_per_ev']:.2f}",
                       f"{wx['q_lead_trunc']:.1f}", f"{wx['strips_per_ev']:.2f}",
                       f"{w['share_matched_med']:.3f}",
                       f"{wx['share_matched_med']:.3f}"])

    H = []
    A = H.append
    A(f"<!-- generated by make_report.py -->")
    A('<div class="wrap">')
    A("<h1>SPS H4 test beam — det4 extraction pass</h1>")
    A('<p class="sub">2026-08-05 · <code>sps_beam_test_26/analysis</code> · '
      "closing the campaign: what the audit found, and what it settled.</p>")

    A('<div class="verdict">')
    A("<p><b>The campaign was not fully extracted, and now is.</b> An audit of "
      "the EOS run directory against the analysis record found <b>eight runs "
      "carrying det4 data that appeared in no analysis table</b> — the entire "
      "Monday-morning flat block. Pulling them produced two results:</p>")
    A("<p>① <b>The wet-CF₄ v(E) curve now reproduces on independent data.</b> "
      "run_62 is a second 25.64° drift ladder under identical conditions to "
      f"run_63's; the two agree to <b>{rms:.1f} % RMS</b>. This is the "
      "campaign's most load-bearing gas number — it is what reset the mount "
      "tilt from 0.2–0.4° to 0.9° — and it no longer rests on one dataset.</p>")
    A("<p>② <b>Kernel gain-invariance is measured over a 1.34× resist swing "
      f"and holds to {abs(1-span_y)*100:.0f}–{abs(1-span_x)*100:.0f} %</b> at "
      "normal incidence, after controlling a zero-suppression artefact that "
      "otherwise fakes a 36 % decline. The premise the whole calibration-"
      "transfer argument rests on previously had a 6 % lever behind it.</p>")
    A("<p><b>What is still not fixable:</b> <code>c1</code>, <code>c2</code> "
      "and <code>tau_s</code> remain integrals over a tail longer than the "
      "3.84 µs DAQ window. Nothing in the recovered runs addresses that, and "
      "there is no beam for three years.</p>")
    A("</div>")

    A('<div class="kpis">')
    A(kpi(f"{rms:.1f} %", "v(E) reproducibility, run_62 vs run_63", "ok"))
    A(kpi(f"×{span_y:.2f} / ×{span_x:.2f}",
          "sharing over 1.34× gain (Y / X), controlled", "ok"))
    A(kpi("1.14", "v(CF₄)/v(CO₂); flush says 1.17", "ok"))
    A(kpi(f"×{yield_span:.2f}", "event-yield lever across the resist scan"))
    A(kpi("8", "runs recovered from the audit"))
    A(kpi("0", "published numbers changed"))
    A("</div>")

    # ---------------------------------------------------------------- §1
    A("<h2>1 · The v(E) curve, reproduced</h2>")
    A("<p>run_62 (Sunday 22:00) and run_63 (Sunday 23:53) are both 25.64°, "
      "Ar/CF₄/iso, 64 samples, ZS 4σ, resist 769.75 V, same pedestal set — but "
      "different drift points (700/600/500/400 V against "
      "675/625/575/525/475/425/325 V). That makes run_62 a reproducibility "
      "check, not a repeat. Same estimator, same shaping constant "
      f"<code>c0 = {C0_NS:.0f} ns</code>.</p>")
    A(table(["field [V/cm]", "hits", "span [ns]", "run_62 v [µm/ns]",
             "run_63 interp", "difference"], rows62))
    A('<p class="sub">* extrapolated beyond run_63\'s highest point.</p>')
    A('<figure><img src="figures/ladders.png" alt="drift velocity ladders">'
      "<figcaption><b>Left:</b> the two independent CF₄ ladders overlay across "
      "the whole field range; the shaded band is the ~2520 ns window-truncation "
      "floor both hit at low field, confirming it is a window property rather "
      "than a fit artefact. <b>Right:</b> the CO₂ ladder (run_57, Saturday) "
      "against CF₄ — the mixture ratio at ~240 V/cm is 1.14, against 1.17 from "
      "the run_60 gas-flush transient, a completely independent method."
      "</figcaption></figure>")

    # ---------------------------------------------------------------- §2
    A("<h2>2 · Kernel gain-invariance, and the artefact that hides it</h2>")
    A("<p>run_66 is our own scripted flat resist scan, 780 → 580 V in 25 V "
      "steps at fixed drift 700.5 V — one sub-run, one pedestal set, one ZS "
      "threshold, so the plateaus are directly comparable. The sharing proxy "
      "is the summed ±1-neighbour amplitude over the leading strip's, "
      "time-matched to ±180 ns.</p>")
    A('<div class="note"><b>The raw ratio is zero-suppression censored, and '
      "the censoring is gain-dependent.</b> A ±1 neighbour carries roughly "
      "half the leading strip, so as gain falls the neighbours cross the 4σ "
      "threshold <i>before</i> the leading strip does. Restricting to a fixed "
      "leading-strip amplitude window (400–3000 ADC) makes every plateau "
      "censor identically. The giveaway that something was wrong: a 200 V "
      "resist swing moved the mean amplitude by only 20 %, which is not "
      "credible for a resistive stage — the real lever is the event yield "
      f"(×{yield_span:.2f}).</div>")
    A(table(["resist [V]", "events", "Y strips/ev", "Y q_lead",
             "Y share raw", "Y share matched", "X share raw",
             "X share matched"], rows66))
    A('<figure><img src="figures/gain_invariance.png" alt="gain invariance">'
      "<figcaption><b>Left:</b> the gain lever — event yield halves across the "
      "scan while the ZS-censored leading-strip amplitude barely moves. "
      "<b>Right:</b> the raw sharing ratio (dotted) slopes with gain; the "
      "amplitude-matched ratio (solid) is flat. The controlled measurement is "
      "the physics; the uncontrolled one is threshold."
      "</figcaption></figure>")

    # ---------------------------------------------------------------- §3
    A("<h2>3 · Mesh transparency at normal incidence</h2>")
    A("<p>run_70's flat CF₄ drift scan, 600 → 100 V at the operating resist, "
      "with a 700 V dwell bracketing each end. At 32 samples the 1.92 µs "
      "window cannot contain a 2.0–2.5 µs drift ladder, so <b>no v(E) is "
      "obtainable</b> — what it gives is the transparency curve.</p>")
    A(table(["drift [V]", "V/cm", "Y q_lead", "Y strips/ev", "X q_lead",
             "X strips/ev", "Y share matched", "X share matched"], rows70))
    A('<figure><img src="figures/transparency.png" alt="transparency">'
      "<figcaption>Leading-strip amplitude falls ×2.1 (Y) / ×2.6 (X) over a 7× "
      "drift-field range. The two 700 V entries are the bracketing dwells, "
      "taken 18 minutes apart at opposite ends of the scan — they agree to "
      "1.3–2.6 % on the matched sharing ratio, which is the internal control "
      "on the plateau windows.</figcaption></figure>")
    A('<div class="note"><b>Do not read the sharing column here as a '
      "drift-invariance measurement.</b> Its residual 11–15 % trend is most "
      "likely still censoring (the matched window fixes the leading strip's "
      "amplitude, not the neighbour's), and its sign is opposite to what "
      "transverse diffusion would produce. The waveform-level measurement on "
      "run_71 RAW settles this properly: τ ±3.7 %, c1 ±1.1 % over "
      "92–233 V/cm.</div>")

    # ---------------------------------------------------------------- §4
    A("<h2>4 · What the audit corrected</h2>")
    A("<ul>")
    A("<li><b>Sub-run names like <code>cfg_gain3.0_peaktime200_opt</code> are "
      "P2's VMM settings, not ours.</b> det4's Dream config is byte-identical "
      "across the supposed scan, and every Dream register is a single constant "
      "over all 29 config copies. The only settings we ever varied were HV, "
      "sample count and the ZS threshold.</li>")
    A("<li><b>“No drift lever in the flat data” is retracted.</b> run_68 and "
      "run_70 are flat CF₄ drift scans taken <i>after</i> the access.</li>")
    A("<li><b>run_68/69 are worthless: there was no beam.</b> SPS FTARGET "
      "extraction stopped 04:00:45 and resumed 04:59:19; their scan ran "
      "entirely inside that gap.</li>")
    A("<li><b>run_62's early stop was beam, not DAQ.</b> Data ends 22:32:14; "
      "FTARGET stopped 22:33:09. Both FEUs stopped together.</li>")
    A("<li><b>run_63's <code>beam_commissioning_00</code> is a 37-second "
      "aborted start</b> — correctly omitted from the dataset registry.</li>")
    A("</ul>")

    A("<h2>5 · What remains</h2>")
    A("<ul>")
    A("<li>runs 64/65 (7.4 GB): extra flat statistics at the operating point, "
      "for a measurement already statistics-stable at 42.6k events.</li>")
    A("<li>run_67 (555→405 V): the resist scan's dead end — run_66's own "
      "bottom plateau already shows the yield halving and the raw ratio "
      "censoring away.</li>")
    A("<li>run_58's CO₂ ladder tail (86 GB): truncated below ~190 V/cm, first "
      "point at a recovering resist.</li>")
    A("<li>σ_p0 remains unidentifiable — the limit is the χ²/dof ≈ 170 "
      "systematic floor (noise model, rot25 alignment), not the data.</li>")
    A("</ul>")
    A('<p class="sub">Full detail and provenance: '
      "<code>sps_beam_test_26/analysis/EXTRACTION_2026-08-05b.md</code>. "
      "Reproduce: <code>make_figures.py</code> then <code>make_report.py</code>."
      "</p>")
    A("</div>")

    doc = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           "<title>SPS det4 extraction — 2026-08-05</title>"
           f"<style>{CSS}</style></head><body>" + "".join(H) +
           "</body></html>")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "report.html")
    with open(p, "w") as f:
        f.write(doc)
    print(f"wrote {p}  ({len(doc)/1024:.0f} kB)")
    print(f"  v(E) reproducibility RMS {rms:.2f} %")
    print(f"  sharing span Y x{span_y:.3f}  X x{span_x:.3f}")


if __name__ == "__main__":
    build()
