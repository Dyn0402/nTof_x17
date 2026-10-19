"""Build dream_return_cea/report.html from the decoded pedestals.

Every number in the text is computed here from `data/ped_stats.npz`;
figures come from `figures.py`.

    ../.venv/bin/python -m dream_return_cea.make_report            # relative figure links
    ../.venv/bin/python -m dream_return_cea.make_report --embed X  # self-contained copy at X
"""

from __future__ import annotations

import argparse
import base64
import html
import os

import numpy as np

from . import figures as F
from . import pedestals as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report.html")
OUTLIER = (0.5, 2.0)          # x FEU median residual sigma


def analyse():
    rows, ctx = P.load()
    for r in rows:
        med = np.median(r["cns_sigma"])
        r["n_out"] = int(np.sum((r["cns_sigma"] < OUTLIER[0] * med) |
                                (r["cns_sigma"] > OUTLIER[1] * med)))
        r["fw_dmean"] = float(np.nanmax(np.abs(r["fw_avr"] - r["mean"])))
        r["fw_dstd"] = float(np.nanmax(np.abs(r["fw_std"] - r["raw_sigma"])))
    allc = lambda k: np.concatenate([r[k] for r in rows])
    return dict(rows=rows, ctx=ctx,
                n_ch=len(rows) * P.NCH,
                n_ev=sum(r["events"] for r in rows),
                raw=np.median(allc("raw_sigma")), cmn=np.median(allc("cns_sigma")),
                cm=np.median(allc("cm_rms")),
                ped_lo=np.min(allc("mean")), ped_hi=np.max(allc("mean")),
                spread=max(np.median(r["cns_sigma"]) for r in rows) /
                min(np.median(r["cns_sigma"]) for r in rows) - 1,
                n_out=sum(r["n_out"] for r in rows),
                all_ids=all(r["id_match"] for r in rows),
                no_loss=all(r["missing"] == 0 and r["completeness"] == 100
                            for r in rows),
                fw_ok=max(max(r["fw_dmean"], r["fw_dstd"]) for r in rows) < 0.05)


def img(name, caption, embed):
    path = os.path.join(F.FIG, name)
    src = ("data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
           if embed else f"figures/{name}")
    return (f'<figure><img src="{src}" alt="{html.escape(caption)}">'
            f"<figcaption>{caption}</figcaption></figure>")


def render(a, embed=False):
    rows, ctx = a["rows"], a["ctx"]
    day = F.date_str(rows)
    t0, t1 = rows[0]["start"], max(r["end"] for r in rows)
    ok = a["all_ids"] and a["no_loss"] and a["n_out"] == 0 and a["fw_ok"]
    verdict = ("All nine FEUs read out, every channel alive." if ok else
               "Not all checks passed — see the table.")

    trs = "".join(
        f"<tr><td>feu{r['slot']}</td><td>{r['feu_id']}</td>"
        f"<td>{'✓ ' if r['id_match'] else '✗ '}{', '.join(map(str, r['stream_ids']))}</td>"
        f"<td>{r['start']:%H:%M}</td><td>{r['end']:%H:%M:%S}</td>"
        f"<td>{r['events']}</td><td>{r['completeness']:.1f} %</td>"
        f"<td>{np.median(r['mean']):.0f}</td>"
        f"<td>{np.median(r['raw_sigma']):.2f}</td><td>{np.median(r['cns_sigma']):.2f}</td>"
        f"<td>{np.median(r['cm_rms']):.2f}</td><td>{r['n_out']}</td></tr>"
        for r in rows)
    failed = "".join(
        f"<li>{f['stamp'][7:9]}:{f['stamp'][10:12]} — feu{f['slot']}: "
        f"<code>{html.escape(f['reason'])}</code>; the retry succeeded.</li>"
        for f in ctx["failed"])

    css = """
:root{--bg:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--band:#f7f6f2;--good:#0ca30c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16161a;--ink:#ecebe6;--ink2:#b8b7b0;--muted:#8d8c86;--line:#34343a;--band:#1e1e23}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
main{max-width:1080px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.15rem;margin:34px 0 8px}
.sub{color:var(--ink2);margin:0 0 20px}
.verdict{border-left:4px solid var(--good);background:var(--band);padding:12px 16px;font-size:1.05rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--band);padding:10px 12px;border-radius:6px}.tile b{display:block;font-size:1.4rem}
.tile span{color:var(--ink2);font-size:.85rem}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:.88rem;width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600}td:first-child,th:first-child{text-align:left}
figure{margin:20px 0}img{max-width:100%;background:#fff;border-radius:4px}
figcaption{color:var(--ink2);font-size:.9rem;margin-top:6px}code{font-size:.88em}
li{margin:4px 0}
"""
    body = f"""
<h1>DREAM DAQ back at CEA — pedestal check</h1>
<p class="sub">{day}, {t0:%H:%M}–{t1:%H:%M} · MX17 DREAM readout, one pedestal run per FEU ·
source <code>~/Desktop/ped_validation_5-10-26</code></p>
<p class="verdict"><b>{verdict}</b> Each FEU was enabled on its own and took a
{ctx['samples']}-sample pedestal run; the FEU ID read back from every data stream is
the one the configuration assigns to that slot, no event or sample was lost, and the
noise is the same on all {len(rows)} cards to within {a['spread']*100:.0f} %.</p>

<div class="tiles">
<div class="tile"><b>{len(rows)} / {len(rows)}</b><span>FEUs read out, IDs match cfg</span></div>
<div class="tile"><b>{a['n_ch'] - a['n_out']:,} / {a['n_ch']:,}</b><span>channels within ½–2× their FEU's median noise</span></div>
<div class="tile"><b>{a['raw']:.2f} ADC</b><span>median raw σ</span></div>
<div class="tile"><b>{a['cmn']:.2f} ADC</b><span>median σ after common-noise subtraction</span></div>
<div class="tile"><b>{a['n_ev']:,}</b><span>pedestal events, 0 missing</span></div>
</div>

<h2>What was taken</h2>
<p>The run-control cfg (<code>ped_val.cfg</code>) lists nine FEU slots; for each run
only one <code>Sys Topo Feu N</code> line was active. A file named
<code>_feuN_</code> is therefore slot <i>N</i>, whose hardware identity is the cfg's
<code>Feu N Feu_RunCtrl_Id</code>. That identity was checked independently against
the FEU ID the decoder reads from the data headers. Clock settings as in the cfg:
<code>RdClk_Div {ctx['RdClk_Div']}</code>, <code>WrClk_Div {ctx['WrClk_Div']}</code>.
Slot 9 (ID 121) is the ninth card, outside the eight-FEU n_TOF readout.</p>
{f'<ul>{failed}</ul>' if failed else ''}

<h2>Per FEU</h2>
<div class="tw"><table>
<tr><th>slot</th><th>cfg FEU ID</th><th>ID in data</th><th>start</th><th>file closed</th>
<th>events</th><th>samples</th><th>pedestal</th><th>raw σ</th><th>sub σ</th><th>CM rms</th><th>outliers</th></tr>
{trs}</table></div>
<p class="sub">Medians over the FEU's 512 channels, in ADC. <i>sub σ</i> is the channel
noise after subtracting, per DREAM chip and per sample, the median of the chip's 64
baseline-subtracted channels; <i>CM rms</i> is the per-chip size of that common
mode. Same definitions as the n_TOF pedestal history (<code>ntof_pedestal_qa</code>).
Pedestals span {a['ped_lo']:.0f}–{a['ped_hi']:.0f} ADC. Baselines and raw σ agree
with the firmware's own <code>_ped.aux</code> to
{'better than 0.05 ADC' if a['fw_ok'] else 'NOT within 0.05 ADC'} on every channel.</p>

<h2>Figures</h2>
{img("01_all_pedestals.png", "Every channel of all nine FEUs side by side. Top: pedestal baseline. Bottom: noise before (orange) and after (blue) common-noise subtraction. The saw-tooth inside each FEU repeats every 64 channels — one DREAM chip — and is the same on every card.", embed)}
{img("02_timeline.png", "When each FEU was taken (top) and its median noise with the 5–95 % range of its channels (bottom). The red cross is the first feu9 attempt, which could not open the FEU; the retry four minutes later worked.", embed)}
{img("03_per_feu_noise.png", "Noise distribution of each FEU's 512 channels, before and after common-noise subtraction.", embed)}

<h2>What this does not show</h2>
<ul>
<li>That the analogue chain responds to charge — a pedestal run has no signal. A
pulser or source run is the test for gain and for dead preamplifier inputs that
still produce a baseline.</li>
<li>Anything about the detectors, cables or the TCM trigger path beyond the constant
pedestal trigger used here; the runs record no detector HV and these noise levels
should not be compared with the n_TOF pedestals, which were taken with chambers
attached and a different write clock.</li>
<li>All nine FEUs running together — each was read out alone.</li>
</ul>
"""
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>DREAM Pedestal Check</title><style>{css}</style></head>"
            f"<body><main>{body}</main></body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", help="also write a self-contained copy here")
    args = ap.parse_args()
    a = analyse()
    open(OUT, "w").write(render(a))
    print(OUT)
    if args.embed:
        open(args.embed, "w").write(render(a, embed=True))
        print(args.embed)


if __name__ == "__main__":
    main()
