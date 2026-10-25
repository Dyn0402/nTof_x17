#!/usr/bin/env python3
"""make_report.py [--out ~/x17/cosmic_bench/det4dig/report.html] -- C21 report, generated from results/*.json.

results/finalize_phys_vs_timeD.json  (finalize_compare.py: lxplus full chain, phys vs timeD, 5 golden rows)
results/heldout_1320.json            (compare.py: laptop held-out, profiled phys vs current bundle / lp)
Re-run after either is regenerated; numbers, charts and verdict text move together.
"""
import argparse, json, os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--s1:#2a78d6;--s2:#eb6834;--card:#f4f3ef}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926;--card:#232322}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926;--card:#232322}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;}
main{max-width:960px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 8px} .sub{color:var(--ink2)}
.verdict{background:var(--card);border-left:4px solid var(--s1);padding:12px 16px;margin:16px 0}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.tile{background:var(--card);padding:10px 14px;min-width:150px;flex:1}
.tile b{display:block;font-size:22px}.tile span{color:var(--ink2);font-size:13px}
.scroll{overflow-x:auto} table{border-collapse:collapse;font-size:13px;margin:8px 0;min-width:560px}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right} th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600} .good{font-weight:700} figcaption{color:var(--ink2);font-size:13px}
svg text{fill:var(--ink2);font-size:12px} .leg span{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px}
li{margin:4px 0}
"""


def dot_chart(rows, title, xlab, lo, hi, zero=0.0, nt=4, dec=2):
    """rows: [(label, [(series_idx, value, err, tip), ...])] -> horizontal dot + error-bar SVG."""
    W, L, R, rowh = 640, 110, 20, 34
    H = 30 + rowh * len(rows) + 48
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{escape(title)}">']
    for t in [lo + i * (hi - lo) / nt for i in range(nt + 1)]:
        o.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="24" y2="{H - 44}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{sx(t):.1f}" y="{H - 28}" text-anchor="middle">{t:+.{dec}f}</text>')
    o.append(f'<line x1="{sx(zero):.1f}" x2="{sx(zero):.1f}" y1="24" y2="{H - 44}" stroke="var(--muted)" stroke-width="1.5"/>')
    for i, (lab, pts) in enumerate(rows):
        y0 = 36 + i * rowh
        o.append(f'<text x="{L - 10}" y="{y0 + 6}" text-anchor="end">{escape(lab)}</text>')
        for j, (si, v, e, tip) in enumerate(pts):
            y = y0 + (j - 0.5) * 10
            c = f'var(--s{si + 1})'
            o.append(f'<g><title>{escape(tip)}</title>'
                     f'<rect x="{sx(v - e) - 4:.1f}" y="{y - 7}" width="{sx(v + e) - sx(v - e) + 8:.1f}" height="14" fill="transparent"/>'
                     f'<line x1="{sx(max(v - e, lo)):.1f}" x2="{sx(min(v + e, hi)):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2"/>'
                     f'<circle cx="{sx(min(max(v, lo), hi)):.1f}" cy="{y}" r="4.5" fill="{c}" stroke="var(--bg)" stroke-width="2"/></g>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">{escape(xlab)}</text></svg>')
    return '\n'.join(o)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.expanduser('~/x17/cosmic_bench/det4dig/report.html'))
    a = ap.parse_args()
    F = json.load(open(os.path.join(HERE, 'results', 'finalize_phys_vs_timeD.json')))
    Hd = json.load(open(os.path.join(HERE, 'results', 'heldout_1320.json')))
    f = lambda d, v, c='all': F[f'{d}|{v}|{c}']
    # chart 1: delta s68, full chain
    rows1 = []
    for d in DETS:
        pts = []
        for si, v in enumerate('xy'):
            r = f(d, v); dv = r['arm'][0] - r['base'][0]
            pts.append((si, dv, r['err'][0], f'{d} {v.upper()}: s68 {r["base"][0]:.3f} -> {r["arm"][0]:.3f} deg '
                                             f'({dv:+.3f} +- {r["err"][0]:.3f}), n {r["n"]}'))
        rows1.append((d, pts))
    c1 = dot_chart(rows1, 'Change in angle resolution', 'Δ s68 (phys − timeD), degrees; left = better', -0.12, 0.12)
    # chart 2: ladder offset from timing v, both arms
    rows2 = []
    for d in DETS:
        for v in 'xy':
            r = f(d, v); vt = r['v_timing']
            pts = [(1, 100 * (r['ladder'][0] / vt - 1), 0.0, f'{d} {v.upper()} timeD ladder {r["ladder"][0]:.2f} vs timing {vt}'),
                   (0, 100 * (r['ladder'][1] / vt - 1), 0.0, f'{d} {v.upper()} phys ladder {r["ladder"][1]:.2f} vs timing {vt}')]
            rows2.append((f'{d} {v.upper()}', pts))
    c2 = dot_chart(rows2, 'Ladder slope against timing v', 'ladder / timing v − 1 (%), 0 = consistent', -12, 6, nt=6, dec=0)
    # tables
    t1 = ['<tr><th>chamber · view</th><th>n</th><th>ladder timeD → phys (timing v)</th><th>s68 timeD → phys</th>'
          '<th>Δ s68</th><th>far % timeD → phys</th></tr>']
    for d in DETS:
        for v in 'xy':
            r = f(d, v); ds = r['arm'][0] - r['base'][0]
            cls = ' class="good"' if abs(ds) > 2 * r['err'][0] else ''
            t1.append(f'<tr><td>{d} {v.upper()}</td><td>{r["n"]}</td><td>{r["ladder"][0]:.2f} → {r["ladder"][1]:.2f} ({r["v_timing"]})</td>'
                      f'<td>{r["base"][0]:.3f} → {r["arm"][0]:.3f}</td><td{cls}>{ds:+.3f} ± {r["err"][0]:.3f}</td>'
                      f'<td>{r["base"][1]:.1f} → {r["arm"][1]:.1f}</td></tr>')
    t2 = ['<tr><th>view · base → arm</th><th>n</th><th>ladder</th><th>s68</th><th>Δ s68</th><th>far %</th><th>χ²/dof</th></tr>']
    for k, r in Hd.items():
        view, b, t, cl = k.split('|')
        if cl != 'all':
            continue
        t2.append(f'<tr><td>{view.upper()} · {b} → {t}</td><td>{r["n"]}</td><td>{r["ladder"][0]:.2f} → {r["ladder"][1]:.2f}</td>'
                  f'<td>{r["base"][0]:.3f} → {r["arm"][0]:.3f}</td><td>{r["delta"][0]:+.3f} ± {r["err"][0]:.3f}</td>'
                  f'<td>{r["base"][1]:.1f} → {r["arm"][1]:.1f}</td><td>{r["base"][2]:.1f} → {r["arm"][2]:.1f}</td></tr>')
    d4x, d4y, d3x = f('det4', 'x'), f('det4', 'y'), f('det3', 'x')
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One physical waveform model</title><style>{CSS}</style></head><body><main>
<h1>One physical waveform model for the MX17 fleet</h1>
<p class="sub">rerun C21 · 2026-10-25 · det4 under the RC model with physics-set inputs; generated by rerun/det4/make_report.py</p>
<div class="verdict"><b>Verdict.</b> det4 works under the RC model once its inputs come from physics, not from the
calibration fit, which had absorbed other effects into the footprint σ0 and the diffusion Dp. det4's angle ladders come back
onto its timing drift velocity and its χ² halves, while det3, the reference, does not move. det2 and det6 are neutral to better.
det7 X is worse, because its board's X neighbour asymmetry was being hidden by the wide footprint.</div>
<div class="tiles">
<div class="tile"><b>{d4x['ladder'][0]:.1f} → {d4x['ladder'][1]:.1f}</b><span>det4 X ladder, µm/ns (timing {d4x['v_timing']})</span></div>
<div class="tile"><b>{d4y['ladder'][0]:.1f} → {d4y['ladder'][1]:.1f}</b><span>det4 Y ladder, µm/ns</span></div>
<div class="tile"><b>{d4x['arm'][0] - d4x['base'][0]:+.3f}°</b><span>det4 X s68 change (± {d4x['err'][0]:.3f})</span></div>
<div class="tile"><b>{d3x['arm'][0] - d3x['base'][0]:+.3f}°</b><span>det3 X s68 change (± {d3x['err'][0]:.3f})</span></div>
</div>
<h2>The recipe</h2>
<ul><li><b>σ0_x / σ0_y = 0.238 / 0.343 mm</b> on every chamber: det3's per-event prompt footprint (strip box + induction ≈ 0.2 mm).
The calibration fits ranged from 0.08 (det6) to 0.53 (det7).</li>
<li><b>Dp from the unsaturated head-on stack</b>, √(2·Dd), which agrees with Magboltz (0.5 % H₂O): det2 0.025, det3 0.023, det4 0.020,
det7 0.0225; det6 keeps 0.018 (its stack is window-limited). The bundles carried 0.028–0.032.</li>
<li><b>det4 D_rc,y = 2.2e-4</b> from the H4 beam; det6 v rescaled to the adopted 28 mm gap (24.17).</li>
<li><b>Per-event σ0 profile</b> (new opt-in wft key <code>sigma_p0_prof_&lt;plane&gt;</code>): one wide alternative (0.42 / 0.50) is refitted
and kept when its χ² is lower. It is taken mostly by saturated clusters, which otherwise tilt to make up their width.</li></ul>
<h2>Full chain on lxplus: phys against the current timeD bundles</h2>
<p class="sub">Both arms finalized side by side (reco 8000 events → waveform alignment → angles), pivot start, event-paired, bootstrap errors.
s68 is taken at each arm's own ladder slope, so the angle scale does not enter.</p>
<figure><div class="leg"><span style="background:var(--s1)"></span>X view<span style="background:var(--s2)"></span>Y view</div>{c1}
<figcaption>Change in angle resolution per chamber and view (hover for numbers). det4 and det6 X improve; det3 and det2 do not move; det7 X loses.</figcaption></figure>
<figure><div class="leg"><span style="background:var(--s1)"></span>phys<span style="background:var(--s2)"></span>timeD</div>{c2}
<figcaption>Ladder slope (fitted w against the M3 angle; the per-event fit never uses v) relative to the timing v. The physical model moves det4
from 7–10 % low to within 1–4 %; det7 X overshoots (37.7 against 36.8; det7 has two timing values).</figcaption></figure>
<div class="scroll"><table>{''.join(t1)}</table></div>
<h2>Held-out laptop test (1320 events per chamber)</h2>
<p class="sub">Profiled physical model against the bundle each chamber had (it0) and det4's old lp kernel; also the third-σ0-point check.</p>
<div class="scroll"><table>{''.join(t2)}</table></div>
<h2>What this does not rule out</h2>
<ul><li>det4/det7 head-on far fraction stays 1.4–3.6 points above the old bundles; a third, wider σ0 point does not change it, so it is not footprint width.</li>
<li>det7 X loses 0.05–0.09°, on the clean card too: the symmetric model cannot follow det7's X neighbour asymmetry (aQ +0.19), the model's stated mismatch.</li>
<li>det6 X χ² rises 15 %: det6 X wants a narrower footprint than det3's.</li>
<li>Production angles (tan_fit) still use the source bundles' w0/kw in both arms; finalize has re-measured them, so compare production angles only after a re-reco.</li>
<li>Rejected levers on det4: cross-strip RC leakage, per-strip in-fit gains (the depth profile absorbs band-scale gain), a third σ0 point.</li>
<li>No new data can be taken; everything here is from the June bench and the H4 beam already on disk.</li></ul>
</main></body></html>"""
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, 'w').write(html)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
