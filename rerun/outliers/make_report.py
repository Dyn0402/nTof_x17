#!/usr/bin/env python3
"""make_report.py [--out ~/x17/cosmic_bench/outliers/report/report.html] -- the outlier study, generated from
report/results.json (summary.py) and report/figures/ (make_figures.py). Re-run both after any upstream change.
"""
import argparse, importlib.util, json, os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser('~/x17/cosmic_bench/outliers')
_spec = importlib.util.spec_from_file_location('c21', os.path.join(HERE, '..', 'det4', 'make_report.py'))
_c21 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_c21)
CSS = _c21.CSS + "img{max-width:100%;height:auto;border:1px solid var(--grid)} details{margin:8px 0}"

LABEL = {'REJECTED_OK': 'fit agrees, rejected by the competing-cluster cut',
         'OUTSIDE': 'M3 track outside the active area', 'EDGE': 'within 30 mm of an edge (field distortion / truncation)',
         'HW_KNOWN': 'known hardware (det4 dead bands, det7 X card 1)', 'MX17_DEAD': 'no charge anywhere in the view',
         'M3_ELSEWHERE': 'MX17 particle not where M3 points', 'POS_TAIL_BENT': 'bent track: M3 straight line misses',
         'POS_TAIL_MX17': 'position tail on a clean track (MX17)', 'MULTI': 'second particle; fit took the other',
         'DELTA': 'delta ray riding on the track', 'ONE_VIEW_ELSEWHERE': 'charge elsewhere in one view',
         'FIT_SAT': 'clean saturated track, fit wrong', 'FIT': 'clean track, fit wrong'}
SHOW = ['REJECTED_OK', 'OUTSIDE', 'EDGE', 'HW_KNOWN', 'MX17_DEAD', 'M3_ELSEWHERE', 'POS_TAIL_BENT', 'MULTI', 'DELTA',
        'POS_TAIL_MX17', 'ONE_VIEW_ELSEWHERE', 'FIT_SAT', 'FIT']


INLINE = False


def fig(name, cap):
    src = f'figures/{name}'
    if INLINE:                                   # self-contained copy for the notes site (offline reading)
        import base64
        src = 'data:image/png;base64,' + base64.b64encode(open(os.path.join(BASE, 'report', 'figures', name), 'rb').read()).decode()
    return f'<figure><img src="{src}" alt="{escape(cap)}"><figcaption>{cap}</figcaption></figure>'


def main():
    global INLINE
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(BASE, 'report', 'report.html'))
    ap.add_argument('--inline', action='store_true', help='embed the figures (data: URIs) for the notes site')
    a = ap.parse_args(); INLINE = a.inline
    R = json.load(open(os.path.join(BASE, 'report', 'results.json')))
    D = R['dets']; C = R['classes']; G = R['groups']
    H = {(h['det'], h['view'], h['cls']): h for h in R['headon']}
    c3 = C['det3']; g3 = G['det3']
    m3_3 = g3['reference (M3)']; bent3 = c3['cls']['POS_TAIL_BENT']['pct']
    sp = R.get('sps', {})
    spsx = sp.get('sps700_dp015', {}); b4 = sp.get('bench_det4_headon', {}); b3 = sp.get('bench_det3_headon', {})
    sx = lambda s, v: f"{s[v]['far_pct']:.1f} ± {s[v]['far_pct_ci']:.1f} %" if s and v in s else 'n/a'
    six = R['sixpt']['worst']

    o = [f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>MX17 outliers vs M3</title><meta name="description" content="Golden June bench events that disagree with M3, classified event by event: bent tracks, second particles, edge zone; det4/det7 head-on; det4 at H4."><style>{CSS}</style></head><body><main>',
         '<h1>Events that disagree with M3: what they are</h1>',
         '<p class="sub">Golden June bench rows under the physical RC model (finalize_phys, lxplus), every event looked at '
         'through its waveforms, the M3 cluster trees rebuilt from raw, and a two-chamber cross-check. Plus det4 at H4. '
         'rerun/outliers/, 2026-10-25.</p>']
    o.append('<div class="verdict"><b>Verdict.</b> '
             f'On det3, {c3["pct_outliers"]:.1f} % of M3-matched events disagree with M3 (angle &gt; 5° or position &gt; 3 mm, '
             f'or rejected for competing clusters). The M3 reference is at fault in about <b>{m3_3:.1f} %</b>, almost all of it '
             f'<b>bent tracks</b>: a muon that scatters between the M3 doublets leaves the straight 4-layer line several mm off at '
             f'the DUT with χ² still inside the cut. Showers and multi-particle events are <b>not</b> M3 failures: M3 sits on one '
             f'of the particles every time, and the MX17 fit takes the other ({g3["second particle / delta ray"]:.1f} % with delta '
             f'rays). Geometry (tracks outside the active area, the drift-field distortion within 20 mm of an X edge) is '
             f'{g3["geometry (acceptance, edge field)"]:.1f} %, MX17 response/reconstruction on clean tracks '
             f'{g3["MX17 response / reconstruction"]:.1f} %, and {g3["not a failure"]:.1f} % are correct fits removed by the '
             'competing-cluster cut.<br><br>'
             '<b>det4 / det7 head-on.</b> Their head-on far tail is explained by hardware already known: det7 X card 1 '
             '(x &lt; 199 mm, the asterisk) and det4\'s dead amplification bands, plus the edge zone. Without those, head-on far is '
             f'det4 {H[("det4","x","head-on")]["far_after_known"]:.1f} / {H[("det4","y","head-on")]["far_after_known"]:.1f} %, '
             f'det7 {H[("det7","x","head-on")]["far_after_known"]:.1f} / {H[("det7","y","head-on")]["far_after_known"]:.1f} % '
             f'(X / Y), against det3 {H[("det3","x","head-on")]["far_after_known"]:.1f} / {H[("det3","y","head-on")]["far_after_known"]:.1f} %. '
             'The +1.4–3.6 pp the physical model added on det4 is mostly low-charge events in dead bands. det7 was never '
             'at SPS. <b>det4 at H4 is not answered yet:</b> the same fitter with the bench-calibrated phys bundle gives '
             f'head-on far X {sx(spsx, "x")}, Y {sx(spsx, "y")} on run_71 (bench det4 under the same script: X {sx(b4, "x")}, '
             f'Y {sx(b4, "y")}), but the beam waveforms are not what that bundle describes (a lumpy 2 µs column in wet '
             'CF₄, a strongly one-sided delayed X copy), so this is a model failure, not a detector rate. It needs a beam '
             'bundle (re-run plan S4).</div>')
    o.append('<div class="tiles">' + ''.join(
        f'<div class="tile"><b>{v}</b><span>{k}</span></div>' for k, v in [
            ('det3 outliers (% of matched events)', f'{c3["pct_outliers"]:.1f} %'),
            ('det3: M3 reference at fault', f'{m3_3:.2f} %'),
            ('det3: second particle / delta ray', f'{g3["second particle / delta ray"]:.1f} %'),
            ('bent-track share of position tails (det3)', f'{c3["bent_f"]:.2f} ± {c3["bent_err"]:.2f}'),
            ('det3 X angle response, 5–15 mm from edge', f'k = {R["edge"]["det3|x"]["k"][1]:.2f}')]) + '</div>')

    # class table
    o.append('<h2>Every outlier, by cause (% of all M3-matched events)</h2><div class="scroll"><table><tr><th>class</th>'
             + ''.join(f'<th>{d}</th>' for d in D) + '</tr>')
    o.append('<tr><td><b>all outliers</b></td>' + ''.join(f'<td><b>{C[d]["pct_outliers"]:.2f}</b></td>' for d in D) + '</tr>')
    for k in SHOW:
        o.append(f'<tr><td>{escape(LABEL[k])} <span class="sub">({k})</span></td>'
                 + ''.join(f'<td>{C[d]["cls"].get(k, {"pct": 0})["pct"]:.2f}</td>' for d in D) + '</tr>')
    o.append('<tr><td colspan="6" class="sub">grouped:</td></tr>')
    for g in R['groups_def']:
        o.append(f'<tr><td>{escape(g)}</td>' + ''.join(f'<td>{G[d][g]:.2f}</td>' for d in D) + '</tr>')
    o.append('</table></div>')
    o.append('<p class="sub">Matched events: ' + ', '.join(f'{d} {C[d]["n_events"]}' for d in D) + '. Classes are assigned in '
             'the order of the table (first match wins), from waveform features over the full plane (charge on the M3 '
             'footprint, other clusters, cluster width, saturation), the M3 cluster trees, and the active area; every class '
             'was checked by eye on random displays (below). POS_TAIL is split into bent / MX17 by the mixture fit described '
             'in the M3 section.</p>')
    o.append(fig('classes.png', 'Outlier classes per chamber. det4 and det7 are dominated by hardware that is already known.'))

    # head-on
    o.append('<h2>det4 and det7 head-on: what the far tail is</h2><div class="scroll"><table><tr><th>chamber, view</th><th>n</th>'
             '<th>far %</th><th>outside</th><th>edge</th><th>known HW</th><th>second particle</th><th>delta</th>'
             '<th>fit (clean)</th><th>fit (sat.)</th><th><b>far after known</b></th><th>inclined, after known</th></tr>')
    for d in D:
        for v in 'xy':
            h = H[(d, v, 'head-on')]; hi = H[(d, v, 'inclined')]; b = h['by']
            o.append(f'<tr><td>{d} {v.upper()}</td><td>{h["n"]}</td><td>{h["far"]:.2f}</td><td>{b["OUTSIDE"]:.2f}</td>'
                     f'<td>{b["EDGE"]:.2f}</td><td>{b["HW_KNOWN"]:.2f}</td><td>{b["MULTI"]:.2f}</td><td>{b["DELTA"]:.2f}</td>'
                     f'<td>{b["FIT"]:.2f}</td><td>{b["FIT_SAT"]:.2f}</td><td><b>{h["far_after_known"]:.2f}</b></td>'
                     f'<td>{hi["far_after_known"]:.2f}</td></tr>')
    o.append('</table></div>')
    o.append('<ul><li><b>det7 X:</b> far share 40–70 % for x &lt; 199 mm (card 1, M1 badly seated: the asterisk), 5–10 % on '
             'card 2. On card 2 det7 behaves like det3.</li>'
             '<li><b>det4:</b> far share follows the amplification stripes in x, in <i>both</i> views (the stripes are in the '
             'amplification gap). Inside the live bands det4 head-on far is 5.9 % X / 8.7 % Y (bench), vs 3.6 / 3.2 % on det3 '
             'interior. What remains is low-gain partial columns and clean-track fit failures (FIT column).</li>'
             '<li><b>The phys increment</b> (+2.7 pp X, +3.6 pp Y head-on on det4): 52–56 % of the newly far events sit in dead '
             'bands, at 0.55× the median charge; timeD fitted them at |tan| ≈ 0.04, phys tilts them to ≈ 0.2. The narrow '
             'physical footprint makes faint, partial columns tilt. On det7 the increment is edge (23–36 %) and saturated '
             'wide clusters.</li></ul>')
    o.append(fig('farmap.png', 'Far share (|dθ| &gt; 5°) against the M3 position at the mesh. det7 X card 1 (x &lt; 199) and '
                 'det4\'s stripes in x stand out; every chamber has hot X edges.'))

    # edge
    e3 = R['edge']['det3|x']
    o.append('<h2>Edge zone: the drift field bends near the cage walls (PAPER_PLAN D4, re-measured on wft angles)</h2>')
    o.append(f'<p>det3 X: the angle response k (fitted tan / M3 tan) is {e3["k"][1]:.2f} at 5–10 mm from the X edge, '
             f'{e3["k"][2]:.2f} at 10–15, {e3["k"][3]:.2f} at 15–20 and 1.0 beyond 20 mm; far share {e3["far"][1]:.0f} / '
             f'{e3["far"][2]:.0f} / {e3["far"][3]:.0f} %. Head-on tracks there read a fake tilt of 0.10–0.15 in tan (6–9°) '
             'within 10 mm and ~0.06 at 10–20 mm, with deep charge pulled <b>inward</b> at both edges (det3 and det6 X). '
             'det3 Y shows nothing: the Y strip plane is passivated for 18 mm at each edge, so Y never sees the wall region. '
             'det2 (chamber B, degrador rings removed) has the <b>opposite</b> sign in Y (outward) and the strongest effect. '
             'This replaces the hits-era −3° (δ ≈ −0.06 tan, ≲ 25–40 mm). It is most of det3\'s head-on excess: inside 20 mm '
             'of the edges det3 head-on and inclined far rates are equal.</p>')
    o.append(fig('edge_response.png', 'Angle response k and far share against the footprint\'s distance to the edge of its own '
                 'coordinate. k &lt; 1: tracks read too vertical.'))

    # follow-ups (Dylan 10-25): edges per band, the competing-cluster events, the two-track refit
    o.append('<h2>Follow-ups: edges by band, the competing-cluster events, the two-track refit</h2>')
    o.append(fig('edgechar.png', 'Per chamber and view, by distance band of the M3 footprint to the edge of its own '
                 'coordinate (fiducial.py): far share, angle response k and the head-on inward tilt. On X, det3/det6/det7 '
                 'tilt inward over 0–15 mm, det2 is broken outward out to 30 mm, det4 shows no tilt but a rising far share '
                 'at the edge. On Y every chamber except det2 stays within a few hundredths.'))
    N = json.load(open(os.path.join(BASE, 'report', 'nearest.json')))
    o.append('<p><b>Competing-cluster events kept.</b> Taking the plausible candidate nearest the reference (within 10 mm, '
             'wide against the 0.5 mm resolution) instead of the rank-0 winner with the n_dropped cut, interior region. '
             'Events whose production track has no candidate in the window count as far.</p>'
             '<div class="scroll"><table><tr><th>chamber, view</th><th>n prod.</th><th>far prod. %</th><th>n nearest</th>'
             '<th>far nearest % (incl. no-match)</th><th>s68 prod. / nearest [°]</th></tr>')
    for k in sorted({k.rsplit('|', 1)[0] for k in N}):
        pr, ne = N.get(k + '|production', {}), N.get(k + '|nearest', {})
        if not pr or not ne:
            continue
        o.append(f"<tr><td>{escape(k.replace('|', ' '))}</td><td>{pr['n']}</td><td>{pr['far']:.2f}</td><td>{ne['n']}</td>"
                 f"<td>{ne.get('far_incl_nomatch', ne['far']):.2f}</td><td>{pr['s68']:.3f} / {ne['s68']:.3f}</td></tr>")
    o.append('</table></div><p>The far rate moves by at most 0.4 pp (det4 slightly up) and the core does not change; what the nearest choice buys is the competing-'
             'cluster events back into the sample. It uses the reference, so it is a bench convention, not a reconstruction.</p>')
    tp = os.path.join(BASE, 'report', 'twotrack_det3.json')
    if os.path.exists(tp):
        T = json.load(open(tp)); arms = [a for a in ('off', 'on', 'onF30') if a in T]
        cls = [r['cls'] for r in T['off'] if r['cls'] in ('MULTI', 'DELTA', 'FIT', 'FIT_SAT', 'EDGE', 'REJECTED_OK', 'CONTROL')]
        get = lambda a, c, k: next((r[k] for r in T[a] if r['cls'] == c), None)
        f = lambda x: '—' if x is None else f'{x:.1f}'
        o.append('<p><b>Two-track refit (det3).</b> reco_subset.py on the outliers + 300 controls in the finalize '
                 'environment; the off arm reproduces finalize_phys bit for bit on all 1 135 events. '
                 '<i>on</i> = WFT_TWO_TRACK_FIT=1 at the default thresholds (F 300, corroborated 120); <i>onF30</i> = '
                 'F 30 / 15, on the MULTI/DELTA/FIT subset and the controls. % of event-views agreeing with M3 '
                 '(|dθ| &lt; 5°, |res| &lt; 3 mm); "any" = some candidate agrees (the ceiling a reference-free choice could reach).</p>'
                 '<div class="scroll"><table><tr><th>class</th><th>n</th>' +
                 ''.join(f'<th>winner {a}</th>' for a in arms) + ''.join(f'<th>any {a}</th>' for a in arms) +
                 f'<th>split % {arms[-1]}</th><th>lowest χ²/dof (on)</th></tr>')
        for c in cls:
            o.append(f"<tr><td>{c}</td><td>{get('off', c, 'n')}</td>" + ''.join(f'<td>{f(get(a, c, "winner"))}</td>' for a in arms)
                     + ''.join(f'<td>{f(get(a, c, "any"))}</td>' for a in arms)
                     + f"<td>{f(get(arms[-1], c, 'split%'))}</td><td>{f(get('on', c, 'rule_chi2dof'))}</td></tr>")
        o.append('</table></div><p>The two-track fit splits 8 % of the MULTI views and raises their agreement with M3 '
                 'from 22 to 25 %; dropping the split threshold ten-fold (F 30) changes nothing further. The right '
                 'track is usually already among the candidates (MULTI "any" ≈ 70 %): the limit is the choice, not the '
                 'splitting. No reference-free rule takes it without cost: lowest χ²/dof reaches ~50 % on MULTI but moves '
                 '~27 % of the correct competing-cluster fits off M3. Controls are untouched in every arm.</p>')

    # M3
    kv = R['kinkvtx']
    o.append('<h2>When is M3 wrong? Bent tracks, and nothing else of size</h2>')
    o.append('<p>The tracker writes only the best ray. Its cluster trees (all clusters on all 8 planes) were rebuilt from the '
             'raw files with the production chain (condor 4546889; rays bit-identical to production, 12 977/12 977 on det3 '
             'file 000) to look for wrong cluster combinations, ghosts and kinks.</p><ul>'
             f'<li><b>Wrong combination / ghosts:</b> an alternative M3 combination pointing at the MX17 particle explains '
             f'{C["det3"]["cls"]["M3_ELSEWHERE"]["pct"]:.2f} % of det3 events. In MULTI events, a second M3 track matching the '
             'second MX17 particle exists in 0–1 % of cases. The extra particles are local (secondaries near the chamber), and '
             'M3 is on the muon.</li>'
             '<li><b>Bent tracks:</b> with 2 + 2 layers, the 4-point fit barely notices a kink between the doublets. A 12 mrad '
             'scatter near the DUT moves the line 3.5 mm at the DUT with χ² = 0.5 mm², inside the χ² &lt; 1 cut. The POS_TAIL '
             'events (clean single track, angle right, &gt; 3 mm off) have M3 χ² 0.37 vs 0.14 for clean events and doublet '
             'kinks of 6–9 mrad vs 3–4. Their residual follows the kink, '
             f'and MX17 sits on the doublet vertex (median |res| {kv["det3"]["med_full_pt"]:.1f} → {kv["det3"]["med_vtx_pt"]:.1f} mm; '
             f'the vertex predictor\'s own spread is {kv["det3"]["s68_vtx_good"]:.1f} mm). A mixture fit gives the bent share '
             + ', '.join(f'{d} {C[d]["bent_f"]:.2f} ± {C[d]["bent_err"]:.2f}' for d in D) + '. det6 sits at z = 232, close to '
             'the lower doublet, where such a kink does not move the M3 line; its zero is the control.</li>'
             f'<li><b>Two chambers at once</b> (6-26 night, det6 + det7, {R["sixpt"]["n_events"]} clean events): in six-point '
             f'lines the single odd point above χ² 50 is always an MX17 chamber ({six["50"]}). That alone does not convict '
             'MX17, because the middle of the lever arm is exactly where a scattered track leaves the chord, and that is where '
             f'det7 is. At milder χ² an M3 layer is the odd point ({six["10"]}), with shifts ≲ 1.5 mm.</li>'
             '<li>A kink-corrected reference (MX17 − M3 − β·kink, cross-validated) moves the core resolution by ≤ 0.03 mm: the '
             'doublet kink is measured too coarsely (~4 mrad) to correct single events.</li></ul>')
    o.append(fig('bent.png', 'det3: the MX17 − M3 residual against the M3 doublet kink (POS_TAIL in orange), and which M3 '
                 'prediction the POS_TAIL events agree with.'))

    # SPS
    o.append('<h2>SPS: det4 at H4</h2>')
    o.append('<p>run_71 RAW, flat mount, 700 V drift plateau (233 V/cm, v = 14.0 µm/ns from the CF₄-epoch timing), uRWELL '
             'reference (truth tan = 0, divergence 0.5 mrad), both views ±8 strips around the reference. Same pivot fitter '
             'and det4 phys bundle (beam D_rc), 64 samples, K = 40, Dp 0.015 (the wet-CF₄ value is not measured). The bench '
             'rows use the same script on the held-out head-on events of the vladder caches.</p>'
             '<div class="scroll"><table><tr><th>set</th><th>X far %</th><th>X s68 [deg]</th><th>Y far %</th><th>Y s68</th><th>n (X/Y)</th></tr>')
    for k, lab in [('sps700_dp015', 'det4 H4 run_71, Dp 0.015'), ('sps700_dp025', 'det4 H4 run_71, Dp 0.025'),
                   ('bench_det4_headon', 'det4 bench head-on (all bands)'), ('bench_det3_headon', 'det3 bench head-on')]:
        s = sp.get(k)
        if s:
            o.append(f'<tr><td>{lab}</td><td>{sx(s, "x")}</td><td>{s["x"]["s68"]:.2f}</td><td>{sx(s, "y")}</td>'
                     f'<td>{s["y"]["s68"]:.2f}</td><td>{s["x"]["n"]}/{s["y"]["n"]}</td></tr>')
    o.append('</table></div>')
    o.append('<p><b>Reading.</b> The H4 tails are symmetric, wide (5th/95th percentile −23° / +32° on X), on high-charge '
             'unsaturated events; 150 GeV beam particles barely scatter. The waveforms show why: the drift column is there '
             '(charge spans 0.95 × the expected 2.1 µs) but its charge sits in one to three short bright bursts, and X carries '
             'the one-sided delayed ±1 copy (0.26 prompt / 0.40 delayed, plane_ratio README) that the symmetric bench kernel '
             'lacks. A bench bundle with only v, Dp and the sample count changed does not describe this, so the far '
             'fraction measures the model mismatch. A K = 24 first attempt was also wrong (the depth grid did not cover the '
             '2.1 µs column, C5); the table is K = 40. <b>Not a measurement of det4.</b></p>')
    o.append(fig('sps.png', 'Head-on angle residual, det4 at H4 against the bench (same fitter, bench-calibrated bundle).'))
    o.append(fig('sps_far_disp.png', 'det4 at H4, X view, strip position × time: four far fits and two good ones (red: uRWELL). '
                 'The charge arrives in short bursts.'))
    pg = R.get('pogo', {})
    o.append('<p><b>Pogo check on the beam data</b> (rerun/pogo_check/run_sps.py, det4 FEU 03 renumbered to M1 channels; '
             f'connectors 3–6 = both cards): X {pg.get("sps_run71_x")}, Y {pg.get("sps_run71_y")}, 0 cross-card links. Only '
             'the beam-lit channels are testable (98 of 256 on X, 201 on Y fired ≥ 50 times). The bench golden rows were '
             'already checked: only det7 X card 1 is POGO.</p>')

    # examples
    o.append('<h2>What each class looks like</h2><p class="sub">Four random events per class. Per view: strip position × time '
             'over ±30 mm (red dashed: M3, white: the fit), and the strip maximum over the whole plane.</p>')
    for f, cap in [('ex_det3_MULTI.png', 'det3 MULTI: two particles; M3 on one, the fit on the other.'),
                   ('ex_det3_DELTA.png', 'det3 DELTA: a delta ray branching off the muon.'),
                   ('ex_det3_POS_TAIL.png', 'det3 POS_TAIL: clean track parallel to M3, 3–4 mm away (mostly bent tracks).'),
                   ('ex_det3_EDGE.png', 'det3 EDGE: within 20 mm of an edge.'),
                   ('ex_det3_FIT_SAT.png', 'det3 FIT_SAT: clean saturated track, wrong fit.'),
                   ('ex_det3_FIT.png', 'det3 FIT: clean track, wrong fit.'),
                   ('ex_det3_OUTSIDE.png', 'det3 OUTSIDE: M3 outside the active area; the fit found something else.'),
                   ('ex_det3_REJECTED_OK.png', 'det3 REJECTED_OK: the fit agrees, the competing-cluster cut removed it.'),
                   ('ex_det4_HW_KNOWN.png', 'det4 dead bands: no or partial column.'),
                   ('ex_det4_FIT.png', 'det4 FIT in live bands.'),
                   ('ex_det7_HW_KNOWN.png', 'det7 X card 1.'), ('ex_det7_MULTI.png', 'det7 MULTI.')]:
        if os.path.exists(os.path.join(BASE, 'report', 'figures', f)):
            o.append(f'<details><summary>{escape(cap)}</summary>{fig(f, cap)}</details>')

    o.append('<h2>What this does not rule out</h2><ul>'
             '<li>The bent-track share is a mixture fit on 35–90 events per chamber; the per-event call is not possible (the '
             'doublet vertex is only good to 1.6 mm). The rest of POS_TAIL (det3 ~0.4 % of events) may be MX17 response or '
             'kinks away from the DUT.</li>'
             '<li>Classes are assigned in a fixed order; an event with two causes counts once (e.g. a second particle near an '
             'edge is EDGE). HW_KNOWN on det7 is event-level: a Y-view outlier in an event whose X sits on card 1 counts there.</li>'
             '<li>The edge zone is defined at 30 mm from the measured k(distance) (bands 0–5 … 20–30 mm, see the follow-ups); the edge-zone sign difference on det2 '
             '(degrador removed) is one chamber.</li>'
             '<li>SPS: the head-on question is open for det4 (no beam-calibrated bundle; the attempt here fails on the model). '
             'det4\'s beam spot sits on a live band, so a future SPS number tests det4\'s live area only. det7 was never at SPS.</li>'
             '<li>The bench second-particle rate (2–4 %) includes secondaries from the stand and the other chamber; it is a '
             'property of the bench, not of n_TOF.</li>'
             '<li>Waveform displays use the local decoded_root (det4 6-24: the pre-fa80c06 decode; fused events are skipped).</li></ul>')
    o.append('<p class="sub">Code: rerun/outliers/ (join, m3feat, wffeat, classify, kinkvtx, sixpt, edge_response, sps_headon, '
             'summary, make_figures, make_report); M3 cluster jobs rerun/outliers/m3clus/; data ~/x17/cosmic_bench/outliers/.</p>')
    o.append('</main></body></html>')
    open(a.out, 'w').write('\n'.join(o)); print('wrote', a.out)


if __name__ == '__main__':
    main()
