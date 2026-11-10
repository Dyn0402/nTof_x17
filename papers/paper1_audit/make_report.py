#!/usr/bin/env python3
"""Paper I audit (2026-11-10): build report.html from the tables below.

    .venv/bin/python papers/paper1_audit/make_report.py            # write report.html
    .venv/bin/python papers/paper1_audit/make_report.py --thumbs   # force a re-render of thumbs/

thumbs/ is rendered from ../paper1/figures/*.pdf and is not tracked (the repo ignores *.png): the script makes it
when it is missing, so on a fresh clone run the script once before opening report.html.

The audit is a snapshot of papers/paper1 at commit 18f3aad8 (identical to the Overleaf head of 2026-11-10).
Figure and page numbers are those of that build (47 pages with the draft notes hidden, 52 figures).
Thumbnails are relative links (thumbs/<figure id>.png), so the page works from disk and from the hub.
"""
import html
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "paper1" / "figures"


def make_thumbs():
    from PIL import Image
    (HERE / "thumbs").mkdir(exist_ok=True)
    for pdf in sorted(FIGDIR.glob("*.pdf")):
        out = HERE / "thumbs" / pdf.stem
        subprocess.run(["pdftoppm", "-r", "80", "-png", "-singlefile", str(pdf), str(out)], check=True)
        png = out.with_suffix(".png")
        im = Image.open(png).convert("RGB")
        if im.width > 520:
            im = im.resize((520, int(im.height * 520 / im.width)), Image.LANCZOS)
        im.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(png, optimize=True)


# ---------------------------------------------------------------------------------------------------------------
# The page strips: (label, colour, pages). Current = the 2026-11-10 build; proposed = the outline below.
CUR = [("Intro", "#8fa8c8", 2), ("Design", "#b7c4d6", 3), ("Setups", "#c9d3df", 3),
       ("Oper.", "#d9c7a3", 1), ("Efficiency", "#e2a45f", 8), ("Gas", "#69a8a0", 3),
       ("Angle, pos.", "#3f6fb0", 2), ("Bow, edge", "#9a8cc2", 5), ("Time", "#c9a0b8", 1),
       ("E", "#b5b5b5", 1), ("Appendix A: loss classes", "#e6d2b5", 9), ("Appendices B–F", "#dedede", 8),
       ("Refs", "#f0f0f0", 1)]
CUR_KEYS = {9: [7], 11: [9], 16: [14], 18: [15], 19: [16, 17], 20: [18], 22: [21]}   # page -> key figures on it

NEW = [("Intro", "#8fa8c8", 2), ("Chambers", "#b7c4d6", 3), ("Setups", "#c9d3df", 2),
       ("Gas, gap", "#69a8a0", 3), ("Tracking", "#3f6fb0", 3),
       ("Efficiency", "#e2a45f", 3), ("2 particles", "#d98c8c", 2),
       ("As built", "#9a8cc2", 2), ("Sum.", "#b5b5b5", 1), ("Appendices A–E", "#dedede", 10),
       ("Refs", "#f0f0f0", 1)]
NEW_KEYS = {8: [15, 16], 9: [17], 10: [18], 11: [21], 14: [7], 15: [9], 18: [14]}


def strip(blocks, keys, title):
    cell, h, x0, y0 = 19, 26, 4, 58
    n = sum(b[2] for b in blocks)
    out = [f'<svg viewBox="0 0 {x0 * 2 + 47 * cell} 118" role="img" aria-label="{html.escape(title)}">']
    out.append(f'<text x="{x0}" y="12" class="st">{html.escape(title)}: {n} pages</text>')
    x = x0
    for i, (lab, col, npg) in enumerate(blocks):
        w = npg * cell
        out.append(f'<rect x="{x}" y="{y0}" width="{w - 1.5}" height="{h}" fill="{col}" rx="2"/>')
        for k in range(1, npg):
            out.append(f'<line x1="{x + k * cell - .75}" x2="{x + k * cell - .75}" y1="{y0}" y2="{y0 + h}" '
                       f'stroke="#fff" stroke-width=".6" opacity=".7"/>')
        ty = y0 + h + (13 if i % 2 == 0 else 26)          # labels alternate between two rows
        out.append(f'<text x="{x + w / 2}" y="{ty}" class="sl" text-anchor="middle">{html.escape(lab)}</text>')
        x += w
    prev, up = -9, False
    for page in sorted(keys):
        up = (not up) if page - prev < 2 else False       # neighbours alternate between two heights
        prev = page
        cx = x0 + (page - 0.5) * cell
        lab = ", ".join(str(f) for f in keys[page])
        pw = 9 + 5.6 * len(lab)
        top = y0 - (40 if up else 23)
        out.append(f'<line x1="{cx}" x2="{cx}" y1="{top + 15}" y2="{y0 + 5}" stroke="#111" stroke-width="1.2"/>')
        out.append(f'<rect x="{cx - pw / 2}" y="{top}" width="{pw}" height="15" rx="7.5" fill="#111"/>')
        out.append(f'<text x="{cx}" y="{top + 11}" class="sk" text-anchor="middle">{lab}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# Storyboard: proposed section -> (title, pages, claim, [(thumb id, "now Fig. N", action, note)], changes)
KEEP, MERGE, UP, NEW_, APP = "keep", "merge", "rebuild", "new", "to appendix"
STORY = [
    ("1 · Introduction", "1.5 pp",
     "Why the angle is the figure of merit, why a compact TPC, why resistive.",
     [("experiment_layout", "Fig. 1", KEEP, "")],
     ["End with the results in one paragraph. The paper never states them up front (the abstract is still a note), "
      "and the last paragraph is only a roadmap.",
      "One sentence on which pairs land in a single chamber. Fig. 1 draws the pair in two chambers, so a reader does "
      "not know why the two-particle study (your third result) matters."]),
    ("2 · The chambers", "3 pp (Stephan's text still to come)",
     "What was built, layer by layer.",
     [("chamber_stack", "Fig. 2", MERGE, "with Fig. 3a,b into one figure"),
      ("readout_layout", "Fig. 3", MERGE, "drop panel (c), the fan-out")],
     ["Fig. 2's 200-word caption is a parts list. Make it a table (layer, material, thickness, X/X0) with a short caption. "
      "The paper calls the chamber a low-material tracker and never gives the material.",
      "§2.1 Design choices quotes results (charge balance, angles, dark current, fringe) before the detector and the setups "
      "are described. State the choices here; judge them in the conclusions.",
      "Replace &ldquo;delayed copies (~29&nbsp;% at &tau;&nbsp;&asymp;&nbsp;47&nbsp;ns)&rdquo;: it is the retired description and "
      "contradicts §3.4."]),
    ("3 · Setups, data and reconstruction", "2.5 pp",
     "Two setups, five chambers, one reconstruction; what one muon looks like.",
     [("bench_setup", "Fig. 4", MERGE, "with the SPS drawing (Fig. 49) as panel (b)"),
      ("sps_setup", "Fig. 49", MERGE, "comes up from appendix D"),
      ("reco_example", "Fig. 5", MERGE, "with Fig. 12a: one muon, both views, data and model")],
     ["Define once, here: reference track, interior, s<sub>68</sub>, the operating HV. Today s<sub>68</sub> is first defined in "
      "the caption of Fig. 21 and used from §2.1; &ldquo;interior&rdquo; is used in §3.3 and defined in §5.2.",
      "§3.3 lists Run 1 against Run 2 in eight numbers the reader cannot yet parse. One sentence, and the unused "
      "time-series figure (<code>det_a_timeseries</code>) in an appendix, says the same."]),
    ("4 · Drift velocity, gap and gas", "3 pp · opens the results",
     "The chamber calibrates itself. Two independent velocities and a geometric gap agree to 2&nbsp;%; they lie on one Magboltz "
     "curve with 0.97&nbsp;% water; the water dries over hours, so v<sub>d</sub> is measured per run; oxygen shows as a falling column.",
     [("two_velocities", "Fig. 15", KEEP, "your key figure"),
      ("gas_vE", "Fig. 16", KEEP, "add the other chambers' points (exists: gas_vE_timing)"),
      ("gas_drying", "Fig. 17", KEEP, "say C and D share a gas line, if they do: that is what (c) shows"),
      ("attachment_charge", "Fig. 18", KEEP, "frame as: the same method finds oxygen where there is some"),
      ("attachment", "Fig. 19", APP, "history of the SPS gas")],
     ["Write it. Today this section is about 200 words of notes for five figures; efficiency has 2,500 for seven.",
      "Add the missing paragraph on the gap: the design says 30&nbsp;mm (§1, Figs. 1, 2), the data say 28.2&nbsp;mm, and nobody "
      "says so. The bow (centre 1.3–1.9&nbsp;mm deeper) belongs in the same paragraph."]),
    ("5 · Tracking performance", "3 pp · the centre of the paper",
     "About 1&deg; per view from one chamber, the chamber's own, flat in angle, gain and drift field; about 120&nbsp;&micro;m; about 40&nbsp;ns.",
     [("angle_resolution", "Fig. 21", UP, "the main figure: see item 1 under Better plots"),
      ("tracking_perf_all", "Fig. 46", MERGE, "all chambers, into Fig. 21"),
      ("bench_position_res", "Fig. 20", MERGE, "with Fig. 22 into one position figure"),
      ("sps_resolution", "Fig. 22", MERGE, ""),
      ("hv_metrics", "Fig. 10", MERGE, "panels a, c, d + timing (Figs. 26a, 45): what the gain buys"),
      ("timing", "Fig. 26", MERGE, "")],
     ["§7.1 is about 50 words today. It needs about a page: what the number is, that the reference adds at most 0.11–0.14&deg;, "
      "that Run 2 is out of sample, whether the bow correction is in it, and what limits it.",
      "Timing joins here (angle, position, time), instead of sitting after the bow and the edge."]),
    ("6 · Operating point and efficiency", "3 pp",
     "A well-built chamber reaches its plateau before discharges cost anything. 94.8&nbsp;% of single tracks over the full area; "
     "the hardware accounts for 0.6&nbsp;pt of the loss.",
     [("hv_scan", "Fig. 7", KEEP, "your key figure; same x range in every panel"),
      ("efficiency_map", "Fig. 9", KEEP, "your key figure; say which data set (see the table of disagreements)"),
      ("loss_budget", "Fig. 11", KEEP, "")],
     ["One efficiency number in the body. Today chamber A has more than a dozen (91.5 to 97.0&nbsp;%).",
      "The definition in one paragraph; Table 2, the &delta;-ray size classes and the radius argument go to the appendix.",
      "Dark current (Fig. 6) and recovery (Fig. 43) are one sentence each here, with the figures in an appendix."]),
    ("7 · Two particles in one chamber", "2 pp",
     "Equal charge on X and Y pairs the views correctly in 97.7&nbsp;%. Two separate tracks are found as often as one. "
     "Overlap in one view is the hard case.",
     [("two_particle", "Fig. 13", KEEP, "panels (a), (b); (c) is a loss class, to the gallery"),
      ("evd_showcase", "Fig. 12", MERGE, "12b duplicates 13a in kind"),
      ("charge_balance", "Fig. 8", MERGE, "becomes panel (a) of the pairing figure: the balance is why pairing works"),
      ("xy_pairing", "Fig. 14", KEEP, "your key figure; add the charge-asymmetry and time panels")],
     ["Results only. The development history (&ldquo;in its first version it rarely fired&rdquo;, &ldquo;an earlier selection "
      "managed 80&nbsp;%&rdquo;, the veto rescue) is Paper II material.",
      "New: efficiency against the separation of the two tracks, and the opening-angle resolution, from overlaid real muons "
      "(done once on the earlier reconstruction; see the opportunities)."]),
    ("8 · The chamber as built: bow and edges", "2 pp",
     "Every readout board is dished by 1.3–1.9&nbsp;mm, seen three independent ways; correcting it gains 16&nbsp;% in angle. "
     "The unpassivated sides fringe over 20–30&nbsp;mm, which sets the fiducial.",
     [("bow_profiles", "Fig. 23", UP, "show data, not five panels of fitted parabolas"),
      ("bow_closure", "Fig. 24", MERGE, "the convincing panel; keep it"),
      ("edge_turnon", "Fig. 25", KEEP, "")],
     ["One page of bow in the body. Alternatives, stability, plausibility and the transfer to the beam (most of its 1,900 words) "
      "become an appendix that answers one question: is the bow real?"]),
    ("9 · Summary and conclusions", "1 p",
     "The five chambers in one table; what varies between boards; the design choices judged; requirements met or not.",
     [("det4_stripes", "Fig. 27", APP, "Fig. 9e already shows the stripes; this map defines region R for the beam appendix")],
     ["A table of the five chambers (efficiency at the run HV and at the scan optimum, angle X/Y, position, time, dish depth, "
      "water, dark current). The paper has no summary table.",
      "Chamber E's stripes as a full section is a weak ending. A paragraph here, the map in the beam-test appendix.",
      "§5.6 Avenues for improvement becomes three sentences here."]),
]

# ---------------------------------------------------------------------------------------------------------------
KEYFIGS = [
    ("21", "angle_resolution", "Angular resolution of chamber A",
     "Agree: the main result. Weakest presentation of your eight.",
     "Half the figure is a diagonal line. It does not show the residual distribution, chamber B, anything beyond &plusmn;20&deg;, "
     "or what the 1&deg; consists of. Rebuild it (item 1 under Better plots)."),
    ("9", "efficiency_map", "Efficiency maps, five chambers",
     "Agree. Honest and readable at a glance.",
     "The caption says &ldquo;golden-run&rdquo; and &ldquo;pool of 30 rows, 70,470 rays&rdquo; in one breath, and its 92.6&nbsp;% is none of "
     "the numbers used elsewhere. C and D are shown 15 and 65&nbsp;V above their own optimum, which makes them look worse than Fig. 7 "
     "says they are."),
    ("14", "xy_pairing", "X/Y pairing by charge",
     "Agree. One message, specific to this readout.",
     "Only the charge discriminant is drawn, though &ldquo;charge + time&rdquo; is the number quoted. The section around it is mostly "
     "reconstruction history, and the paper never says why two tracks in one chamber matter."),
    ("15", "two_velocities", "Two velocities and the gap",
     "Agree. With Fig. 16, the best argument in the paper: three independent quantities close.",
     "Nothing to fix in the figure. The text does not say that 28.2&nbsp;mm is 1.8&nbsp;mm short of the design."),
    ("16", "gas_vE", "v<sub>d</sub>(E) against Magboltz",
     "Agree.", "Could carry the other chambers' points with their water curves (the unused <code>gas_vE_timing</code> has them)."),
    ("17", "gas_drying", "The gas dries overnight",
     "Agree, as support: it is why v<sub>d</sub> is measured per run.",
     "It shows C and D, the two discharge-limited chambers, not A. The x axis runs to 15&nbsp;h for a &ldquo;night&rdquo;."),
    ("18", "attachment_charge", "Charge against drift time: attachment",
     "A good figure, but the one of your eight least about this detector.",
     "It is chamber E in the SPS gases, the cross-check data set. The independent reader moved 17–19 to an appendix. "
     "I would keep it only as the closing paragraph of the gas section."),
    ("7", "hv_scan", "HV scans: plateau against discharges",
     "Agree.", "Panels have different x ranges; chamber C's long-run star sits off its own curve. "
     "Fig. 10b repeats panel (a); Fig. 48 repeats all of it."),
]

LEFTOUT = [
    ("23 / 24", "The bow", "The independent reader ranked it third: new, general, on all five chambers. "
     "I agree it earns one body figure. Not as drawn: Fig. 23 is five panels of fitted parabolas with no data points."),
    ("10", "Flat over a factor 13 in gain", "The independent reader ranked it second. Efficiency, angle and position do not move "
     "from 425 to 490&nbsp;V; only the time does. That is an operating-margin result and belongs next to the angle."),
    ("11", "Loss budget", "The compact answer to &ldquo;where does the 5&nbsp;% go&rdquo;. It replaces most of §5.4 and appendix A in the body."),
    ("25", "Edge turn-on", "Justifies the fiducial every headline number uses."),
]

# ---------------------------------------------------------------------------------------------------------------
# All 52 figures: (no, id, what, verdict)
ALLFIGS = [
    (1, "experiment_layout", "experiment layout and TPC principle", "keep"),
    (2, "chamber_stack", "chamber cross-section", "merge with 3; caption becomes a table"),
    (3, "readout_layout", "readout board", "merge with 2; drop (c)"),
    (4, "bench_setup", "cosmic bench", "merge with 49"),
    (5, "reco_example", "one muon: data and model", "merge with 12a"),
    (6, "currents_vs_hv", "dark current of A", "appendix, with 44 (log y)"),
    (7, "hv_scan", "efficiency and discharges against HV", "<b>key</b>: keep, opens §6"),
    (8, "charge_balance", "X/Y charge balance", "merge into the pairing figure (14)"),
    (9, "efficiency_map", "efficiency maps", "<b>key</b>: keep"),
    (10, "hv_metrics", "A against HV: charge, efficiency, angle, position", "merge with 26a/45 into one &ldquo;against gain&rdquo; figure; (b) repeats 7a"),
    (11, "loss_budget", "where the single-track loss goes", "keep in the body"),
    (12, "evd_showcase", "one muon; two muons", "12a merges with 5, 12b with 13a"),
    (13, "two_particle", "three two-particle displays", "keep (a), (b); (c) to the gallery"),
    (14, "xy_pairing", "pairing by charge", "<b>key</b>: keep, add panels"),
    (15, "two_velocities", "two velocities and the gap", "<b>key</b>: keep"),
    (16, "gas_vE", "v(E) and Magboltz", "<b>key</b>: keep"),
    (17, "gas_drying", "drying overnight", "<b>key</b>: keep"),
    (18, "attachment_charge", "attachment in the column", "<b>key</b>: keep, as the end of §4"),
    (19, "attachment", "water and O<sub>2</sub> at SPS against time", "appendix (beam test)"),
    (20, "bench_position_res", "bench position resolution", "merge with 22; panel (b) is hard to read"),
    (21, "angle_resolution", "angular resolution", "<b>key</b>: rebuild"),
    (22, "sps_resolution", "position resolution in the beam", "merge with 20"),
    (23, "bow_profiles", "mesh height, five chambers", "rebuild from data, merge with 24"),
    (24, "bow_closure", "tilt measured against predicted", "keep as a panel"),
    (25, "edge_turnon", "edges: efficiency, angle bias, column", "keep"),
    (26, "timing", "time resolution", "merge into the &ldquo;against gain&rdquo; figure; (a) disagrees with 45a"),
    (27, "det4_stripes", "chamber E stripes at SPS", "appendix (beam test)"),
    (28, "loss_gallery", "six loss classes", "one two-page gallery of all classes"),
    ("29–42", "loss_*", "fourteen single-event displays", "cut: six repeat the panels of 28; the rest join the gallery"),
    (43, "spark_deadtime", "efficiency and charge after a discharge", "appendix; it is the only evidence for &ldquo;no dead time&rdquo;"),
    (44, "currents_vs_hv_all", "dark current, all boards", "appendix, with 6"),
    (45, "hv_metrics_extra", "time and angle scale against HV", "merge with 10 and 26"),
    (46, "tracking_perf_all", "angular resolution, all chambers", "promote: into Fig. 21"),
    (47, "timing_all", "timing, all chambers", "appendix; panel (c) is the summary"),
    (48, "hv_scan_single", "alternative to Fig. 7", "delete"),
    (49, "sps_setup", "SPS setup", "merge with 4"),
    (50, "eff_vs_radius", "efficiency against matching radius", "appendix"),
    (51, "residuals", "residual distributions", "appendix"),
    (52, "timing_driftscan", "time against drift field", "appendix"),
]

# ---------------------------------------------------------------------------------------------------------------
# Disagreements: (group, quantity, what the paper says, note)
INCONS = [
    ("These change what a reader concludes", [
        ("Drift gap", "30&nbsp;mm in §1, Fig. 1 and Fig. 2 (&ldquo;30.000&nbsp;mm, nominal&rdquo;); 28.2&nbsp;&plusmn;&nbsp;0.5 and 28.19&nbsp;&plusmn;&nbsp;0.19&nbsp;mm measured in §6; chamber E 27.9 or 28.6&nbsp;mm in the caption of Fig. 19.",
         "Never discussed. The bow section adds that the centre is 1.3–1.9&nbsp;mm further from the cathode than a point 150&nbsp;mm out."),
        ("Time resolution against HV", "Fig. 26a: uncorrelated part about 83&nbsp;ns and event time about 102&nbsp;ns at 425&nbsp;V. Fig. 45a and §5.3: 109 and 105&nbsp;ns.",
         "Two versions of one curve, one in the body and one in the appendix."),
        ("Gain doubling", "16.7&nbsp;V in §4 (behind I&nbsp;&prop;&nbsp;G<sup>1.25</sup>) and Fig. 44; 18.8&nbsp;V in §5.3 and Fig. 10a.",
         "&ldquo;A factor 13 from 425 to 490&nbsp;V&rdquo; fits neither (15 or 11)."),
        ("What the resistive layer does", "§2.1: delayed copies, ~29&nbsp;% at &tau;&nbsp;&asymp;&nbsp;47&nbsp;ns. §3.4: diffusion, the prompt footprint and RC spreading. §7.3: a copy 0.13 of the centre pulse, about 110&nbsp;ns late.",
         "§2.1 is the retired description."),
        ("Dead time after a discharge", "Below 1.2&nbsp;ms (§4); &ldquo;fitted below 0.1&nbsp;s&rdquo; (appendix B).", ""),
        ("Dark current", "&ldquo;No measurable ohmic part&rdquo; (§4); &ldquo;a leakage floor of about 0.2&nbsp;&micro;A&rdquo; (Fig. 6 caption).", ""),
        ("Range of the field fringe", "&ldquo;Decays within ~10&nbsp;mm of the edges&rdquo; (§7.5, as an argument for the bow); &ldquo;fringes over 20–30&nbsp;mm&rdquo; (§2.1, §7.6).", ""),
        ("What the 1&deg; includes", "§7.1 does not say whether the bow correction (16&nbsp;% gain, §7.5) is applied. &ldquo;Interior&rdquo; is &ge;&nbsp;8&nbsp;mm from the y sides (§5.2), but §7.6 says angles need &ge;&nbsp;30&nbsp;mm.", ""),
        ("Angle error budget", "§5.3: the fit's own error is &sigma;<sub>tan</sub>&nbsp;=&nbsp;0.014–0.021, which is 0.8–1.2&deg;. §7.1: 0.56–0.65&deg; is unexplained.",
         "A reader cannot reconcile the two; the paper gives no budget."),
        ("Run 2 drift velocity", "37.2&nbsp;&micro;m/ns (§3.3). That is also the scan value at 390&nbsp;V/cm (§6); Run 2 ran at 355&nbsp;V/cm, and the unused time series shows 36.2–36.6.",
         "Check: possibly a copied number."),
        ("C and D", "Called production defects (§5.3). Their long runs are 15 and 65&nbsp;V above their own optimum, where Fig. 7 has them near 85 and 90&nbsp;%.",
         "A referee will ask whether it is the board or the operating point."),
    ]),
    ("Efficiency: too many versions of one number", [
        ("A, interior, the track the reconstruction keeps", "92.6&nbsp;% (§5.3, Fig. 9), 92.3&nbsp;% (§5.2), 92.29&nbsp;% (Table 3).", "Fig. 9 is a pool of 30 &ldquo;rows&rdquo; (70,470 rays), not Runs 1 and 2 (29,398)."),
        ("A, interior, single track", "95.7&nbsp;% (§5.2, §5.4, appendix E); 95.94&nbsp;% (Table 2).", "Table 2 removes the reference and geometry classes; §5.2 says nothing is removed."),
        ("A, interior, all tracks", "95.1 (Fig. 50), 95.31 (Table 2), 95.48 / 95.26 (§3.3).", ""),
        ("Hardware floor; edge cost", "0.6&nbsp;% (§5.4, §5.6) against 0.72&nbsp;% (appendix A). 0.9&nbsp;pt (§5.4, §7.6) against 1.5&nbsp;pt (§5.6).", "Different denominators, same words."),
        ("Plateau of A", "91–94&nbsp;% (Fig. 7 caption); 90–94&nbsp;% (§5.3).", ""),
        ("Merged-projection events", "146, 176, 195 and 232 in different places. &ldquo;620 to 662, 232 to 176&rdquo; does not conserve events (852, 838). &ldquo;0.8&nbsp;% of tracks&rdquo; (§5.6) against 0.46&nbsp;% (Table 3).", ""),
        ("Two-particle samples", "931 (§5.5), 276 (§5.4), 567 (Fig. 14), with no sentence connecting them. Class (i): 95.6&nbsp;&plusmn;&nbsp;0.8 (Fig. 13) against 95.9&nbsp;&plusmn;&nbsp;0.8 (Table 2).", ""),
        ("Pairing on A by the standard reconstruction", "96.6&nbsp;% and 95.9&nbsp;% in the same subsection.", "The sentence &ldquo;The production reconstruction makes the same choice&hellip;&rdquo; is printed twice."),
    ]),
    ("Other numbers", [
        ("Angle", "A, X: 0.98&deg; (§2.1) and 0.99&deg; (§7.1). B: 1.03/1.08&deg; (§2.1) and 1.05/1.13&deg; (§7.1). Abstract: 1.15–2.51&deg;.", "0.82–0.94&deg; by charge decile (§5.3) is below the headline with no explanation."),
        ("Position", "Abstract: 176&nbsp;&micro;m, 0.30 of the pitch. §7.4: 138 / 120&nbsp;&micro;m. Core 0.50/0.51&nbsp;mm (§3.3), 0.55/0.53 (appendix E).", ""),
        ("Charge balance", "Abstract 0.49/0.53; text 0.485/0.513. D's spread 0.116 (§2.1, §5.1) against &sigma;(ln) 0.109 in the Fig. 8 caption, which is about 0.027 in f.", ""),
        ("Active area", "399.4 &minus; 18.0 &minus; 18.7 = 362.7&nbsp;mm; quoted 359.9. The x edge is at 398.6&nbsp;mm in §7.6.", ""),
        ("Water", "Fig. 19's band is labelled A–D, 0.42–1.00&nbsp;%, but 0.42 is chamber E (§6).", ""),
        ("Time at fixed amplitude", "44&nbsp;&plusmn;&nbsp;5 to 30&nbsp;&plusmn;&nbsp;4&nbsp;ns is called &ldquo;barely changes&rdquo;. It is the uncorrelated part in §5.3 and the event time in appendix F.", ""),
        ("Bow gain", "&ldquo;3–29&nbsp;% per view&rdquo; and &ldquo;2–29&nbsp;% on the X views&rdquo; in one sentence.", ""),
        ("Discharge", "A chip with &ge;&nbsp;8 strips over 3000&nbsp;ADC (Fig. 7); more than 50 strips (the veto, §5.4). About 4&nbsp;% of triggers at 490&nbsp;V against a 0.47&nbsp;% loss.", "Say how the two relate."),
        ("Dangling references", "&ldquo;Noise bands (G<sup>1.4–1.6</sup>, appendix A)&rdquo; is not in appendix A. §3.2 promises the angle scale at 25.4&deg; and the footprint at normal incidence; §7.4 has neither.", ""),
        ("Operating points", "&ldquo;480/480/440&nbsp;V&rdquo; (§4 placeholder); long runs at 490–495&nbsp;V; optima 480 and 430&nbsp;V (Fig. 7).", ""),
    ]),
]

LEAKS = [
    ("File paths and line numbers in text and captions", "Tables 2 and 3, §5.5 (four), §5.6, §7.5 (<code>xcheck.py</code>, <code>summary.py</code>, <code>ref_bow_bound.py</code>), appendix E (<code>budget.py:174</code>), Fig. 37"),
    ("A name and a date in body text", "&ldquo;(Dylan 2026-11-02)&rdquo; in §7.6. It is plain text, so it survives when the draft notes are hidden."),
    ("Pipeline words", "pass 1, the pass, re-pass, pre-pass, golden run, rows, plateau pools, production pick, multi-track credit, bench switch, A1, &ldquo;outlier class FIT&rdquo;"),
    ("Hardware shorthand", "M3 (Fig. 4, appendix E, Fig. 52), DUT, FEU, CM, ZS and RAW, L4–L7, run numbers 55/63/71/149, &ldquo;June boards&rdquo;, &ldquo;capsule focus&rdquo;"),
    ("Never defined", "ladder, column, head-on stack, box model, seed gap, rail, w<sub>0</sub>, k (Fig. 17), K (§7.3), &ldquo;trigger term&rdquo; (Fig. 47)"),
    ("One symbol, two meanings", "f (charge fraction; back-plane resolution), R (matching radius; region), v and v<sub>d</sub>, q and Q"),
    ("One thing, three names", "resist HV, resistive-strip HV, mesh voltage. Dates as 6-27 with no year. &ldquo;Paper II&rdquo; and [1]."),
]

# ---------------------------------------------------------------------------------------------------------------
# Opportunities: (title, status, text)
EXISTS, RERUN, TEXT = "numbers exist", "analysis on existing data", "text or drawing only"
OPPS = [
    ("Rebuild Fig. 21 as the paper's main figure", EXISTS,
     "<ul class='in'>"
     "<li><b>Panel (a):</b> the residual &theta;<sub>fit</sub>&nbsp;&minus;&nbsp;&theta;<sub>ref</sub>, X and Y, on a log scale with s<sub>68</sub> marked, so the tails are visible.</li>"
     "<li><b>Panel (b):</b> s<sub>68</sub> against angle for A and B, plus a beam point at 25&deg;. Chamber E's inclined view at SPS reads 0.96–1.10&deg; with the beam itself as the "
     "reference. That is another setup, chamber and gas, beyond the bench's &plusmn;20&deg;. Two versions of the number exist (<code>gb_m3_angle</code>, and the Paper II hits study); it needs one clean one.</li>"
     "<li><b>Panel (c):</b> all five chambers, X and Y (now Fig. 46 in the appendix).</li>"
     "<li><b>In the text:</b> the reference and scattering add at most 0.11&deg; (X) and 0.14&deg; (Y) (<code>rerun/model_audit/followups/gb_m3_angle</code>), so the 1&deg; is the chamber's own. "
     "Run 2 is out of sample and gives the same number. The resolution is flat over a factor 13 in gain and from 166 to 366&nbsp;V/cm "
     "(<code>rerun/paper1_res/report/figures/res_vs_drift.png</code>).</li>"
     "<li><b>Optional, and worth it:</b> one grey line for the best hit-time method on the same events (2.1 / 2.7&deg;, the Paper II result). It shows in one glance why the waveform fit matters.</li>"
     "</ul>"),
    ("Say what the 1&deg; is made of", EXISTS,
     "The budget exists (<code>rerun/paper1_floor/README.md</code>): reference negligible, ideal-model simulation 0.3–0.46&deg;, saturation and bright tails 12–37&nbsp;% of the excess, "
     "and a remainder that behaves as event-to-event charge-pattern fluctuation larger than the simulation makes. Three sentences and a pointer to Paper II. "
     "Today the headline section says only that 0.6&deg; is &ldquo;not identified&rdquo;, which raises the question and leaves it."),
    ("Material budget, and the 1&deg; against scattering", TEXT,
     "No X/X0 exists anywhere in the repo. From the thicknesses in the Fig. 2 caption I get about 0.10&nbsp;% X0 in front of the drift volume "
     "(60&nbsp;&micro;m Mylar 0.021, 50&nbsp;&micro;m Kapton 0.018, 9&nbsp;&micro;m copper 0.063), which scatters an 8&nbsp;MeV electron by about 2.2&deg; "
     "(1.3&deg; without the copper), against 1.7&deg; for the 20&nbsp;cm of air quoted in the introduction. This is my estimate, and the copper thickness is marked &ldquo;assumed&rdquo;: check it. "
     "If it holds, the conclusion writes itself: the chamber's 1&deg; is below the scattering in its own entrance foils and the air at every lepton energy of interest. "
     "It also says the cathode copper matters more than the unexplained 0.6&deg;."),
    ("A table of the five chambers", EXISTS,
     "Efficiency (run HV and scan optimum), angle X/Y, position, time, dish depth, water, dark current. Every number is already in the text or "
     "<code>rerun/PAPER1_NUMBERS_2026-10-31.md</code>."),
    ("The gap: 30 designed, 28.2 measured", TEXT,
     "One paragraph in §4, with the bow. Two independent methods agree on 28.2&nbsp;mm, which makes the 1.8&nbsp;mm shortfall a result, not an embarrassment."),
    ("Two-track separation and the opening-angle resolution", RERUN,
     "Overlaying two real single-muon events at a chosen separation gives the probability of finding both against separation, and the resolution of the opening angle of a pair in one chamber. "
     "It was done on chamber A with the earlier reconstruction (<code>rerun/multiparticle/</code>, 26 Oct): nothing resolved below 12&nbsp;mm by the standard chain, 28/43/50&nbsp;% at 0–6/6–12/12–18&nbsp;mm with the joint fit, "
     "and an opening-angle &sigma; of 1.47&deg; above 24&nbsp;mm. Re-run on the final reconstruction, this is the figure that turns §7 into a two-track study. It is a batch job, not a laptop one."),
    ("The bow, from data", RERUN,
     "A measured map of the mesh height for chamber A in cells, profiles with binned points and the fit, the dish depth of the five chambers as five points, and the closure plot (Fig. 24). "
     "A map of the angle bias before and after the correction would show the effect directly. The fit products exist in <code>fb_bow_maps</code>."),
    ("Pairing: when it works", RERUN,
     "Correct-pairing fraction against the charge ratio of the two particles (the text says 100&nbsp;% above a factor 1.65, 95&nbsp;% when similar) and the time-difference distribution as a second panel of Fig. 14."),
    ("Charge spectrum and gain curve", EXISTS,
     "<code>rerun/paper1_charge/figures/</code> has the track-charge spectra and the gain curves of four chambers, in working style. §4 is titled &ldquo;gain&rdquo; and shows neither. "
     "State that the absolute gain is not calibrated (no charge calibration of the electronics exists)."),
    ("Stability over ten hours", EXISTS,
     "<code>papers/paper1/figures/det_a_timeseries.pdf</code> is made and unused: charge, efficiency, position, angle and time flat through Runs 1 and 2 while v<sub>d</sub> rises as the gas dries. "
     "It is the chamber-A version of Fig. 17 and a standard figure for a detector paper. Appendix, or panel (d) of Fig. 17."),
    ("Efficiency against angle and drift field", RERUN,
     "Against drift field the numbers exist (84&nbsp;% at 100&nbsp;V, 92–94&nbsp;% from 300 to 1100&nbsp;V); against angle nothing exists. Both are one-panel plots a referee expects."),
    ("Compare with other detectors", TEXT,
     "No number in the paper is set against the literature (ATLAS NSW micro-TPC, the T2K resistive Micromegas TPCs, charge-dispersion readouts). One paragraph in the conclusions."),
]


# ---------------------------------------------------------------------------------------------------------------
CSS = """
:root{--ink:#1c2330;--mut:#5d6776;--line:#d9dee6;--bg:#fbfbfc;--card:#fff;--key:#0b5cad;--warn:#b3541e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px}
h2{font-size:21px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:22px 0 6px}
p{margin:8px 0}
.sub{color:var(--mut);font-size:14px;margin-bottom:18px}
code{font:13px ui-monospace,Menlo,Consolas,monospace;background:#eef1f5;padding:1px 4px;border-radius:3px;overflow-wrap:anywhere}
.verdict{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--key);border-radius:8px;padding:10px 18px 12px}
.verdict ol{margin:6px 0;padding-left:22px}.verdict li{margin:7px 0}
svg{width:100%;height:auto;display:block;margin:6px 0 2px}
.st{font:600 12px system-ui,sans-serif;fill:#1c2330}.sl{font:9.5px system-ui,sans-serif;fill:#3b4350}.sk{font:700 9.5px system-ui,sans-serif;fill:#fff}
.cap{color:var(--mut);font-size:13.5px;margin:2px 0 10px}
.sec{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
.sec h3{margin:0}.pp{color:var(--mut);font-size:13px;font-weight:400;margin-left:8px;white-space:nowrap}
.claim{margin:6px 0 10px;font-size:15.5px}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin:8px 0}
.fig{border:1px solid var(--line);border-radius:8px;padding:6px;background:#fff;font-size:12.5px;line-height:1.35}
.fig img{width:100%;height:132px;object-fit:contain;display:block;background:#fff}
.tag{display:inline-block;font-size:11px;font-weight:600;border-radius:10px;padding:1px 8px;margin-right:5px;color:#fff;white-space:nowrap}
.t-keep{background:#2f7d4f}.t-merge{background:#62708a}.t-rebuild{background:#b3541e}.t-new{background:#7a4fb0}.t-app{background:#8a8a8a}
.sec ul{margin:6px 0 0;padding-left:20px;font-size:14.5px}.sec li{margin:4px 0}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}
th{font-size:12.5px;color:var(--mut);font-weight:600;white-space:nowrap}
td.n{white-space:nowrap;font-weight:600}
td img{width:150px;height:auto;display:block;border:1px solid var(--line);border-radius:4px}
details{margin:10px 0}summary{cursor:pointer;font-weight:600}
.st1{font-size:11.5px;font-weight:600;border-radius:10px;padding:1px 8px;white-space:nowrap}
.s-ex{background:#dcefe2;color:#1d5a36}.s-re{background:#fde7d3;color:#8a3f10}.s-tx{background:#e6e9f2;color:#39456a}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:760px){.two{grid-template-columns:1fr}}
.note{color:var(--mut);font-size:14px}
ul.in{margin:0;padding-left:18px}ul.in li{margin:3px 0}
table.w3 td:nth-child(1){width:17%}table.w3 td:nth-child(2){width:50%}
"""

TAGCLS = {KEEP: "t-keep", MERGE: "t-merge", UP: "t-rebuild", NEW_: "t-new", APP: "t-app"}
STCLS = {EXISTS: "s-ex", RERUN: "s-re", TEXT: "s-tx"}


def storyboard():
    out = []
    for title, pp, claim, figs, changes in STORY:
        out.append(f'<div class="sec"><h3>{title}<span class="pp">{pp}</span></h3><p class="claim">{claim}</p><div class="figs">')
        for fid, now, act, note in figs:
            out.append(f'<div class="fig"><img src="thumbs/{fid}.png" alt="{fid}" loading="lazy">'
                       f'<span class="tag {TAGCLS[act]}">{act}</span>now {now}'
                       + (f'<br>{note}' if note else '') + '</div>')
        out.append('</div><ul>' + "".join(f"<li>{c}</li>" for c in changes) + '</ul></div>')
    return "\n".join(out)


def keyfig_table():
    rows = []
    for no, fid, what, verdict, issue in KEYFIGS:
        rows.append(f'<tr><td class="n">{no}</td><td><img src="thumbs/{fid}.png" alt="{fid}" loading="lazy"></td>'
                    f'<td><b>{what}</b><br>{verdict}</td><td>{issue}</td></tr>')
    return ('<div class="tw"><table><tr><th>Fig.</th><th></th><th>What it is, and whether I agree</th>'
            '<th>What holds it back</th></tr>' + "".join(rows) + '</table></div>')


def leftout_table():
    rows = "".join(f'<tr><td class="n">{no}</td><td><b>{what}</b></td><td>{why}</td></tr>' for no, what, why in LEFTOUT)
    return '<div class="tw"><table><tr><th>Fig.</th><th>Result</th><th>Why it should be more visible</th></tr>' + rows + '</table></div>'


def allfigs_table():
    rows = "".join(f'<tr><td class="n">{no}</td><td>{what}</td><td>{v}</td></tr>' for no, fid, what, v in ALLFIGS)
    return '<div class="tw"><table><tr><th>Fig.</th><th>What</th><th>Verdict</th></tr>' + rows + '</table></div>'


def incons_tables():
    out = []
    for group, rows in INCONS:
        out.append(f'<h3>{group}</h3><div class="tw"><table class="w3"><tr><th>Quantity</th><th>What the paper says, and where</th><th>Note</th></tr>')
        out += [f'<tr><td class="n" style="white-space:normal">{q}</td><td>{s}</td><td>{n}</td></tr>' for q, s, n in rows]
        out.append('</table></div>')
    return "\n".join(out)


def leaks_table():
    rows = "".join(f'<tr><td class="n" style="white-space:normal">{k}</td><td>{v}</td></tr>' for k, v in LEAKS)
    return '<div class="tw"><table><tr><th>Kind</th><th>Where</th></tr>' + rows + '</table></div>'


def opps_table():
    rows = "".join(f'<tr><td class="n" style="white-space:normal">{i}. {t}<br><span class="st1 {STCLS[s]}">{s}</span></td><td>{x}</td></tr>'
                   for i, (t, s, x) in enumerate(OPPS, 1))
    return '<div class="tw"><table><tr><th>What</th><th>Why, and what already exists</th></tr>' + rows + '</table></div>'


def build():
    n_inc = sum(len(r) for _, r in INCONS)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper I audit</title><style>{CSS}</style></head><body><main>
<h1>Paper I audit: organisation, key figures, what to add</h1>
<div class="sub">10 November 2026 · the Overleaf version of that day (same as repo <code>main</code>, 18f3aad8) · 47 pages, 52 figures.
Figure and section numbers are the current ones.</div>

<div class="verdict"><ol>
<li><b>The content is there. The weight is in the wrong places.</b> The introduction names the track angle as the figure of merit.
The paper then gives it about 50 words (§7.1, on page 21 of a 30-page body), against 2,500 words on efficiency accounting, 1,000 more in appendix A, and 1,900 on the bow.
The order follows how the analysis was done, not what a reader needs first.</li>
<li><b>Your eight figures are the right core, with two qualifications.</b> A second reader, given only the PDF and not your list, also put
Fig. 21 first and picked 9, 14, 16 and 7. It did not pick 15, 17 or 18, and it ranked three results you left out above them: everything flat over a
factor 13 in gain (Fig. 10), the bow (Figs. 23, 24) and the edge (Fig. 25). My own view sits between the two: 15 and 16 are the best argument in the
paper, 17 supports them, 18 is the one of your eight least about this detector, and the bow earns one body figure.</li>
<li><b>Fig. 21 undersells the main result.</b> Half of it is a diagonal line. Chamber B, a beam point at 25&deg;, the fact that the reference adds at
most 0.14&deg;, and the comparison with the hit-time method all exist in the repo and none is in the figure or its 50 words.</li>
<li><b>Proposed order: follow the measurement, tan&nbsp;&theta;&nbsp;=&nbsp;w/v<sub>d</sub>.</b> Calibration from the gas (15–18), then the angle (21), then
efficiency (7, 9), then two particles (14), then the chamber as built. Every results section opens on one of your key figures. About 21 body pages and
18 figures, from 30 and 27; appendices from 17 pages to about 10.</li>
<li><b>{n_inc} numbers or statements disagree between places.</b> Most are bookkeeping. A few change what a reader concludes: the gap (30&nbsp;mm designed,
28.2 measured, never discussed), two versions of the timing curve, two gain-doubling voltages, and three descriptions of what the resistive layer does.</li>
<li><b>The largest additions need no new data</b>: the rebuilt Fig. 21, a table of the five chambers, the material budget (which may show the chamber is
already below its own scattering floor), and a two-track separation curve from overlaid real muons.</li>
</ol></div>

<h2>Where the pages go now, and where they would go</h2>
{strip(CUR, CUR_KEYS, "Now")}
{strip(NEW, NEW_KEYS, "Proposed")}
<p class="cap">One cell is one page, both strips on the same scale. Black labels are your eight key figures at the page they appear on. Now: the first arrives on page 9, the most important on
page 22, and the efficiency section with appendix A takes 17 pages. Proposed: the gas figures open the results on page 8 and the angle follows on page 11.
Page counts for the proposal are targets; the current count is with all draft notes hidden and about a third of the prose still unwritten, so the real paper would be longer than 47.</p>

<h2>Why it reads the way it does</h2>
<div class="two"><div>
<h3>One topic, many places</h3>
<p>The same scan or data set is cut by observable and spread over the paper, so no single place carries the point.</p>
<ul>
<li><b>Chamber A's HV scan</b> is in Figs. 7a, 10, 26a and 45, and argued in §4, §5.3, §8 and appendices C and F.</li>
<li><b>Timing</b> is in §5.3, §8, appendix C and appendix F.</li>
<li><b>The geometry as built</b> (gap, bow, fringe, passivation) is in §2.1, §5.2, §6, §7.5 and §7.6.</li>
<li><b>Chamber E and the beam test</b> are in §3.2, §6, §7.4, §9 and appendix D.</li>
<li><b>Event displays</b>: Figs. 5, 12, 13, 28 and 29–42. Six of the fourteen single displays repeat the panels of Fig. 28.</li>
</ul></div><div>
<h3>Effort shows, importance doesn't</h3>
<p>Words of running text per section (captions and draft notes left out): efficiency 2,480; bow 1,860; introduction 725; setups 690; design 445; angle, position and beam cross-check together 365; edge 280; gas 205; operation 205; timing 60.</p>
<ul>
<li>§5.3 is one 1.5-page paragraph that goes from maps to discharges to gain to timing to angle against charge.</li>
<li>§5.5 and §5.6 tell the history of the reconstruction (&ldquo;in its first version it rarely fired&rdquo;). That is Paper II, or nowhere.</li>
<li>§5 to §7 are still notes with bold run-in heads: up to forty numbers in a paragraph, and no sentence saying which one matters.</li>
<li>The paper ends on chamber E's defect, then an empty conclusion. There is no summary table.</li>
</ul></div></div>

<h2>Proposed structure, section by section</h2>
<p class="note">Thumbnails are the current figures. Green: keep. Grey-blue: merge. Orange: rebuild. Grey: to an appendix.
If you prefer the conventional order, move §6 (operating point and efficiency) in front of §4; nothing else changes.</p>
{storyboard()}
<div class="sec"><h3>Appendices<span class="pp">about 10 pp, each answering one question</span></h3>
<ul>
<li><b>A · How is the efficiency defined, and where does each lost track go?</b> Definition and radius (Figs. 50, 51), Table 2, Table 3, one two-page gallery of the loss classes.</li>
<li><b>B · Discharges and currents.</b> Recovery (Fig. 43), dark current (Figs. 6, 44).</li>
<li><b>C · Gain and timing in detail.</b> Figs. 45, 47, 52, the charge spectrum and gain curves, the ten-hour stability.</li>
<li><b>D · Is the bow real?</b> Alternatives, stability over nineteen runs, plausibility, transfer to the beam.</li>
<li><b>E · The beam test of chamber E.</b> Stripes and region R (Fig. 27), gas history (Fig. 19).</li>
</ul>
<p class="note">To Paper II, or cut: the development of the two-track fit, the discharge-veto rescue, the saturated-blob model tests, §5.6, the X neighbour asymmetry (§7.3: no figure, hypothesis &ldquo;unproven&rdquo;, effect at most 5&nbsp;&micro;m), and the signal-to-noise against walk argument.</p></div>

<h2>Your eight key figures</h2>
{keyfig_table()}
<h3>Results you did not list that deserve more room</h3>
{leftout_table()}

<h2>Better plots and analyses, in order of value</h2>
<p class="note">Nothing here needs new data. Green: the numbers or the figure already exist. Orange: an analysis to re-run on what we have. Blue: text or a drawing.</p>
{opps_table()}

<h2>Numbers and statements that disagree</h2>
<p class="note">Each row is the same quantity quoted differently in two or more places. Some differ for a reason (another denominator, another data set); the paper does not say so, and a referee will stop at each.</p>
{incons_tables()}

<h2>Internal language a reader will not follow</h2>
{leaks_table()}

<details><summary>All 52 figures, one line each</summary>{allfigs_table()}</details>

<h2>What the independent reader said</h2>
<p>I gave a second Claude instance only the compiled PDF and asked it to referee the paper cold: no list of key figures, none of my notes. Its ranking of the figures that carry the paper:</p>
<p><b>21</b>, 10, 23 (+24), <b>9</b>, 25, 11, <b>14</b>, 20, <b>16</b>, <b>7</b>. Bold are yours.</p>
<ul>
<li><b>Where it agrees with you:</b> Fig. 21 is the result, and the reader meets it 21 pages in. The maps, the pairing figure, v(E) and the HV scan are in its ten.</li>
<li><b>Where it differs:</b> it would send Figs. 17, 18 and 19 to an appendix and keep only Fig. 16 (with 15c folded in) for the gas. It rated the bow, the flat response against gain, and the edge higher than any gas figure.</li>
<li><b>Its order</b> was the conventional one: gas and calibration, operation, efficiency, tracking, two particles, timing, chamber-to-chamber variation. Mine differs only in putting tracking straight after the calibration it depends on.</li>
<li><b>What it asked for as a referee</b> matches the list above: a material budget, the chamber's own angular resolution with the reference removed, a hit-time result on the same data, binned data for the bow, efficiency against two-track separation, absolute gain and electronics, a table of the five chambers, the SPS angle result promised in §3.2, and a comparison with the literature.</li>
</ul>

<h2>How this audit was done, and what it did not check</h2>
<ul>
<li>I read every section file, built the paper (47 pages with the draft notes hidden), and looked at every page and figure. The Overleaf head and the repo were identical, so nothing was pulled or pushed.</li>
<li>The disagreements were found by reading, and by the independent reader; I checked each one against the text. I did not re-derive any number from the data, so where two values disagree I do not say which is right.</li>
<li>The scattering estimate in the opportunities is mine, from the thicknesses in the Fig. 2 caption. Treat it as a prompt to check, not a result.</li>
<li>Stephan's sections (§2.2–2.5) are empty, so the design and production half of the title could not be assessed.</li>
<li>I did not check the references or the reference list.</li>
</ul>
</main></body></html>
"""
    (HERE / "report.html").write_text(page, encoding="utf-8")
    print("wrote", HERE / "report.html", f"({len(page) / 1e3:.0f} kB, {n_inc} disagreements)")


if __name__ == "__main__":
    if "--thumbs" in sys.argv or not any((HERE / "thumbs").glob("*.png")):
        make_thumbs()
    build()
