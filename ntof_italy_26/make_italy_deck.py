#!/usr/bin/env python3
"""
make_italy_deck.py -- the n_TOF Italy talk (15 Oct 2026): the Athens deck with
two new slides near the end and the meeting/date/page numbers updated.

    python ntof_italy_26/tools/make_figures.sh        # (bash) pulls the figures
    python ntof_italy_26/make_italy_deck.py --src <athens.pptx>

New slides (both built on the Athens header/footer frame):
  * after "Now: sorting the pairs by where their legs land":
        the Al pairs are soft -- few reach the trigger, they do not point back,
        but correlated double tracks are there.
        Source: status deck for 15 Oct (sept26_prelim_analysis/make_status_oct15_deck.py,
        published as dylan-neff.web.cern.ch/notes/x17-status-2026-10-15.html).
  * after "Plan A after LS3":
        LNL, 7Li(p,e+e-)8Be, against n_TOF; the big plastics; the other facilities.
        Source: x17_facility_search lnl/ (FEASIBILITY_SIM.md, lnl/out/talk/numbers.json),
        published as dylan-neff.web.cern.ch/notes/lnl-x17-feasibility.html.

The Athens deck is not edited; the output is written next to this file.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
OUT = HERE / 'slides' / 'neff_x17_ntof_italy_26.pptx'
DEFAULT_SRC = HERE.parents[0] / 'ntof_athens_26' / 'slides' / 'neff_x17_ntof_athens_26.pptx'
FAC = Path.home() / 'PycharmProjects' / 'x17_facility_search'

MEETING = 'n_TOF Italy Meeting'
DATE = '15 October 2026'
OLD_FOOT = 'n_TOF Collaboration Meeting, Athens · 1 October 2026'

# Athens palette
INK = RGBColor(0x23, 0x37, 0x3B)      # titles, body
BODY = RGBColor(0x3C, 0x52, 0x57)     # captions
MUTE = RGBColor(0x5D, 0x71, 0x76)
ORANGE = RGBColor(0xEB, 0x81, 0x1B)   # accent
KICK = RGBColor(0xC6, 0x6A, 0x0F)
CARD = RGBColor(0xFD, 0xF0, 0xE4)
PURPLE = RGBColor(0x8A, 0x3F, 0x8F)   # big plastics (as in the LNL figures)
RED = RGBColor(0xC8, 0x3E, 0x4D)      # Al (as in the status deck)
GREEN = RGBColor(0x2E, 0x8B, 0x57)    # X17 (as in the status deck)
ROWA = RGBColor(0xF4, 0xF6, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT, DISPLAY = 'Noto Sans', 'Noto Sans Display'


# --------------------------------------------------------------------------- #
def runs_of(prs):
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        yield s, sh, r


def retitle_meeting(prs):
    """Footers, the title slide, and the page numbers (25 -> 27 main slides)."""
    renum = {24: 25, 25: 27}           # Plan A outlook, Conclusions
    n_foot = 0
    for s, sh, r in runs_of(prs):
        if OLD_FOOT in r.text:
            r.text = r.text.replace(OLD_FOOT, f'{MEETING} · {DATE}')
            n_foot += 1
        elif r.text == 'n_TOF Collaboration Meeting 2026':
            r.text = f'{MEETING} 2026'
        m = re.fullmatch(r'(\d+)((?:\.\d+)?) / 25', r.text.strip())
        if m:
            n = int(m.group(1))
            r.text = f'{renum.get(n, n)}{m.group(2)} / 27'
    left = [r.text for _, _, r in runs_of(prs) if 'Athens' in r.text]
    print(f'footers updated: {n_foot}; runs still naming Athens: {left}')


def new_slide_like(prs, template, keep):
    """A new slide on the template's layout carrying copies of its frame shapes."""
    s = prs.slides.add_slide(template.slide_layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    for sh in template.shapes:
        if keep(sh):
            s.shapes._spTree.append(copy.deepcopy(sh._element))
    return s


def set_frame(slide, kicker, title, page):
    for sh in slide.shapes:
        if not sh.has_text_frame or not sh.text_frame.paragraphs[0].runs:
            continue
        r0 = sh.text_frame.paragraphs[0].runs
        t = sh.text_frame.text
        if sh.top < Inches(0.6) and sh.left > Inches(0.8) and sh.left < Inches(1.0):
            r0[0].text = kicker
            for r in r0[1:]:
                r.text = ''
        elif abs(sh.top - Inches(0.69)) < Inches(0.05):
            r0[0].text = title
            for r in r0[1:]:
                r.text = ''
        elif re.fullmatch(r'\d+(\.\d+)? / \d+', t.strip()):
            r0[0].text = page
            sh.left, sh.width = Inches(12.37), Inches(0.38)


def text(slide, x, y, w, h, paras, name, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, space_after=0):
    """paras: list of paragraphs; each a list of (text, size, bold, color[, font])."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.0)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    for i, runs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        for rr in runs:
            t, size, bold, color = rr[:4]
            font = rr[4] if len(rr) > 4 else FONT
            r = p.add_run()
            r.text = t
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
            r.font.color.rgb = color
    return tb


def picture(slide, path, x, y, w=None, h=None, name=None):
    W, H = Image.open(path).size
    if w is not None and h is None:
        h = w * H / W
    elif h is not None and w is None:
        w = h * W / H
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic.name = name or Path(path).stem
    return pic, w, h


def card(slide, x, y, w, h, name, fill=CARD, line=ORANGE):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.name = name
    sh.adjustments[0] = 0.08
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    sh.text_frame.text = ''
    return sh


def notes(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt


# --------------------------------------------------------------------------- #
def s_al(prs, template):
    s = new_slide_like(prs, template, lambda sh: sh.shape_type != 13 and sh.name not in
                       ('Plan A outlook bullets', 'TextBox 8'))
    set_frame(s, 'Plan C · What we found', 'The Al pairs are soft: few trigger, poor pointing', '24 / 27')

    pic, w, h = picture(s, FIG / 'al_penetration.png', 0.6, 1.42, w=6.95, name='Al penetration plot')
    y = 1.42 + h + 0.06
    text(s, 0.6, y, 6.95, 0.42, [[
        ('Geant4: a lepton from the capsule aimed at a plastic bar, by its energy. Shaded: leg spectra. '
         'Al conversion legs have ~2 MeV at the gap; X17 legs 4–16 MeV.', 11, False, BODY)]], 'Al plot caption')

    cy = y + 0.55
    card(s, 0.6, cy, 6.95, 7.02 - cy, 'Al outlook card')
    text(s, 0.78, cy + 0.1, 6.6, 7.02 - cy - 0.2, [
        [('Not a total loss: ', 14, True, ORANGE),
         ('the double tracks are still there. A soft, close, same-chamber pair is the ', 13, False, INK),
         ('hardest', 13, True, INK),
         (' two-track case, so it tests the pair reconstruction. X17 legs (~10 MeV, opposite arms) are the easy one.',
          13, False, INK)]], 'Al outlook text', anchor=MSO_ANCHOR.MIDDLE)

    x0, wcol = 7.95, 4.8
    blocks = [
        ('11 %', ' vs 82 %', 'of leptons reach the trigger level: Al vs X17',
         'Plastic ≥ 2.5 MeV. An Al pair fires the trigger itself only 8 % of the time (X17: 45 %); most are '
         'recorded because a Compton electron of the same capture triggered.'),
        ('~9°', ' → 33 mm', 'scattering on the way out spoils the pointing',
         'A 2 MeV leg bends ~9° leaving the capsule and its line misses the vertex by 33 mm (an X17 leg: 9 mm). '
         'Pairs pass a vertex test only at the chance rate (~2 %).'),
        ('~3,700', ' pairs', 'correlated double tracks in the data',
         '0.15 % of late events in A and C, flat while the rate falls ×3.5. Opening angle prefers Geant4 Al e⁺e⁻; '
         'the rate is ~⅛ of the Geant4 expectation (under study).'),
    ]
    yb, step = 1.42, 1.88
    for i, (big, small, label, detail) in enumerate(blocks):
        yy = yb + i * step
        text(s, x0, yy, wcol, 0.5, [[(big, 28, True, ORANGE, DISPLAY), (small, 18, True, ORANGE, DISPLAY)]],
             f'Al stat {i + 1} number')
        text(s, x0, yy + 0.52, wcol, 0.3, [[(label, 14, True, INK)]], f'Al stat {i + 1} label')
        text(s, x0, yy + 0.84, wcol, 0.9, [[(detail, 11.5, False, BODY)]], f'Al stat {i + 1} detail')
    notes(s, """The Al conversion pairs from the capsule are much softer than X17 legs: ~2 MeV at the drift gap against 4-16 MeV. Consequences:
- Trigger: per lepton reaching a gap, 11 % of Al legs reach the plastic at trigger level (>= 2.5 MeV) against 82 % of X17 legs; 2.6 % vs 57 % reach the liquid. An Al pair can fire the trigger itself 7.7 % of the time (X17 pair: 45 %). Most recorded Al pairs ride on a Compton electron trigger from the same capture (76 % of trigger legs).
- Pointing: p*beta*theta = 0.40 MeV rad; a median 2 MeV leg bends ~9 deg and misses the true vertex by 33 mm (X17 ~10 MeV: 2.3 deg, 9 mm). Same-chamber e+e- pass a DCA<30 mm vertex test at 2.3 % with perfect tracks, the event-mixed chance rate is 3.0 %.
- Still statistics: a rate-independent second same-chamber track in 0.15 % (A) / 0.16 % (C) of events after 12 ms: ~2,051 (A) + ~1,660 (C) correlated pairs; Geant4 says ~75-78 % of such pairs are one photon's e+e-. Opening angle (A) chi2 6.6/6 for Geant4 e+e- vs 26/6 for unrelated tracks. Rate = 0.13 (A) / 0.12 (C) of Geant4 x digitised reconstruction: pair-specific, the singles agree to ~20 %. Open.
Source: status deck for 15 Oct, dylan-neff.web.cern.ch/notes/x17-status-2026-10-15.html (slides 4, 5, 8, 9, 10, 11); sept26_prelim_analysis/leg_penetration.py, make_status_oct15_deck.py, SAME_CHAMBER_PAIRS.md. is2_v2 re-pass, capsule-focus k, run_69 onward.""")
    return s


def table(slide, x, y, w, rows, colw, name, size=11):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(0.3 * nr))
    gt.name = name
    tbl = gt.table
    tbl.first_row = tbl.horz_banding = False
    for j, cw in enumerate(colw):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(0.3)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = ROWA if (i % 2 == 0 and i > 0) else WHITE
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = val
            head = i == 0
            r.font.name, r.font.size = FONT, Pt(size if not head else size + 0.5)
            r.font.bold = head or j == 0
            r.font.color.rgb = (KICK if j == 1 else PURPLE if j == 2 else INK) if head else (INK if j else MUTE)
    return gt


def s_lnl(prs, template, num):
    s = new_slide_like(prs, template, lambda sh: sh.shape_type != 13 and sh.name not in
                       ('Plan A outlook bullets', 'TextBox 8'))
    set_frame(s, 'Outlook · LNL Legnaro', 'Next: repeat the ATOMKI measurement at LNL', '26 / 27')
    big, asb = num['cases']['big plastics|1'], num['cases']['as built|1']

    # top left: what changes from n_TOF
    rows = [['', 'n_TOF EAR2 2026 (ran)', 'LNL AN2000 / CN (proposed)'],
            ['Reaction', 'n + ³He → ⁴He*', '⁷Li(p,e⁺e⁻)⁸Be: ATOMKI’s own'],
            ['Precedent', 'none: a new channel', 'yes: ATOMKI 2016, Hanoi 2024'],
            ['Beam', 'white n: the X17 edge moves 110°→66°', 'DC protons ~1.1 MeV: one edge, ~135°'],
            ['Target', '500 bar capsule: σ₆₈ 14.5°', '1.5 µm Li₂O film: σ₆₈ 4.6°'],
            ['Background', 'n capture on Al/C, 10⁴–10⁶ × ³He', 'the reaction’s own IPC (QED) + cosmics'],
            ['Statistics', 'no X17 expected at thermals',
             f'~{big["x17_per_day"]:.0f} X17 / day selected at 1 µA']]
    table(s, 0.6, 1.40, 6.15, rows, [1.05, 2.45, 2.65], 'LNL vs nTOF table', size=10.5)

    # top right: 10 days of pairs
    pic, w, h = picture(s, FIG / 'syst_spectra.png', 6.98, 1.36, w=5.77, name='LNL 10-day spectra')
    text(s, 6.98, 1.36 + h + 0.03, 5.77, 0.42, [[
        ('Statistics are plentiful; the fight is the background. ', 11, True, INK),
        ('Geant4, 10 days at 1 µA, big plastics: S/B ≈ 8 %, so the background shape must be known to ~1 %. '
         'An angle × energy-split × E_sum fit measures it from the signal-free regions.', 11, False, BODY)]],
         'LNL spectra caption')

    # bottom left: rate and time to ATOMKI sensitivity
    yb = 4.22
    pic, w, h = picture(s, FIG / 'lnl_summary.png', 0.6, yb, w=6.15, name='LNL rate and significance')

    # bottom right: big plastics card with the face-on arms to scale
    cx, cw, ch = 6.98, 5.77, 1.98
    card(s, cx, yb, cw, ch, 'Big plastics card', fill=WHITE, line=PURPLE)
    hh = 1.62
    _, w1, _ = picture(s, FIG / 'lnl_arm_ntof.png', cx + 0.1, yb + (ch - hh) / 2, h=hh, name='arm face-on, as built')
    _, w2, _ = picture(s, FIG / 'lnl_arm_big.png', cx + 0.1 + w1 + 0.02, yb + (ch - hh) / 2, h=hh, name='arm face-on, big plastic')
    tx = cx + 0.1 + w1 + w2 + 0.14
    gain = num['pair_tag']['big plastics'] / num['pair_tag']['as built']
    ps = num['price_keur']['PS']['plates_keur']
    text(s, tx, yb + 0.1, cx + cw - tx - 0.12, ch - 0.2, [
        [('New: big plastics', 14, True, PURPLE)],
        [('four 75 × 75 × 5 cm plates replace the 20 × 30 cm bars (red) and the liquid', 10.5, False, INK)],
        [(f'×{gain:.0f}', 22, True, PURPLE, DISPLAY), (' trigger acceptance', 11, True, INK)],
        [(f'{100 * num["pair_tag"]["as built"]:.1f} % → {100 * num["pair_tag"]["big plastics"]:.0f} % of X17 pairs', 10.5, False, INK)],
        [(f'3σ in ~{big["days_3sigma"]:.0f} day, not ~{asb["days_3sigma"]:.0f}', 11, True, INK)],
        [(f'≈ €{ps[1]:.0f}k (polystyrene, budgetary)', 10.5, False, BODY)],
    ], 'Big plastics text', anchor=MSO_ANCHOR.MIDDLE, space_after=1)

    # bottom strip: the other facilities
    text(s, cx, yb + ch + 0.1, cw, 7.05 - (yb + ch + 0.1), [
        [('Also studied, all with neutrons on ³He: ', 10.5, True, INK),
         ('ILL PF1B', 10.5, True, BODY), (' (cold beam; ~7× short of ATOMKI’s level in a 50-day cycle) · ', 10.5, False, BODY),
         ('GANIL-NFS', 10.5, True, BODY), (' (fast n; sub-2 MeV flux unmeasured) · ', 10.5, False, BODY),
         ('NCSR Demokritos', 10.5, True, BODY), (' (tandem MeV neutrons; first look).', 10.5, False, BODY)]],
         'Other facilities', anchor=MSO_ANCHOR.MIDDLE)
    notes(s, f"""LNL: 7Li(p,e+e-)8Be at the 18.15 MeV resonance, the ATOMKI 2016 reaction, energy and target thickness (Li2O 300 ug/cm2, E_p = 1.04-1.10 MeV, 1 uA on the AN2000; ~4 uA on the CN). An exact re-measurement with an independent detector: the MX17 apparatus from n_TOF (Micromegas tracking points each lepton back to the target and rejects cosmics).
Why it is a cleaner problem than n_TOF: no neutrons below E_p = 1.88 MeV, so no capture background; one proton energy fixes the X17 edge at 133-139 deg; a 1.5 um film instead of a 500 bar capsule (sigma68 4.6 deg vs 14.5 deg).
Numbers (Geant4 of the full apparatus, calibrated n_TOF-anchored response, MM 15 deg + TOF, theta x |y| x E_sum fit with the background shape free, R = 5.8e-6, m = 16.7 MeV): of ~{big['produced_per_day']:.0f} X17 produced per day at 1 uA, {asb['x17_per_day']:.0f}/day selected with the n_TOF stack as built and {big['x17_per_day']:.0f}/day with big plastics (x{num['x17_gain']:.1f}). 3 sigma at the ATOMKI ratio: {asb['days_3sigma']:.0f} d as built (needs a repaired liquid scintillator) vs {big['days_3sigma']:.1f} d with big plastics; 5 sigma {big['days_5sigma']:.1f} d.
The limit is signal/background, not statistics: one week at 1 uA with big plastics is ~750 X17 on ~10,700 IPC pairs in 130-172 deg. The energy split |y| takes S/B from 8 % to 19 %; the 3D fit measures the background shape in the signal-free parts of the plane. Open: scattering in the real LNL beam pipe and target chamber (3 sigma 1.0 -> 2.5 -> 6.8 d for sigma68 4.5 -> 9 -> 14 deg); bench-test a plate as a calorimeter.
Big plastics: one 75x75x5 cm plate per arm replaces the two 20x30x2 cm bars and the liquid. Pairs triggered {100 * num['pair_tag']['as built']:.1f} % -> {100 * num['pair_tag']['big plastics']:.1f} % (x{num['trigger_gain']:.1f}). Price (trigger_scint/costs.py, budgetary, no labour/VAT): polystyrene plates EUR {ps[1]:.0f}k ({ps[0]:.0f}-{ps[2]:.0f}k), ~EUR {num['price_keur']['PS']['all_in_keur'][1]:.0f}k with the SiPM readout; PVT EUR {num['price_keur']['PVT']['plates_keur'][1]:.0f}k.
Other facilities (x17_facility_search): ILL PF1B, cold beam, Geant4 done: one 50-day cycle reaches X17/IPC(M1) ~1.2e-2 at 3 sigma, ~7x short of the ATOMKI 8Be level. GANIL/SPIRAL-2 NFS: fast neutrons 1-40 MeV, the sub-2 MeV flux the physics wants is unmeasured. NCSR Demokritos: quasi-monoenergetic MeV neutrons from the tandem, first look only.
Sources: x17_facility_search/lnl/FEASIBILITY_SIM.md, lnl/out/talk/numbers.json, lnl/out/syst/figures/syst_spectra.png; dylan-neff.web.cern.ch/notes/lnl-x17-feasibility.html (slides 3, 6, 8, 10).""")
    return s


def move_to(prs, slide, after_slide):
    lst = prs.slides._sldIdLst
    ids = list(lst)
    new = ids[[prs.slides.index(sl) for sl in [slide]][0]]
    lst.remove(new)
    ids = list(lst)
    pos = [prs.slides.get(int(e.id)) for e in ids].index(after_slide) + 1
    lst.insert(pos, new)


def build(src: Path, out: Path):
    prs = Presentation(str(src))
    retitle_meeting(prs)
    S = list(prs.slides)
    pairs_slide, planA, tmpl = S[41], S[42], S[42]
    assert 'sorting the pairs' in ''.join(sh.text_frame.text for sh in pairs_slide.shapes if sh.has_text_frame)
    assert 'Plan A after LS3' in ''.join(sh.text_frame.text for sh in planA.shapes if sh.has_text_frame)
    num = json.loads((FIG / 'lnl_numbers.json').read_text(encoding='utf8'))
    al = s_al(prs, tmpl)
    lnl = s_lnl(prs, tmpl, num)
    move_to(prs, al, pairs_slide)
    move_to(prs, lnl, planA)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    order = [i + 1 for i, s in enumerate(prs.slides) if s in (al, lnl)]
    print('wrote', out, f'({len(prs.slides)} slides; new slides at {order})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=DEFAULT_SRC, help='the Athens deck (not modified)')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    build(a.src, a.out)


if __name__ == '__main__':
    main()
