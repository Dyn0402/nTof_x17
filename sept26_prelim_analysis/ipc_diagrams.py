#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipc_diagrams.py -- the inline SVG figures that carry the physics arguments.

    python -m sept26_prelim_analysis.ipc_diagrams > /tmp/preview.html

WHY SVG AND NOT MATPLOTLIB.  These four are not plots of numbers; they are the
mechanisms behind the numbers, and a reader who has not met internal pair
creation before needs them before any of the spectra mean anything.  A
mechanism drawing has no data to ship, scales to any width, stays sharp, and --
because it is written with the page's own CSS variables -- follows the reader's
light/dark theme instead of being a white rectangle in a dark page.  Anything
with a number on an axis stays in ``make_ipc_figures.py``.

The four:

``mechanism``   how a transition of energy W turns into two tracks with an
                opening angle, and why the virtual photon's MASS is the whole
                story.  Light photon, big boost, collimated pair; heavy photon,
                no boost, wide pair.
``channels``    n + 3He below 2 eV: two entrance channels, and only one of them
                has a photon to emit.  This is the argument of section 4 as a
                picture.
``al_scheme``   the 27Al(n,g) capture scheme, drawn FROM THE DATA rather than
                by hand -- level energies, parities and primary intensities all
                come from ``ipc_aluminium.line_list()``, so it cannot drift out
                of step with the tables beside it.
``capsule``     where each pair is born and what it has to cross, which is the
                whole of sections 8 and 9 in one picture.

Every drawing is a plain string of SVG with no external references, so the page
stays a single file.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SCHEMA = 'sept26_prelim/ipc_diagrams/1'

#: Colours are the page's CSS variables, so the drawings follow the theme.
INK, INK2, INK3 = 'var(--ink)', 'var(--ink-2)', 'var(--ink-3)'
LINE, ACCENT, WARN, GOOD = 'var(--line)', 'var(--accent)', 'var(--warn)', 'var(--good)'
PANEL = 'var(--panel)'

_HEAD = ('<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         'aria-label="{alt}" style="max-width:{w}px;display:block;'
         'margin:18px auto">')


def _wrap(body: str, w: int, h: int, alt: str) -> str:
    return _HEAD.format(w=w, h=h, alt=alt) + body + '</svg>'


def _t(x, y, s, size=12, fill=INK, anchor='middle', weight='400',
       family='var(--sans)', style=''):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{family}"{style}>{s}</text>')


def _p(x, y, w, html, size=11.5, colour=INK2, align='center', h=None):
    """A block of PROSE inside the drawing, which wraps.

    SVG ``<text>`` does not wrap, and every caption in these diagrams is a
    sentence rather than a label -- a long one silently ran off both edges of
    the viewBox before this existed.  ``foreignObject`` puts real HTML inside
    the SVG, so the browser wraps it and it inherits the page's font.
    """
    h = h or (size * 3.4)
    return (f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h:.0f}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" style="font:400 '
            f'{size}px var(--sans);color:{colour};line-height:1.45;'
            f'text-align:{align}">{html}</div></foreignObject>')


def _arrow_defs(name: str, colour: str) -> str:
    return (f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker>')


# --------------------------------------------------------------------------- #
def mechanism() -> str:
    """Three panels: a light virtual photon, a heavy one, and E0.

    The one idea a reader has to take away before any spectrum makes sense:
    the pair is emitted back-to-back in the virtual photon's own frame, always,
    and what decides the LAB opening angle is how hard that frame is moving.
    ``gamma = W/M``, so the mass of the virtual photon is the only knob.
    """
    w, h = 940, 388
    b = [f'<defs>{_arrow_defs("am", INK)}{_arrow_defs("aa", ACCENT)}</defs>']

    panels = [
        (40, 'light &nbsp; M = 2 MeV', '&gamma; = W/M &asymp; 10', '&asymp;&thinsp;12&deg;',
         [14], INK, 'The pair is thrown forward. A fast frame collimates it, '
                    'and it never reaches the X17 region.'),
        (335, 'heavy &nbsp; M = 16 MeV', '&gamma; = W/M &asymp; 1.3',
         '&asymp;&thinsp;110&deg;',
         [58], GOOD, 'Barely moving, so the back-to-back pair stays nearly '
                     'back-to-back. This is the X17 region.'),
        (630, 'E0 &nbsp; no photon exists', 'contact term, free recoil',
         'anything',
         [14, 32, 50, 68], ACCENT, 'The nucleus absorbs any momentum at no '
                                   'cost, so no mass is preferred and the '
                                   'angle is unconstrained.'),
    ]
    for x0, title, gam, ang, halves, col, note in panels:
        cx, cy = x0 + 135, 162
        b.append(f'<rect x="{x0}" y="46" width="270" height="212" rx="8" '
                 f'fill="{PANEL}" stroke="{LINE}"/>')
        b.append(_t(cx, 72, title, 12.5, col, weight='600'))
        b.append(_t(cx, 90, gam, 11, INK3, family='var(--mono)'))
        b.append(f'<line x1="{cx - 104}" y1="{cy}" x2="{cx - 36}" y2="{cy}" '
                 f'stroke="{INK3}" stroke-width="1.6" stroke-dasharray="4 3" '
                 f'marker-end="url(#am)"/>')
        b.append(_t(cx - 70, cy - 8, 'k', 11, INK3, family='var(--mono)',
                    style=' font-style="italic"'))
        mk = 'aa' if col == ACCENT else 'am'
        for j, half in enumerate(halves):
            op = 1.0 if len(halves) == 1 else 0.35 + 0.2 * (len(halves) - j)
            for sgn in (+1, -1):
                th = np.radians(sgn * half)
                x2 = cx + 74 * np.cos(th)
                y2 = cy - 74 * np.sin(th)
                b.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" '
                         f'y2="{y2:.1f}" stroke="{col}" stroke-width="2.3" '
                         f'opacity="{min(op, 1.0):.2f}" '
                         f'marker-end="url(#{mk})"/>')
        b.append(f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="{col}"/>')
        b.append(_t(cx + 54, cy + 4, '&theta;', 13, col, family='var(--mono)'))
        b.append(_t(cx, 246, f'opening angle {ang}', 11.5, INK2))
        b.append(_p(x0 + 6, 264, 258, note, 10.5, INK3))

    b.append(_p(120, 6, 700, 'ONE transition energy, W = 20.58 MeV. The pair '
                             'is always back-to-back in the virtual '
                             'photon&rsquo;s own frame &mdash; the lab angle '
                             'is set only by how fast that frame moves.',
                12, INK2))
    b.append(_p(120, 346, 700, 'So a multipole is a statement about which '
                               'MASSES the virtual photon is allowed to have, '
                               'and nothing else. Everything on this page '
                               'follows from that one sentence.', 11.5, INK3))
    return _wrap(''.join(b), w, h,
                 'three panels showing how the virtual photon mass sets the '
                 'lab opening angle of the pair')


# --------------------------------------------------------------------------- #
def channels() -> str:
    """n + 3He below 2 eV: two entrance channels, one photon between them."""
    w, h = 940, 372
    b = [f'<defs>{_arrow_defs("cg", GOOD)}{_arrow_defs("ca", ACCENT)}</defs>']

    b.append(_p(120, 4, 700, 'Below 2 eV only s-wave survives, and a '
                             'spin-&frac12; neutron on a spin-&frac12; nucleus '
                             'makes exactly two states.', 12.5, INK2))

    # the entrance
    b.append(f'<rect x="40" y="120" width="150" height="86" rx="8" '
             f'fill="{PANEL}" stroke="{LINE}"/>')
    b.append(_t(115, 150, 'n + &sup3;He', 15, INK, weight='600'))
    b.append(_t(115, 172, '&frac12;&#8314; &otimes; &frac12;&#8314;', 12, INK2,
                family='var(--mono)'))
    b.append(_t(115, 192, 'E&#8345; &lt; 2 eV', 11, INK3, family='var(--mono)'))

    rows = [
        (80, GOOD, '1&#8314;  (&sup3;S&#8321;)', 'M1',
         '&#8308;He 20.58 MeV &rarr; 0&#8314;',
         '55 &mu;b, MEASURED',
         'emits a real photon; the pair is the 3.7&times;10&#8315;&sup3; that '
         'converts internally'),
        (230, ACCENT, '0&#8314;  (&sup1;S&#8320;)', 'E0',
         '0&#8314; &rarr; 0&#8314;',
         '5333 b of (n,p) &mdash; the channel the neutron actually takes',
         'CANNOT emit a real photon. Its only electromagnetic exit is the '
         'pair, so no (n,&gamma;) measurement bounds it'),
    ]
    for y, col, jp, mult, dest, sig, note in rows:
        b.append(f'<line x1="196" y1="163" x2="268" y2="{y + 34}" '
                 f'stroke="{col}" stroke-width="2" marker-end="url('
                 f'#{"cg" if col == GOOD else "ca"})"/>')
        b.append(f'<rect x="272" y="{y}" width="128" height="68" rx="7" '
                 f'fill="{PANEL}" stroke="{col}"/>')
        b.append(_t(336, y + 28, jp, 14, col, weight='600'))
        b.append(_t(336, y + 50, dest, 10.5, INK2))
        b.append(f'<rect x="424" y="{y + 12}" width="56" height="44" rx="6" '
                 f'fill="{col}" opacity="0.14" stroke="{col}"/>')
        b.append(_t(452, y + 40, mult, 15, col, weight='600',
                    family='var(--mono)'))
        b.append(_p(498, y - 2, 420, sig, 11.5, INK, align='left', h=20))
        b.append(_p(498, y + 20, 420, note, 10.5, INK3, align='left', h=52))

    b.append(f'<line x1="498" y1="206" x2="918" y2="206" stroke="{LINE}"/>')
    b.append(_p(120, 322, 700, 'The channel with no photon is populated '
                               '10&#8312; times more strongly than the one the '
                               'pair rate is normalised to. That is why the E0 '
                               'fraction, and not the QED, is the open '
                               'question.', 11.5, INK3))
    return _wrap(''.join(b), w, h,
                 'the two s-wave entrance channels of n plus helium-3 and the '
                 'multipole each one decays by')


# --------------------------------------------------------------------------- #
def al_scheme(lines=None, sn_kev: float = 7725.10, n_show: int = 9) -> str:
    """The 27Al capture scheme, drawn from ``ipc_aluminium.line_list()``.

    Nothing here is typed in: the level energies, their parities and the
    primary intensities are read off the same frame the tables use, so the
    picture cannot disagree with them.  ``n_show`` strongest primaries are
    drawn; the rest are summarised as one grey band.
    """
    from sept26_prelim_analysis import ipc_aluminium as AL
    if lines is None:
        lines = AL.line_list()
    p = lines[lines.is_primary].nlargest(n_show, 'intensity').copy()
    p = p.sort_values('e_gam')

    w, h = 940, 452
    top, bot = 104, 372         # y of the capture state and the ground state

    def ylev(e_lev):
        return bot - (bot - top) * (e_lev / sn_kev)

    b = [f'<defs>{_arrow_defs("ae", GOOD)}{_arrow_defs("am2", ACCENT)}'
         f'{_arrow_defs("au", INK3)}</defs>']
    b.append(_p(120, 2, 700, '&sup2;&#8311;Al(n,&gamma;) at thermal. The '
                             'capture state is 2&#8314; or 3&#8314;, so the '
                             'parity of the level a primary lands on fixes its '
                             'multipole &mdash; and that is the whole '
                             'assignment rule.', 12.5, INK2))

    # capture state and ground state
    b.append(f'<line x1="120" y1="{top}" x2="806" y2="{top}" stroke="{INK}" '
             f'stroke-width="2.5"/>')
    b.append(_t(112, top + 4, f'{sn_kev:.0f} keV', 11.5, INK, anchor='end',
                family='var(--mono)'))
    b.append(_t(816, top + 4, '2&#8314; / 3&#8314; &nbsp;capture state', 12,
                INK, anchor='start', weight='600'))
    b.append(f'<line x1="120" y1="{bot}" x2="806" y2="{bot}" stroke="{INK}" '
             f'stroke-width="2.5"/>')
    b.append(_t(112, bot + 4, '0', 11.5, INK, anchor='end',
                family='var(--mono)'))
    b.append(_t(816, bot + 4, '3&#8314; &nbsp;&sup2;&#8312;Al g.s.', 12, INK,
                anchor='start', weight='600'))

    n = len(p)
    for i, (_, r) in enumerate(p.iterrows()):
        e_lev = sn_kev - r.e_gam
        y = ylev(e_lev)
        x = 152 + (630 * i) / max(n - 1, 1)
        col = GOOD if r.multipole == 'E1' else (
            ACCENT if r.multipole == 'M1' else INK3)
        mk = 'ae' if r.multipole == 'E1' else (
            'am2' if r.multipole == 'M1' else 'au')
        # the level it feeds, and its parity to the right of it
        b.append(f'<line x1="{x - 30:.0f}" y1="{y:.0f}" x2="{x + 30:.0f}" '
                 f'y2="{y:.0f}" stroke="{col}" stroke-width="2.4"/>')
        jp = (r.final_jpi or '?').replace('-', '&#8315;').replace('+', '&#8314;')
        b.append(_t(x + 34, y + 4, jp, 10.5, col, anchor='start',
                    family='var(--mono)'))
        # the primary, width = intensity
        lw = 1.2 + 9.0 * float(r.intensity)
        b.append(f'<line x1="{x:.0f}" y1="{top + 2}" x2="{x:.0f}" '
                 f'y2="{y - 4:.0f}" stroke="{col}" stroke-width="{lw:.1f}" '
                 f'opacity="0.9" marker-end="url(#{mk})"/>')
        # intensity at the TOP, energy at mid-arrow -- both away from the
        # crowded ground state, where the two hard primaries land
        b.append(_t(x, top - 8, f'{100 * r.intensity:.0f}%', 10, col,
                    family='var(--mono)', weight='600'))
        b.append(_t(x - 6, (top + y) / 2 + 3, f'{r.e_gam:.0f}', 9.5, col,
                    anchor='end', family='var(--mono)'))
        # the rest of the cascade
        if y < bot - 8:
            b.append(f'<line x1="{x:.0f}" y1="{y + 4:.0f}" x2="{x:.0f}" '
                     f'y2="{bot - 4:.0f}" stroke="{INK3}" stroke-width="1" '
                     f'stroke-dasharray="3 4" marker-end="url(#au)"/>')

    b.append(_t(463, top - 26, 'intensity per capture (line width is the same '
                               'number); primary energy in keV', 10.5, INK3))
    b.append(_t(463, bot + 26, 'dotted: the rest of the cascade, which also '
                               'makes pairs', 10.5, INK3))
    b.append(f'<rect x="150" y="{bot + 44}" width="16" height="4" '
             f'fill="{GOOD}"/>')
    b.append(_p(174, bot + 36, 660, '<b>E1</b> &mdash; lands on a '
                                    'negative-parity level. These are the soft '
                                    'ones, and they make most of the '
                                    'wide-angle pairs.', 11, INK2,
                align='left', h=20))
    b.append(f'<rect x="150" y="{bot + 68}" width="16" height="4" '
             f'fill="{ACCENT}"/>')
    b.append(_p(174, bot + 60, 660, '<b>M1</b> &mdash; lands on a '
                                    'positive-parity level. The two hard '
                                    'primaries at 7724 and 7693 keV, which '
                                    'convert less, and more collimated.', 11,
                INK2, align='left', h=20))
    return _wrap(''.join(b), w, h,
                 'the aluminium-27 thermal capture gamma scheme with primary '
                 'multipole assignments')


# --------------------------------------------------------------------------- #
def capsule() -> str:
    """Where each pair is born, and what it has to cross to be seen."""
    w, h = 940, 340
    cx, cy, r_out, r_wall = 300, 178, 122, 104
    b = [f'<defs>{_arrow_defs("kn", WARN)}{_arrow_defs("kp", ACCENT)}'
         f'{_arrow_defs("kc", GOOD)}</defs>']

    b.append(_p(120, 2, 700, 'The capsule: 0.5 mm of aluminium and 1.2 mm of '
                             'carbon fibre around 500 atm of &sup3;He.',
                12.5, INK2))

    b.append(f'<circle cx="{cx}" cy="{cy}" r="{r_out}" fill="{LINE}" '
             f'stroke="{INK2}" stroke-width="1.5"/>')
    b.append(f'<circle cx="{cx}" cy="{cy}" r="{r_wall}" fill="{PANEL}" '
             f'stroke="{INK3}" stroke-width="1"/>')
    b.append(_t(cx, cy - 34, '&sup3;He gas', 13, INK, weight='600'))
    b.append(_t(cx, cy - 16, 'optical depth ~150', 10.5, INK3,
                family='var(--mono)'))
    b.append(_t(cx, cy + 2, 'to (n,p) at thermal', 10.5, INK3))
    b.append(_t(cx, cy + 24, 'so essentially every neutron', 10.5, INK2))
    b.append(_t(cx, cy + 40, 'stops in here', 10.5, INK2))

    # the beam
    b.append(f'<line x1="30" y1="{cy}" x2="{cx - r_out - 6}" y2="{cy}" '
             f'stroke="{WARN}" stroke-width="3" marker-end="url(#kn)"/>')
    b.append(_t(96, cy - 12, 'neutrons', 11.5, WARN, weight='600'))

    # a wall pair
    wx, wy = cx - r_out + 9, cy - 76
    b.append(f'<circle cx="{wx:.0f}" cy="{wy:.0f}" r="4" fill="{GOOD}"/>')
    for dx, dy in ((60, -40), (30, 62)):
        b.append(f'<line x1="{wx:.0f}" y1="{wy:.0f}" x2="{wx + dx}" '
                 f'y2="{wy + dy}" stroke="{GOOD}" stroke-width="2" '
                 f'marker-end="url(#kc)"/>')
    b.append(_t(wx - 6, wy - 12, 'capsule pair', 11, GOOD, anchor='end',
                weight='600'))

    # a gas pair
    gx, gy = cx + 22, cy + 62
    b.append(f'<circle cx="{gx}" cy="{gy}" r="4" fill="{ACCENT}"/>')
    for dx, dy in ((78, 26), (18, 84)):
        b.append(f'<line x1="{gx}" y1="{gy}" x2="{gx + dx}" y2="{gy + dy}" '
                 f'stroke="{ACCENT}" stroke-width="2" '
                 f'marker-end="url(#kp)"/>')

    rows = [
        (GOOD, 'born IN the wall',
         '2&ndash;4 MeV between the two tracks, from a &sup2;&#8311;Al or '
         '&sup1;&sup2;C capture &gamma;. Crosses on average half the wall, so '
         'it is the one that gets scattered.'),
        (ACCENT, 'born in the gas',
         '20.6 MeV between the two tracks, from &sup3;He(n,&gamma;) or from '
         'the E0 channel. Crosses the whole wall and barely notices it.'),
        (WARN, 'and the counting',
         'the wall takes ~1.4&times;10&#8315;&sup3; of the neutrons; the gas '
         'takes ~1 but only 10&#8315;&#8312; of those go radiative. That '
         'single ratio is the entire problem.'),
    ]
    y = 92
    for col, head, txt in rows:
        b.append(f'<rect x="470" y="{y}" width="6" height="52" rx="3" '
                 f'fill="{col}"/>')
        b.append(_t(488, y + 16, head, 12.5, col, anchor='start', weight='600'))
        b.append(f'<foreignObject x="488" y="{y + 22}" width="430" height="46">'
                 f'<div xmlns="http://www.w3.org/1999/xhtml" '
                 f'style="font:400 11px var(--sans);color:{INK2};'
                 f'line-height:1.45">{txt}</div></foreignObject>')
        y += 72
    return _wrap(''.join(b), w, h,
                 'cross section of the helium-3 capsule showing where capsule '
                 'pairs and gas pairs are born')


# --------------------------------------------------------------------------- #
def ganil_sweep() -> str:
    """Why a MeV neutron moves the signal and not the background.

    Three columns, one chain each: the neutron brings energy in, the compound
    excitation rises, and the X17 opening angle falls.  The fourth row is the
    point -- the capsule photon is the same photon in all three columns.
    """
    from sept26_prelim_analysis import ganil_background as GB
    w, h = 940, 400
    b = [f'<defs>{_arrow_defs("gs", INK3)}</defs>']
    b.append(_p(100, 2, 740, 'The neutron energy goes into the compound '
                             'nucleus, so at NFS the transition energy is a '
                             'variable and not a constant. Everything that '
                             'comes out of &#8308;He moves with it.',
                12.5, INK2))

    cases = [(0.0, 'n_TOF', INK, 'thermal'),
             (1.5, 'NFS, low', GOOD, '1.5 MeV'),
             (20.0, 'NFS, high', WARN, '20 MeV')]
    rows = [('neutron energy', lambda e: f'{e:.4g} MeV' if e else '&lt; 2 eV'),
            ('&#8308;He excitation E<tspan baseline-shift="sub" '
             'font-size="8">x</tspan>',
             lambda e: f'{GB.excitation(e):.1f} MeV'),
            ('X17 opening angle', lambda e: f'{GB.x17_min_angle(e):.0f}&deg;'),
            ('capsule photon', lambda e: '2.2&ndash;4.4 MeV'),
            ('its pairs', lambda e: 'median 45&ndash;50&deg;')]

    x0, colw = 386, 186
    for j, (label, _) in enumerate(rows):
        y = 96 + 52 * j
        b.append(_p(20, y - 16, 258, label, 11.5,
                    INK if j < 3 else INK3, align='right', h=38))
        if j == 3:
            b.append(f'<line x1="20" y1="{y - 26}" x2="920" y2="{y - 26}" '
                     f'stroke="{LINE}"/>')
    for i, (en, title, col, _) in enumerate(cases):
        cx = x0 + colw * i
        b.append(f'<rect x="{cx - 82}" y="58" width="164" height="260" rx="8" '
                 f'fill="{PANEL}" stroke="{col}" stroke-opacity="0.5"/>')
        b.append(_t(cx, 80, title, 12.5, col, weight='600'))
        for j, (_, fn) in enumerate(rows):
            y = 96 + 52 * j
            c = col if j < 3 else INK3
            b.append(_t(cx, y + 6, fn(en), 14 if j < 3 else 11.5, c,
                        family='var(--mono)',
                        weight='600' if j == 2 else '400'))
            if j < 2:
                b.append(f'<line x1="{cx}" y1="{y + 14}" x2="{cx}" '
                         f'y2="{y + 34}" stroke="{c}" stroke-width="1.4" '
                         f'marker-end="url(#gs)"/>')
    b.append(_p(100, 334, 740, 'The bottom two rows are the same in every '
                               'column: a 2.2 MeV inelastic photon from '
                               '&sup2;&#8311;Al is 2.2 MeV whatever the neutron '
                               'did. <b>The signal tracks the neutron energy '
                               'and the capsule background does not</b> '
                               '&mdash; which is a discriminant n_TOF cannot '
                               'have.', 11.5, INK3))
    return _wrap(''.join(b), w, h,
                 'how neutron energy moves the excitation and the X17 opening '
                 'angle while leaving the capsule background fixed')


DIAGRAMS = {'mechanism': mechanism, 'channels': channels,
            'al_scheme': al_scheme, 'capsule': capsule,
            'ganil_sweep': ganil_sweep}


def main() -> int:
    from sept26_prelim_analysis.make_funnel_report import CSS
    parts = [f'<meta charset="utf-8"><style>{CSS}</style><div class="wrap">']
    for name, fn in DIAGRAMS.items():
        parts.append(f'<h2>{name}</h2>' + fn())
    parts.append('</div>')
    sys.stdout.write('\n'.join(parts))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
