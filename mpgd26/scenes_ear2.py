#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes_ear2.py -- the n_TOF EAR2 vertical beam line, as an in-house render.

What the figure has to say, in one picture: 20 GeV/c protons arrive
horizontally, strike the lead spallation target, and the neutrons leaving it at
90 deg to the proton beam are collimated into a vertical line whose measuring
station -- where our He-3 capsule and its Micromegas pinwheel sit -- is ~20 m
above the target.  The **beam pipe ends about a metre above the EAR2 floor**; the
neutrons cross the hall in air, through the experiments, and are stopped by the
beam dump on the bunker roof.  It replaces both the borrowed facility schematic
and the borrowed control-room photograph on the facility slide.

Frame: **beam along +Y** (EAR2 is a vertical line, so +Y is up), the proton beam
along +X, origin at the drawn bottom of the figure, millimetres.  Transverse
axes are the same convention as ``scenes_ntof.py``, so the setup render drops
straight into the measuring station.

The vertical scale, and why it is broken
----------------------------------------
The flight path is ~20 m and the apparatus is ~1 m, so a true-scale elevation is
useless: the experiment would be a few pixels at the top of a blank strip.  The
figure therefore keeps a **single uniform 1:1 scale everywhere** -- transverse
and vertical, so nothing is squashed -- and removes the empty stretch with one
explicit **break**.  ``ACTS`` is the whole mapping: two true-height windows, each
drawn 1:1, separated by an annotated gap.  ``y_of()`` converts a true height in
metres to a drawn coordinate and raises if you ask for a height that falls in the
gap, which is what stops an element being drawn in the wrong act.

The figure stops above the station (2026-08-11, Dylan)
------------------------------------------------------
There used to be a third act at the top -- the bunker ceiling, its second break,
and the beam dump on the roof drawn in full (a 3.2 m concrete block around 1.6 m
of iron around the borated-PE core).  It is **gone**.  It was 2.7 m of drawn
height and the widest object in the picture, spent on shielding, and it made the
one thing the figure is about -- the measuring station -- small.  Dropping it
takes ``DRAWN_H`` from 8.67 m to 5.91 m, so at the same canvas everything left is
drawn about **1.5x larger**, and it removes the upper break with it.

Nothing is asserted away: the beam still leaves the top of the frame, now inside
the wide pipe that really carries it there, under a label that says where it goes
and how high (entrance 24.73 m, above this frame).  The dump's own three layers,
which used to be the drawing's own evidence for the citation, now live only in
``FACTS`` and in the slide bullet.  If you ever want them back, the geometry is
in git history at the commit before 2026-08-11.

Provenance
----------
Every position, aperture and material below is from the EAR2 design paper:

  C. Weiss et al. (n_TOF Collaboration), "The new vertical neutron beam line at
  the CERN n_TOF facility design and outlook on the performance", Nucl. Instrum.
  Meth. A 799 (2015) 90-98, doi:10.1016/j.nima.2015.07.027
      - Table 1 for the element heights above the target centre,
      - Sec. 2.2 and Table 2 for the two collimators,
      - Sec. 2.4 and Fig. 5 for the beam dump and its three layers,
      - Sec. 2.7 for the floor at 18.16 m and the ceiling at 23.66 m.

the third-generation target's own design paper:

  R. Esposito et al. (for the n_TOF Collaboration), "Design of the
  third-generation lead-based neutron spallation target for the neutron
  time-of-flight facility at CERN", Phys. Rev. Accel. Beams 24 (2021) 093001,
  arXiv:2106.11242
      - Sec. III A for the six-slice lead core and the anti-creep plates,
      - Sec. III B for the vessel, its two windows and the two moderators,
      - Sec. II B for the 4 cm water layer, Sec. II C for the 5 cm lead plate.

and the post-upgrade characterisation:

  J. A. Pavon-Rodriguez et al. (n_TOF Collaboration), "Characterisation of the
  neutron beam in the n_TOF-EAR2 experimental area at CERN following the
  spallation target upgrade", Eur. Phys. J. A 61 (2025), arXiv:2505.00042
      - the 19.95 m reference flight path and the ~3 cm FWHM beam there,
      - the vertical line perpendicular to the proton beam,
      - the two swappable second-collimator bores (21.8 / 60 mm).

The structure at the top of the line **is** the beam dump, and that is a
citation, not a reading of a photograph: Weiss Sec. 2 -- "The beam dump for this
new vertical flight path is installed on the roof of the bunker" -- and Sec. 2.4,
which gives the three layers drawn here (borated-polyethylene core 400^3 mm with
a 340 mm x 250 mm entrance bore, iron 1600^3 mm, concrete 3200 x 3200 x 2400 mm).
An earlier note in this repo called it "tentatively the beam dump, unverified";
that caveat is discharged.

What is drawn but not measured is collected in ``ASSUMPTIONS`` and repeated in
the figure caption.

Where the beam pipe stops (corrected 2026-08-11, Dylan)
-------------------------------------------------------
An earlier version of this figure ran the pipe continuously from the collimator,
past the station and on through the bunker roof into the dump, and then **cut it
away** above the floor as a drawing device so that the station was not inside an
opaque tube.  That was wrong about the facility, not just about the drawing: the
pipe **really does end** about a metre above the EAR2 floor, and the space above
it is open hall for experiments to stand in.  So the pipe now terminates for
real, in a flat circular end with a flange, and the neutrons cross the
experimental space **in air**.  Nothing is cut away any more, and the ``section``
accent is used only for the break in the vertical scale, which is what it was
for.

Above that space the beam goes back into a pipe -- the wide section that runs on
up through the bunker ceiling to the dump, and the largest-diameter thing on the
line (``add_uppipe``, added 2026-08-11 at Dylan's request).  It is cut off by the
top of the frame, not ended.  So the hall reads the way the photograph beside the
figure reads: narrow tube up to a termination, open space with the experiment in
it, wide tube away.

The in-hall shape is a reading of that photograph rather than of a drawing --
stepped diameters, a reducer, and the white shielding around the floor
penetration.  See the ``H_CHAMBER_TOP`` block for the numbers and ``ASSUMPTIONS``
for how far they can be trusted.

The one remaining departure from as-built geometry is the measuring station's
**support frame, which is not drawn** -- the real station hangs in an
aluminium-profile frame standing on the floor.  It is left out because four grey
uprights sit in front of the one object the figure is about.  It is declared in
``ASSUMPTIONS``, on the figure label and in the slide caption.

The build-up
------------
``STAGE_PARTS`` names **five** strict subsets of the same picture, for the five
frames ``make_ear2.py`` writes (2026-08-11, Dylan asked for the build to be cut
finer): the target, then the neutrons leaving it at 90 deg, then the collimation
and the pipe ending in the hall, then the pipe on up to the dump, then the
measuring station.  Nothing moves between them -- the camera, the scale, the
light rig and ``ACTS`` are all independent of what is shown -- so the frames
overlay exactly and elements only ever appear.

**One exception, and it is the only one:** the collimated beam is drawn at a
whisper across the 0.56 m the sample occupies in the frame that introduces the
sample (``add_neutrons_hi(fade_sample=True)``).  Drawn at full strength there it
is a grey line down the middle of a translucent capsule, and it reads as the
capsule being *behind* the beam.  See that function.

It was three frames earlier the same day, and four for a few hours before that.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

import meshes as M
import style as S

MM = 1000.0                              # metres -> millimetres

# --------------------------------------------------------------------------- #
# The facility numbers that appear as text on the figure.  Collected here rather
# than typed into label strings, so the figure and its caption cannot disagree.
# --------------------------------------------------------------------------- #
FACTS = dict(
    proton_momentum='20 GeV/c',
    proton_source='CERN PS',
    pulse_width='7 ns rms',
    rep_rate='up to 0.8 Hz',
    n_per_proton='~300 n / proton',
    target='lead spallation target (Target #3, installed 2021)',
    target_core='6 lead slices, 5 x 50 mm + 150 mm, 600 x 600 mm',
    target_len_mm=449.25,
    moderator='4 cm of water',
    pb_plate='5 cm of lead',
    p_per_pulse='up to 1e13 protons per pulse',
    beam_spot='15 mm rms on target',
    angle_to_proton='90°',
    # The FIRST collimator is 7.4-8.4 m above the target -- inside the shaft,
    # roughly 10 m BELOW the EAR2 floor, which is why it never appears in the
    # drawing: it falls inside the lower break.  1 m of iron, 200 mm bore, and
    # its job is to define the beam, not to shape it (Weiss Table 2).
    h_coll1=(7.4, 8.4),                  # 1 m Fe cylinder, 200 mm bore
    coll1_bore='200 mm',
    h_magnet=10.4,                        # permanent sweeping dipole
    h_filters=11.4,                       # 8-slot neutron-filter station
    h_coll2=(15.04, 18.04),               # 2 m Fe + 1 m B-PE, bore 70 -> 21.8
    coll2_bore='70 → 21.8 mm',
    h_floor=18.16,
    h_pipe_end=19.16,                     # ~1 m above the floor; see H_PIPE_END
    h_station=19.95,                      # the reference flight path today
    h_ceiling=23.66,
    h_dump=24.73,                         # beam enters the B-PE core
    fwhm_station='≈ 3 cm FWHM',
    pipe_bore_mm=317.0,                   # circular section, inner diameter
)

CITATION = ('C. Weiß et al. (n_TOF), Nucl. Instrum. Meth. A 799 (2015) 90 '
            '· R. Esposito et al. (n_TOF), Phys. Rev. Accel. Beams 24 '
            '(2021) 093001, arXiv:2106.11242 · J. A. Pavon-Rodriguez et al. '
            '(n_TOF), Eur. Phys. J. A 61 (2025), arXiv:2505.00042')

ASSUMPTIONS = [
    'The vertical scale is 1:1 everywhere, but the empty stretch of pipe is '
    'REMOVED at the drawn break -- so heights read off the figure are wrong and '
    'the labelled ones are right.  What the break skips: the first collimator, '
    'the sweeping magnet and the neutron-filter station.',
    'The figure STOPS about 1.15 m above the measuring station. The bunker '
    'ceiling (23.66 m), its roof, and the beam dump standing on that roof '
    '(entrance 24.73 m -- a borated-PE core inside 1.6 m of iron inside 3.2 m '
    'of concrete, Weiss Sec. 2.4) are all real and all above the frame: they '
    'were drawn until 2026-08-11 and were removed because 2.7 m of drawn '
    'height spent on shielding left the station small. The beam leaving the '
    'top of the frame is labelled with where it goes.',
    'NOTHING at the measuring station is drawn 1:1.  The He-3 capsule is 23 mm '
    'across against a 600 mm target, so at true scale it is one pixel; it is '
    'drawn 5.5x oversize.  The chambers are drawn 1.35x oversize -- board, gap '
    'and frame scaled together, so each is still a chamber and only its size is '
    'wrong -- because at 400 mm they read as small against the capsule they have '
    'to be seen watching.  Their STANDOFF from the beam is the real 330 mm.  The '
    'frame claims "detectors, set back, watching the sample", and nothing '
    'dimensional beyond that.',
    'TWO of the four chambers are drawn, in section, and their drawn azimuths '
    '(136° apart) are chosen to present that section to this camera -- the real '
    'station has FOUR, 90° apart, in a pinwheel. That is a choice about which '
    'two to show and from where, not a claim about the station: drawn as the '
    'true pinwheel the two solid arms come out 45° to the screen and close on '
    'the capsule from both sides until they touch it. The four-arm version is '
    'rendered as well, and is on record in the figures directory.',
    'The TARGET is Target #3 -- the third-generation target installed during LS2 '
    'and the one our data were taken on -- and it is now drawn from its design '
    'paper (Esposito et al., PRAB 24 (2021) 093001) rather than guessed. Sourced '
    'and drawn to scale: six slices of high-purity lead on the proton axis, '
    '600 x 600 mm in cross-section, five of them 50 mm thick with the 150 mm one '
    'at the downstream end against the EAR1 moderator; 9.85 mm aluminium '
    'anti-creep plates between the slices, which carry the nitrogen cooling '
    'channels; an AISI 316L vessel; the 4 mm stainless neutron window welded to '
    'its top; the 50 mm lead plate that window supports, which buys back the '
    'factor 6 in prompt gamma that the better EAR2 coupling cost; and the EAR2 '
    'moderator sitting outside the vessel on that plate, holding the 40 mm water '
    'layer that FLUKA picked as the optimum for the EAR2 resolution function. '
    'Until 2026-08-11 this was the Target #2 CYLINDER instead -- a real object, '
    'but the wrong one, retired at the end of 2018, and none of the '
    'EAR2-facing assembly was drawn at all.',
    'FOUR things about the target are drawn and not sourced, all of them wall '
    'thicknesses or extents the design paper does not give: the vessel wall '
    '(18 mm of clearance here; the paper gives only the 3 mm proton window and '
    'the 4 mm neutron window), the moderator cans\' 12 mm walls and their plan '
    'size (drawn to match the lead plate), the extent of the EAR1 moderator '
    'along the beam (320 mm; the paper says only that it is the larger of the '
    'two), and the radius of the hemispherical aluminium vacuum window above the '
    'moderator, drawn at the pipe bore. The moderator circuits can run '
    'demineralised or borated water independently per area; which was circulating '
    'for our runs is not asserted, and the water is drawn as water.',
    'The proton beam is drawn as an arrow onto the upstream face. The real spot '
    'is 15 mm rms and the drawn one is far wider, because a 15 mm arrow on a '
    '600 mm face is invisible at slide size.',
    'The second collimator is drawn with a straight bore.  The real one is a '
    'stepped approximation to a cone, ' + FACTS['coll2_bore'] + ' over 3 m '
    '(Weiss Table 2), and only its last 0.74 m is inside the drawn window.',
    'The neutron envelope is a drawn illustration of the collimation -- the '
    'beam filling the pipe below the collimator and a few-cm pencil above it. '
    'It is not a transported flux profile.',
    'The beam pipe ENDS about 1 m above the EAR2 floor and the neutrons cross '
    'the experimental space in air -- that is the facility, not a drawing '
    'device. Above that space the beam goes back into the wide pipe that '
    'carries it to the dump, which is the largest section on the line and is '
    'cut off by the top of the frame. The exit and entrance windows are drawn '
    'schematically; their material and thickness are not from a drawing, and '
    'the height at which the upper pipe begins is drawn, not sourced.',
    'The SHAPE of the line inside the hall -- the white segmented shielding '
    'disc and collar on the floor, the lead-disk vacuum chamber at the shaft '
    'bore, the reducer, the narrow tube to the exit window, and the wide '
    'section above -- is scaled off the photograph of the hall beside the '
    'figure, against the ~1.2 m white disc. It is a reading of a photograph, '
    'not a drawing: treat the steps as "the line changes diameter here, about '
    'this much" and nothing finer. The shielding is drawn as polyethylene '
    'because that is what it looks like; no document was consulted.',
    'The lead disks are drawn as a 0.36 m stack, against the 0.57 m effective '
    'height Weiss Sec. 2.3 quotes, because the drawn hall segment is 1 m tall '
    'and also has to carry the shielding collar, the reducer and the exit '
    'flange. The figure claims that there is further collimation in the hall '
    'and roughly where, not how much of it there is.',
    'The measuring station is drawn WITHOUT its support structure, so the '
    'capsule and its four chambers appear to float above the pipe. They do '
    'not: the station hangs in an aluminium-profile frame standing on the EAR2 '
    'floor, which is left out for the same reason as the pipe.',
    'The hall floor is drawn for context.',
]

# --------------------------------------------------------------------------- #
# Palette -- every colour is either style.COL or one scenes_ntof already uses,
# so a red track means "proton beam" and a grey one "neutron" on every slide.
# --------------------------------------------------------------------------- #
COL = dict(
    proton=S.COL['track_beam'],       # the beam-particle red, both scenes
    # Neutrons: scenes_ntof's cool grey, DARKENED here (2026-08-11).  It was
    # '#8b96a3', which is the same value as the amber-tinted pipe interior it has
    # to be seen against -- Dylan could not find the arrows leaving the target at
    # all.  A darker slate is the one change that works on BOTH of this figure's
    # backgrounds: the tan inside the pipe below, and the near-white open hall
    # above, where making the arrows lighter (the obvious fix) would have lost
    # them instead.
    neutron='#55657a',
    envelope=S.COL['beam_env'],       # the amber beam envelope
    lead='#7d838c',                   # lead: dark, slightly blue, matte
    steel='#b9c2cc',                  # the beam pipe
    flange=S.COL['alu'],
    bpe='#e6e9ec',                    # borated polyethylene: pale, matte
    iron='#8a929c',
    concrete='#cfd3d8',
    # The hall shielding is a WARM white and the floor slab is a cool grey, and
    # the two values are deliberately far apart: they touch each other over the
    # whole width of the frame, and at the first attempt (bpe on floor) the
    # shielding disc simply disappeared into the slab it stands on.
    pe='#f6f3ea',
    floor='#cbd0d6',
    strip=S.COL['copper'],            # readout strips on the board
    window='#dfe6ea',                 # the 40 um mylar entrance window
    pcb=S.COL['pcb'],
    gas=S.COL['gas'],
    he3='#8ad0f0',                    # scenes_ntof
    water='#8fc9e3',                  # the moderator water: cool, and a
                                      # different cast from the He-3 gas, or
                                      # the two translucent blues in the
                                      # figure read as the same substance
    al='#c8ccd2',
    cfrp='#2b2f36',
    section='#9c93a0',                # section markers at a break: a muted cast
                                      # of the deck's accent, which is no
                                      # hardware here, so it reads as annotation
                                      # and not as a part.  Desaturated on
                                      # 2026-08-11 -- at the collimator's 680 mm
                                      # radius the old value read as a pink fin
                                      # sticking out of the block, and Dylan
                                      # asked what the part was.
)

# --------------------------------------------------------------------------- #
# The three drawn windows.  (true_lo, true_hi) in metres above the target
# centre, and the drawn y [mm] of true_lo.  Inside a window, 1 m of beam line is
# 1000 mm of drawing.
# --------------------------------------------------------------------------- #
GAP = 560.0                              # drawn height of each break [mm]

ACTS = []


def _build_acts():
    # Each window is as short as it can be and still carry its own act.  The
    # figure's aspect ratio is set here and nowhere else: 1:1 means a metre of
    # window is a millimetre-for-millimetre metre of drawing, so every metre
    # kept in a window is a metre the 340 mm pipe has to be thin against.
    # Two windows since 2026-08-11: the ceiling-and-dump act is gone (see the
    # module docstring).  The second one runs ~1.35 m past the station, which is
    # what it takes to CENTRE the sample between the two pipes (see H_UP0) and
    # still show the first two rings of the upper one before the frame cuts it.
    windows = [(-0.60, 1.20),            # the target and the start of the line
               (17.55, 21.30)]           # the collimator exit, floor, station
    d = 0.0
    for lo, hi in windows:
        ACTS.append(dict(lo=lo, hi=hi, d0=d))
        d += (hi - lo) * MM + GAP
    return d - GAP                        # total drawn height


DRAWN_H = _build_acts()

# the drawn extent of each break, for the anchors and for the cut faces
BREAKS = [(ACTS[i]['d0'] + (ACTS[i]['hi'] - ACTS[i]['lo']) * MM,
           ACTS[i + 1]['d0']) for i in range(len(ACTS) - 1)]

# deg -- the tilt of the break planes.  ZERO since 2026-08-11: the breaks are
# drawn FLAT, and only the COL['section'] accent now marks them.
#
# It was 24 deg, on the theory that a slanted section reads as a drawing cut
# while a square one reads as a pipe that simply stops.  In this figure it did
# not: a 350 mm disc tipped 24 deg, seen from a camera only 6 deg above the
# horizon, projects to a thin dark diagonal streak, and Dylan's reaction to it
# was to ask what the diagonal line across the top of the beam pipe was.  Flat,
# the same disc projects to a shallow ellipse capping the tube -- which is what
# the collimator's own break face was changed to a few hours earlier, for the
# same reason.  The break is carried by the gap and its label instead.
BREAK_TILT = 0.0


def y_of(h):
    """Drawn y [mm] for a true height ``h`` [m] above the target centre."""
    for a in ACTS:
        if a['lo'] - 1e-9 <= h <= a['hi'] + 1e-9:
            return a['d0'] + (h - a['lo']) * MM
    raise ValueError(f'height {h} m falls in a drawn break -- it cannot be '
                     f'placed; windows are '
                     f'{[(a["lo"], a["hi"]) for a in ACTS]}')


def act_span(i):
    """(drawn_lo, drawn_hi) of act ``i``."""
    a = ACTS[i]
    return a['d0'], a['d0'] + (a['hi'] - a['lo']) * MM


# --------------------------------------------------------------------------- #
# Drawn geometry [mm].  Transverse sizes are TRUE sizes.
# --------------------------------------------------------------------------- #
R_BORE = FACTS['pipe_bore_mm'] / 2.0      # 158.5
R_PIPE = R_BORE + 12.0

# --------------------------------------------------------------------------- #
# The spallation target -- TARGET #3, the one our data were taken on
# --------------------------------------------------------------------------- #
# Rebuilt 2026-08-11 from the design paper, after Dylan asked whether the shape
# was known or guessed:
#
#   R. Esposito et al. (for the n_TOF Collaboration), "Design of the
#   third-generation lead-based neutron spallation target for the neutron
#   time-of-flight facility at CERN", Phys. Rev. Accel. Beams 24 (2021) 093001,
#   arXiv:2106.11242 -- Sec. III A for the core, Sec. III B for the vessel and
#   the two moderators, Sec. II B for the 4 cm water layer, Sec. II C for the
#   lead plate above the core.
#
# It used to be the Target #2 cylinder (400 mm long, 600 mm diameter, Weiss
# Sec. 2.1), which is a real object but the WRONG one: Target #2 was retired at
# the end of 2018 and Target #3 went in during LS2.  They are not the same shape
# -- the Pavon-Rodriguez characterisation paper puts it plainly, "the previous
# spallation target was a monolithic lead cylinder coupled to EAR2 via a
# polygonal window", where Target #3 is a stack of square slices with a
# purpose-built flat lead plate and water moderator on top of it.
#
# Everything below is from the paper except the four things ASSUMPTIONS names.
TGT_XY = 600.0                            # slice cross-section, 0.6 m x 0.6 m
# Six slices of high-purity lead (UNS L50006, >= 99.98 wt%) along the proton
# beam, "5-cm thick, with the exception of the slice close to the EAR1
# moderator, which is 15-cm thick" -- so the thick one is at the DOWNSTREAM end.
TGT_SLICES = (50.0, 50.0, 50.0, 50.0, 50.0, 150.0)
ACP_T = 9.85                              # anti-creep plate, 9.85 +/- 0.05 mm
TGT_LEN = sum(TGT_SLICES) + (len(TGT_SLICES) - 1) * ACP_T      # 449.25
VES_GAP = 18.0                            # DRAWN vessel clearance; see ASSUMPTIONS
NWIN_T = 4.0                              # the 4 mm stainless neutron window
PBPLATE_T = 50.0                          # the 5 cm lead plate under the EAR2
                                          # moderator, which buys back the factor
                                          # 6 in prompt gamma that the new,
                                          # better-coupled vacuum chamber cost
MOD_WATER = 40.0                          # 4 cm water: the FLUKA optimum for the
                                          # resolution function, both moderators
MOD_WALL = 12.0                           # DRAWN Al wall; see ASSUMPTIONS
# Plan size of the lead plate and the EAR2 moderator can.  DRAWN: the paper gives
# the plate's 50 mm thickness and the can's 40 mm of water but no footprint.  Set
# to the vessel's own top face less a 26 mm ledge, so the can is bolted ONTO
# something rather than overhanging it, and so the gas outlets on the vessel top
# have somewhere to be.  It is comfortably wider than the 317 mm the vacuum
# window above it has to cover.
EAR1_MOD_X = 320.0                        # DRAWN extent of the EAR1 moderator
CRADLE_T = 64.0                           # DRAWN cradle under the core

# Kept as aliases because the proton track, the neutron envelope and the label
# anchor are all positioned off the target's size rather than off its shape.
TGT_L, TGT_R = TGT_LEN, TGT_XY / 2.0
MOD_XY = TGT_LEN + 2 * VES_GAP - 52.0     # see MOD_WALL

# The stack the neutrons going to EAR2 actually cross, in mm above the target
# centre.  This is the part of the target that matters for THIS figure and it was
# not drawn at all before: the paper's Sec. III B order is core -> vessel -> the
# 4 mm neutron window welded to its top -> the 5 cm lead plate the window
# supports -> the EAR2 moderator, which sits OUTSIDE the vessel, bolted to it and
# resting on that plate -> the hemispherical aluminium vacuum window into the
# beam pipe (that one from Pavon-Rodriguez Sec. 3).
Y_CORE = TGT_XY / 2.0                                   # 300 -- top of the lead
Y_VES = Y_CORE + VES_GAP                                # 318
Y_NWIN = Y_VES + NWIN_T                                 # 322
Y_PB = Y_NWIN + PBPLATE_T                               # 372
Y_MOD = Y_PB + 2 * MOD_WALL + MOD_WATER                 # 436 -- top of the can
R_VACWIN = R_BORE                                       # DRAWN; see ASSUMPTIONS

# Where the vertical pipe starts: just clear of the vacuum window's apex.  It was
# 0.30 m, which is now inside the lead plate -- the whole target-moderator
# assembly reaches 0.44 m and the window's dome another 0.16 m above that.
H_PIPE_START = 0.60

COLL2_R = 340.0                           # vacuum vessel, 680 mm outer dia
COLL2_BORE = 21.8 / 2.0                   # the small-configuration exit
H_COLL2_FE = 17.64                        # last 0.4 m is B-PE + B4C

# Where the beam pipe ends -- a real end of pipe, ~1 m above the EAR2 floor top
# at 18.16 m, with open hall above it (Dylan, 2026-08-11; the earlier version of
# this figure ran the pipe to the dump and then cut it away, which was wrong).
# 19.16 m clears the lead disks, which end at 18.77 m, and leaves 0.79 m of air
# below the station, so the capsule and its chambers sit in open beam.
H_PIPE_END = 19.16

# --------------------------------------------------------------------------- #
# The in-hall pipe, read off the photograph (2026-08-11)
# --------------------------------------------------------------------------- #
# ``slides/assets/img/ear2_hall_photo.jpg`` shows what the design paper does not:
# the line does NOT cross the hall at the shaft's 317 mm bore, and it is not one
# smooth tube either.  It comes out of the floor at about that size, inside a
# white segmented shielding disc lying on the floor with a collar above it,
# widens into the vacuum chamber that carries the lead disks, then steps DOWN
# through a reducer into a narrow tube -- the collimated beam is only a few
# centimetres across, so nothing above the collimator needs to be wide -- and
# above the experimental space it steps back UP into the widest section on the
# line, which is the one that goes to the dump.
#
# The radii here are scaled off that photograph against the ~1.2 m white disc,
# and the heights of the steps are chosen to fit between the floor and the end
# of the pipe.  Both are ROUGH and neither is from a drawing -- see ASSUMPTIONS.
H_CHAMBER_TOP = 18.86        # top of the lead-disk vacuum chamber
H_NECK = 18.97               # top of the reducer, bottom of the narrow tube
R_NECK = 78.0                # bore radius of the narrow in-hall tube
R_NECK_OUT = R_NECK + 10.0

# 20.74 is not a free choice: it puts the sample at 19.95 m exactly HALFWAY
# between the end of the lower pipe (19.16 m) and the start of the upper one, so
# the one object the figure is about sits in the middle of the open space instead
# of crowding the pipe above it (Dylan, 2026-08-11).
#
# 160 mm rather than the photograph's own proportions: there the upper pipe is
# the widest thing on the line, and drawn that way it OUTWEIGHS the station,
# which is the opposite of what removing the beam dump was for.  It is still
# visibly the wide section -- the tube below it is 78 mm.
H_UP0 = 20.74                # bottom of the wide pipe on up to the beam dump
H_UP_RING = 21.02            # the one flange ring above its entrance flange:
                             # "only the first 2 rings", so the pipe reads as
                             # continuing rather than as a short stub
R_UP = 160.0                 # ... and its bore radius
R_UP_OUT = R_UP + 12.0

R_SHIELD = 620.0             # the white disc lying on the floor
H_SHIELD_DISC = 18.32        # its top
R_SHIELD_COLLAR = 250.0      # the collar around the pipe above the disc
H_SHIELD_COLLAR = 18.48      # its top

# Slabs are drawn as SECTIONS: wide enough to read as a floor, narrow enough
# not to become the subject.  A slab drawn at its real plan size is a grey wall
# across the whole frame from this camera.
FLOOR_T = 240.0                           # drawn thickness of the EAR2 floor
# Narrowed from 2400 on 2026-08-11: with the 3.2 m dump gone the drawn scene is
# 1.5x larger, so a 2.4 m floor slab now reaches almost the full frame width and
# competes with the station for attention.
FLOOR_W = 1900.0
HALL_W = 1600.0

PLATE_SIZE = 400.0                        # MX17 readout board, drawn square
PLATE_THICK = 12.0                        # the board itself, not the assembly
DRIFT_GAP = 30.0                          # the settled MX17 drift gap
PLATE_R = 330.0                           # board centre to the beam axis
PLATE_PHI0 = 25.0                         # pinwheel phase [deg]; see add_station
# Opacity of the frames of the arms the camera looks THROUGH.  0.7, not the 0.24
# scenes_ntof uses for its ghosts: ``frame_ring`` is a hollow rectangular band, so
# a near arm drawn as its frame alone does not cover the sample at all, and at
# 0.24 it was invisible -- which made the 4-arm and 2-arm variants identical
# pictures and the comparison Dylan asked for meaningless.
NEAR_ALPHA = 0.70
GAS_ALPHA = 0.34                          # lit gas, with the board behind it

# The two drawn scales at the station, and NEITHER of them is 1:1 any more
# (2026-08-11, Dylan: "make the capsule slightly smaller and the detectors a bit
# larger ... I don't care much about how the detectors look here or their
# accuracy -- this is mostly just a generic example diagram").
#
# The capsule was already drawn oversize -- it is 23 mm across against a 600 mm
# target, so at 1:1 it is a pixel.  The chambers used to be the one true-size
# thing in this part of the figure, and they are not any more: PLATE_DRAW scales
# the whole chamber, board and gap and frame together, so it stays a chamber and
# only its size is a lie.  Their STANDOFF (PLATE_R) is still the real 330 mm, so
# what the frame claims is "detectors, set back, watching the sample", which is
# all a motivation slide should be claiming.  Both factors are in ASSUMPTIONS and
# in the backup slide's caption.
CAPSULE_SCALE = 5.5
PLATE_DRAW = 1.35
# 24 strips, not 512.  The real 0.78 mm pitch at this scale is 0.26 px, i.e. a
# flat grey wash; ~24 is what actually reads as "strip readout" on a 400 mm board
# drawn 135 px across.  Declared in ASSUMPTIONS.
N_STRIPS_DRAWN = 24

# How many of the four chambers to draw.  **2 is the default** (Dylan,
# 2026-08-11): two of the four, in section, left and right of the sample and
# clearly behind it.  4 is the real pinwheel and is still rendered, as the
# alternate -- ``make_ear2.py`` writes the last frame both ways.
#
# Why the pinwheel is not the default, having been tried: at this camera the two
# arms the cutaway leaves solid come out 45 deg to the screen, ~90 px wide, and
# they close on the 160 mm drawn capsule from both sides until they touch it.  You
# get a green box with a sliver of cyan in it, which was Dylan's original
# complaint about this part of the figure and is not fixed by making the chambers
# more detailed.
STATION_ARMS = 2

# How far the section pair is rotated off exactly edge-on [deg].  0 would give
# two 42 mm-thick vertical bars -- a true cross-section and unreadable as a
# detector.  22 deg opens ~150 mm of foreshortened board face, enough for the
# strips to read, while keeping each chamber ~40 px wide and 70+ px clear of the
# capsule.
SECTION_TILT = 22.0


def _section_angles():
    """Azimuths [rad] of the two chambers drawn IN SECTION, left then right.

    Defined **relative to the camera**, not to ``PLATE_PHI0``: the whole point is
    that each one presents its section to this particular viewpoint.  Both sit
    just past the pair of azimuths that are exactly edge-on (the camera azimuth
    +/- 90 deg), on the far side, so that

      * they project to the left and the right of the beam rather than in front
        of and behind it,
      * the cutaway plane slices them -- which is what makes them sections and
        not slabs -- and every surviving piece is BEHIND the sample in depth, and
      * the sample is left alone on the beam with ~200 mm of clear air either
        side of it.

    The price is that the two drawn azimuths are 136 deg apart rather than the
    real 90 deg.  That is a drawing choice about which two of four to show and
    from where, it is declared in ASSUMPTIONS, and the four-arm alternate exists
    precisely so the true arrangement is on record.
    """
    if _CUT is None:                                         # pragma: no cover
        a = -90.0                                            # any fixed pair
    else:
        a = np.degrees(np.arctan2(_CUT[2], _CUT[0]))
    back = a + 180.0
    off = 90.0 - SECTION_TILT
    return [np.radians(back - off), np.radians(back + off)]


def _arm_angles():
    """Azimuths of the drawn chambers [rad] -- the pinwheel, or the section pair."""
    if STATION_ARMS == 4:
        return [np.radians(PLATE_PHI0 + 90.0 * k) for k in range(4)]
    return _section_angles()


# How far above and below the sample the neutron beam is drawn as a whisper --
# see add_neutrons_hi.  0.28 m covers the drawn capsule (23 mm x CAPSULE_SCALE,
# plus its nose) with a little air, and nothing else.
SAMPLE_BAND = 280.0

# ONE opacity for the amber beam envelope, above the collimator and below it
# (2026-08-11).  It used to be 0.17 below and 0.34 above, on the reasoning that a
# 22 mm pencil needs more tint than a 317 mm cone to be seen at all.  What that
# actually produced was two different-COLOURED beams in one figure: the wide cone
# is pale amber, and the same amber at twice the opacity on a thin tube seen
# through both its walls came out olive-brown, which is what Dylan asked about.
# The pencil is instead carried by its tracks, which are drawn far wider than the
# beam really is for exactly this reason -- see _beam_tracks.
ENV_ALPHA = 0.16
ENV_FADE = 0.05                           # ... across the sample; see SAMPLE_BAND

# The beam dump (Weiss Sec. 2.4 / Fig. 5: a 400^3 mm borated-PE core with a
# 340 x 250 mm entrance bore, inside 1600^3 mm of iron, inside a 3200 x 3200 x
# 2400 mm concrete block, entrance at FACTS['h_dump'] = 24.73 m) is NO LONGER
# DRAWN -- see the module docstring.  It is above the top of the frame, and the
# beam leaving the frame carries a label that says so.  Its dimensions stayed in
# this comment rather than as dead constants, because they are what the slide
# bullet quotes.


# --------------------------------------------------------------------------- #
# Small builders the shared meshes module does not have
# --------------------------------------------------------------------------- #
def _ring(r, n=96):
    a = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def annulus(y0, y1, r_in, r_out, n=96):
    """A hollow cylinder about the beam axis -- one section of beam pipe.

    Built as the extruded band between two circles (``meshes.band_prism``)
    rather than as a boolean difference of two cylinders: VTK's boolean filters
    are unreliable on tessellated surfaces of revolution, and the band keeps a
    regular tessellation, which is what the aluminium highlight needs.  It is
    also a closed solid, so after the longitudinal cut below you look at a real
    inner wall with an outward normal, and it lights correctly.
    """
    return M.band_prism(_ring(r_in, n), _ring(r_out, n), y0, y1 - y0,
                        normal_axis='y')


def conical_shell(y0, y1, ri0, ro0, ri1, ro1, n=96):
    """A hollow cone frustum about the beam axis -- a pipe reducer.

    ``annulus`` can only make straight sections (``band_prism`` extrudes one
    pair of rings), so the taper needs its own builder.  Closed, like
    ``annulus``, so that after the longitudinal cut you look at a real inner
    wall with an outward normal and it lights correctly.
    """
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    rings = [(ri0, y0), (ro0, y0), (ri1, y1), (ro1, y1)]
    pts = np.vstack([np.column_stack([r * c, np.full(n, y), r * s])
                     for r, y in rings])
    faces = []

    def band(k0, k1, flip=False):
        for i in range(n):
            j = (i + 1) % n
            q = [k0 * n + i, k0 * n + j, k1 * n + j, k1 * n + i]
            faces.append([4] + (q[::-1] if flip else q))

    band(1, 3)                  # outer wall
    band(0, 2, True)            # inner wall
    band(0, 1, True)            # the annular face at y0
    band(2, 3)                  # ... and at y1
    return pv.PolyData(pts, faces=np.hstack(faces)).compute_normals(
        auto_orient_normals=True, inplace=False)


def pie_segments(y0, y1, r_in, r_out, n_seg=8, gap_deg=1.4):
    """A ring of wedge blocks -- the segmented white shielding on the floor.

    Drawn as separate wedges with a gap between them rather than as one
    annulus, because the radial seams are the only thing that identifies the
    disc in the photograph as demountable shielding blocks and not a plinth.
    """
    out = []
    for k in range(n_seg):
        a0 = 2 * np.pi * k / n_seg + np.radians(gap_deg)
        a1 = 2 * np.pi * (k + 1) / n_seg - np.radians(gap_deg)
        a = np.linspace(a0, a1, 16)
        poly = np.vstack([
            np.column_stack([r_out * np.cos(a), r_out * np.sin(a)]),
            np.column_stack([r_in * np.cos(a[::-1]), r_in * np.sin(a[::-1])])])
        out.append(M.polygon_prism(poly, y0, y1 - y0, normal_axis='y'))
    return out


def frustum(y0, y1, r0, r1, n=72):
    """A closed cone frustum about the beam axis -- the neutron beam envelope."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lo = np.column_stack([r0 * np.cos(a), np.full(n, y0), r0 * np.sin(a)])
    hi = np.column_stack([r1 * np.cos(a), np.full(n, y1), r1 * np.sin(a)])
    pts = np.vstack([lo, hi, [[0, y0, 0], [0, y1, 0]]])
    c_lo, c_hi = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([4, i, j, n + j, n + i])
        faces.append([3, c_lo, j, i])
        faces.append([3, c_hi, n + i, n + j])
    return pv.PolyData(pts, faces=np.hstack(faces)).compute_normals(
        auto_orient_normals=True, inplace=False)


def bored_box(y0, y1, w, r_bore, bore_from_top=None):
    """A square block with a cylindrical bore up its axis.

    ``bore_from_top`` bores only that far down from ``y1`` (the beam dump's
    B-PE core has a 250 mm entrance hole, not a through hole).
    """
    box = M.slab((0, (y0 + y1) / 2, 0), w, w, y1 - y0, normal='y')
    yb0 = y0 if bore_from_top is None else max(y0, y1 - bore_from_top)
    bore = pv.Cylinder(center=(0, (yb0 + y1) / 2 + 2, 0), direction=(0, 1, 0),
                       radius=r_bore, height=(y1 - yb0) + 8, resolution=64,
                       capping=True)
    try:
        out = box.triangulate().boolean_difference(bore.triangulate())
        if out.n_points:
            return out
    except Exception:                                        # pragma: no cover
        pass
    return box                            # the bore is a detail, not the point


# --------------------------------------------------------------------------- #
# The longitudinal cutaway
# --------------------------------------------------------------------------- #
# Drawn whole, an opaque steel pipe hides the thing the figure exists to show:
# the neutron flux going up inside it, and the dump's three layers hide each
# other.  Drawn translucent, five nested shells turn into soup.  So everything
# is cut on a vertical plane through the axis with the near half removed --
# exactly the way scenes_ntof cuts the He-3 capsule open, and for the same
# reason.
_CUT = None


def set_cut(normal):
    """Set the cutaway plane's normal (the horizontal look direction)."""
    global _CUT
    if normal is None:
        _CUT = None
        return
    n = np.array([normal[0], 0.0, normal[2]], float)
    _CUT = tuple(n / np.linalg.norm(n))


def cut(mesh):
    if _CUT is None or mesh is None:
        return mesh
    return mesh.clip(normal=_CUT, origin=(0, 0, 0), invert=True)


def add_cut(p, mesh, **kw):
    """``add_mesh(cut(mesh))``, but tolerant of the cut removing everything.

    Every other part in this scene straddles the beam axis, so the cutaway always
    leaves something behind.  A ring of separate wedge blocks does not: the ones
    on the far side of the cut plane survive whole, the ones on the near side are
    removed *entirely*, and VTK raises on an empty mesh.
    """
    m = cut(mesh)
    if m is not None and m.n_points:
        p.add_mesh(m, **kw)


def _tilt_cut(mesh, y, sign, tilt=BREAK_TILT):
    """Slice a pipe section on a tilted break plane (sign +1 = keep below)."""
    th = np.radians(tilt)
    return mesh.clip(normal=(np.sin(th) * sign, np.cos(th) * sign, 0.0),
                     origin=(0, y, 0), invert=True)


def _cut_face(y, sign, radius=None, tilt=BREAK_TILT):
    """The pipe's cross-section at a break, drawn as a face.

    ``clip`` leaves the pipe open, and an open tube end reads as a pipe that
    simply stops -- which is the one thing the break must not look like.  A thin
    filled disc on the break plane, in the deck's caution accent, turns the two
    ends into a *section* instead, so the gap reads as "the drawing is cut here"
    without needing the label to say it.

    ``tilt`` is 0 everywhere now; see BREAK_TILT for why the slanted version was
    withdrawn.  The parameter stays because the rotation is what makes the face
    coincide with whatever plane ``_tilt_cut`` used, and the two must not drift
    apart.
    """
    r = R_PIPE + 6.0 if radius is None else radius
    face = M.polygon_prism(_ring(r), -5.0, 10.0, normal_axis='y')
    R = np.eye(4)
    th = np.radians(tilt) * sign
    c, s = np.cos(th), np.sin(th)
    R[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    R[1, 3] = y
    return face.transform(R, inplace=False)


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #
# The three slabs are three separate parts, not one ``hall``, because each
# belongs to a different frame of the build-up: the target stands on the first,
# the dump stands on the third, and the second is the EAR2 floor the apparatus
# is referenced to.
def add_hall(p):
    """The target-hall floor, under the spallation target."""
    y0, _ = act_span(0)
    p.add_mesh(cut(M.slab((0, y0 + 90, 0), HALL_W, HALL_W, 180, normal='y')),
               **S.mat('plastic', COL['concrete'], opacity=1.0))


def add_floor(p):
    """The EAR2 floor -- its top face is the 18.16 m reference."""
    yf = y_of(FACTS['h_floor'])
    p.add_mesh(cut(M.slab((0, yf - FLOOR_T / 2, 0), FLOOR_W, FLOOR_W, FLOOR_T,
                          normal='y')),
               **S.mat('plastic', COL['floor'], opacity=1.0))


def add_target(p):
    """The lead spallation target, at true size, on the proton axis.

    This is **Target #3**, the one our data were taken on, from Esposito et al.
    PRAB 24 (2021) 093001 -- see the TGT_XY block for the citation and for what
    was wrong with the cylinder that used to be here.  Six square lead slices on
    the proton axis, aluminium anti-creep plates between them, a stainless vessel
    around the lot, and then the part this figure exists to show: the 4 mm
    neutron window, the 5 cm lead plate, the 4 cm water moderator and the
    hemispherical vacuum window, which is everything a neutron on its way to
    EAR2 crosses before it is in the pipe.

    Four things in here are drawn rather than sourced and ASSUMPTIONS names them:
    the vessel's wall thickness, the moderator cans' wall thickness and plan
    size, the EAR1 moderator's extent, and the vacuum window's radius.
    """
    y = y_of(0.0)
    box = lambda x0, x1, y0, y1, xy: M.slab(
        ((x0 + x1) / 2, y + (y0 + y1) / 2, 0.0),
        x1 - x0, xy, y1 - y0, normal='y')

    # --- the core: six slices, thick one downstream, plates between them ------
    x = -TGT_LEN / 2.0
    for i, t in enumerate(TGT_SLICES):
        p.add_mesh(cut(box(x, x + t, -Y_CORE, Y_CORE, TGT_XY)),
                   **S.mat('alu_matte', COL['lead'], opacity=1.0))
        x += t
        if i < len(TGT_SLICES) - 1:
            p.add_mesh(cut(box(x, x + ACP_T, -Y_CORE, Y_CORE, TGT_XY)),
                       **S.mat('alu', COL['al'], opacity=1.0))
            x += ACP_T

    # --- the AISI 316L vessel, and the cradle that carries the core -----------
    # Translucent, or it is an opaque steel box with the whole subject inside it.
    xv = TGT_LEN / 2.0 + VES_GAP
    p.add_mesh(cut(box(-xv, xv, -Y_VES, Y_VES, TGT_XY + 2 * VES_GAP)),
               **S.mat('alu_matte', COL['steel'], opacity=0.16))
    p.add_mesh(cut(box(-xv, xv, -Y_VES - CRADLE_T, -Y_VES,
                       TGT_XY + 2 * VES_GAP)),
               **S.mat('alu', COL['al'], opacity=0.85))

    # --- and up towards EAR2 --------------------------------------------------
    p.add_mesh(cut(box(-xv, xv, Y_VES, Y_NWIN, TGT_XY + 2 * VES_GAP)),
               **S.mat('alu_matte', COL['steel'], opacity=1.0))
    p.add_mesh(cut(box(-MOD_XY / 2, MOD_XY / 2, Y_NWIN, Y_PB, MOD_XY)),
               **S.mat('alu_matte', COL['lead'], opacity=1.0))
    # the moderator can: aluminium walls with the 4 cm water layer inside it
    p.add_mesh(cut(box(-MOD_XY / 2, MOD_XY / 2, Y_PB, Y_MOD, MOD_XY)),
               **S.mat('alu', COL['al'], opacity=0.32))
    p.add_mesh(cut(box(-MOD_XY / 2 + MOD_WALL, MOD_XY / 2 - MOD_WALL,
                       Y_PB + MOD_WALL, Y_MOD - MOD_WALL, MOD_XY - 2 * MOD_WALL)),
               **S.mat('gas', COL['water'], opacity=0.72))
    # the hemispherical aluminium vacuum window the neutrons leave through
    dome = pv.Sphere(radius=R_VACWIN, center=(0, y + Y_MOD, 0), theta_resolution=48,
                     phi_resolution=48).clip(normal=(0, -1, 0),
                                             origin=(0, y + Y_MOD, 0),
                                             invert=True)
    p.add_mesh(cut(dome), **S.mat('alu', COL['al'], opacity=0.45,
                                  smooth_shading=True))

    # --- the EAR1 moderator, downstream: the larger of the two ---------------
    # It is what the 15 cm slice sits against, so leaving it out would make the
    # slice thicknesses look arbitrary.  Its extent is drawn, not sourced.
    # Deliberately quiet: aluminium only, no water fill.  Filled to match the
    # EAR2 can it is a 0.3 m^3 block of blue on the downstream face, and it then
    # outweighs the 40 mm layer that this figure is actually about -- which is
    # backwards, EAR1 being 185 m in the other direction and irrelevant here.
    p.add_mesh(cut(box(xv, xv + EAR1_MOD_X, -Y_CORE, Y_CORE, TGT_XY)),
               **S.mat('alu', COL['al'], opacity=0.20))


def add_protons(p):
    """20 GeV/c protons arriving horizontally, with the head at the target."""
    y = y_of(0.0)
    for m in M.tracks_with_heads([((-1600, y, 0), (-TGT_L * 0.45, y, 0))],
                                 radius=22.0, head_len=180.0,
                                 head_radius=62.0):
        p.add_mesh(m, **S.mat('glow', COL['proton'], opacity=1.0))
    p.add_mesh(pv.Cylinder(center=(-1050, y, 0), direction=(1, 0, 0),
                           radius=46.0, height=1100, resolution=48),
               **S.mat('envelope', COL['proton'], opacity=0.07))


def add_pipe_lo(p):
    """The pipe in the shaft, from just above the target up to the break.

    Split from the hall run (2026-08-11) because the build-up now shows the
    neutrons leaving the target before it shows any of the collimation: frame 2
    needs this segment and nothing above the break.
    """
    a0, a1 = act_span(0)
    p.add_mesh(cut(_tilt_cut(annulus(y_of(H_PIPE_START), a1 + 300, R_BORE, R_PIPE),
                             a1, +1)),
               **S.mat('alu_matte', COL['steel'], opacity=1.0))
    # The lower break, sectioned.  ``clip`` leaves the pipe open, and an open
    # tube end reads as a pipe that simply stops -- which at a BREAK is the one
    # thing it must not look like, so the tilted accent face turns it into a
    # section instead.  (The break's upper face belongs to the collimator, not
    # the pipe, and add_collimator draws it.)
    p.add_mesh(cut(_cut_face(BREAKS[0][0], +1)),
               **S.mat('alu_matte', COL['section'], opacity=1.0))
    # Flanged joints: what makes a grey tube read as a beam pipe.
    for h in (0.78, 1.08):
        y = y_of(h)
        p.add_mesh(cut(annulus(y - 30, y + 30, R_PIPE - 2, R_PIPE + 54)),
                   **S.mat('alu', COL['flange'], opacity=1.0))


def add_pipe_hi(p):
    """The stepped run across EAR2, from the collimator exit to the exit window.

    There is no pipe between ``H_PIPE_END`` and ``H_UP0`` -- the lower line ends
    ~1 m above the EAR2 floor, the experimental space above it is open, and a
    separate wide section (``add_uppipe``) takes the beam on to the dump.  So the
    only cut in this part is the lower break in the vertical scale.

    Across the hall the pipe is drawn STEPPED and tapered, from the photograph:
    the lead-disk chamber at the shaft's bore, a reducer, then a narrow tube to
    the exit window.  See the H_CHAMBER_TOP block for what that is and is not.
    """
    y_end = y_of(H_PIPE_END)
    segs = []
    # part 1: the collimator exit through the floor into the vacuum chamber that
    # carries the lead disks, all at the shaft's bore
    segs.append(annulus(y_of(FACTS['h_coll2'][1]), y_of(H_CHAMBER_TOP),
                        R_BORE, R_PIPE))
    # part 3: the narrow tube.  Not clipped -- ``annulus`` is a closed prism, so
    # it already ends in the flat annular face an end of pipe has.
    segs.append(annulus(y_of(H_NECK), y_end, R_NECK, R_NECK_OUT))
    # Phong, not PBR: the studio cubemap reflected off the pipe's *concave*
    # inner wall (which is what the cutaway shows) reads as wood grain.
    for s in segs:
        p.add_mesh(cut(s), **S.mat('alu_matte', COL['steel'], opacity=1.0))
    # part 2: the reducer between them
    p.add_mesh(cut(conical_shell(y_of(H_CHAMBER_TOP), y_of(H_NECK),
                                 R_BORE, R_PIPE, R_NECK, R_NECK_OUT)),
               **S.mat('alu_matte', COL['steel'], opacity=1.0))
    # The end of the pipe: a flange ring and the exit window across the bore.
    # Both are here to make the termination read as DELIBERATE at slide size --
    # a bare tube end and a break's section face look alike from the back row,
    # and this one is real hardware, so it gets hardware colours and a square
    # end rather than the breaks' tilt and accent.
    p.add_mesh(cut(annulus(y_end - 48, y_end, R_NECK_OUT - 2, R_NECK_OUT + 46)),
               **S.mat('alu', COL['flange'], opacity=1.0))
    p.add_mesh(cut(M.polygon_prism(_ring(R_NECK - 3.0), y_end - 10.0, 6.0,
                                   normal_axis='y')),
               **S.mat('plastic', COL['flange'], opacity=0.30))
    # 18.08 m is load-bearing rather than decorative: it closes the bottom of the
    # hall segment onto the collimator's exit face, so that in a frame where the
    # collimator is not drawn the segment ends in a joint instead of hanging in
    # the air as an open tube.
    y = y_of(18.08)
    p.add_mesh(cut(annulus(y - 30, y + 30, R_PIPE - 2, R_PIPE + 54)),
               **S.mat('alu', COL['flange'], opacity=1.0))
    # the joint at the top of the lead-disk chamber, just under the reducer
    y = y_of(H_CHAMBER_TOP)
    p.add_mesh(cut(annulus(y - 52, y, R_PIPE - 2, R_PIPE + 54)),
               **S.mat('alu', COL['flange'], opacity=1.0))


def add_shield(p):
    """The white shielding around the floor penetration, from the photograph.

    A segmented disc lying on the EAR2 floor with a collar around the pipe above
    it.  Drawn in ``COL['pe']``, a warm white -- close to the collimator's borated
    PE, and deliberately far from the floor slab it stands on.  The material is
    read off a photograph, not a drawing, and ASSUMPTIONS says so.

    It is worth drawing because it is the one piece of the hall a viewer can
    match between the render and the photograph beside it, and because without
    it the pipe comes out of a bare slab, which is not what the floor looks
    like.
    """
    yf = y_of(FACTS['h_floor'])
    for seg in pie_segments(yf, y_of(H_SHIELD_DISC), R_PIPE + 24.0, R_SHIELD):
        add_cut(p, seg, **S.mat('plastic', COL['pe'], opacity=1.0))
    p.add_mesh(cut(annulus(y_of(H_SHIELD_DISC), y_of(H_SHIELD_COLLAR),
                           R_PIPE + 6.0, R_SHIELD_COLLAR)),
               **S.mat('plastic', COL['pe'], opacity=1.0))


def add_uppipe(p):
    """The wide pipe above the experimental space, on up to the beam dump.

    In the photograph this is the largest thing on the line: above the
    experiments the beam goes back into a pipe, much wider than the narrow tube
    below, and that section runs on up through the bunker ceiling to the dump.

    It is CUT OFF BY THE TOP OF THE FRAME rather than ended -- the drawing stops
    above the station, the beam line does not.  That is why it is drawn past
    ``act_span(1)[1]``: the camera's own margin then does the cutting, which
    reads as "continues" in a way no drawn cap can.
    """
    y0 = y_of(H_UP0)
    y1 = act_span(1)[1] + 300.0           # past the top of the visible frame
    p.add_mesh(cut(annulus(y0, y1, R_UP, R_UP_OUT)),
               **S.mat('alu_matte', COL['steel'], opacity=1.0))
    # its entrance: a flange, and a window across the bore drawn like the one at
    # the end of the pipe below
    p.add_mesh(cut(annulus(y0, y0 + 66, R_UP_OUT - 2, R_UP_OUT + 60)),
               **S.mat('alu', COL['flange'], opacity=1.0))
    p.add_mesh(cut(M.polygon_prism(_ring(R_UP - 3.0), y0 + 14.0, 7.0,
                                   normal_axis='y')),
               **S.mat('plastic', COL['flange'], opacity=0.30))
    y = y_of(H_UP_RING)
    p.add_mesh(cut(annulus(y - 32, y + 32, R_UP_OUT - 2, R_UP_OUT + 54)),
               **S.mat('alu', COL['flange'], opacity=1.0))


def add_collimator(p):
    """The second collimator: 2 m Fe + 1 m borated PE, last 0.74 m drawn."""
    # 120 mm INTO the break, so the block reads as cut by it rather than as
    # starting at its edge.  Its section face goes at the same place -- drawn at
    # the act boundary instead, it floated 120 mm up inside the block.
    lo = act_span(1)[0] - 120.0
    y_fe = y_of(H_COLL2_FE)
    y_ex = y_of(FACTS['h_coll2'][1])
    p.add_mesh(cut(annulus(lo, y_fe, COLL2_BORE, COLL2_R)),
               **S.mat('alu_matte', COL['iron'], opacity=1.0))
    p.add_mesh(cut(annulus(y_fe, y_ex, COLL2_BORE, COLL2_R)),
               **S.mat('plastic', COL['bpe'], opacity=1.0))
    # The lower break cuts THROUGH the collimator, so the section face there is
    # drawn at the collimator's radius and with the collimator -- a pipe-sized
    # disc floating inside a 680 mm block would read as neither.  FLAT and inset
    # (2026-08-11): at this radius the break's usual tilt puts a third of the
    # disc outside the block, and it then reads as a pink diagonal fin bolted to
    # the collimator rather than as the block being cut off.  The tilt survives
    # on the pipe's own face below, where the disc is small enough to stay
    # inside its own silhouette.
    p.add_mesh(cut(_cut_face(lo, -1, radius=COLL2_R - 4.0, tilt=0.0)),
               **S.mat('alu_matte', COL['section'], opacity=1.0))


def add_lead_disks(p):
    """The in-hall lead collimation just above the EAR2 floor.

    Weiss Sec. 2.3: lead disks inside the first vacuum chambers after the second
    collimator's exit, 0.57 m of effective height by default, with inner
    diameters matched to the diverging beam -- "an extension of the collimation
    system".  Real hardware, and it happens to occupy the one stretch of the
    drawn line that would otherwise be blank pipe.
    """
    # 0.36 m of the documented 0.57 m, and stacked above the shielding collar
    # rather than starting at the floor: the drawn hall segment is only 1 m tall
    # and it also has to carry the collar, the reducer and the exit flange.  What
    # the figure is claiming is "there is more collimation in the hall, here",
    # not a length -- see ASSUMPTIONS.
    y0, y1 = y_of(H_SHIELD_COLLAR + 0.02), y_of(H_CHAMBER_TOP - 0.06)
    n = 4
    h = (y1 - y0) / n
    for i in range(n):
        r_in = 16.0 + 4.0 * i
        p.add_mesh(cut(annulus(y0 + i * h + 10, y0 + (i + 1) * h - 10,
                               r_in, R_BORE - 6.0)),
                   **S.mat('alu_matte', COL['lead'], opacity=1.0))


def add_neutrons_lo(p):
    """The flux leaving the target: everything the pipe subtends, going up.

    Split from the collimated pencil above (2026-08-11) so that the build-up can
    say "and the neutrons leave at 90 deg" one frame before it says anything
    about collimation.  The contrast between this and ``add_neutrons_hi`` is the
    point of the whole figure: the target radiates into the full 317 mm bore, and
    what survives 20 m and two collimators is a few-centimetre pencil.
    """
    y_t = y_of(0.0)
    a1 = act_span(0)[1]
    # From the top of the water, not from the middle of the lead: the neutrons
    # are BORN in the lead but they reach the pipe through the moderator, and
    # an envelope starting inside the slices just tinted the lead.
    p.add_mesh(frustum(y_t + Y_MOD - MOD_WALL, a1 + 40,
                       TGT_XY * 0.22, R_BORE * 0.97),
               **S.mat('envelope', COL['envelope'], opacity=ENV_ALPHA))
    # a handful of neutrons, so the direction of travel is not a caption.  Drawn
    # thicker than the tracks above the collimator (12 vs 10 mm) and in the
    # darkened COL['neutron']: these five are the only thing in the frame that
    # says "at 90 deg to the protons", and they have to say it from the back row
    # against a tinted pipe interior.
    rng = np.random.default_rng(7)
    for k in range(5):
        phi = 2 * np.pi * k / 5 + 0.4
        r = R_BORE * 0.92 * np.sqrt(rng.uniform(0.10, 1.0))
        a = np.array([0.0, y_t, 0.0])
        b = np.array([r * np.cos(phi), a1, r * np.sin(phi)])
        for m in M.tracks_with_heads([(a, b)], radius=12.0, head_len=110.0,
                                     head_radius=30.0):
            p.add_mesh(m, **S.mat('glow', COL['neutron'], opacity=1.0))


def add_neutrons_hi(p, fade_sample=False):
    """The collimated pencil: out of the collimator, on out of the top of frame.

    ``fade_sample`` drops the envelope and the tracks to a whisper across the
    0.5 m the sample occupies.  It is the one thing in this scene that is NOT
    independent of what else is shown, and it is deliberate: drawn at full
    strength the beam is a grey line straight down the middle of a translucent
    capsule, and Dylan read the result as the capsule being behind the beam
    rather than in it.  So the frame that introduces the sample also gets out of
    its way.  Every other part is drawn identically in every frame.
    """
    b1 = act_span(1)[1]
    y_ex = y_of(FACTS['h_coll2'][1])
    ys = y_of(FACTS['h_station'])
    top = b1 + 300.0

    # above the collimator: the 21.8 mm exit, diverging slowly.  It carries on
    # past the end of the pipe, across the open experimental space, into the wide
    # pipe above it and out of the top of the frame, because that is what the beam
    # does -- the dump is 4.8 m above the station.
    if not fade_sample:
        p.add_mesh(frustum(y_ex, top, COLL2_BORE, 26.0),
                   **S.mat('envelope', COL['envelope'], opacity=ENV_ALPHA))
    else:
        lo, hi = ys - SAMPLE_BAND, ys + SAMPLE_BAND
        f = lambda y: COLL2_BORE + (26.0 - COLL2_BORE) * (y - y_ex) / (top - y_ex)
        for y0, y1, op in ((y_ex, lo, ENV_ALPHA), (lo, hi, ENV_FADE),
                           (hi, top, ENV_ALPHA)):
            p.add_mesh(frustum(y0, y1, f(y0), f(y1)),
                       **S.mat('envelope', COL['envelope'], opacity=op))

    # above the collimator the beam is a few cm across, i.e. thinner than a line
    # at this scale -- so the tracks in it are drawn at a legible width instead.
    # The heads go in the OPEN space above the end of the pipe, where they are on
    # clean background at their full size and where they say the thing this part
    # of the figure is for -- the beam crosses the hall in air.  The tracks then
    # carry on past them, through the sample and into the wide upper pipe, and out
    # of the top of the frame with it.
    _beam_tracks(p, y_ex, top, 13.0, head_at=y_of(19.62),
                 fade=(ys - SAMPLE_BAND, ys + SAMPLE_BAND) if fade_sample
                 else None)


def _beam_tracks(p, lo, hi, r, head_at=None, fade=None):
    """Three neutron tracks up the axis, drawn wider than the beam really is.

    ``head_at`` puts the arrow heads part-way up and lets the tracks carry on
    past them, which is what a beam that keeps going looks like.  It exists
    because the two jobs are not in the same place: the heads have to go where
    they are LEGIBLE -- open background, at full size -- and the tracks have to
    go where the beam goes, which here is off the top of the frame inside a pipe.

    ``fade`` is a (y0, y1) band drawn at a whisper instead of at full strength --
    see ``add_neutrons_hi``.
    """
    def seg(a, b, op):
        p.add_mesh(M.tube(a, b, 10.0),
                   **S.mat('glow', COL['neutron'], opacity=op))

    for j in (-1, 0, 1):
        off = j * r * 0.8
        a = np.array([off, lo, off * 0.4])
        b = np.array([off * 1.3, hi, off * 0.5])
        at = lambda y: a + (b - a) * ((y - lo) / (hi - lo))
        mid = at(head_at) if head_at else b
        for m in M.tracks_with_heads([(a, mid)], radius=10.0, head_len=110.0,
                                     head_radius=25.0):
            p.add_mesh(m, **S.mat('glow', COL['neutron'], opacity=0.95))
        if head_at is None:
            continue
        if fade is None:
            seg(mid, b, 0.95)
        else:
            f0, f1 = at(fade[0]), at(fade[1])
            seg(mid, f0, 0.95)
            seg(f0, f1, 0.10)
            seg(f1, b, 0.95)


def _capsule_meshes():
    """The real He-3 vessel profile if the Geant4 repo is reachable, else a stub.

    The vessel is a G4Polycone in ``MX17_Full_Geant``; ``scenes_ntof`` already
    imports it, so reuse that rather than re-typing the profile.  This figure is
    schematic enough that a missing checkout must not stop it rendering, hence
    the fallback.
    """
    try:
        import scenes_ntof as N
        m = N.capsule_meshes()
        return [(m['cfrp'], 'plastic', COL['cfrp'], 0.95),
                (m['al'], 'alu', COL['al'], 0.95),
                (m['gas'], 'glow', COL['he3'], 0.85)]
    # SystemExit, not Exception: scenes_ntof raises SystemExit when the Geant4
    # checkout is missing, and SystemExit is not an Exception -- so the obvious
    # `except Exception` lets it through and kills the whole render.
    except (Exception, SystemExit):                          # pragma: no cover
        body = pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0), radius=11.5,
                           height=52.0, resolution=48)
        nose = pv.Cone(center=(0, 34, 0), direction=(0, -1, 0), height=18.0,
                       radius=11.5, resolution=48)
        return [(body.merge(nose), 'alu', COL['al'], 0.95),
                (pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0), radius=8.0,
                             height=46.0, resolution=48), 'glow', COL['he3'],
                 0.85)]


def _chamber_meshes():
    """One MX17 chamber, in its own frame: +z points AT the sample.

    Built here with ``meshes.rect_chamber`` rather than imported from
    ``scenes_ntof``: that module reads the real geometry out of the Geant4
    checkout, which this figure must not depend on, and at 400 mm across a
    six-metre frame nothing finer than "board, gas, window, frame" survives
    anyway.  ``drift_dir=-1`` puts the gas on the -z side, so with +z outward the
    drift volume faces the sample -- which is how the real pinwheel is set up.
    """
    k = PLATE_DRAW                        # every length together; see PLATE_DRAW
    return M.rect_chamber(center=(0.0, 0.0, 0.0),
                          pcb_size=(PLATE_SIZE * k, PLATE_SIZE * k),
                          # The active area is NOT square: 399.4 mm across the
                          # chamber, 360 mm along the beam, the short axis being
                          # the passivated one (measured 2026-08-11; the sim's
                          # SimConfig carries the same pair).  Local +y is the
                          # world +y here, i.e. the beam, so this order is the
                          # one that puts the short side along it.
                          active_size=(399.4 * k, 360.0 * k),
                          frame_size=((PLATE_SIZE + 34.0) * k,
                                      (PLATE_SIZE + 34.0) * k),
                          pcb_thick=PLATE_THICK * k, normal='z',
                          drift_gap=DRIFT_GAP * k, drift_dir=-1,
                          n_strips=N_STRIPS_DRAWN)


def add_station(p):
    """The measuring station: the He-3 sample and the Micromegas around it.

    Drawn with no support structure at all -- see ASSUMPTIONS.  The real station
    hangs in an aluminium-profile frame on the EAR2 floor; four tall uprights and
    a shelf were drawn here until 2026-08-10 and they were the least legible
    thing in the figure, four grey columns in front of the one object the figure
    is about.  Without them the sample sits in open beam -- which is also where it
    really sits, the pipe having ended 0.79 m below it.

    The chambers were plain teal slabs until 2026-08-11, and Dylan's reaction was
    *"I'm also not sure what the green is supposed to be there"* -- which is fair:
    a 400 mm square of PCB colour seen obliquely is a green box and nothing else.
    They are now drawn as chambers -- aluminium frame, readout board with STRIPS
    on it, 30 mm of lit drift gas facing the sample, and the entrance window as a
    tint -- because the strips are what makes a green square read as a detector at
    a glance.

    And **two of them, in section**, not four in a pinwheel (``STATION_ARMS``,
    ``_section_angles``): the pinwheel's own geometry puts two boards hard against
    the drawn capsule from either side, so the sample -- the thing the beam is
    for -- ends up as a sliver between two green rectangles.  Sectioned and set
    back, the sample is alone on the beam and the detectors are visibly watching
    it from a distance, which is the sentence this frame has to say.
    """
    ys = y_of(FACTS['h_station'])
    T = np.eye(4)
    T[0, 0] = T[1, 1] = T[2, 2] = CAPSULE_SCALE
    T[1, 3] = ys
    for mesh, mat, col, op in _capsule_meshes():
        p.add_mesh(cut(mesh.transform(T, inplace=False)),
                   **S.mat(mat, col, opacity=op, smooth_shading=True))

    m = _chamber_meshes()
    for th in _arm_angles():
        n = np.array([np.cos(th), 0.0, np.sin(th)])
        u = np.array([-np.sin(th), 0.0, np.cos(th)])
        R = np.eye(4)
        R[:3, 0], R[:3, 1], R[:3, 2] = u, (0, 1, 0), n
        R[:3, 3] = n * PLATE_R + np.array([0.0, ys, 0.0])
        place = lambda key: m[key].transform(R, inplace=False)
        put = lambda key: cut(place(key))

        # A chamber the camera looks THROUGH is drawn as its FRAME ALONE, and
        # drawn UNCUT.  Both halves of that matter:
        #
        #   * uncut, because the cutaway plane passes through the beam axis and
        #     the two near arms lie entirely on its near side -- run through
        #     ``cut`` they are not faded, they are DELETED, which is why the
        #     first version of the 4-arm variant was pixel-identical to the
        #     2-arm one, and
        #   * frame alone, because ``frame_ring`` is a hollow rectangular band:
        #     you can count four chambers and still see the sample through the
        #     two nearest.  Four solid boards around a 160 mm sample leave
        #     nothing to see, and a faded solid board is worse -- it greys the
        #     sample out instead of framing it.
        if _CUT is not None and float(np.dot(n, _CUT)) > 0.25:
            p.add_mesh(place('frame'),
                       **S.mat('alu_matte', COL['al'], opacity=NEAR_ALPHA))
            continue
        p.add_mesh(put('frame'), **S.mat('alu', COL['al'], opacity=1.0))
        p.add_mesh(put('pcb'), **S.mat('pcb', COL['pcb'], opacity=1.0))
        p.add_mesh(put('strips'),
                   **S.mat('alu_matte', COL['strip'], opacity=1.0))
        # The drift volume is the chamber's message, so it gets real presence:
        # from the sample you look into 30 mm of lit gas with the board behind it.
        p.add_mesh(put('gas'), **S.mat('gas', COL['gas'], opacity=GAS_ALPHA))
        # the entrance window is 40 um of mylar -- a tint on the front face, not
        # a wall; drawn any heavier it hides the sample behind it
        p.add_mesh(put('cathode'),
                   **S.mat('plastic', COL['window'], opacity=0.12))


# --------------------------------------------------------------------------- #
# Anchors for the 2-D label pass
# --------------------------------------------------------------------------- #
def _detector_anchor():
    """A point on the LEFT chamber of the section pair, for the label leader.

    Always the section pair, never ``_arm_angles()``, even when the pinwheel is
    the thing being drawn: the label sits in the left-hand text column, and its
    leader has to end on the left.  (In the pinwheel alternate it then lands on
    the near-left arm's frame rather than on a board, which is close enough --
    the pinwheel is the alternate, not the slide.)
    """
    th = _section_angles()[0]
    d = np.array([np.cos(th), 0.0, np.sin(th)]) * PLATE_R
    return (d[0], y_of(FACTS['h_station']) + PLATE_SIZE * PLATE_DRAW * 0.30,
            d[2])


def anchors():
    yb0, yb1 = BREAKS[0]
    return dict(
        protons=(-1300.0, y_of(0.0), 0.0),
        target=(0.0, y_of(0.0) - TGT_R * 0.55, 0.0),
        # The 4 cm water layer, which is worth its own label now that it is
        # drawn: the resolution function of EAR2 is "purely dictated by the
        # geometry of the moderator" (Esposito Sec. II B), and it is the one part
        # of the target a neutron on its way here has to go through.  Anchored
        # off-axis so its leader does not run down the beam.
        moderator=(TGT_XY * 0.30,
                   y_of(0.0) + (Y_PB + Y_MOD) / 2.0, 0.0),
        neutrons=(R_BORE * 0.55, y_of(0.95), 0.0),
        gap_lo=(0.0, (yb0 + yb1) / 2, 0.0),
        collimator=(COLL2_R * 0.82, y_of(17.75), 0.0),
        # The white shielding on the floor, and the floor itself: on OPPOSITE
        # sides of the pipe, and pulled apart radially, or the two leader lines
        # land on the same pixel.  The shielding's anchor is at -x, i.e. screen
        # LEFT, which is the side its label is on (2026-08-11, Dylan) -- with
        # both on the right the leader had to cross the beam pipe to reach a
        # part that is present on both sides of it anyway.
        shield=(-R_SHIELD * 0.62, y_of(18.24), 0.0),
        floor=(FLOOR_W * 0.44, y_of(FACTS['h_floor']), 0.0),
        pipe_end=(R_NECK_OUT * 1.6, y_of(H_PIPE_END), 0.0),
        # two labels where there used to be one (2026-08-11, Dylan): the thing
        # the beam hits and the things that watch it are different objects and
        # the slide should not make the audience work that out.  ``sample`` is on
        # the capsule itself, ``detectors`` on a chamber board.
        sample=(0.0, y_of(FACTS['h_station']) - 40.0, 0.0),
        detectors=_detector_anchor(),
        # the wide pipe the beam goes on up in -- this label is all that is left
        # of the beam dump, so it has to carry the height too
        to_dump=(R_UP_OUT * 1.05, y_of(20.86), 0.0),
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
PARTS = ('hall', 'target', 'protons', 'pipe_lo', 'neutrons_lo',
         'pipe_hi', 'neutrons_hi', 'floor', 'shield', 'collimator', 'lead',
         'uppipe', 'station')

# The build-up: three STRICT SUBSETS of PARTS, in the order they appear, with the
# label keys (see anchors()) each frame is allowed to carry.  Each stage is the
# previous one plus what is listed, so the frames overlay exactly and a label
# never moves -- make_ear2.py lays the whole column out once and then draws the
# subset.
STAGE_PARTS = [
    # 1. the source: protons in, lead target, the hall floor under it
    ('target', ('hall', 'target', 'protons'),
     ('target', 'moderator', 'protons')),
    # 2. and the neutrons leave, at 90 deg, filling the pipe.  The break appears
    #    with the pipe, so its label belongs here.
    ('neutrons', ('pipe_lo', 'neutrons_lo'), ('neutrons', 'gap_lo')),
    # 3. what makes the beam: the second collimator, the lead disks, the EAR2
    #    floor and its shielding, and the pipe ending a metre above it
    ('collimation', ('pipe_hi', 'neutrons_hi', 'floor', 'shield', 'collimator',
                     'lead'),
     ('collimator', 'shield', 'floor', 'pipe_end')),
    # 4. and back into a pipe, on up to the dump above the frame
    ('dump', ('uppipe',), ('to_dump',)),
    # 5. the experiment: the sample in the open beam, and the trackers round it
    ('station', ('station',), ('sample', 'detectors')),
]

def stage_parts(i):
    """The parts shown in build frame ``i`` (1-based) -- cumulative."""
    out = []
    for _, parts, _ in STAGE_PARTS[:i]:
        out += list(parts)
    return tuple(out)


def stage_labels(i):
    """The label keys build frame ``i`` (1-based) carries -- cumulative."""
    out = []
    for _, _, keys in STAGE_PARTS[:i]:
        out += list(keys)
    return set(out)


def build(p, show=PARTS, cut_normal=None):
    set_cut(cut_normal)
    show = set(show)
    if 'hall' in show:
        add_hall(p)
    if 'target' in show:
        add_target(p)
    if 'protons' in show:
        add_protons(p)
    if 'pipe_lo' in show:
        add_pipe_lo(p)
    if 'pipe_hi' in show:
        add_pipe_hi(p)
    if 'uppipe' in show:
        add_uppipe(p)
    if 'floor' in show:
        add_floor(p)
    if 'shield' in show:
        add_shield(p)
    if 'collimator' in show:
        add_collimator(p)
    if 'lead' in show:
        add_lead_disks(p)
    if 'neutrons_lo' in show:
        add_neutrons_lo(p)
    if 'neutrons_hi' in show:
        add_neutrons_hi(p, fade_sample='station' in show)
    if 'station' in show:
        add_station(p)
    return anchors()


def scene_center():
    return (0.0, DRAWN_H / 2.0, 0.0)


def scene_scale():
    """Characteristic radius of the drawn line [mm], for the light rig."""
    return DRAWN_H / 3.0
