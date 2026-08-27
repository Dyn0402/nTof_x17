# mpgd26 — 3-D setup renderings for the MPGD 2026 conference

Publication-grade 3-D views of the two setups:

* **SPS H4 beam telescope** (P2 zone) — three P2 BASKET fans between two EIC
  uRWELL references, optionally with MX17 "Detector E".
* **Saclay cosmic bench** — four M3 reference Micromegas around the P1/P2 test
  slots, triggered by a top/bottom scintillator coincidence.

…and the physics case that motivates them:

* **X17 signature** — capture → the three de-excitation channels → the
  opening-angle distribution that separates the hypothesis from the known
  channel. A diagram, not a render, but it shares the palette.

Everything is driven from the run configs and the measured records, so a figure
can be re-rendered from a different angle, at a different size, in a different
theme, or with a different chamber in a slot, without redrawing anything.

Start with **`report.html`** — it carries the geometry tables, the figure set
and the list of what is drawn but not measured.

## Quick start

**Not all of these run off the bench.** Ten of the figure families need
`/media/dylan`, `~/CLionProjects` or one small exported file that `*.npz` keeps
out of git; [`slides/HANDOFF_offline_rebuild.md`](slides/HANDOFF_offline_rebuild.md)
says which, and what to copy.

```bash
cd mpgd26
../.venv/bin/python make_figures.py                # the setup stills
../.venv/bin/python make_chamber.py                # the exploded chamber
../.venv/bin/python make_microtpc.py --right waveforms   # micro-TPC operation (+ deck copy)
../.venv/bin/python make_share.py                  # sharing cartoon, kernels, model diagram, real-data split
../.venv/bin/python make_efficiency_breakdown.py   # loss budget, |r| tail, r<2 mm map (deck assets only)
../.venv/bin/python make_resolution.py             # angle correlation + sigma68 vs angle (deck assets only)
../.venv/bin/python make_x17.py --layout both --theme both   # physics-case diagrams
../.venv/bin/python make_x17.py --layout beats --capsule     # ...one beat per file
../.venv/bin/python make_x17.py --layout detect --slides     # ...the hand-over to the detector (slide 6.3)
../.venv/bin/python make_x17.py --layout outlook --slides    # ...and the Summary's outlook figure
../.venv/bin/python make_ear2.py                   # the n_TOF EAR2 beam line (5 frames)
../.venv/bin/python make_ntof.py                   # the n_TOF setup, 9 build frames
../.venv/bin/python make_ntof_plan.py --bare       # ...and the same thing as a plan
../.venv/bin/python make_target.py                 # the spallation target, in detail (2 views)
../.venv/bin/python make_photos.py                 # slide copies of the two station PHOTOGRAPHS
../.venv/bin/python make_anim.py                   # turntables + build-ups
../.venv/bin/python make_report.py                 # rebuild report.html
../.venv/bin/python make_status_plots.py           # the Status-section DATA plots
../.venv/bin/python make_flash_slides.py            # the reworked slides 28-30 flash figures (+ 1 backup)
../.venv/bin/python make_timeline.py               # the project timeline, in full (backup)
../.venv/bin/python make_campaign.py --slides      # mini timeline + the daily event census
../.venv/bin/python make_x17_rate.py --slides      # where the X17 rate is, with/without the dead time
../.venv/bin/python make_run145_pointing.py --slides  # the two closing pointing figures (run_145, arms A and C)

../.venv/bin/python make_figures.py --draft        # fast, for framing checks
../.venv/bin/python make_figures.py --theme both   # + dark theme
```

Light theme is the primary set; `--theme dark` works but has had less tuning.

Each figure is written twice:

* `figures/<name>_<theme>.png` — the bare render, for when you want to place
  your own labels in Keynote / Beamer;
* `figures/<name>_<theme>_labelled.png` and `.pdf` — titled, captioned and
  labelled, with the type set in matplotlib so the **PDF carries live text**
  (it scales to any slide or page without going fuzzy).

## The figure set

| name | what it is |
|---|---|
| `sps_hero` | three-quarter hero from downstream-left; every readout face towards the camera |
| `sps_hero_mx17` | the same, with MX17 at z = 1155 mm |
| `sps_side` | near-elevation — the spacing-along-the-rail figure |
| `sps_beam` | beam's-eye view; the fan's pad structure at its clearest |
| `bench_hero` | the bench as a rack, **both slots MX17** |
| `bench_side` | near-elevation with a slight lift — the stacking figure |
| `bench_p2` | the same bench, **both slots P2 BASKET fans** |
| `bench_p2_side` | elevation of the two-P2 configuration |
| `bench_mixed` | P2 fan in P1, MX17 in P2 (the 6-27 configuration) — available, not headline |
| `chamber_exploded` | one MX17 chamber pulled apart, with a muon and its drifting ionisation — **landscape**: a `44 × 34 mm` window on the chamber (`scenes_chamber.WIN_MM`) with the labels on the render down its left side, sized for the 56 % column of the deck's "Chamber design" slide. **The readout side is the as-built board**, re-sourced 2026-08-17 from `MX17_Geant` (`shared/MX17ModuleGeometry.hh` + the gerbers): 0.68 mm **L4 pads** on the 0.78 mm grid, 0.5 mm L5 (Y, along x) and L6 (X, along y) strips, and over them the **black** ESL film — 550 µm strips, 250 µm gaps, its own 0.80 mm pitch — with the colours and L-numbers of the board-peel figure beside it. `WIN_MM` and `EXPLODE` are one setting: they set the figure's aspect between them — the window went `120 × 30` → `60 × 18` (in, to resolve the strip structure) → `60 × 34` (deeper along the strips, so the layers read as *planes* and not ribbons; 48 mm clips the near corner at this camera) → **`44 × 34`** (in again across the strips, 56 of them on screen instead of 77). The last step is a true **magnification, not a crop**: `make_chamber.VIEW`'s `view_angle` came down 17.8 → 16.6° with it, so the frame width is unchanged and the deck's column weights did not have to move. It came with the muon — the track tube was 0.9 mm across, i.e. 1.2 strip pitches, drawn at the scale of the structure it crosses, and is now 0.30 with 0.10 drift lines and 15 smaller ionisation clusters. The deck copy `chamber_exploded_slide` drops the title/caption bands and is written straight into `slides/assets/img/`; `make_anim`'s turntable uses the centred `ANIM_VIEW`, not this one (`make_chamber.py`) |
| `microtpc` | **how the chamber measures an angle**: primaries drifting to the mesh coloured by arrival time, plus the strip-time ladder and its fit (`make_microtpc.py`) |
| `microtpc_waveforms` | the same event with the **raw per-strip waveforms** instead of the ladder — the measured impulse response, sampled at 32 × 60 ns (`make_microtpc.py --right waveforms`). **This is the variant the deck shows** since 2026-08-17: the ladder is one estimator built on the waveforms, and it is the estimator the forward-fit slides exist to replace. `compose(bare=True)` writes the deck copy — no title band, no caption paragraph (they were 36 % of the height), the operating point burned onto the render in three lines — straight into `slides/assets/img/microtpc.png` |
| `share_cartoon` | **the sharing mechanism as a drawing**: the avalanche onto the resistive film, sideways through the film's own sheet resistance, then down onto the strips — so the neighbours' copies are **late** (146 / 291 ns) and dispersed (`make_share.py`) |
| `share_kernels` | **the kernels production uses**, per plane, from the frozen det3 bundle: charge on the strip itself, and what ±1 and ±2 see. X 5 / 6 %, Y 15 / 17 % — the film's strips run along y, so Y shares ~3× more (kY = 2.9). ⚠️ **c₁ sits on its C1_MIN = 0.05 calibration floor on a cosmic fit** — read `make_share.py`'s docstring before quoting an amplitude off this figure |
| `share_build` | **what the model does**, four stages: 60 ns depth slices with free q ≥ 0 → the geometric strip integral → the kernel copies onto ±1/±2 → the fold with the measured impulse response |
| `share_decompose` | **the same split on real data**: four *consecutive* strips of event 1663 (Y plane), each fitted waveform stacked into own / ±1 / ±2 against the measurement. Exact, not estimated — the model is a sum of those three terms, so the split is a difference of three builds of the design matrix |
| `efficiency_breakdown` | det3's **loss budget** — where every crossing muon goes, every percentage read from `efficiency_breakdown.json`. **Not on a slide since 2026-08-17**: the deck draws the same five bars in HTML (`.bar-chart.loss`), because matplotlib set the sentence-length labels in its own font at the saved size and they arrived smaller than the deck's body text. Kept as the standalone/handoff copy — the two must agree bar for bar (`make_efficiency_breakdown.py`) |
| `efficiency_residual_tail` | the **|r| distribution**, core and tail, one panel and no burned-in title. The second panel (efficiency vs match radius) was dropped 2026-08-17 — it re-plotted this histogram's own cumulative. Saved at 5.6 × 3.5 in *because* it is shown at about that size: a 7.4 in figure displayed at 2 in has 4-pixel tick labels |
| `efficiency_map_sliding` | efficiency **across the chamber face**, as a **20 mm circle swept over it 0.5 mm at a time** — efficiency inside the circle is *reconstructed within 5 mm / all reference muons*, the same 5 mm the loss budget beside it uses (`make_efficiency_map.py`, on `g_det3_wknd`, the highest-statistics det3 set: **21,948** reference muons against `sat_det3`’s 7,049, and the two agree to 0.15 points). Replaced the 40 × 40 **binned** `efficiency_map_2mm` on 2026-08-18. ⚠️ **the kernel cannot be 2 mm**: at 0.16 rays/mm² a 2 mm circle holds two muons, and even a 12 mm one holds ~75, where a single missed muon paints its own 24 mm disc and the map becomes a field of blue circles that is pure counting noise. 20 mm holds ~224. The **0.5 mm step** is exactly as briefed and free — the map is an FFT convolution, not a double loop. Yellow = efficient (viridis), which is the one deviation from `plotstyle.efficiency_cmap()` in the deck |
| `angle_correlation` | **reconstructed against reference track angle**, X and Y, as a 2-D density with the line of equality — the plot that says the fit *measures* the angle rather than regressing to the mean of it. σ₆₈ 1.19° / 1.16°, bias < 0.1°, no `slope_reliable` gate (`make_resolution.py`) |
| `angle_resolution` | σ₆₈ of the same residual **in bins of |reference angle|**, both planes, against the ~1° physics floor. Flat at 0.94–1.48° from head-on to 18° — **including the head-on bin**, where a drift-time ladder has no lever arm at all. Replaces the 2026-07-14 hits-basis `angular_resolution.png`, which showed 1.66° |
| `x17_signature` | the physics case, compact: capture → three de-excitation channels → the e⁺e⁻ opening-angle distribution (`make_x17.py`) |
| `x17_story` | the same in five beats over two rows, including **why** the boost sets the opening angle (`make_x17.py --layout story`). ⚠️ **Re-flowed 2026-08-18 onto a 124-unit canvas** (`scenes_x17.SW`, against the compact layout's `W = 160`) so that each row alone is 2.16 : 1 — the shape of the figure hole on a deck slide. A slide figure is width-limited, so the number of canvas units a row spans is the *only* lever on how big its type and its drawing come out: 160 units across 12.4 in renders 9 pt type at 7 pt, 124 units renders it at 9 pt. Making these rows taller and making them bigger was the same operation. The two-row compilation is consequently **portrait** now. Do not widen `SW` back out without re-flowing the beats |
| `x17_story_1of2`, `x17_story_2of2` | the same five beats split across two slides — 1–3 then 4–5 (`--layout split`) |
| `x17_story_capsule` | the story layout with the real Geant4 ³He vessel in beat 1 (`--layout story --capsule`) |
| `x17_story_bot_3_detect` | **deck frame 6.3**: the bottom story row with the **micro-TPC cartoon in beat 4’s box** and the spectrum untouched beside it, so the frame changes the argument and not the picture the audience is reading (`make_x17.py --layout bot3`). It was two stacked full-width pictures in one figure box until 2026-08-18, which cost both ~41 % of their width. One claim — one gap gives a direction, two give the angle. The **opening angle is drawn true**; the 204 mm standoff, the 30 mm gap and the 400 mm chamber are not. The 21° tilt is on the **chamber**, not the track: square incidence puts all the charge at one depth and leaves a micro-TPC nothing to reconstruct. `--layout detect_solo` writes the cartoon on a canvas of its own |
| `x17_outlook` | **the Summary slide's figure** (2026-08-24): find the two-track events → histogram their opening angle (`make_x17.py --layout outlook`). The opening figure answered — the physics case says the observable is an angle, this says how the banked data become one. **Left is to scale**: four chambers, 204 mm standoff, 399 × 360 mm active, 90° apart, so a 110° pair visibly *cannot* land in one chamber while a narrow one can, and the audience can check it rather than be told. ⚠️ **It is a view from ABOVE and `mm()` mirrors x to make it one** — the beam is along +Y and EAR2's is vertical going up, so a plan view looks along −Y and, right-handed with +Z up the page, `X = Y × Z = left`. Drawn unmirrored (as it was until 2026-08-24) the figure is the station seen from *underneath*, with arms D and B on the wrong sides and the pinwheel the wrong handedness. `ray()` therefore takes **canvas** azimuths, not simulation ones; `_arm_hit` is what checks which arm a drawn leg really enters (canvas az → sim az is `180 − az`). **Right is one background with a breakdown under it** (re-emphasised 2026-08-24): the bold curve is *every* accepted IPC pair, whatever it hit; the two thin curves under it are the one- and two-chamber topologies that explain its shape — a one-chamber peak dying by ~95°, handing over to a **flat** two-chamber tail (the acceptance rises about as fast as the physics falls). X17 is drawn **only over the bump**, on top of the total, so the eye reads *background plus something above threshold* and not *a third curve*; the ~3° merging cutoff is the shaded band at the left. Neither component is clamped to the axis floor, so the one-chamber curve falls off the bottom of the frame where it dies instead of running along the axis as yield that is not there. ⚠️ **Computed vs drawn**: the channel shapes (MX17_Simulation generators) and the one-/two-chamber split (`scenes_x17.pair_acceptance`, ray-traced on the as-built geometry) are computed; the **X17 yield is drawn**, at a declared 30 % of the whole background above threshold — the relative rate is the measurement, not an input — and so is the 12 mm two-track separation the cutoff comes from, which is the single-track fit's merged-cluster limit and not a measured two-track efficiency. The X17 bump carries a big tilted **?** and the word *sketch* — placed off the drawn curve's actual apex, so it follows the bump — because a legend entry saying *(drawn, not predicted)* is not what a room looks at. Canvas 152 × 63 units = 2.42 : 1, the **measured** shape of the Summary slide's figure hole under three one-line bullets. ⚠️ **Two type scales, and they are not the canvas width**: `OUTLOOK_FS = 1.6` scales every label in the figure, `OUTLOOK_FS_SPEC = 1.18` scales the spectrum panel again on top of that. Narrowing the canvas (the `x17_story` lever) magnifies type *and* drawing together; these magnify the type alone, which is what a conference room actually needs. Raising `OUTLOOK_FS` further needs the station drawing to give ground — it is already at `sc = 0.083` |
| `x17_beat1_beam`, `x17_beat2_capture`, `x17_beat3_channels`, `x17_beat4_boost`, `x17_beat5_spectrum` | **the same five beats, one per file**, for dropping into slides individually — a build, another deck, a poster (`--layout beats`, or `--layout beat3` for one; `--capsule` applies to beat 1, exactly as in the story layout). Each is the story drawing cropped to its own beat, **not a redrawn version of it**: the beats keep their absolute coordinates and the canvas is cropped to the window that holds them, so an edit lands in the compilation and in the single file together, at the same size and in the same style — the identity is asserted by `x17_story_capsule_light.png` coming out **byte-identical** after the split was added. By default each beat keeps its **row's full height**, so beats dropped one after another land in exactly the register they have in the compilation; `--tight` trims each to its own ink instead and writes `…_tight_…` alongside |
| `ntof_build_1…9` | **the n_TOF setup, built up** — capsule → neutron → pair → +Micromegas (close) → (zoom out) +Micromegas → +SiPM wall → +plastics → (rotate overhead) plastics → +liquid scintillator (`make_ntof.py`) |
| `ntof_plan` | **the same setup and event as a plan** — orthographic, down the beam, 1:1 in both axes, with a dimension chain; a matplotlib drawing, not a render (`make_ntof_plan.py`). `ntof_plan_bare` drops the headline and the note, and is what the slide uses |
| `x17_signature_bare`, `x17_story_bare` | either layout with the title and caption bands cropped off (`--no-title`) |
| `ear2_build_1…5`, `ear2_onfig_1…5`, `ear2_beamline` | **the n_TOF EAR2 vertical beam line, built up in five frames**: **Target #3** and the 20 GeV/c protons — six 600 × 600 mm lead slices (5 × 50 mm + 150 mm) with aluminium anti-creep plates between them, the AISI 316L vessel, and on top of it the 4 mm neutron window, the 50 mm lead plate and the EAR2 moderator's 40 mm water layer, all from Esposito et al., *PRAB* **24** (2021) 093001 (rebuilt 2026-08-11; it used to be the retired Target #2 cylinder, with the EAR2-facing assembly not drawn at all) → the neutrons leaving at 90° and filling the pipe → the middle of the line, both collimators, the floor and its shielding, and the lower beam pipe *ending* about a metre above the EAR2 floor → back into a pipe, on up to the dump → the measuring station in the open beam, halfway between the two pipes. The drawing **stops ~1.15 m above the station** — the bunker ceiling and the beam dump are real but above the frame (removed 2026-08-11 so the station is not dwarfed by 2.7 m of shielding); the **wide upper pipe** that carries the beam there is drawn and cut off by the top of the frame. Cut open along the beam axis, with one annotated break that removes the empty pipe. Inside the hall the shape of the line — polyethylene shielding on the floor, the lead-disk chamber, a reducer, the narrow tube — is **scaled off the CERN photograph beside it**, not off a drawing; that and the station's deliberately omitted support frame are in `scenes_ear2.ASSUMPTIONS`. The measuring station is drawn as real chambers — aluminium frame, strip readout board, 30 mm of drift gas facing the sample — with **24 strips drawn rather than 512**, since the real 0.78 mm pitch is a quarter of a pixel here. ⚠️ **Nothing at the station is 1:1**: the capsule is drawn 5.5× oversize and the chambers 1.35× (`CAPSULE_SCALE`, `PLATE_DRAW`), their 330 mm standoff being the one real dimension left; frame 5 is written twice: **two chambers in section** (the default, and what the slides use — the pair is placed relative to the *camera*, so their drawn azimuths are 136° apart and not the real 90°, which `ASSUMPTIONS` says) and the true four-arm pinwheel (`STATION_ARMS`, `_4arm`). The frames are strict subsets of one picture — same camera, same scale, no label moves — with one deliberate exception, the beam faded across the sample on the last frame. **Two label layouts of the same render** are written every run: `ear2_onfig_*` puts the labels on the drawing's own background (left and right), `ear2_build_*` / `ear2_beamline_*` put them in a gutter column; frame 3 is also written under the plain `ear2_beamline` names (`make_ear2.py`) |
| `target3_layers`, `target3_cooling` | **the n_TOF Target #3 spallation target, in detail** — the two target backup slides (**46 and 47**), which are the deck's source of record for the target. *Layers*: the assembly cut open along the beam, cradle to vacuum window — six 600 × 600 mm lead slices (5 × 50 mm + one 150 mm, thick one downstream), 9.85 ± 0.05 mm aluminium anti-creep plates carrying the nitrogen channels, the AISI 316L vessel at 0.5 bar with its 3 mm proton window, and towards EAR2 the 4 mm neutron window, the 50 mm lead plate and the moderator's 40 mm water layer. The 10° beam-to-target yaw is visible here and **only** here — it is a yaw about the vertical axis, so it foreshortens to nothing on a view of a face. *Cooling*: one anti-creep plate exploded off its slice, with the 3 mm channels, the wedge obstructions that throttle the outer ones so the flow goes where the beam is, and the nitrogen path. Everything from R. Esposito et al., *Phys. Rev. Accel. Beams* **24** (2021) 093001, arXiv:2106.11242, except four wall thicknesses/extents the paper does not give — listed in `scenes_target.ASSUMPTIONS` and on both slides. ⚠️ Shared dimensions are **imported from `scenes_ear2`**, never re-typed: the facility figure draws the same object (`make_target.py`) |

The X17 diagrams are written straight out as `figures/<name>_<theme>.png` and
`.pdf` — no separate `_labelled` version, because their type is drawn in from
the start. **Prefer the PDF on a slide**: all of it is live text.

Which layout: `x17_signature` when the diagram shares a slide with something
else, `x17_story` when it gets a slide of its own. The story layout's fourth
beat is the part the compact one has to assert. The pair is always
back-to-back in the parent's rest frame, so the lab angle is *only* the boost,
and which way it is bounded depends on whether the parent outruns its own
leptons (crossover at m = √(2mₑE) ≈ 4.6 MeV):

| | X17, m = 16.8 | a 2 MeV IPC pair |
|---|---|---|
| β of parent | 0.58 — slower than the leptons | 0.995 — faster |
| bound | **≥ 109°**, reaches 180° | **≤ 11°**, closes to 0° |

Three worked orientations per channel make that visible: no orientation lets
X17 close below 109°, and none lets a light IPC pair open past 11°. Since IPC
draws its pair mass from dN/dM ∝ 1/M it gets a band for every mass, and those
bands between them are the smooth slope panel 5 shows under the X17 peak.

**One slide or two.** `--layout split` writes the same five beats as two
figures: beats 1–3 (which set up the physics and end on *"detect the e⁺e⁻
pair"*) and beats 4–5 (which derive the measurement from the boost). They are
**not crops** — each part is the same drawing seen through a different canvas
band, with its own title, subtitle and caption, so editing a beat updates the
combined figure and its slide together and there is no second layout to
maintain. The parts come out roughly 16:5, which fills the width of a slide
and leaves room for a title bar.

Beat 1 is generic (a beam and some ³He) because early in a talk the target
hardware has not been introduced. `--capsule` swaps in the real vessel, drawn
from the `He3Gas` / `He3Cap_Al` / `He3Cap_CFRP` polycones in
`~/CLionProjects/MX17_Full_Geant/src/DetectorConstruction.cc` (sectioned from
the STEP solid), true aspect, mounted nose-first as the simulation mounts it.

## The n_TOF setup, built up (`make_ntof.py`)

```bash
../.venv/bin/python make_ntof.py                 # the nine build frames
../.venv/bin/python make_ntof.py --only full     # just the last one
../.venv/bin/python make_ntof.py --view top      # a different camera
../.venv/bin/python make_ntof.py --draft         # small and fast
```

Nine frames of one figure: the ³He capsule, a neutron reaching it, the e⁺e⁻
pair leaving it, and then the four detector layers going on one at a time.
Slides 16–24 of the talk are these nine frames with the step's explanation
beside them; flipping forward assembles the detector.

**Four cameras, not nine.** The subject changes scale by a factor of fifty —
the capsule is 23 mm across and the setup is 1.2 m — and then changes what it
has to show, so the sequence runs in four acts: `micro` (frames 1–3, the vessel
and the event in it, portrait), `close` (frame 4, the chambers arriving around
it), `hero` (frames 5–7, the apparatus assembled) and `over` (frames 8–9,
straight down — 89°, since 90° leaves the camera with no defined roll). The last cut earns itself: the layers are stacked
**radially**, so from any three-quarter view the trigger wall stands in front
of the plastics and the liquid and a leg arriving in them cannot be seen. That
act also runs **bare** — frames, boards and vessel shells drop to a whisper and
only the active volumes keep their colour, because looking down the beam the
four arms are seen through each other and the aluminium becomes a lid.
Within each act the frames share a camera and a size exactly, so the layers
grow onto a still picture. Frame 4 repeats frame 5's content at a larger scale
and is the first to cut if the section runs long.

⚠️ The overhead act draws the capsule **whole** (`VIEWS['over']` sets
`cut=False`). Everywhere else the vessel is sectioned on a plane through the
beam axis with the near half removed, which is the only way to show 0.6 mm of
wall and a lit gas core at the same time — but that plane *contains* the
overhead view direction, so from up there it does not open the vessel, it
deletes the half of it nearest the bottom of the frame and the capsule reads as
broken. It is also the one thing `BARE` does **not** whisper: everything else
that drops to `BARE_ALPHA` up there is a box the eye can still infer from its
neighbours, but the capsule is 23 mm on a 1.2 m frame and whispered it is a
smudge at the exact point the picture converges on. Solid, it is a small dark
disc — the CFRP overwrap seen end-on, with the gas bore a speck at its centre —
where the two legs meet.

**The frames carry their own labels.** The first one names the vessel's layers —
the wall is 0.6 mm of Al under 0.9 mm of CFRP on a 20 mm bore, which at slide
size is two thin bands that nothing else identifies. Every *build* frame names
**the layer it just added**, and only that one: the label moves outward with the
build rather than accumulating, so the picture says what it is showing without
the audience having to find it in the bullets. The text is pinned to a corner of
the frame (`LAYER_LABEL`, `LAYER_POS`) — top-left on the build acts, and at the
**bottom** on the close-up, where the chambers fill the frame and the only empty
space left is the see-through one. **One label, a leader to each solid arm**: a
layer is four objects and the frame draws two of them solid, so a single leader
would quietly imply the label is about that one; which arms get a line is read
off the view's own `near`, so it cannot fall out of step with the ghosting. It is
a leader per drawn **object**, not per arm — `scenes_ntof.LAYER_PARTS` says the
plastics are two separate bars, so that label carries four, and its text drops
the "2 ×" the lines already say. The
anchors are that layer's own centre in each arm, taken from the geometry by
`scenes_ntof.layer_anchor` — so a label cannot end up pointing at where a layer
used to be. Sizes on these labels are in **centimetres**: they are read from
across a room, not quoted. The leaders are drawn in matplotlib over the render
at 1:1 (`make_ntof.overlay`) — by hand rather than by `annotate(arrowprops=)`,
since one block of type has to serve several anchors, so each line is struck
from the text's own bounding box towards its target and two leaders leave it
from the two edges that face their arms. The frame keeps its size and its alpha;
`--no-labels` turns all of it off. The package's usual label path, `annotate.compose`, is not
used here because it grows the canvas and writes an opaque page.

**The plastics are lavender, not the package's scintillator blue** (`style.COL`
still has that, and the bench figure still uses it). Here they sit two layers out
from a 30 mm slab of drift gas drawn in `#6fd0e8`, and a light blue beside a
light cyan reads as the same material twice; lavender is the one pastel left that
is far in hue from all of the gas, the gold trigger bars, the orange liquid and
the red/crimson tracks.

**What is drawn.** Only the charged pair: the event's bremsstrahlung gammas are
real but neutral, so on the picture they are rays crossing the frame with no
visible cause (`scenes_ntof.DRAWN_PARTICLES`). The two arms nearest the camera
(B in front, A on the right) are drawn as outlines, not as translucent solids —
four half-transparent layers still hide what is behind them, and in a different
colour at every overlap. The two the pair actually crossed (D, C) stay solid,
and on the overhead act everything that is not an active volume drops to
`BARE_ALPHA`. The camera azimuth, `NEAR_ARMS` and the extractor's
`--prefer-arms` are one decision in three places — change one and change all.

**The legs grow with the apparatus.** Each is cut where it first reaches a layer
that is not on the frame yet (`truncate_at_next_layer`), so a track runs on to
the trigger wall only on the frame that puts the trigger wall there. Drawn full
length from the start they assert three frames early that all of it is measured,
and the layer being added stops being the thing that changes.

**The beam** is a plain grey column at the sampled 90 % radius, with a slim dart
lying on the axis inside it for direction — an arrowhead on the beam itself has
to be bigger than the 23 mm target it points at before it reads at slide size,
and one standing off to the side is a second object to explain. The dart's
*length* comes from the frame (`spherical(..., arrow=, arrow_y=)`) and its girth
from the beam, so it can never grow out through the column; only the close-up
act asks for one. Once any detector is placed the column stops at the vessel's
**nose** (`CAPSULE_NOSE_Y`), with a short stub below it: drawn past the target it
runs through the middle of the apparatus and reads as a pole holding it up, and
drawn even alongside the capsule it sleeves a 23 mm object in translucent grey
and the target goes hazy instead of crisp.

**The neutron is drawn, not transported.** It is a straight line up the beam axis
to the pair vertex, running off the bottom of the frame. The real Geant4 history
is still selected and stored in the JSON, but it belonged to a different event
from the pair, so drawing it meant translating it onto this vertex and then
showing its own in-gas scattering — a fact about that neutron rather than about
this figure. The neutron *run* is still needed: the beam envelope is measured
from its sampled primaries.

The frames are written with an **alpha channel** so they sit on a slide of any
colour (`--opaque` keeps the theme background instead).

**Where the geometry comes from.** Every dimension is imported at run time from
`~/CLionProjects/MX17_Full_Geant/scripts/plot_geometry.py`, which is written
against the simulation's own `SimConfig.hh` — so the figure and the Geant4
model cannot drift apart. Point `MX17_FULL_GEANT` elsewhere if that repository
is not at the default path. Two things are drawn rather than measured, and both
are flagged in `report.html`: the liquid's fill dome is extruded at constant
height along the vessel, and the beam envelope is drawn at the radius the
simulation's own sampled primaries occupy.

One number is **typed here rather than imported**, because `plot_geometry` does
not carry it: the chambers' **440 mm outer frame square** (`scenes_ntof.FRAME_HU`;
as-built, CAD plus two gerber cross-checks, `MX17_Full_Geant/docs/
HANDOFF_ARM_GEOMETRY.md`). It used to be drawn as a constant 30 mm cheek around
the active area, which came to the same 440 while the active area was the
unsourced 38 × 34 cm. On **2026-08-11** the sim took the measured **39.9 × 36.0 cm**
(the short axis is the one *along the beam* — ~19 mm at each end of the strip
plane is passivated), and a constant cheek around that is 490 mm: too big to
fit, since the pinwheel has only 15.5–17.3 mm of tangential shift to buy
clearance with and 440 is exactly what it clears. The frame is therefore a fixed
square and the cheek is what is left over — 20.3 mm across the chamber, 40.0 mm
along the beam. If the active area is remeasured again, change the sim, not the
cheek.

**Where the event comes from.** The neutron and the pair are real Geant4
events, picked out of `--trajdump` step dumps by `tools/extract_ntof_event.py`
into `data/ntof_event.json`:

```bash
# in MX17_Full_Geant, after sourcing the Geant4 environment
build/mx17_full_sim -n 400  -t 1 --trajdump 400  --ipc 0 -s 20260810 -o pairs
build/mx17_full_sim -n 4000 -t 1 --trajdump 4000 -s 77 \
    --neutron data/fluxEAR2-Ph3_in_different_units.root \
             data/lamda2DvsEn_EAR2.root \
    --emin 1e-3 --emax 1000 -o neutrons

../.venv/bin/python tools/extract_ntof_event.py \
    --pairs /path/pairs_traj_t0.csv --neutrons /path/neutrons_traj_t0.csv
```

**The pair on the figures is one real Geant4 event**; the neutron that made it
is *drawn*, not simulated alongside it, and it has to be that way — the
radiative branch that forms the ⁴He* is ~10⁻⁸ of ³He(n,p)t, so no neutron run
will ever contain one. A real neutron event is still selected and stored in the
JSON, translated so its interaction point lands on the pair vertex (28.3 mm of
shift on the shipped one), and the neutron run is needed regardless because the
beam envelope is measured from its sampled primaries — but no scene draws that
history: what it added to the picture was its own in-gas scattering. The tool
records the pairing, the shift and the residual in the JSON's `provenance`
block, and the talk says it on the slide.

## The same setup as a plan (`make_ntof_plan.py`)

```bash
../.venv/bin/python make_ntof_plan.py           # titled, for report.html
../.venv/bin/python make_ntof_plan.py --bare    # no headline/note, for the slide
```

The tenth slide of the setup section, and the only figure in it that is **not a
render**. Every frame above it has perspective: the four arms sit at four
different distances from the lens, so every length is foreshortened by its own
amount and the distances can only be *written* in a caption. This one is
orthographic and 1:1 in both axes — the beam is the view axis, so the drawing
plane is the X-Z plane the apparatus is symmetric in, and a length measured off
the page is a real length.

**What the drawing can say that the renders cannot.** The **204.5 mm** standoff
every arm's window sits on, and the layer radii out to the vessels (330 / 410 / 487 mm), drawn as a dimension
chain on arm B — the one arm no leg crosses, and the numbers are that arm's:
the plastics and the vessel sit a few mm differently on each; and the **size of the target**,
which at 1:1 is a 23 mm dot in the middle of a 1.1 m apparatus — which is why
the vertex marker on it is deliberately tiny, a bigger one covering the one
object whose size is the point. Two mechanical facts also become visible: the
~16 mm pinwheel offsets, and that **two of the four liquid vessels are laid on
their side**, with their necks and PMTs pointing sideways into the plane
(`LS_ROT`) — from above the two orientations really do look different, upright
showing the pillow cross-section with its domed edges and laid-over showing the
flat flank of a 451 mm slab, so both are drawn as they are.

⚠️ **What the plan gives up is the beam axis.** Both legs also rise ~135 mm
along it before they reach the liquid, so the opening angle *as drawn* is 122°
while the space angle is 110°; the figure quotes both. Two small departures
from the 3-D frames, and both are legibility rather than geometry: the 40 µm
mylar window gets no line of its own (the stroke would be four times its
thickness, and in the only honest colour for it — red — it reads as a highlight
box in almost the positron's own colour, so the drift gas's own front edge
stands for it), and a deposit marker is drawn in the **depositing leg's**
colour rather than the layer's, since a lavender dot inside a lavender bar is
not on the page at all.

Geometry and event come from `scenes_ntof` — so from the same
`plot_geometry.py` and the same `data/ntof_event.json` as the renders, and the
two cannot drift apart. Every number on it, including the dimension-chain
labels, is computed from that geometry rather than typed in.

**This is the view where the chamber's outer size is not decoration.** The plan
is the only figure that shows all four arms in one plane, so it is the only one
where a chamber drawn too big visibly runs into its neighbour — see the note at
`scenes_ntof.FRAME_HU` before changing either the active area or the frame.

## Photographs of the real station (`make_photos.py`)

```bash
../.venv/bin/python make_photos.py           # rebuild the slide copies
../.venv/bin/python make_photos.py --list    # what is in photos/
```

Two pictures taken in EAR2 on **2026-08-10**: down into the assembled station
(the photographic answer to the plan above) and one arm from outside. The
full-resolution originals live in `photos/` under the names the camera gave
them — that is the point of the directory, since a Downloads folder is not a
durable location — and `make_photos.py` writes the 1125 × 1500 copies the deck
loads. No cropping is done: both are portrait phone frames with about a third
to spare, and the right crop depends on where they land.

⚠️ They are in the deck as **placeholders** (slides 25–26, after the plan and
before the status section) and are not yet worked into the argument. The open
questions are listed in the comment above them in `slides/index.html`.

## Animations (`animations/`)

| name | what it is |
|---|---|
| `turn_sps`, `turn_bench`, `turn_bench_p2`, `turn_chamber` | turntables, 270 frames, 18 s per turn, seamless loop — MP4 + GIF |
| `build_sps` | table → uRWELL references → P2 fans → beam |
| `build_bench` | rack → trigger paddles → M3 reference → chambers → muons |
| `build_ntof` | the n_TOF setup assembling around the target — the build act of `make_ntof.py`, at its fixed camera |
| `turn_ntof` | turntable of the full n_TOF setup |

Build-ups are written **both** as a slow MP4 and as numbered stills
(`build_bench_1_rack.png` …). Drop the stills on successive slides and the
setup assembles itself as you speak — no video embedding needed.

**The deck does exactly that for the cosmic bench** since 2026-08-18: slide 12
was `build_bench.gif` and is now the five stills as a five-frame overlay build,
with HTML `.pin` labels calling out what each frame adds (60 × 60 cm trigger
paddles, 50 × 50 cm M3 references, 40 × 40 cm chambers under test). A GIF cannot
be paged, cannot be held on the frame being talked about, and prints as whatever
frame the PDF exporter happened to catch. Refresh the deck copies with

```bash
cp animations/build_bench_?_*.png slides/assets/img/
```

Individual scenes, with all the switches:

```bash
../.venv/bin/python make_sps.py   --views hero,side,beam [--mx17] [--envelope] \
                                  [--no-spot] [--no-tracks] [--theme dark]
../.venv/bin/python make_bench.py --views hero,side,low --slots p2,mx17 \
                                  [--no-structure] [--no-tracks]
../.venv/bin/python make_anim.py  --only turn_bench --frames 120
```

`--slots lower,upper` takes `mx17`, `p2` or `none` for each test level.

## Measured alignment and real tracks

The bench scene will draw a chamber **where the fit says it is**, and will draw
**real reconstructed muons** instead of sampled ones:

```bash
../.venv/bin/python make_bench.py --views hero \
    --align P2=/media/dylan/data/x17/cosmic_bench/Analysis/<run>/<sub>/<det>/alignment_tpc_veto50/alignment.json \
    --rays  /media/dylan/data/x17/cosmic_bench/<run>/<sub>/m3_tracking_root
```

* **`--align SLOT=alignment.json`** — the file written by
  `cosmic_micro_tpc_analysis.save_alignment`. It maps detector-local strip
  coordinates into the M3 frame; `geometry.load_bench_alignment` pushes the
  chamber's own active-area centre through that transform to get its transverse
  offset. The M3 frame is centred on zero, so the result is directly the
  chamber's (x, y) in the bench frame — typically a few mm to a few cm, e.g.
  `(-4.3, +12.7) mm` for det2 on 6-22.
  * The **in-plane angle** (~89–90°) turns the *strip direction only*, not the
    chamber body. On a square chamber a 90° body rotation carries no
    information but does swing the frame's specular reflection from bright to
    dark, making two identical chambers look like different objects.
  * The fit's own `z_x`/`z_y` (typically 713–715 mm against a configured 702 —
    the known origin offset) is **reported, not applied**, and an alignment
    whose z belongs to the other slot triggers a warning rather than being
    drawn silently.

* **`--reference`** — the shortcut: `geometry.BENCH_REFERENCE` names the one
  June run (`mx17_det2_det3_overnight_6-22-26/long_run`) that carries **both**
  slots' alignment fits *and* its own M3 rays, so the whole figure comes from a
  single dataset — mx17_3 in P1 (fit z = 242 mm), mx17_2 in P2 (fit z = 714 mm).
  The headline `bench_*` figures use it automatically when the data disk is
  mounted, and fall back to nominal positions and sampled muons when it is not.

* **`--rays DIR`** — an `m3_tracking_root*` directory. The ray files carry
  `Z_Up = 1302` and `Z_Down = 24`, exactly the top and bottom M3 plane heights
  in `geometry.py`, so a ray is a straight line through the scene with **no
  transform at all**. Quality cuts default to the recipe in
  `mx_june_cosmic_qa/qa_config.py` (χ² < 1.0 **and** NClus = 4 on both planes),
  and only tracks that cross both trigger paddles are drawn.

### The SPS alignment, and what it says

`sps_beam_test_26/analysis/urw_mapping/mapping_alignment.json` fits a rigid
(dx, dy, θ) per P2 station, taking a uRWELL track into the P2 pad frame. Two
results come out of it, and **neither moves anything in the drawing** — which
is itself the finding:

* the three P2 stations agree to **0.7 mm in x, 1.6 mm in y and 0.38°**, so the
  telescope is transversely aligned far below anything visible at figure scale;
* the fitted uRWELL→pad rotation (−59.68 / −59.77 / −60.07°) plus the fan's own
  pad→lab rotation of −(90 + 30.074)° closes to **180.24 / 180.16 / 179.86°**.
  A multiple of 90° means the uRWELL strips are square to the lab: the −60° is
  the fan's mounting geometry, not a tilted detector.

So the stations are drawn on the nominal axis because that is what the
alignment says. `SPS_STATIONS` still carries per-station `x`, `y` and `yaw`, so
a survey would drop straight in.

### Real SPS beam tracks

`data/urwell_tracks.csv` holds **measured** two-point tracks from the two EIC
uRWELLs, produced by `tools/extract_urwell_tracks.py` **on lxplus** (the merged
hit file is 11 GB and stays there):

```bash
scp tools/extract_urwell_tracks.py \
    ../sps_beam_test_26/analysis/urw_mapping/mapping_urwell.csv lxplus:~/mpgd26_tracks/
ssh lxplus 'cd ~/mpgd26_tracks &&
  source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh &&
  python3 extract_urwell_tracks.py --mapping mapping_urwell.csv --subrun 23'
scp lxplus:~/mpgd26_tracks/urwell_tracks.csv data/
```

Both uRWELLs are on FEU 1 (front x/y = channels 0–255, back x/y = 256–511) and
`mapping_urwell.csv` already carries the resolved wiring and the final
`position_mm`, so nothing has to re-derive the connector order — the part with
four candidate answers and a mirror ambiguity.

**The script will not write tracks unless it reproduces the published
front→back alignment.** On `highstat_eff_1/beam_commissioning_00`:

| axis | slope (published) | offset (published) | core σ (published) |
|---|---|---|---|
| x | 0.99940 (0.99960) | −0.96 mm (−0.96) | 0.67 mm (0.77) |
| y | 1.03933 (1.04177) | −3.98 mm (−4.11) | 0.64 mm (0.72) |

3 759 events out of 1.19 M had exactly one cluster in all four views. As a
further check the extracted spot sits at uRWELL local y = 51.0 mm against the
documented 50.9 mm. The σ come out slightly *better* than published because
the four-view single-cluster requirement is tighter than the published cut.

## What is actually measured

* **SPS station z** (0 / 320 / 630 / 940 / 1155 / 1370 mm) — `run_59`'s
  `det_center_coords`, via `sps_beam_test_26/analysis/run_inventory.json`.
* **The P2 fan** — annulus sector r 150.7 → 635.0 mm, φ 2.30° → 57.85°, apex
  back-solved from the group's Gerber-derived pad map, mounted bisector-vertical
  with the apex 765 mm above the table. All **1280 pads** are placed as their
  true rotated rectangles from `P2_BASKET_mapping.csv`, using the same
  construction as `15_sps_beam_board.py` so the two can't disagree.
  Sectors 3–6 of 10 are the instrumented ones and are drawn brighter.
* **The beam spot on P2 MID** — the stage-22 `n_tag` illumination summed over
  the ten `eff_nominal_1` sub-runs (15.1 M tagged tracks), shaded onto the pads.
  Pushed through the mounting transform it reproduces the documented
  250 mm height, σ_h = 28.6 mm and σ_v = 37.8 mm.
* **The beam** — parallel to < 0.5 mrad, uniform across the hard-edged
  186–311 mm trigger slab (that band is the scintillator aperture, not the
  beam), Gaussian horizontally.
* **Bench plane z** (−110 … 1420 mm) and the P1/P2 levels — `bench_geometry` and
  `detectors` in `mx17_det2_det3_overnight_6-22-26/run_config.json`.
* **MX17 dimensions** — 470 mm PCB, 398.58 mm metallised, 30 mm drift gap,
  512 strips per view at 0.78 mm pitch, 150 µm amplification gap
  (`garfield_sim/mm_config.py`).
* **Both opening-angle curves** in `x17_signature` — sampled from
  `X17PhysicsSpectrum` and `IPCPhysicsSpectrum` in
  `MX17_Simulation/MX17_Simulator.py`, which that module documents as matching
  the Geant4 `X17PrimaryGenerator` event for event. 400 000 events per channel,
  cached under `.cache/`. `make_x17.py --validate` cross-checks the sampled X17
  minimum against an independent analytic solution in `scenes_x17.py` (they
  agree to <0.01°) and reports the IPC shape. **IPC is not a small-angle-only
  background**: its pair mass is drawn from dN/dM ∝ 1/M, giving a median
  opening angle of 30° and ~30 % of the yield above 60°, right under the X17
  peak.
* **Cosmic muons** — sampled from cos²θ and kept only if they cross both
  60 × 60 cm paddles, i.e. the bench trigger. Nothing steeper than ~15° survives
  that, which is why the drawn tracks are near-vertical.

## What is drawn but not measured

Collected in `geometry.ASSUMPTIONS` and repeated in every figure caption:

* **SPS transverse alignment.** Every run config carries `x = y = 0` — nominal,
  not surveyed. The uRWELLs and MX17 are drawn centred on the beam axis.
* **MX17 at z = 1155 mm** is flagged a placeholder in the run config itself.
* **The P2 mounting** (bisector vertical, apex up, centred, 130 mm from the
  lowest active point to the table) is supplied by the P2 group and corroborated
  by the trigger aperture coming out horizontal — but not surveyed.
* **The bench scintillators.** The paddles sit outside the DAQ geometry, so no
  run config records them; they are drawn just beyond the stack at z = −110 /
  +1420 mm, matching the engineer-package side-view schematic.
* **The bench rack and the SPS table** are drawn for context.
* **Drawn thicknesses** are chosen for legibility. The 30 mm MX17 drift gap is
  the only real one.
* **The relative rate of the two channels** in `x17_signature`. The *shapes*
  are both sampled from `MX17_Simulation` (see below), but each is normalised
  to unit peak — their ratio is the measurement, so the figure must not appear
  to assert it.
* **…and in the story panel 5 it *is* drawn**, deliberately: since 2026-08-17
  that panel stacks the X17 yield on the IPC background at
  `scenes_x17.SIG_FRAC` (4 % of the IPC yield over the plotted window) so the
  slide shows what a measurement looks like — a bump on a background. The
  fraction is illustrative, is printed on the panel in words, and is not a
  prediction. Change it in one place if you want a different-looking bump.
* **The kinematic minimum it marks, 109°**, follows from m = 16.8 MeV carrying
  the full 20.58 MeV transition, recoil neglected. It is *not* the ~120° quoted
  from the ATOMKI ⁷Li measurements, which sit at a different transition energy.
* **The pad etch gap** — pads are shrunk 16 % about their own centres so the
  1280 of them don't read as one solid copper sheet. Pad centres, angles and
  count are the measured ones.

## How it is built

| file | role |
|---|---|
| `geometry.py` | every number and its provenance; `ASSUMPTIONS` |
| `meshes.py` | geometry → PyVista meshes (slabs, frames, fan prisms, pad quads, tubes, cast shadows) |
| `style.py` | palette, materials, procedural studio cubemap, light rig, render harness — for the 3-D renders |
| `plotstyle.py` | the same house look for **matplotlib data plots**: ink/muted/line matched to the slide CSS, recessive frame, and the four-detector categorical palette (Okabe-Ito subset, validated with the dataviz `validate_palette.js`; the CVD floor-band warning is discharged by per-detector marker shapes) |
| `scenes_sps.py` | the H4 telescope |
| `scenes_bench.py` | the cosmic bench |
| `scenes_chamber.py` | one MX17 chamber, exploded |
| `scenes_x17.py` | the X17 physics case; matplotlib, not PyVista — the decay kinematics live here too |
| `scenes_target.py` | **the n_TOF Target #3 spallation target, in detail** — two views for the target backup slides: `build_layers` (the whole assembly cut open along the beam, cradle to vacuum window) and `build_cooling` (one anti-creep plate exploded off its slice, with the channels, the wedge obstructions and the nitrogen path). ⚠️ Every shared dimension is **imported from `scenes_ear2`**, never re-typed — the facility figure draws the same object. Records three rendering traps in its own docstrings: the beam-axis cutaway is wrong for a plate, lead/aluminium/groove are three greys within 0.2 in value, and the plates in the layers view need PBR to separate on a single cut plane. The 10° beam yaw is only visible in `build_layers` |
| `make_target.py` | the two target detail figures, and the only writer of `slides/assets/img/target3_{layers,cooling}.png` |
| `scenes_ear2.py` | the n_TOF EAR2 vertical beam line; every sourced height and aperture is cited in its docstring, `ACTS` / `y_of()` are the broken vertical scale, `STAGE_PARTS` is the five-frame build, `H_PIPE_END` is where the lower pipe really stops and `H_UP0` where the separate upper one starts, `_section_angles()` places the two drawn chambers relative to the camera, the acts end just above the station (no ceiling, no dump), and everything read off the photograph rather than a drawing is in `ASSUMPTIONS` |
| `make_ear2.py` | the EAR2 build-up in both label layouts, and the only writer of `slides/assets/img/ear2_{onfig,beamline}_{1_target,2_neutrons,3_collimation,4_dump,5_station}.png` |
| `annotate.py` | 3-D anchors → pixels, then the type layout |
| `make_sps.py`, `make_bench.py`, `make_chamber.py`, `make_x17.py` | per-scene drivers with camera presets |
| `make_couplings.py` | the two X17-theory teaching figures for the backup slides — the "What ε is" vertex/chain explainer (drawn with the `scenes_x17` primitives) and the three-lane coupling-windows chart (`plotstyle`); the only writer of `slides/assets/img/x17_{epsilon,couplings}.png`. Every number is the verified set from the theory backup slides, and the docstring says to change slide and script together |
| `make_figures.py` | the deliverable still set |
| `make_anim.py` | turntables and build-up sequences |
| `make_report.py` | `report.html` |
| `make_flash_slides.py` | the six deck-only flash figures behind slides 28&ndash;30 and one backup &mdash; kept out of `make_status_plots.py` on purpose, because that module's `render()` is called with no name list by the flash-charge report and would then re-open a 154 MB waveform archive to build figures the report does not use |
| `make_status_plots.py` | the seven `status_*.png` data figures for the Status section — reads only committed reductions plus a small local mirror; `render()` is importable so `ntof_july_analysis/flash_charge/make_report.py` builds the same figures from the same code |

`make_sps.build()` and `make_bench.build()` take a `show=` subset of their
`PARTS` tuple; that is the one hook the build-up sequences need, and it means a
stage of the animation is the *same scene* as the final still, not a
re-creation of it.

Rendering is PyVista/VTK. Three choices are worth knowing about, because each
was made after the obvious alternative failed:

* **PBR for metals only.** VTK drives PBR almost entirely from the environment
  cubemap, which is what makes aluminium and copper read as metal — but it also
  swamps the directional lights, so a scene rendered *entirely* in PBR comes out
  flat and unshaded. Everything non-metallic uses classic Phong instead, where
  the light rig does the modelling.
* **Analytic cast shadows.** VTK's shadow-map pass is unusable at a two-metre
  scene scale — even at 4096 px the map frustum shows up as straight-edged
  brightness steps across the table. Since every object here has a known
  outline, `meshes.ground_shadow` projects that outline along the key direction
  onto the ground plane, with a few nested copies for a penumbra. Exact,
  artefact-free, and under full control.
* **Type after the fact.** VTK's 3-D text is hard to place and reads badly at
  print size, so the render is composed onto a titled canvas in matplotlib with
  a label gutter, and 3-D anchors are projected through the same camera.


## Which way things face

Both scenes had this wrong at first and it is worth stating explicitly.

* **SPS.** The beam runs along **+Z** and meets `EIC_uRWELL_front` (z = 0)
  first. Every readout plane faces **upstream**: a particle enters through the
  drift window, crosses the gas, and the pads sit on the face the gas is on —
  so the pads look back into the beam and the PCB substrate is downstream. All
  three SPS cameras therefore sit at negative z, looking downstream.
* **Bench.** Muons travel **downwards**. Both scintillator PMTs point along
  **−y**, and the rack has its two uprights on **+y** with the rails
  cantilevered out over the open −y side — which is the side every hero camera
  looks from, and the side detectors slide in from.
* **Tracks carry arrow heads on the exit end** in both scenes, so the direction
  of travel does not depend on the caption or the camera. `cosmic_tracks`,
  `real_tracks` and `beam_tracks` all return `(entry, exit)` in travel order so
  the head always belongs on the second point.

## Known conventions that are not pinned by data

The transverse **handedness** of the P2 pad frame is a view-side convention,
not something the data fixes (`P2_MIRROR`, and the same caveat in the P2
group's own handoff). It sets the sign of the beam's 6 mm lateral offset and
the sense of the uRWELL local axes in the drawing. Track *angles*, the spot
size and the front→back offset are unaffected by it.

## Slides

**The deck of record is `slides/mpgd26_talk.pptx` from 2026-08-26** — it is
edited by hand in PowerPoint, and `slides/index.html` is frozen. The HTML is
kept as the provenance record (per-slide comments naming the script, the
source and the reasoning behind each slide) and as what `slides/to_pptx.py`
can re-export from — but a re-export overwrites the .pptx and every hand edit
in it, so it is a start-over, not a sync. Figures are unaffected: the
`make_*.py` scripts still write `slides/assets/img/`, and a changed figure is
swapped into the .pptx by hand.

`slides/` builds the actual MPGD2026 talk from a subset of the figures above
plus curated material from elsewhere (ATOMKI evidence, the n_TOF/EAR2
facility, cosmic-bench characterization results). Start with
`slides/NOTES.md` — open items by slide, and how to regenerate every image
in `slides/assets/img/`, which is **not tracked in git** (see that file for
why). `slides/STATUS_PLAN.md` is the map for the Status section: what each
slide is for, where every number comes from, which slides are load-bearing if
the talk has to be cut, and the three things still open (the physics framing,
the charge measurement's one systematic, and the PRELIMINARY status of the
target-imaging reconstruction).

`slides/mpgd26_talk_draft.pdf` is regenerated by `slides/make_pdf.sh` and is
**deliberately not committed** — it is ~23 MB of embedded raster and rebuilds
in one command, which is the same policy the repo applies to `*.png`.
