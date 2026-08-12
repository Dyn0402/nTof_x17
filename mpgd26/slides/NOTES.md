# mpgd26/slides — status and open items

MPGD2026 talk sketch, built as a single self-contained `index.html` (no build
step, no external network dependency — open it directly in a browser).

**View it:** open `index.html` in any browser, or `index.html#9` to jump to
slide 9. Arrow keys / Space / click the left-right edges to navigate.

**PDF backup:** `./make_pdf.sh` regenerates `mpgd26_talk_draft.pdf`. Don't use
the browser's own File → Print → Save as PDF — Chrome's headless print
pipeline blanks whichever slide is first whenever the output has more than one
page (confirmed positional, not content-specific, by swapping slide order).
`make_pdf.sh` sidesteps it by printing each slide to its own single-page PDF
and merging with `pdfunite`.

## Design reboot — 2026-08-12

**The deck was restyled wholesale for a scientific-conference aesthetic** (Dylan:
the tech look was wrong for the room, and HTML is risky to hand a conference —
so the **PDF from `make_pdf.sh` is now the deliverable**, with the HTML as the
authoring/presenting format). Slide *content* and *structure* are untouched: the
whole look lives in the `<style>` block, and the only markup edits were the
Backup divider (now `class="slide divider"`), a `dense` class on four
overstuffed backup slides (see below), and two column-ratio tweaks on the target
slides.

**The theme is "Modern", chosen by Dylan the same day from four rendered
options** (journal didone serif + mx17 purple / Times "preprint" + navy /
modern all-sans / slate banded-header TDR). The comparison snapshots are in
`theme_preview/` (`compare.html` + 20 screenshots) — a record of the decision,
regenerable from git history, delete at will. The three variant decks themselves
were deleted after the choice was merged into `index.html`. What the system is
now:

- **All-sans, Metropolis-inspired**: Noto Sans Display titles (bold, dark rule
  under), Noto Sans body — **local system fonts**, so the file stays
  offline-self-contained. White paper, dark teal-black ink (`#23373b`), **one
  orange accent** (`#eb811b`), hairline rules, booktabs-style `spec-table`s, a
  running footer (`D. Neff · MPGD 2026, Prague · …` + slide number) on every
  slide but the title. The copper caution accent survives; the figures' mx17
  purple now reads as a *figure* colour rather than a chrome colour, which
  keeps the two layers distinguishable.
- **Slide numbers are a CSS counter**, and `make_pdf.sh` injects a per-slide
  `counter-reset` so they survive its one-page-at-a-time printing. If you add or
  reorder slides, numbers update themselves in both paths.
- **Photographs get a hairline frame automatically** (`img[src$=".jpg"]`);
  renders and plots sit borderless on the paper. Don't re-add panel boxes — a
  height-limited figure in a bordered box reads as dead space.
- ⚠️ **`class="slide dense"`** shrinks `spec-table`/`callout`/`bullets`/caption
  on the four reference slides that carry more than a page of content (the EAR2
  documentation slide, both Target #3 slides, and "Why that is a measurement…").
  **The two target tables overflowed the page in the old design too** — checked
  against git HEAD before fixing, so this was a pre-existing fit bug, not a
  regression. If a new slide overflows, `dense` is the intended lever.
- The old body font stack resolved to Liberation Sans, which is ~7 % narrower
  than Noto Sans — that is why the densest slides needed refitting, and why
  eyeballing "it fit before" is not a check after any font change. The full-deck
  QA loop is: `./make_pdf.sh`, render pages with `pdftoppm`, look.

`mpgd26_talk_draft.pdf` regenerated on the new design, 56 pages.

Current: **56 slides** (title, outline, motivation incl. the **5-frame EAR2
beam-line build-up, 2026-08-11**, detectors, **n_TOF setup — the 9-frame 3-D
build-up, 2026-08-10**, **Status — 11 slides, drafted 2026-08-09**, summary,
backup incl. the two imon teaching slides, 3 status backups, the old
measured-drawing setup slide and the **EAR2 render's documentation slide**).

**The Status section is deliberately over-built.** It is written as a menu to
cut from, not as a running order — see [`STATUS_PLAN.md`](STATUS_PLAN.md) for
what each slide is for. Its own "load-bearing if time is tight" set was D1, D4,
D6, D8, D9; that has since been superseded — **D9 was removed by decision**
(no yield claim, see below) and [`RUNNING_ORDER.md`](RUNNING_ORDER.md) is now the
authority on what to show in the 15-minute slot.

## Where this stands — 2026-08-10

**The slot is 15 min + 5 min**, Prague, 3 September 2026, speaker Dylan Neff.
The deck is far over length by design; [`RUNNING_ORDER.md`](RUNNING_ORDER.md)
holds the proposed cut. Read that first.

**Done this pass:** title byline · ⁴He-channel bridge on the evidence slide ·
D9 removed and the section reframed as a status report · efficiency numbers and
figures refreshed (three chambers moved 14–21 points — the old values were wrong,
not merely stale) · the spatial-resolution caveat withdrawn as wrong-premised and
replaced with a deconvolved number · the ideal e⁺e⁻ kinematics figure built from
Geant4 truth · the zoomed board-peel figure · the strip pitch corrected package-wide
(0.7785 → 0.78 mm) · the EAR2 beam line rendered in house, EAR1 dropped, and the
beam-dump identification verified against the literature.

**Also done, second pass, all from Dylan's review:** the EAR2 render split into a
build-up with his geometry changes (pillars gone, capsule floating) · the
board-peel strip layers redrawn as **strips with the vias suppressed** and the
pillars off the resist band · and **the imon systematic MEASURED AND CLOSED**,
with a two-slide teaching backup.

**Also done, third pass, 2026-08-11, from Dylan's review of the EAR2 render:**
**the beam pipe now really ends ~1 m above the EAR2 floor**, in a flanged
circular end, with the neutrons crossing the hall in air. That was a **factual
correction, not a styling change**: the previous render ran the pipe to the dump
and then cut it away as a drawing device, and the pipe does not run there. His
question *"where is the first collimator?"* is answered on frame 2 and in the
facility section below — **7.4–8.4 m above the target, ~10 m below the EAR2
floor**, 1 m of iron with a 200 mm bore, invisible in the render because it falls
inside the break.

**Fourth pass, 2026-08-11, same review round:** **the ceiling and the beam dump
came out of the drawing entirely** and the upper break went with them, on Dylan's
"cut off the full ceiling beam dump to make more space on top and put more
emphasis on the measuring station". `DRAWN_H` 8.67 m → 5.71 m, so everything left
is drawn **~1.5× larger**; the beam now leaves the top of the frame under a label
carrying the dump's height, and what is above the frame is stated in
`ASSUMPTIONS` and in the slide caption. The build went **back to three frames** —
the dump had its own frame for a few hours and then had nothing left to reveal.
**And the CERN photograph of the hall is back**, cropped to its right-hand panel
and set beside the render (`ear2_hall_photo.jpg`). ⚠️ It shows the vertical line
**closed from the floor to the ceiling** with detectors on a chamber in the beam
— a different configuration from the one the render draws. The backup slide's
caption says exactly that and claims nothing about which section is removable.
See "Left:" for the open question. (The photograph has since been promoted from
illustration to **source**: the shape of the line inside the hall is scaled off
it — see the fifth pass below.)

**The imon result is the big one.** The CAEN readback is a ~1 s averager, so
`mean − median` really is the time-average current: **the deck now says "142 nC"
with no inequality and no correction factor**, and four independent estimators of
the charge agree to ±3 %. Every lower-bound hedge is gone from the slides. Detail:
[`HANDOFF_imon.md`](HANDOFF_imon.md) and §8 of
`ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md`.

**Fifth pass, 2026-08-11, same review round — the figure got more real and the
slide got taller.** Six changes:

1. **The facility slides carry no caption.** Both pictures on them are portrait,
   so height is the scarce resource, and the six-line caption was costing them a
   fifth of it. It moved, in full, onto **one new backup slide** under the same
   figure ("The EAR2 render, and what in it is drawn rather than measured"). The
   main slides keep a one-line `.fig-label` credit under each picture, which is
   the minimum attribution and is not a caption. **The photograph is now on the
   left**, the render in the middle, the text on the right.
2. **The column widths are proportional to the two images' aspect ratios**
   (0.62 / 1.00 / 1.04 for photo / render / text). That is not cosmetic: get it
   wrong and both images are *width*-limited, stop growing, and leave a band of
   dead space under them — which is exactly what the first caption-free attempt
   looked like. Both pictures now run 505 px of a 720 px frame.
3. **The slides use the on-figure-labelled render** (`ear2_onfig_<n>_<tag>.png`,
   labels on the drawing's own background, left *and* right) instead of the
   gutter variant. Dylan asked for that variant to compare; with no caption to
   pay for, a gutter is dead width, so it won. Both variants are built by every
   `make_ear2.py` run and swapping is three `img src` edits. The **backup slide
   keeps the gutter variant** — under a caption the figure is short, and
   on-figure labels are the first thing to stop being readable at that size.
4. **The line inside the hall is modelled from the photograph**: the white
   segmented polyethylene shielding disc and collar on the floor, the lead-disk
   vacuum chamber at the shaft bore, a **reducer** down to a narrow tube, and the
   flanged exit window. ⚠️ Radii are scaled off the photo against its ~1.2 m
   white disc and the step heights are chosen to fit — a reading of a photograph,
   not of a drawing, and `ASSUMPTIONS` says so in those words.
5. **There is a pipe above the experimental space again — a different pipe.**
   Dylan: *"the top pipe toward the beam dump included and cut off at some
   point."* So `add_uppipe` draws the wide section from `H_UP0 = 20.45` m up
   through the top of the frame, cut off by the camera margin rather than capped.
   The line now reads the way the photograph reads: narrow tube → flanged end →
   **open space with the experiment in it** → wide tube away. Drawn at 160 mm
   bore rather than the photo's own proportions, because at full size it
   *outweighed the measuring station*, which is the opposite of what removing the
   beam dump was for. `DRAWN_H` 5.71 → 5.91 m.
6. **"What is this pink diagonal thing in the collimator?"** — it was the
   *break's own section marker*, `COL['section']` on a 24°-tilted disc at the
   collimator's 680 mm radius, i.e. a third of it hanging outside the block. It
   was never hardware. Fixed by drawing that one face **flat and inset**, at the
   block's own bottom rather than 120 mm up inside it, and desaturating
   `COL['section']`. The tilt survives on the pipe's face below, where the disc
   stays inside its own silhouette.

**Sixth pass, 2026-08-11 — the build got finer and the station got rebuilt.** All
from Dylan's review of the fifth:

1. **Five frames, not three.** Target → **the neutrons leave at 90°** → the whole
   middle of the line (second collimator, floor and its shielding, lead disks,
   the pipe *ending*) → **back into a pipe, on up to the dump** → the measuring
   station. That needed `pipe` split into `pipe_lo`/`pipe_hi` and `neutrons` into
   `neutrons_lo`/`neutrons_hi`, so each frame can show the shaft without the hall
   and the raw flux without the collimated pencil. Deck 49 → **51 slides**,
   main flow 30 → **32**.
2. **The chambers are chambers now.** Dylan: *"I'm also not sure what the green is
   supposed to be there"* — fair, a 400 mm square of PCB colour seen obliquely is
   a green box. Each is built with `meshes.rect_chamber`: aluminium frame,
   readout board **with strips on it**, 30 mm of lit drift gas facing the sample,
   entrance window as a tint. ⚠️ **24 strips are drawn, not 512** — the real
   0.78 mm pitch is 0.26 px at this scale, i.e. a flat grey wash; declared in
   `ASSUMPTIONS`.
3. **Frame 5 exists in two versions, because Dylan asked to compare**: all four
   chambers (`ear2_onfig_5_station_4arm.png` — two solid, two the camera looks
   through drawn as bare frames) or two in section
   (`ear2_onfig_5_station.png`). `STATION_ARMS` picks; `make_ear2.py` writes both
   every run. **The section pair is the default and is on the slide** — see the
   seventh pass. ⚠️ In the four-arm version the near two must be drawn **uncut**:
   the cutaway plane passes through the beam axis and would *delete* them, not
   fade them, which is why the first four-arm attempt came out pixel-identical to
   the two-arm one. `frame_ring` is a hollow band, so drawing them solid frames
   the sample instead of hiding it.
4. **"Measuring station" became two labels**, `sample` and `detectors` — the thing
   the beam hits and the things that watch it are different objects and the slide
   should not make the audience work that out.
5. **The sample is centred between the two pipes.** `H_UP0` 20.45 → **20.74 m**,
   which puts 19.95 m exactly halfway between the end of the lower pipe (19.16 m)
   and the start of the upper one, and the upper pipe now shows **only its
   entrance flange and one ring** before the frame cuts it. `DRAWN_H` 6.11 m.
6. **The beam is faded across the sample** (`add_neutrons_hi(fade_sample=True)`,
   frame 5 only). At full strength it is a grey line down the middle of a
   translucent capsule and reads as the capsule being *behind* the beam. ⚠️ This
   is the **one exception** to "every frame is a strict subset of one picture" —
   documented in `scenes_ear2.py`'s docstring and in the slide comment.
7. **`Lead target` moved to the right column** (Dylan): its anchor is on the beam
   axis and `protons` comes in from the left, so with both on the left their
   leader lines crossed right under the target.

**Seventh pass, 2026-08-11 — legibility, all of it from Dylan looking at the
renders.** No slide was added or removed; every change is inside the drawing.

1. **The neutrons leaving the target are darker.** `COL['neutron']`
   `#8b96a3` → **`#55657a`**, and the five arrows are drawn at 12 mm instead of
   9 mm. Dylan could not find them: the old grey is the same *value* as the
   amber-tinted pipe interior it has to be seen against. ⚠️ The obvious fix —
   making them white — is the wrong one, and was rejected on purpose: the same
   neutrons cross the open hall higher up on a near-white background, where white
   would lose them completely. A darker slate is the only single colour that
   works on both of this figure's backgrounds.
2. **One opacity for the beam envelope, above the collimator and below it**
   (`ENV_ALPHA = 0.16`). It was 0.17 below and **0.34** above, on the reasoning
   that a 22 mm pencil needs more tint than a 317 mm cone. What that produced was
   two different-*coloured* beams in one figure — the same amber at twice the
   opacity, on a thin tube seen through both its walls, comes out olive-brown,
   which is what Dylan asked about. The pencil is carried by its tracks instead,
   which is what they are drawn oversized for.
3. **The break is drawn flat.** `BREAK_TILT` 24° → **0**. The tilt was meant to
   make a break read as a drawing cut rather than as a pipe that stops; from a
   camera 6° above the horizon a 350 mm disc tipped 24° projects to a thin dark
   **diagonal streak**, and Dylan asked what the line across the top of the beam
   pipe was. Flat, it caps the tube in the section accent. Same fix the
   collimator's own break face got in the sixth pass, same reason.
4. **The PE-shielding leader crosses to the left of the pipe.** Its label is in
   the left column and its anchor was at +x, so the leader ran across the beam
   pipe to reach a part that is present on *both* sides of it.
5. ⚠️ **Two chambers in section, not the four-arm pinwheel — and this is a drawn
   arrangement, not the station.** `STATION_ARMS = 2` is now the default, and the
   pair's azimuths come from `_section_angles()`, which is defined **relative to
   the camera**: just past edge-on, on the far side, so the cutaway slices them
   and every surviving piece is behind the sample. `SECTION_TILT = 22°` sets how
   much board face shows. **The price is that the two drawn azimuths are 136°
   apart, not the real 90°** — declared in `ASSUMPTIONS`, and
   `ear2_onfig_5_station_4arm.png` keeps the true pinwheel on record. Why it was
   worth it: drawn as the pinwheel, the two solid arms come out 45° to the screen
   and close on the drawn capsule until they touch it, so the sample is a sliver
   between two green rectangles.
6. **The station's slide labels are generic**, `sample` and `detectors`, not
   `³He sample` / `Micromegas trackers`. This is slide 9 and the experiment is
   introduced on slide 16: the specific words made the facility slide look like
   the setup slide. The standalone figure's `LABELS` still carry the full
   description, including the four-vs-two caveat.
7. ⚠️ **Nothing at the station is drawn 1:1 any more.** `CAPSULE_SCALE` 7 →
   **5.5** and a new **`PLATE_DRAW = 1.35`** on the chambers (Dylan: *"make the
   capsule slightly smaller and the detectors a bit larger — I don't care much
   about how the detectors actually look here or their accuracy, this is mostly
   just a generic example diagram"*). `PLATE_DRAW` scales board, gap and frame
   **together**, so each is still a chamber and only its size is wrong; the
   `STANDOFF` (`PLATE_R = 330 mm`) is still real. The chambers used to be the one
   true-size thing at the station, so the claim *"the four chamber plates are
   true size (400 mm)"* had to come **out of the backup slide's caption** — it is
   now the opposite statement. If you ever restore true scale, that caption is
   the thing to put back.

**Two new backup slides: the target, in full** (2026-08-11, Dylan — *"add a
backup slide with full details on how this target works (cooling, layers, etc).
Would be good for me in the future. Feel free to design a more detailed
visualization"*). Slides **46 and 47**, and they are the **source of record for
the target in this deck**:

1. *"The n_TOF spallation target, layer by layer — Target #3"* — the assembly cut
   open along the beam, plus a spec table (core, lead grade, plate gaps, vessel,
   both windows, the moderators, the beam).
2. *"How 2.7 kW leaves a block of lead that creeps at 135 °C"* — one anti-creep
   plate exploded off its slice, plus the nitrogen circuit end to end, the duty
   figures, the wedge trick, the temperatures and the creep numbers.

New files: **`scenes_target.py`** and **`make_target.py`** (`../.venv/bin/python
make_target.py`, ~40 s, two views). ⚠️ **Every shared dimension is imported from
`scenes_ear2`, never re-typed** — the facility render on slides 5–9 draws the same
object, and a divergence between the two would be worse than either being wrong.
That is also why `MOD_XY` was added to `scenes_ear2` while building these: at the
detail scale it was obvious the moderator can was drawn *overhanging* the vessel
it is bolted to, and there was nowhere for the gas outlets to be.

⚠️ **Three rendering traps recorded in `scenes_target.py`, all found the hard way:**

- **The beam-axis cutaway is wrong for a plate.** It slices a 9.85 mm part
  diagonally into a wedge — right for a tube, wrong for a face. The cooling view
  is an *exploded* plate instead, with no clip plane at all.
- **Lead, aluminium and a milled groove are three greys within 0.2 in value.** At
  this light rig they all saturate to the same pale grey. The groove interiors are
  pushed to a much darker slate, and the plate brighter, than the shared palette.
- **The plates in the layers view use PBR, uniquely in these two figures.** The
  cutaway leaves all six slices and five plates on *one flat plane*, where only
  shade separates them, and matte they merge. The metallic highlight is what makes
  9.85 mm of aluminium visible between 50 mm of lead — at the cost of some
  apparent thickness, which is why the plate's real thickness is shown in the
  other view.

And one thing that **cannot** be drawn on the cooling view and should not be
added: the **10° beam-to-target angle** is a yaw about the *vertical* axis, so it
foreshortens to nothing on a view of the plate face. It is in the layers figure,
where the horizontal proton arrow and the yawed stack are both in frame.

**Deck is now 56 printed slides** — 43 of main flow, a `Backup` divider, then 12
backup slides. Fine: it is a menu, and `RUNNING_ORDER.md` holds the 16-slide cut.
`mpgd26_talk_draft.pdf` is current at 56 pages.

⚠️ **Slide numbers in this file are a moving target.** The deck went 51 → 53 → 56
in two days and the setup section was renumbered when it landed, so any "slide N"
written here may be a day stale — the two target slides quoted as **20 and 21**
when they were written are now **46 and 47**. Identify a slide by its **title** and
re-derive the number; the check is one command:

```
cd mpgd26/slides && grep -n 'class="title' index.html | nl
```

**Left:**

- **Review the nine-frame setup build-up** (slides 16–24) — the largest unreviewed
  block in the deck. See "Still to do" below.
- **Slide 9's two superseded figures** — regenerating from `sat_det3` on the
  waveform-first basis gives a *better* resolution number.
- **The bench-slide flag** — needs Alexandra's talk order.
- ~~**Re-run `./make_pdf.sh`.**~~ Done 2026-08-11 — **56 pages**: the plan slide
  (25) plus the two photograph placeholders (26–27, PDF 30–31). Re-run it after
  the nine-frame review below, since that will change slides 16–25.
- **Work the two photographs in, or drop them** (slides 26–27) — they are parked
  after the plan as placeholders, with the open questions in the comment above
  them in `index.html`. The pairing that pays is the top-down photo *beside* the
  plan drawing rather than after it; both want a crop, and the "gamma-flash
  recovery" silkscreen bullet needs an inset or it is a claim about something
  the audience cannot see.
- **Confirm the beam-pipe configuration** — narrower than it was, but still open.
  The photograph beside the render shows the vertical line **closed all the way
  from the floor to the ceiling**, with detectors on a chamber in the beam. The
  render has a lower pipe ending ~1 m above the floor (Dylan's own description of
  the hall), open experimental space with the capsule in it, and a separate wide
  section from 20.45 m carrying the beam on to the dump. That upper section came
  from Dylan on 2026-08-11, so the two pictures now differ in **one** thing rather
  than two: whether the gap in the middle exists in the photograph's
  configuration. The presumption is that the section over the experimental space
  is removable and the photograph is an in-vacuum experiment, but **nobody has
  confirmed it**, so the backup caption states what each picture shows and stops
  there. Three numbers would close it: the height at which the lower pipe's exit
  flange sits, the height at which the upper section starts, and whether the
  middle section is a fixed part of the line.
- **Two things in the render are scaled off a photograph and could be replaced by
  a drawing if one exists** — the in-hall diameters and step heights (the lead-disk
  chamber, the reducer, the narrow tube, the upper pipe), and the white shielding
  on the floor, whose material is assumed to be polyethylene because it looks like
  it. Both are declared in `scenes_ear2.ASSUMPTIONS` and in the backup caption. If
  an EAR2 layout drawing turns up, these are the numbers to correct.
- **One question for Dylan:** which CAEN board is card 5 of the crate at
  128.141.177.244? It is the only thing left for the readback's absolute nA
  accuracy — integral conservation is closed regardless and nothing in the talk
  depends on it. The model is in neither repo, and the live crate was deliberately
  not probed during production running.

## Images are intentionally not tracked in git

The repo's top-level `.gitignore` blanket-excludes `*.png`/`*.gif` (see the
comment there — 250 MB was purged from history on 2026-08-04, policy is
"regenerate from the analysis scripts, don't commit the bulk output"). That
rule silently catches every image in `assets/img/` too, including several
that have **no one-command regeneration path** — unlike `mpgd26/figures/`,
this folder mixes mpgd26-native renders with curated external material. If
you clone this repo fresh, `assets/img/` will be empty. Regenerate it with
the steps below, or ask about carving out a `.gitignore` exception for
`mpgd26/slides/assets/` if that's preferable to re-deriving everything.

| File | Source / how to regenerate |
|---|---|
| `x17_signature.png`, `x17_story_capsule.png` | `mpgd26/make_x17.py` (see `mpgd26/README.md`) |
| `chamber_exploded.png` | `mpgd26/make_chamber.py` |
| `microtpc.png` | `mpgd26/make_microtpc.py` |
| `build_bench.gif` | `mpgd26/make_anim.py --only build_bench` |
| `setup3d_1_capsule.png` … `setup3d_9_full.png` (slides 16–24) | **`mpgd26/make_ntof.py`, then `cp ../figures/ntof_build_<n>_<tag>_light.png assets/img/setup3d_<n>_<tag>.png`.** One command for all nine. Geometry is imported at run time from `~/CLionProjects/MX17_Full_Geant/scripts/plot_geometry.py` (itself written against `SimConfig.hh`), so the frames track the simulation; the e⁺e⁻ pair is a real Geant4 event picked by `mpgd26/tools/extract_ntof_event.py` into `mpgd26/data/ntof_event.json` (**which is tracked** — regenerating the frames does not need the sim). **The neutron is drawn, not transported** — a straight line up the beam axis to the pair vertex; the real history is selected and stored but deliberately not drawn (it belonged to a different event), and the beam envelope is measured from the neutron run's sampled primaries. Written with an alpha channel so they sit on the slide gradient. |
| `ntof_plan.png` (slide 25) | **`mpgd26/make_ntof_plan.py --bare`, then `cp ../figures/ntof_plan_bare_light.png assets/img/ntof_plan.png`.** A matplotlib **drawing**, not a PyVista render: orthographic plan down the beam, 1:1 in both axes. Geometry and event come from `scenes_ntof`, i.e. the same `plot_geometry.py` and the same `data/ntof_event.json` as the nine 3-D frames, and every number on it — including the dimension-chain labels — is computed from that geometry rather than typed in. The same command without `--bare` writes the titled version `figures/ntof_plan_light.{png,pdf}` that `report.html` cites. ⚠️ The beam axis is projected away, so the drawn opening angle (122°) is not the space angle (110°); the figure says both. |
| `photo_station_topdown.jpg`, `photo_arm_outside.jpg` (slides 26–27) | **`mpgd26/make_photos.py`** — slide-sized copies of the two photographs taken in EAR2 on 2026-08-10. The full-resolution originals are in `mpgd26/photos/` under the camera's own filenames; **do not go back to `~/Downloads` for them**, that is where they came from and it is not a durable location. No crop is applied (both are portrait phone frames); when the crop is decided, put the box in `PHOTOS` rather than cropping the original. ⚠️ **Placeholders** — parked after the plan, not yet worked into the argument; the open questions are in the comment above the slides in `index.html`. |
| `setup_1_mm.png`, `setup_2_sipm.png`, `setup_3_plastic.png`, `setup_4_full.png`, `setup_topdown.png` | copied from `~/CLionProjects/MX17_Full_Geant/scripts/mx17_buildup_clean_*.png` (`plot_buildup.py --style clean`) and `mx17_mm_layout_topdown.png` (`plot_mm_layout.py`). **Demoted to backup 2026-08-10** when the 3-D build-up took over the main flow — kept because they are the versions that carry the survey distances, i.e. what to show if someone asks "how far apart, exactly?" |
| `mx17_board_peel.png` | **`cd ~/CLionProjects/MX17_Geant && ~/PycharmProjects/nTof_x17/.venv/bin/python scripts/model/plot_mx17_model.py --only peel`** (~4 min), then copy from `design/figures/`. Refreshed 2026-08-10 to pick up two depth-key label fixes: the L5/L6 strip directions were **swapped** (the Y-measuring strips run along *x*, gerber-verified) and the resist thickness still read "100 µm" after `AsBuiltSpec` dropped to 10 µm — now read from `mx17_model.PASTE` so it cannot go stale again |
| `mx17_board_peel_zoom.png` | **`cd ~/CLionProjects/MX17_Geant && ~/PycharmProjects/nTof_x17/.venv/bin/python scripts/model/plot_mx17_model.py --only peel_zoom`**, then `cp design/figures/mx17_board_peel_zoom.png ~/PycharmProjects/nTof_x17/mpgd26/slides/assets/img/`. Same `fig_peel()` as the full-board figure via a new `zoom=True` mode (added 2026-08-10) — 25 × 25 mm of board (matching the figure's own title), split into four 6.25 mm bands so each peeled band shows 8 pad columns, with burned-in 5 × 0.78 mm readout-pitch and 5 × 0.80 mm resist-pitch calipers plus a 2 mm scale bar. **Revised 2026-08-10 (second pass, Dylan's review): the bulk pillars are gone from the resist band, and the L5/L6 bands now draw continuous strips on the gerber's own 0.78 mm grid with the vias suppressed, because the literal dot-and-stub artwork made the strip direction unreadable.** That makes those two bands **schematic, not copper**, so the title no longer says "real gerber copper" — it now names per band what is artwork (the pads) and what is schematic (resist ①, strips ③④, vias suppressed), and the band captions read "along x / along y — schematic". **How the dot-and-stub interconnect actually completes is still unresolved** (`HANDOFF_board_peel.md` §1) — the schematic asserts direction and pitch, not the connection. `--only peel_zoom` is deliberately **not** in the default all-figures run, and the full-board `peel` output is unchanged (it keeps the pillars and the literal artwork; both changes are gated on `zoom`). The pitch is 0.78 mm exactly, not 0.7785 mm; **the slide-7 caption was corrected to match on 2026-08-10** and the error was traced out of the whole package (see the pitch section below) — see [`HANDOFF_board_peel.md`](HANDOFF_board_peel.md) |
| `charge_sharing_schematic.png`, `unsharing_depth_bias.png`, `event_display_3d.png`, `angular_resolution.png`, `spatial_residuals.png`, `time_resolution.png` | `pdftocairo -singlefile -png -r 300` from the matching PDF in `mx_june_cosmic_qa/engineer_package/figures/` (e.g. `10-det3A-spatial-residuals.pdf` → `spatial_residuals.png`) — those source PDFs **are** tracked in git. **All still 2026-07-14 hit-chain vintage**, i.e. the same staleness the efficiency figure had; `angular_resolution.png` in particular is superseded by the waveform-first σ_θ (1.08°/1.11° on det3, not 1.66°) |
| `efficiency_breakdown.png`, `efficiency_residual_tail.png` | **`mpgd26/make_efficiency_breakdown.py`** — one command, reads only `mx_june_wft/02_efficiency.py`'s JSON reductions, writes straight into `assets/img/`. Regenerated 2026-08-10 on the waveform-first chain, `sat_det3`, 7,055 reference muons: **93.5 %** within 5 mm. Replaces the `pdftocairo` route from `engineer_package/figures/21-det3A-efficiency-breakdown.pdf`, which was 2026-07-14 hit-chain vintage on a *different* det3 run (`g_det3_wknd`, 92.9 %) **and** carried the stale "88.8 %" annotation — that literal was hardcoded in the engineer-package script's annotation string and is now derived there too, so the inconsistency cannot recur on either path. Full old-vs-new accounting and the slide markup: [`HANDOFF_efficiency.md`](HANDOFF_efficiency.md) |
| `ear2_onfig_{1_target,2_neutrons,3_collimation,4_dump,5_station}.png` plus `ear2_onfig_5_station_4arm.png` (**on slides 5–9**) and `ear2_beamline_{1_target,…,5_station}.png` (frame 5 is **on the backup slide**) | **`mpgd26/make_ear2.py`** — one command for everything: `cd mpgd26 && ../.venv/bin/python make_ear2.py` (~1 min 40 s), written straight into `assets/img/`. Two label layouts of the same render, both built every run: `ear2_onfig_*` puts the labels **on the drawing's own background, left and right** (wider canvas, 1740×2050, `annotate.side_labels`, `ONFIG_SIDES` assigns the sides by hand); `ear2_beamline_*` puts them in a **gutter column** down one side (`annotate.column_labels`). Swapping which the slides use is five `img src` edits — see the fifth pass at the top of this file for why each is where it is. It also writes `figures/ear2_build_<n>_<tag>_light{,_labelled}.png` with a live-text `.pdf`, `figures/ear2_onfig_<n>_<tag>.pdf`, and the standalone-figure aliases `figures/ear2_beamline_light{,_labelled}.png` that README.md and report.html cite. In-house PyVista render of the EAR2 vertical beam line, **a five-frame build-up**: target → the neutrons leaving at 90° → the middle of the line (collimator, floor, shielding, the pipe ending) → back into a pipe on up to the dump → the measuring station. The three frames are **strict subsets of one drawing**: same camera, lens, light rig, canvas and drawn scale, and each label column is solved once from the full set and then filtered, so **nothing moves and no label shifts between frames**. Parts and per-frame labels are `scenes_ear2.STAGE_PARTS`. Sourced heights, apertures and materials are in `scenes_ear2.py`'s docstring from C. Weiß et al. (n_TOF), *NIM A* **799** (2015) 90 and J. A. Pavon-Rodriguez et al. (n_TOF), *EPJ A* **61** (2025), arXiv:2505.00042 — **no sourced number has changed in any of the reworks.** Frame 5 is written **twice**: **two chambers in section** (the default and what the slide uses) and the true four-arm pinwheel (`STATION_ARMS`, `_4arm` suffix) — Dylan asked to compare, and chose the section pair because the pinwheel closes on the drawn capsule from both sides. ⚠️ Six things to know before editing. (1) **The lower beam pipe ends for real at `H_PIPE_END = 19.16` m**, ~1 m above the EAR2 floor, with the beam crossing the experimental space in air (Dylan, 2026-08-11 — 19.16 m is his recollection, not a drawing, hence the figure's `≈ 1 m above the floor`). (2) **A separate wide pipe starts at `H_UP0 = 20.45` m** and carries the beam on to the dump, cut off by the top of the frame; do not merge it with the lower one and do not delete it. (3) **The shape of the line inside the hall is scaled off the photograph** beside it, not off a drawing — shielding, stepped diameters, reducer, and the lead disks drawn at 0.36 m of the documented 0.57 m; all of that is in `ASSUMPTIONS`. (4) **The drawing stops ~1.35 m above the station**, so the bunker ceiling, its roof and the beam dump are real but above the frame. (5) **The chambers draw 24 strips, not 512** — the real pitch is 0.26 px here. (6) **The two drawn chambers are placed relative to the CAMERA** (`_section_angles`), so their drawn azimuths are 136° apart and the real station's 90° pinwheel is not what frame 5 shows — that is in `ASSUMPTIONS`, and the `_4arm` render keeps the true arrangement on record. In *that* render the near two chambers must be drawn UNCUT; run through `cut()` the cutaway deletes them outright, which silently makes the four-arm variant identical to the two-arm one. (7) **`BREAK_TILT` is 0** — a tilted section face reads as a stray diagonal line at this camera elevation. The one remaining departure from as-built geometry is the station's **support frame, not drawn** — declared in `scenes_ear2.ASSUMPTIONS`, in the on-figure label and in the backup slide's caption, and now also visible in the photograph beside it. Slide-copy labels must keep every line to **~21 characters** — `annotate` does not wrap, a longer line is silently clipped at the edge of the PNG, and the on-figure variant reuses `LABELS_SLIDE` for exactly that reason. **Replaced the borrowed schematic on slide 5.** |
| ~~`ntof_facility_schematic.png`~~ | cropped from Fig. 1 of G. Tagliente et al. (n_TOF Collaboration), *EPJ Web Conf.* 292, 12002 (2024), page 4 of `https://cds.cern.ch/record/2939795/files/fulltext.pdf` — **no longer used on any slide** (2026-08-10): its EAR2 content is in `ear2_beamline.png` and the only thing it added was the 185 m horizontal line to EAR1, which the deck no longer discusses |
| `ear2_hall_photo.jpg` | **crop of `ear1_ear2_photo.jpg`** (below), its right-hand panel, downsized: `cd mpgd26/slides/assets/img && ../../../../.venv/bin/python -c "from PIL import Image; Image.open('ear1_ear2_photo.jpg').crop((1960,0,3640,3264)).resize((860,1671), Image.LANCZOS).save('ear2_hall_photo.jpg', quality=86, optimize=True, progressive=True)"`. **Restored on 2026-08-11** at Dylan's request, and moved to the **left** of the render the same day at his request. It is the only thing in the deck that shows the real EAR2 hall and the **aluminium-profile support frame the render deliberately omits**, and since 2026-08-11 it is also the **source** for the shape of the line inside the hall (the white shielding, the stepped diameters, the reducer, the wide upper pipe — scaled against the ~1.2 m white disc at its foot). ⚠️ In it **the line is closed all the way from the floor to the ceiling**, with detector snouts on a chamber in the beam — a different configuration from the one the render draws. The backup slide's caption states both and claims nothing about which section is removable; **do not delete that sentence.** Credit **© CERN** is on all four slides that use it (a one-line `.fig-label` on the main three, in the caption on the backup) — keep it there. |
| `ear1_ear2_photo.jpg` | CERN Document Server, `https://cds.cern.ch/record/2148416/files/n-TOF-EAR1-EAR2.jpg` (OPEN-PHO-EXP-2016-006, © CERN) — this one **is** tracked (not a `.png`). Two panels; the deck uses only the right-hand one, via the `ear2_hall_photo.jpg` crop above. Kept as the uncropped source, and it is the photograph the beam-dump question originally came from. |
| `atomki_angular_correlations.png` | extracted (`pdfimages`) from `~/Downloads/Neff n_TOF Analysis Meeting X17 Update 3_24.pdf`, page 4; original data: Krasznahorkay et al., *PRL* 116, 042501 (2016) / *PRC* 104, 044003 (2021) / *PRC* 106, L061601 (2022) |
| `atomki_spectrometer_schematic.png` | cropped from Fig. 4 of J. Gulyás et al., arXiv:1504.00489 (*NIM A* 808, 21 (2016)), page 5 |
| `ideal_pair_spectrum.png` | `mpgd26/make_pair_kinematics.py` — **one command, reads a committed 394-line histogram reduction** (`mpgd26/data/ideal_pair_kinematics.csv`), so it rebuilds offline with no lxplus. The reduction itself comes from the Geant4 `pairs_thermal_trig_2cm` campaign's generator-truth vertices (`--reduce`, needs the 64 MB npz); provenance and the IPC model caveat are in the script's docstring. **The two curves are normalised separately and are 50/50 by construction — shapes, not a branching ratio.** |
| `status_*.png` (6 files) | `mpgd26/make_status_plots.py` — **one command, and it reads only committed reductions plus a small local mirror**; see that file's docstring for the four inputs and `~/.cache/mpgd26_status/` for where two of them are staged |
| `status_imon_response.png` | **`.venv/bin/python ntof_july_analysis/flash_charge/make_imon_figure.py`** (seconds; regenerate its input first with `ntof_july_analysis/flash_charge/imon_response.py --src /media/dylan/data/x17/beam_july`, ~2 min). Added 2026-08-10: the direct measurement of the HV-monitor's imon impulse response, which **closes the open systematic** the main charge slide and its backup used to carry — the readback is a ~1 s averager, so `mean − median` is the time-average current and the charge numbers are measurements, not lower bounds. Deliberately **not** in `make_status_plots.py` (that file was being edited concurrently); it reads only the committed reduction `flash_charge/results/imon_response_run_79.json` + the fold CSVs, uses `mpgd26/plotstyle.py` and adds **no new hues**, so folding it in later is one entry in that file's `FIGURES` dict. Measured on **run_79** (2026-07-26), not run_158 — same production setpoint, and the only production-point `hv_monitor.csv` in the local mirror; det C agrees with run_158's det C to 5 %. Full method, the timestamp-drift trap it had to survive, and the slide markup: `ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md` §8 and [`HANDOFF_imon.md`](HANDOFF_imon.md) |
| `target_pointing_fans.png` | copied from `…/analysis/wft/run_79/stat090_0000/mx17_A/wall_segment_tour/wall_segment_tour_all.png`, made by `ntof_tracking/run79_wall_segment_gif.py` |
| `target_pointing_slope.png` | copied from the same analysis dir, `figures/01_target_pointing_A.png` (not currently used on any slide — kept as the alternative to the fans figure) |

## Open items, by slide

**Setup build-up (16–27), added 2026-08-10** — one figure in nine states with
the step's explanation beside it, since 2026-08-11 a tenth slide that draws the
same thing as a plan, and after that two **placeholder photographs** of the real
station. Two decisions worth knowing before editing:

* **It is four acts at four cameras**, because the capsule is 23 mm and the
  setup is 1.2 m — and then because the layers are stacked *radially*. Slides
  16–18 are the close-up act (`micro`), slide 19 is the chambers arriving
  around the vessel (`close`), slides 20–22 are one fixed camera (`hero`), and
  slides 23–24 are the same setup **from straight down** (`over`, 89°,
  and drawn *bare* — only the active volumes keep their colour), because in any
  three-quarter view the trigger wall stands in front of the plastics and the
  liquid and a leg arriving in them cannot be seen. Within each act the frames
  are interchangeable stills. Slide 19 repeats slide 20's layer at a larger
  scale, so **cut it first** if the section runs long, then slide 17.
* **Only the charged pair is drawn.** The gammas in the event are real but
  neutral, so they crossed the picture as rays with no visible cause; the beam
  is a plain grey column with a slim direction dart on the axis inside it (an
  cut off at the vessel's **nose** once any detector is placed — past the target
  it reads as a pole holding the apparatus up, and alongside it a column wider
  than a 23 mm vessel makes the vessel look hazy. **The neutron is drawn,
  not transported** — a straight line up the beam axis to the vertex; the real
  history is still in the JSON but belonged to a different event. The legs are
  cut at the first layer not yet placed, so they grow with the apparatus. All
  fixed 2026-08-10.
* **The event is selected on how it leaves the vessel.** Both legs must exit
  through the *barrel* wall, transversely, rather than through a domed end
  where they would cross several times the wall thickness at a glancing angle;
  and the extractor prefers events landing in the two arms the figure draws
  solid. `--prefer-arms` must be kept in step with `scenes_ntof.NEAR_ARMS`.
* **The pair is one real Geant4 event; the neutron that made it is drawn.** It
  cannot be otherwise: the radiative branch that makes the ⁴He* is ~10⁻⁸ of
  ³He(n,p)t, so no simulated neutron history contains one. Slide 18 says this in
  its caption. **Do not quietly drop that caption** — it is the honest part, and
  for this audience it is also the interesting part.
* **Every build frame names the layer it just added**, and only that one — the
  label moves outward with the build instead of accumulating
  (`make_ntof.LAYER_LABEL` / `LAYER_POS`, anchored from the geometry by
  `scenes_ntof.layer_anchor`). **One label, a leader to each solid arm**: a layer
  is four objects and two are drawn solid, so a single line implies the label is
  about that one — and it is a line per drawn *object*, so the plastics (two bars
  per arm) carry four and the text drops the "2 ×" the lines already say.
  Top-left on the build frames, **bottom** on the close-up, where
  the chambers fill the frame and the only empty space is the see-through one.
  Sizes on the figure are in **cm** (read across a room); the bullets keep mm.
  It repeats the first bullet's size on purpose, so a frame lifted out of the
  deck still says what it is. Added 2026-08-11.
* **The overhead act draws the capsule whole and solid** (`VIEWS['over']`,
  `cut=False`). The cutaway plane contains that view direction, so from above it
  deletes the half of the vessel nearest the bottom of the frame instead of
  sectioning it — and the capsule is the one thing `BARE` does *not* whisper,
  because 23 mm on a 1.2 m frame faded to 10 % is a smudge at the exact point
  the picture converges on.
* **Slide 25 closes the section with a plan** (`make_ntof_plan.py --bare`,
  added 2026-08-11) — the one figure here that is **not a render**. Every frame
  before it has perspective, so the four arms sit at four different distances
  from the lens and every length is foreshortened by its own amount; the plan is
  orthographic and 1:1 in both axes, so the **204.5 mm standoff** (the dashed
  circle all four windows are tangent to), the layer radii (330 / 410 / 487 mm,
  as a dimension chain on arm B — the one arm no leg crosses, and the numbers
  are that arm's; the outer two layers shift a few mm per arm) and the **23 mm target**
  can be measured off the slide instead of being asserted in a caption. It also
  shows two things a three-quarter camera cannot: the ~16 mm pinwheel offsets,
  and that two of the four liquid vessels are laid on their side with their PMTs
  pointing sideways. What it gives up is the beam axis — the legs rise ~135 mm
  along it, so the opening angle **as drawn is 122° against a 110° space
  angle**, and the figure quotes both. Same geometry module and same event JSON
  as the renders. It repeats no content, so it is a clean place to stop if the
  section overruns.
* **The chambers grew on 2026-08-11**, and the frame did not. The sim's MM
  active area went from an unsourced 38 × 34 cm to the measured **39.9 × 36.0**
  (the short axis is the one *along the beam*: ~19 mm at each end of the strip
  plane is passivated), and every figure here imports it, so all nine renders and
  the plan were re-run. The **frame** is now drawn as the fixed **440 mm outer
  square** it really is (`scenes_ntof.FRAME_HU`) instead of a constant 30 mm
  cheek — which used to give the same 440 around the old size but gives 490
  around the new one, and 490 does not fit: the pinwheel has only 15.5–17.3 mm
  of tangential shift, and 440 is exactly what it clears. The plan view is where
  this shows, since it is the only figure with all four arms in one plane; drawn
  the old way the neighbouring arms interpenetrate by ~10 mm on the page. The
  MM bullet on slide 20 and the layer label on the figures were updated with it.
* **Two photographs, slides 26–27, are placeholders** (`make_photos.py`, added
  2026-08-11). Top-down into the station, and one arm from outside. They are
  parked here rather than placed: the section has already said everything they
  show, and the pairing that would actually pay is the top-down photo *beside*
  the plan drawing. Open questions are in the comment above them in `index.html`.

Not done: only the light theme is rendered. `make_ntof.py --theme dark` works if
the deck ever goes dark; `make_ntof_plan.py` is light-only by construction
(`plotstyle` has no dark variant, as the deck's CSS has none).

**Title** — **done 2026-08-10.** Dylan Neff, MPGD 2026 Prague, 3 September 2026.

**Running order** — **the slot is 15 min + 5 min**, so the deck's 28 main-flow
slides do not fit. A concrete 16-slide cut, with what moves to backup and why,
is in [`RUNNING_ORDER.md`](RUNNING_ORDER.md). Read that before doing any more
slide work — some of the open items below are on slides that cut proposes to
demote, which changes how much they are worth.

**Motivation / ATOMKI** (comment above slide 3): condensed to 2 main-flow
slides + backup already done. Still open — beam orientation at ATOMKI
unconfirmed (deliberately left off every slide); PyVista remake of the
5-telescope schematic in house style not started; highlighting the
³H(p,e⁺e⁻)⁴He channel specifically as the bridge into the EAR2 slide not
done (evidence slide still gives ⁸Be/⁴He/¹²C equal weight).

**n_TOF / EAR2 facility** (slides 5–9, plus one backup) — **done 2026-08-10, revised four times on 2026-08-11.** EAR1 is out of
the discussion and the borrowed *schematic* is gone: the slides are the in-house
render (`mpgd26/make_ear2.py` + `scenes_ear2.py`) with the cropped CERN
photograph of the hall beside it. The Tagliente Fig. 1 schematic was **folded in
rather than kept as a second panel** — everything it said about EAR2 is in the
render, and the one thing it added was the 185 m line to EAR1.

**The five main-flow slides carry no caption; one backup slide carries it in
full.** Both pictures are portrait, so a six-line caption was eating a fifth of
the height they needed. Everything that was in it — the provenance, the citations
and every drawn-rather-than-measured item — is on the backup slide *"The EAR2
render, and what in it is drawn rather than measured"*, under the same figure. The
main slides keep a one-line `.fig-label` credit under each picture. ⚠️ **If you
change what the render draws, change that backup caption with it** — it is now
the only place in the deck where the disclosures live. Column widths on the main
slides are **proportional to the two images' aspect ratios** (0.62 / 1.00 / 1.04);
that is load-bearing, not cosmetic — see the fifth pass at the top of this file.

**It is five consecutive slides, one drawing built up** (2026-08-10 second pass,
then four rounds on 2026-08-11, all on Dylan's review): frame 1 the lead target
and the protons, frame 2 **the neutrons leaving at 90°** and filling the pipe with
the break annotated, frame 3 the whole middle of the line — second collimator,
EAR2 floor and its shielding, lead disks, the pipe **ending** a metre above the
floor — frame 4 **back into a pipe, on up to the dump**, frame 5 the measuring
station. Same camera, same scale, same labels-in-place — the deck has no fragment
mechanism (`make_pdf.sh` prints one page per `<section>`), so a build *is*
consecutive slides. The bullets accumulate, with the already-narrated ones set
`li.dim` and **shortened as they pile up** so frame 5 still fits, and the figure
markup is **deliberately identical on all five** so neither picture can move
between them. The **photograph is on all five frames** and is not part of the
build. ⚠️ One thing in the *render* is not a strict subset and it is deliberate:
the beam is drawn at a whisper across the 0.5 m the sample occupies, on frame 5
only — see the station paragraph below.
**At 15 minutes all five go to backup as one unit** — this does not change the
16-slide cut in [`RUNNING_ORDER.md`](RUNNING_ORDER.md), and if you show only
one, show frame 5.

**Three legibility fixes, all from Dylan looking at the output** (2026-08-11
seventh pass), each of which is a trap worth knowing before you touch the palette
or the break:

- **`COL['neutron']` is dark on purpose** (`#8b96a3` → `#55657a`, arrows 9 → 12 mm).
  The old grey has the same *value* as the amber-tinted pipe interior it sits in,
  so the five arrows leaving the target were invisible. ⚠️ The obvious fix —
  white — is wrong: the same neutrons cross the open hall higher up on a
  near-white background. A darker slate is the one colour that works on both.
- **One `ENV_ALPHA = 0.16` for the beam envelope everywhere.** It used to be 0.34
  above the collimator against 0.17 below, and the same amber at twice the opacity
  on a thin tube seen through both its walls came out **olive-brown** — two
  different-coloured beams in one figure. The thin pencil is carried by its
  tracks, which are drawn far wider than the beam really is for exactly this
  reason.
- **`BREAK_TILT = 0`.** The 24° tilt existed so a break would read as a drawing
  cut rather than as a pipe that stops. From a camera 6° above the horizon a
  350 mm disc tipped 24° projects to a thin dark **diagonal streak** across the
  pipe, which is what it actually read as. The collimator's own break face was
  flattened for the same reason a few hours earlier; this finished the job.

⚠️ **The target is now Target #3, drawn from its design paper** (2026-08-11,
second answer). Dylan asked whether the shape was known or guessed; the first
answer was that the lead body was sourced but was the **Target #2 cylinder** —
a real object, retired at the end of 2018, and *not the one our data are on* —
with none of the EAR2-facing assembly drawn at all. He asked for it as accurate as
feasible, so it was rebuilt from

> R. Esposito et al. (n_TOF), *Design of the third-generation lead-based neutron
> spallation target for the neutron time-of-flight facility at CERN*,
> **Phys. Rev. Accel. Beams 24 (2021) 093001**, arXiv:2106.11242 — Sec. III A the
> core, Sec. III B the vessel and the two moderators, Sec. II B the 4 cm water,
> Sec. II C the 5 cm lead plate.

Sourced and drawn to scale: **six lead slices** on the proton axis, **600 × 600 mm**
in cross-section, **five 50 mm + one 150 mm** with the thick one at the
**downstream** end against the EAR1 moderator (high-purity UNS L50006, ≥ 99.98 wt%);
**9.85 ± 0.05 mm** aluminium (EN AW-6082 T6) **anti-creep plates** between them,
which carry the **nitrogen** cooling channels — the slice gaps are held to
45–195 µm; an **AISI 316L vessel** at 0.5 bar N₂ with the proton window locally
thinned to **3 mm**; the **4 mm** stainless **neutron window** electron-beam welded
to its top; the **50 mm lead plate** that window supports, which buys back the
factor-6 prompt-γ increase the better EAR2 coupling cost; the **EAR2 moderator**
outside the vessel, bolted to it and resting on that plate, holding the **40 mm
water layer** FLUKA picked as the optimum for the EAR2 resolution function
(aluminium EN AW-5083 H112, two independent circuits, demineralised **or** borated
water); and the **hemispherical aluminium vacuum window** above it (that one from
Pavon-Rodriguez Sec. 3 — the previous target coupled to EAR2 through a *polygonal*
window instead).

That last stack is why the rebuild was worth doing rather than just correcting the
caption: **it is the part of the target this figure is about and it was missing.**
`H_PIPE_START` moved **0.30 → 0.60 m** to clear it, its flanges with it, and the
neutron envelope now starts at the top of the water rather than inside the lead.
A **twelfth label**, `water moderator + lead plate`, was added to frame 1.

⚠️ **Four target details are drawn, not sourced**, all wall thicknesses or extents
the paper does not give: the vessel wall (`VES_GAP`), the moderator cans' walls and
plan size (`MOD_WALL`), how far the EAR1 moderator extends (`EAR1_MOD_X` — the
paper says only that it is the larger of the two), and the vacuum window's radius
(`R_VACWIN`, drawn at the pipe bore). All four are in `ASSUMPTIONS` and on the
backup slide. The **EAR1 moderator is drawn as aluminium with no water fill** on
purpose: filled to match the EAR2 can it is a 0.3 m³ block of blue that outweighs
the 40 mm layer the figure is about.

**The PE-shielding leader crosses to the left of the pipe** — its label is in the
left column and its anchor was at +x, so the leader ran over the beam pipe to
reach a part that is present on both sides of it anyway.

**The measuring station, rebuilt** (2026-08-11, Dylan: *"I'm also not sure what
the green is supposed to be there"*). It was four 400 mm slabs in PCB colour with
translucent gas in front, and seen obliquely that is a green box and nothing else.
Each chamber is now built with `meshes.rect_chamber` and drawn as one: aluminium
frame, readout board **with strips on it**, 30 mm of lit drift gas facing the
sample, and the entrance window as a tint. The strips are what makes it read as a
detector at a glance — ⚠️ **24 are drawn, not 512**, because the real 0.78 mm
pitch is 0.26 px at this scale and comes out a flat grey wash; that is in
`ASSUMPTIONS`. `"Measuring station"` also became **two labels**, and on the slide copy they are deliberately **generic** — `sample` and `detectors`, not ³He and Micromegas. This is slide 9; the experiment is introduced on slide 16, and the specific words made the facility slide look like the setup slide. The standalone figure's `LABELS` still carry the full description. And the beam is **faded across the sample** on this frame only
(`add_neutrons_hi(fade_sample=True)`): at full strength it is a grey line down the
middle of a translucent capsule and reads as the capsule being *behind* the beam.

**Frame 5 exists in two versions and both are built every run** (Dylan asked to
compare): `ear2_onfig_5_station.png` shows **two chambers in section** and is the
default and what the slide uses; `ear2_onfig_5_station_4arm.png` shows the true
**four-arm pinwheel** — two solid, two the camera looks through drawn as bare
frames. `scenes_ear2.STATION_ARMS` picks; swapping is one `img src` edit.

⚠️ **The section pair is a drawn arrangement, and it is not the station's.**
`_section_angles()` places the two relative to the **camera** — just past edge-on,
on the far side, so the cutaway slices them and every surviving piece sits behind
the sample — which makes their drawn azimuths **136° apart rather than the real
90°**. That is declared in `ASSUMPTIONS` and the `_4arm` render keeps the truth on
record. It is worth the departure because the pinwheel's own geometry, at this
camera, brings the two solid arms in at 45° to the screen and closes them on the
drawn capsule until they touch it: the sample ends up a sliver between two green
rectangles, which was the original complaint about this part of the figure and is
not fixed by drawing the chambers in more detail. `SECTION_TILT = 22°` sets how
much board face shows; at 0° you get two 42 mm bars that read as nothing.

⚠️ In the **four-arm** render the near two must be drawn **uncut**: the cutaway
plane passes through the beam axis, so run through `cut()` they are not faded but
*deleted*, which is why the first four-arm attempt came out pixel-identical to the
two-arm one. `frame_ring` is a hollow band, so drawing them solid frames the
sample rather than hiding it.

**The sample is centred between the two pipes** (2026-08-11, Dylan): `H_UP0` went
20.45 → **20.74 m**, which puts 19.95 m exactly halfway between the end of the
lower pipe (19.16 m) and the start of the upper one, and the upper pipe shows
**only its entrance flange and one ring** before the top of the frame cuts it.
`DRAWN_H` is 6.11 m. And `Lead target` moved to the **right** label column,
because with it and `protons` both on the left their leaders crossed under the
target.

**It was four frames for a few hours** (2026-08-11): the dump got its own frame,
because collimation and the dump had been sharing frame 2 and that buried the
collimators — Dylan's reaction to that version was *"where is the first
collimator, I actually don't know?"*. Then the dump came out of the drawing
altogether, so the dump frame had nothing left to reveal and it went back to
three. Collimation still has its own frame, which was the point, and the
measuring station still appears **only on the last frame**: everything before it
is beam line, and the last thing to land is us.

**The drawing stops just above the station** (2026-08-11): the bunker ceiling, its
break and the beam dump drawn in full used to occupy a third act at the top. It
was 2.7 m of drawn height and the widest object in the picture, all of it
shielding, and it left the station small. Removing it took `DRAWN_H` from 8.67 m
to 5.71 m (5.91 m once the upper pipe needed room), so at the same canvas
everything left is drawn **~1.5× larger**, and the upper break went with it, which
Dylan did not want either. Nothing is asserted away: the beam leaves the top of
the frame inside the wide pipe that really carries it there, under a label giving
the dump's entrance height, and both `ASSUMPTIONS` and the backup caption say what
is above the frame. `FLOOR_W` was narrowed 2400 → 1900 mm in the same pass,
because at 1.5× a 2.4 m floor slab reaches almost the full frame width and
competes with the station. The dump's own dimensions survive as a comment in
`scenes_ear2.py` and in the slide caption; the geometry is in git history.

**The photograph is back, on the left of the render** (2026-08-11, Dylan asked for
it, and asked for it on the left): `ear2_hall_photo.jpg`, the right-hand panel of
the CERN photo cropped and downsized. It earns its place three times over — it
shows the real hall, it shows the **aluminium-profile support frame the render
deliberately omits** (the honest way to have that omission both ways), and it is
now the **source** for the shape of the line inside the hall. ⚠️ **In it the line
is closed all the way from the floor to the ceiling**, with detector snouts on a
chamber in the beam. Ours ends the lower pipe about a metre above the floor and
puts the capsule in air above it, with a separate wide section above taking the
beam on to the dump. The backup caption states both and claims nothing about which
section is removable — **do not delete that sentence**, and do not "reconcile" the
two pictures by guessing. Open question in "Left:" above.

**The line inside the hall is modelled from that photograph** (2026-08-11, Dylan:
*"can we try to add a bit more realism to the bottom beam pipe, with the white
polyethelyne somewhat modeled and a rough attempt at tapering the pipe"*). What is
drawn now, bottom to top: the white **segmented polyethylene shielding disc**
lying on the floor around the pipe penetration with a **collar** above it
(`add_shield`, `pie_segments`), the **lead-disk vacuum chamber** at the shaft's
317 mm bore, a **reducer** (`conical_shell`), the **narrow tube** at 156 mm, and
the flanged **exit window**. ⚠️ Radii are scaled off the photo against its ~1.2 m
white disc and the step heights are chosen to fit between the floor and the end of
the pipe: this is a reading of a *photograph*, not of a drawing, and
`ASSUMPTIONS` says so in those words. Two consequences worth knowing. The
shielding needed **its own colour** (`COL['pe']`, a warm white) because on
`COL['bpe']` it vanished into the floor slab it stands on — so `COL['floor']` went
darker in the same pass. And the **lead disks are drawn as a 0.36 m stack against
the documented 0.57 m** effective height, because the drawn hall segment is 1 m
tall and also has to carry the collar, the reducer and the exit flange; the figure
claims *that* there is further collimation in the hall and roughly where, not how
much.

**There is a pipe above the experimental space, and it is a different pipe**
(2026-08-11, Dylan: *"the top pipe toward the beam dump included and cut off at
some point"*). `add_uppipe` draws the wide section from `H_UP0 = 20.45` m up past
the top of the frame, **cut off by the camera margin rather than capped** — which
reads as "continues" in a way no drawn cap does. So the line reads the way the
photograph reads: narrow tube → flanged end → **open space with the experiment in
it** → wide tube away. It is drawn at a **160 mm bore rather than the photo's own
proportions**, because at full size it outweighed the measuring station, which is
the opposite of what removing the beam dump was for. The height at which it
begins is drawn, not sourced.

**"What is this pink diagonal thing in the collimator?"** (Dylan, 2026-08-11) — it
was the **break's own section marker**: `COL['section']` on a 24°-tilted disc at
the collimator's 680 mm radius, so a third of it hung outside the block and read
as a fin bolted to it. It was never hardware, and it was in the wrong place too —
drawn at the act boundary, it floated 120 mm up inside the block instead of
sitting on the block's own bottom. Now **flat, inset and at the block's bottom**,
with `COL['section']` desaturated. The tilt survives on the pipe's face below,
where the disc stays inside its own silhouette.

**The beam pipe really does end, ~1 m above the EAR2 floor** (Dylan,
2026-08-11) — and the space above it is open hall for experiments to stand in,
which is where our station is. The 2026-08-10 version of this render ran the
pipe continuously from the collimator, past the station and up through the roof
into the dump, and then **cut it away** above the floor as a drawing device with
a section marker and a three-line disclaimer. That was **wrong about the
facility**, not just about the drawing, so the pipe now terminates for real, in
a flat circular end with a flange (`H_PIPE_END = 19.16`, and `H_PIPE_TOP` is
gone), the neutrons cross the experimental space **in air**, and the disclaimer
went with it. The `section` accent is once again used for **only** the break in
the vertical scale, which is what it was for. **Do not join the lower pipe to the
upper one** — the wide section above 20.45 m is a *separate* pipe with open space
under it, and merging the two is precisely the mistake this correction undid.
19.16 m itself is not from a drawing — it is Dylan's recollection of the hall,
which is why the figure says `≈ 1 m above the floor` and not a height.

**One departure from the as-built geometry remains, and the deck says so.** The
station's support frame is not drawn — it really hangs in an aluminium-profile
frame standing on the floor — so the capsule appears to float. That was Dylan's
call, for legibility: drawn as built, the capsule and its four chambers sit
behind four grey uprights. It is disclosed three times over: the station
label's `(its support frame is not drawn)`, `scenes_ear2.ASSUMPTIONS` (hence the
standalone figure's caption), and the slide caption. **Do not quietly drop any
of those.** Smaller things are declared in `ASSUMPTIONS` only, because nothing on
the slide turns on them: the exit and entrance windows are schematic (no material
or thickness is claimed), and the in-hall diameters and step heights are the
photograph reading described above.

**Where the first collimator is** — 7.4–8.4 m above the target, i.e. **~10 m
below the EAR2 floor**, down in the shaft: 1 m of iron with a 200 mm bore
(Weiß Table 2). It is invisible in the render because it falls inside the lower
break, which is exactly why the break's label now names it and frame 2's bullet
spells it out. The second collimator is 15.04–18.04 m, 2 m of iron plus 1 m of
borated PE, bore 70 → 21.8 mm, and the sweeping magnet (10.4 m) and the 8-slot
filter station (11.4 m) are in that same break.

**The structure at the top of the pipe is the beam dump, and it is now
verified** — the old "tentatively the beam dump, UNVERIFIED" caveat is
discharged. C. Weiß et al. (n_TOF), *NIM A* **799** (2015) 90, §2: *"The beam
dump for this new vertical flight path is installed on the roof of the
bunker."* §2.4 and Fig. 5 give the three layers the render draws — a borated-PE
core 400³ mm with a 340 × 250 mm entrance bore, iron 1600³ mm, concrete
3200 × 3200 × 2400 mm — and Table 1 puts its entrance at 24.73 m above the
target. Independently confirmed by J. A. Pavon-Rodriguez et al. (n_TOF),
*EPJ A* **61** (2025), arXiv:2505.00042 §5 (beam *"stopped at the beam dump
downstream (not shown), placed above the ceiling"*) and by the CERN Bulletin of
29 July 2014. **Do not write "19.5 m" for the flight path** — no source says
it. What the literature actually quotes: ~20 m nominal, EAR2 floor 18.16 m,
ceiling 23.66 m, nominal sample position 19.76 m, and the current reference
flight path **19.95 m** (Pavon-Rodriguez, Figs. 10–14), which is the number the
render and the slide use.

**Chamber design** (comment above slide 7): ✅ **done 2026-08-10** —
`mx17_board_peel_zoom.png` is built and in `assets/img/`. The slide markup
change (swap the right-hand `img`) and the **caption pitch correction
(0.7785 → 0.78 mm)** are written up in
[`HANDOFF_board_peel.md`](HANDOFF_board_peel.md); `index.html` was
deliberately not edited. The HTML comment above slide 7 can go once the swap
is made.

**Cosmic bench** slide: flagged in-slide as pending the running-order
decision with Alexandra's P2 talk.

**Efficiency** — **done 2026-08-10, markup ready in
[`HANDOFF_efficiency.md`](HANDOFF_efficiency.md), not yet applied to
`index.html`.** All five numbers were re-derived on the current
(waveform-first) chain: det3 92.9→**93.5**, det2 91.3→**91.9**, det6
57.8→**75.4**, det7 43.1→**56.9**, det4 20.7→**41.9 %**. The three
spark/gain-limited chambers move by 14–21 points, so the old fleet bars and
the old caption are both wrong, not merely stale — det4 in particular now
*detects* 95.8 % of crossings, not "~70 %". The breakdown figure is
regenerated and self-consistent (annotation derived, no literals); the
residual pairing is a new tail-focused figure, `efficiency_residual_tail.png`,
rather than a crop of `spatial_residuals.png`, because that image is
hit-chain vintage and would not match the refreshed breakdown's basis.
Both blocks of markup from the handoff are now **applied** in `index.html`
(main slide → det3 alone, fleet bars → new backup slide).

**Resolution** — **the old caveat here was wrong and has been withdrawn
(2026-08-10).** It said the tile was reference-limited because the M3 detectors
"only have ~500 µm resolution"; that is the **per-plane** figure. What matters is
the four-plane fit's pointing interpolated to the DUT plane, measured on the same
run at **0.21 / 0.24 mm (X/Y)** — the DUT sits ~50 mm from the telescope
centroid, where interpolation is best. The reference is 10–21 % of the residual
*variance*, so the quadrature subtraction is stable to < 0.01 mm and the tile now
carries the deconvolved **0.57 / 0.69 mm**. Do not call this measurement
reference-limited on stage. Derivation and sources:
[`HANDOFF_resolution.md`](HANDOFF_resolution.md), which also documents two
pre-existing repo errors it uncovered (a "0.40 mm M3 pointing floor" quoted in
three places that is actually the *deconvolved DUT* value, and a radial-vs-per-axis
units mismatch in `m3_self_resolution/analyze.py`).

*Still open on this slide, and worth doing before 3 September:* both its figures
are superseded. `spatial_residuals.png` is hits-basis from `g_det3_wknd`, a run
never reprocessed after the 2026-07-25 significance-floor fix — regenerating from
`sat_det3` on the waveform-first basis (the basis the rest of the deck now uses)
gives **≈ 0.50 / 0.47 mm**, better than the slide claims. And
`angular_resolution.png` is hits-basis at 1.66° while the tile advertises the
waveform-first 1.0–1.1°; the tile states both honestly today, but regenerating the
figure would let it lead with 1.1°.

**Status** section — **built 2026-08-09, 11 main-flow slides + 3 backups.**
Everything on them is sourced; `STATUS_PLAN.md` is the map. What is still open:

- **Slide D9 (the physics reach) is DECIDED and REMOVED — 2026-08-10.** Dylan
  chose `STATUS_PLAN.md` §5 framing **option (c): say nothing about yield**,
  because there will not be enough analysis by 3 September to support a reach
  statement. The talk is framed as a status report with analysis ongoing. The
  slide's markup is in git history (last present in `d3e4993`) and its content is
  unchanged in `STATUS_PLAN.md` §5 if it ever needs to come back. Two knock-ons
  applied: D0's "physics reach is far below projection" bullet became an
  "analysis is ongoing" bullet, and D9's **σ(p)/p ≲ 30 % tracker requirement**
  survived onto the outlook slide — it is a detector statement, not a yield
  statement. The thermal-window rate table stays in **backup**; having it ready
  for a question is not "saying it".
- **The charge measurement's one systematic is still open** — it assumes the CAEN
  imon readback preserves the time-average of a short current burst. If it does
  not, every charge number is a lower bound. Closing it is item 4 of
  `ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md`. Note the
  "cheapest route is a spec lookup" line is optimistic: **the board model is
  recorded nowhere in any of the repos** (the Python wrapper is generic and the
  crate sits behind a CFE server at 128.141.177.244), so the lookup needs either
  Dylan's knowledge or a crate-map query — and the crate must **not** be touched
  while production data taking is live. A stronger route that needs no hardware
  and no new data: phase-fold imon against the beam-intensity log's per-pulse
  timestamps to measure the monitor's impulse response directly. **That is now in
  flight** (third attempt — the first two were killed by infrastructure failures;
  the time reconstruction had been validated to ~1 ms jitter). It will also deliver
  a deliberately over-detailed **teaching slide** explaining the method, which
  Dylan asked for so he can follow the argument before it is pared back for an
  audience. Watch for `HANDOFF_imon.md`.
- **The target-imaging slide is on PRELIMINARY reconstruction** — one arm, one
  sub-run, 3 of 13 file tags of run_79, on a transferred bench calibration whose
  own document says "nothing here is quotable". Either do the in-situ
  calibration or keep the caveat visible. None of the ~1.7 TB of production
  statistics has been reconstructed yet; a four-arm back-projection over even a
  slice of it is what would turn "the fans converge" into an actual image.
- **It is too long** — and now quantified: the slot is 15 + 5 min against 28
  main-flow slides. The cut is proposed in [`RUNNING_ORDER.md`](RUNNING_ORDER.md),
  which keeps D1, D1b, D5 and D6 in the main flow, demotes D2/D3/D4/D7 to backup,
  and argues for demoting D8 too on the preliminary-calibration grounds below.

**Target slide** (n+³He physics) — **done 2026-08-10.** The ideal-case e⁺e⁻
kinematics figure did not exist anywhere in the repos; it does now, as a Geant4
campaign product rather than a hand-drawn curve — `ideal_pair_spectrum.png` from
`mpgd26/make_pair_kinematics.py` (provenance row above). It is on a new slide 6b,
"What the detector has to separate", after the n+³He channel slide, because the
kinematic confinement of a two-body 17 MeV decay above 109° is the cleanest
argument in the deck for *why this needs a tracker*. **At 15 minutes it is a cut
candidate** — `RUNNING_ORDER.md` treats it as promotable/demotable at Dylan's
discretion. Two things to say if asked: the two curves are shapes normalised
separately (50/50 by construction, **not** a branching ratio), and the IPC curve
is the standard dN/dM_ee ∝ 1/M_ee virtual-photon stand-in with no E0/M1/E2
multipole correlation.

## The strip pitch was wrong everywhere — corrected 2026-08-10

The deck (and the mpgd26 package, and `report.html`) said **0.7785 mm**. That is an
off-by-one: 398.58 mm is the first-to-last strip *centre* span, i.e. **511**
pitches, and dividing it by the strip *count* 512 gives 0.77852. The design pitch
is a round **0.78 mm** and the active area is 512 × 0.78 = **399.36 mm** —
gerber-verified, strip dots at ±0.39, ±1.17 mm, spacing exactly 0.780. Found while
building the zoom figure, which now carries a burned-in 0.78 mm caliper that would
have contradicted the old caption on the same slide.

Corrected in: the slide-7 caption, `scenes_chamber.py` (both the docstring and
`STRIP_PITCH_MM`, which computed it wrongly — now `/(N_STRIPS - 1)`),
`scenes_microtpc.py`, `make_chamber.py`, `make_report.py`, `README.md`, and
`report.html` (patched in place rather than regenerated, so as not to discard
hand edits in it).

**Not affected:** no analysis code uses 0.7785 — grepped across `common/`,
`wft/`, `mx_june_wft/`, `mx_june_cosmic_qa/`, `ntof_tracking/`,
`cosmic_bench_analysis/`. The 398.6 in `common/mx17_active_area.py` is a fiducial
box edge, which is legitimately the strip-centre span. **So no reconstructed
position is wrong** — this was a documentation error only.

*Loose end:* `chamber_exploded.png` and `microtpc.png` were rendered with the old
constant, so their drawn geometry is 0.2 % off. That is invisible at slide size and
the code is now right, so they will self-correct on the next re-render. Not worth a
re-render on its own.

## The plan slide, answered: two questions it invites — 2026-08-12

Both came from Dylan reading the finished figure, and both are the sort of thing
that gets asked from the floor. Numbers below are from `data/ntof_event.json`
itself (event #208), not estimates. **This is ONE event** — do not generalise
either answer into a statement about the setup.

### "110° in space but 122° as drawn — is that scattering in the capsule?"

No. Both numbers come from the *same* primary directions at the vertex: the
generator recorded 110.1°, and recomputing from the direction vectors the figure
actually draws gives 110.2°. Nothing about the material enters.

It is pure projection, and the intuition that projecting can only widen an angle
is not right — it can go either way:

```
cos θ_3D = a⊥·b⊥ + a_y b_y            (unit vectors, y = beam)
cos θ_2D = a⊥·b⊥ / (|a⊥| |b⊥|)
```

In this event **both legs leave going up** (+y components +0.284 and +0.407;
polar 73.5° and 66.0° from the beam), so `a_y b_y = +0.116` — the one term
pulling cos θ *up*, i.e. making the space angle **less** obtuse than the drawn
one. Drop it and renormalise by `|a⊥||b⊥| < 1` and you get 121.7°. Had the legs
gone to opposite sides of the drawing plane, projection would have **shrunk** the
angle instead. Recorded in the `make_ntof_plan.py` docstring too.

### "14.8 of 19.6 MeV seen — where did the other 4.7 go? The MM PCB?"

The PCB is the biggest *dead-material* sink but it is not most of it. Full
budget, by tracking each primary's KE loss region by region (sums exactly to the
19.558 MeV the pair started with):

| Where the energy went | e⁻ (arm D) | e⁺ (arm C) | total |
|---|---|---|---|
| **Seen** — deposits in sensitive volumes | 8.39 | 6.46 | **14.84** |
| ³He capsule itself (gas + Al + CFRP wall, ~13 mm crossed) | 0.63 | 0.50 | 1.13 |
| MM stack dead material | 0.61 | 0.52 | 1.13 |
| air, containers, plastic wrapping (~460 mm of path) | 0.06 | 0.08 | 0.14 |
| radiated out of the plastic/LS and never re-absorbed | 1.06 | 1.27 | 2.33 |
| | | | **19.56** |

Inside that MM row, per leg (e⁻ arm D / e⁺ arm C), read off a 0.2 mm dE/dx
profile through the stack rather than from volume names:

| | e⁻ | e⁺ |
|---|---|---|
| **readout board laminate** (1.7 mm of Cu/FR4/kapton, w = 35.0–36.8 mm) | **0.507** | **0.421** |
| Rohacell backing (5 mm, w = 36.8–41.8) | 0.048 | 0.048 |
| back mylar/Al foil (w ≈ 41.8, 40 µm) | 0.003 | 0.003 |
| mylar + Al window, drift cathode | 0.012 | 0.014 |
| **the 30 mm of drift gas** | **0.046** | **0.032** |
| gas behind the mesh (w = 30–35, dE/dx ≈ 0) | ~0 | ~0 |

So the chamber's whole material cost to a 10 MeV electron is the **readout
board** — 0.93 MeV of the 1.13, about a fifth of the total deficit — and the gas
it actually measures in costs 0.08 MeV. Everything else is foils.

⚠️ **The 8 mm Al support plate is NOT in this budget, and must not be put in it**
(Dylan, 2026-08-12). It is built by `addRing` in
`MX17_Geant/shared/MX17ModuleGeometry.hh` — a plate with a **402 mm square
through-aperture concentric with the active area** (`AsBuiltSpec`:
`plate_mm = 8.0`, `plateAp_mm = 402.0`), i.e. a frame around the outside with
nothing over the active area. These legs cross at |u| ≤ 83 mm and |v| ≤ 92 mm,
deep inside the ±201 mm aperture, and the profile shows air (0.1–0.2 keV/mm)
from w = 42 mm out. A track that did cross 8 mm of Al would lose ~3.7 MeV —
i.e. if the plate ever appears in an energy budget, something is wrong.
An earlier draft of this note credited it with 0.14 / 0.11 MeV; that energy is
the Rohacell and the foil behind it.

**The dominant term is escaping photons, ≈2.3 MeV — half the deficit.** Mostly
arm C: the e⁺ spent 5.97 MeV in the 25 mm plastic, arrived at the liquid with
~1.2 MeV, stopped in its first 2.2 mm, and arm C's LS recorded only 0.077 MeV.
The two 511 keV annihilation gammas evidently left as well — and note they are
*extra* energy, not part of the 19.56, so they do not close the gap either.

Rebuild the table: the region-by-region KE decomposition is a few lines over
`ev['legs'][i]['ke']` and `['layers']`; `layers` labels every step, and `'world'`
means any non-sensitive volume.

## The SiPM dead bars were mirrored — corrected 2026-08-12

Dylan, off the top-down slide: "the SiPM bars seem to be shifted in the wrong
direction — there should be one unread on the left and three on the right,
looking from the top and behind the MMs." Correct, and the deck had it backwards.

The 16-of-20 read-out window is shifted **one bar toward the MM**, and the MM is
pinwheel-shifted along **−u**, so the window goes to −u: **live bars 1–16**, dead
**{0}** on the MM side (left, seen from behind the wall) and **{17,18,19}** on the
far side (right). `DetectorConstruction.cc` and `mx_july_beam_qa/mx17_geom.py`
both had that sign; `MX17_Full_Geant/scripts/plot_geometry.py` — which every
figure in this package imports — had `+shift` instead of `−shift`, so it drew
bars 3–18 and mirrored the dead ones. **Drawing-only: the simulation is
unaffected.** One-line fix in `plot_geometry.py`, dated comment there and a full
write-up in `GEOMETRY_COORDINATE_CONVENTION.md` §6.

Regenerated and re-copied: `setup3d_6_sipm`, `_7_plastic`, `_8_plastic_top`,
`_9_full` (slides 26–29), `ntof_plan.png` (slide 30), the backup
`setup_2_sipm` / `_3_plastic` / `_4_full` (slide 51, from `plot_buildup.py
--style clean`), and the `build_ntof` / `turn_ntof` animations in `report.html`.
`setup_topdown.png` does not draw individual bars and was left alone.

⚠️ **Still unverified against hardware.** Sim, analysis, figures and the written
convention now agree, but they are all the *same* convention — no measurement has
confirmed which side the real dead bars are on. The check is to project
reconstructed MM tracks onto the wall and see which group fires; logged as an
open item in `GEOMETRY_COORDINATE_CONVENTION.md` §6, with the σ ≈ 47 mm pointing
blur caveat. Nothing on the slides depends on it — the wall is drawn, never used
for a number — so this is not a blocker for 3 September.

## Dylan's review of the two new renders — 2026-08-10, in flight

Both were reviewed and both got a verdict of broadly good with specific changes.
Sessions are working on these now; if they are not reflected in the figures on
disk, they did not land.

~~**EAR2 beam line** — split into **three progressive frames** that are strict
subsets of one final picture (1: lead target alone · 2: + the broken ~20 m flight
path, pipe and beam dump · 3: the full picture), same camera and scale throughout.
In the full frame: the beam pipe should **end not far above the floor**, the
pillar-like supports around the station come **out**, and the capsule should
**float above the top of the beam**.~~ **DONE 2026-08-10, then SUPERSEDED
2026-08-11** by a second round of review on the result — see below. Its guess
that "the real pipe continues through the roof to the dump" was **wrong**, and
the resulting cut-away-pipe disclaimer is gone.

**EAR2 beam line, round two — 2026-08-11. DONE.** Four changes, all landed:

1. **The beam pipe genuinely ends** about a metre above the EAR2 floor, and the
   hall above it is open space for experiments — so it is drawn ending, in a flat
   circle with a flange, and the neutrons cross the hall in air through a bore in
   the roof slab. This **replaced a factual error**, not a styling choice.
2. **One extra frame**, so the sequence is target → collimation and the beam
   arriving in EAR2 → the beam dump → the measuring station. The station appears
   **only on the last frame**.
3. **The first collimator is named**, on the break label and in frame 2's bullet:
   7.4–8.4 m above the target, ~10 m *below* the EAR2 floor, 1 m of iron with a
   200 mm bore. It is inside the lower break, which is why the drawing alone
   invited the question.
4. A slide-copy label was **clipped** by the fix in 3 (`annotate` does not wrap);
   every line in `LABELS_SLIDE` is now ≤ ~21 characters and `make_ear2.py` says so.

~~**Board peel zoom** — drop the **pillars** from the resistive-strip band, and on
the two strip layers **suppress the vias and draw the strips themselves** as plain
horizontal / vertical lines, because as gerber artwork "it isn't immediately obvious
what is going on". ⚠️ That makes two of the four bands **schematic rather than
gerber copper**, so the figure's "real gerber copper" title has to be narrowed and
the pre-existing note about the unresolved dot-and-stub interconnect must stay.~~
**DONE 2026-08-10** — asset regenerated, title narrowed to name artwork vs
schematic per band, interconnect note kept. `index.html` untouched; if the
fig-label should change, the exact replacement text is in
[`HANDOFF_board_peel.md`](HANDOFF_board_peel.md) §4.

## Still to do

- **Review the nine-frame setup build-up.** `scenes_ntof.py` / `make_ntof.py` →
  `setup3d_1..9_*.png`, now on deck slides 16–24. It **is** in the main flow, but
  it was written by a session that crashed before reporting, so **nobody has
  reviewed the wording** and there is no handoff note for it. The renders and the
  provenance are sound (geometry imported from the simulation's own
  `plot_geometry.py`, real Geant4 neutron and pair, and slide 18's caption admits
  the pair frame is two events spliced at the vertex). Two captions asserting the
  `~10⁻⁸` radiative branch have already been removed per the no-yield decision.
  **This is the largest unreviewed block in the deck** — and at 15 minutes it has
  to collapse to about two frames anyway, so review it with the cut in hand.
- **Slide 9's two figures are superseded** — see the Resolution item above.
  `spatial_residuals.png` is hits-basis from a run never reprocessed after the
  2026-07-25 fix, and regenerating from `sat_det3` on the waveform-first basis
  gives a *better* number (≈ 0.50/0.47 mm); `angular_resolution.png` is hits-basis
  at 1.66° while the tile advertises 1.0–1.1°.
- **The bench slide's running-order flag** — waiting on whether Alexandra's P2 talk
  covers the cosmic bench. Unknown as of 2026-08-10.
- ~~**The imon systematic**~~ — **CLOSED 2026-08-10.** Measured, not looked up: the
  readback is a ~1 s averager, the charges are measurements. Two teaching backup
  slides added. One residual, which is a *different* question from the one that was
  open and which nothing in the talk depends on: the absolute nA scale of the
  readback is uncalibrated, and that needs the CAEN card model.
  **Caveat worth knowing:** run_157/158/159 are not in the local mirror and the DAQ
  and lxplus were unreachable, so this was measured on **run_79** — the same
  production setpoint, and its det C charge (97–98 nC) agrees with run_158's det C
  (97–101 nC). Worth re-running when the August tree is back on disk.

### Done, for the record

- **EAR2 beam-line scene** (2026-08-10, revised three times on 2026-08-11):
  `scenes_ear2.py` + `make_ear2.py` → `assets/img/ear2_onfig_{1_target,
  2_collimation,3_full}.png` on slides 5–7 and `ear2_beamline_{…}.png` on the new
  backup slide, EAR1 dropped, beam dump identification verified. **Split into a
  three-frame build-up per Dylan's review** (it was four for a few hours),
  together with the legibility changes he asked for (support columns removed,
  capsule floating), one **factual correction — the lower beam pipe ends ~1 m
  above the EAR2 floor** and the beam crosses the experimental space in air — and
  then the **ceiling and beam dump removed from the drawing**, which took
  `DRAWN_H` from 8.67 m to 5.71 m and made the station ~1.5× larger. Third round:
  the **line inside the hall modelled from the photograph** (polyethylene
  shielding, stepped diameters, reducer), the **wide upper pipe** to the dump
  restored as a separate section cut off by the frame, the **break's section
  marker fixed** (Dylan's "pink diagonal thing"), the **caption moved to a backup
  slide** so the pictures get the height, and a **second label layout** built
  (`ear2_onfig_*`, labels on the figure's own background) which is what the main
  slides now use. The 08-10 "pipe cut away above the floor" drawing device and its
  disclaimer are both gone; the support frame is the only as-built departure left,
  disclosed on the figure and in the backup caption, and now also visible in the
  restored **CERN photograph of the hall** beside the render (see the facility
  section above for the configuration caveat that photograph carries).
  Note `RUNNING_ORDER.md` demotes these slides to backup **as a unit** at 15
  minutes, so their value is as backup figures and as the replacement for the
  deck's weakest borrowed image.
- **Zoomed board-peel figure** (2026-08-10): built, swapped onto the chamber-design
  slide, and the caption's pitch corrected. **Second pass done the same day** per
  Dylan's review — pillars removed from the resist band, L5/L6 vias suppressed and
  the strips drawn as continuous lines, and the "real gerber copper" title narrowed
  because those two bands are now schematic.
