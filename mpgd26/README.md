# mpgd26 — 3-D setup renderings for the MPGD 2026 conference

Publication-grade 3-D views of the two setups:

* **SPS H4 beam telescope** (P2 zone) — three P2 BASKET fans between two EIC
  uRWELL references, optionally with MX17 "Detector E".
* **Saclay cosmic bench** — four M3 reference Micromegas around the P1/P2 test
  slots, triggered by a top/bottom scintillator coincidence.

Everything is driven from the run configs and the measured records, so a figure
can be re-rendered from a different angle, at a different size, in a different
theme, or with a different chamber in a slot, without redrawing anything.

Start with **`report.html`** — it carries the geometry tables, the figure set
and the list of what is drawn but not measured.

## Quick start

```bash
cd mpgd26
../.venv/bin/python make_figures.py                # the setup stills
../.venv/bin/python make_chamber.py                # the exploded chamber
../.venv/bin/python make_anim.py                   # turntables + build-ups
../.venv/bin/python make_report.py                 # rebuild report.html

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
| `chamber_exploded` | one MX17 chamber pulled apart, with a muon and its drifting ionisation (`make_chamber.py`) |

## Animations (`animations/`)

| name | what it is |
|---|---|
| `turn_sps`, `turn_bench`, `turn_bench_p2`, `turn_chamber` | turntables, 90 frames, seamless loop — MP4 + GIF |
| `build_sps` | table → uRWELL references → P2 fans → beam |
| `build_bench` | rack → trigger paddles → M3 reference → chambers → muons |

Build-ups are written **both** as a slow MP4 and as numbered stills
(`build_bench_1_rack.png` …). Drop the stills on successive slides and the
setup assembles itself as you speak — no video embedding needed.

Individual scenes, with all the switches:

```bash
../.venv/bin/python make_sps.py   --views hero,side,beam [--mx17] [--envelope] \
                                  [--no-spot] [--no-tracks] [--theme dark]
../.venv/bin/python make_bench.py --views hero,side,low --slots p2,mx17 \
                                  [--no-structure] [--no-tracks]
../.venv/bin/python make_anim.py  --only turn_bench --frames 120
```

`--slots lower,upper` takes `mx17`, `p2` or `none` for each test level.

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
  512 strips per view at 0.7785 mm pitch, 150 µm amplification gap
  (`garfield_sim/mm_config.py`).
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
* **The pad etch gap** — pads are shrunk 16 % about their own centres so the
  1280 of them don't read as one solid copper sheet. Pad centres, angles and
  count are the measured ones.

## How it is built

| file | role |
|---|---|
| `geometry.py` | every number and its provenance; `ASSUMPTIONS` |
| `meshes.py` | geometry → PyVista meshes (slabs, frames, fan prisms, pad quads, tubes, cast shadows) |
| `style.py` | palette, materials, procedural studio cubemap, light rig, render harness |
| `scenes_sps.py` | the H4 telescope |
| `scenes_bench.py` | the cosmic bench |
| `scenes_chamber.py` | one MX17 chamber, exploded |
| `annotate.py` | 3-D anchors → pixels, then the type layout |
| `make_sps.py`, `make_bench.py`, `make_chamber.py` | per-scene drivers with camera presets |
| `make_figures.py` | the deliverable still set |
| `make_anim.py` | turntables and build-up sequences |
| `make_report.py` | `report.html` |

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
