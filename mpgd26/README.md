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

```bash
cd mpgd26
../.venv/bin/python make_figures.py                # the setup stills
../.venv/bin/python make_chamber.py                # the exploded chamber
../.venv/bin/python make_microtpc.py               # micro-TPC operation
../.venv/bin/python make_x17.py --layout both --theme both   # physics-case diagrams
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
| `microtpc` | **how the chamber measures an angle**: primaries drifting to the mesh coloured by arrival time, plus the strip-time ladder and its fit (`make_microtpc.py`) |
| `microtpc_waveforms` | the same event with the **raw per-strip waveforms** instead of the ladder — the measured impulse response, sampled at 32 × 60 ns (`make_microtpc.py --right waveforms`) |
| `x17_signature` | the physics case, compact: capture → three de-excitation channels → the e⁺e⁻ opening-angle distribution (`make_x17.py`) |
| `x17_story` | the same in five beats over two rows, including **why** the boost sets the opening angle (`make_x17.py --layout story`) |
| `x17_story_capsule` | the story layout with the real Geant4 ³He vessel in beat 1 (`--layout story --capsule`) |
| `x17_signature_bare`, `x17_story_bare` | either layout with the title and caption bands cropped off (`--no-title`) |

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

Beat 1 is generic (a beam and some ³He) because early in a talk the target
hardware has not been introduced. `--capsule` swaps in the real vessel, drawn
from the `He3Gas` / `He3Cap_Al` / `He3Cap_CFRP` polycones in
`~/CLionProjects/MX17_Full_Geant/src/DetectorConstruction.cc` (sectioned from
the STEP solid), true aspect, mounted nose-first as the simulation mounts it.

## Animations (`animations/`)

| name | what it is |
|---|---|
| `turn_sps`, `turn_bench`, `turn_bench_p2`, `turn_chamber` | turntables, 270 frames, 18 s per turn, seamless loop — MP4 + GIF |
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
  512 strips per view at 0.7785 mm pitch, 150 µm amplification gap
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
| `style.py` | palette, materials, procedural studio cubemap, light rig, render harness |
| `scenes_sps.py` | the H4 telescope |
| `scenes_bench.py` | the cosmic bench |
| `scenes_chamber.py` | one MX17 chamber, exploded |
| `scenes_x17.py` | the X17 physics case; matplotlib, not PyVista — the decay kinematics live here too |
| `annotate.py` | 3-D anchors → pixels, then the type layout |
| `make_sps.py`, `make_bench.py`, `make_chamber.py`, `make_x17.py` | per-scene drivers with camera presets |
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
