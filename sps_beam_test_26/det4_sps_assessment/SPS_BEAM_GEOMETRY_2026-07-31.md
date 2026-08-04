# The SPS H4 beam spot, read off the P2 telescope — 2026-07-31

Input for placing det4's live band (X 177–215 mm, see `DET4_SPS_ASSESSMENT.md`)
in the H4 beam. Everything here is **read** from the P2 test-beam machine
(`ssh banco_cern`, dedippcq196); nothing was written or run there beyond
read-only Python over their finished analysis products.

Sources on that machine:

- `~/P2_basket_analysis/sps_beam_analysis/` — the stage battery (20–31)
- `/local/home/banco/P2_data/TB_July2026_H4/analysis/<det>/<run>/<sub>/`
  - `23_beam_profile/beam_profile_*.csv` — their Gaussian core fits
  - `22_tag_probe_efficiency/eff_map_*.csv` — per-pad `n_tag` = the *illumination*
    (this is the map I used: 1.5 M tagged tracks per sub-run, unbiased by the
    probe plane's efficiency)
- `~/dylan/urw_analysis/out_fixed/.../summary.json` — uRWELL DUT beam spot
- `.../urw_reference/URW_TRACKING_HANDOFF_2026-07-25.md` — uRWELL↔P2 frame fit

## 1. The setup the beam was measured in

Five detectors on one rail, from `runs/run_32/run_config.json`:

| detector | z [mm] | type |
|---|---|---|
| EIC_uRWELL_front | 0 | uRWELL, 127 × 127 mm |
| P2_IN | 320 | P2 BASKET fan |
| P2_MID | 630 | P2 BASKET fan |
| P2_OUT | 940 | P2 BASKET fan |
| EIC_uRWELL_back | 1370 | uRWELL, 127 × 127 mm |

`det_center_coords` are `x = y = 0` for all five — **nominal, not surveyed**;
the run config carries no transverse geometry. Gas Ar/iso 95/5, trigger =
external scintillator coincidence into the TCM. Beam: SPS slow extraction,
43 s cycle, ~5 s spill, **~11 kHz in-spill / ~1.6 kHz run-averaged** trigger rate.

## 2. The beam is parallel — det4's z does not matter

Illumination centroid in the P2 pad frame, same sub-run, three stations
620 mm apart (`p2_mesh_drift_eff_1/scan`):

| station | z [mm] | centroid [mm] | σ major / minor [mm] |
|---|---|---|---|
| P2_IN | 320 | (414.83, 232.61) | 37.60 / 28.81 |
| P2_MID | 630 | (414.58, 232.17) | 37.76 / 28.80 |
| P2_OUT | 940 | (414.76, 232.56) | 37.75 / 28.80 |

Spread ≤ 0.3 mm over 620 mm → **< 0.5 mrad**, and the spot does not grow with z.
Consistent with the uRWELL two-point tracks (0.6–1.2 mrad). Put det4 anywhere
along the rail; the spot is the same.

## 3. Where the spot is, and how big

All in the **P2 BASKET pad frame** (the PCB's own Gerber frame, `pad_cx/pad_cy`).

- **Centre (414.7, 232.4) mm.** Stable to ±2 mm across 17 runs, 07-24 → 07-30
  (`eff_nominal_1` alone: rms 0.05 mm over 14 sub-runs). The one outlier is the
  first run, `beam_nominal_meshscan_1` on 07-24, at (423.6, 237.2) — 9 mm away,
  i.e. the beam was retuned early and has not moved since.
- **Second moments of the full illumination** (halo included):
  σx 35.7, σy 31.2, ρ = +0.238 →
  **major σ 37.8 mm at 30.2°, minor σ 28.7 mm.**
- **2σ elliptical core**, truncation-corrected: σ_major 35.0, σ_minor 16.5 at
  30.1°, holding **53 %** of the tracks (a 2D Gaussian would hold 86 %). So there
  is a narrow ridge sitting on a broad non-Gaussian halo — do not model this spot
  as a single Gaussian. The core numbers are estimator-sensitive (σ_minor moves
  16–19 mm depending on how many maps are summed); **quote the moments, use the
  core only to say "there is a ridge"**.
- Their own stage-23 fits give σx 39.5 / σy 32.4 mm. The x fit is good; the y
  fit is visibly bad (pad-row spikes, see `p2_beam_profiles_P2_MID.png`) — use
  the moments above instead.

Figures copied here: `p2_beam_spot_P2_MID.png` (per-pad map + centroid density),
`p2_beam_profiles_P2_MID.png` (projections).

### On the board — `15_sps_beam_board.py` → `sps_beam_board.png`

`15_sps_beam_board.py` redraws all this on the board itself: every one of the
1280 pads as its true rotated rectangle from the Gerber-derived map, shaded by
illumination, with the spot, the fan geometry and det4's band on top. Numbers in
`sps_beam_board.json`. It sums the 10 `eff_nominal_1` sub-run maps — **15.1 M
tagged tracks**.

Two things the board view makes obvious that the numbers alone do not:

- **Only sectors 3–6 of the 10 are instrumented** (channels 384–895). The beam is
  measured through a wedge that reaches only **97 / 109 mm** along the arc either
  side of the spot — ±3.4–3.8 σ across the beam. The core is unaffected; the far
  azimuthal tail is simply not measured, so treat the halo fraction as a lower
  limit.
- The spot is well clear of every boundary: **120 mm to the outer arc**, 364 mm to
  the inner arc, 243 / 256 mm along the arc to the two fan edges. Nothing about
  this spot is clipped by the board.

Empirical widths through the spot centre (12 mm bins, the pad pitch):
**FWHM 88 mm along the spot, 65 mm across it.**

### In the fan's own mechanical coordinates

The P2 BASKET is a fan. Apex at pad **(−33.69, −20.58)** — back-solved exactly
from the map's `radius`/`phi`, which are built on the map's `x`/`y`, *not* on
`pad_cx`/`pad_cy`; using the pad centroids instead puts the apex 11 mm off.
Metallised area (true pad corners): radius 150.7 → 635.0 mm, φ 2.30° → 57.85°,
bisector 30.074°. The beam sits at

> **radius 514.7 mm from the apex, φ = 29.40°**

i.e. **on the fan's azimuthal centre-line** (0.67° off it, 6.0 mm laterally) and
120 mm inside the outer edge. Its long axis (29–30°) is the local radial
direction — which §3b shows is the trigger scintillator's slab, not the beam.

### In the uRWELL DUT frame

`EIC_uRWELL_back`, active 127.4 × 127.25 mm, core fit at **(61.3, 50.9) mm** —
2.4 mm and 12.7 mm inside the geometric centre. Its fitted σ (19.8, 15.8 mm) are
**truncation-biased low** (a 30–38 mm spot in a 127 mm aperture); ignore them.

The uRWELL→P2 frame fit is a clean rotation of **−60.0°**, det +1, no shear, on
all three stations, so the beam's long axis lands at ~90° in the uRWELL frame —
along one of the uRWELL's own strip axes.

## 3b. In the table frame — `16_sps_beam_lab_frame.py` → `sps_beam_lab_frame.png`

Given the mounting (fan bisector vertical, apex/inner radius up, centred, and a
130 mm gap from the lowest point of the active area to the table top), the whole
map rotates into the lab. The fan apex ends up **765 mm** above the table.

**The beam is 250 mm above the table**, 6 mm off the fan centre-line. Both P2 and
det4 numbers below are heights above the mechanical table top.

The figure is drawn **from upstream, looking downstream** (beam into the page).
Both det4 connector banks are drawn, X4/X5 and Y4/Y5 picked out, so the
perspective can be checked at a glance.

The det4 margins are **not symmetric, and that fixes which way round it goes**.
`DFS3498A_det.gbr` puts all 32 mezzanine mounting holes at local 425.18–440.18 mm
— 16 on the +X edge, 16 on the +Y edge — so **both connector banks live in the
wide 50.32 mm margins**, and the −X/−Y edges are bare with 20.32 mm of margin.
Hence:

- det4 rests on a **bare −X edge**: active area starts 20.3 mm above the table.
- The **+Y edge, with the X bank, is on the right**, so local Y increases to the
  right and the active area is inset 50.3 mm from the right-hand PCB edge and
  20.3 mm from the left-hand one.
- `board_map_g_det4_rot90ccw.png` is the **mirror** of this view (it has local Y
  increasing left) — it is drawn from the other side of the board.

Nothing in the *data* ties the handedness of the P2 pad frame to det4's Gerber
frame — each is drawn from its own board's side — so P2's view side is a
convention, `--p2-mirror`, set to match. The only quantity that depends on it is
the sign of the beam's 6 mm lateral offset.

**The rotation exposed something the board view hid.** In the lab frame the
illumination is not one shape but two different ones:

- **Vertically it is a hard-edged slab, 186 → 311 mm, 125 mm tall.** The
  intensity falls by a factor ~50 within 20 mm at each edge. That is not a beam
  profile, it is an aperture — the **external trigger scintillator**. The
  measurement that it is lab-fixed and not a detector artefact: the edge heights
  are flat in x to ~1 mm across the central ±60 mm, whereas an edge at constant
  fan radius would sag 3.5 mm at x = 60 mm.
- **Horizontally it is the beam**, smooth and peaked: **σ_h = 28.6 mm**, 10–90 %
  span 73 mm, no hard edge (it is down to a couple of percent by ±80 mm, inside
  the ±97/109 mm readout aperture).

So **the σ_v = 37.8 mm of §3 is the trigger slab, not the beam**, and the earlier
reading that the spot is "elongated along the fan radius" was that slab seen in
the detector frame. This does not undo the acceptance numbers — det4 would run on
the same trigger, so the triggered flux is exactly the right thing to count — but
it does change what they mean: vertically you are slicing a flat slab, horizontally
you are slicing a peaked beam.

As a by-product, the fact that the aperture comes out *horizontal* in this frame
is an independent confirmation of the stated mounting.

## 3c. det4 on its 30 mm riser

det4 with the bands horizontal (local X vertical, local Y horizontal), resting on
the bare **PCB** edge — local X = −20.32 mm, one of the two 20.32 mm margins, not
a connector edge — on top of the **30 mm riser** available since 2026-07-31. So
height = local X + 20.32 + 30:

| | height above the table |
|---|---|
| riser | 0 → 30 mm |
| det4 PCB | 30 → 500 mm |
| active area | 50 → 450 mm |
| **live band** (local X 177–215) | **227 → 265 mm** |
| readout square (X4+X5, Y4+Y5) | 200 → 299 mm |

The readout square from those **four cables** is 149.76–248.82 mm on both axes:
**99.06 mm square, 256 channels**, and the live band crosses it horizontally.

**The riser is what makes this work.** Flat on the table the beam axis was 34 mm
above the band centre and the band caught a quarter of the triggers; on the riser
the band centre is at 246 mm and the beam axis at 250 mm — **4 mm apart, i.e.
centred** — and the readout square lands almost exactly on the trigger slab
(200–299 mm against 186–311 mm), which is why it jumps to 71 %.

| | on the 38 mm band | in the 99 mm square |
|---|---|---|
| flat on the table | 25 % | 58 % |
| **on the 30 mm riser** | **32 %** | **71 %** |
| +17 mm more (total 47, the formal optimum) | 34 % | 72 % |

Three things worth knowing before the run:

1. **The 30 mm is essentially the whole win.** The remaining 17 mm to the formal
   optimum buys ~2 points on the band and ~1 on the square. The curve is flat
   enough that anything from about +5 to +30 mm of further shim stays within a
   point of the peak, so build tolerance is not a concern either — do not spend
   effort packing det4 higher.
2. **Horizontal bands cost about a third of the acceptance.** The same 38 mm band
   turned *vertical* would collect **51 %** instead of 32 %, because vertically it
   slices the flat 125 mm trigger slab while horizontally it slices a beam with
   σ = 28.6 mm. Horizontal bands were chosen so a left–right yaw runs the track
   along a band; that is a real advantage for a micro-TPC angle scan, but it is
   being paid for in rate, and the trade should be made deliberately.
3. **Beam-weighted efficiency is better than the band average**: 83 % on the band
   (against the 78.5 % band-average of `DET4_SPS_ASSESSMENT.md`), because the beam
   sits on the band's good middle rather than being spread over its full 360 mm.
   Across the whole readout square it is 62 % — the square straddles a dead stripe
   above and below the band. See §3d.

## 3d. The same thing from det4's side — `17_det4_board_with_beam.py`

`det4_board_with_beam.png` is the inverse view: det4's own board map (within-5 mm
efficiency, 3 mm sliding kernel, the same content as
`board_map_g_det4_rot90ccw.png`) with the triggered flux contoured on top at
10 / 50 / 90 %. It runs §3b's transform backwards, so the two figures cannot
disagree.

With the mounting above and the readout square centred on the beam horizontally,
the beam centre lands at detector-local **(200, 199) mm** — essentially the board
centre, inside the live band — and the trigger slab covers local X **136–260 mm**,
spanning the band with ~40 mm to spare each side.

Note it is drawn in the **beam's-eye view** in the same orientation as
`board_map_g_det4_rot90ccw.png` — detector-local Y increasing to the LEFT, so
that with the X cards on the right X1 is at the bottom and with the Y cards on
top **Y1 is on the right**, which is what a 90 deg CCW rotation of the bench view
(X cards bottom / X1 left, Y cards right / Y1 bottom) gives. The board margin and
the X bank are mirrored about the active-area centre so the wide 50.32 mm
connector margin and the X cards land together on the right, exactly as
`14_board_map.py` does. (Before 2026-07-31 this figure and
`det4_board_with_beam.png` had the Y bank numbered the wrong way round.)
The flux contours are measured through P2's readout aperture
(±~100 mm horizontally), so the 10 % contour sits near the edge of what was
measured; 50 % and 90 % are well inside it.

## 4. Lab orientation — supplied, not measured

There is no transverse survey anywhere in the P2 data, and their handoff says so
explicitly: *"The absolute lab direction of x and y is still not determined by
this data … tying that to the lab needs the P2 frame's own orientation or a
survey."*

The frame used in §3b/§3c is therefore **given, not derived**: fan bisector
vertical, apex up, centred, 130 mm from the lowest active point to the table top.
The data does corroborate it — the trigger aperture comes out as a *horizontal*
slab with edges flat in x to ~1 mm, which only happens if the bisector really is
vertical — but the 130 mm and the "centred" are taken on trust. If either turns
out different, re-run `16_sps_beam_lab_frame.py --gap <mm>`; every height in
§3b/§3c moves with it, one for one.

## 5. What this means for det4's live band

det4's usable band is **X 177–215 mm — 38 mm wide, 360 mm long** (not 50 mm; the
widest single band, X 355–398, is the stale-pedestal connector region and is
excluded). Fraction of triggered beam particles landing on a band of a given
width, computed directly from the illumination map (no Gaussian assumed):

| band width | vertical band (along the slab) | horizontal band (across it) |
|---|---|---|
| 25 mm | 41.0 % | 29.5 % |
| **38 mm** (det4) | **54.6 %** | 40.0 % |
| 50 mm | 66.7 % | 49.8 % |
| 60 mm | 75.8 % | 58.4 % |
| 80 mm | 85.0 % | 68.5 % |
| 100 mm | 90.6 % | 81.8 % |

(These are the same numbers as §3c, seen before the lab frame explained *why* the
two orientations differ: a vertical band runs along the 125 mm trigger slab, a
horizontal one cuts across it while spanning the full peaked beam horizontally.)

Two practical consequences:

- **Orientation is worth ~15 points but is not critical.** Over the whole 180°
  the 38 mm band gives 40 → 55 %, and within ±20° of the optimum it only moves
  49 → 55 %.
- **Centring is what matters.** At the best orientation, for the 38 mm band:

  | centring error | 0 | ±5 | ±10 | ±15 | ±20 | ±25 mm |
  |---|---|---|---|---|---|---|
  | fraction on band | 54.6 % | 48–54 % | 46–50 % | 41–43 % | 36–47 % | 30–40 % |

  So aim to place the band centre within **~1 cm** of the spot centre; beyond
  ~2 cm you start losing half of what you could have had.

Rate, for planning, at the ~1.6 kHz run-averaged trigger and the §3c mounting
(measured from the illumination map, not from a Gaussian):

| | average | in spill |
|---|---|---|
| 38 mm live band, horizontal, on the riser | **520 Hz** | 3.6 kHz |
| 99 mm readout square | 1.14 kHz | 8.0 kHz |
| peak areal flux at the spot centre | 0.17 Hz/mm² | 1.2 Hz/mm² |

This does not change the recommendation in `DET4_SPS_ASSESSMENT.md` (det4 is not
a characterisation target) — it just says that *if* a slot opens, the band can be
put in the beam and will collect a third of the triggers as mounted (half, if the
bands were turned vertical).

## 6. Reproduce

The board figure and every number in it:

```bash
.venv/bin/python sps_beam_test_26/det4_sps_assessment/15_sps_beam_board.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/16_sps_beam_lab_frame.py
#   16_ takes --gap <mm> (table to lowest active point) and --shim <mm> (under det4)
.venv/bin/python sps_beam_test_26/det4_sps_assessment/17_det4_board_with_beam.py
#   17_ takes the same --gap/--shim, plus --key for the det4 cosmic run
```

`15_` and `16_` read only `sps_beam_data/`, which holds copies of the P2 group's
pad map and stage-22 illumination maps, so they re-run offline. `17_` additionally
needs the det4 cosmic run (`ray_hit_miss_list.csv` + its alignment) for the
efficiency map. To refresh the P2 inputs from their machine:

```bash
ssh banco_cern      # dedippcq196; banco_ext / banco_daplxa are unreachable from here
scp banco_cern:/local/home/banco/P2_basket_analysis/Detector_Mapping/P2_BASKET/P2_BASKET_mapping.csv \
    sps_beam_test_26/det4_sps_assessment/sps_beam_data/
scp 'banco_cern:/local/home/banco/P2_data/TB_July2026_H4/analysis/P2_MID/eff_nominal_1/scan/22_tag_probe_efficiency/eff_map_*.csv' \
    sps_beam_test_26/det4_sps_assessment/sps_beam_data/
```

`--maps` picks a different set (the `p2_mesh_drift_eff_1` maps for all three
stations are in `sps_beam_data/` too, which is how the z-independence in §2 was
checked).
