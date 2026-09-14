# pair_vertex_imaging — what the two-track vertex does and does not tell us

## Read this first: the series, in order

Five studies, 2026-09-12 → 09-13, each building on the last. All run from
`ntof_athens_26/` with `X17_ROOT=D:/x17` on the Windows box; every product lands
in `<out>/pair_vertex/`; `chain.sh` runs the lot.

| # | module | question | answer in one line | note |
|---|---|---|---|---|
| 1 | `vertex_lab`, `diagnostics`, `make_report` | why does the pair vertex radius not image the 10 mm capsule? | it is the two legs' pointing, amplified by the crossing geometry; the y view drags the 3D point | `figures/report.html` (local) |
| 2 | `vertex_image` | is it a *blurred* image? | yes in x (A/C legs): centre −9.4 ± 0.2 mm vs −9.3 from single tracks, σ ≈ 16 mm; not in z | [pair-vertex-capsule-image](https://dylan-neff.web.cern.ch/notes/pair-vertex-capsule-image.html) |
| 3 | `z_image` | where did z go? | chamber D: noisy columns, unmeasured slopes, a dead band. D–D pairs image z; A–A/C–C do not; A–C weakly | [pair-vertex-z](https://dylan-neff.web.cern.ch/notes/pair-vertex-z.html) |
| 4 | `vertex3d` | one 3D density of every clean pair? | the raw sum is the A–D/C–D x slab; x slab + D–D z slab at equal weight cross 5.5 mm from the capsule, 10σ | [pair-vertex-3d-intra](https://dylan-neff.web.cern.ch/notes/pair-vertex-3d-intra.html) |
| 5 | `y_image` | the vertex along the beam | the three y peaks were averaging a good A/C y with a junk D y; y from the A/C leg: ≈ +30 mm, blur ≈ 28 mm | [pair-vertex-y](https://dylan-neff.web.cern.ch/notes/pair-vertex-y.html) |
| 6 | `intra_vertex` | do two tracks in one chamber share a vertex? | untestable today: a second track in the chamber makes both 3–4× worse in y, and close tracks are reconstructed as one | [pair-vertex-3d-intra](https://dylan-neff.web.cern.ch/notes/pair-vertex-3d-intra.html); handoff `sept26_prelim_analysis/HANDOFF_INTRA_TWO_TRACK_RECO.md` |

Conventions that hold across all of them:

- **Two nulls, two questions.** *Shuffled* (each track's direction swapped with
  another of the same chamber, run and quality flags, same cuts re-applied):
  "is there any source". *Event-mixed* (legs from different triggers): "is this
  more than two independent capsule tracks". Never read one for the other.
- **Cleaning** (`z_image`, `y_image`): slope measured (`*_slope_reliable`,
  |tan| ≥ 0.08) and not in a noisy column found per run, per view; scintillator
  confirmation (`coinc_this_arm`) as an optional third tier.
- **The capsule position** is the single-track band crossing
  (`imaging_campaign`), never the frame origin.
- **The source shape is not settled.** Fits in studies 2 and 5 use the He-3 gas
  volume as a reference shape; the expectation is that the pairs come from the
  capsule's aluminium (bottom end, perhaps top) and that the capsule is not at
  (0, 0, 0). Centres are free in every fit; blurs and fractions depend on the
  shape and should be refitted with an aluminium end-cap model before quoting.
- **Chamber B** is excluded everywhere (no usable angle).

---

## Study 1 — why the pair vertex does not image the capsule

**The question.** `sept26_prelim_analysis/source_imaging.py` locates the He-3
capsule to a few tenths of a millimetre and the answer repeats across 33 runs.
`pair_qa.py` then takes the *same* tracks, forms every in-trigger pair, takes
the closest approach of the two lines — and the vertex lands tens of
millimetres off the beam axis with a distribution the event-mixed null
reproduces. One of the two looks wrong.

**The answer.** Neither is. They are different measurements.

- The single-track image is an **ensemble centroid** — the zero crossing of
  median(tan) against position over millions of tracks. Its error falls as
  1/√N and reaches 0.25 mm.
- The pair vertex is a **per-event position**. It inherits the full per-track
  resolution, which on this campaign is a **44 mm median miss in chamber A,
  75 mm in C and 114 mm in D**. You cannot make a 10 mm image out of that one
  pair at a time.

Three things then make it worse, and all three are measured here:

1. **The pairing amplifies rather than averages.** The transverse crossing of
   two lines is fixed algebraically by their two miss distances and the angle
   between them, `|c| = √(e₁²+e₂²−2e₁e₂cosψ)/|sinψ|` — verified in the data to
   1e-16. So the pair vertex is a *function* of the two single-track pointings
   and carries nothing they did not already have, and `1/|sinψ|` is never
   below 1. Median gain 1.12 perpendicular, 2.26 intra, 2.86 opposing —
   exactly the order the three classes come out in.
2. **The 3D closest approach spends transverse accuracy on y.** The capsule is
   10 mm across but **80 mm long** along the beam, so there is no y image to
   find; and the y view is not a pointing measurement anyway — band slope 0.26
   (A), 0.35 (C), 0.06 (D) against 1.0 for a point source. The two legs
   disagree in y by 158 mm and the 3D fit slides both tracks along themselves
   to reconcile it, dragging the vertex off the axis. **Dropping y halves it:**
   perpendicular 46 → 26 mm, opposing 112 → 51 mm.
3. **None of it is a geometry problem or a bug.** Replace both legs'
   directions with directions pointing exactly at a random capsule point,
   keeping their measured impact points, and every class returns **7.0 mm with
   100 % inside the bore** — 7.0 mm being the median radius of a uniform 10 mm
   disc, i.e. the source itself. The crossing geometry imposes no floor.

**What to do with it.** Read `v_r` as a track-quality variable, not as imaging,
and leave the imaging claim on the band crossing. A per-event capsule image
from pairs does exist: at a **5 mm** per-leg cut the transverse crossing puts
**96.6 %** of perpendicular pairs inside the capsule against 8.0 % at the
published 30 mm cut — on 946 pairs of 29 617. It costs 96.8 % of the sample and
it is, by (1), a restatement of the leg cut.

Report: `figures/report.html`.

## The follow-up: is it a *blurred* image? (`vertex_image`)

**Half yes.** Taken one coordinate at a time, the perpendicular pairs make a
real, blurred image of the capsule **in x** and none **in z**.

- The crossing of an A–D or C–D pair is nearly square, so its **x is the A or
  C leg's pointing and its z is the D leg's**. The two coordinates are two
  chambers and are judged separately.
- Against a **no-source null** — every track keeps its impact point, its
  direction is shuffled among the tracks of its own chamber and run, and it
  goes through the identical cut and pairing — x shows a peaked excess at the
  capsule. Fitted as the projected He-3 gas ⊗ a Gaussian + the null shape
  (legs < 60 mm): **centre −9.4 ± 0.2 mm against −9.3 from single tracks,
  σ ≈ 16 mm, capsule term ~33 %**.
- Per chamber the centre is good to **~2 mm, not 0.2**: A's pairs read
  −10.2 (its band crossing −8.3), C's −7.7 (band −10.3). The offsets are
  equal and opposite and are in every run; the A–D fit's χ²/ndf ≈ 17 says the
  null's shape — non-capsule tracks with capsule-like angles — is not exact.
- z (chamber D) wants **no capsule term** (0.2 %, 2ΔlnL = 0.6); D's per-track
  pointing barely beats its own null.
- **sep** (the 3D distance between the two lines) barely separates data from
  the null for perpendicular pairs (135 against 136 mm). It measures the y
  disagreement, not a common transverse origin.
- Do **not** tighten the cut about the beam axis to sharpen the image: the
  capsule sits 9.9 mm off it, so a tight axis-centred cut drags the image
  toward the axis; a capsule-centred cut images the capsule by construction.

Note (published): `figures/vertex_image_note.html`, with an interactive 3D
view (`figures/image_3d.html`, plotly from its CDN).

```bash
python -m pair_vertex_imaging.vertex_image --jobs 8        # build + fits, ~5 min
python -m pair_vertex_imaging.vertex_image --derive-only   # fits only
python -m pair_vertex_imaging.make_image_figures
python -m pair_vertex_imaging.make_image_note
```

`vertex_image` writes `pairs_image.parquet` (data `variant == 0`, two shuffles
`1, 2`), `image_stats.csv` (model-free, per class × cut × cut centre ×
estimator), `image_fit.csv` (the 1D fits) and `image_fit_per_run.csv`. At the
30 mm axis cut its data sample is checked pair for pair against
`pairs_vertex.parquet` (`image_verify.csv`, zero difference on all six A/C/D
arm pairs).

## The second follow-up: where z went (`z_image`)

**The uncorrelated vertices in every map with z in it are chamber D.** In a
perpendicular pair z comes from the D leg, and D's gated tracks carry almost no
pointing:

- a median **third of D's tracks per run sit in noisy readout columns** (whole
  columns that make tracks at every angle), against 4 % of A's and none of C's;
- **31 % have |tan| < 0.08**, where `wft` says the drift timing carries no slope
  information and piles them at tan ≈ 0 (`x_slope_reliable`, recorded, never
  used as a cut);
- D's readout is **dead from z ≈ −57 to 0 mm**, so the D leg of a pair mostly
  contributes *where it hit D* — the x–z map is a stripe at the capsule's x
  shaped by D's live area.

Cutting all three (slope measured, not in a noisy column, own scintillators
fired) makes a D track point as well as an A track: excess over the shuffled
null within 30 mm goes 6 % → 30 %. **It does not rescue z in A–D / C–D pairs**:
the production trigger lights one arm, so a confirmed D leg rarely shares a
trigger with a good A/C track (2 282 pairs).

**z does come from two other places:**

- **D–D pairs** — both lines run along x, so their crossing fixes z. A clear
  image, σ ≈ 13–21 mm, but its centre moves −1.6 → +5.0 mm with the selection
  and always sits above the single-track −3.5; D's dead region is the first
  suspect.
- **A–C pairs**, weakly — z ≈ −10 mm, σ ≈ 33 mm with legs < 60 mm, but a broad
  hump at −2.4 mm with no leg cut. **A–A and C–C give no z at all.** A single
  A or C track's depth gives its angle (so its x), not where along z the vertex
  is.
- Shifting A and C onto their common single-track x (`--align-ac`) moves the x
  images by ∓1 mm as it should and **does not move A–C z**: the slope
  difference takes both signs, so an offset widens that image instead.

Note: `figures/z_image_note.html`.

```bash
python -m pair_vertex_imaging.z_image --jobs 8              # build + fits, ~1 min
python -m pair_vertex_imaging.z_image --jobs 8 --align-ac   # the A/C-aligned rebuild
python -m pair_vertex_imaging.make_z_figures
python -m pair_vertex_imaging.make_z_note
```

`z_image` writes `pairs_z.parquet` (per-leg flag codes `code1`/`code2`:
1 slope measured, 2 noisy column, 4 confirmed), `z_hot_columns.csv`,
`z_pointing.csv`, `z_band.npz` and `z_fits.csv`; `--align-ac` writes the same
under `_ac_aligned`. The null is shuffled within chamber × run × flag code, so
every cut's null is the same selection as its data. Its default data sample is
checked against `pairs_image.parquet` at the 30 mm cut.

## Every clean pair in one 3D density (`vertex3d`)

A pair always gives a 3D point, but each class measures one direction: A–D/C–D
x (the A/C leg), D–D z, A–A/C–C and A–C nothing above noise, and y is poor for
everyone (see `y_image` for the fix). So each class images a slab.

- Each class minus its own shuffled null, the null normalised in the transverse
  corners (|x − cx| and |z − cz| both > 35 mm).
- **Raw sum**: 84 % A–D/C–D, so it is their x slab — peak 13.5 mm from the
  capsule, no z localisation.
- **Balanced** (x slab and z slab at unit weight): peak (−10, +2) mm, 5.5 mm from
  the single-track capsule, 10.3σ in a 12 mm box against 2.2σ for the identical
  construction with shuffle 1 as data and shuffle 2 as null.
- A back-projection, not a deconvolution: right centre, cross-shaped spread.

```bash
python -m pair_vertex_imaging.vertex3d      # reads pairs_z.parquet; ~1 min
```

Writes `v3d_classes.csv`, `v3d_measure.csv`, `v3d_maps.npz`; figures
`v3d_projections`, `v3d_significance`, `v3d_profiles`, `v3d_volume.html`.

## Same-chamber pairs, and why they cannot be tested yet (`intra_vertex`)

Do two tracks in one chamber (A–A, C–C) share a vertex? Legs cleaned in both
views, pointing within 30 mm of the capsule, clones (shared x or y cluster)
removed. Three tests against **event-mixed** pairs: Δx/Δy at the capsule's
depth; whether the x view and the y view give the same crossing depth; where the
agreeing pairs sit.

- **No test separates real from mixed** (A–A Δy robust σ 152 vs 168 mm; C–C 230
  vs 231; depth correlations ≈ 0). A–A with tight pointing has a ~2σ hint.
- **The tests are blind** (`--multiplicity`): a track sharing its chamber with a
  second one is 3–4× worse in y at the capsule (A 41 → 139 mm, C 48 → 170 mm),
  1.4× in x, with 2–3× the strips and 6–10× the χ²/dof — at *every* separation
  of the two tracks, 80–400 mm included.
- **Close tracks are one track**: of 25,292 selected tracks in two-track A
  chambers, 1 has its partner within 12 mm in y (`wft.seed.GAP_THRESHOLD_MM`) and
  4 in x; C 2 (y) and 2 (x) of 44,813.
- **The damage grows with separation**: 16–24 mm apart y is 79 mm (A) / 83 mm (C)
  with ~19 strips; 80–400 mm apart 137 / 172 mm with 35 / 57 strips — the
  opposite of overlapping charge.
- In `wft/reco.py:select_tracks`, two time-coincident candidates per plane are
  paired by rank of χ² improvement, not geometry.

Handoff for the reconstruction work:
`sept26_prelim_analysis/HANDOFF_INTRA_TWO_TRACK_RECO.md`.

**Follow-up, 2026-09-14.** The handoff's truth bench (`sept26_prelim_analysis/intra_bench.py`)
separated the causes: x/y swaps, a significance floor relative to the whole plane that erases a
fainter partner, and merging under the 12 mm seed gap — while found tracks fit like single ones,
so the widened fits above are busier events, not the fit. Two opt-in `wft` fixes take same-chamber
pairs ≥ 24 mm apart from 47/39 % to 71/66 % (A/C) without moving a production track
(`wft/MULTITRACK_2026-09-14.md`). They are not in the track table yet, so the numbers in this
section are still production reconstruction.

```bash
python -m pair_vertex_imaging.intra_vertex --jobs 8          # pairs + tests, ~1 min
python -m pair_vertex_imaging.intra_vertex --multiplicity    # the quality study
python -m pair_vertex_imaging.make_v3d_intra_note            # the note (with vertex3d)
```

Writes `pairs_intra.parquet` (`kind` real / mixed / shuffled), `intra_summary.csv`,
`intra_multiplicity.csv`, `intra_twotrack_separation.csv`; figures
`intra_test1–3`, `intra_multiplicity`.

## The third follow-up: y (`y_image`)

**The three y peaks were an averaging artefact.** The vertex y was the mean of
the two legs' y at the transverse crossing; in A–D / C–D the A/C leg's y is a
single peak and the D leg's is junk, uncorrelated with it, so the mean draws
D's structure around A's peak.

- **The y plane is cleaned like x**: `y_slope_reliable` (|tan y| ≥ 0.08) and
  noisy y columns per run. Flag bits 8 and 16 on top of z_image's 1/2/4; the
  null is shuffled within all 32 strata.
- **Pair y = the A or C leg's y at the crossing**: A–D ≈ +30 mm, C–D ≈ +29 mm,
  blur ≈ 28 mm on top of the 80 mm gas. Single tracks agree per chamber, and the
  scale-free **y band crossing** gives +30…+32 mm in all three chambers. The
  nominal gas centroid is +0.8 mm — the offset is common to every chamber and is
  measured, not interpreted.
- **D's y is not usable in pairs**, even cleaned; D–D gives no y image.
- **The y angle scale is uncalibrated and bracketed**: the band slope comes out
  < 1 (background) and widens the images if applied; the focus minimum comes out
  > 1 and the width is flat there. Images use the scale as reconstructed.
- **3D**: x slab (A–D, C–D) + z slab (D–D) at equal weight, y from the A/C legs,
  is one compact blob.

Note: `figures/y_image_note.html`.

```bash
python -m pair_vertex_imaging.y_image --jobs 8        # build + fits, ~3 min
python -m pair_vertex_imaging.make_y_figures
python -m pair_vertex_imaging.make_y_note
```

`y_image` writes `pairs_y.parquet` (per-leg `py`, `ty`, `dw` so y can be
re-evaluated under any y scale), `tracks_y.parquet` (x-clean single tracks
pointing < 30 mm), `y_band_scale.csv`, `y_focus.csv`, `y_fits.csv` and
`y_map3d.npz`. The leg-y rebuild is asserted against `cross_xz` and the
single-track y against the stage-3 `target_y_mm`.

## Running it

```bash
export X17_ROOT=D:/x17                       # the data tree, on this box
python -m pair_vertex_imaging.vertex_lab   --jobs 8   # build,  ~20 s
python -m pair_vertex_imaging.diagnostics  --jobs 8   # measure, ~30 s
python -m pair_vertex_imaging.make_figures
python -m pair_vertex_imaging.make_report
```

From `ntof_athens_26/`. `vertex_lab` writes to `<out>/pair_vertex/`;
everything downstream reads from there. `--only <name,name>` restricts both
`diagnostics` and `make_figures`.

## What is built

`vertex_lab.py` rebuilds the pair table with the **same selection**
`source_imaging._track_table` uses — gated, angle-calibrated, a ceiling on each
leg's `dca_axis_mm` — so any difference found downstream is a difference of
estimator and not of sample. **At the published 30 mm cut it reproduces
`pair_qa`'s real-pair counts exactly, 0 difference on all ten arm pairs.** The
build ceiling is left loose at 60 mm so the cut can be scanned instead of
assumed.

Four quantities per pair where the published table keeps one:

| column | what it is | what it isolates |
|---|---|---|
| `v_r` | radius of the 3D closest-approach midpoint | the published quantity — a mixture of everything below |
| `v_r_xz` | the same, **in the transverse plane only** | the in-plane angles, which is exactly what the band crossing uses |
| `dy_cross` | the two legs' y at that crossing, subtracted | the y information, transverse information divided out |
| `sin_psi_xz` | sine of the transverse crossing angle | the conditioning — how much the crossing multiplies the legs' error |

## The measurements

Each writes one CSV into `<out>/pair_vertex/`.

| table | what it decides |
|---|---|
| `classes` | the observation, per arm pair, real and mixed, with a per-class `pair == 'all'` row first |
| `decomposition` | the closed form, verified; the gain distribution per class |
| `leg_scan` | tighten the leg cut 60 → 5 mm and watch the vertex follow it linearly |
| `pointing` + `pointing_hist` | single-track pointing per chamber against a **tan-shuffled null** with the source information removed |
| `scale` / `scale_focus` | what a tan rescale does to each estimator |
| `ybudget` | how much of `v_r` − `v_r_xz` is the y mismatch (answer: all of it) |
| `yband` | the pointing band in both views, per chamber |
| `floor` | the ideal-leg substitution — what perfect angles would give |
| `lift` | whether a vertex cut enriches real pairs over event-mixed ones |
| `scint_purity` | at a fixed pointing cut, does a scintillator-confirmed leg beat an unconfirmed one? |

### The null that makes section 2 a measurement

Shuffling `tan` among the tracks of one chamber keeps every marginal — the same
impact points, the same angular distribution, the same acceptance — and
destroys only the correlation between where a track landed and which way it was
going. That correlation *is* "this track came from the capsule", so the
shuffled sample is the same chamber reading the same rates with no source in
it. The difference between the two histograms is the pointing information, in
units anyone can check.

### Three things to know before reading the tables

**The event-mixed null does not mean here what it means in the opening-angle
spectrum.** Mixing decorrelates the *trigger*, not the *origin*: two tracks
from two different neutron captures in the same capsule both still came out of
that capsule, so a mixed pair has a genuine common source and a working vertex
detector would image it just as well. Real ≈ mixed is **not** evidence that the
imaging failed — it is the separate statement that the vertex carries no
coincidence information, which `lift` quantifies.

**`scale_scan` is circular and `scale_focus` is not.** The pair sample is
selected on pointing evaluated at *s* = 1, so its scale scan is biased toward
finding its minimum there. `scale_focus` runs on every gated track with no
pointing cut at all. Read that one for the angle scale and `scale` for what a
scale error does to the vertex.

**Only the `yband` y row is a measurement.** The leg cut is a pure XZ quantity
(`build_tracks.pointing` computes `dca_axis_mm` from the in-plane angle alone
and is blind to the y slope), so the x band is pinned near 1 by that cut and
says nothing. The y band is uncut and says everything. The x row is printed
anyway, as the demonstration that the cut does what is claimed.

## Two things this leaves open

**The angle scale.** The vertex is genuinely sensitive to it — ×1.33 doubles
the perpendicular vertex, 26 → 55 mm — and the band crossing is exactly
invariant, which is half the answer to why one survived the open `k` question
and the other did not. But the uncut focus scan prefers *s* ≈ 0.90 (A), ≤ 0.7
(C) and is flat for D; **none of them asks for the +33 % the arm-A scintillator
wall measures.** Two different samples: the wall runs on the
scintillator-confirmed one, this on the gated one where roughly half the tracks
are not from the target and dilute any focus. Recorded as an open question, not
as a competing number — a diluted focus scan is the weaker measurement.

**Resolution versus background, beyond the cut.** `scint_purity` says a
confirmed arm-A leg improves the A–D vertex from 26.7 to 24.0 mm and its bore
fraction from 6.8 % to 10.0 % — real, and roughly a tenth of the factor of
three that would be needed. So what survives the 30 mm cut is mostly
resolution. Separating the angle error from the source size, from scattering
and from the residual background needs a sample with an independent direction
reference, which is what a per-detector recalibration would produce.

*(The A–A and A–C rows of that table are not comparable and the report says so:
confirming an arm-A leg selects tracks pointing at arm A's own wall, which for
those two classes collapses the crossing angle — sin ψ falls from 0.36 to 0.15
on A–C — so those rows compare two different geometries.)*

## What this does not touch

The **opening angle** is a difference of two directions and does not use the
vertex at all, so none of this changes the campaign spectra. It does mean a
vertex cut cannot be used to clean them.

**Chamber B** is in the tables for completeness only — no field-shaping rings,
no uniform drift field, no usable angle. Its rows are printed so they can be
seen to be empty of information.

## Style

`mpgd26/plotstyle.py` and `sept26_prelim_analysis/report_style.py`, imported
rather than copied, so a chamber does not change colour between this and the
Athens deck.
