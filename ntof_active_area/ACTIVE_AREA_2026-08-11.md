# MX17 active areas — the measurement, the numbers, and what was changed

**2026-08-11.** The Geant simulation's MX17 active area was an unsourced
estimate of 38 × 34 cm. It is **39.9 × 36.0 cm** — 11 % more area — and the
4 cm that goes missing goes missing on the **beam** axis, not the tangential
one. Both `SimConfig.hh` files and the geometry documentation in both Geant
repos are updated; the scintillators are deliberately untouched.

This note is the durable record for whoever picks this up next. The
human-facing version with figures is
[`report.html`](report.html); the code is this directory.

---

## 1. Why anyone looked

`~/CLionProjects/MX17_Full_Geant/include/SimConfig.hh` carried

```c++
double mm_size_u_cm    = 38.0;   // MM active area: u [cm]
double mm_size_v_cm    = 34.0;   // MM active area: v (along beam) [cm]
```

Those two lines were the **only** dimensions in that file without a provenance
comment. Everything else — SiPM wall 50 × 50 cm with 20 bars of 2.5 cm,
plastics 20 × 30 × 2.0 cm, the LS slab at 451.2 × 450.6 mm from the STEP export
— carries a survey date and is a tape or CAD measurement good to millimetres.
The MM numbers were a guess from the start of the project and had never been
revisited.

So the scope that mattered was narrower than it first looked: **only the
chambers needed measuring.** That is worth knowing before spending a day on the
scintillators, which is the natural first instinct when the question is phrased
as "check all the detectors".

---

## 2. The answer

| | was | is | source |
|---|---|---|---|
| `mm_size_u_cm` (tangential) | 38.0 | **39.9** | full metallised strip region, no passivation |
| `mm_size_v_cm` (along beam) | 34.0 | **36.0** | 359.9 ± 1.8 mm between the passivation bands |

* **u = 399.36 mm** of metal: 512 strips at 0.78 mm pitch is 398.58 mm centre to
  centre, plus half a pitch of metal at each end.
* **v = 359.9 ± 1.8 mm**: the strip plane is passivated over a ~19 mm band at
  each end of this coordinate, leaving roughly [19, 379] mm of the 398.6 mm
  strip span.
* **Centred.** Midpoint 199.1 mm against a strip-plane centre of 199.3 mm. No
  placement offset comes with the size change.

### ⚠ The axis trap

The passivated plane is the chamber's **FEU-Y plane**, and detector-local Y is
the coordinate **along the beam**. So the number that shrinks is `v`. Putting
the small number on `u` reproduces the old error with new digits.

The mapping (`x` plane = u = tangential, `y` plane = v = along beam) is fixed in
`ntof_tracking/run79_merge_prelim.track_frame` and independently confirmed in
the beam data by the wall-segment correlation (§4).

---

## 3. How the beam measurement works

### The observable

A **paired track**: exactly one particle-like strip cluster on each plane of the
same chamber in the same event, with the two planes' charges balanced
(ratio 0.6–1.6). An MX17 avalanche splits about 50/50 between the two strip
planes, so the balance requirement is a physical statement about a real
avalanche.

**That requirement is the whole measurement.** Without it there is nothing to
see. The raw `y`-plane occupancy *outside* the chamber — beyond 380 mm — is
**higher than the chamber's own interior** on three of the four chambers, because
the outer channels are noisy:

| chamber | raw interior | raw beyond 380 mm | ratio | paired tracks beyond 380 mm |
|---|---|---|---|---|
| A | 6 459 | 4 940 | 0.8× | 2 |
| B | 4 475 | 16 313 | **3.6×** | 1 |
| C | 5 454 | 11 428 | **2.1×** | 6 |
| D | 3 658 | 53 489 | **15×** | 7 |

(median counts per strip; interior = 100–300 mm.) A raw occupancy gives the
wrong answer confidently. Paired tracks go to zero at 379 on all four.

### Why the edges are real

The beam illuminates each chamber smoothly and well past its edges — the source
is the He-3 target 235 mm away plus the whole neutron flight path — so nothing
about the illumination changes over a few millimetres. A physical edge does. So
the measurement is a search for **steps**, and the smooth 5–10× illumination
gradient across each chamber is not a problem, it is the control.

### The estimator, and why it is not a fit

The first attempt fitted an error-function turn-on to get a 50 % point. It kept
railing the width parameter at its lower bound, which is the fit saying the step
is sharper than it can resolve. It is: the boundary is a **strip boundary**, and
strips are either read or not. So the primary estimator is per-strip live/dead:

* build a **span profile** — per strip, how many selected tracks had that strip
  inside their cluster. Spans, not centroids, because a centroid can never reach
  the outermost live strip (the cluster is truncated there) but a span can;
* walk **outward from the interior**, comparing each strip to the median of the
  30 strips just *inside* it. Never a symmetric local window — that lets the
  dead region set its own reference and the edge dissolves;
* call the chamber ended when 40 consecutive strips are below a quarter of that
  reference. 40 because chamber C has a real ~20-strip interior dead stripe near
  u = 190 mm and D is worse; a shorter persistence reports that stripe as the
  edge of the chamber.

The erf fit is kept as a cross-check (`mm_edges.fit_edge`) but is not the quoted
number.

### Three guards that each caught a wrong answer

| guard | what it stops |
|---|---|
| **hot-strip mask** (`hot_strip_mask`) | One noisy channel on each plane firing together makes a "track" at a fixed (u, v). Chamber A had a 245-count spike in a 0.78 mm bin at 377 mm — right where the edge is. Real track centroids spread continuously across the pitch; a single-strip spike does not. |
| **noise floor** (`NOISE_FRAC`) | Once the reference level itself has decayed into noise, "the edge" is wherever the noise dipped. Chamber D hits this and is reported as *undetermined* rather than given a number. |
| **board-end vs undetermined** | A walk that reaches strip 511 without going dark means the plane is live all the way out — that is an *answer* (it is how we know u has no passivation), not a failure. Merging it with "ran out of contrast" would have thrown away the u result. |

---

## 4. Results, chamber by chamber

Run 79, sub-runs `stat090_0000`/`0001`, 27 files, 215 481 DREAM events.

| chamber | bench | pairs | u live | v live | June telescope, v |
|---|---|---|---|---|---|
| A | det3 | 15 056 | 1.6 – 348.7 † | **19.5 – 376.7** | 18.0 – 379.9 |
| B | det2 | 2 546 | 0.8 – 398.6 (board end) | **18.7 – 379.1** | 19.7 – 378.8 |
| C | det6 | 13 868 | 14.0 † – 397.8 | **20.3 – 379.1** | 17.9 – 380.7 |
| D | det7 | 3 622 | 199.7 † – undetermined | undetermined | 20.4 – 379.3 |

† = readout defect in this run, not a chamber boundary (§5).

**The v edges are the result.** Three chambers, both ends each, agreeing with
each other and with a completely independent June measurement to 1–2 mm. The two
methods do not even define the edge the same way — June is a 50 % efficiency
point against an external M3 telescope track, this is the outermost strip that
ever takes part in a track — so 1–2 mm is about as close as they can come.

**u has no passivation.** Chamber B is live at strip 0 *and* strip 511. That is
the whole argument, and it matches June's finding of sharp geometric edges on X.

Combined over all seven measurements (3 beam + 4 June): v span **359.9 ±
1.8 mm**, midpoint **199.1 mm**.

---

## 5. Run-79 readout defects — kept out of the geometry

These are real and useful, and they must never end up in `mm_size_*`:

* **Chamber A's X-plane connector 8 is dead** — strips 448–511, u = 349–399 mm,
  exactly one 64-channel connector. It was **alive on 18 July in run_55** (per
  connector occupancy 1.00) and dead in run_79 on 26 July, so it failed during
  the campaign. Arm A read only 87.5 % of its u width in run_79.
* **Chamber D's u plane is largely dark** in run_79 — only u ≈ 85–145 and
  200–320 mm produce tracks. D is not measurable in this run.
* **Chamber C has a genuine interior dead stripe** near u = 190 mm, ~20 strips
  wide, plus smaller ones.

If a *run-79-specific* simulation is ever wanted, these belong in a readout mask
layered on top of the geometry.

---

## 6. Scintillators — why nothing changed

The merged n_TOF ↔ DREAM sample for arm A (6 839 tracks with a full
waveform-first fit and angle) was used to point tracks at each scintillator and
ask whether it tagged the event. Two limits, both fitted rather than assumed:

* **pointing blur σ ≈ 47 mm** at the plastic plane (190 mm lever arm, angle
  scale ~0.8 of truth, and not every trigger particle is the reconstructed one);
* **accidental pedestal ~40 %** of the plateau — the DREAM trigger is an OR over
  all four arms, so an arm-A track routinely carries a tag some other particle
  produced.

| quantity | survey | fit | verdict |
|---|---|---|---|
| plastic half-length along beam | 150 mm | 253 ± 27 | not constrained |
| plastic pair half-width | 200 mm | 277 ± 5 | not constrained |
| SiPM wall half-length along beam | 250 mm | 184 ± 11 | not constrained |

The fits land on **both sides** of the survey, which is the signature of an
unconstrained parameter, not of a disagreement. A tape measure beats this by a
factor ~30. **Keep the surveyed sizes.**

What the merge *does* establish, and it is worth having:

* **the plastic pair is centred on the chamber** — L/R boundary at
  −6.8 ± 5.3 mm where the surveyed geometry (two 200 mm bars abutting on the
  pinwheel-shifted chamber centre line) puts it at 0. This is the sharpest
  statement available because it is a boundary between two live detectors, not
  an edge, so no acceptance falls off across it. It is a genuine check on the
  pinwheel shift and the plastic centring;
* **the wall segment ordering is right**, r = +0.97 across the four n_TOF
  channels. The 0.53 slope against geometry is accidental-tag dilution, and it
  matches the pedestal the acceptance fits find independently;
* **σ ≈ 47 mm** is itself a number worth having for anyone planning to use
  chamber pointing for anything.

The liquids were skipped: further out, so worse on every axis of this argument.

---

## 7. A plausible mechanism for the Y band (not measured)

`MX17_Geant/design/RESPONSE_SIM_PLAN.md` §1 records that the ESL resistive
strips contact copper bus strips **at both y-ends of the active area and nowhere
in between** (user, 2026-08-07). A dead band of the same width, at the same two
ends, on the same coordinate, is what a covered bus termination would look like
— and it would explain why the passivation is on Y and not X, which is otherwise
an odd asymmetry.

**Nobody has checked this against the gerbers.** If the bus footprint measures
~19 mm, this is closed. `MX17_Geant/design/gerbers/` is right there.

---

## 8. What was changed

### `MX17_Full_Geant` — the values

| file | change |
|---|---|
| `include/SimConfig.hh` | `mm_size_u_cm` 38.0 → **39.9**, `mm_size_v_cm` 34.0 → **36.0**, with the sources and the axis warning in a comment block |
| `scripts/plot_geometry.py` | `CFG` mirror |
| `scripts/plot_mm_layout.py` | `MM_U_HALF`, which carried its own copy; legend now quotes one decimal |
| `README.md`, `HANDOFF_FULL_SIM.md` | quoted numbers |
| `GEOMETRY_COORDINATE_CONVENTION.md` | new **§3a** — the full record: both numbers, the axis warning, the two-method comparison table, what is deliberately not modelled, and why the scintillators stayed |
| `GEOMETRY_CHANGE_CHECKLIST.md` | "Last change" banner recording which items were touched and which correctly were not |

`src/DetectorConstruction.cc` needed no edit — it reads the config
(`mmU_hf = fConfig.mm_size_u_cm * cm / 2`). Checklist items 3, 4, 5 unaffected
(no volume added, renamed or removed). Item 11 (the DAQ's
`run_config_beam.py`) correctly untouched: it stores MM *positions*, and the
active area is centred, so nothing moved.

**Verified:** rebuilds clean; a 30-event run checks **212 volumes for overlaps,
all OK**; all figures regenerated.

### `MX17_Geant` (response sim) — documentation only

This repo never had the 38 × 34 guess. It builds the **399.36 mm metallised
window, square**, with real pad structure from the gerbers, which is right for
the readout. But it does not model the Y passivation, and that is now stated in
three places rather than being silently absent:

| file | change |
|---|---|
| `README.md` | block quote after the module description |
| `design/GEOMETRY_FROM_CAD.md` | new entry under "What the CAD does *not* settle", including the bus-strip mechanism from §7 |
| `design/GEOMETRY_IMPLEMENTATION_NOTES.md` | the cross-repo comparison table's "380 × 340" corrected |
| `include/ActiveAreaFrame.hh` | comment at `activeWidth_mm` |

**Deliberately not modelled there.** That sim runs at chosen impact points, not
over a flood, so the band costs it nothing — but **any Y sweep or acceptance
number out of it must apply the band by hand.** Rebuilt clean.

---

## 9. Open items

- [ ] **Sims are stale.** The chamber is 11 % larger, so every existing
      `MX17_Full_Geant` acceptance number predates this. Re-run.
- [ ] **Check the bus-strip footprint in the gerbers** (§7). Cheap, and it
      would turn a plausible mechanism into a known one.
- [ ] **Chamber D was never measured** on beam data. If it matters, it needs a
      run where its u plane is alive — or just take June's numbers, which cover
      all four chambers.
- [ ] **Only run_79, two sub-runs.** Nothing here tests whether the band moves
      with time, though there is no mechanism by which it would.
- [ ] The v edge is measured on **strip liveness**, not efficiency. A region
      that is live but at reduced efficiency would not show as an edge here.
      June's telescope measurement carries the efficiency information and puts
      the 50 % points within 1–2 mm of these, so the two together bound it —
      but neither says the efficiency *inside* is uniform.

---

## 10. Reproducing

```bash
cd ~/PycharmProjects/nTof_x17
.venv/bin/python -m ntof_active_area.run_all     # ~2 min, rebuilds report.html
```

Inputs: `run_79/stat090_{0000,0001}` `combined_hits` (the only two sub-runs
mirrored locally) and `merged_prelim.parquet` from `RUN79_PRELIM_2026-07-30`
for the scintillator section.

`clusters.py` holds the vectorised clusterer, validated against a plain-loop
reference on a real file — identical to float summation order.

No hit times are used anywhere, so this stays inside `RECONSTRUCTION_BASIS.md`:
strip identity and charge are detection and QA quantities, and an occupancy edge
is exactly that.
