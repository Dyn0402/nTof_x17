# HANDOFF — where the capsule sits along the beam, by ray tracing

**Written 2026-09-14.** Everything below is measured on run_145's condor full
pass unless it says otherwise.

> ## Outcome — §4 was run the same day. Read this before §3–§6.
>
> Code: `capsule_y.py` (`--validate`, `--sensitivity`, `--campaign`), report
> `<out>/capsule_y/report.html`. 36 runs, 32 with ≥ 500 tracks in the fit.
>
> **The height comes from the band crossing, not the ray trace.** Campaign
> medians A **+31.1** (run-to-run s.d. 0.7), C **+35.6** (1.6), D +29.4 (noisy):
> **y_s ≈ +33 ± 3 mm**, the error being half the A–C difference and a ±10 %
> eff(v) tilt. The CAD +0.8 is wrong by ~30 mm.
>
> **The ray trace is good for one thing: it excludes a *common* v-origin
> error.** A v offset moves the band crossing 1× and the acceptance fit ~2.1×;
> if the band's +33 were all frame offset the fits would sit near +104, and
> they sit at +39 (A) and +31 (C). The SiPM wall changes nothing (§4 item 3).
> D cannot do the shape fit (§6.6 confirmed).
>
> **The ray trace is NOT a height measurement, and §5's "the difference is the
> v origin" is wrong as stated.** The acceptance fit reads the *shape* of the v
> distribution, which each chamber's own eff(v), dead strips and noisy columns
> distort; the band crossing reads the *correlation* of tan_y with v, which
> thinning tracks along v does not move. Measured by thinning the same tracks
> (`--sensitivity`, run_145): per 100 % efficiency tilt the band crossing moves
> 0.3 / 2.6 / 3.7 mm (A/C/D), the acceptance fit 32 / 38 / 65 mm. A first
> reading solved the two estimators chamber by chamber, got y_s = +24 (A) vs
> +39 (C), and called it an 11 mm A–C v misalignment with a ±8 mm error. That
> was eff(v). It was caught because the side view showed all three band
> crossings agreeing, which a real 11 mm offset (gain 1 on the band) forbids.
>
> **Rule for anything like this:** take positions from pointing, not from the
> shape of a distribution; when two estimators respond differently to a
> nuisance, split them only on what is common to all chambers.
>
> **Still open:** A vs C at 4.4 mm (s.d. 2.0) in the band crossing — a real
> relative v offset or a band-crossing bias that differs between chambers —
> and it should be settled before §5's placement constant is written. Also:
> the *median* of y at the target in the side-view histogram is set by
> mean(v) and mean(tan_y), not by pointing; the band crossings in its box are
> the measurement.

Companions:

- `ntof_athens_26/make_overhead_figure.py --projection y` and the README
  section *The side view* — the figure that raised this, and the numbers in §2.
- `ntof_athens_26/pair_vertex_imaging/y_image.py` — the campaign-wide version
  of the same measurement, and the y-plane cleaning every number here uses.
- `plastic_acceptance.py` — **the ray tracer already exists**; §4 is mostly
  about giving it a free parameter and a v projection.
- `PLAN.md` §Y, route (b) — this test, scoped but never built.
- `CLAUDE.md` → *Reconstruction basis*: geometry from the waveforms. Respected
  throughout; the y band is a fit to reconstructed slopes, not to hit times.

---

## 1 · The question

Where is the ³He capsule along the beam, **in the detector frame**? That frame
is the reference by construction: the chamber alignment is millimetric and the
capsule's placement is centimetric, so the capsule is the thing being located
and the chambers are what locate it.

Today the answer in the geometry is **not measured at all**. `run_config.json`
carries no target entry — only `target_type: 3He` — so the capsule's height is
whatever `geometry.HE3_GAS_Y/R` says, a STEP-derived polycone ported from the
Geant4 stack on 2026-07-15 (`MX17_Full_Geant/scripts/plot_geometry.py`) and
placed with its axis on +Y. Nominal gas centroid: **+0.8 mm**.

Three chambers now disagree with that by ~30 mm, and agree with each other.

---

## 2 · What is measured today

**The y band crossing is scale-free.** A track from a source at `y_s` has
`tan_y · L = s (y_s − v)`, so a robust line of `tan_y · L` against `v` crosses
zero at `y_s` whatever the angle scale `s` is. That matters because the y scale
is *not* calibrated (§6.2).

Run_145, confirmed sample (`k_arm.coincident_tracks`), x-clean **and** y-clean
(`|tan| ≥ 0.08` in both views, no noisy column in either), inside the active
area:

| chamber | tracks | y band crossing | campaign `y_image` |
|---|---:|---:|---:|
| A | 3 726 | **+30.7 ± 0.8 mm** | +30.4 |
| C | 2 481 | **+32.6 ± 2.1 mm** | +31.9 |
| D | 1 593 | **+34.1 ± 2.2 mm** | +31.9 |

Against a nominal **+0.8 mm**. Three independent geometries — two opposing
across Z, one across X — landing within ~3 mm of each other, on a run's own
data, and reproducing the campaign-wide numbers from 33 runs.

**The cleaning is what brought D in.** On uncleaned `target_y_mm`, PLAN.md
recorded A +16.2, C +14.3, **D −33.5** — D fifty millimetres from the others.
The y-plane cleaning removes that disagreement, which is itself evidence the
cleaned estimator is the trustworthy one.

**The campaign forward comparison already flagged the same thing** and could
not resolve it: `imaging_campaign`'s `y_per_run` gives run_145 offsets of
obs − model **+27.3 (A), +25.2 (C), +13.9 (D)** mm with width ratios 2.4, 2.7,
5.6, and STATUS.md's verdict was "either the polycone acceptance model or the y
reconstruction, and this does not separate them".

---

## 3 · Why the crossing is not yet the number to put in the geometry

Not because it is wrong — because it is an **ensemble estimator on a selected
sample with the acceptance left out**:

1. **The acceptance in v is not folded.** The chambers are 340 mm tall, the
   plastic bars 300 and the wall 500, and the trigger demanded a coincidence in
   both. The v acceptance is therefore *not* flat over the range the band is
   fitted on, and a robust line through an asymmetrically illuminated band is
   pulled toward the populated side — exactly the effect
   `source_imaging.symmetric_crossing` exists to remove in u, with no
   counterpart in v.
2. **The source shape is assumed symmetric and it is not.** The polycone runs
   −29.5 → +50.7 mm and tapers above +20; its area-weighted centroid is +0.8 mm
   but its *emission* profile is not the gas profile if the pairs come from the
   aluminium (see §6.1).
3. **The estimator has no error model beyond a bootstrap.** ±0.8 mm on A is a
   statistical error on a fit whose systematic — the acceptance — is the thing
   not yet included.

A forward comparison fixes all three at once: predict what each chamber should
see, with the capsule's y as the one free parameter, and fit it.

---

## 4 · The test

**Reuse `plastic_acceptance.py`. It already ray-traces this.** For each point
on a chamber's active surface it samples source points, draws the straight line
to the surface point, extends it to the plastic plane and asks whether it lands
on active scintillator. `acceptance_map()` **already computes the v coordinate
of that intersection** (`vp`, against `v_half = BSC_V_MM/2`) — the v acceptance
is in there and has simply never been projected or compared.

Four changes, in order:

1. **Give the source a y offset.** `sample_source` builds a stadium (cylinder +
   hemispherical caps) **centred on y = 0** — not the polycone, and not
   off-centre. Replace it with `geometry.HE3_GAS_Y/R` (as `y_image` and
   `vertex_image` already do) and add a `y0` argument that shifts every sampled
   point. `y0` is the parameter being fitted.
2. **Project the acceptance in v**, as `measured_profile` does in u. Note that
   one reads `<out>/fullpass/run_145`, which despite the name is the
   **allowlist** pass (STATUS.md, 2026-09-09); use `stage3_fullpass` /
   `reco_fullpass` so the sample matches everything else.
3. **Impose the SiPM wall as well as the plastics.** The module deliberately
   imposes only the plastics ("the trigger geometry's ceiling"). In v that is no
   longer good enough: the wall is ±250 mm and the plastic ±150 mm, and the
   trigger required both. The wall's surveyed position is in `run_config.json`
   (`sipm_<arm>_01..16`, 25 × 500 mm, structure-centred, 97.4 mm past the
   strips) — `make_overhead_figure.sipm_wall()` already reads it.
4. **Fit one `y0` per chamber** against the measured v profile of the
   **trigger-matched** sample, with the chamber's own v efficiency and dead
   channels divided out or masked. Report `y0` per chamber per run, and the
   chamber-to-chamber spread as the alignment number — the way A−C is in X.

**Use the matched sample, not the bystander one.** This is the opposite of
`athens_insitu`: there the trigger's acceptance was the contaminant to remove,
here it is the estimator. A bystander track carries no plastic-acceptance edge
and so carries no information about `y0`.

---

## 5 · What makes it decisive, and what it also buys

**It closes the one loophole the band crossing cannot.** A rigid offset δ in a
chamber's v coordinate moves the band crossing by exactly δ, and all four
chambers share one strip-map convention (`y_local = ±(y_p0 − 199.29)`), so
their mutual agreement does not test it. The ray trace does, because the
acceptance edges come from the **scintillators**, which are surveyed
independently in `run_config.json`. If the fitted `y0` agrees with the band
crossing, both the capsule offset and the chambers' v origin are confirmed
together; if they disagree by ~30 mm, that difference is the v origin and is
worth knowing for its own sake.

**Acceptance criteria.** The test has answered when:

- `y0` is fitted per chamber on run_145 and on ≥ 10 campaign runs;
- the three chambers agree within their spread (A and C are the ones to trust:
  D's v profile carries its dead quarter and its noisy y columns);
- the fitted `y0` is quoted **with** the systematic from the source-shape
  choice (gas polycone vs aluminium end caps, §6.1), not only the fit error;
- the answer is compared against the band crossing, and any difference is
  attributed rather than averaged away.

**Then record it in one place.** The outcome is a measured capsule offset in
the detector frame. It should live as an explicit, dated constant that the
polycone is placed by — not by editing `HE3_GAS_Y`, which is CAD and should
stay CAD. Every consumer (`y_image`, `vertex_image`, `campaign_imaging`,
`acceptance.py`, the athens figures) then reads the offset from that one place.

---

## 6 · Traps

**6.1 The source may not be the gas.** The pairs are expected from the
capsule's **aluminium** (bottom end, perhaps top), not the helium — and at
y ≈ +30 the polycone has tapered to R ≈ 4.8 mm, which is where the aluminium is
thickest. So a fitted `y0` of +30 on a *gas* source shape may be the right
number for the wrong model. Fit the gas shape and an aluminium end-cap shape
and quote the difference as the systematic; keep the source position free.

**6.2 The y angle scale is uncalibrated and bracketed.** `y_image`: the band
slope comes out below 1 (0.39–0.86 depending on tier and sample) and the focus
scan above 1, in the directions their biases predict, and **neither is
adopted**. The band crossing is scale-free so §2 survives this, but any
quantity that extrapolates a track along y — including the ray trace's
comparison sample — carries it. Run the fit at s = 1 and repeat at the bracket
ends.

**6.3 A fifth of arm A's track table rails outside the chamber in v**
(`det_a_scint`, 2026-09-10): 13.9 % of all tracks pile into one 20 mm window at
v ≈ −195 mm where the fitted y rails, and the scintillators confirm those at
6.6 % against 46 %. The `|v| ≤ 170` fiducial removes it; a v profile built
without that cut is measuring the rail.

**6.4 The trigger particle need not be the reconstructed track.** This is why
the cheap version of this test failed. Extrapolating the confirmed tracks of
run_145 out to the plastic gives v spans of −168 → +190 mm (A), −157 → +183
(C), −225 → +168 (D) against a ±150 mm bar: no readable edge, because tracks
that are not the triggering particle are free to miss it. The acceptance fit
handles this correctly (it predicts a smooth profile, not an edge); an
edge-finder does not.

**6.5 The coincidence never tested v.** `run145_target_imaging.pointing_coincidence`
checks the u coordinate only — the wall group and the plastic bar, both in u.
So v is a free observable in the confirmed sample, which is what makes this
test possible; it also means nothing upstream has ever validated v against the
scintillators.

**6.6 Chamber D's v profile is the least trustworthy.** A quarter of its x
plane is dead, a median third of its tracks per run sit in noisy columns, and
32 % of its confirmed sample fails the y-slope cut. It should be fitted, but A
and C decide.

---

## 7 · One paragraph, if you read nothing else

Three chambers say the source sits **~+31 mm** along the beam where the CAD
polycone puts +0.8, they agree to ~3 mm, the statement is scale-free, and the
capsule's position was never surveyed in the first place. The ray tracer that
would turn that into a fitted number with a real systematic **already exists**
in `plastic_acceptance.py` and already computes the v intersection; what it
needs is the polycone instead of a y-centred stadium, a free `y0`, the SiPM
wall imposed alongside the plastics, and a v projection to fit against.
