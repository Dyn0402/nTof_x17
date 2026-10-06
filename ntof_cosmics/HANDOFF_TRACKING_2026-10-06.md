# HANDOFF — pooled run_149 cosmic tracks: the capsule-free angle scale, then validate on beam

> **⚠ Read §7 first (review, 2026-10-06 afternoon).** Several statements in
> §0–§2 below are wrong and are kept only as the record. The slope ratio is
> s/j = k_borrowed / k_true, **not** k_true ≈ k_borrowed × ratio. So A's tans
> read ~10 % too **large** (not "too shallow"), and step 4's `--k-scale A=1.10`
> would double the error. The ratio is not flat in angle (step 1 answered: stop,
> do not apply). The C "estimator problem" is not regression dilution. Beam
> tracks are not near-normal. Results: `results/tracking/pooled/report.html`.

**Written 2026-10-06.** Continues [`HANDOFF_TRACKING_2026-10-02.md`](HANDOFF_TRACKING_2026-10-02.md)
(read its §0, §1b and §5 first — this file does not repeat them). Entry point
for the package: [`README.md`](README.md). Live note:
<https://dylan-neff.web.cern.ch/notes/beam-off-cosmics.html> (not read in the
session that wrote this file; it does **not** yet contain anything below).

---

## 0 · Where we are

All of run_149 is now reconstructed and tracked: 87 sub-runs (`cosbounce_cos_0000`
to `_0086`), 1,936,312 triggers, condor cluster 4355060 (1,028 jobs, all exit 0),
EOS `/eos/user/d/dneff/x17/cosmics_fullpass`. Every sub-run was fetched, built
and analysed under both borrowed k (`--k-from run_147,run_150`). Commit `a6f7e5f`
holds the code and per-sub-run products; the parquet is gitignored.

New this session: **`pool_tracking.py`** (pools the per-sub-run pair tables;
rows keyed `(subrun, event_id)` because `event_id` repeats across sub-runs;
errors are a **bootstrap over sub-runs**, not over pairs).

### Pooled numbers (what the next steps start from)

| | k from run_147 | k from run_150 |
|---|---|---|
| triggers with ≥1 gated track | 10.0 % | 10.0 % |
| triggers with tracks in ≥2 arms | 0.60 % | 0.60 % |
| opposing triggers (A–C, B–D) | 5,226 (all A–C) | 5,776 |
| clean opposing (lines meet < 20 mm) | 3,233 | 3,440 |
| clean, fraction > 170° | 81.0 ± 0.7 % | 79.2 ± 0.7 % |
| all opposing, fraction > 170° | 63.3 ± 0.6 % | 58.0 ± 0.6 % |
| clean open-angle quantiles 5/16/50/84/95 % | 131 / 167 / 175.4 / 177.7 / 178.7° | 131 / 165 / 174.9 / 177.3 / 178.4° |

Capsule-free slope ratio, median of (track slope ÷ joined-line slope), clean A–C:

| arm, axis | n | k from 147 | k from 150 |
|---|---|---|---|
| A x | ~3,200 | 1.117 | 1.099 |
| A y | ~3,200 | 1.110 | 1.094 |
| C x | ~3,200 | 1.016 | 0.968 |
| C y | ~3,200 | 0.935 | 0.892 |

Bootstrap errors are ~0.002–0.004: **statistical only**. The k choice moves the
ratio by ~0.02, and for C the estimator choice moves it far more (below).

**Reading:** A reads ≈ 10 % too shallow, both axes, both k — the robust result.
C is **not** settled. The median ratio is 0.89–1.02, but the least-squares ratio
(Σsj/Σj²) is 0.80–0.87. They disagree because C has outliers (corr 0.79–0.83
against 0.91 for A) and the lsq ratio is biased low by noise in the joined-line
slope (regression dilution). The handoff-of-10-02 figure "C ~12 % too steep" came
from ~40 events and does **not** survive as a number; do not quote it.

**Two things that look like bugs and are not:**

- `k_run_147` has no pairs involving B (no AB/BC/BD): B has no calibrated k
  under run_147's table, so B tracks are not `angle_calibrated` there. run_150
  has all four arms. B–D ratios from run_150 rest on ~75 pairs and mean nothing.
- The two k choices give different "all opposing" fractions (63 vs 58 %) mainly
  because run_150 brings B in, adding opposing B–D pairs.

---

## 1 · What the analysis is, and the framing to keep

We are **testing** the beam-derived k on through-goers. Nothing has been
calibrated and nothing has gone back to beam data.

1. Cosmics are reconstructed with the beam k (borrowed; `k_arm` cannot measure
   k from through-goers because it assumes capsule origin — and **never run
   `k_arm` on cosmic runs**: it has no `--out` and silently overwrites
   `kcal/k_arm_<run>.json`).
2. A straight particle crossing A and C is one line. The joined line through the
   two chambers' track points is the reference; each chamber's own slope should
   equal the joined-line slope. `k_true ≈ k_borrowed × ratio`.
3. Pairs are selected with `sep_mm < 20` (`CLEAN_SEP_MM`). This pulls the ratio
   toward 1, so a departure from 1 is conservative. It does **not** exclude two
   particles that happen to line up, or a shower.

Dylan's framing (agreed 2026-10-06): **cosmics first, as an independent
measurement; then return to beam to validate; only then apply.** Not "calibrate
on cosmics and ship". The reasons it can fail to transfer are the point of
the steps below:

- **Angle range.** Through-goers span different angles from near-normal capsule
  tracks. The earlier run_55 bench regression failed to transfer to near-normal
  beam tracks because the angle scale there was charge-sharing driven. If k
  depends on angle, a cosmic k is right for cosmic angles and may be wrong for
  beam.
- **Run condition.** k is not stable: the 128–147 block moved A +2.3 %, C +7.8 %,
  D +7.3 %, and run_149 sits just after it. CLAUDE.md: calibrations are per
  detector **and** per run condition.
- **Coverage.** Only A–C has statistics. B and D cannot be tested until there is
  a horizontal-cosmic sample (B–D is empty in run_149 under run_147's k, and
  ~75 pairs under run_150's).

Reconstruction basis (CLAUDE.md, top): positions and angles come from the
waveforms (`wft`), never from `combined_hits` times. Everything here already
does; do not introduce a hits-based position.

---

## 2 · Next steps, in order

### Step 1 — Is the ratio flat in angle? (decides whether the rest is worth doing)

Needs only the data already on disk. In
`results/tracking/pooled/slope_samples_run_149_k<run>.parquet` (columns
`subrun, arm, axis, s, j` — `s` is the track slope, `j` the joined-line slope)
bin by `|j|` (or by joined-line zenith angle) and take the per-bin ratio with
sub-run-bootstrap errors, per arm and axis, both k.

- Flat across angle → a cosmic k can plausibly transfer to near-normal beam
  tracks. Go on to step 2.
- Trending with angle → k is angle-dependent; a single multiplicative k is the
  wrong model, and the beam-vs-cosmic difference is physics (charge sharing /
  drift response), not a calibration error. **Stop and report; do not apply.**

Note `slope_check` and `slope_samples` already drop `|j| < 0.1` (the ratio
blows up at small `j`), so the near-normal end of the cosmic distribution is
exactly what is excluded. Quantify how little overlap there is with the beam
tracks' angle range (`d_x/d_z`, `d_y/d_z` of the beam `tracks_campaign.parquet`
for arm A/C, run_147/150) before claiming anything transfers.

### Step 2 — Settle C

The ratio for C is estimator-dependent (median 0.89–1.02, lsq 0.80–0.87).

- Fit with an **errors-in-variables** / orthogonal or Deming regression, or a
  robust (Huber / Theil–Sen) slope through the origin. The joined-line slope
  has noise from both track points; ordinary least squares of `s` on `j`
  dilutes toward 0.
- Quantify the outliers: which sub-runs, which `sep_mm`, which chamber y or x
  region. Is C's scatter localised (dead strips, a noisy column, the known
  chamber-C lp-kernel difference — C is on the `lp` bundle, A/B/D on `r06`)?
- Compare against A under the **same** estimator so the two are like for like.
- Deliverable: one defensible ratio per arm and axis with a systematic
  (estimator spread + k-choice spread) next to the bootstrap error.

### Step 3 — Straightness: are the "clean" pairs one particle?

Not checked yet. For clean A–C pairs look at:

- the lateral miss `sep_mm` distribution against the multiple-scattering
  expectation (roughly how far two straight-line extrapolations should miss
  over the A–C baseline for a ~GeV muon) — a long tail suggests showers or
  two-particle coincidences;
- the open-angle tail below 165° (clean 5 % quantile is 131°, i.e. a real tail
  of clean-by-`sep` pairs that are far from collinear);
- track multiplicity in each arm on those triggers (more than one gated track
  per arm → likely not a single clean line);
- scintillator coincidence if available (the scintillator stack is calibrated
  from the MM tracks — commit b7d927f).

Report the surviving sample and rerun the slope ratio on it. If the ratio moves
with a stricter straightness cut, the first sample was contaminated.

### Step 4 — Redo the 170° efficiency with a corrected k (A first)

Only once steps 1–3 hold. To apply a cosmic k you need a track table built with
that k: `cosmic_tracks.py build` takes `--k-from <beam run>` and reads
`kcal/k_arm_<run>.json`, so add a small `--k-json` (or `--k-scale A=1.10,C=…`)
option that builds the k dict directly. **Do not write into `kcal/`** and do not
touch the campaign track table. Output to a new `results/tracking/k_<tag>/`
directory (the `_guard` in `cosmic_tracks.py` already refuses `/media`).

Then repeat the pooled fractions above. Quote the 170° efficiency under (a) beam
k_147, (b) beam k_150, (c) cosmic k. If the near-180° peak moves with k, that
shift is the systematic on any cut set from it. Only then propose a value for
`BACK_TO_BACK_DEG`.

### Step 5 — Validate on beam (the actual test)

Take the cosmic k (A, and C if step 2 settles it) and reconstruct beam tracks
with it, on data the correction was **not** fitted to: run_147, run_150, and the
beam runs neighbouring run_149 on both sides. Judge by quantities independent of
the ratio fit:

- the A–C opposing opening-angle peak in beam `back_to_back` pairs
  (`tight_coincidence.py`), same axes as the cosmic distribution;
- capsule pointing by the **pointing estimator** (the band crossing, CLAUDE.md
  "Take positions from pointing"), not a shape fit — shape fits are dominated by
  chamber efficiency and do not measure alignment;
- agreement with the existing `k_arm` capsule-based value on the same runs.

If cosmic k and beam `k_arm` agree within error → a real cross-check and the
128–147 block gets an independent test (run_133 and run_134 are cosmic runs
inside it; they are in the 41 h list below). If they disagree, the disagreement
is the result: report it, with the angle-range and run-condition candidates from
§1, and do not ship a correction.

### Step 6 — Smaller items, any order

- **Split the 10 % track-yield loss** into gate vs. geometry (what fraction is
  scintillator/coverage outside the chambers, vs. the gate cutting valid tracks).
  Last session's numbers: A y-plausibility ~55 %, D quality ~77 %.
- **The non-meeting tail**: clean-by-`sep` fails for opposing pairs whose lines
  miss by cm with opening angle 120–160°. What are they?
- **Other cosmic runs** not yet tracked: run_133, run_89, run_103, run_134
  (together with run_149, 41 of the 57.6 h). Same recipe: `make_stage2_campaign.py
  --full-pass --tags-json` (written for runs with no stage 1), then
  `cosmic_tracks.py fetch/build/analyse`, then `pool_tracking.py --run <run>`.
  run_133/134 carry their own k history (inside the 128–147 block); borrow k
  from beam runs on **both** sides of each.
- **The clock match at scale** (README "Next"): needed only for the arm-to-arm
  Δt handle, which needs a large matched two-arm sample.

---

## 3 · Mechanics you will need

All commands from the repo root; the venv is `.venv/bin/python`.

**Reproduce the pooled products** (a few minutes; nothing on EOS is needed — the
per-sub-run tracks and pairs are already under `results/tracking/k_run_*/`):

    .venv/bin/python -W ignore ntof_cosmics/pool_tracking.py

Writes `results/tracking/pooled/` : `pooled_run_149.json`,
`per_subrun_run_149_k<run>.csv`, `pairs_run_149_k<run>.parquet`,
`slope_samples_run_149_k<run>.parquet`. The parquet is gitignored; the JSON and
CSV are tracked.

**If the per-sub-run products are gone** (the loop used a scratchpad that will
not survive the session): refetch from EOS.

    S=<scratch dir off /media>
    for s in $(ssh lxplus 'ls /eos/user/d/dneff/x17/cosmics_fullpass' \
        | sed -nE 's/^run_149_(cosbounce_cos_[0-9]{4})_beam_.*/\1/p' | sort -u); do
      .venv/bin/python ntof_cosmics/cosmic_tracks.py fetch --subrun $s --tarballs $S/tarballs
      .venv/bin/python -W ignore ntof_cosmics/cosmic_tracks.py build --subrun $s
      .venv/bin/python -W ignore ntof_cosmics/cosmic_tracks.py analyse --subrun $s
    done

It ran at ~1 sub-run per minute (87 in ~90 min), 0 failures.

**Data layout:** `results/tracking/reco/run_149/<sub>/mx17_{A,B,C,D}` (reco),
`results/tracking/k_run_{147,150}/` (`tracks_*.parquet`, `pairs_*.parquet`,
`slope_check_*.csv`, `summary_*.json`).

**Conventions already in the code, keep them:**

- `ARMS = ('A','B','C','D')`, `DCA_MAX = 30 mm`, `BACK_TO_BACK_DEG = 170`,
  `CLEAN_SEP_MM = 20`, `MATCH_NS = 50` (`cosmic_tracks.py`).
- `slope_check` / `slope_samples`: opposing pairs only, AC uses normal z and
  in-plane x,y; BD uses normal x and in-plane z,y; `|j| > 0.1` kept.
- Pass `M3_MIN_NCLUS` explicitly to any `M3RefTracking(...)` call (CLAUDE.md);
  nothing here uses M3.
- The sharing kernel needs c2 < c1; the shipped bundles are `calib_bundle_r06`
  (A, B, D) and `lp` (C). Read c2 with `wft.calib.effective_c2`, never
  `hyper['c2']`. Bundles are per detector **and** per run condition.

---

## 4 · Hazards

- **The working tree is shared with other work.** At the end of the writing
  session `git status` showed staged `scint_stack` → `ntof_scint_stack/` renames,
  a modified `sept26_prelim_analysis/paths.py`, and untracked
  `ntof_scint_stack/` files — **none of them from this work.** Commit with a
  pathspec: `git commit -- ntof_cosmics`, never a bare `git commit -a`.
- **No writes to `/media/dylan/data`** (`~/x17` is a symlink to it; every
  `paths.out` default resolves there). `cosmic_tracks.py` refuses any output
  path under `/media`; new scripts must do the same (reuse `CT._guard`).
- **The noise-floor and chamber-A-connector conditions do not touch these
  runs** (beam-off, run_149 is August); but if you ever compare against beam
  runs 69–79, read "Two n_TOF conditions" in CLAUDE.md first.
- **Bootstrap errors are statistical.** For anything in a report, quote the k
  spread and the estimator spread next to them; the headline systematic here is
  not the bootstrap one.
- Do not edit the campaign track table, `kcal/`, or the `cosmics_fullpass` EOS
  output (an input shared with the lxplus package `~/cosmics_stage2_run149`).

---

## 5 · Reporting (what "done" looks like)

- `results/tracking/pooled/report.html`, generated by a
  `make_pooled_report.py` (CLAUDE.md: generate, do not hand-write; reference
  model `ntof_july_analysis/leadshield_compare/make_report.py`; relative
  `figures/` links). Lead with the verdict, then headline numbers, tables,
  figures with captions, and **what the result does not rule out** (angle range,
  run condition, B/D coverage, two-particle coincidences).
- Figures: ratio vs angle (per arm and axis, both k); the joined-line vs track
  slope scatter for A and C with the estimator lines; the clean opening-angle
  distribution with the 170° line; `sep_mm` against the scattering expectation.
  Use the `dataviz` skill, and the house style in `figures-document-not-slides`
  (ordinary ~3:2, 10.5 pt, light ink).
- Extend the live note by adding slide functions to `make_deck.py` and
  re-running the deck and publish commands in README §Reproduce (`slide-note`
  and `publish-note` skills). Only after steps 1–3, so the page does not carry a
  C number that later moves.
- Log it on the X17 board (`x17-board` skill): the 170° fraction, the A-arm
  result, and the open question "is the cosmic k transferable to beam angles?".
- Then `/wrap-up` style: a short dated handoff, commit `ntof_cosmics` only.

---

## 6 · What not to claim yet

- Not "k_C is wrong". Only "the median ratio is 0.89–1.02 and the estimator
  decides it".
- Not "the 170° cut should be X". It depends on k at the ~5-point level
  (63 vs 58 % on all opposing, 81 vs 79 % on clean).
- Not "A should be rescaled by 1.10 in the beam analysis". That needs steps 1
  and 5.
- Not anything about B and D.

---

## 7 · Review and steps 1–2, 2026-10-06 afternoon

Code: **`angle_response.py`** (sample, response, resolution, flag test, beam
test) and **`make_pooled_report.py`** → `results/tracking/pooled/report.html`
+ `figures/` (each PNG has its CSV beside it). Reproduce:

    .venv/bin/python -W ignore ntof_cosmics/angle_response.py      # ~10 s, needs the per-sub-run tables + campaign track table
    .venv/bin/python -W ignore ntof_cosmics/make_pooled_report.py

### 7a · Corrections to §0–§2

- **Direction.** s = k_b·tan_raw and j is truth, so s/j = k_b/k_true. Under a
  single-k summary (sep < 20 mm, |j| > 0.1): k_A = 1.110–1.119, k_C = 1.50 (x)
  and 1.63 (y), the same under either borrowed k to < 0.01. Beam k_arm is A
  1.22–1.24, C 1.45–1.53. "The k choice moves the ratio by ~0.02" is just
  ratio ∝ k_b. It is **not** a systematic: `raw = s/k_b` agrees to 1e-15 across
  the two tables.
- **Regression dilution (step 2) does not apply.** The lever arm is 469 mm, so
  σ_j ≈ 0.002, and least squares handles the noise in s. Least squares and the
  median differ because the ratio trends with |j|. Deming/Theil–Sen would not
  settle C. The trend does.
- **Angle overlap.** Beam pointing tracks have |tan| at 16/50/84 % of
  0.07–0.18 / 0.23–0.31 / 0.42–0.58 (A, C; x and y). That is the cosmic range,
  so the angle-range worry in §1 does not apply.

### 7b · Step 1 answered: the ratio is not flat. Stop, do not apply.

Primary selection: one gated track per arm, sep < 60 mm. The 20 mm cut and
no cut agree. raw/true per unit |tan| (sub-run bootstrap):

| | A x | A y | C x | C y |
|---|---|---|---|---|
| trend | −0.28 ± 0.02 | −0.09 ± 0.02 | −0.34 ± 0.02 | −0.23 ± 0.02 |

A gradient plus a small outward offset (~0.02 raw) describes it. That is why
k_arm's band k > track k in **every** arm and run (A 1.29/1.24, C 1.71/1.53,
D 1.87/1.73, B 2.67/2.17). So k is not purely a drift-velocity number, and
`build_tracks`' `v_insitu = v/k` carries an angle effect into depth. The drift
span of through-goers (all cross the full gap) gives A ≈ 39 µm/ns and C ≈
34 µm/ns. That is indicative only (5 % threshold, 60 ns bins, 13–21 % railed).

### 7c · Beam test (step 5, first half): the cosmic response does NOT transfer

This uses k_arm's pointing-coincident sample, rebuilt from the campaign table
(`coinc_this_arm` + charge window + lever window); it reproduces k_arm per
sub-run to ~3 %. The local `k_arm.coincident_tracks` finds only ~10 % of
condor's sample: the slim export differs, not debugged. Runs 145/147/150/152:

- The beam response is **much steeper** than the cosmic one. A x raw/true goes
  1.01 → 0.71 over |tan| 0.1 → 0.5 (cosmic 0.95 → 0.88); C x 0.89 → 0.46
  (cosmic 0.74 → 0.60). They meet only at |tan| ≈ 0.15–0.22.
- It is a shift of the whole distribution in each bin, not a contamination
  tail.
- Cosmic-corrected beam tans leave k_arm's band/track at 1.14–1.18 / 1.10–1.13
  in A and 1.06–1.18 / 0.90–1.03 in C, not 1.
- What has been ruled out:
  - A's position: the cosmic ratio varies 3–7 % over the lever range, with no
    trend.
  - Charge: ~5 % lowest-to-highest quartile, in both samples.
- Partly explained: C reads ~9–10 % lower for outward-going tracks, and every
  beam track is outward.
- Untested candidates: the particle (low-energy Compton electrons vs MIP
  muons; the beam ratio scatter is 2–3× the cosmic one), and the beam truth
  model (lever/234.6 assumes a line source on the axis at the pinwheel foot).

**This is the open question now.** By §2 step 5's own rule the disagreement is
the result. No correction ships, and neither k (cosmic or beam) is a validated
single number.

### 7d · Near normal incidence, and the `slope_reliable` flag

- Per-track σ_tan against the joined line, after the cosmic correction:
  - |tan| 0.12–0.45: 0.027–0.058
  - |tan| < 0.04: 0.24–0.41
  - |tan| 0.04–0.08: 0.09–0.30
- `tan_err` is a constant (0.022 A / 0.026 C); pulls run 1.3 in the core and
  9–16 near normal.
- `wft` `slope_reliable` (|raw tan| ≥ `TAN_MIN_SLOPE` = 0.08) is set on the
  *reconstructed* tan. The fit pushes near-normal tracks away from zero, so
  64–90 % of truly near-normal gated tracks are flagged reliable, and the median
  true |tan| of "unreliable" tracks is 0.09–0.12. `det_a_intra`'s `slope`
  selection rests on this flag.
- Straightness (step 3, partial): of 2,327 clean crossings (sep < 10 mm), 1.9 %
  (A) and 1.5 % (C) have a second gated track. All of them are ≥ 6 cm away and
  share no view, so they are second particles, not ghosts or splits.

### 7e · Revised next steps

1. **Why beam ≠ cosmic** (blocks every angle calibration):
   - Test the beam truth model: is the response curve stable under a tighter
     pointing / capsule selection, and per y-band?
   - Test the particle: Geant4 Compton electrons through the forward model, or
     beam tracks split by q_per_len / chi2.
   - C outward vs inward is already measured on cosmics.
2. **Replace the single k by a binned response** wherever opening angles
   matter: `k_arm` in |tan| bins on beam data, the same table as
   `beam_response.csv`. This is for the beam analysis and its truth caveat
   applies. Then reconsider `v_insitu = v/k`.
3. **Near-normal:** find a flag that works against truth (chi2/dof, n_strips,
   the x/y q_uend asymmetry), because `slope_reliable` does not.
4. **For the same-chamber two-track work** (worktree `nTof_x17_tt`):
   - add a near-normal case to the `pair_angle` oracle (it sits at tan 0.3);
   - build an angle-dependent σ(|tan|) error model from `resolution.csv`;
   - consider cosmic donors with joined-line truth for the intra bench. That
     needs the run_149 waveforms, which are on EOS, not local.
5. Unchanged from §2 step 6: run_133/134 (the 128–147 block), B/D (needs
   horizontal cosmics), the 170° cut (deferred until the angle response is
   settled), and the clock match at scale.

Not done: the live note (`make_deck.py`) and the X17 board were **not**
updated; both are outward-facing and wait for Dylan.
