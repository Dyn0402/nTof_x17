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
- ~~No writes to `/media/dylan/data`~~ — **lifted by Dylan 2026-10-07**: the data
  disk may be used (check free space first; it was 69 GB free). The `_guard`
  helpers are now no-ops.
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

---

## 8 · In-situ calibration against the A–C line (2026-10-06 evening, unfinished)

Harness: **`insitu_calib.py`** (truth → cache → profile / hyper / reco / score).
Its `reco` step reproduces production bit-for-bit (60 events, Δtan = 0). The
work dir was the session scratchpad (`.../scratchpad/insitu`, with 17 GB of A/C
waveforms from 14 run_149 sub-runs). **It will not survive.** To refetch,
use `fetch.sh` logic: FEUs 03/04/07/08 decoded_root + combined_hits_root for
the 14 truth-richest sub-runs, then `truth --subs`, then `cache`. 815 events
per arm; 1/3 train.

Established:

1. **The bulk angle-scale error is the v substitution, not physics.** The bench
   fitted v together with the kernel (det3 36.6, det6 26.7). `wft_beam.make_bundle`
   keeps the kernel but swaps v for the 42.6 prior. In-situ free-fit w/tan_true
   gives A 37.5 (x) / 35.7 (y) and C 25.8 / 25.4, i.e. the bench values;
   42.6/36.6 ≈ 1.16 and 42.6/26.7 ≈ 1.60 are most of k_arm.
2. **The fit finds its χ² minimum** (within 0.2 µm/ns), but under the production
   bundle the minimum is off the truth: Δχ²(truth) is 28–220 scaled units,
   against a statistical width < 0.5 µm/ns. That is a systematic mismatch.
3. **A ref-pinned in-situ hyper refit is NOT the fix.** For A it gives
   v 39.2, Dp 0.031, τ_s 413, σ_s 105 and halves χ². Held out, x improves
   (median tan/true 1.025, σ 0.028–0.036) but y gets worse (kw_y 0.79,
   non-linear). The fitted v = 39.2 is the known χ²(v)-valley bias
   (ANALYSIS_STATE S8): use the geometric v from free fits, never the
   ref-pinned one. The C fit (kY 0.39, τ_s 30) looks degenerate; unchecked.
4. **Ruled out:**
   - zero suppression: off, full 512-channel readout;
   - template / shaping: the 600 fC vs 200 fC CSA range was suspected, but
     angle-matched strip FWHM is only 2–5 % narrower, explained by the faster v;
   - fixed per-channel gain pattern;
   - the search/optimiser.
5. **Prime remaining suspect: the dropped t0 prior.** The bench showed t0
   trades against slope, with near-degenerate t0 minima 60 ns apart; only ~35 %
   of free fits land right without the external-clock prior (T1.1). The n_TOF
   bundles drop `t0_abs`/`t0_prior_sigma`. Cosmic t0 ≈ −40 ns, so the leading
   edge is outside the window. This plausibly drives both the residual angle
   trend (implied v 40 → 37 over |tan| 0.15–0.5, 3× the bench) and the head-on
   failure.

Next, in order:

1. Measure t0 per (plane, ftst) from truth-pinned fits (profile machinery,
   production hypers, v = 37.5/35.7). Build a bundle with `t0_abs` +
   `t0_prior_sigma = 5` and the geometric v, then `reco` → `score` on the test
   split. Check the angle trend and the head-on bins.
2. If that closes, the in-situ recipe is: bench kernel, geometric v and w0/kw
   from free fits against the A–C line, in-situ t0 prior. Then the beam t0 prior
   (different trigger path) and redo the beam/cosmic comparison of §7c with it.
3. Widen `W_SCAN_HALF` (it covers only |tan| ≤ 0.56 at A's v); beam tracks
   exceed that.

---

## 9 · Head-on solved (seeder), readout ruled out, C kernel open (2026-10-06/07)

Work dir is now **durable**: `~/scratch/ntof_insitu` (17 GB waveforms, caches,
reco tables; `fetch.sh`, `subs.txt` there). New: **`seed_test.py`**, and
`insitu_calib.py` steps `t0meas`, `mkbundle --hyper`, `dtxy`, `corridor`,
`joint`, `implied`. Bench scripts `degrade.py`, `degrade_crop.py` and
`noise_inject.py` are in the work dir.

**The head-on failure was the seeder, not the fit.**
- Beam seeder minimum `MIN_STRIPS_BEAM = 5`; bench `wft.seed.MIN_STRIPS = 3`.
  At n_TOF S/N a near-normal track has only 3–4 strips over threshold.
- Same triggers (clusters in both A and C), same bundle and fit, min 5 → 3:
  - one-track A–C pairs 945 → **1,570**;
  - A x true |tan| 0.02–0.08: 34 → 205 tracks, σ_tan 0.18–0.22 → **0.03–0.06**;
  - C x 0.04–0.08: σ 0.24–0.44 → 0.06–0.08;
  - core unchanged.
- Only |tan| < 0.02 remains poor (σ ~0.15).
- The |j| distribution still dips below 0.04 (≈50 vs ≈85 per 0.02), so some
  loss remains.
- **This loss applies to beam data too**: capsule tracks have a 16 % quantile
  of |tan| 0.07–0.12.

**Ruled out, with bench M3 truth.** Bench det3 events, production fit, bench
bundle (`degrade*.py`, `noise_inject.py`). Head-on and linearity stay fine
under all of:
- S/N ÷ 3 and ÷ 8;
- cropped to 20 samples starting at bench sample 6/7/8;
- real n_TOF noise injected.

So n_TOF's 7–10× lower S/N (brightest strip 40–57σ against 300–440σ),
framing and noise character are **not** the cause. Also ruled out on n_TOF
data: fit window/seeder pad (truth-corridor windows give identical results);
the w-scan range (0.021 → 0.035: no change); Dp scan (small-angle push only).

**t0.** Per-plane t0 is unconstrained in the free fit: truth-pinned t0x − t0y
scatters by 89 ns even at equal ftst. It moves (t0, p0) together. Fitted-t0
late tracks read 4–12 % steeper, but a t0 prior from these medians would be
arbitrary. The joint x–y fit (`wft.model.fit_joint`, unused in production)
flattens A y slightly and improves core σ to ~0.024. Head-on is unchanged.

**Remaining response (seed min 3; implied v = w/tan_true, µm/ns):**

| plane | 0.10–0.15 | 0.20–0.25 | 0.35–0.45 | 0.45–0.60 |
|---|---|---|---|---|
| A x | 40.2 | 38.2 | 36.7 | 32.3 |
| A y | 38.9 | 38.3 | 37.8 | 36.1 |
| C x | 30.9 | 28.3 | 26.2 | 21.9 |
| C y | 30.1 | 27.5 | 25.2 | 24.5 |

- A y is flat; A x is mildly S-shaped.
- C is the problem chamber. It runs the old det6 lp kernel. Swapping in the
  det3 r06 kernel flattens it partly (C x 32.3 → 27.9) but its core tails stay
  10–17 % (A 4–7 %).

**Recipe so far (to validate before shipping):**
1. seeder minimum 3;
2. per-plane geometric v from free fits against the A–C line (A ≈ 38.6/37.7,
   C ≈ 28.4/27.7 with the r06 kernel), never the 42.6 prior and never the
   ref-pinned v;
3. joint x–y fit as an option to evaluate.

**Next:**
1. Run seed min 3 on a beam sub-run (run_145/147). Check junk/purity (gate
   pass rates, χ²/dof, isochronous deposits) and the near-normal capsule-track
   yield. `MIN_STRIPS_BEAM = 5` was set for beam junk ("a column, not a
   single-strip deposit").
2. C: in-situ kernel work. Fit only the kernel (template, v and diffusion
   fixed) with a free-fit-closure objective, or start from r06. Then the
   residual A x S-shape.
3. Then a campaign re-pass (condor) with the new seeder, v and bundles, and
   redo §7c's beam/cosmic comparison on it.

---

## 10 · The 3-strip seeder on beam data, and the C kernel (2026-10-07 night)

**Read first:** `sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md`. This tracking
work (T2) and the two-track separation work (T1, branch `two-track-joint-fit`)
share the seeder and the bundles.

### 10a · Beam purity of seeder min 3 (§9 next step 1)

Code: **`seed_beam_test.py`** (`reco` / `verify` / `build` / `compare` /
`scint`) and **`make_seed_beam_report.py`** →
`results/seed_beam/report.html`. Work dir `~/scratch/ntof_insitu/beamseed`.
Sample: run_145 stat090_0000, all 7 tags, every trigger. Reconstruction uses
the full pass's own saved bundle per arm, and only `MIN_STRIPS_BEAM` changes.
- `verify`: a local min-5 re-run reproduces production on A tag 000 to 99.8 %
  of fits (5/2288 x, 1/2657 y; laptop-vs-condor minimum flips).
- `build --min 0` (production reco, today's code) reproduces the stage-3 gated
  counts exactly (A 7 497, C 7 089), which is also T1's split-ab baseline.

Note: `seeds_from_hits_beam` binds `min_strips` at definition time, so patching
`MIN_STRIPS_BEAM` alone only relabels the sidecar. The test patches both.

**Results (A, C):**

| | A prod → min 3 | C prod → min 3 |
|---|---|---|
| seeded events / tag | ~3 230 → ~3 860 | |
| gated tracks | 7 497 → 12 978 | 7 089 → 10 812 |
| wall-confirmed minus accidentals | 2 957 → **4 289 (+45 %)** | 2 212 → **3 352 (+52 %)** |
| gained tracks' wall confirmation (prod) | 23.8 % (42.5 %) | 31.9 % (35.2 %) |
| near-normal confirmed excess | 74 → 264 | 14 → 71 |
| events with ≥ 2 gated tracks | 621 → 1 504 | 580 → 1 091 |

Chamber D (no near-normal truth, but the same test): gated 16 259 → 23 164,
confirmed excess 1 991 → 2 790 (+40 %), gained tracks confirm at 78 % of
production's rate. B was not run: it has no angle scale, so its tracks cannot be
extrapolated (run it with `reco --min 3 --arms B` if the counts are wanted).

- **Nothing is lost as a particle.** 952 (A) / 357 (C) production gated tracks
  have no min-3 track within 2 mm, but nearly all of them reappear: at least one
  view's fit survives in a gated min-3 track, **re-paired** with another
  partner. Only 22 (A) / 5 (C) vanish entirely. They are busy events (median 2
  candidates per plane), junk-heavy (47 % q_sum > 1e6 in A, 43 % late), and the
  re-pairing is a tie on |t0x − t0y| and on the x/y charge ratio. The min-3
  tracks shared with production confirm *better* than production (A 47.9 %
  vs 42.5 %).
- Gained tracks are later (t0 > 300 ns: 31 % vs 17 %) and less confirmed, so
  they are real but dirtier.
- Two-track events: every pair is still ≥ 24 mm apart (median ~210 mm). Min 3
  neither makes close fake pairs nor recovers close pairs.

**Verdict:** min 3 is a net gain, but it must be validated together with T1's
`xy_pairing` (the re-pairing above is exactly its problem) through the
split-ab contract, not shipped alone.

### 10b · An in-situ bundle for A that closes on cosmic truth

`WFT_BEAM_MIN_STRIPS` is now an opt-in env switch in `wft_beam` (default 5,
production unchanged). The driver passes it explicitly, so a condor job must
carry it in its environment. `seed_test.py` was updated to match.

Recipe (scripts `~/scratch/ntof_insitu/recipe_A.sh`, `recipe_A2.sh`):
production kernel, `mkbundle --v 38.0`, seeder 3, then the new
`insitu_calib.py kwmed` step. It sets kw = median w/(v·tan_true) over |tan|
0.15–0.45 on the TRAIN third, with w0 = 0. **`w0kw` (least squares) is pulled
by the tails** and over-corrected A y by 7 %. Bundle:
`~/scratch/ntof_insitu/bundles/is2_A` (kw x 1.020, y 0.993).

Held-out cosmic test, median reco/true per |tan| bin (σ = MAD of reco − true):

| | 0.08–0.15 | 0.15–0.25 | 0.25–0.35 | 0.35–0.45 | 0.45–0.60 |
|---|---|---|---|---|---|
| prod x | 0.968 σ.026 | 0.937 σ.033 | 0.900 σ.053 | 0.876 σ.065 | 0.888 σ.110 |
| is2 x | 1.057 σ.029 | 1.027 σ.026 | 0.990 σ.029 | 0.964 σ.041 | 0.970 σ.072 |
| prod y | 0.988 σ.025 | 0.903 σ.034 | 0.901 σ.052 | 0.899 σ.066 | 0.889 σ.090 |
| is2 y | 1.104 σ.033 | 1.005 σ.025 | 1.001 σ.033 | 1.004 σ.036 | 1.000 σ.043 |

y is flat at 1.00, x keeps a ±3 % S-shape, and the resolution at large angle
halves. The truth sample (§8) was selected from min-5 production tracks, so it
has almost no near-normal events; head-on is judged on `seed_test`'s sample.

### 10c · On beam the in-situ A bundle does NOT read k = 1, and the two beam truths disagree

run_145 stat090_0000 A was re-reconstructed with `is2_A` at seeder 3 (label
`is2`, built with `--k-one`). `seed_beam_test.py kbeam` measures k_arm's capsule
estimators: band/track **1.19 / 1.14** (production raw: 1.29 / 1.24). The
response still falls with angle, 1.05 at |tan| 0.10 to 0.73 at 0.50, while on
cosmics the same bundle is flat.

Chamber C with its in-situ bundle (`is2_C`: r06 det7 kernel, v 28.7, robust
kw x 1.001 / y 0.957, seeder 3) reads capsule band/track **1.19 / 1.09**
(production raw 1.77 / 1.54). **With in-situ bundles A and C agree on beam**
(1.19/1.14 and 1.19/1.09). C's large excess over A was entirely calibration;
what remains is common to both chambers. Table: `~/scratch/ntof_insitu/kbeam_AC.txt`
and `beamseed/compare/run_145/stat090_0000/kbeam_*`.

The capsule-free test, wall-edge shift against tan (`det_a_scint.edge_vs_tan`,
threshold lowered for one sub-run), says the opposite. In-situ tans are about
18 % **too large** (k_ratio 1.16–1.19). The sign was checked by rescaling the
tans: the edge fit reaches 1 at s ≈ 0.82–0.85.

**Three answers for A's true tan, in units of production raw tan:**
- the scintillator edges, campaign (2.5 M tracks): wall ε 0.334 ± 0.055 and
  plastic 0.339 ± 0.082 → true ≈ 0.67 × 1.266 = **0.85**;
- the cosmic A–C line: **1.11**;
- capsule pointing (k_arm): **1.24–1.29**.

Possible explanations, with what has been tried:
- **A depth-reference bias of the track position** (p0 shifts by c·tan when the
  fitted t0 is off, §9). Alone it reconciles one sub-run with s ≈ 1.07 and
  c ≈ 23 mm. The campaign's two levers (wall 97.4, plastic 190.6 mm) give
  m(L) = L(1 − s) + c·s → s = 0.66 ± 0.18, **c = −1 ± ~25 mm**: no support,
  but not excluded. A t0 split on one sub-run is too thin to say.
- **Regression dilution with beam-specific angular noise (current favourite).**
  The edge test bins in the measured tan, so noise pulls it toward "too large".
  The capsule band regresses the measured tan on a precise u, so it is
  unaffected by noise but flattened by non-capsule tracks, which pulls it toward
  "too small". The truth sits between, where cosmics put it. Low-energy
  electrons scatter far more than cosmic muons, which would make this
  beam-only. Against it: the campaign's `slope` and `fiducial` selections barely
  change ε, so reconstruction resolution is not the driver; scattering would
  have to be.
- **Dilution is RULED OUT, and the capsule estimator is what fails (2026-10-07,
  later that night).** The wall test can be binned in the precisely measured
  strip position instead of the measured tan:
  - Where the fired group switches across a surveyed boundary U_b, half the
    tracks cross on each side. So at that strip position u_b the median TRUE
    tan is (U_b − u_b + foot)/L, and it is compared with the median raw reco
    tan of the same tracks. No tan binning, no capsule.
  - Campaign, 31 runs, 990 k single-track single-group events, each run's tans
    de-k'd with its own stage-3 k:

    | boundary | u_b [mm] | true tan | raw reco tan | true/raw |
    |---|---|---|---|---|
    | 0\|1 | −82.6 ± 1.2 | −0.267 | −0.294 | **0.909** |
    | 1\|2 | −0.5 ± 1.2 | −0.084 | −0.081 | 1.03 (no lever) |
    | 2\|3 | +71.9 ± 0.9 | +0.200 | +0.229 | **0.874** |

  - Solving both outer boundaries with a free rigid wall offset: true/raw =
    **0.893**, offset 0.4 mm. So the beam scale from the scintillators is
    **0.89 × production raw** whether binned in tan (0.85) or in u (0.89).
    Dilution is small.
  - The same edges give the effective source distance directly, tan-free and
    foot-free: the two outer edges are 9 % farther apart than a point source at
    234.6 mm predicts, so **D_eff ≈ 330 mm**. Particles reaching A's wall are
    less divergent than radial tracks from the axis. 330/234.6 = 1.41 is
    exactly the factor between capsule k (1.27) and the wall (0.89). The
    capsule estimator (k_arm, and so the campaign's stage-3 k) assumes D = 234.6
    and inherits the whole factor.
  - Split (offset-free two-boundary combination; joined to the campaign track
    table):

    | class | n | true/raw | D_eff |
    |---|---|---|---|
    | all | 990 664 | 0.894 | 331 mm |
    | t0 ≤ 100 ns | 864 605 | 0.891 | 331 mm |
    | 100 < t0 ≤ 300 ns | 72 956 | **1.052** | **250 mm** |
    | q_per_len below / above median | 329 k each | 0.892 / 0.883 | 341 / 332 mm |

    The result does not depend on charge. The 100–300 ns t0 class reads close to
    the capsule geometry and to cosmics. Its fitted t0 is where the t0–p0 trade
    (§9) would move the position reference, so the t0 dependence is the next
    thing to chase. (t0 > 300: fit fails, too few tracks at the edges.)
  - Reproducible: `ntof_cosmics/wall_edge_scale.py` (from the scint-stack
    per-track tables; validated on A only, where it reproduces the
    det_a_scint number exactly) → `results/wall_edge_scale/`. The scint-stack
    package's own edge likelihood (`ana.fit_pointing`, λ as a fraction of
    k·tan) gives the same A value, λk = 0.703 × 1.266 = 0.89, and covers C and D:

    | arm | wall λ (pass 1–2) | stage-3 k | beam true/raw | cosmic true/raw | beam/cosmic |
    |---|---|---|---|---|---|
    | A | 0.70–0.64 | 1.27 | 0.81–0.89 | 1.11 | 0.73–0.80 |
    | C | 0.66–0.63 | 1.62 | 1.01–1.06 | 1.50 (x) | 0.67–0.71 |
    | D | 0.68–0.59 | 1.77 | 1.04–1.20 | — | — |

    **The beam/cosmic gap is systematic across chambers.** Beam tracks read
    20–33 % shallower at the wall than the cosmic calibration predicts.
  - **Ruled out as the cause of the beam/cosmic gap** (A, wall scale in
    sub-samples; campaign stack tables joined to stage 3):
    - **t0 / the t0–p0 trade:** flat at 0.88–0.89 in every bin from −400 to
      +100 ns, cosmic-like t0 (−40..0) included. The earlier "100–300 ns reads
      1.05" was a thin, unstable edge fit.
    - **partial tracks:** drift length 20–28 mm 0.895, full gap 0.901,
      railed 0.910.
    - **fit quality:** χ²/dof < 5 gives 0.880.
    - **charge:** 0.883–0.891.
    - (x_n_strips > 15 reads 1.01, but strip count rises with |tan|, so that
      cut selects on angle and the estimator is not valid there.)
    - D_eff drifts with t0 (382 → 315 mm from early to late) while the scale
      does not; not understood.
  - **What remains open is beam vs cosmics:** 0.89 (scintillators, beam) against
    1.11 (A–C line, cosmics), a 25 % difference that §7c already saw as the
    "steeper beam response". Remaining candidates: the particle (low-energy
    electrons vs muons — a Geant4 forward-model test), the cosmic truth itself
    (the A–C line uses the chambers' own p0; an A/C position bias ∝ tan would
    tilt it), or the wall/plastic lever arms. The two-lever agreement makes a
    single survey error unlikely, but both levers come from one survey.
- **Tests to run next:**
  1. ~~The edge test binned in u~~ — **done** above (0.89, dilution excluded).
  2. Why D_eff ≈ 330 mm: the same u-binned edge test per run and per t0 class
     (in-time vs late), and per charge (MIP vs low-energy). Is it a population
     not from the axis, or a property of all tracks?
  3. **(Now the decisive test.)** Run_149 cosmics that fire A's wall: the
     u-binned edge test (`wall_edge_scale.py`), plus the A–C line's own
     prediction at the wall. If cosmics read 1.11 at the wall, then beam really
     differs from cosmics, which points at the particle. If they read 0.89,
     the A–C truth or the levers are off. Needs the run_149 slim: beam-off
     triggers go on the n_TOF clock with `clock_match.py` first, then
     `slim_export` (it reads the n_TOF processing's slim ROOT).
  4. Electron scattering in Geant4 (`MX17_Full_Geant`): σ(tan) between the gap
     and the wall for the beam's electron spectrum.

### 10d · The cosmic wall test: the beam/cosmic gap is REAL (2026-10-07 afternoon)

The decisive test of §10c item 3 is done. Code: `ntof_cosmics/cosmic_wall_scale.py`
(build / ana / report). Output and `report.html`:
`/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale/` (the data disk is now allowed, see §4).

**Plumbing**
- `clock_match.py` now runs on every run_149 sub-run that n_TOF recorded: cos_0000–0034 ×
  n_TOF 224678–224687, 43 pairs (42 new + cos_0000), 88–99 % matched, 93 % overall, core residual 9–18 ns.
  - Two changes. DREAM timestamps fall back to a timestamps-only extract
    (`/media/dylan/data/x17/beam_july/dream_ts/run_149/ts_<sub>.npz`, made on lxplus with LCG_106
    uproot), so the 330 MB decoded files are not needed.
  - Sub-runs that straddle two n_TOF runs keep only the DREAM triggers within `SPAN_PAD_S` = 15 s
    of that run's bunches. cos_0000 reproduces exactly.
- n_TOF partials for 224679–687 are in `beam_july/ntof_data/`.
- A's wall channels are read in ±50 ns around the matched singles time, on triggers from ANY arm.
  The WALA dt peak is within 2 ns of 0 for A-, B-, C- and D-triggered events, so raw tof is
  effectively on a common zero for these trees.
- Yield: n_TOF records only ~16 % of the time, so 4 886 A tracks; 3 088 single x-plane tracks.
  Only 17 have A–C truth, too few for a truth-scale wall fit.

**Estimator.** The u-binned median test (`wall_edge_scale`) needs a pointing source, so it is
useless on cosmics. The edge likelihood `ntof_scint_stack.ana.fit_wall_u` (s in u + L·s·tan, set
by edge sharpness) is the right tool for cosmics. **It is NOT valid on beam**: there tan ≈ (u−u_c)/D,
so s is degenerate with the per-boundary offsets. It returns 0.52 at > 20 ms where the u-binned
test gives 0.92. Do not quote fit_wall_u numbers for beam.

**Result (arm A, L = 97.4 mm)**

| sample | estimator | true / raw |
|---|---|---|
| cosmics, x-plane single tracks (3 088) | edge likelihood, profile minimum | **1.15** (ΔNLL to 0.89 = 85) |
| same | sub-run bootstrap | 1.15 (68 % 1.12–1.19) |
| cosmics, beam `gated` selection (1 358) | edge likelihood | 1.13 ± 0.03 |
| cosmics, by \|tan\| 0–0.15 / 0.15–0.3 / 0.3–0.6 | edge likelihood | 0.96 ± 0.11 / 1.09 ± 0.02 / 1.18 ± 0.03 |
| cosmics | A–C line (§7) | 1.11 |
| beam, campaign | u-binned | 0.89 |
| beam, 10–15 / 15–20 / 20–30 / 30–45 / 45–80 ms after flash | u-binned | 0.77 / 0.87 / 0.91 / 0.93 / 0.92 |

**Verdict.** On cosmics, A's wall agrees with the A–C line (1.15 vs 1.11), so the wall survey,
the lever arms and the A–C truth are mutually consistent. **The beam reads steeper than cosmics,
for real:** about 20 % on the late-time plateau, more at 10–20 ms.

New observation: the beam scale has a **flash transient** (0.77 → 0.92 between 10 and 20 ms,
flat after). The scint-stack/wall_edge_scale campaign average (0.89) mixes it in.

**What is left as the cause (beam-specific, in the chamber):**
1. **Beam-on coherent noise.** The §9 noise injection used run_149 (beam-OFF) noise blocks. Beam CM
   wander is 10–20× beam-off (ZS study). Rerun `~/scratch/ntof_insitu/noise_inject.py` with the
   noise bank built from a run_145 sub-run (signal-free windows), against bench M3 truth. It is cheap
   and decisive for this candidate.
2. **The particle.** Low-energy electrons vs muons: a Geant4 forward-model test (MX17_Full_Geant).
3. **The 10–20 ms transient.** Flash space charge or baseline recovery. Split the beam scale by
   time on C and D too (`beam_reference()` covers A only).

Caveat: the cosmic scale rises with |tan|, and beam large-|tan| tracks sit at large |u|, so the two
samples weight angle and position differently. The plateau gap (0.92 vs 1.09 at |tan| 0.15–0.3, the
range of the beam boundary tans) is about 15 %, not 25 %.

### 10e · Is it gain / the beam environment? No — muons under beam read like muons (2026-10-07 evening)

Dylan's question: the chambers recover gain after the flash, so is the scale a function of gain?
Three tests, all chamber A.

**1. Charge.** The q_per_len medians (gated A tracks):

| population | median q_per_len |
|---|---|
| run_149 cosmics, all gated | 221–245 |
| run_149 A–C through-goers | 165–177 |
| muons in beam runs (A–C through-goers, 20–80 ms) | 115–121 |
| beam particles at the wall, 20–80 ms | 98 |
| beam particles at the wall, 10–12.5 ms | 106 |

- So the gain IS lower under beam: the same muon selection loses about 34 %.
- But it does not recover between 10 and 80 ms. Muon q is flat at 115–121. Beam-particle q falls
  slightly (106 → 98), the wrong way for gain recovery.

**2. Scale vs charge on beam.** u-binned wall scale in q_per_len quintiles:
- 20–80 ms: flat at 0.915–0.933 across a factor ~3 in q.
- 10–20 ms: no monotonic trend.
- Run by run (25 runs), corr(q, scale) = 0.4, over a q range of only 93–105.

**3. The decisive one: cosmic muons crossing A and C DURING beam runs**
(`ntof_cosmics/inbeam_through_goers.py` → `/media/dylan/data/x17/ntof_cosmics/inbeam_through_goers/`).
- Selection: one gated track each in A and C, lines within `sep`, joined line > 60 mm from the axis.
- Beam runs at 20–80 ms (0–10 ms is flash-correlated junk).
- Estimator: median(A–C line tan / raw tan) over 0.1 < |raw| < 0.6.

| sep cut | run_149 (beam off) | beam runs 20–80 ms | beam runs 40–80 ms | beam runs 10–20 ms |
|---|---|---|---|---|
| < 60 mm | 1.050 | 0.930 | 0.958 | 0.861 |
| < 20 mm | 1.080 | 1.024 ± 0.010 | 1.035 ± 0.015 | 0.975 ± 0.017 |
| < 10 mm | 1.102 | 1.061 ± 0.016 | 1.085 ± 0.018 | 1.011 ± 0.027 |

The beam-run values climb toward run_149 as the cut tightens (accidental A–C coincidences dilute
the loose cuts). At sep < 10 mm, muons under beam read **2–4 % below beam-off**, with 34 % less
gain and the full beam-on noise. Beam particles read ~20 % below (0.92 vs 1.15 at the wall).

**Verdict.**
- Gain, beam-on noise and steady space charge do NOT drive the ~20 % beam gap. A small (≤ 5 %)
  environment effect is allowed, and the 10–20 ms muons hint at one (~1.0).
- The beam/cosmic difference belongs to the **beam particles**: their kind, energy, or where and
  how they cross the chamber.
- The planned beam-on noise injection is superseded: the in-beam muons already carry that noise.
- The 0.77 → 0.92 transient in beam particles comes with D_eff 450 → 330 mm, so it looks like a
  population change (neutron energy changes with time since the flash), not recovery.

**Next:**
1. Geant4 forward model of the beam electrons (MX17_Full_Geant): the true chamber-gap tan against
   the reconstructed one, with the real material between gap and wall. Scattering inside the gap
   for ~MeV electrons is the lead.
2. Beam particles split by what they are: q_per_len tails, n_strips, chamber position (|u|
   matched to the muons' range).
3. Same in-beam muon test for C, where the beam/cosmic gap was 0.67–0.71.

### 10f · Cosmic A–C rate, activation, the late-trigger clock, and the Geant4 angle test (2026-10-07 night)

**A–C cosmic rate makes sense** (`ntof_cosmics/ac_cosmic_rate.py`).
- At EAR2 the beam is vertical (global y), so A–C through-goers are near-horizontal.
- Monte Carlo inputs: Chirkin-corrected I₀cos²θ*, I₀ = 70 m⁻²s⁻¹sr⁻¹, open sky; run_149 geometry;
  measured active area; gate |tan| < 0.6; wall∧plastic trigger on A or C.

| | expected | observed (run_149) |
|---|---|---|
| A–C through-goers, 100 % efficient chambers | **578 /h** | **189 /h** clean (sep < 60), 228 /h all single A–C pairs |
| zenith median | 74.4° | 73.6° |

The zenith histograms agree bin by bin. Implied ε_A·ε_C ≈ 0.33–0.39 (≈ 0.6 per chamber), in line with
the bench's 40–65 %. Building shielding would raise the implied efficiency somewhat.

**Activation (minutes and longer) is negligible.**
- Beam-off sub-runs starting 3–7 min after the last n_TOF pulse run at 24.7–25.4 Hz, the same as 20 h
  later (slow-control beam_class logs).
- Regressing rate on decay-weighted proton history (²⁸Al, ⁶⁶Cu, ⁴¹Ar — the gas is Ar/iso 90/10,
  ⁵⁶Mn, ²⁴Na) gives ≤ 0.2 Hz, against ≈ 110–600 Hz of late triggers with tracks during beam.
- Note: Geant4 DOES include RadioactiveDecay (²⁸Al shows up), but every analysis cuts t < 100 ms.

**The late-trigger clock** (`ntof_cosmics/late_trigger_clock.py`).
- At ≥ 30 ms the trigger rate is a single exponential, T½ = 23.5 ± 0.8 ms (C, χ² 65/50) and
  24.1 ± 1.3 ms (A).
- Direct beam captures in a thin 1/v absorber would fall ~t⁻⁴, a local T½ of 7–12 ms. The evaluated
  EAR2 flux has ~nothing below 2.5 meV, i.e. after ~28 ms.
- ¹²B (20.2 ms, ¹²C(n,p) by flash neutrons) is disfavoured as the main component: Δχ² = 125 on C with
  T fixed. No common isotope sits at 23.5 ms.
- Best reading: the die-away of thermalised neutrons in the EAR2 hall (τ ≈ 34 ms; ¹⁴N capture in air
  alone gives ~60 ms, leakage shortens it). The late triggers are then captures of AMBIENT neutrons
  all around the setup, not beam captures in the capsule. That is consistent with D_eff ≈ 330 mm.
  **This population is absent from the Geant4 beam campaign.**

**Geant4 angle test (submitted 2026-10-07, condor clusters 4402864 / 4402865).**
- Code: `ntof_cosmics/g4_angle/` (`reduce_gap_wall.py`, `condor/`). Lxplus job dir
  `~/condor/mx17_angle_scale/`; output `/eos/experiment/ntof/data/x17/full_sim/angle_scale/{single,neutrons_nose}/`.
- Truth-level "reconstruction": an edep-weighted line u(w) through all DriftGas steps of the arm.
  It is compared with where the dominant gap track first hits the SiPM wall (mesh → wall lever 97.4 mm,
  the same as data).
- (a) single e⁻ 1/2/3/5/8 MeV and μ⁻ 1 GeV from the capsule centre into arm A (sim arm 2),
  tan_u 0–0.5, 20k each;
- (b) the 100 files of `neutrons_thermal_trig_2cm_nose` (beam captures).
- Built with the lxplus checkout at 3d97437 (the build that produced the nose campaign).

### 10g · Geant4 result: the beam/cosmic wall gap is electron scattering, not reconstruction (2026-10-07 night)

`ntof_cosmics/g4_angle/analyze.py` → `/media/dylan/data/x17/ntof_cosmics/g4_angle/report.html`
(inputs pulled from EOS `full_sim/angle_scale/`).

**Method.** The data's own outer-pair u-binned wall estimator is applied to Geant4 tracks.
- The reconstruction is replaced by its ideal: an edep-weighted line through the true DriftGas
  ionisation.
- Virtual group boundaries are placed at u = ±100 mm on the wall plane (sim mesh → wall lever 97.4 mm,
  as in data).

**Beam-capture population** (`neutrons_thermal_trig_2cm_nose`): 45 626 gap tracks reaching the wall;
91 % e⁻, KE in the gap 2.0 / 3.1 / 4.6 MeV quartiles.

| sample | true / raw (ideal reco) | D_eff |
|---|---|---|
| A (sim arm 2) | **0.60** | 405 mm |
| C (sim arm 3) | **0.59** | 411 mm |
| dominant-track line only | 0.59 | 410 mm |
| KE > 4 MeV | 0.92 | 260 mm |
| KE 2–4 MeV | 0.63 | 385 mm |
| KE < 2 MeV | ≈ 0 (no correlation) | — |
| μ 1 GeV (single) | 1.000 | — |

**Verdict.**
- With a PERFECT reconstruction, the wall estimator reads well below 1 for few-MeV electrons and
  exactly 1 for muons, and it depends strongly on energy.
- The data's beam/muon ratio (~0.80: 0.89–0.92 against 1.10–1.15) sits inside the simulated range.
  The beam/cosmic gap is therefore what electron scattering between the gap and the wall (gas, mesh,
  PCB, air, SiPM container) does to this estimator.
- **The wall cannot be used as angle truth for beam electrons without a forward model.** The
  cosmic in-situ scale (A–C line, confirmed by cosmics at the wall, 1.11–1.15) stands as the
  reconstruction's scale, and §10e showed the beam environment does not move it.

**Single particles** (capsule centre → arm A, fixed gun angles):
- The gap fit against the gun direction has slope 0.80 at 5 MeV and 0.90 at 8 MeV (all full-gap
  tracks).
- Per-angle wall/gap ratios carry wall-edge truncation: fixed guns aim near the wall's outer edge at
  large angle. Read them for trend only.

**Open / next.**
1. The data number (0.92 vs sim 0.59) depends on the real population's energy mix and on how the real
   fit weights scattered charge. The late data population is ambient hall-neutron captures (§10f),
   absent from the sim. A digitised forward model through `wft` would close it, but it is not needed
   for the conclusion.
2. **X17 consequence to check:** few-MeV electrons' gap angle is compressed relative to their emission
   direction (~10 % at 8 MeV in the ideal fit). The pair/opening-angle simulation should include this,
   if it does not already go through the same physics.
3. The early 10–20 ms transient (0.77) is still unexplained. It is now best read as a population/energy
   change, consistent with this energy dependence.


## 11 · Next steps after 2026-10-07 (in order)

The beam angle question is closed. Beam/cosmic wall gap = electron scattering (§10g), not the
reconstruction and not the beam environment (§10e).

1. **Adopt the cosmic in-situ calibration for beam angles.**
   - Bundles: `is2_A` (v 38, robust kw) and `is2_C` (det7 r06 kernel, v 28.7), seeder min 3.
   - Inventory what the campaign applies now (stage-3 `k_arm`, the capsule-pointing k) and every
     consumer (opening angle, `tight_coincidence` 170° cut, `det_a_intra`'s `slope_reliable`, T1's F).
   - Make the switch a re-pass with an explicit version tag, not an overwrite.
   - Ask Dylan before writing campaign products.
2. **X17 opening angle.** Single-electron Geant4 (ideal fit): median gap tan / gun tan is 0.73 at 5 MeV
   and 0.91 at 8 MeV, before any reconstruction.
   - Find whether the opening-angle templates (IPC/X17 pairs) come from Geant4 hits through the reco
     (then included) or from truth directions (then missing).
   - X17 electrons are ~8–10 MeV, so expect a few-% compression.
3. **Same-chamber pairs:** the combined split-ab (seeder min 3 + T1 `xy_pairing`), then T1's F on the
   in-situ bundles; then the campaign re-pass (OCTOBER_2026 O4).
4. **The late population.** Triggers after ~30 ms follow one 23.5 ms exponential (A 24.1, C 23.5):
   ambient thermal neutrons in the hall, not beam captures.
   - Add an ambient-neutron mode to MX17_Full_Geant: capture vertices in the arms' Al frames, PCB and
     plastics, time ∝ exp(−t/34 ms).
   - Then: the late background and its share of the wall maps, and the data's 0.92 against simulation.
5. **Optional closures:**
   - (a) `wft`-digitise Geant4 DriftGas hits and reconstruct, to replace the ideal fit;
   - (b) a Geant4 flash run (> 13.6 MeV n) with RadioactiveDecay kept to 100 ms, to bound ¹²B (up to
     ~20–30 % of the late clock is not excluded);
   - (c) the in-beam muon test and the cosmic wall test on C (and D);
   - (d) the 10–20 ms dip (0.77): split by wall-group population/charge in time;
   - (e) the clock match for run_103 (4.3 h of n_TOF recording).


## 12 · Capsule estimators on Geant4 (2026-10-07, late) — **conclusion SUPERSEDED by §13**

> §13 digitised the same Geant4 tracks through the production reco and applied k_arm's *reco*
> charge window. Most of the falling response below is the estimator on this population, not the
> reconstruction. The reco follows the electrons' ideal line as well as it follows muons. The
> comparison here applied the charge window to G4 edep, not to reco q_sum, and had no failed fits.
> The tables below are correct; the "Reading" is not.

`ntof_cosmics/g4_angle/capsule_estimator.py [--data]` → `/media/dylan/data/x17/ntof_cosmics/g4_angle/capsule_estimator.csv`.

**Why.** §10g closed the *wall* gap as electron scattering, but the wall estimator is population-dominated
(ideal 0.59). Before adopting `is2` for beam (§11 step 1), I checked k_arm's own capsule estimators on the
same Geant4 beam-capture population (`neutrons_thermal_trig_2cm_nose`, ideal edep-weighted gap line).
- Sim geometry: 1 GeV μ guns from the capsule give u_mesh = 16.3 + 234.6·tan. That is the data's D_PERP; foot 16.3 mm.
- The same lever window (30–130 mm) and the same charge window (25–75 %) as k_arm.

**Result.** With an ideal reconstruction the capsule estimators are unbiased on beam electrons:

| | band | track | response 0.10 → 0.40–0.50 (median tan / tan_expected) |
|---|---|---|---|
| G4 ideal A, reaches wall | 1.02 | 0.97 | 1.05 → 0.98 (flat) |
| G4 ideal C, reaches wall | 1.02 | 0.97 | 1.06 → 0.97 (flat) |
| G4 ideal, full gap, no wall requirement | 1.03–1.04 | 0.88–0.89 | flat (0.95–1.09) |
| G4 ideal, KE > 2 MeV | 1.01 | 0.98–1.00 | flat |
| data run_145 `is2_A` (cosmic-closing bundle) | 1.19 | 1.14 | 1.04 → 0.73 |
| data run_145 `is2_C` | 1.19 | 1.09 | 1.07 → 0.71 |
| data campaign production raw A (> 10 ms) | 1.26 | 1.23 | 1.00 → 0.71 |
| data campaign production raw C | 1.62 | 1.47 | 0.93 → 0.56 |

- The data response falls by ~30 % between |tan| 0.10 and 0.45.
- It falls the same at 10–20, 30–60 and > 60 ms, so a late non-capsule population does not drive it. It
  also falls the same in every split: χ² above/below median, q/len, full gap, railed, slope_reliable.
- The simulated population doesn't make it fall even with no wall selection at all, so it isn't dilution
  by the physical population either.
- On cosmics the same `is2_A` bundle is flat to ±3 % against the A–C line (§10b).

**Reading.**
- The reconstruction compresses beam-electron angles increasingly with angle; on muons it doesn't.
- The cosmic in-situ scale is the right scale for *muons*, and for beam near |tan| ≈ 0.1. At |tan| 0.3–0.45
  it under-reads beam electrons by 20–27 %.
- §10g still stands for the wall (scattering after the gap). But "the cosmic in-situ scale stands for beam"
  is **not** established, and §11 step 1 (adopt `is2` for beam) should not go ahead as a flat scale.

**Not ruled out:**
1. A data population absent from the sim, flat in time, e.g. electrons from captures in the arm structure,
   not the capsule. The fall would then be dilution after all.
2. Sim/data mismatch in the source: an extended capsule-Al source (D_eff ≈ 330 mm at the wall, §10c) vs
   the point-ish sim.

**What would close it:**
- the `wft`-digitised Geant4 tracks through the real reconstruction (§11 5a) — now the decisive test,
  not optional;
- the response vs angle by the strip-cluster width at fixed angle (scattered or curled electron segments).


## 13 · Digitised Geant4 through the production reco: the reco does not compress electrons (2026-10-07, night)

Package `ntof_cosmics/g4_digi/` (outputs `/media/dylan/data/x17/ntof_cosmics/g4_digi/`).

**What the digitiser does** (`digitise.py`, docstring has the detail):
- DriftGas steps become Poisson electrons. Each drifts with the bundle's own diffusion pair and gets a
  Polya gain.
- Electrons land on the nearest strip of the run's map. The bundle's impulse template and resistive
  kernel are applied (wft.model's own pieces), then the per-channel gain.
- The signal is added to the raw ADC of a real quiet run_145 trigger on the same FEUs, clipped at the
  rail, then gets FeuReader's exact pedestal and 64-channel common-mode subtraction.
- Hits are emulated (5σ, σ from the real hit finder's amplitude/significance).
- Then the unchanged production path: `seeds_from_hits_beam` (min 3) → `extract_window` →
  `wft.reco._worker_fit`.

The response model is the fit model. What the test isolates is what the model lacks:
- non-straight ionisation;
- fluctuations;
- real noise and CNS;
- seeding and window truncation.

It does not test model-vs-chamber mismatch; that part is calibrated on cosmics.

**Inputs:**
- Step files from `extract_steps.py`, on EOS under `full_sim/angle_scale/steps_nose/` (100 files) and
  `steps_single/`.
- Charge scale set so the reco median x_q_sum matches the data's (A 1553 → 11.8 ADC/e; C 1585 → 10.8).
  The y/x split comes from data (A 0.88, C 0.69).
- t0 is sampled from the data's is2 fitted t0.
- Sample: the (event, arm) groups with a full-gap track whose own track reaches the wall (the sim
  analogue of the pointing coincidence).

**Gates (synthetic straight muons, 2000 per arm).** reco/true, flat:
- A: x 0.98 (= 1/kw_x), easing to 0.95 at |tan| 0.5; y 1.00.
- C: x 1.00, easing to 0.98; y 1.035 (= 1/kw_y).
- Capsule band/track on A: 1.02 / 1.01.
- 18–23 % wrong-sign x fits at |tan| > 0.35. The data's `sign_ok` is 0.87–0.91 there.

**Result 1: against the ideal gap line, electrons reconstruct like muons** (`analyse.py`):

| x, reco / ideal | 0.10–0.30 | 0.35–0.45 | 0.45–0.55 |
|---|---|---|---|
| A muons (gate) | 0.98 | 0.96 | 0.95 |
| A G4 beam electrons | 0.97–0.98 | 0.96–0.97 | 0.94 |
| C muons (gate) | 1.00 | 0.99 | 0.98 |
| C G4 beam electrons | 0.99–1.00 | 0.98–0.99 | 0.97 |

**Result 2: like-for-like capsule view** (`compare_data.py`: k_arm's selection, *reco* q_sum window
25–75 %, reco position, lever 30–130 mm; bootstrap errors). Full statistics: the 100-file reruns
`g4_{A,C}_is2_full` (condor cluster 4404719, 2026-10-08; ~4800 tracks per arm, 4× the 24-file sample).

| | band | track | r 0.1–0.2 | r 0.2–0.3 | r 0.3–0.4 | r 0.4–0.55 |
|---|---|---|---|---|---|---|
| A G4 ideal line | 1.108 ± .007 | 1.014 | 0.96 | 0.94 | 0.92 | 0.90 |
| A G4 → reco | 1.153 ± .007 | 1.038 | 0.94 | 0.91 | 0.89 | 0.84 |
| A data is2 | 1.194 ± .009 | 1.143 | 0.94 | 0.86 | 0.81 | 0.75 |
| C G4 ideal line | 1.069 ± .007 | 0.997 | 0.99 | 0.99 | 0.97 | 0.92 |
| C G4 → reco | 1.082 ± .007 | 0.988 | 0.99 | 0.99 | 0.95 | 0.90 |
| C data is2 | 1.187 ± .010 | 1.088 | 1.00 | 0.90 | 0.86 | 0.78 |

(24-file numbers, superseded: A reco band 1.141 ± .016, C 1.057 ± .013; the shifts are within 1–2σ.)

**Reading:**
- §12's "the reco compresses beam electrons 20–27 %" is wrong.
  - Against the ideal line, the reco is as good on beam electrons as on muons, within 1–2 %.
  - Most of the falling capsule response is the estimator on this population: the reco-charge window
    selects angle-correlated tracks, plus angular spread and failed fits. Even the ideal line in the
    same selection falls to 0.87 on A.
- **A significant residual remains.** In the data the capsule estimators read 5–13 % higher (band A
  +0.05, C +0.13) and the response 8–12 % lower at |tan| > 0.2 than the full simulation, on both arms.
  At full statistics the band residual is A +0.041 (3.6σ), C +0.105 (7.7σ); response at |tan| 0.2–0.55
  is 5–10 % lower in data on both arms (3–7σ per bin).
- Not in the simulation:
  - (a) a population that doesn't point at the capsule: hall die-away captures in the arms' structure,
    §10f; data capsule k is flat 10 → 60 ms, which weighs against it unless the early window is
    already die-away-dominated;
  - (b) model-vs-chamber mismatch that is larger for electron tracks than for muons;
  - (c) source/geometry: the sim capture vertices are capsule-Al.
- **Consequence for §11 step 1:** the cosmic in-situ scale is supported for beam electrons to within a
  ~5–10 % systematic. The capsule estimators are not truth for beam either way.
  - Any beam-angle truth needs this forward model, not a raw estimator.
  - The adoption decision is Dylan's.
- Still to do:
  - the 100-file reruns;
  - single-gun trends (`steps_single/`, e.g. 2 MeV tan 0.1–0.5, 5 MeV 0.1/0.5);
  - the data split by early/late on run_145 with is2;
  - prod-bundle (v 42.6) reruns, to check that the simulation reproduces production's k_arm 1.27/1.62;
  - a non-quiet overlay (busy triggers) as a systematic.

**Chasing the residual (2026-10-08, full statistics; data is2 run_145 vs `g4_*_is2_full`):**
- **Time since flash: no trend.** The data band is 15–30 ms A 1.16 / C 1.19, 30–60 ms 1.19 / 1.16,
  > 60 ms 1.17 / 1.10. Below 15 ms it is noisy (± 0.1–0.3). A late die-away population does not explain it.
- **No excess non-pointing population.** The x miss at the capsule plane, `lever − D·tan_raw`, is
  *narrower* in data than in the sim: median |miss| A 15 vs 17 mm, C 16 vs 16; fraction > 50 mm A 0.10 vs
  0.15, C 0.11 vs 0.15. Hypothesis (a) is disfavoured.
- **The residual survives in the pointing core.** With |miss| < 40 mm on both sides, the band is A 1.134
  data vs 1.069 sim reco, and C 1.118 vs 1.032. That is the same +6–9 % as with no cut. It is a scale on
  tan-vs-lever: either the angle scale is wrong for electrons (b), or D is wrong (c). The two are
  degenerate in one view. Both arms move the same way, so a displaced capsule (opposite signs on A and C)
  is disfavoured. A chamber stand-off common to both arms (~15–20 mm on D = 234.6) is not.
- **The y view does not share it.** I fitted the band with its own zero crossing, so no y foot is needed.
  Data y band A 1.44 ± .03, C 1.74 ± .08. Sim reco y: A 1.14, C 1.06; sim ideal y: A 1.12, C 1.09. The y
  residual is 3–7× the x one. That argues against a pure common-D error, which would hit both views
  equally. Something y-specific is much larger in data: the y kernel/scale on electrons, the y source
  extent, or the y lever window running into the active edge (data x0 is 33 / 44 mm). This has not been
  checked yet.
- **The band is a source-distribution estimator as much as a scale one.** In the sim, with the TRUE
  source position at the capsule plane held fixed (any quartile), the band is 1.00 ± 0.02 (x) and
  0.96–1.05 (y): the reco is right. It rises steadily with source width: A x 1.005 (3 mm rms) → 1.068
  (26 mm) → 1.09 (all, long tails). The 1.07–1.14 of the full sim sample is the spread of the source
  times the acceptance. So the data/sim residual can be a source mismatch as well as a scale one.
  Against a simply broader source in x: the data's x miss distribution has fewer tails than the sim's
  (> 50 mm: A 0.12 vs 0.17).
- **The charge dependence is flat.** The data/sim ratio is constant across charge quintiles in x and y,
  so charge-dependent sharing does not drive it.
- **Production-bundle reruns** (cluster 4405795, `g4_{A,C}_fp145_full`). is2 physics is reconstructed
  with the production run_145 bundles (`calib_bundle_prelim`, v 42.6; C on det6 `lp`). This uses the
  new `--sim-bundle`.
  - Sim reproduces production roughly: band A 1.266 vs data 1.287; C 1.605 vs 1.770 (100 files).
  - data/sim is A 1.018 (is2: 1.036), C 1.103 (is2: 1.097).
  - **C's ~10 % is the same under two different kernels (r06-det7 and lp)**, so it is not the
    kernel or the fit model.
- **The per-run gas drift is real, and in-beam muons confirm it** (`inbeam_through_goers.py`, which now
  carries C's tans and the y view). Sep < 15 mm, 20–80 ms, median true/raw on production raw tans:

  | | run_149 (beam off) | 79–98 | 100–124 | 128–147 | 150–162 |
  |---|---|---|---|---|---|
  | A x muons | 1.091 ± .005 | 1.04 ± .04 | 1.04 ± .02 | 1.06 ± .03 | 1.08 ± .03 |
  | A y muons | 1.106 ± .004 | 1.00 ± .04 | 1.04 ± .02 | 1.07 ± .03 | 1.02 ± .03 |
  | C x muons | 1.470 ± .007 | 1.27 ± .04 | 1.25 ± .03 | 1.41 ± .04 | 1.31 ± .04 |
  | C y muons | 1.611 ± .007 | 1.40 ± .04 | 1.42 ± .03 | 1.58 ± .05 | 1.39 ± .04 |
  | k_arm band A / C (median) | — | 1.26 / 1.59 | 1.26 / 1.63 | 1.29 / 1.73 | 1.26 / 1.59 |

  - Production's k_arm band varies per run by A ±1.2 %, C ±3.8 %, D ±4.3 %, coherently. It peaks at
    runs 135–147 and drops by run_150. C muons give 128–147 / 150–162 = 1.08 ± .05 (x); k_arm gives 1.09.
    The capsule band tracks the scale *ratio* between runs, so the source distribution is stable run to run.
  - **The muons do not support the band's residual.** The sep cut interacts with the mis-scaled
    production tans (run_149 C x moves 1.44 → 1.53 from sep < 20 to < 4 mm), so only ratios at equal
    cut mean anything. At sep ≤ 10 mm, runs 128–147 (run_145's period) read true/raw 3–5 % *below*
    run_149 on A x and C x (± 3–8 %), and equal on C y. That is consistent with no environment shift
    at ~± 4 %. If anything, is2 on beam makes tans slightly too large. The capsule band says too
    small, by 3.6 % (A) and 10 % (C): about 2σ (A) and 3σ (C) against the muons. That points the
    residual at the band's source and population modelling (hypotheses a/c), not at the scale. It is
    not proven; it needs more in-beam muon statistics (pool all periods after the gas correction).
  - Muon statistics per period are thin (n ≈ 50–250 at sep < 10).
  - **Pooled, gas-normalised to run_145.** Each in-beam muon's raw tan is scaled by k_run / k_145
    (k_arm band), then everything 20–80 ms is pooled. true/raw relative to run_149, i.e. what is2
    reads on run_145 beam muons:

    | | sep < 10 | sep < 6 |
    |---|---|---|
    | A x | 0.992 ± .016 | 0.991 ± .019 |
    | A y | 0.959 ± .013 | 0.971 ± .020 |
    | C x | 0.957 ± .018 | 0.945 ± .024 |
    | C y | 0.967 ± .016 | 0.964 ± .026 |

    So is2 reads beam-period muons right to 1–4 % (tans slightly too large, if anything). The capsule
    band's data/sim residual claims the opposite sign: +3.6 % (A), +10 % (C). That is ~3σ (A x) and
    ~7σ (C x) away. **The band residual is not an angle-scale error.** It belongs to the source and
    population model that the band depends on.
- **y stays open.** With the same cuts, the y residual is ~3× the x one. In-beam muons read y like
  x within errors, so it is beam-particle or source specific, and only in y. The sim's source is
  already wider in v (80 mm, 10–90 %) than in u (53 mm), so v is probably the capsule's long axis.
  An under-modelled source along that axis would hit y hardest.
- Code: `residual_checks.py` (time / miss / yview / charge / source; output
  `residual_checks.txt`), `compare_data.py --data prod`, `run_digi.py --sim-bundle`.

**Consequence for adopting is2 (§11 step 1):**
- The scale truth for beam should be the **in-beam muons** (A–C line), not the capsule band. The band
  needs a source model that the residual shows we do not have to better than ~10 %.
- A per-run (or per-period) correction is needed. The gas moves C's scale by up to ±7 % across the
  campaign. The relative per-run k_arm band (k_run / k_ref) tracks it, as the muons confirm.
- Proposed calibration (Dylan's decision): is2 bundles × per-run relative factor (k_arm band ratio to
  a reference run) × the pooled in-beam-muon normalisation (A x 0.99, A y 0.96–0.97, C x 0.95–0.96,
  C y 0.97; ± 1.5–2.5 %). The muons are muons, so for electrons this assumes the reco does not tell
  them apart. §13 Result 1 tested that against the ideal line (1–2 %).

**lxplus (2026-10-07):** the single-gun jobs had filled the AFS quota. Their ROOT files were left in
condor scratch and copied back, and 19 jobs went on hold. The ROOT files are now on EOS
`full_sim/angle_scale/single_root/` (16 kept; the 20 that failed transfer are lost). The job scripts are
fixed to leave nothing in scratch. The general rule and the `lxstore` tool are in `~/.claude/CLAUDE.md`.
