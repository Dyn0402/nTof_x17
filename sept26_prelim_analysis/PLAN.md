# The preliminary X17 analysis — plan of record

**Goal: an opening-angle distribution for e⁺e⁻ pair candidates from the 2026
n_TOF EAR2 campaign, preliminary, in time for the 30 September collaboration
meeting.**

Written 2026-09-07. The working week is **8–15 September**; Dylan is away
16–30 September. Everything here is scoped to what can be *finished* in that
week. Everything that cannot is written down in §9 rather than attempted — an
unfinished check is worse than a recorded one.

Companion documents:

| | |
|---|---|
| [`STATUS.md`](STATUS.md) | live state — what has run, what it produced, what is next |
| <https://dylan-neff.web.cern.ch/x17/analysis.html> | the board: pipeline status, log, deferred register, open questions |
| [`../RECONSTRUCTION_BASIS.md`](../RECONSTRUCTION_BASIS.md) | why geometry comes from waveforms and never from hit times. Binding. |
| [`../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`](../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md) | the DREAM ↔ n_TOF clock. Authoritative. |

---

## 1 · Where the physics lives, and what that implies

X17 at m = 16.8 MeV produced at E = 20.58 MeV has γ = 1.225, β = 0.577. The
two-body opening angle is minimised at symmetric energy sharing:

```
cos θ_min = 1 − 2 m² / E²  =  −0.333    →    θ_min ≈ 109°
```

so the signal piles up at **110–140°**, on top of an internal-pair-creation
continuum that falls steeply with angle from ~0°.

The four chambers sit at 90° in azimuth around the target, each a 400 × 400 mm
strip plane at 234.6 mm perpendicular distance, so **one chamber subtends about
±40°** in each in-plane direction (~90° corner to corner).

Two consequences that shape the whole plan:

- **The signal region is a two-chamber topology.** A 110–140° pair cannot fit
  inside one chamber. It lands in two different chambers — usually one of the
  good pair (A, C) and one of the poor pair (B, D), because A and C are
  *opposite* each other and each neighbours B and D.
- **The one-chamber topology is the control, not the signal.** Two tracks in a
  single chamber span 0–90°: the IPC continuum below the X17 threshold. It is
  the easier measurement, it normalises the continuum, and it is where the
  double-track machinery gets exercised — but no X17 peak can appear there.

We therefore build **both**, quote both, and are explicit that the interesting
one is the hard one. Naming, used consistently from here on:

| name | meaning |
|---|---|
| **intra-chamber pair** | both tracks in one chamber. θ ≲ 90°. Control / continuum. |
| **inter-chamber pair** | one track in each of two chambers. θ ≳ 60°. Signal region. |
| **implied pair** | one reconstructed track, plus an n_TOF wall+plastic coincidence in a *second* arm with no track from that chamber. Recovers B/D misses. Kept strictly separate, with its own false-positive rate. |

---

## 2 · What we are standing on (inherited, not to be rebuilt)

| | state | source |
|---|---|---|
| **DREAM ↔ n_TOF match** | 279 938 of 281 488 beam pulses joined since run_79 = **99.45 %**; 1 516 pulses had no n_TOF running (irrecoverable, outside the denominator). Per-trigger match efficiency 95.8 % (wall AND plastic), accidental 0.049 %, resolution 6 ns. | [qa-match](https://dylan-neff.web.cern.ch/x17/qa-match.html), `DREAM_NTOF_CALIBRATION.md` |
| **n_TOF reprocessing** | 445/445 runs have a complete product at the campaign recipe (`v12_liqpileup`). | `ntof_processing/STATUS.md` |
| **Reconstruction** | `wft/` waveform-first, per-arm bundles seeded from the June bench, `calib_bundle_r06` (kernel ordering fixed, bench t0 prior dropped for beam). Multi-track output exists (`n_tracks`, `*.candidates.parquet`). | `wft/`, `wft/MULTITRACK_2026-08-12.md` |
| **Geometry** | pinwheeled chambers, in-plane sign fixed, perpendicular lever, active-volume model ported from the as-built Geant4. | `ntof_tracking/RUN145_ALIGNMENT_2026-08-20.md`, `ntof_tracking/reco/geometry.py` |
| **Compute** | condor packaging, one job per (arm, file tag), fetch-at-CERN / return-parquet. Proven on run_145 (60 jobs). | `ntof_tracking/condor/` |
| **Pointing** | arms A and C independently back-project to the same spot, **−9.6 and −8.9 mm** in their common transverse coordinate, inside the r = 10 mm capsule bore. | `RUN145_ALIGNMENT_2026-08-20.md` §5 |

### Two campaign-wide conditions that must never be pooled across

Both from `ntof_pedestal_qa/README.md`, restated because every table in this
analysis has to respect them:

1. **The noise floor doubled on 23 July** (`RdClk_Div` 6 → 4) and never came
   back. run_67 and earlier are the quiet configuration; run_69 onward the
   noisy one; **run_68 is inside the bracket and unplaced**. The entire
   production period is on the noisy side.
2. **Chamber A x-view connector 8 was dead through run_79** (channels 448–511
   of FEU 3; 41 of 64 recorded nothing). Live again from run_83.

### Two calibrations that are known to be wrong or unmeasured

3. **The per-arm in-situ angle scale k is not physical on B and D.** A 1.25,
   C 1.58, B 1.99, D 1.70 → in-situ drift velocities from 34 down to 21 µm/ns
   against a Magboltz prior of 42.6. A's is credible, C's is quotable, B's and
   D's are not. B additionally truncates **46–56 % of its columns**; D has a
   dead region and a vertical stripe at u ≈ −135 mm.
4. **The residual ~9 mm offset common to A and C** is either a real target/beam
   offset or a survey error, and no measurement in hand separates them. Treat
   every absolute position as good to ~1 cm, not better.

---

## 3 · The chain, end to end

```
 [0] sample        which runs/sub-runs/tags enter, and what each cut costs   → sample.csv
        │
 [1] candidates    hits-level filter on combined_hits, ALL triggers.         → candidates.parquet
        │          per-arm cluster taxonomy + n_TOF arm coincidence            (one row per DREAM
        │          → one topology class per trigger                            trigger, ~31 M)
        │
 [2] reco          full wft waveform fit, ONLY on the classes worth it       → events/*.parquet
        │          + a prescaled control sample of the rest                    + candidates sidecar
        │
 [3] database      X/Y pairing → 3D segments → global frame → predictions    → tracks.parquet
        │          n_TOF hits attached per event, with derived positions       + scint.parquet
        │
 [4] pairs         intra-chamber / inter-chamber / implied, + backgrounds    → pairs.parquet
        │
 [5] spectrum      opening angle, per topology, acceptance-corrected         → the figures
```

Stages 0–1 are cheap and run over everything. Stage 2 is the expensive one and
runs only on what stage 1 selected — that asymmetry is the whole design.

---

### Stage 0 — Freeze the sample

**Input:** the frozen run registries already on the site (`x17-runs.json`,
`x17-match.json`, `x17-ntof-runs.json`) plus the slim and reco inventory on EOS.

**Cuts, each recorded with its cost in beam hours:**

- `mode = beam`, `phys = true`, target ³He, gas Ar/iso 90/10
- production trigger (wall AND plastic coincidence), i.e. **run_79 onward**
- all 8 FEUs present, `st = complete`
- not an HV or threshold scan
- the sub-run is joined to n_TOF in the pulse ledger
- **run_67 and run_68 excluded** (quiet / unplaced noise configuration — §2.1)
- **run_79 kept but flagged**: its arm-A x view must mask channels 448–511
  (§2.2), and any A-x occupancy or efficiency number from it is caveated
  separately.

**Output:** `sample.csv`, one row per (run, sub-run, file tag): events, matched
pulses, n_TOF run, conditions, flags.
**Figure:** beam-time timeline with the cuts shaded — the first slide of the talk.

---

### Stage 1 — The candidate filter (the key cost decision)

Full waveform reconstruction runs at ~1 000 events per core-hour per arm. The
campaign is **31.2 M triggers × 4 arms**, i.e. of order **10⁵ core-hours** to
reconstruct blind. That is not a week. So we filter first, on `combined_hits`,
which is small, already on EOS, and needs no fitting.

**Per trigger, per arm, per plane** — reusing `ntof_tracking/reco/noise.py` and
`reco/segments.py`, which already do exactly this:

1. remove the coherent noise bands (the ~1.3 MHz whole-plane oscillation) and
   isolated hits; apply the dead/hot channel mask for that run condition
2. cluster the surviving hits in (strip, time), Chebyshev link
3. classify each cluster: `track` / `point` / `band_fragment` / `blob`
4. count spatially separated (> 12 mm) track-like clusters per plane

**Per trigger, from the n_TOF side** — read straight off the slim file, joined
on `eventId`, no clock fit needed:

5. which arm(s) have a wall+plastic coincidence in the accept window
6. which arms have wall-only or plastic-only hits (the weaker evidence tier)
7. `t_since_flash`, the flash veto flag, and the derived E_n

**The classes.** One terminal class per trigger, so the ledger is a partition
and every downstream number has an honest denominator:

| class | definition | what it feeds |
|---|---|---|
| `INTER` | track-like clusters in exactly **2** arms | inter-chamber pairs (signal) |
| `INTRA` | ≥ 2 separated track-like clusters in **1** arm | intra-chamber pairs (control) |
| `IMPLIED` | track-like clusters in 1 arm, **but n_TOF coincidence in 2 arms** | B/D recovery |
| `SINGLE` | one track-like cluster, one arm | QA, efficiency, scintillator calibration (prescaled) |
| `BUSY` | ≥ 3 arms with track-like activity, or > 120 clean strips in ≥ 3 arms | pile-up: vetoed, but counted |
| `NONE` | nothing track-like | denominator only |

**Output:** `candidates.parquet`, partitioned by run — one row per DREAM
trigger with the class, the per-arm cluster counts, the n_TOF arm flags and the
flash time. Order 31 M rows of narrow columns: a few GB, and **the denominator
for every efficiency and rate in the analysis**. It is a first-class product,
not scratch.

**Deliverables:** the class census table (how much of the campaign is in each
class), and the class populations vs run, vs time-in-pulse, vs E_n.

> **Known unmeasured:** the filter's own efficiency. We do not know how many
> real pairs `NONE` and `SINGLE` swallow. The stage-2 control sample is what
> measures it — see D13.

---

### Stage 2 — Full reconstruction on what survives

Run `wft` through `ntof_tracking/wft_beam.py` on condor, one job per
(arm, file tag), **with an event-id allowlist** so only selected triggers are
fitted. Everything else about the driver is unchanged.

- **All of** `INTER`, `INTRA`, `IMPLIED` — and for `IMPLIED`, reconstruct the
  silent arm too, forced, so we can see whether there was any hint of signal.
- **A prescaled control** of `SINGLE`, `BUSY` and `NONE` (target ~1 %, tuned
  once the class census exists). This is not optional: it is the only thing
  that measures the stage-1 filter's efficiency, and without it the spectrum
  has no acceptance.
- **Double tracks.** `wft` returns every candidate; the sidecar carries
  `track_id` and `n_tracks`. Spatially separated doubles (> 12 mm) work today.
  **Doubles merged into one cluster (< 12 mm) are not resolved** — they come
  back as one bad-χ² track with `quality_ok = False`. We count them and report
  the loss; fixing it is D1.
- **Benchmark**, on the first sub-run and again at scale: core-hours per 1 000
  events per arm, wall-clock per file tag, output size. Recorded in `STATUS.md`.
  If the cost is out of line the filter tightens; the scope does not.

**Bundles.** `calib_bundle_r06` per arm, bench t0 prior dropped, DAQ constants
from the run's own `run_config.json`. A bundle is per detector **and** per run
condition — if the sample spans conditions that means a bundle per condition,
and the reco records which one it used.

---

### Stage 3 — The track database

This is the artefact the week has to leave behind, because every later analysis
reads it instead of re-running anything.

**`tracks.parquet` — one row per 3D track segment.**

| group | columns |
|---|---|
| identity | `run, subrun, tag, event_id, bunch, arm, track_id, cand_rank, class` |
| local | `x_local, y_local, tanx, tany, z_entry, z_exit, drift_len` |
| global | `p0_{x,y,z}, d_{x,y,z}` — a line, not a point; everything downstream is geometry |
| quality | `chi2dof_x, chi2dof_y, nstrip_x, nstrip_y, slope_reliable_{x,y}, quality_ok, n_tracks, pair_score, pair_ambiguous, pair_alt_id` |
| charge | `q_total, q_per_len, q_profile_p16/50/84` — the dE/dx handle: populated now, interpreted later (D5) |
| timing | `t0, ftst, t_since_flash, e_neutron` |
| pointing | `dca_axis, dca_target, target_x, target_y, in_bore` |
| predictions | `pred_sipm_bar, pred_sipm_seg, pred_plastic_bar, pred_ls, pred_u, pred_v` per crossed volume |
| provenance | `bundle, k_arm, v_drift, geom_version, code_commit` |

Two design rules, both learned the hard way upstream:

- **Never collapse an ambiguity at write time.** For a two-track event the X↔Y
  pairing is genuinely ambiguous (`wft/MULTITRACK_2026-08-12.md`: the t0 prior
  *increases* swaps). Write **both** hypotheses with their scores and a
  `pair_ambiguous` flag. Choosing is a downstream cut, and D3 is how we choose
  properly later.
- **Carry the calibration provenance in the row.** A bundle used outside its
  conditions is a silent error; the row must be able to say which bundle it is.

**`scint.parquet` — one row per n_TOF hit in a selected event.**
`detector (WAL/PSS/LIQ/PKUP), arm, segment/bar, amp_top, amp_bot, t_top, t_bot,
saturated, pileup`, plus **the derived position** (stage 4) and the id of the
track predicted to have crossed it. This is what makes "characterise the liquid
scintillators with the other detectors defining the track" a query rather than a
project.

---

### Stage 4 — Scintillator positions

The handles already exist and are already validated —
`ntof_processing/quality_metrics.py` metrics A1/A2 establish them on n_TOF data
alone:

- **Transverse (across the wall):** the wall segment that fired. 4 segments of
  4 bars × 25 mm = **100 mm bins**. Light sharing at a segment boundary may do
  better; check, don't assume.
- **Along the bar (the beam direction — EAR2's beam is vertical, hence
  "top/bottom"):** two independent estimators, both already characterised —
  - `Δt = t_top − t_bot`, the transit time along the bar;
  - `log(A_top / A_bot)`, linear in position with the attenuation length λ.

  A2 fits one against the other and reports the residual scatter. It also
  reports the caveat: `√(A_top·A_bot)` varies **+9.1 %** across a bar, which is
  either light-collection asymmetry or a reconstruction bias and has never been
  separated (`ntof_processing/REVIEW.md` §7).

**What is new this week:** calibrate both estimators against **Micromegas
tracks**. A confirmed track predicts the impact point on the bar — truth the
earlier study did not have. Fit λ and the Δt scale, quote a resolution, write
the resulting position into `scint.parquet`.

Plastics: 2 PVT bars 200 × 300 × 20 mm → bar id gives ~100 mm in u, nothing
along the bar. Liquid: known volume, position from the MM track only.

**Deferred:** the gain map across the LS surface and vs distance from the PMT
(D7), and any energy calibration (D6). This stage delivers *position*, not
energy.

---

### Stage 5 — Pairs and the opening angle

**Intra-chamber.** Two 3D segments in one arm, time-coincident, both
`quality_ok`. Opening angle from the two direction vectors; vertex from the
distance of closest approach of the two lines; require the vertex near the
capsule. Both X↔Y pairing hypotheses carried through, and the spread between
them quoted as a systematic.

**Inter-chamber.** One 3D segment in each of two arms on the same trigger. Each
must point back at the target (`dca_axis` cut, tuned on the single-track
sample, not invented). Opening angle from the two global directions. Reported
split by which arms: **A×C separately from anything using B or D**, because §2.3
says the B/D angle scale is not quotable.

**Implied.** One good track plus a wall+plastic coincidence in a second arm
with no track from it. The second direction is estimated from the target to the
fired wall segment — coarse, ~100 mm at ~330 mm, so tens of degrees. Reported
**separately, never merged into the measured spectrum**, with a measured
false-positive rate from:

- the accidental n_TOF two-arm coincidence rate (0.15 % of triggers have ≥ 2
  arms in the window — already measured; extend it per arm pair), and
- a data-driven control: events where the second arm *did* reconstruct a track,
  with that track hidden, so the implied direction can be compared to the real
  one.

**Backgrounds, all data-driven:**

| background | how it is measured |
|---|---|
| accidental two-track | event mixing — pair tracks from different bunches through the identical selection |
| cosmic | the beam-off cosmic runs at the production operating point (run_83, run_146) give the rate directly |
| flash-induced | the `t_since_flash` veto; the residual measured from the veto sideband |
| external conversion in material | geometry/material argument only this week — D12 |

**Acceptance.** A straight-line toy through the existing active-volume model:
throw pairs from the capsule with a flat opening-angle distribution and
isotropic orientation, propagate, apply the volumes and the per-arm
reconstruction efficiency measured on the control sample → `acceptance(θ)` per
topology. Preliminary, geometry only, no multiple scattering. The spectrum is
plotted raw **and** acceptance-corrected side by side, so the correction is
visible rather than baked in.

**Not attempted:** the invariant mass. m_ee needs the energy sharing, which
needs scintillator calorimetry, which needs a calibration we do not have (D6).
**The opening-angle distribution is the deliverable.**

---

## 4 · The local development run

Debug the whole chain on one run here before anything goes to condor.

**run_145** is the choice:

- 4 sub-runs, 190 109 events, 62 GB — small enough to move, big enough to be real
- 2026-08-05: post-23-July (production noise condition) and post-run_83 (arm A's
  connector 8 alive), so neither §2 condition bites
- production trigger, ³He target, Ar/iso 90/10, at the run_67 optimum
- **already reconstructed on all four arms** with `calib_bundle_r06` — the
  parquet exists at CERN at ~1 MB per (arm, tag), so stage 3 onward can be
  exercised without moving a single waveform
- its slim n_TOF join exists (`ntof_hits_run_145_stat090_0000_224670.root`)
- its geometry defects are already found and fixed, and its pointing result
  (A and C agreeing at −9 mm) is the sanity check the chain must reproduce

**What to pull, in this order** — the link is the constraint (310 kB/s was
measured to CERN in August, so measure it again first):

1. the existing `events_prelim.parquet` + `*.candidates.parquet`, all 4 arms — MB
2. the slim n_TOF root file for `stat090_0000` — hundreds of MB
3. `combined_hits` for all 4 sub-runs — the stage-1 input, modest
4. `decoded_root` for **one file tag, all 8 FEUs** — the stage-2 input, ~1 GB,
   enough to prove the reco path end to end locally

**run_79** is the cross-check once the chain works: it has the most existing
analysis to disagree with, and it carries the arm-A channel mask — exactly the
kind of condition the pipeline has to handle without being told.

---

## 5 · Compute

| | |
|---|---|
| stage 1, campaign | `combined_hits` only, no fitting. Order 10³ core-hours, one condor job per sub-run. Overnight. |
| stage 2, campaign | driven by the class census. If `INTER + INTRA + IMPLIED + control` is ~1 % of triggers, that is ~10³ core-hours — a day on condor. **If it is 10 %, the filter tightens.** |
| stages 3–5 | laptop-scale. The whole campaign track database should be < 10 GB. |

The benchmark is not a footnote: measure it on run_145, write it in
`STATUS.md`, re-measure once at scale.

---

## 6 · The week

| day | target |
|---|---|
| **Mon 8** | lxplus access; verify the slim and reco inventory at CERN; pull the run_145 bundle; scaffold this package and the presentation figure style |
| **Tue 9** | stage 0 (sample frozen) and stage 1 written and run on run_145; class census |
| **Wed 10** | stage 2 on run_145 candidates + benchmark; stage 3 database schema and writer |
| **Thu 11** | stage 4 (scintillator positions vs MM tracks); first stage-5 look on run_145; **decide the campaign scope from measured yields** |
| **Fri 12** | launch the campaign stage-1 pass, then stage 2, on condor. While it runs: acceptance toy, event-mixing background |
| **Sat 13 – Sun 14** | merge, QA, the spectra, the figures |
| **Mon 15** | note published, board current, October hand-off written |

The analysis runs on the **Ubuntu laptop** — repo `~/PycharmProjects/nTof_x17`,
python `.venv/bin/python`, data `/media/dylan/data/x17/`, and a working
`kinit` / `ssh lxplus`. `STATUS.md`'s **Resume here** section is the ordered
checklist for the first session there.

**The fallback, decided now rather than on Friday:** if the campaign pass does
not complete, run_145 + run_79 + run_147 alone is ~3.6 M triggers and still
produces every figure. The spectrum's statistics change; nothing else does.

---

## 7 · Figures

Every figure is built for a **projected slide**, not a page:

- 16:9, generated at a fixed size, minimum font size readable from the back of a
  room (base ≥ 18 pt at final size, axis labels larger)
- one message per figure; the message is the title
- a `Preliminary` badge on anything that touches the reconstruction — the
  convention `ntof_run_report` already holds itself to
- a shared style module in this package so every figure in the deck matches
- exported PNG at presentation resolution **and** the underlying numbers as CSV
  beside it, so a figure can be rebuilt without rerunning the analysis

---

## 8 · What this analysis will explicitly NOT establish

Stated up front so no figure has to carry the disclaimer alone:

1. **No resolution measurement.** There is no reference telescope at n_TOF.
   Every width is an upper limit.
2. **No absolute position better than ~1 cm.** The A/C common ~9 mm offset is
   unresolved between a real target offset and a survey error.
3. **B and D angles are not quotable.** Their in-situ k is unphysical. They
   enter as *tagging* chambers (was there a track at all?), their angles carry a
   flag, and any inter-chamber pair using a B or D angle is reported separately
   from an A×C one.
4. **No invariant mass.** No energy calibration.
5. **No efficiency from first principles.** Acceptance is geometry plus a
   measured reconstruction efficiency, both preliminary.
6. **Merged double tracks (< 12 mm) are lost** — counted, not recovered.

---

## 9 · Deferred to October and beyond

The register. Each item has a stable ID so the board, `STATUS.md` and future
notes can point at it; each says **why it is safe to defer** and **what
unblocks it**.

### Reconstruction

- **D1 — Merged-cluster double tracks.** Two tracks < 12 mm apart fit as one
  bad-χ² compromise. Needs a two-column NNLS design matrix (2K basis, 6 outer
  parameters) plus a 1-vs-2-track model-selection penalty.
  *Safe because* they are flagged (`quality_ok = False`) and counted, so they
  are a measured loss rather than a silent one. *Unblocked by* model work in
  `wft/model.py`.
- **D2 — Simulate the double-track finding efficiency.** Push Geant4 pairs
  through the response sim and the full chain and measure what we miss, by
  opening angle and by track separation. *The single most valuable October
  item* — it turns every "we probably lose some" in this plan into a number.
- **D3 — Break the X↔Y pairing ambiguity for simultaneous tracks.** Two handles,
  neither used yet: (a) **does the pairing point back at the ³He sample** — the
  wrong pairing generally does not; (b) **charge deposition along the track** —
  the correct X and Y projections of one particle must carry the same charge
  profile. Both are already columns in the stage-3 schema (`dca_target`,
  `q_profile_*`), unused this week.
- **D9 — In-situ v_drift and diffusion, per arm and per run condition.** The
  transferred bench values are the largest unvalidated assumption in the chain.
  Port `mx_june_wft/bench/gap_study.py` (stacked NNLS + erfc endpoint); costs no
  refit.
- **D13 — Measure the stage-1 filter efficiency properly.** This week's
  prescaled control gives a first number; a dedicated unfiltered pass over one
  sub-run gives the real one.

### Detectors

- **D4 — Chambers B and D.** B truncates 46–56 % of its columns and its k is
  1.99; D has a dead region, a vertical stripe at u ≈ −135 mm and a k that is
  not single-valued (1.18 / 1.96 on two sides). Both need a dedicated
  calibration study. *Safe because* they are flagged and their angles are never
  quoted alone. **This is the biggest single limit on the signal region**, since
  the X17 topology usually needs one of them.
- **D5 — dE/dx in the Micromegas.** Expect gain variation across a chamber to
  dominate; a per-region gain map has to come first (D10).
- **D8 — Absolute alignment.** Separate the residual ~9 mm A/C offset into a
  real beam/target offset and a survey error. Needs survey information or a
  beam-spot measurement, not more tracks.
- **D10 — Per-channel gain and dead-strip maps** per run condition, from the
  pulser and source runs.

### Scintillators

- **D6 — Energy calibration and calorimetry** for plastics, wall and LS. This is
  what unlocks the invariant mass, and it is a project, not a task.
- **D7 — Liquid scintillator gain vs position.** Specifically: how extreme is
  the gain change with distance from the PMT, and across the surface? Verify
  before any LS amplitude is used as energy. The +9.1 % `√(A_top·A_bot)`
  variation on the wall bars is the same question in its milder form, and is
  also unresolved.

### Physics

- **D11 — The neutron-energy axis.** `t_since_flash` → E_n over 19.5 m, and the
  opening-angle spectrum resolved in E_n. The timing is already in the database;
  only the analysis is deferred.
- **D12 — External pair conversion** in the capsule and frame material, as a
  measured background shape rather than an argument.
- **D14 — run_67 and run_68.** The quiet-configuration runs, excluded this week
  by §2.1. Either bring them in with their own calibration, or record the loss.
- **D15 — Pile-up within a bunch.** ~112 triggers per bunch; the `BUSY` class is
  vetoed wholesale this week and its contents never examined.
