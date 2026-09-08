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

---

## 10 · The alignment and spectrum phase — S1 to S4

**Added 2026-09-08**, from Dylan, after the run_145 chain closed end to end.
Stages 0–3 run; the local development run is complete. What is left before a
full-statistics campaign pass is worth launching is *understanding the local
run* — and specifically, understanding the **geometry**, because every number
in §5 is divided by an acceptance and the acceptance is geometry.

Four workstreams. They are ordered by dependency, not by size: **S4 needs S2**,
and S2's y handle needs one measurement out of S1.

| | workstream | ships |
|---|---|---|
| **S1** | the n_TOF scintillators — what is integrated, and what is only a filter | `x17/scintillators/` |
| **S2** | imaging the ³He capsule; what the chamber-to-chamber spread says about alignment | `x17/source-imaging/` |
| **S3** | drift velocity along the gas chain, and the contamination it implies | folded into `x17/reco-funnel/` |
| **S4** | the opening-angle spectrum against an acceptance-folded expectation | `x17/opening-angle/` |

Every one of them is a **generated** page (`make_*_report.py`), figures with
CSV beside them, relative links, `Preliminary` badge — the same contract as
`reco-funnel` and `detector-response`.

---

### S1 — The scintillators are a *filter*, and that is all they are

**The answer to "has the n_TOF scintillator information been fully integrated?"
is: it is fully integrated as a tag, and not at all as a measurement.**

Three roles today, all of them boolean:

1. **Stage 1** — `candidate_filter.py` reads `det`/`detn`/`dt_ns` from the slim
   and asks *which arms have a wall+plastic coincidence in the accept window*.
   That is what separates `INTER` from `IMPLIED` from `NONE`.
2. **Efficiency** — `efficiency.py` uses "wall AND plastic fired in arm X" as
   the MM-independent denominator, accidental-corrected against the untagged
   control.
3. **Funnel** — `funnel.py` asks the sharper question: does the track
   extrapolate to the wall *segment* and the plastic *bar* that actually fired.
   That is the pointing confirmation, and it is the strongest column in the
   report.

**Nothing reads an amplitude or a time except as "in the window".** No
position, no energy, no `scint.parquet`. PLAN §Stage 4 is unstarted.

The slim carries more than we use — verified 2026-09-08 on
`ntof_hits_run_145_stat090_0000_224670.root`: `amp`, `amp_0`, `area_0`,
`fwhm`, `risetime`, `chi2`, `satuflag`, `pileup1`, `pulseshape`, `shadow_amp`,
`shadow_dt`, and `tof` as a `double`. **So stage 4 needs no reprocessing and no
EOS** — only the analysis.

**Deliverables**

- the audit page: the three roles, drawn; the branch inventory; and an explicit
  list of what a scintillator-derived number would unlock (S2's y handle,
  D6 energy, D7 gain-vs-position) and what it would cost.
- **one new measurement, because S2 needs it**: the wall is read out
  **top and bottom** (`detn` odd/even within a group of 4 bars), so
  `Δt = t_top − t_bot` and `log(A_top/A_bot)` both measure **position along the
  bar — the beam axis, y** — the one coordinate the capsule's 80 mm length
  denies the pointing method. Calibrate both against Micromegas tracks that
  predict the crossing point on the fired bar; quote λ, the Δt scale and a
  resolution. If it works it is a *second, independent* handle on the capsule's
  y position, and it is the only one that does not go through the chambers.
- the caveat that already stands: **LIQ C is effectively dead in run_145**
  (891 in-time hits against A's 7 227), so LIQ stays out of the partition.
- and the one that is deferred and must be repeated on the page: **a
  scintillator tag is not proof a charged particle crossed the gas.** Chamber A
  measures 63 % where 80–90 % is expected, and a neutron or gamma converting in
  the PCB would push it exactly that way.

---

### S2 — Where the ³He capsule is, and what the disagreement means

The pointing band's **zero crossing is scale-free**: it is the strip coordinate
at which a track from the source is normal to the plane, so it does not depend
on `k`, on `v_drift`, or on the bundle. That is what makes it the right
alignment observable — it survives every calibration doubt in §2.3.

Each chamber measures **one** transverse coordinate, the one along its own
`u_hat`:

| chamber | u_hat | measures | run_145 |
|---|---|---|---|
| A | +x | global **X** | −7.86 ± 0.31 mm |
| C | −x | global **X** | −8.97 ± 0.41 mm |
| B | +z | global **Z** | −3.82 ± 1.97 mm *(one sub-run; see below)* |
| D | −z | global **Z** | −4.33 ± 0.52 mm |

> **MEASURED 2026-09-08, and it corrects this table's first version.** The
> numbers here were read from a cached `imaging_summary.json` written the day
> before, which said D was **−48 / −36 / −36 mm** — 40 mm off axis. That cache
> went stale under the y in-plane sign fix: the fix changed which tracks are
> pointing-coincident (the coincidence predicts a *v* on the wall and the
> plastic) and the sample shrank 30–45 %. Re-running the *identical* estimator
> on the current reconstruction gives D at **−10.0 / −9.6 / −6.1 mm**, and
> −4.3 ± 0.5 with the charge window the angle scale already uses. `k_arm` now
> computes the crossing from the sample in memory instead of reading the file.
>
> **And chamber B does contribute.** The crossing asks only where the
> reconstructed angle is zero — where the track is perpendicular to the plane —
> which a distorted but symmetric field still puts in the right place. B has no
> *slope* and never will, but it gives one Z point at ±2 mm, so **Z is
> cross-checked after all**. Everything below about "Z is measured once and the
> answer is not believable" is retracted: the source is at
> **X = −8.4, Z = −4.1 mm, 9.3 mm from the beam axis, inside the r = 10 mm
> bore**, with a chamber-to-chamber alignment spread of ±0.6 mm.
> Published: <https://dylan-neff.web.cern.ch/x17/source-imaging/>

**X is the measurable one, and it is measured twice.** A and C look at the same
number from opposite sides, so

- their **mean** is the capsule's X, and
- their **difference** is a *relative in-plane offset* between A and C — a
  chamber's strip-coordinate origin shifted by δ moves only that chamber's
  estimate. A − C = **+3.8 mm**, reproducing across all three sub-runs.

That is the measurement to make final (preliminary): **X = −9.6 mm, with a
±1.9 mm chamber-to-chamber alignment systematic**, against a capsule bore of
r = 10 mm. It is already consistent with the r = 10 mm bore — the point of the
work is to quote it *with its systematic* and to say which part of it is target
and which is survey. That split is D8 and stays open.

**Z is measured once, and the answer is not believable as it stands.** D says
the source is 36–48 mm off axis in z — four to five capsule radii, and a gross
installation error if true, while A and C see only ~1 cm in X. The obvious
suspect is in the chamber, not the target: **D has ~130 dead channels of 512,
and they are one-sided** (x_local +0…+57 mm and +150…+178 mm). A band fit whose
acceptance is asymmetric about the crossing pulls the crossing. So:

1. re-measure D's zero crossing **with the dead runs masked**, and with a
   lever window forced symmetric about the crossing;
2. re-measure it **on the outer-ring-excluded sample** that already moved D's
   `k` by 2 %;
3. if it stays at −40 mm, say so, and record that Z is uncertain at the 4 cm
   level with **no second chamber to check it** — which is itself a result, and
   the strongest single argument for what chamber B costs us.

**Y — the coordinate with no lever.** The capsule is a point in XZ (r = 10 mm)
but 80 mm long in y, so no zero crossing exists. Three routes, in increasing
order of how much they assume:

- **(a) the pointing distribution.** `target_y_mm` at closest approach to the
  beam axis is a per-track estimate of the emission height. Its *distribution*
  is the capsule's y profile convolved with the y resolution. The gas profile
  is known exactly (`geometry.HE3_GAS_Y/R`, a STEP-derived polycone from
  y = −29.5 to +50.7 mm), so this is a forward comparison, not a fit: predict
  the `target_y` distribution from the polycone, compare to what each chamber
  sees, and let the **offset between chambers** be the alignment number the
  way A−C is in X. Run_145 medians on all gated tracks are **A +16.2, C +14.3,
  D −33.5 mm** — A and C agree to 2 mm and D is 50 mm away, the same shape of
  disagreement as in Z, and quite possibly the same cause.
- **(b) the expected hit map in v.** The occupancy along the chamber's vertical
  coordinate is the capsule's y profile projected through the plastic-bar
  acceptance (`plastic_acceptance.py` already ray-traces the capsule, and the
  measured occupancy already sits inside its contours). Shifting the capsule in
  y shifts that pattern; the comparison is a one-parameter fit per chamber.
- **(c) the wall's top/bottom ratio** — S1's new measurement, and the only one
  that does not use a Micromegas angle at all.

Agreement among (a), (b) and (c) is the result. Disagreement localises the
problem to the chamber it comes from, which is the point.

**Double-track vertexing.** If two tracks in one event come from a common
vertex in the capsule, their distance of closest approach *to each other*
locates that vertex in all three coordinates at once — no beam-axis assumption,
and the y lever comes back. Run_145 has, with `dca_axis < 30 mm` and both
tracks gated: **160 intra-chamber and 444 inter-chamber** candidate events.
That is enough to *test the method* and not enough to *make the measurement*, so
this week's job is:

- build the two-line vertex (DCA point, DCA distance, per-track pull);
- show, on the intra-chamber sample, that the vertex distribution collapses
  onto the capsule and not onto the chamber — and quantify how much of it is
  accidental by running the identical estimator on **mixed events** from
  different bunches;
- from the observed vertex resolution, state **how many events a y measurement
  needs**, so the campaign pass has a target rather than a hope.

**Deliverables:** `x17/source-imaging/` — the geometry drawn, the three
coordinates each with its estimate and its systematic, the chamber-to-chamber
disagreement table, the double-track vertex demonstration and its statistics
projection.

---

### S3 — The gas chain, made obvious

The four chambers are **daisy-chained on one gas line, A → B → C → D →
exhaust** (`mx_july_beam_qa/DRIFT_WINDOW_HANDOFF.md` §0, confirmed twice). In
run_145 **all four drift cathodes are at the same 700 V** (HV monitor:
v0 700.0, vmon 699.8–700.0), so E = 233 V/cm everywhere and **the velocity
differences are gas, not field.** The measured in-situ velocities fall
monotonically along the chain:

| position on the line | A | (B) | C | D |
|---|---:|---:|---:|---:|
| v_insitu [µm/ns] | 33.6 | — | 26.4 | 24.2 |
| deficit vs Magboltz 42.6 | −21 % | | −38 % | −43 % |

**That is the signature of pickup along a series line**, and it is the same
mechanism the June fleet and the July run_58 work already established: each
chamber outgasses water into the gas that feeds the next one.

**Changes to the figure** (`make_figures.fig_k_summary`):

- **drop B's marker** — B has no drift field, so its `k` is not a velocity —
  but **keep its slot on the x axis**, labelled, so the series reads
  A → B → C → D and the gap is visibly *a chamber we cannot measure* rather
  than a chamber that is not there;
- draw the chain explicitly: an arrow along the axis, gas in at A, exhaust
  after D;
- put a **second y axis in implied H₂O %**, from the Magboltz curves we already
  have.

**The contamination estimate**, from `garfield_sim/results/drift_9010_contam_cern.json`
(Ar/iso 90/10, CERN 720.8 Torr, evaluated at 233 V/cm):

| mixture | v [µm/ns] | η [cm⁻¹] |
|---|---:|---:|
| pure 90/10 | 42.6 | 0 |
| +0.3 % H₂O | 35.6 | 0 |
| +0.5 % H₂O | 30.1 | 0 |
| +1.0 % H₂O | 19.3 | 0 |
| +5 % N₂ | 35.2 | 0 |
| +1 % O₂ | 41.6 | **3.70** |
| +3 % air | 38.2 | **2.61** |

Interpolating: **A ≈ 0.35 %, C ≈ 0.6 %, D ≈ 0.7 % H₂O**, rising monotonically
down the line. Two exclusions come free and must be stated with the number:

- **O₂ and air cannot do this.** They barely move v (1 % O₂ still gives 41.6)
  and they attach at η = 2–4 cm⁻¹, which would strip the cathode-side charge to
  a per-cent of the anode side. `garfield_sim/attachment_run58.py` measured the
  opposite on real data — flat-to-rising amplitude across the full 30 mm.
- **N₂ cannot explain any of them.** The ladder tops out at 5 % N₂ =
  35.2 µm/ns, which is *above every chamber here* — so N₂ does not reach even
  the driest, and it has no natural source that does not also bring O₂.
  (This corrects the first version of this line, which said N₂ could account
  for A; the inversion returning NaN is what caught it.)

**Caveats that go on the figure, not in a footnote:** the in-situ v is
`v_prior / k` and `k` is the angle scale, whose focus objective is flat over
±20 % — so the *ordering* is much better established than the absolute level;
and v is derived assuming a 30 mm effective gap, so a smaller effective gap
shrinks every deficit and every implied water fraction together.

---

### S4 — The opening angle, against something

The current figure is a **geometry check**: 939 inter-chamber pairs from
run_145 (A–D 419, C–D 291, A–C 229, no B), opposing chambers at 144° and
perpendicular at 83–97°, which is what the geometry demands. It is not yet a
measurement, because there is nothing to compare it to.

**What is missing is the expected shape**, and it factorises:

```
   expected(θ | topology)  =  physics(θ)  ⊗  acceptance(θ | topology)
```

**The acceptance** is ours to build, and S2 is what makes it credible. A
straight-line toy through the as-built model:

- vertices sampled from the **He-3 polycone** (`geometry.HE3_GAS_Y/R`), shifted
  by the S2-measured capsule offset — that is the coupling between the two
  workstreams;
- pairs thrown with a flat opening-angle distribution and isotropic
  orientation, so the toy measures acceptance and not physics;
- propagated through the **as-built volumes**, the **plastic-bar trigger
  acceptance** (`plastic_acceptance.py` — already validated against the
  measured occupancy, both lobes and the gap), the **dead-channel masks**
  (D's ~130, C's 10), and the **measured per-chamber efficiency map**
  (`efficiency.py`, binned in u);
- **B enters as a tagging chamber only**, never with an angle, so a B pair is
  `IMPLIED` and is reported separately.

**The physics** is the part we do not own, and it needs care. There is a full
Geant4 pair simulation at `~/CLionProjects/MX17_Full_Geant`:
`analysis/al_pair/signal_reco.npz` holds **300 000 events** with
`type` (0 = X17, 1 = IPC), `theta_truth`, `theta_reco` (MSC-smeared),
`n_mm`, `same_arm` — X17 median 117°, IPC median 30°, and 2-MM acceptance
already applied. That is the right starting point and it is **not** the whole
answer, because its IPC generator samples

```
dN/dM_ee  ∝  1/M_ee      (log-uniform, 2m_e → E_transition), isotropic in the γ* frame
```

which is an ansatz, not the IPC matrix element. The real internal-pair rate
depends on the multipolarity of the 20.58 MeV ⁴He transition and carries its
own angular correlation. So:

1. take the Geant IPC truth as the **baseline** shape;
2. implement the standard IPC angular correlation for the relevant
   multipolarity as an **alternative** shape;
3. quote the difference as the **dominant modelling systematic**, as a band on
   the expectation and never as a single curve.

**The categories**, split identically in data and in the expectation, because a
shape comparison is only as good as the topology it is made in:

| category | arms | why |
|---|---|---|
| **intra-chamber** | both tracks in one chamber | θ ≲ 90°: pure IPC continuum, no X17 possible. **The normalisation and the strongest background test.** |
| **perpendicular** | A–B, A–D, C–B, C–D | θ ~ 60–120°, the acceptance turns over here |
| **opposing** | A–C, B–D | θ ≳ 110° — **the signal region** |
| **implied** | one track + a second arm's scintillator coincidence | B recovery; separate, with its own false-positive rate |

**The test that carries the least model dependence** is the *ratio* between
categories: a steeply-falling IPC continuum and a 110–140° X17 peak differ
most between intra-chamber and opposing, and a great deal of the acceptance
uncertainty divides out. Quote the raw spectrum, the acceptance-corrected
spectrum, and the category ratios, in that order.

**Statistics.** run_145 gives 229 A–C pairs. The campaign is ~50× run_145, so
the campaign pass is what turns this from a shape check into a measurement —
and S4's job this week is to prove the machinery and **state the yield the
campaign needs**, exactly as S2 does for the vertex.

**Deliverables:** `x17/opening-angle/` — the four categories in data, the
acceptance per category, the folded expectation with its IPC band, the raw and
corrected spectra side by side, and the statistics projection.
