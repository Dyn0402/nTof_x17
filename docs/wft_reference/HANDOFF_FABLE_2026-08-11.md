# Handoff — critical review of the waveform-first reconstruction

**For:** a fable session, tasked with looking critically at `wft/` in the light
of Dylan's feedback on the reference document and of what the end-to-end
response simulation has learned.
**From:** a gatekeeper pass, 2026-08-11.
**Companion file:** [`FEEDBACK_2026-08-11.md`](FEEDBACK_2026-08-11.md) — the
full catalogue of the 33 comments, item by item, with sources. **Read it after
this file and before starting work.** Items are referenced below as F1…F33.

---

## 1 · What this is and what it is not

Dylan read the nine-part reference document (`docs/wft_reference/`) and left 33
comments. A gatekeeper pass has already answered the ones that were factual
questions, located every source, and killed the ones that were already done.
**What is left is genuine open work.** Do not re-derive the closed items; the
value of this session is in the ranked queue in §5.

Two constraints on scope:

- **MPGD2026 is the near-term deadline.** The conference deliverable is the
  presentation (`mpgd26/`), and the reconstruction is a supporting story, not
  the subject. Anything in Tier 1 should be doable and *checkable* before then;
  anything in Tier 2/3 is post-conference unless it falls out cheaply.
- **The end-to-end simulation (`~/CLionProjects/MX17_Geant`) does not yet match
  data and is being set down until after the conference.** Use what it has
  *established* — geometry, mechanism, and a handful of hard numbers. Do not
  start simulation work, and do not treat its absolute amplitudes as targets
  (the T14 verdict is "same ballpark, not in agreement", and the amplitude
  deficit is an open contradiction, not a calibration).

---

## 2 · Orientation — where everything lives

### The document under review
```
docs/wft_reference/
  README.md            what it is, how to regenerate, its own two findings
  sections/*.html      the document, nine parts, §1–58
  figsrc/*.py          the 42 figures, all from live sat_det3 products
  build.py             inlines the figures into page.html
```
Plain-text extraction of the whole document, if useful:
`/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17/*/scratchpad/wftdoc.txt`
(regenerate by stripping tags from `sections/*.html`).

**Reading stopped at §32.** Parts VI (from §33), VII, VIII and IX are
un-reviewed — a second batch of feedback is expected. Do not assume silence on
those sections means approval.

### The code
```
wft/model.py    410 lines  forward model, design matrix, kernels, NNLS chi2, raw fits
wft/reco.py     493        per-event reco, global start scan, errors, candidates, driver
wft/seed.py     124        the only place hits are read; clustering constants
wft/io.py       138        the only place raw waveforms are read; pedestal, CNS, noise
wft/calibrate.py 441       the four calibration stages
wft/calib.py    175        CalibrationBundle: save/load, conditions, provenance
wft/compat.py   133        bridge to the position-side analysis machinery
wft/tests/      317        seeding/selection, model regression, both share modes

wft/archive/rc_ladder_2026-07-31/   the OTHER implementation — read its README
```

### The driver chain
```
mx_june_wft/01_alignment.py  02_efficiency.py  03_angles.py  04_maps.py  digest.py
mx_june_wft/05_beam_kernel_xcheck.py    model-free kernel measurement
mx_june_wft/bench/run_bench.py          the A/B harness (hyper_patch variants)
mx_june_wft/condor/                     the fleet job packages
```

### The prior records, in dependency order
```
RECONSTRUCTION_BASIS.md                            the rule and the migration table
mx_june_cosmic_qa/waveform_first_threading/
    WAVEFORM_FIRST_THREADING.md                    the original study
    THREADING_DISPLAYS_2026-07-28.md               displays + the 600-event census
mx_june_wft/ANALYSIS_STATE_2026-07-31.md           50 KB — the RC-ladder line. IMPORTANT
mx_june_wft/DET3_GATE_2026-07-29.md                both chains, identical accounting
mx_june_wft/FLEET_2026-07-29.md                    the fleet run and the det7 lesson
mx_june_wft/BEAM_CONSTRAINED_CALIB_2026-08-05.md   beam kernel as a bench constraint
mx_june_wft/GAP_STUDY_2026-07-30.md                the cathode/column saga
mx_june_wft/WINDOW_ABLATION_2026-07-30.md          readout-window sensitivity
sps_beam_test_26/analysis/M70V_FLAT_ANALYSIS.md    direct kernel at normal incidence
sps_beam_test_26/analysis/EXTRACTION_2026-08-05b.md the share_lp port and its A/B arms
```

### The simulation
```
~/CLionProjects/MX17_Geant/                  the response sim (the one that matters here)
    README.md                                geometry + the real readout pattern
    design/GEOMETRY_FROM_CAD.md              as-built vs the models, discrepancies ranked
    design/GEOMETRY_IMPLEMENTATION_NOTES.md  what was implemented
    design/NEEDED_INPUTS.md                  what is still assumed
    design/RESPONSE_SIM_PLAN.md              the parameter table (§1), S1/S2/S3, §9 targets
    design/report/S3_ION_CLOSEOUT_2026-08-09.md
    shared/MX17ModuleGeometry.hh             the single geometry description
~/CLionProjects/MX17_Full_Geant/             the physics/neutron sim (shares the geometry)
mx17_sim_wft/T14_CAMPAIGN_2026-08-09.md      sim-vs-data verdict — read the top section
mx17_sim_wft/ANGLED_LADDER_2026-08-09.md     the angle-dependent v mismatch
```
Published T14 note: <https://dylan-neff.web.cern.ch/notes/t14-sim-vs-data-waveforms.html>

### Data
```
/media/dylan/data/x17/cosmic_bench            (mirror: ~/x17/cosmic_bench)
  Analysis/<run>/<subrun>/mx17_<det>/wft/
      calib_bundle*/bundle.json + arrays.npz
      calib_work/calib_cache.pkl              400 ref-pinned calibration events
      events.parquet (+ .meta.json)           the reconstruction
      alignment_lp/, efficiency/
  <run>/<subrun>/decoded_root/*_<feu>.root     the waveforms

reference run for everything in the document:
  Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/
```
Python: `.venv/bin/python` from the repo root (`../../.venv/bin/python` from a
subdirectory).

**Excluded by instruction: `mx17_det4_day_6-24-26`.** Dylan flagged it as
looking corrupted; it is separately known to mix 32- and 37-sample event windows
inside single files. Do not use it for calibration, ablation or validation, and
do not spend time diagnosing it — it is deferred (F3).

---

## 3 · Ground rules

1. **Never build position, angle or depth from `combined_hits` times.** The
   repo `CLAUDE.md` and `RECONSTRUCTION_BASIS.md` state this; §5–9 of the
   document measure it. Hits are for *detection* (which events, which strips)
   and QA only.
2. **The reference telescope may be seen by calibration and never by
   reconstruction** (§32). Seeding from the reference and then measuring
   alignment or efficiency against it is circular.
3. **Judge changes on implied-velocity flatness, not on χ².** w/tan θ_ref must
   be the same in every angle bin (§8.1). It needs no new data and has caught
   every calibration pathology so far. `mx_june_wft/03_angles.py` computes it.
   χ² alone is degenerate along the v↔kernel valley (§35) — a better optimiser
   finds the wrong bottom faster.
4. **Anything calibrated is per detector and per run condition.** A bundle used
   outside its conditions produces no error, no warning, and wrong physics.
5. **Do not tune to the simulation's absolute amplitudes.** Its verdict is
   "same ballpark, not in agreement", the deficit is ×0.53–0.64 with a surviving
   f_eff contradiction, and the data/sim gain slope differs by ×1.52. Use its
   *geometry* and its *mechanisms*, which are solid, and its measured
   microscopic numbers (σ₀, prompt induction) which are untuned predictions.
6. **A/B properly.** `mx_june_wft/bench/run_bench.py` exists for this and takes
   `hyper_patch` variants. The 2026-07-31 validation pass exists because a
   single-split scan overstated a gain by 2× (the winner's curse) —
   `ANALYSIS_STATE §10.1`.

---

## 4 · The four things that reframe the review

Details and evidence in `FEEDBACK_2026-08-11.md`; summarised here because they
change what the priorities *are*.

**4.1 · The resistive layer is strips running along y, and that makes X and Y
physically different (F6).** 515 discrete ESL strips, 550 µm wide, 250 µm
grooves, 800 µm pitch, along y. DC transport runs *along* the strips → it moves
charge in y → it lands on the **y-measuring** channels. The grooves block
transport across x, so the **X view cannot have resistive sharing at all** — its
±1 sharing must be drift diffusion. The simulation supports this with two
untuned numbers (avalanche footprint σ₀ ≈ 33 µm; point-charge prompt induction
to ±1 ≈ 0) and one prediction that came out right (c1_X = 0.202 untuned, and
**rising by a factor 45 across z** — the diffusion signature). Data agrees on
the asymmetry: τ_X ≈ 230 ns vs τ_Y ≈ 410 ns, ±1 peak-time shift +37 ns (X) vs
+87 ns (Y).

`wft` gives both planes the same kernel *form* with a depth-independent `c1` and
an amplitude-only `kY`. If the simulation is right, X's `c1` is duplicating the
`Dp·√u` term. And two long-standing "pathologies" are what you would expect if
it were: the calibration degeneracy always drives c1_X → 0 with σ_p0 inflating,
and under the RC-ladder kernel c1 sits on its 0.05 floor because "the data
genuinely want very little discrete sharing".

**4.2 · The joint two-plane fit exists and nothing calls it (F23).**
`wft/model.py:339 fit_joint()` — not exported, zero callers. §22 describes it as
part of the chain; the executed chain is independent 1D fits plus an
off-by-default *selection* coupling.

**4.3 · There are two kernel implementations and the kept one lacks the
measured per-plane time constant (F12).**
`wft/archive/rc_ladder_2026-07-31/` has `kTauY = 1.78` (τ_X 230 / τ_Y 410 ns,
measured directly on ~1000 near-vertical tracks per plane), `tau2_fac_y`,
`sigma_sY`, and a model-error term in the χ² weights. It was set aside at the
2026-08-06 merge because the shipped copy was newer and SPS ran on it — **not**
because it was better. Its README says neither is a superset of the other.

**4.4 · The bench has an external clock and the fit does not use it (F26).**
The DREAM trigger is a 60×60 scintillator coincidence at ~5 ns. `ftst` records
the trigger's phase against the DREAM clock, per FEU, and demonstrably carries
real timing information (`measure_dt_xy` separates det3 into two clean classes,
−7.65 and −16.01 ns). t0 is nevertheless fitted freely per plane per event, and
§20's warning against fixing t0 "from an external clock" was aimed at a
bundle-level constant, not at a per-event measurement. A t0 prior would break
the p0–t0 degeneracy that sits behind three separate known weaknesses.

---

## 5 · Scope of this session — the underlying model

**Dylan's instruction, 2026-08-11: work on the underlying model. The second
feedback batch is not coming for a while.**

That means the session's subject is **Parts III and IV — the forward model
(§10–18) and how it is fitted (§19–24)** — together with the pieces of Parts I,
II and V that feed them directly. This is the material he has read and can
argue with.

**In scope:** everything that changes what the model *is* — the kernel and its
per-plane structure, the transverse-spread parameterisation, the depth basis,
saturation, gain, the charge solve, the error definition, and the χ² itself
(which is where a t0 prior would live).

**Deferred, and say so rather than half-doing it:** the seeding and clustering
cut-hardening (F29–F31), two-track extension (F28), and the calibration-stage
internals in §33–37, which sit in territory he has not reviewed. F32 is a
partial exception — its first step is a one-line output change, worth doing
while nearby.

Do not treat "underlying model" as licence to redesign it. The document's
§43 ablation is the governing fact: **every single-component ablation stays
within ±0.05° of the full model on per-event angle σ.** The components earn
their place through *ensemble physicality* (implied-v spread), not per-event
resolution. So a change that improves χ² or per-event σ and worsens implied-v
flatness is a regression, however good it looks.

### The ranked queue

Ordering is by (importance × prospect) ÷ cost, with Tier 1 chosen so each item
either finishes before MPGD26 or produces a decisive answer cheaply.

### Tier 1 — before MPGD2026

**T1.1 · Test the absolute-t0 prior. (F26) — cheap, and it pays for T2.4.**
Do this first. **Read F26 in full before starting** — its claims were corrected
on 2026-08-11 and the naive version of this item sets the wrong expectation.

*What to expect.* **σ_t0 ≈ 3.5–4.5 ns is available.** Dylan confirms the ~5 ns
is *measured* on the cosmic bench reading out the actual scintillators, so it is
end-to-end and includes walk, discriminator and cables — there is no unbounded
term hiding in it. It is the **pair** figure, so a single paddle is ≈ 3.5 ns.
The range spans whether the 10 ns `ftst` quantisation (≈ 2.9 ns) is inside or
outside that measurement; either end is tight enough and check (b) measures the
total anyway. **This will not move σθ or the core position σ.** The
degeneracy direction is δp₀ = w·δt₀, which at typical w buys 0.04 mm against a
0.33 mm floor, and §42's ~1.0° floor is centroid jitter that no clock touches.
If you judge this item on headline resolution you will wrongly conclude it
failed.

*What it is actually for.* (i) It collapses the t₀ axis of `_global_start()`
stage 1 — an 11-point t₀ scan becomes 1 — which frees the grid budget to widen
the p₀ window, i.e. it **pays for T2.4's §21.1 fix**. (ii) Near-vertical planes,
where it is strictly stronger than the joint fit (T2.3): `fit_joint` ties t0x to
t0y but leaves the shared value free; an external clock pins it.

*Steps.*
- Confirm the trigger configuration for the det3 saturday run and whether an
  `ftst` correction is already applied anywhere (`qa_config.py`, the run config,
  the analyzer). The June time-resolution work established the bench has a
  scintillator trigger and an applied ftst — find that record first. While there,
  note *how* the 5 ns was measured (scintillators read out through DREAM ⇒ the
  clock quantisation is already inside it; through a separate TDC ⇒ it is not).
  Do not spend long — it is the 3.5 vs 4.5 ns difference and (b) settles it.
- **Prerequisite:** extend `_errors()` (`reco.py:87`) to three parameters. There
  is currently no `t0_err`, and the diagnostic needs one. Worth doing anyway —
  it is the same omission behind F25's caveat (2), where the p0–t0 correlation
  never reaches `p0_err`.
- Plot the **spread of fitted t0 within a single ftst class**, and read it
  **against the fit's own t0 uncertainty**, not an absolute threshold: spread ≈
  fit uncertainty ⇒ the trigger term is unresolvably small and the prior can be
  as tight as the budget allows (this is the *good* outcome and looks like a
  null result); spread ≫ fit uncertainty ⇒ the excess in quadrature is the real
  jitter, and is a better σ_t0 than any budget.
- If it holds: Gaussian penalty ((t0 − t0_pred)/σ_t0)² in `chi2_plane`, t0_pred
  per ftst class the way `dt_xy` already is, then A/B. Judge on the
  near-vertical bins (|tan θ| < 0.08) and the §21.1 wrong-basin rate — **not** on
  fleet σθ.

**T1.2 · The X/Y sharing asymmetry, as understanding and as presentation
material. (F6, F12)**
Not a code change first — a *measurement*. Test a depth-dependent X-view
sharing term against the constant `c1`, on det3, judged on implied-velocity
flatness. Whatever the outcome, the geometric story (strips along y → Y shares,
X cannot) is the clearest physical explanation of `kY` anyone has produced and
belongs in the talk.

**T1.3 · Dead strips as censored samples. (F1)**
Build a per-run dead mask from **hit rate**, not pedestal (a broken connection
downstream of the preamp still gives a normal pedestal). `find_dead_strips()` in
`beam_track_finding.py:296` already does this. Drop masked strips from the χ²
sum entirely — same machinery as the saturation mask in `chi2_plane`, but with
no penalty in either direction. Store the mask in the bundle. Expected effect:
removes a systematic pull on p0/w wherever a dead strip sits inside a window.

**T1.4 · The explanatory set — pure write-up, zero risk, and Dylan needs it for
the talk. (F7, F8, F9, F15, F24, F25)**
- F7 · a cartoon of the seven v_D estimators acting on one event display. The
  enumeration is done in `FEEDBACK_2026-08-11.md`; the drawing is not. Note the
  angle-dependent (not constant-scale) sim mismatch.
- F8 · two figures showing *why* deconvolution is unstable per event — the
  kernel transfer function against the noise floor, and per-event scatter vs the
  forward fit. **Soften §9**: deconvolution is not dead, it is what §45 already
  uses for the line-free displays.
- F9 · say plainly in the document that line-free 3-D clusters already exist
  (§45, `THREADING_DISPLAYS_2026-07-28.md`). Dylan did not realise.
- F15 · erf = the Gaussian CDF, i.e. "area under the Gaussian between the
  strip's two edges". One paragraph.
- F24 · how NNLS actually solves the 18 charges (Lawson–Hanson active set), and
  why the resulting sparsity is the §19.1 trap.
- F25 · what "profile likelihood" means here, plus the three honest caveats
  (boundary breaks Wilks; t0 is profiled but not propagated into `p0_err`; the
  pull width of 1.19/1.13 is consistent with that omission).

**T1.5 · Record the closed items and correct the document. (F3, F17, F20, F33,
plus findings A–F)**
Small, and it stops the same questions recurring:
- `~29 mm` → **30.000 mm** (the Al frame is the spacer). §11, §50.
- `K = 18` needs the note that the fleet was refit at K = 22 for slow chambers.
- §22 should say the joint fit is not in the executed path.
- §14.3 should say a per-plane τ exists in `wft/archive/` and was *measured*.
- Chase the `−18.8 ns` `dt_xy` default in `reco.py:270` — it matches neither
  measured det3 class.

### Tier 2 — high value, real work

**T2.1 · Merge the per-plane kernel. (F12)** Bring `kTauY` (and consider
`tau2_fac_y`, `sigma_sY`) into the shipped model, keeping the shipped RC-tail
treatment (F17 — the archived copy truncates at 6τ, the shipped one zero-pads to
6 µs and has a test). Note `ANALYSIS_STATE §10.1` found the timescale knobs did
*not* survive validation as free parameters — so pin them to the direct
measurement rather than fitting them.

**T2.2 · Saturation recovery. (F21)** Reference implementation:
`~/CLionProjects/mm_strip_reconstruction/waveform_analysis/src/WaveformAnalyzer.cpp:916`.
Recover the peak by two-sided linear extrapolation, then refit with a **large,
asymmetric** error — the recovered value is a lower bound, and a symmetric σ
would let the fit pull the peak back down, which is what censoring was
introduced to prevent. Confirm `SAT = 3550` against the real clipping behaviour
first.

**T2.3 · Turn on the joint 2D fit and A/B it. (F23)** It is written. Wire
`fit_joint` into `reco.py` behind a flag, export it, and A/B across the fleet.
Expected signature per §22: gains concentrated at |tan θ| < 0.08 where the
single-plane σθ degrades to 1.42° (X) / 2.35° (Y). While there, A/B
`WFT_PAIR_SELECT` too — §55 lists it as untested at fleet scale.

**T2.4 · The document's own top two open items — and note the dependency.**
- §21.1: the p0 global scan is centred on the window's charge centroid, but p0
  is the position *at the mesh*; on an inclined track those differ by ~half the
  transverse span (3.6 mm at |tan θ| = 0.25 vs a ±2.5 mm scan). 21 % of planes
  start outside the box and carry 5× the catastrophic-failure rate. Fix by
  centring on the *earliest* charge or widening with |w| in stage 2.
- §27: the candidate score is not scale-free — Δχ² is computed in each
  candidate's own window. **This is a prerequisite for T2.5**, not a nicety:
  two fits cannot be ranked until it is fixed.

**T2.5 · Make the algorithm two-track-extendable. (F28)** Design now, build
after the conference. Three pieces: the scale-free score (T2.4), overlapping
tracks in one cluster (iterative fit-and-subtract is natural because the model
is linear in charge), and X↔Y pairing with N > 1 (the ghost problem — this is
where `select_pair`'s t0 coincidence becomes essential). Validate by
**superimposing two real single-track events**: linearity makes that a
physically correct two-track event with known truth, no simulation required.

### Tier 3 — right ideas, lower prospect or later

- **T3.1 · Hunt the 31.2 mm beat, then consider a 1D (x) response map. (F22)**
  The resistive/readout pitch mismatch (800 vs 780 µm) predicts a
  position-dependent *sharing kernel* with a 31.2 mm superperiod in x. Look for
  it in det3 residuals and sharing before implementing anything — it is a
  falsifiable prediction either way. Per-strip gain is measured at 1.4–1.5 % on
  det3 and ablates to nothing, so gain is not the motivation; the kernel phase
  is.
- **T3.2 · Harden the fixed cuts. (F29, F30, F31) — DEFERRED per §5**, except
  the one piece below. Dynamic spark veto (charge and time structure, not a flat
  count of 50 — and check efficiency-vs-angle, because det3's sparks are
  muon-induced and edge-dominated); angle-aware clustering as a second pass
  alongside the production 12 mm split; `MIN_STRIPS` 3 → 2 behind a flag with
  `slope_reliable = False`. **All three change detection semantics or efficiency
  — gate each behind a flag and quote both ways** (§48).
- **T3.2a · Isochronous rejection — the one-line part, worth doing now. (F32)**
  Confirmed with Dylan as *isochronous*. The discriminator is already computed:
  `q_uend` (the fitted charge column's duration) and the `plausible` window
  250–1100 ns in `_candidate_score()`. It is used only to rank candidates and is
  **never written to the event table**, so nothing downstream can cut on it. Add
  `{plane}_q_uend` and an `{plane}_isochronous` flag to `row_from_fits()`. The
  two follow-ups — deriving the bound from `v_drift` and the local gap instead of
  the fixed 250–1100 ns det3-at-1000-V window, and adding a model-free companion
  measure for events where the fit does not converge — are described in
  `FEEDBACK_2026-08-11.md` F32 and are Tier 3 proper.
- **T3.3 · CNS guard for high occupancy. (F5)** Measured safe for cosmics
  (median 6 of 64 channels, p99 18–20). Not established for n_TOF. Build the
  two-pass CNS (identify signal, re-estimate the common mode without it) before
  n_TOF, not before MPGD26.
- **T3.4 · Window-independent kernel parameterisation. (F19)** Parameterise by
  the sheet transport D = 1/(ρ_s·c′(d_k)) — what the sim's S1 solver produces —
  and *derive* the window-truncated τ per configuration, instead of fitting a τ
  that means different things in a 1.92 µs and a full window.
- **T3.5 · Depth bins finer than 60 ns. (F14)** Decouple `DT` from
  `cal.sample_ns` (currently hard-bound, `model.py:84`) and ablate at 30 ns. Not
  rank-limited, so it is allowed; expected gain small because the template's own
  rise sets the resolution. Re-validate the column estimators (§50) afterwards —
  changing bin width changes NNLS sparsity.
- **T3.6 · Pin `sigma_p0` to the simulated σ₀. (F16)** The sim measures 33 µm;
  the det3 bundle carries 242 µm. §12 already calls this "the parameter that
  absorbs model error". Interacts with T1.2 — if X's sharing is really
  diffusion, a large `sigma_p0` may be absorbing exactly that.
- **T3.7 · Pedestal cross-check. (F4)** `wft` builds the pedestal from the first
  300 events of the physics run, not from the dedicated pedestal run that
  exists. Compare once; expected difference small.
- **T3.8 · ML, after the conference. (F10)** The best first target is not the
  fit or the deconvolution — both are bounded by a measured ~1.0° physics floor
  (§42). It is **candidate selection**: a classification problem with abundant
  labels and a measured 40-point gap (right cluster present 95 % of the time,
  found 55 %).

---

## 6 · Answered, and the one question left

**F32 — resolved 2026-08-11.** Dylan confirms *isochronous*. Written up in
`FEEDBACK_2026-08-11.md` F32, including what `q_uend` is (he had not met it) and
why it, rather than any spatial or multiplicity cut, is the right
discriminator — a vertical muon is flat in *position* but not in *time*, a
gamma flash is flat in both. Action is T3.2a.

**Second feedback batch — not coming for a while.** Parts VI (§33–37), VII, VIII
and IX stay un-reviewed. Per §5, do not start work whose justification lives in
those sections; if a Tier 2 item turns out to depend on the calibration-stage
internals, say so and stop rather than reviewing them by proxy.

**F26 — also resolved 2026-08-11.** The ~5 ns is the scintillator **pair**
figure, and it is *measured* on the cosmic bench reading out the actual
scintillators, so it is end-to-end: PMT spread, discriminator, walk and cables
are all inside it. Single paddle ≈ 3.5 ns. That removes the one unbounded term
from the σ_t0 budget and leaves **σ_t0 ≈ 3.5–4.5 ns** — see F26 for the revised
budget, the corrected expectation (it will *not* move headline resolution), and
the corrected gate.

**Nothing is blocking.** T1.1 can start.
