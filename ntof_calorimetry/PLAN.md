# ntof_calorimetry — what energy information the n_TOF data can give, and MM dE/dx

**Plan written 2026-10-08.** Steps 1-2 (C1.1, C1.3, C4) ran the same day:
see **Results so far** at the end, and the generated report
`/media/dylan/data/x17/calorimetry/report.html`.

Numbers below marked *(est.)* are the plan's original hand estimates. They are
hand estimates: textbook ranges and stopping powers for PVT and Ar.
Phase C2 replaces them with Geant4. Kinematic numbers are exact: two-body decay,
m_X = 16.9 MeV, Q = 20.578 MeV.

This plan picks up deferral **D6** of `sept26_prelim_analysis/PLAN.md` ("energy
calibration and calorimetry … what unlocks the invariant mass … a project, not a
task") and **D7** (liquid gain vs position). It also picks up the 2026-09-07
STATUS decision "no invariant mass", which was taken because there was no
calibration.

---

## 0. Bottom line (expected, before any analysis)

1. **We never had a calorimeter for the hard lepton, and that does not depend
   on the liquids.** Behind each MM, the stack is 3 mm PVT (wall), then 20 mm
   PVT (plastic), then one ~18 mm LAB cell. That cell is 21.2 mm outer
   (`MX17_Full_Geant` `ls_slab_thick_cm`), not the 4 × 15 mm the early design
   had. The full stack, liquid included, stops an electron of about 8–9 MeV
   *(est.)*. Without the liquid it stops about 5 MeV *(est.)*. A 10–15 MeV lepton
   leaves about 4 MeV in the plastic and goes on. **What we have is a range
   telescope with dE/dx sampling, not a calorimeter.**
2. **At wide opening angles the X17 soft lepton is in exactly the range the
   plastic stops.** At θ_open ≥ 140° the soft leg has T = 3.9–4.8 MeV, and at
   ≥ 160° T = 3.9–4.1 MeV (table §2). After about 0.5–1 MeV of upstream loss
   *(est.)*, it reaches the wall with about 3–4 MeV and mostly stops in wall +
   plastic. For those pairs the plastic measures one leg's full energy. X17
   then predicts a **line** E_soft(θ_open), where IPC gives a **continuum**.
   That is a discriminant independent of the opening-angle peak, and the main
   hypothesis this plan tests (C3).
3. **Even without that, a per-leg "minimum energy" cut looks clean.** Every X17
   lepton has T ≥ 3.9 MeV. The non-capture backgrounds are below that:
   - ²⁸Al β, endpoint 2.86 MeV;
   - Compton electrons from H (2.2 MeV) and most structural capture γs;
   - random low-energy coincidences.

   A cut "both legs deposit ≳ 2.5 MeVee in the plastic" should keep almost all
   X17 and remove most of those backgrounds. Exactly how much is C3's job.
4. **The liquids matter less than they seem for energy, but they would have
   told us which leg is the hard one.** At large angle, the soft leg (stopping
   at about 3.5 MeV) and the hard leg (MIP-like, about 4 MeV Landau) leave
   **similar plastic deposits**. The liquid breaks that degeneracy, since only
   the hard leg reaches it. The salvage study (C4) asks whether any part of any
   liquid can still do that.
5. **MM dE/dx will not measure energy.** Electrons at 1–15 MeV sit on the Ar
   ionisation plateau to within ~10 % *(est.)*, and one 30 mm gap gives
   ≳ 30 % resolution *(est.)*. It can plausibly do two other things:
   - flag **two unresolved MIPs** (external conversions, close same-chamber
     pairs);
   - flag **heavily ionising** tracks (slow electrons below ~300 keV,
     recoils).

   Both are worth knowing for the next iteration. The cosmic test (M2–M3)
   should give a clean yes/no on the 2-MIP flag. **Prerequisite:** the stored
   `q_sum`/`q_total` cannot be used (NNLS runaway, OCTOBER O10). We need a
   model-independent raw charge first (M1).

---

## 1. Inventory — what each arm actually has

The per-arm stack, outward from the capsule. Levers are from `ntof_scint_stack`
`geometry.csv` (run_config.json). Status is from the scint-stack report of
2026-10-06 (`/media/dylan/data/x17/scint_stack/report.html`).

| layer | thickness | ≈ g/cm² | an electron stops if T ≲ *(est.)* | through-going MIP | read-out | status / calibration |
|---|---|---|---|---|---|---|
| capsule wall + ³He (500 bar) + air | TBD from G4 | ~0.2–0.4 *(est.)* | — | — | — | upstream loss and straggling; take from G4 |
| MM window + 30 mm drift gas + PCB (FR4/Cu/Rohacell) | 30 mm gas; PCB ~0.15 g/cm² | ~0.15 | ~0.3 MeV | ~3 keV in gas | DREAM, ZS | gain −34 % under beam vs cosmics (`inbeam_through_goers`); `q_sum` unusable (O10) |
| SiPM wall | 3 mm PVT + tape/foil | ~0.33 | ~0.7 MeV cumulative | ~0.6 MeV | 4 groups × 2 ends | 75–91 % eff (unbiased); MIP response flat to 2–2.5 % run-to-run; WALA ~30 % low, WALD g3 ~35 % low |
| plastic | 20 mm PVT, 2 bars 200×300 mm | 2.06 | ~5 MeV cumulative | ~3.6–4 MeV *(est.)*; **measured 3.0–3.4 MeVee** | 1 PMT per bar | keVee scale from the 2026-07-28 two-source calibration (`mx_july_beam_qa/calib/srccal_energy_calib.json`); ~1 % run-to-run; face maps ±10 % (B bar 2 +20 %) |
| liquid | ~18 mm LAB in 21.2 mm vessel, 451×450 mm | ~1.6 | ~8–9 MeV cumulative | ~3 MeV | 1 PMT, amp + area | **A, D**: answer only near their +u PMT edge (9 % vs 0.2–1 %); **C**: only above ~8 MeVee in the plastic; **B**: roughly uniform but low (4 %). "Response = punch-through × efficiency", never separated |

Not measured yet, needed below:
- the plastic and wall **saturation** level in MeVee;
- the **gain vs time since flash** of the plastic PMTs;
- the trigger's **plastic threshold in keVee** (PLAS_THR is 112–151 mV);
- why the through-going plastic deposit reads 10–20 % below the expected MPV *(est.)*.

---

## 2. What energy information is worth having — the physics targets

**X17 kinematics** (exact): E_X = 20.560 MeV, γ = 1.2165, β = 0.5695.
- Lepton kinetic energy in the lab: **3.93–15.61 MeV**, flat in between.
- θ_min = 110.5°.
- 9 % of legs have T < 5 MeV, 18 % have T < 6 MeV.
- At a given θ_open the energy split is fixed (two-body decay of a fixed mass):

| θ_open | 111° | 120° | 130° | 140° | 150° | 160° | 170° | 178° |
|---|---|---|---|---|---|---|---|---|
| soft T [MeV] | 8.95 | 6.52 | 5.44 | 4.79 | 4.38 | 4.12 | 3.97 | 3.93 |
| hard T [MeV] | 10.59 | 13.02 | 14.10 | 14.75 | 15.16 | 15.42 | 15.57 | 15.61 |

So **near θ_min both legs punch through**: no energy information, only "both
above threshold". At **wide angles the soft leg stops in the plastic**, and its
energy is a prediction. Opposite-arm pairs are by construction the wide-angle
ones. Adjacent-arm pairs straddle 110–150°. The acceptance-weighted split
between them comes from G4 (C2).

**Backgrounds and what energy does to each:**

| background | lepton energies | energy handle |
|---|---|---|
| IPC of ³He(n,γ) 20.58 MeV (irreducible) | same sum as X17, continuous sharing. **Spectrum from `ipc_born.py` (M1 + E0 mix), not `pair_physics.py`** (CLAUDE.md) | soft-leg energy at fixed θ: X17 line vs IPC continuum (C3b) |
| ²⁸Al β (activation) | ≤ 2.86 MeV | per-leg threshold (C3a) |
| capture-γ Compton / external conversions (H 2.2, Al 7.7, structural) | mostly ≲ 5 MeV, sum ≲ 8 MeV | threshold. Conversions are also close pairs, so the MM 2-MIP flag applies (M4) |
| through-goers (cosmic or beam) | MIP in both arms, ~3.5 MeVee each | **no** plastic handle (looks like X17 at 180°). Capsule vertex, collinearity and timing do this (`ntof_cosmics/README.md`) |
| random coincidences | spectrum of the singles: plastic|wall only 57 %, median 1.6 MeVee | threshold |

**Observables this can buy, in order of ambition:**
- **(a)** per-leg minimum energy (a cut);
- **(b)** soft-leg energy vs θ_open (a 2D discriminant, X17 line vs IPC
  continuum);
- **(c)** full kinematics for stopped-soft-leg pairs. With E_soft measured and
  the capture Q fixed, E_hard = 19.56 MeV − E_soft − losses, so
  m² ≈ 2 E₁E₂(1 − cos θ). That is an invariant-mass estimate for a subset.
  Whether it beats θ_open alone depends on the resolution (C3c).

---

## 3. Scintillator calorimetry — feasibility steps

Each step lists its question, its inputs, what it produces, and the result that
would stop the line of work (*kill*).

### C1 — Plastic and wall energy scale *for electrons*

The plastic keVee scale exists (2026-07-28 source calibration, flat to ~1 %).
What is missing is proof that it holds for **electrons of 1–5 MeV at the
production operating point**, linearly, across the face, and at late times
after the flash.

1. **MIP check with a known path.** Use cosmic muons on run_149 and run_103.
   These are the only beam-off runs with n_TOF scintillator data, already
   clock-matched by `ntof_cosmics/clock_match.py` and joined in
   `cosmic_wall_scale.py`. Also use the in-beam through-goers
   (`inbeam_through_goers.py`). Take the MPV of the plastic MeVee × cosθ, with
   the path from the MM track (waveform reco, k-calibrated). Compare with the
   G4 MPV for 20 mm PVT.
   - **Decides:** whether the measured 3.0–3.4 MeVee vs ~3.6–4 *(est.)* is a
     scale error, a path or angle error, or light collection.
   - **Kill:** a scale discrepancy above 20 % with no explanation. Calorimetry
     would then wait for a fresh source calibration that we cannot take.
2. **In-situ electron endpoint: ²⁸Al β, Q_β = 4.64 MeV, β endpoint 2.86 MeV.**
   Activation decays are late (s–h; ~half of the sim DriftGas hits) and are
   in the data between flashes. Select MM-tagged single tracks late after the
   flash, then fit the plastic deposit spectrum's endpoint against G4
   (²⁸Al β + 1.78 MeV γ). This puts an electron energy point in the same
   detector at the same time.
   - **Kill (soft):** no visible endpoint above the trigger threshold. Then
     fall back to Y-88 Compton edges + MIP only.
3. **Linearity / saturation.** Find where `psat` sets in, in MeVee per bar,
   from the high tail of the late-trigger plastic spectrum. Stopped soft legs
   need linearity to ≳ 6 MeVee, and Landau tails beyond that.
4. **Time since flash.** MIP-like plastic deposits vs t_since_flash
   (10 ms–80 ms) on the late sample. PMT gain sag after the flash is the
   suspect.
5. **Face non-uniformity.** Reuse the scint-stack response maps (25 mm cells).
   Turn them into a per-cell correction and give the residual non-uniformity
   as a resolution term.

**Product:** `calib_plastic_e.json` per bar: scale, linearity, per-cell map,
flash-time correction, saturation. Same for the wall, used as a dE/dx sample,
not as energy.

### C2 — Detector response matrix from Geant4

Use the existing full-sim geometry (`MX17_Full_Geant`, current per-arm stack
including capsule, PCB, wall, plastic and the 21 mm LS vessel).

- **Mono-energetic electrons, and positrons for annihilation γs**: T = 0.5–16
  MeV, from the capsule, over the arm's angular acceptance.
- Score E_dep in wall, plastic and liquid per leg, the exit flag, and MS angle
  at the wall.
- Fold in the measured light response: C1 scale, Birks for PVT, the C1
  non-uniformity, photostatistics.

**Questions:**
- E_dep(plastic) vs T: where does it plateau? What are the backscatter and
  escape tails?
- What is the **upstream loss and straggling** (capsule + MM)? This sets the
  floor on any soft-leg energy resolution.
- What is P(liquid fires | T)? That gives the punch-through threshold.

**Data validation:** the unbiased plastic spectrum of in-time MM-tagged tracks
(the scint-stack `unbiased` sample, median 1.6 MeVee) against G4 neutron-run
electrons (`neutrons_thermal_trig_2cm_nose`, already on EOS, with the < 100 ms
cut for ²⁸Al).

**Product:** response tables R(E_dep_wall, E_dep_plastic, liq | T, θ_inc).

### C3 — Pair-level sensitivity: is it worth anything?

Inputs:
- the existing pair sim `pairs_thermal_trig_2cm_nose` (10⁷, X17 + IPC 50/50,
  thermal capture vertices), passed through C2's response;
- **IPC re-weighted to `ipc_born.py`** (M1 + E0). The sim's IPC came from the
  older generator, so check which before using it. Re-weight per event in
  (E₊, E₋, θ), or regenerate.

Then, comparing against **opening angle alone**, the status-quo deliverable,
with the same Asimov-Z machinery as the trigger/acceptance studies:

- **(a)** per-leg threshold scan, E_dep(plastic) > E_cut on both legs. Measure
  X17 efficiency, IPC efficiency, and the low-energy background rejection, using
  the late-time beam data itself as the background sample.
- **(b)** the 2D distribution (θ_open, E_dep of the lower-deposit leg). Find
  where the X17 line separates from the IPC continuum, given the C2 resolution
  and the hard/soft degeneracy of §0.4.
- **(c)** mass reconstruction for stopped-soft-leg pairs (Q-constrained). Is
  its resolution better than the θ_open peak width for those pairs?

**Kill:** if (b)/(c) give < 10 % gain in expected Z over θ_open + (a), stop at
(a) and ship the threshold cut only. **Go:** apply (a) to the beam data and
compare the selected yield with expectation. Stop there; the X17 search itself
is not part of this plan.

### C4 — Liquid salvage: efficiency for MIPs, separated from punch-through

The scint-stack report could not separate "the particle did not reach it" from
"it did not see it". Cosmic muons reach it with certainty.

- On run_149 / run_103, take cosmic tracks through A and C (A–C line). Every
  one crosses that arm's whole stack, liquid included. On B and D use
  steep single-arm tracks. Map **liquid MIP efficiency and amplitude** over the
  face, 50 mm cells.
- Hypotheses to test with the maps, not assume:
  - an air gap or under-fill at the PMT side. A/D are horizontal (PMT at +u),
    B/C vertical (PMT on top), and the response pattern follows the
    orientation.
  - a dead or low-gain PMT (C).
  - PSA/threshold settings in the `v12_liqpileup` processing. Check amp vs
    `area_0` consistency, and the recording threshold relative to a ~3 MeV
    deposit.
- **Product:** per-liquid usable-region masks and a MIP efficiency inside them.
  Even a usable strip (A/D +u, behind the R bar) gives a **hard-leg tag** for
  pairs that hit it, which resolves the §0.4 degeneracy for that subset.
  Quantify the subset's size with C3.
- **Kill:** MIP efficiency below ~50 % everywhere. The liquids are then
  documented as lost, and the lesson goes to §5.

### C5 (stretch) — multiple-scattering soft/hard tag

At 4 MeV the scattering angle in the MM PCB is θ₀ ≈ 17°, against ≈ 5° at 15
MeV *(est., x/X₀ ≈ 0.011)*. At the wall's 97 mm lever that is ~30 mm against
~8 mm. The wall gives only 100 mm groups in u and ~52 mm along the bars in v,
so this works statistically at best: the wall-edge width split by plastic
deposit. It is cheap because the scint-stack per-track tables already hold
everything.

---

## 4. MM dE/dx — feasibility steps (cosmics first)

### M0 — What it can and cannot do (expectation to test)

| use | physics | expected verdict |
|---|---|---|
| lepton energy 1–15 MeV | Ar plateau, ≲ 10 % relativistic rise over the range *(est.)* | **no**, within any achievable resolution |
| 2 unresolved MIPs vs 1 | ratio 2 | **maybe**: separation ~1.5–2.5 σ, depending on whether truncated sampling beats the whole-gap Landau |
| heavily ionising (p, α, recoils, e⁻ ≲ 300 keV) | ≥ 1.5–100 × MIP | **yes**, if saturation and ZS are handled |
| gain monitor per run / cell / time since flash | MIP electrons are a constant source | **yes**: already used informally as `q_per_len` |

One 30 mm Ar gap holds ~300 primary electrons for a MIP *(est.)*. A single
whole-gap sample has the full Landau tail. A truncated mean over N depth slices
of 2.6 mm (60 ns at ~44 µm/ns, so N ≈ 11) might reach 25–35 % *(est.)*. The
slices are correlated through diffusion, shaping, and the resistive kernel's
delayed neighbour copies (τ ≈ 47 ns ≈ one slice), so "might" is the honest
word.

### M1 — A charge estimator we can trust

- **Do not use `q_sum` / `q_total` / `q_per_len`.** The NNLS fills unobservable
  depth bins without bound (STATUS 2026-10-02, OCTOBER O10). The guard there is
  "treat q_sum and q_total as unusable". `q_per_len` was used only as a gain
  proxy, as a median.
- **Estimator:** the raw charge from `decoded_root`. Take the
  pedestal-subtracted, common-mode-corrected ADC samples, summed over the
  strips in the track's road (geometry from the wft reco, not from hits) and
  over the drift window, per view: Q_x, Q_y, Q = Q_x + Q_y. Summing the road
  captures the resistive sharing, which conserves charge across strips.
- **Path:** use the geometric gap, path = 30 mm × sec θ, with θ from the
  k-calibrated reco. Do not use `drift_len_mm`, which rails. Require
  full-gap tracks (from the capsule, or through-going).
- **Systematics to quantify:**
  - ZS censoring of small samples. Production and cosmics share the post-23 July
    noisy configuration, so they are comparable to each other but not to
    anything earlier.
  - DREAM saturation.
  - Q_x/Q_y sharing ratio vs position.
  - Strips in the road that are dead or masked (A-x connector 8 on run_79/80).
- **Also (input from O10):** if O10's fix stores `q_obs`, compare it to the raw
  Q. Do not wait for it.

### M2 — Cosmic test, run_149 (21.8 h) + run_133 / 089 / 103

Through-going muons on the A–C line: 1 MIP, known path, two independent
measurements of the **same** particle.

1. **Q distribution per chamber.** Landau MPV and width vs sec θ: the MPV
   should scale like path, and the width should shrink slowly.
2. **Charge vs drift depth**, from the raw time profile summed over the road.
   Look for attachment loss along the drift (A dry; B/C/D carried ~0.8 % H₂O in
   July) and for the late-window cut-off.
3. **Gain map.** MPV per 20 mm cell. Compare with the efficiency maps (dead
   regions, D's bad cells, B's low efficiency).
4. **Same-muon A vs C.** corr(Q_A, Q_C) after the corrections. Residual
   correlation means a common-mode or shared systematic, not dE/dx.
5. **Resolution by estimator.** Whole-gap Q vs truncated mean over depth
   slices (keep the lowest 60–70 %) vs truncated mean over strips, for
   inclined tracks. Quote σ/MPV and the high-tail fraction.

**Product:** a cosmic dE/dx calibration per chamber (path, depth and cell
corrections) and the achieved resolution.
**Kill:** truncated resolution ≳ 45 %. Then 2-MIP separation is below ~1 σ;
document and stop the 2-MIP line.

### M3 — 2-MIP separation power

- **Data overlay:** sum the raw waveforms of two cosmic events whose tracks
  cross the same chamber within a few mm, then reconstruct as one. This gives
  true noise, true kernel and true Landau. Caveat: the inputs are ZS'd, so
  samples that would cross threshold only in the sum are missing. Quantify that
  with the next item.
- **Simulation:** `ntof_cosmics/g4_digi`, which injects G4 steps into quiet
  run_145 raw ADC through the production reco. Generate 1 and 2 electrons/muons
  with exact truth. Validate the 1-MIP Q against M2 first.
- Report the efficiency for 2-MIP at 10 % 1-MIP mis-tag, as a function of the
  pair's separation. This connects to the two-track work: below ~2–3 mm the reco
  sees one track (SAME_CHAMBER_PAIRS T1), and that is exactly where dE/dx
  would have to take over.

### M4 — Transfer to beam data

- **Gain under beam.** Muons cross ~34 % lower in beam than in run_149. Measure
  the beam-on gain per run, per cell and vs time since flash, from the bulk
  late electrons (median Q per path) and the in-beam through-goers. Apply on
  top of the M2 calibration.
- **Electrons vs muons.** Compare the late-time single-electron Q per path with
  M2's muons. The prediction is equal within a few %. A difference is either
  physics (low-energy electrons on the rising branch) or the estimator.
- **Look for 2-MIP populations** in the beam:
  - the high-Q excess over the M2 Landau shape, vs time since flash, per
    chamber;
  - whether it points at the capsule or at structure. External conversions of
    capture γs are close pairs, and their conversion points image the material.
- **Heavily ionising:** census of tracks at ≥ 3× MIP (early times, out-of-time
  tracks). QA value only.

### M5 — What a future iteration would need (write-up, from M2–M4)

- Whether dE/dx is limited by gap length, sampling (time bins, ZS), the
  resistive kernel's mixing, or gain stability.
- What a design with real 2-MIP or PID power would need: thicker or segmented
  gap, no ZS, finer time sampling.
- Separately and more importantly: the **calorimetry lesson** from C2–C4. It
  takes ~5 cm PVT-equivalent behind the tracker to contain a 10 MeV lepton,
  and ~8 cm for 15 MeV *(est.)*. A single 18 mm liquid cell was never going to
  measure the hard leg. Also record whether the dead-liquid cause (C4) was
  mechanical, so the next build avoids it.

---

## 5. Package, inputs, outputs

- **Package:** `ntof_calorimetry/` (this file is its first content). Modules:
  - `scint_ecal.py` (C1);
  - `g4_response.py` (C2; condor submit files under `condor/`);
  - `pair_sensitivity.py` (C3);
  - `liquid_salvage.py` (C4);
  - `mm_charge.py` (M1 estimator, shared);
  - `mm_dedx_cosmics.py` (M2–M3);
  - `mm_dedx_beam.py` (M4);
  - `make_report.py`. Per CLAUDE.md the report is generated, with relative
    figure links, and leads with the verdict.
- **Outputs:** `/media/dylan/data/x17/calorimetry/` (add a `calo` entry to
  `sept26_prelim_analysis/paths.py`). Large G4 products go to
  `/eos/experiment/ntof/data/x17/full_sim/calorimetry/`. Condor jobs write to EOS
  from inside the job and clean their scratch dir (global CLAUDE.md).
- **Reuse, do not copy:**
  - `ntof_scint_stack.extract` / `ana`: per-track scint tables, `plastic_kevee`,
    `liquid_kevee`, the trigger emulation, the response maps;
  - `ntof_cosmics.clock_match` and `cosmic_wall_scale`: cosmic ↔ slim join;
  - `ntof_cosmics/g4_digi`;
  - `sept26_prelim_analysis/ipc_born.py`;
  - `wft` reco for geometry.
- **Conditions:** every number is per arm *and* per run condition. Production
  and the cosmic runs are all post-23 July (noisy DREAM side), with the run_79
  HV/readout. Exclude run_79/80/81 for A-x unless masked, and run_68.

## 6. Order of work and decision points

Effort rough, in sessions. M and C are independent and can run in parallel.

| # | step | needs | effort | decision after it |
|---|---|---|---|---|
| 1 | C1.1 MIP check (cosmics + in-beam) | slim join (exists) | 1 | is the plastic scale right to ~10 %? |
| 2 | C4 liquid MIP maps on cosmics | same sample as 1 | 1 | any usable liquid region? |
| 3 | M1 raw-charge estimator + M2 on run_149 | decoded_root on EOS (condor) | 2 | resolution → is M3 worth it? |
| 4 | C2 G4 electron response | condor | 1–2 | upstream straggling vs needed resolution |
| 5 | C1.2–1.5 (²⁸Al endpoint, saturation, flash, maps) | 1, 4 | 1–2 | `calib_plastic_e.json` |
| 6 | C3 pair sensitivity (+ ipc_born reweight) | 4, 5 | 2 | **ship threshold only, or pursue soft-leg / mass** |
| 7 | M3 2-MIP separation | 3, g4_digi | 1–2 | is a 2-MIP flag useful? |
| 8 | M4 beam transfer | 3, 7 | 1–2 | — |
| 9 | report + M5 lessons | all | 1 | — |

Steps 1–2 are the cheapest and settle the two questions that most change the
scope: is the scale right, and is any liquid alive. Do them first.

## 7. What this plan cannot deliver, stated up front

- **No calorimetric energy for any lepton above ~5 MeV**, so no invariant mass
  for near-symmetric pairs. Those are the pairs near θ_min, where X17 peaks.
  Their energy information is "both above threshold", nothing more.
- The soft-leg measurement rests on upstream straggling and on telling which
  leg is soft. Without a liquid, the second is ambiguous whenever both
  deposits are ~3.5–4 MeV.
- MM dE/dx gives no energy. Its best case is a probabilistic 2-MIP / PID tag.
- Through-goers are not touched by any of this. They stay with the topology
  handles.

## 8. Coordination

- Another session works in this checkout on the is2_v1 re-pass. It has
  uncommitted changes in `wft/reco.py`, `build_tracks.py`, `campaign_tracks.py`
  and `make_stage2_campaign.py`. **This package must not edit those files.**
  M1 reads the reco outputs; it does not change the reco. If the is2_v1 re-pass
  is adopted, re-run the steps that use track angles (path length, the
  extrapolation in C1/C4) on its tables.
- O10 (NNLS guard / `q_obs`) and the two-track threads (`SAME_CHAMBER_PAIRS.md`)
  are the neighbours. M3's separation-vs-distance result belongs in
  `SAME_CHAMBER_PAIRS.md` when it exists.

---

## Results so far (2026-10-08)

Code: `mip_sample.py` (samples + layer crossings), `landau.py` (Bichsel MPV,
truncated Landau(x)Gauss), `scint_ecal.py` (C1), `liquid_salvage.py` (C4),
`make_report.py`. Outputs under `paths.spell('calo')`.

- **C1.1: the plastic keVee scale under-reads at the MIP by 3-24 %**
  (production line reads cosmic MIP at 0.76-0.97 of the Bichsel Delta_p
  3.41 MeV). Cause: the line through the 477/699 keVee edges extrapolated x5.
  The Y-88 1.6 MeV edge falls below it by the same amount on A and D. The
  response is linear in path to 1.45 MIP (~5 MeV). **Not a kill.** Product:
  `c1/calib_plastic_e.json`, MIP-anchored mV/MeVee per bar, 4-7 %, beam-off
  cosmics. Trigger thresholds on that scale: 2.1-2.9 MeVee. Arm B not
  measured.
- **Fit gotcha:** on A the trigger threshold is at 0.85-0.9 of the MIP peak.
  Fit with per-event truncation at thr x cos (`landau.fit_trunc`) and a
  sigma/MPV prior; a free fit runs to a low-MPV, wide-sigma solution.
- **C1.3:** `satuflag` is never set. Clipping starts at >= 15 MeVee. Irrelevant.
- **C1.4 blocked:** the beam plastic peak rises ~10 % from 10 to 80 ms, but
  the beam through-goers do not penetrate (below), so population and gain
  cannot be separated. An in-beam energy reference is needed: 28Al endpoint
  (C1.2), or the H-capture Compton edge.
- **C4: the liquids are threshold-limited, not dead.** MIP in 18 mm LAB =
  20-28 mV (6-9 mV/MeV, i.e. 0.13-0.19 of the source scale). The n_TOF ZS
  threshold of 16-18 mV is therefore 1.8-2.8 MeV. The source "edges" were not
  Compton edges (LIQA Cs = Y-88 amplitude). MIP efficiency: A 56 % (65-86 % on
  u >= 0, the PMT half), D 36 % (50-70 % at u >= 50), C 14 % (falls toward its
  top-mounted PMT: a bubble?). B not measured. The signal was never recorded,
  so reprocessing recovers nothing.
- **By-product:** in-beam A-C "through-goers" at >= 10 ms reach the liquid at
  0.10 +- 0.04 (A) / 0.04 (C) of the position-matched cosmic expectation. They
  are mostly **not penetrating muons**, which matters for the through-going
  background picture (`ntof_cosmics/README.md`).
- **Plastic thickness is a single-point dependency.** 20 mm per the Geant4
  geometry (`SimConfig.hh`, corrected 2026-07-20 from 2.5). The run_config
  text and `pss_mip_calib` still say 2.5 cm. At 25 mm the MIP is 4.30 MeV,
  the MIP-anchored scale drops 21 %, and the source line reads 23-40 % low.
  Measure a bar.
- **M1 done** (`mm_charge.py`): the raw road charge over the waveform reco's
  own corridor, all 20 samples, with the common mode taken from each block's
  channels OUTSIDE the road (the stock CNS eats a steep track's charge). The
  off-road control averages -0.06 % of the signal. Inputs: the 14 run_149
  sub-runs whose waveforms the in-situ work staged in
  `~/scratch/ntof_insitu/beam/run_149` (A, C only).
- **M2 done; the kill condition is met.** Whole-gap FWHM/MPV 0.95
  (sigma_eq 40 %), truncated means 47 % (no gain: shaping and the kernel
  correlate samples). Best-case 2-MIP flag is ~40 % efficient at 10 % 1-MIP
  mis-tag. The 2-MIP line stops; M3 is not run. Chamber C (v = 28 um/ns)
  needs > 1.1 us for the gap, so its deep charge falls off the 1.2 us window
  on ~94 % of tracks; only the plateau estimator covers it. Gain spread per
  40 mm cell: 19 % on A, 39 % on C. **No attachment visible** with the
  unbiased (absolute, all-track) depth profile; a per-track-normalised
  in-window version faked a 3x fall on C.
- **C2/C3 staged, not submitted:** `condor/submit_c2.py` (single e-/e+,
  0.5-16 MeV, 2 angles) and `condor/submit_c3_reduce.py` (the pair sim ->
  per-event-arm deposits), both through `condor/reduce_edep.py` (prompt hits,
  t < 1e8 ns). They run on lxplus against
  `/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant`. Test the reducer on a
  200-event file first: it has not been run on real sim output.
