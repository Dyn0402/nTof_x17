# The reconstruction basis: waveforms, not hits

**Decided 2026-07-28. This is the canonical statement; other documents link
here rather than restating it.**

## The rule

> **Never reconstruct position, angle or drift depth from `combined_hits` times.**
> A per-strip hit time is an *aggregate* of that strip's own charge and delayed,
> dispersed copies of its neighbours' charge. It is not a drift-time measurement
> and no threshold, CFD or matched-filter refinement makes it one.
>
> Hits remain legitimate for two things: **finding candidates** (which strips and
> which events to look at) and **QA/monitoring** (rates, amplitudes, occupancy,
> detection efficiency). Everything geometric comes from the waveforms via
> `wft/`.

## Why — the measurement

**Mechanism (revised 2026-11-05, `papers/paper2/hits_daq/P0_XBIAS_2026-11-05.md`).** A single time per strip
reads a strip whose waveform holds charge from *several drift depths*. Three things put it there:

1. **drift diffusion** (Magboltz-consistent D_T; up to ~0.8 mm at the cathode on a 0.78 mm pitch) — present in any
   µTPC, metallic or resistive;
2. **the prompt footprint** σ0 (avalanche ⊕ induction through the resistive layer and insulator ⊕ the 550 µm
   resistive-strip box; 0.30 mm in the model against a ~0.07 mm induction floor);
3. **RC spreading along the resistive strips**, which grows after landing (rc_D_y 2–7e-4 mm²/ns) and therefore
   moves charge across the strips of the **Y view only**; across them (X) there is none (`rc_D_x = 0`).

With the pass kernel and real events, hits emulated on the model windows reproduce the data's hit-ladder compression
(det3 X 1.21 vs 1.18 data, Y 1.43 vs 1.37; det2, det7 alike; det4 over-predicted). Switching ingredients off:

| det3 | full model | metallic (σ0 0.07, no RC, diffusion kept) | diffusion off | all off |
|---|---|---|---|---|
| X | 1.21 | 1.17 | 1.09 | 1.04 |
| Y | 1.43 | 1.22 | 1.25 | 1.05 |

So **most of the bias is generic same-time mixing — diffusion compounded by the footprint — and a metallic readout in
the same gas would keep most of it; the resistive RC spreading roughly doubles the excess, on Y only.** X is
effectively a metallic readout in this design. The 2026-07 wording, "29 % of each ±1 neighbour delayed by τ ≈ 47 ns"
(det3, kY = 1.375 on Y), was the fitted *copy kernel* of the R&D model: an effective description that lumps all three
effects, retired with that kernel. Do not quote it as the mechanism.

Why the ladder compresses: a leading-edge (or peak) time is pulled toward the earliest substantial charge on the
strip. Deep charge is the most diffused, so the strips near the cathode see earlier charge from shallower
neighbours and are pulled earlier; strips near the mesh see little mixing (and, through RC on Y, later charge).
The time ladder is compressed by **20–30 %** (hits against M3, det3: X 1.19, Y 1.37), which at
a fixed drift velocity reads as a track **~4° too steep**, and every strip lands
at the wrong depth. Measured on 600 det3 muons (`THREADING_DISPLAYS_2026-07-28.md`),
charge-weighted median |cluster − M3 reference line| against drift depth:

| depth [mm] | 3 | 9 | 15 | 21 | 26.5 |
|---|---|---|---|---|---|
| production (hits), common frame | 0.45 | 0.50 | 0.61 | 0.67 | 0.63 |
| production (hits), own t0 and v — what the old 3-D displays draw | 0.43 | 0.63 | 0.88 | 1.07 | 1.17 |
| **waveform-first** | **0.39** | **0.44** | **0.49** | **0.50** | **0.55** |

Both agree at the mesh, at the ~0.4 mm M3 pointing floor. The hits-based cluster
then walks away from the true track as it ascends the gap; the waveform-first
one does not. The comparison is not circular: the displayed cluster comes from a
line-free 2-D deconvolution and a sub-pitch free ladder, never from the fitted
track.

**This is estimator-independent.** Production rising-edge, leading-edge 20 %,
CFD and full matched-filter all show the same compression
(`WAVEFORM_FIRST_THREADING.md` §3) — because all of them read a *mixed*
waveform. The mixing has to be modelled forward or removed, not thresholded
away.

## What replaces it

The forward-model reconstruction (`WAVEFORM_FIRST_THREADING.md`, packaged in
`wft/`): the charge arriving in each 60 ns slice of drift, folded through the
measured per-plane impulse response and the sharing kernel, generates the whole
(strip × sample) picture, and that whole picture is fitted per event. The
neighbours' delayed copies become part of the model instead of contamination.

Validated: per-event angle σ ≈ 1.0–1.1° (the July claim that this is the
diffusion/charge-granularity physics floor was **withdrawn 2026-10-27**, model
audit F3: the floor is 0.30–0.35°, `rerun/model_audit/FINDINGS.md`) — against
1.58/1.53° for the SOTA hybrid and 4.97/5.74° for the raw ladder on the same
events; mesh σ ≈ 0.58 mm; ~70 % of medium-angle events thread < 1 mm over the
full gap. Portable: det2 and det4 both calibrate and reconstruct with their own
kernels.

**Prior art**: `wft/REFERENCES.md`. The model is a re-derivation, not a
borrowing, but the ingredients are twenty years old — Dixit et al. (2004) for
charge dispersion on a resistive anode, Galan et al. (2012) for the
resistive-strip transmission line, and the T2K ND280 ERAM analysis (Attié et
al., 2023) for a multi-channel waveform model of resistive charge sharing (leading pad = electronics response,
neighbours = that response convolved with the RC dispersion) -- but used there to **map RC**, with positions from the
pad response at the waveform maximum, not as a per-event reconstruction (`papers/paper2/hits_daq/DAQ_SURVEY.md`,
2026-11-05; the earlier "simultaneous multi-channel waveform fit, the closest published method" overstated it).
Credit them in any write-up; the part to claim is
solving the drift-depth profile inside the fit, which is what turns a sharpened
centroid into a micro-TPC. The µTPC literature this section argues against is
the ATLAS NSW resistive-strip work — cite it as the foil.

## Consequences already known

- **v_drift**: the hits ladder implies v = 47–50 µm/ns and the gap estimators
  read 31.5–34; the forward fit gives **36.6 ± 0.3 ± 0.9** at 1000 V, which
  matches Magboltz Ar/iso 95/5 + 0.8 % H₂O. Old v numbers derived from either
  ladder or gap carry this bias.
- **Drift gap**: gap-based estimators inherit −3…−7 % from assuming 29 mm. The
  charge-visible column, all five chambers on the RC-ladder kernel at the K = 22
  basis (X plane, ± 1 mm calibration systematic each): **det2 30.6, det3 27.9,
  det6 27.9, det7 27.5, det4 25.6 mm** — det4's is not usable (its two charge
  halves read 20.9 and 30.2 mm) and det7's is marginal. det3 additionally has a
  reproducible dished topography (`ANALYSIS_STATE_2026-07-31.md` §3.5).
- **Angles and depth-resolved anything** from the old chain are compressed.
- **Positions at the mesh and efficiency are much less affected** — the July-25
  significance-floor fix left det3 at 93.1 % within 5 mm with core σ 0.45 mm, and
  that still holds. Efficiency is a property of the analyzer's trigger, not of
  the fit.

## Migration status

Updated in the same commit as each ported analysis — a number is only trustworthy
if you know which basis produced it.

| analysis | basis | where |
|---|---|---|
| Threading / cluster displays | **waveform** | `waveform_first_threading/37_threading_displays.py` |
| Per-event tracking (angles, positions) | **waveform** | `wft/` — packaged, regression-tested against the R&D code |
| Angular resolution | **waveform** — det3, det2, det6, det7 | `mx_june_wft/03_angles.py` |
| Alignment | **waveform** — det3, det2, det6, det7 | `mx_june_wft/01_alignment.py` |
| Efficiency breakdown | **waveform** positions, hits detection | `mx_june_wft/02_efficiency.py` |
| Efficiency / resolution maps | **waveform** — det3, det2, det6, det7 | `mx_june_wft/04_maps.py` |
| v(E), gas fit | **waveform** | `waveform_first_threading/` scripts 19–21 (re-run under the RC-ladder kernel pending) |
| Gap / column maps | **waveform (RC-ladder, K = 22 basis)** — all five | `mx_june_wft/bench/gap_study.py` / the condor path; verdict in `GAP_STUDY_2026-07-30.md` as amended by `ANALYSIS_STATE_2026-07-31.md` §3.5, §10.4 |
| Hybrid tracking | hits — **superseded in accuracy**, do not extend | `mx_june_cosmic_qa/34`, `36` |
| Time resolution | hits + waveform (port pending) | `mx_june_cosmic_qa/42` |
| Sparks, charge balance, fringe field | hits — QA-level, unaffected | `mx_june_cosmic_qa/38`–`40` |

**det4 recalibrated beam-anchored (2026-08-05,
`mx_june_wft/BEAM_CONSTRAINED_CALIB_2026-08-05.md`).** The H4 beam kernel
observables reproduce on bench cosmics (±1 delay +60 ns, matched-window
charge budget — the kernel is the layer's, not the gas's), and a calibration
with the kernel, σ_p0 and v pinned to measured values beats the lost legacy
bundle: σ_θ 2.63/2.44° (was 2.73/2.58), implied-v spread 4.1/4.5. Same doc:
the plain-delay kernel shape is now measurably wrong (share_lp port is the
next structural gain), and the fitted-v-runs-high failure mode is
demonstrated with the kernel known.

**Fleet result (2026-07-29, `mx_june_wft/FLEET_2026-07-29.md`).** Against the
hits chain at its best, within 5 mm: det3 −1.3, det2 +0.1, **det6 +17.6,
det7 +14.2** points; core σ improves on every chamber except det3. σ_θ roughly
halves everywhere (det3 1.20/1.14°, det2 1.31/1.56°, det6 2.28/2.52°,
det7 1.96/1.71°) with bias consistent with zero, and the implied-v spread across
angle bins — the direct compression signature — falls from ~17 to 1.3–6.4 µm/ns.
det4 is still running.

**det3 gate result (2026-07-29, `mx_june_wft/DET3_GATE_2026-07-29.md`)**, both
chains through identical accounting: σ_θ **2.42/2.60° → 1.20/1.14°** with the
implied-v spread across angle bins falling from ~17 to 2.3 µm/ns (the
compression signature, essentially gone); position at parity (within 5 mm
93.13 → 92.15 %, core σ 0.448 → 0.472 mm); detection unchanged (`has_any`,
`spark_frac` identical). The position gap is multi-cluster seeding — the
"largest cluster wins" rule picks the wrong charge in ~5 % of events, where the
reference sits a median 37 mm outside the fit window — and a track-compatibility
seeder is the open follow-up.

**Update 2026-07-30.** The `wft/` fit itself was benchmarked and upgraded
(3 % model-error chi2 weighting, per-plane `w0`/`kw` angle constants, the
`share_lp` RC-ladder sharing kernel — the neighbour copy is a low-passed
template, measured directly; coarse pre-scan). det3 runs on it; the fleet
procedure is approved-but-not-run. Campaign record:
`mx_june_wft/RECO_BENCH_2026-07-29.md`; state + procedures:
`mx_june_wft/HANDOFF_2026-07-30.md`. The drift-gap question is resolved as
chamber geometry rather than gas or reconstruction
(`mx_june_wft/GAP_STUDY_2026-07-30.md`, narrowed by the 07-31 fleet maps: it is
det2 that reads full, and three chambers that read ~2.5 mm short).

**Update 2026-07-31.** The fleet was put on the RC-ladder production
configuration (`mx_june_wft/rollout_lp.sh`), which fixed det7 (σ_θ 7.05° → 1.98°
— its stored chain had been produced by a rejected calibration); the drift
column was re-measured on a deeper charge basis, which fixed det6; and three new
diagnostics (angle-optimality scan with a disjoint validation split, readout
framing corner, model residual audit) were run on the grid.

**Current state, and the audit entry point (2026-07-31):**
`mx_june_wft/ANALYSIS_STATE_2026-07-31.md` — what each chamber's numbers were
produced by, the full systematics register, and what an external reviewer should
attack. Read that before quoting any number from the tables above.

> **Merge note, 2026-08-06 — the two updates above describe a `wft/` that is no
> longer the one in this tree.** The RC-ladder kernel was implemented twice,
> independently: the July bench R&D, and the 2026-08-05 port (`8e52e69`) that
> the SPS campaign ran on. The port ships; the R&D copy is kept for reference at
> `wft/archive/rc_ladder_2026-07-31/`. The shipped model has **no** `MODEL_FRAC`
> (so no 3 % model-error weighting), no per-plane `kTauY`/`tau2_fac_y`, and no
> `sample_weights`. Every number in the 07-30 and 07-31 updates — including the
> gap column above — came from the archived kernel and does **not** reproduce by
> re-running today's scripts against today's `wft/`. Reconciling the two is
> still to do; see that directory's README for the feature-by-feature diff.

## Reading order for the evidence

1. `mx_june_cosmic_qa/waveform_first_threading/THREADING_DISPLAYS_2026-07-28.md`
   — the displays and the 600-event census (this document's table)
2. `mx_june_cosmic_qa/waveform_first_threading/WAVEFORM_FIRST_THREADING.md`
   — the full study: §3 estimator-independence, §12 physics floor, §15 the
   recommended reconstruction, §17–21 v and gap consequences
3. `mx_june_cosmic_qa/REFERENCE_TRACK_THREADING_REPORT.md` — the earlier
   systematics report that posed the question
