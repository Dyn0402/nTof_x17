# mpgd26 — plan for the Status section

**Drafted 2026-08-09.** Replaces the `[fill in]` placeholders on slide 17 and the
last bullet of slide 18. Everything below is sourced; the provenance column says
where each number comes from and how solid it is.

---

## 0. The framing decision that drives everything else

The full story is: *the physics case collapsed twice (branching ratio, then
aluminium capture), and the DAQ could not survive the environment it was put
in.* For an MPGD audience, only one of those three is theirs — **the DAQ
failure in a novel environment**, and what we did about it. That is also the
part that is unambiguously ours to tell: it is instrumentation, not a
collaboration physics result.

So the recommended shape of the section is:

1. **What the environment does to the readout** (measured — the interesting part)
2. **What we did about it** (trigger + operating point chosen against a measured
   dead-time map — the transferable lesson)
3. **What we get out of it** (target imaging with reconstructed tracks)
4. **One honest sentence about the physics reach** (see §5 for options)

The physics reversals get *one line each*, stated as facts about the reaction
and the capsule, not as a verdict on the experiment. That keeps you inside what
you can say without pre-empting the collaboration, and it is also the correct
weighting for this audience.

---

## 1. The DAQ-saturation story — what is measured

### 1.1 What the flash does to a DREAM channel

| Fact | Value | Source |
|---|---|---|
| Prompt flash rails every channel | +rail 4095 ~0.4 µs, then −rail 0 ~0.4 µs, ~2 µs railed total | `daq:analysis/dream_saturation_7-12-26/DREAM_GAMMA_FLASH_SATURATION.md` §3.1 |
| ADC returns to a *flat* baseline in ~3 µs | but that is the shaper AC-coupling, not recovery | ibid §4(iii) |
| The real tell is **absence of noise** | dead ≈ 2 ADC scatter, alive ≈ 50–130 ADC | `daq:analysis/flash_recovery/*` |
| Mechanism (best theory) | CSA pinned against its rail while input current exceeds the AICON feedback limit I_max ≈ 9–90 nA; zero small-signal gain while pinned ⇒ no tracks *and* no noise | ibid §4(i)(ii) |
| Ruled out as the fix | mesh charge-injection compensation — trades +rail for −rail, total railed time unchanged | ibid §3.3 |

The "flat baseline is not proof of recovery" point is the single best slide-line
in the whole study, and it is exactly the kind of thing this audience will not
have seen.

### 1.2 Recovery time vs gain — **the headline measurement**

`flash_random` mode: flash defines t = 0, a Poisson pulser then fires random
triggers across the ~30 ms gate; each probe measures per-channel baseline
scatter, so recovery = when the noise comes back and stays back.

run_57 (Ar/iC₄H₁₀ 90/10, ³He, no filter), drift 800 V —
`daq:docs/flash_recovery_run57_HV_map_2026-07-20.md`, full grid in
`analysis/flash_recovery/run57/metrics_run_57_perdet.csv`:

| resist HV [V] | A [ms] | B [ms] | C [ms] |
|---:|--:|--:|--:|
| 580 | 21.5 | 21.5 | 24.9† |
| 560 | 13.9 | 10.4 | 16.1 |
| 548 | 4.3 | 4.3 | 7.7 |
| **540** | **5.0** | **1.2** | 8.9 |
| 532 | 2.4 | 0.9 | 3.7 |
| **524** | 1.0 | 0.5 | **2.4** |
| 520 | 0.9 | 0.5 | 0.9 |

† 24.9 ms = top bin = "does not recover inside the gate". Det D is
window-limited (≥22 ms) above ~540 V and only becomes resolvable below ~520 V.

Reproduced independently on run_19 (different detectors-at-max, circuit removed)
and cross-checked against run_42 in the 520–560 V overlap to within ~1 log-bin.

**This is the plot to show.** It is a clean, monotonic, two-decade dead-time
curve against a single detector knob, and it is what every operating decision
downstream was made against.

### 1.3 Why that matters: it lands on top of the signal window

- Thermal-neutron arrival peaks at **~5.3 ms** (Geant4 `timedist_2cm`), and on
  real beam there is **nothing below ~1 ms** — earliest track 0.993 ms, 75 of
  12 458 events under 1 ms (`ntof_tracking/RUN79_PRELIM_2026-07-30.md` §6b).
  **Correction to that document:** its "rate peaks at ~4–7 ms" does not survive
  1 ms binning of the same parquet — the distribution turns on at 1 ms and
  decays monotonically, with 29 % of all recorded tracks inside 3–8 ms. The
  figure built here says the honest thing instead.
- At the production operating point the slowest recovering chamber is still
  coming back *inside* that peak.

That is the whole failure in one sentence: **the dead time the flash imposes is
the same size as the physics window it precedes.**

---

## 2. Quantifying the charge — first pass, and how to firm it up

This is the number the talk is missing. Three independent handles; #1 is done
below, #2 and #3 are proposed.

### 2.1 Handle A — HV supply current (DONE, first pass, 2026-08-09)

The resistive-layer supply carries the avalanche ion current, so its average
current *is* the charge delivered to the amplification stage, integrated over
everything the pulse does. Beam-on vs beam-off at the identical setpoint:

- beam-on: run_158 (production, 2026-08-09 08:10→13:50, 20 382 monitor samples)
- beam-off: run_157 + run_159 (cosmic reference runs the same day, same HV)
- beam pulse rate over the same window: **0.302 Hz** (6 152 pulses > 100e10 in
  20 398 s, from `slow_control/beam_intensity/`; median 847e10 protons)

| det | resist V | I beam-off [µA] | I beam-on, mean [µA] | ΔI [µA] | **Q per pulse** |
|---|---:|---:|---:|---:|---:|
| A | 540 | 0.016 | 0.059 | 0.043 | **143 nC** |
| C | 525 | 0.008 | 0.039 | 0.031 | **102 nC** |
| D | 520 | 0.708 | 0.874 | 0.166 | 551 nC (leaky — see below) |
| B | 540 | 2.13–2.86 | 2.36 | — | unusable (2 µA standing leakage that drifts) |

Take **~100–150 nC per beam pulse per chamber** as the first-pass number, from
the two clean channels.

What that means for the front end:

- Spread over a chamber's 1 024 DREAM channels (2 FEUs × 512): **~140 pC per
  channel, average.** The DREAM CSA full scale is **50–600 fC**. So the average
  channel is asked to swallow **~2×10²–3×10³ × full scale**, and the channels
  under the beam spot see far more than the average.
- Areal: ~0.1 nC/cm² per pulse over the 398.6 × 362 mm active area.
- Primary ionisation (needs the gain at this working point — the weak link):
  at G ~ 2×10³ that is ~70 pC of primaries ≈ 4×10⁸ electrons ≈ 10⁵ MIP-crossings
  per pulse per chamber.

**Caveats to check before this goes on a slide:**

1. *Monitor response.* The CAEN readback is sampled at ~1 Hz and 27.8 % of
   samples sit above baseline on **both** A and C. With a 3.3 s pulse period
   that is consistent with the monitor smoothing each burst over ~1 s, in which
   case the sample mean does recover the time-average current — but this should
   be verified against the supply's actual imon integration, not assumed. If the
   monitor instead reports a short instantaneous average, the quoted Q is a
   **lower bound**.
2. *A and C agree to 40 %*, which is reassuring but is also two chambers at
   different HV; a proper statement wants ΔI vs resist HV across the run_57 /
   run_67 scans, which would let you plot **charge delivered vs recovery time
   on one axis** — the single most persuasive figure this section could carry.
3. The current integrates flash **and** thermal tail. It does not by itself say
   what fraction is prompt.

### 2.2 Handle B — the single-channel n_TOF digitiser (**DONE**, 2026-08-09)

> **Done in a parallel session the same day** — `ntof_processing/mm_flash/`,
> published at
> <https://dylan-neff.web.cern.ch/notes/ntof-micromegas-gamma-flash.html>.
> **The chamber returns below threshold 0.87 µs after the flash peak** and
> delivers hits at the first instant the DAQ allows, so the millisecond dead
> time is a DREAM-chain property and nothing in the gas, field or amplification
> stage has a ms time constant. Median flash charge 930 pC into 50 Ω (224302).
> This became slide **D1b**, and it is the slide that makes the section about
> MPGDs rather than about a DAQ. Its own caveats: chamber identity is
> undecidable from the data, the gas was Ar/CF4/Iso 88/10/2, and 930 pC assumes
> direct 50 Ω termination — and it is *not* directly comparable to the 142 nC
> above (one electrode over ~1 µs vs a whole chamber over 80 ms).

The original reasoning, kept because the remaining gap is still real:

The premise of the whole story ("in single channels on the n_TOF DAQ it was
fine, in DREAM it saturated") is directly measurable: **MMA/MMB** are micromegas
channels in the n_TOF DAQ, S014, **1 GS/s, ~504 mV full scale**, runs
224297–224339 (2026-07-05→09). MMB is the live one (−4.0 mV threshold, real
zero-suppressed blocks out to ms). Raw waveforms survive for **224302, 224325,
224327** (~11 800 bunches). See `ntof_processing/NTOF_MICROMEGAS_SIGNALS.md`.

Integrating an unsaturated flash pulse over 50 Ω gives the charge on that
electrode *at waveform level*, on the same detector, in the same beam. It also
gives the time profile — which is what separates the prompt flash from the tail
and tests the "sustained current pins the CSA" mechanism directly.

**Blocker:** nothing has yet correlated MMB with the DREAM data — we do not know
what electrode it was on, at what gain, or under what HV. That is the first
question to answer, and it is a couple of hours of work, not a campaign.

### 2.3 Handle C — Geant4 (proposed, cross-check)

The full sim already transports the EAR2 flux through the real geometry and
knows the drift gas volumes. An energy-deposition-in-DriftGas histogram vs time
since flash, × W = 26 eV, × gain, is an independent prediction of both the
charge and its time profile. It would also say how much of the load is the
prompt γ flash versus the neutron-induced tail — which Handle A cannot.

---

## 3. What we did about it — the operating point

This is the "we know how to run in this environment" slide, and it is a real
result rather than an apology.

- **Deliberately below the efficiency plateau.** Production is drift 700 V, resist
  **A 540 / B 540 / C 525 / D 520 V** — chosen off the §1.2 dead-time map, not off
  the gain curve. Every ~10 V of resist buys a few ms of blindness.
- **Trigger tuned into the thermal peak.** run_67 scanned plastic threshold ×
  drift × resist and found efficiency *rises* toward lower threshold in every
  time window (1–10 / 10–30 / 30–80 ms), so the low threshold was taken:
  **0.90 MIP**, wall 0.5 MIP (`analysis/July_HV_Scan/run67_scan/compare/recommendation.md`).
- **FEU watermark forced to Hwm 1/Lwm 0**, measured to cut the 1–10 ms
  acceptance-comb CV from 1.50 to 0.51 for −10 % triggers (run_158 config).
- **Readout window measured, not assumed:** latency 27 puts the drift onset at
  sample 2, 20 samples × 60 ns holds 95 % of the drift charge (run_78 latency scan).
- **Result:** ~1.65 TB of production statistics banked since 2026-08-02 across
  runs 120–158, still running as of 2026-08-09, with beam-off cosmic reference
  runs interleaved (157, 159).

The honest framing: *the operating point is a dead-time optimum, not an
efficiency optimum, and we can show the map we chose it from.*

---

## 4. Target imaging — what to show

**Already exists and is good:** `ntof_tracking/run79_wall_segment_gif.py` →
`…/wft/run_79/stat090_0000/mx17_A/wall_segment_tour/wall_segment_tour_all.png`
(+ GIF). Waveform-first tracks on beam data, coloured by which SiPM wall segment
fired, in the 3-D model. The four bundles are **251 mm apart at the wall and
9 mm apart at the target plane**, against a label-shuffled null of 28 mm / 3.2 mm.

Supporting numbers from `RUN79_PRELIM_2026-07-30.md`:

- position↔angle pointing slope **0.80× expected** with an arm-A scintillator
  tag, **0.00×** for the other-arm null (§3)
- wall-segment ordering correlation **−0.98**, four separated peaks (§4)
- angle scale consistent with unity to ~10 % once the beam's non-radial
  dilution is divided out (§5.2)

**Two things to fix before it is talk-ready:**

1. It is **explicitly marked PRELIMINARY and "nothing here is quotable"** — the
   in-situ calibration of `TRACK_PLAN_08` §6 (template, sharing kernel,
   diffusion, v_drift) has not been done. Either do that, or caption the slide
   as a demonstration of the chain rather than a measurement.
2. It is **one arm, one sub-run, 3 of 13 file tags of run_79** — none of the
   1.65 TB of production data has been reconstructed yet. A four-arm
   back-projection over even a slice of run_120–158 would turn "the fans
   converge" into an actual **image of the capsule**, which is the thing you
   said you want to end on.

Also worth a caveat line on the slide: the convergence is an **ensemble**
statement about medians. Per track, X at the target plane has median −23 mm with
IQR [−46, −4]; only 15 % land within 10 mm of the beam axis. A viewer who reads
the waist as per-track pointing resolution is reading it wrong.

---

## 5. The physics reach — three framings, pick one with the collaboration

The facts, all from our own simulation work, none of them secret:

| Fact | Number | Source |
|---|---|---|
| At thermal, n+³He goes to ³H+p, not ⁴He(g.s.) | σ_np = 5333 b vs σ_nγ = 54 µb ⇒ **σ_nγ/σ_np ≈ 1.0×10⁻⁸** | ENDF / Wolfs 1989; measured in G4 at 1.0×10⁻⁸ |
| Self-shielding caps it further | (1.1–2.3)×10⁻⁴ IPC/pulse vs the table's 1.21×10⁻² — **×50–100** | `MX17_Full_Geant/docs/report/thermal_note.pdf` |
| X17 yield at thermal | **0.035 produced/day → 0.012 MM-acceptance pairs/day** | `.claude/al_pair_background/VERDICT.md` |
| Aluminium capture on the capsule dominates the trigger | **97 % of trigger legs are Al γ**, 122 legs/pulse at 0.5 MIP | `analysis/trigger_provenance/` |
| Al pair background | 5.95×10⁶ produced/day → **6.5×10⁵ MM pairs/day**, i.e. ~5×10⁷ : 1 over X17 | `al_pair_background/VERDICT.md` |
| It is not hopeless *in principle* | a total-pair-energy cut at 13 MeV works **iff** σ(p)/p ≲ 30 % per lepton | ibid |

Three ways to say this on one slide:

- **(a) Fully open.** "At thermal the ⁴He channel is 10⁻⁸ of (n,p), and Al capture
  on the capsule produces 97 % of our triggers. The thermal X17 reach is
  ~10⁻² events/day. We are operating this as a detector demonstrator." — Correct,
  and the strongest version of the DAQ story, but it is a statement about the
  experiment's prospects and the collaboration should sign it off first.
- **(b) Factual, no verdict** *(recommended default)*. State the branching ratio
  and the Al capture as **properties of the reaction and the capsule**, note that
  they set the rate, and say the thermal window is "background-characterisation
  territory" — which is the phrase already used inside our own campaign
  documents. Let the audience draw the conclusion.
- **(c) Instrument-only.** Say nothing about yield; frame the whole section as
  "operating an MPGD tracker in the EAR2 flash environment", ending on the target
  imaging. Safest, and loses the most interesting tension.

**Recommendation: (b), plus have (a)'s numbers on a backup slide** so you can
answer the question if it is asked — it will be asked.

There is also a genuinely positive line available and it is true: *the Al pair
background is defeatable by a tracking-based total-energy cut at ≲30 % per-lepton
momentum resolution* — which makes "can this tracker measure momentum to 30 %?"
a real, forward-looking MPGD question to leave the room with.

---

## 6. The slides as built (2026-08-09)

12 main-flow slides replacing the placeholder, plus 3 backups. All seven data
figures regenerate with `../make_status_plots.py`.

| ID | Slide | Figure | Note |
|---|---|---|---|
| D0 | Where things stand | — | four bullets, the section's contract |
| **D1** | What the γ flash does to a DREAM channel | `status_flash_waveform` | rails +4095 → 0 → flat; the noise panel is the point |
| **D1b** | The chamber is fine. The front end is not. | `status_two_readouts` | **added late** — the same chamber on a direct 1 GS/s channel is usable 0.87 µs after the peak; exonerates the detector by measurement |
| D2 | How much charge are we talking about? | `status_charge_scale` | 142 nC · ×231 · 0.1 nC/cm² as stat tiles |
| D3 | Charge delivered, versus gain | `status_charge_vs_hv` | the gain curve, measured as charge |
| **D4** | And what it costs: milliseconds of blindness | `status_recovery_vs_hv` | ★ marks the production point; thermal band shaded |
| D5 | Dead time is set by charge, not by voltage | `status_deadtime_vs_charge` | **the new result** — three chambers, one power law |
| **D6** | We chose the operating point off the dead-time map | — | bullets + 1.7 TB / 4-of-4 stat tiles |
| D7 | What we record, and when | `status_track_rate` | the overlap of data window and dead window |
| **D8** | The tracks point back at the capsule | `target_pointing_fans` | ensemble caveat is on the slide |
| **D9** | Two facts about the reaction and the capsule | — | §5 framing option (b); carries a `flag` |
| D10 | What we take away | — | outlook, incl. what did *not* work |
| B1 | How the charge per pulse is measured | — | method + the three validations |
| B2 | Thermal-window rates, as simulated | — | §5 option (a)'s numbers, kept off the main flow |
| B3 | Why the trigger threshold went *down* | — | the run_67 efficiency table |

**Bolded IDs are the load-bearing six** if the talk is tight. D2/D3/D5 collapse
to a single number on D4's slide; D0 and D10 are the first things to go if the
summary slide already does that work.

---

## 7. To-do, in the order that de-risks the section

1. ~~ΔI vs resist HV off the scan sub-runs~~ — **DONE**, and it gave more than
   expected: charge and recovery measured on the *same* 31 sub-runs collapse
   three chambers onto one power law, t ∝ Q^1.2 (slide D5). Reduction in
   `ntof_july_analysis/flash_charge/`, which also reduced run_19/42/58/61/64/67
   and 79 — **five more scans sit in the CSV unexploited**, including a drift
   axis and a 95/5-vs-90/10 gas comparison.
2. **Confirm the imon method** (§2.1 caveat 1). Still the one systematic that
   bounds the headline number; a beam-off null and a 10× rate-scaling test both
   pass, but neither proves the readback preserves a short burst's integral.
   Cheapest route is a spec lookup. Full detail:
   `flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md` §4.
3. **Find out what MMB was connected to** (§2.2) — an unsaturated, 1 GS/s,
   waveform-level charge measurement with a *time axis*, which is the only thing
   that separates the prompt flash from the tail. Highest value per hour on the
   list.
4. **Decide the §5 framing with the collaboration** before D9 is shown.
5. **In-situ calibration of the run_79 reconstruction**, or keep D8's
   PRELIMINARY caveat visible.
6. *(stretch)* Reconstruct a slice of the production runs on all four arms and
   make the actual capsule image, rather than the four-fan convergence.
