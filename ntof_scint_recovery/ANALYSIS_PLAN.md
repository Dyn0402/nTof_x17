# Scintillator recovery after the γ-flash: analysis plan and state (2026-10-21)

The question, from a collaborator: **how long do the scintillators take to recover after the n_TOF
γ-flash?** Answered in three layers. Each layer is a stricter definition of "recovered".

| layer | "recovered" means | status |
|---|---|---|
| 1. quick pass | the baseline is back within ±3 / ±1 mV of its pre-flash level | **done**: slide note <https://dylan-neff.web.cern.ch/notes/scint-flash-recovery.html> |
| 2. pulses | pulses are found again with their normal shape | prototype `pulses.py`; histograms for every run running on condor |
| 3. amplitude | a found pulse has its full amplitude (no gain sag) | first look below; the core of the full analysis |
| 4. variation | layers 1–3 per channel, per bar end, and along the bar | handles demonstrated below |
| **5. reconstruction (headline)** | a pulse is found and within 10 % of its true amplitude (≥ 90 %), gain back within 5 %; bias and resolution in mV, MeV and ns vs time | **done**: Round 5 below, <https://dylan-neff.web.cern.ch/notes/scint-flash-reconstruction.html> |

**Rules that hold throughout**
- Always split by beam intensity: parasitic (~4×10¹² ppp) and dedicated (~8.5×10¹²).
- Never pool across configuration epochs (below).
- Count only live channels: some runs had whole families unpowered.

## Round 5 (2026-10-21): THE HEADLINE — from when a pulse is reconstructed accurately, and how well

This is the final analysis Dylan asked for (2026-10-21). The headline is no longer "when is the baseline
back" but **from when after the flash a pulse can be reconstructed accurately, and how biased and how well
resolved (amplitude, energy, time) it is against time since the flash**. Slide note
<https://dylan-neff.web.cern.ch/notes/scint-flash-reconstruction.html> (`make_reco_deck.py`); report
`out_dir('reco')/report.html` (`headline.py`); every number in `out_dir('reco')/{headline,performance}.csv`.

**Three independent pieces**

1. **Injection** (`inject.py`, condor `lxplus/inject.sub` → `/eos/.../flash_tail_inject/`, one flash_tail file per
   production run, all bunches; `inject_analysis.py`). Family-template pulses of known amplitude and sub-ns time are
   added to the real unsuppressed head traces (walls to ~40–60 µs, plastics ~250 µs, liquids 100–200 µs after the
   flash), clipped at saturation, and reconstructed: running-median baseline → peak finder → template fit with a
   local linear baseline and a derivative term. Two reconstructions: the `pulses.py` single-pulse fit, and a
   **multi-pulse** fit (prominence finder + joint fit of every neighbour; the one quoted). Control = the same
   1.5–8 µs before the flash. Each injection is flagged clean/piled: piled if a real pulse ≥ 5 % of it lies within
   the template body (217 / 42 / 18 ns).
2. **Detector gain** (`gain_ref.py`). The injection cannot see a PMT/SiPM gain change. The amplitude spectra can:
   a gain change scales **every quantile by the same factor**, a change of particle mix does not. Quantile ratios
   q50…q99 over their 1–10 ms values, for plastic pulses behind a wall hit, plastic singles, wall bars.
   Plastic gain = mean of the coincident and singles q99, ± half their difference.
3. **Geant4** (`g4_mix.py` over the existing `full_sim/neutrons_fullrange` run, 5×10⁸ EAR2 neutrons;
   `g4_mix_analysis.py`). Particle mix and deposit spectra vs time since the flash, Birks-quenched light.

**Results (dedicated; parasitic in `headline.csv`)**

- **What arrives** (Geant4): fast-neutron proton recoils dominate until ~1.5–2 µs (the ~1 µs bump on the
  plastic/liquid traces is 2 MeV neutrons). From ~2 µs every deposit above 0.1 MeVee is e± from capture γ, and the
  spectrum keeps one shape to ~1 ms (95 % at 1.99 MeVee, the H-capture Compton edge). Rates within ×2 of the data at
  3–300 µs on the plastics, ×6 low after 1 ms; the data spectrum is harder (q95 behind a wall 4.6–5.3 MeVee vs ~2):
  the hall and the in-beam γ are not simulated.
- **Gain** (data): plastics 0.38 / 0.68 / 0.92 at 3 / 5 / 10 µs, within 5 % (± 4–5 %) from ~10–20 µs.
  All quantiles drop together before ~10 µs (gain); after ~20 µs q50 and q99 are back while q75–q95 stay 10–20 %
  low to ~0.3 ms (mix). This **replaces** the round-4 reading of the plastic light edge as an upper bound on gain:
  most of that edge deficit after ~20 µs is mix. Walls: no sag resolved (q99 1.07 → 0.97, ±3–8 %). Liquids:
  quantiles move ×1.4–2.9 at 2–100 µs; no gain reference.
- **Reconstruction** (injection, multi-pulse, ~1 MIP = plastic 200 mV ≈ 3.6 MeVee, wall end 30 mV ≈ 1 MIP,
  liquid 100 mV): found ≥ 99 % from ~1 µs (plastics, liquids), ≥ 95 % from ~5 µs (walls); median bias ≤ 1 %
  from ~2 µs (plastics, liquids), ≤ 4 % from ~5 µs (walls). **The limit is pile-up with real beam-induced
  pulses**: the fraction within 10 % is 0.90 at 2–3 µs on the plastics, falls to ~0.72 at 5–20 µs (the
  capture-γ pulse train is densest there), and is back to 0.92 by 200 µs; liquids 0.87–0.92 from 5 µs; walls
  ≤ 0.43 within the head (dedicated; 0.71 parasitic). Wall "noise" (high-pass rms) is 0.9 mV before the flash,
  15 / 6 / 2.5 mV at 10 / 20 / 30 µs: unresolved small pulses.
- **Beyond the head** (`occupancy_model`): good10 = exp(−a R(t)) with R the measured pulse rate, fitted on the
  injection (residuals 0.02–0.05 at ~1 MIP) and carried to 10 ms. Walls reach 90 % within 10 % at ~0.75 ms
  (dedicated) / ~0.27 ms (parasitic) for 1 MIP.
- **Headline times, ~1 MIP, ≥ 90 % within 10 % and (plastics) gain ≥ 0.95**: plastics ~160 µs (parasitic ~50 µs),
  liquids ~80 µs (~3 µs), walls ~0.75 ms* (~0.27 ms*), * from the model. Bigger pulses are earlier (plastic
  7 MeVee 100 µs, 14 MeVee ~10 µs, gain-limited). The flash tail on its own (clean pulses) stops mattering within
  0.5–8 µs for ≥ 1 MIP plastics and liquids.
- **Energy and time resolution** (σ68, ~1 MIP, dedicated): plastic 0.12 / 0.43 / 0.41 / 0.14 / 0.06 MeVee at
  2 / 10 / 20 / 100 / 200 µs (control 0.4 %); time 0.14–0.29 ns (control 0.07). Walls 0.2–0.3 MeV and 2.5–4.5 ns
  at 5–50 µs (control 2 %, 0.4 ns). Full grid: `performance.csv`.
- **Energy scale**: plastics 39–72 mV/MeVee (mean 56), muon calibration (6 bars) + 30–32 ms cosmics in this data
  (B bars; in-situ/calibration 0.95); wall ends 22–31 mV per MIP (0.47 MeV in 3 mm).

**What it does not rule out**: a slow change of the hard part of the data spectrum would read as gain (Geant4
doesn't reproduce it); before ~2 µs gain and mix can't be separated; injection only inside the head block (the
model beyond); family template (per-channel shapes cancel against the control); pile-up scales with the rate.

## Data

| | where | made by |
|---|---|---|
| raw blocks, 0–10 ms + 30–32 ms, ~40 bunches/run | `/eos/experiment/ntof/data/x17/flash_tail/<run>/` | `ntof_processing/flash_pull` |
| per-bunch log-time baseline/mean traces | `/eos/.../flash_tail_summary/summary_<run>.npz` (local copy `out_dir('summaries')`) | `flash_summary.py` via `lxplus/summary.sub` |
| pulse histograms (log t × amp, log t × width) per channel × class | `/eos/.../flash_tail_pulsehist/pulsehist_<run>.npz` | `pulses.py --hist` via `lxplus/pulsehist.sub` |
| quick-pass tables, curves, examples | `out_dir('quick')` = `/media/dylan/data/x17/scint_stack/recovery/quick/` | `quick.py`, `examples.py` |

The lxplus submit directory is `~/x17rec` (code rsynced from here; logs in `logs/`). 70 runs were still
on the tape recall when this was written. When they land, rerun both condor steps for the new runs, then
run `quick.py` and the deck again.

## Configuration epochs (found in this data, causes not identified)

All detectors of a family move together at these runs (`recovery_by_run.csv`):

| from run | change |
|---|---|
| 224355 | walls 0.25 → 0.55 ms |
| 224401 | walls → 0.83 ms; plastics 12 → 31 µs; liquids first appear |
| 224450 (14 July) | plastics 31 → 114 µs; liquids 20 → 8 µs; walls and liquids flat from here |
| 224450–224487 | the plastics still move run to run (114 → 43 → 20 → 126 µs, dedicated) |
| **224488** (17 July) | **everything flat to the end** = "production" (`PRODUCTION_FROM`) |

Neither 22 July (wall gate delay 200 → 1000 ns) nor 23 July (DREAM clock divider) moves anything.
`quick.EPOCHS` and `PRODUCTION_FROM` hold this.

**What the DAQ settings say** (`DAQsettings` tree of the official files: ZS threshold, full scale, offset;
no HV):

- 224354 → 224355: **nothing changed**. The wall step is hardware (HV, SiPM gain or the gate), not recorded there.
- 224381 → 224401: **every plastic ZS threshold moved by 5 mV**; the liquids were added (an old `LIQS` channel removed).
- 224448 → 224450: the walls (32 channels) enter the DAQ configuration (224448 had only 12 channels).
- Within production, ZS thresholds keep changing (e.g. liquids −1 → −16 mV between 224450 and 224712)
  while the recovery times stay flat.

Method caveat this exposes: the DAQ extends the unsuppressed head block while the signal is outside the ZS
band. On the plastics that is 140–210 µs, which is where their recovery time falls. Inside the head the
quick pass takes a per-bin median of all samples; after it, block pre-samples. So the plastics' number
partly depends on where the head block ends, and therefore on the ZS threshold. The 224401 step may be
this rather than a detector change. The template fit (step 1 below) removes the dependence.
The PMT and SiPM HV history is not in these files: look in the e-log or the slow control.

## Layer 1 results (production, ±3 mV, median of four detectors; see the deck for the current numbers)

Liquids ~5 µs (parasitic) / ~6 µs (dedicated); plastics ~25 / ~90 µs; walls ~0.5 / ~0.64 ms.
Dedicated takes ×1.5 longer (median over detectors). The wall divert gate does not set the wall time:
divert-off 224464/466 recover exactly like the other production runs.

## Layer 2–3: first look at the pulses (one file, 224712, 20 bunches)

`pulses.py`, baseline from block pre-samples (ZS blocks) or a 200–400 ns chunked running median (flash
block). Peaks above 4 mV after family-specific smoothing.

1. **The walls lose small pulses while the baseline undershoots.** Between ~10 µs and ~1 ms, 5–20 mV
   wall pulses are missing. A ZS block opens only when the signal crosses a fixed threshold relative to
   the original baseline, so a baseline 10–19 mV low raises the effective threshold by that much. This
   is an efficiency loss the quick pass cannot see. It also matters for the hardware wall trigger (sum
   of two ends, 26–38 mV), if the trigger path sees the same undershoot.
2. **Plastics: the 99th-percentile amplitude climbs from ~80 mV at 1 µs to its ~300 mV plateau by ~10 µs.**
   It then stays flat to 10 ms and matches the 30–32 ms reference. This is either a real gain recovery
   over ~10 µs or the chunked baseline lagging in the steep tail. A template fit will tell.
3. **Liquids: the low-amplitude feature jumps 10 → 23 mV at ~0.5 ms.** That is a change of population
   (capture γs arriving), not gain. It shows why a spectrum's shape cannot be read as gain without a
   fixed reference.
4. **Wall position handle.** For 85–90 % of wall pulses the other end of the bar fires within 30 ns.
   Their Δt (rms 4–6 ns) and ln(A₁/A₂) (rms 0.6–0.7) correlate at −0.6 to −0.74, so either one places
   the hit along the bar.

## Layer 2–3 on the whole production set (`pulse_analysis.py`, deck slides 12–14; numbers from the first 190-run pass)

- **Pulse shape.** Wall pulses are narrow (45–65 ns against 75 ns) until ~6 µs (parasitic) / ~27 µs
  (dedicated). Plastic widths are normal within a few µs. Liquid widths are below the histogram resolution.
- **Amplitude scale** (the 90 % quantile of pulses above a floor, over its 1–10 ms value). Plastics:
  ~0.5 at 2 µs, ~0.85–0.9 by 10 µs, 1.0 by ~1 ms. Liquids: ~0.65 at 2 µs, then 0.85 → 1.0 by ~1 ms.
  Walls: never compressed. **Not a gain measurement yet**: the same scale at 30–32 ms differs from 1–10 ms
  by −15 % (walls) to +35 % (plastics) with no flash nearby. That is the particle mix (cosmics there), and
  the same can happen early (quenched fast-neutron recoils).
- **Walls lose small pulses** while the baseline undershoots: 4–15 mV pulses per pulse above 30 mV are
  ×3.9 fewer at 10–100 µs than at 1–10 ms (dedicated).

### First try at a fixed reference: plastic pulses in coincidence with the wall of the same arm

On 224712 (20 bunches, `pulses_v2` list): plastic minus wall Δt peaks at −15…−5 ns in every arm,
5–7× the side level. With an accidental subtraction (signal −20…0 ns, side +50…+150 ns ÷ 5):

| slice | net pairs | median plastic amp |
|---|---|---|
| 1–3 µs | 245 | 40 mV |
| 3–10 µs | 905 | 61 mV |
| 10–100 µs | 7 481 | 82 mV |
| 0.1–1 ms | 14 089 | 79 mV |
| 1–10 ms | 19 567 | 80 mV |
| 30–32 ms (cosmics) | 81 | 177 mV |

There is no MIP peak in the beam window: the coincident particles are mostly not minimum-ionising (the
cosmics at 30–32 ms are). Above 10 µs the coincident spectrum is flat, with a sharp upper edge at
~400–450 mV that is stable to 10 ms and sits lower at 3–10 µs. If that edge is set by the light yield or
the electronics, rather than the particles, its position is the gain reference. Next: build the
coincidence into the condor reduction (pairs per time slice, both amplitudes) to get campaign statistics,
and find out what sets the edge.

## The fixed-reference layer, whole production set (`coinc_analysis.py`, v2 histograms, deck slides 15–18)

Reprocessed all 242 runs with the template fit and the coincidence histograms (`pulses.py --hist`, v2 →
`/eos/experiment/ntof/data/x17/flash_tail_pulsehist_v2`). Production runs: 3 004 parasitic + 4 073 dedicated bunches.

- **Template fit** (`templates.py` → `templates.npz`: median shape of clean 1–10 ms pulses; fit A·T + b₀ + b₁t).
  80 % of the early misfit excess is gone by ~6 µs (plastics), ~7.5 µs (walls), ~60 µs (liquids), dedicated.
  The walls' last 20 % takes until ~0.1 ms. The plastic/liquid misfit floor (~0.1) is sub-sample alignment
  of 4–13 ns pulses; a template with a fitted time shift would lower it.
- **Plastic behind a wall hit** (−25…+5 ns, sideband subtracted): the median is back (±10 %) by ~13 µs, the
  95 % upper edge only by ~0.5 ms (~0.85 from 10 µs to 0.1 ms). Still not a fixed population (no MIP peak).
- **Wall bar ends**: the median over 16 bars of Δ ln(A₁/A₂) at fixed position stays within 4 % after 2 µs. So a
  gain loss on ONE end is ruled out; a common loss is not tested by this.
- **Along the bar** (Δt → mm with the scint_stack MM-track calibration, bands v < −300, |v| < 150, v > 300 mm):
  the three bands agree within 7 % after 3 µs. All dip together to ~0.92 (parasitic) / 0.96 (dedicated) at
  30–300 µs: either a common SiPM gain sag or the particle mix.

## Round 3 (2026-10-21): gain reference, per channel, trigger path (deck slides 19–21)

- **The plastic upper edge is light, not electronics.** Behind a wall hit, the edge (95 %) over the bar's own
  pulse scale is 1.19–1.36 across the 8 bars, while the scales span 176–278 mV (dedicated, 1–10 ms). So the
  edge follows each bar's gain and is usable per bar as a gain reference, as long as the coincident
  particles' energy distribution holds.
- **Single-photoelectron peaks:** the plastics show a bump at ~2.5–4 mV above the noise in the quiet tails of
  ZS blocks (local 224712 file), but its fitted position wanders 3.0–4.0 mV between time slices and is mixed with
  real low-energy pulses. Not yet a clean reference: it needs per-channel fits and a cleaner sample. The walls show
  none (a single SiPM photoelectron is under the noise); the liquids show only a weak shoulder.
- **Per channel** (`channel_maps.py`, slide 19): walls uniform (baseline 0.55–0.75 ms; amplitude back within
  ~3 µs except A6). Plastics: 48 µs (A1) to 300 µs (C1). Liquids C and D slower in shape (0.3–0.5 ms).
- **The wall undershoot follows the channel's gain** (slide 20): depth vs pulse scale, correlation 0.73–0.74,
  depth ≈ 10 % (parasitic) / 15 % (dedicated) of the channel's scale; deeper undershoots recover later
  (correlation 0.80). In units of its own pulses every wall end loses the same fraction.
- **Trigger path** (`trigger_cand.py` on condor → `flash_tail_trigger/`, `trig_analysis.py`, slide 21):
  - **DREAM's gate opens at ~1.0 ms**: the first trigger of a bunch is at 992–1061 µs (1–95 %).
  - The expected undershoot of the wall trigger sum is then ~1.6 mV, ≤ 5 % of the 26–38 mV thresholds, which
    bounds any trigger bias.
  - A direct turn-on measurement (emulated wall-sum + plastic candidates matched to DREAM triggers: 8.9 M
    candidates, 121 runs with slims) fails. DREAM's dead time is ~205 µs after a trigger, but the busy state
    lasts up to several ms (strong-candidate efficiency 0.2 at 0.3–1 ms, 0.46 at 2–5 ms, 0.76 beyond), and
    some bunches aren't recorded. Before a bunch's first trigger the efficiency is 0.72.
- **MM tracks: deferred.** They start with DREAM at ~1 ms, by which point only the slow last 5–15 % is left.
  The scint_stack per-track tables are gone from the desktop and not on EOS. Rebuilding them needs the stage-3
  full pass (`/eos/experiment/ntof/data/x17/desktop_scratch/x17_local/yfix_work/sept26_prelim_stage3_fullpass`,
  11.4 GB) and the slims.

## Round 4 (2026-10-21 night): sub-ns template, plastic edge per bar, photoelectrons (deck slides 17, 23)

v3 reprocessing of the 181 production runs (`flash_tail_pulsehist_v3`): template fit with a derivative term (a
sub-ns time shift), templates re-made with sub-sample alignment (`templates.py refine`), and a per-channel
small-peak histogram from the quiet tails of ZS blocks (`SPE`).

- **Sub-ns template:** the misfit floor moved only a little (plastics 0.119 → 0.106; liquids unchanged ~0.11). Shape
  recovery (80 %, dedicated): plastics 7.5 µs, walls 9.4 µs, liquids 133 µs. The liquid number moved 60 → 133 µs
  between template versions, so it is soft: it needs per-channel templates.
- **Plastic light edge per bar** (behind a wall hit, each bar over its own 1–10 ms value; dedicated): 0.71 at
  10 µs, 0.87 at 100 µs, 0.93 at 300 µs. All 8 bars agree within 2–7 % (rms). This is the best common-mode
  reference we have. A change of particle mix would give the same curve, so it remains an upper bound on the
  gain sag. Geant4 of the coincident population is the way to separate them.
- **Photoelectron peak:** 6 of 8 plastic bars show an accepted peak at 2.6–4.5 mV (dedicated, 5–12 σ). Early on it
  moves UP (×1.2–1.6 at 0.03–0.3 ms, settling by ~1 ms), the opposite of a gain sag. While the rate is high, the
  "quiet" tails are not quiet (multi-photoelectron and small real pulses), so it cannot measure gain. It is at
  least not evidence for a large sag. Walls: no peak (one wall end passing the cuts is excluded). Liquids: none.
- Operational: some condor jobs of a cluster ran late; the log names are shared between passes, so judge
  completion by the EOS file count (`xrdfs ls`), not by `condor_q` being empty or by `=== done` in the logs.

## The full analysis, step by step

1. **Pulse model.** Fit per-family templates on clean late-time pulses (30–32 ms window, isolated, flat
   baseline). Then refit pulses in the tail with template + local exponential baseline, so the steep
   flash tail stops biasing amplitudes. Report χ²/ndf against time: the earliest time where the shape
   is normal is "how far back pulses can be recovered".
2. **Gain references, independent of the particle mix.** In order of preference:
   - The **ratio of the two ends of a wall bar** for coincident pulses at fixed Δt (fixed position).
     It is population-independent, and a sag on one end shows directly. A common sag on both ends would
     not show, so pair it with the next one.
   - A **fixed spectral feature**: the Compton edge of the 2.22 MeV H(n,γ) line in plastics and liquids
     (thermal captures dominate ~0.1–20 ms), and the plastic ~100 mV peak if it proves stable. Its
     position against time, divided by its position at 30–32 ms, is the gain.
   - **Single-photoelectron / dark pulses**, if they are above the ZS threshold anywhere (check the
     low-amplitude spectra at late times).
   - The amplitude of **MM-tagged tracks** (`ntof_scint_stack` tables): through-going particles of known
     path length, but only from ~0.5 ms (DREAM triggers) and thin statistics.
3. **Effective threshold against time** (walls especially): the ZS/trigger threshold relative to the
   local baseline, from the lowest-amplitude pulses found per time bin. Turn it into an efficiency loss
   for a given spectrum.
4. **Detector to detector.** All 44 channels against each other in layers 1–3: arms A–D, the two ends of
   a bar, the eight PMTs. The quick pass already shows walls uniform to ~10 % and plastic B/C slower than A/D.
5. **Position dependence.** Along each wall bar, bin coincident pairs in Δt (or ln A₁/A₂) and repeat
   layers 1–3 per bin. Across the stack, use the MM tracks: from ~0.5 ms only, matched to flash_tail
   bunches by (run, bunch) through the slim `events` tree. Calibrate Δt → mm from the bar length (edges of
   the Δt distribution) or against the MM tracks.
6. **Intensity scaling.** Two classes is two points. Within each class PulseIntensity spreads ~±5 %. Look
   at whether recovery times scale with the flash charge (PKUP amplitude) bunch by bunch.

Deliverables: extend the slide note (or a second note, "full analysis") with the template-fit recovery,
gain against time per family, the wall efficiency loss, and the per-channel and along-bar maps. Plus a
`report.html` per the repo rules.

## What can bite

- A spectrum's shape changes with the particle mix (prompt γ, fast neutrons, then capture γ). Never read a
  shift in shape as gain without a fixed reference.
- The plastics are a continuous pulse train in the first ~100 µs, so a baseline estimate there includes
  pile-up.
- The unprocessed runs have no PulseIntensity; they are classed on the PKUP flash amplitude
  (`quick.classify`, cut ~650 mV), which separates as cleanly as the intensity does.
- The 57 runs from 1–9 July have no scintillators (PKUP only).
- Everything here is the digitized path. The analog trigger path may differ.
