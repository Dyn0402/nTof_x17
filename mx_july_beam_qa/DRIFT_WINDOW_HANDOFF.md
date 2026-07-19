# Drift-voltage & readout-window sizing — HANDOFF

> **ANSWERED 2026-07-19 → see `DRIFT_WINDOW_ANALYSIS.md`.** Headline: latency 32,
> n_samples 28; det A to 700 V (staged); B/D "lateness" was flash-ringing +
> spark artifacts, not slow gas; B/C/D breathe ~0.8 % H2O (A dry); small on-beam
> det-A drift scan recommended as the 700 V spark gate.

## Task for you (the follow-up model)

Decide, for the MX17 Micromegas stack now running at nTOF EAR2:
1. **Drift voltage(s)** — in particular whether the spark-limited "good" detector
   (det A) should stay at 600 V or push to 700 V.
2. The **minimum `n_samples` (and `latency`) for the 60 ns DREAM readout** such that
   **no drifting primary is lost on ANY of the four detectors**, trimming the empty
   ends of the current 32-sample / 1.92 µs window as much as safely possible.

This document is a **data gather** done 2026-07-18: (1) Magboltz theory, (2) the June
Saclay cosmic micro-TPC measurement on an identical 30 mm gap, and (3) a direct
measurement from last night's nTOF cosmics (run_54). Numbers are cross-checked and the
tooling/paths to reproduce and extend are in §7. **The deep reasoning, the de-noising of
det B/C/D, and the final recommendation are yours to do.** Read §0 and §6 first — the gas
topology (below) makes this a 4-detector worst-case problem, not a single-detector one.

> **⚠ Governing fact discovered after the first gather: the four detectors are
> DAISY-CHAINED in series, A → B → C → D → exhaust (single gas line).** Gas gets
> progressively wetter/more-attached down the chain (each detector outgasses H2O/O2
> into the next; June established H2O rises along a shared line). So there are **two
> opposing gradients** that set the drift time:
> - **field:** A (600 V, 200 V/cm) is the SLOWEST by field; D (800 V, 267 V/cm) fastest.
> - **water:** A gets the FRESHEST/driest gas (fastest); D the wettest (slowest).
> These partly cancel, so **the binding constraint — the detector with the longest
> drift time / latest deep-drift edge — is not obvious a priori and must be found
> empirically across all four.** It is plausibly **det D at 800 V (wettest gas)** OR
> **det A at 600 V (lowest field)**. The window must cover whichever is worst.
> This also means the "800 V looks slower than 600 V" puzzle in §4 may be **partly
> real** (water gradient), not purely the noise artifact I first assumed.

---

## 0. The question and the hard constraints

- **Drift gap:** 30 mm mechanical (all four detectors A–D, confirmed in run_config).
- **Gas:** Ar/iC4H10 90/10 (config `"gas": "Ar/Iso 90/10"`), **series flow A→B→C→D→exhaust**
  (single line; wetness/attachment increases down the chain — see the ⚠ box above).
- **Sampling:** DREAM, `sample_period = 60 ns`, currently `n_samples_per_waveform = 32`
  → **1.92 µs window**. `latency = 35` (DREAM units; see §3 for the sample↔time map).
- **Resist HV:** fixed/constrained (560 V in run_54; not a free parameter here).
- **Drift HV — the live decision:**
  - "Good" detector = **det A**: sparks at 800 V, so has only run at **−600 V**.
    Hope to push to **−700 V**. (run_54: det A drift = 600 V, I_drift 0.14 µA,
    stable, no sparking.)
  - det B/C/D currently at **−800 V** (E = 267 V/cm).
- **E field = V / 3.0 cm** for a 30 mm gap:
  600 V → **200 V/cm**, 700 V → **233 V/cm**, 800 V → **267 V/cm**.
- Objective: trim samples from both ends ("wiggle room at start and end") without
  clipping the latest-arriving primaries.

**One-line preview (see §5–6):** on the *driest* detector (**det A @600 V**) last-night's
gas gives full-gap drift **≈ 0.9–1.0 µs (≈ 16 samples)**, column in samples **~9–11
(prompt) → ~27 (deep edge p99)**, comfortably inside the 32-sample window (<1–3 %
truncation) → the nTOF gas is far drier than June's. **But det A is the best case;** the
noisy 800 V detectors (esp. **det D**, wettest) show deep edges piling at the sample-31
ceiling and MUST be de-noised and re-measured before trimming — they may already be
truncating. The main trim lever is **reducing `latency`** to remove the ~8 empty
pre-prompt samples; the tail budget is set by the worst detector, not det A.

---

## 1. Theory — Magboltz drift velocity (clean gas)

Garfield++/Magboltz is installed and runs locally (`garfield_sim/`, ROOT 6.30,
`GARFIELD_INSTALL=/home/dylan/Software/garfield/install`).

**Fresh run for the exact gas at CERN pressure** (nTOF EAR2, 450 m, 720.8 Torr):
`garfield_sim/mm_drift_9010_cern.py` → `results/drift_velocity_Ar_iC4H10_90_10_CERN.json`.

Clean Ar/iso 90/10, drift velocity vs field (Magboltz; CERN P). Low-field points
verified from the run; the operating-range rows are the interpolated/rising-branch
values (curve still rising toward its peak in this range):

| drift HV | E [V/cm] | v_clean [µm/ns] | full-gap drift 30 mm | # 60 ns samples |
|---:|---:|---:|---:|---:|
| **600** | **200** | **40.5** | 741 ns | 12.4 |
| **700** | **233** | **42.6** | 704 ns | 11.7 |
| **800** | **267** | **44.1** | 681 ns | 11.3 |

*(exact, from the completed CERN Magboltz run; 166 V/cm = 37.3 µm/ns anchors the low end.)*

Cross-check (existing table, Saclay P 745.8 Torr, `results/drift_velocity_candidates2.json`,
`Ar90_iso10`): 206→40.4, 239→42.5, 258→43.4, 278→44.1 µm/ns — same shape, CERN is
~3 % faster at equal E (lower pressure → higher E/N). 95/5 is within ~1 µm/ns of 90/10
here (`drift_velocity_Ar_iC4H10_95_5_Saclay.json`).

**Take-away:** in *clean* gas the full-gap drift is only ~12 samples at any of these
voltages, and drift voltage barely changes it (750→682 ns from 600→800 V). **Clean
gas is not the regime we are in** — see §2 and §4.

---

## 2. June Saclay cosmic micro-TPC — the *measured* v(E) on a 30 mm gap

This is the most relevant prior: **same 30 mm gap**, Ar/iso **95/5**, cosmic muons,
micro-TPC mode, full v(E) scan. Full writeup:
`mx_june_cosmic_qa/DET3_WEEKEND_ANALYSIS.md` (and `report_det3_weekend/main.pdf`).

Bias-free **geometry** drift-velocity estimator (E = HV/3 cm, 30 mm gap):

| drift HV | E [V/cm] | v_geom [µm/ns] | T_sat [ns] | note |
|---:|---:|---:|---:|---|
| 500 | 167 | 12.4 | 1361 | **window-truncated** (primaries lost!) |
| 700 | 233 | 21.6 | 992 | |
| 900 | 300 | 30.0 | 754 | |
| 1000 | 333 | 33.9 | 690 | headline |
| 1100 | 367 | 35.1 | 661 | |

**Critical lessons for us:**
1. **Water dominates the low-field drift velocity.** This gas carried **0.7–1.2 %
   H2O + ~1 % air** (fleet equilibrium — a *system* property of the gas line, see
   §6b of that doc). Water roughly **halves** v at 200–233 V/cm vs clean Magboltz
   (21.6 vs ~41 at 233 V/cm). It ENHANCES O2 attachment (deep charge lost, ~1/4 of
   the column, λ_att 16–18 mm).
2. **At low drift field + wet gas the column overran the readout window.** The 500 V
   point was truncated at the (then 32-sample × 60 ns) window — exactly the failure
   mode to avoid. This is why 600 V is the dangerous corner if the gas is wet.
3. Time-based track fits are biased **low** ~10–20 % by prompt resistive charge
   sharing (c1≈0.45–0.52 between neighbours); the geometry estimator above is the
   unbiased one. (Only relevant if the follow-up re-derives v from nTOF tracks.)

**The open variable that decides everything: how wet is the nTOF gas right now?**
→ answered directly in §4.

---

## 3. Ground-truth DAQ config (nTOF, run_54, last night)

Source: `daq_lxplus:/mnt/data/x17/beam_july/runs/run_54/run_config.json`
(also the human-readable `trigger` string). Beam was DOWN 2026-07-18, so run_54 is
**cosmic singles** — muons crossing the full gap, a direct analog of the Saclay measurement.

- Gas Ar/iso 90/10; all four dets 30 mm gap.
- **32 smp × 60 ns = 1920 ns**, `latency 35`, `sample_period 60`, zero-suppress OFF.
- Drift: **det A 600 V**, det B/C/D 800 V; **resist 560 V** all four (HV monitor confirms:
  9:0=600 V @0.14 µA, 9:1/9:2/9:3=800 V, 5:1–5:4=560 V).
- **FEU map** (decoded-root filename suffix `_NN.root` = FEU number):
  | det | drift | FEU x | FEU y | cable |
  |---|---:|---:|---:|---|
  | **A** | **600 V** | 3 | 4 | 1.5 m |
  | B | 800 V | 5 | 6 | 1.5 m |
  | C | 800 V | 7 | 8 | 1.5 m |
  | D | 800 V | 1 | 2 | 2 m |
- **Sample↔time map** (from config note "MM pulse peak ≈ latency − 24"):
  a zero-drift (near-mesh, prompt) MM pulse peaks at sample **latency − 24 = 11**.
  This is confirmed by the data (§4). So: prompt at sample 11 ⇒ samples 0–10 (0–660 ns)
  are *pre-signal baseline*; drift adds on top of sample 11.

---

## 4. Direct measurement from last night's nTOF cosmics (run_54, subrun 000)

Decoded tree `nt`, branches `eventId, timestamp, sample, channel, amplitude`
(per event: full 512 ch × 32 samples). Method (scratchpad `drift_time_v2.py`/`v3.py`,
copied to `daq_lxplus:/tmp/`): per event subtract per-sample common-mode (median over
channels) + per-channel baseline; a **real hit strip** = peak > ~400–500 ADC over
baseline with pulse width ≥ 2 samples (MIP strips are 400–3800 ADC; coherent noise
RMS ≈ 60 ADC). Record each hit strip's **peak sample**; per event take earliest
(prompt), latest (deep-drift edge), span.

### det A — 600 V — CLEAN, and it is the *driest / best-case* detector (first in the chain)
~10 % of events have a track cluster (531/5212), median 6 strips/cluster — clean cosmics.

| quantity | x (FEU3) | y (FEU4) | meaning |
|---|---|---|---|
| earliest peak sample (p5/p50) | 9 / 11 | 9 / 11 | **prompt = sample 11 → confirms latency−24** |
| latest peak sample p90/p95/p99 | 23 / 24 / 27 | 24 / 27 / 31 | deep-drift edge |
| span p95 / p99 | 14 / 16 smp | 16 / 22 smp | full-column drift time |
| span p95 in ns | 840 ns | 960 ns | → v ≈ 30 mm / 0.9 µs ≈ **31–36 µm/ns** |
| **at ceiling (latest ≥ 31)** | **0.3 %** | **3.1 %** | **NOT truncated** |

**Interpretation:** at 600 V the nTOF gas drifts at **≈ 31–36 µm/ns** — i.e. **close to
clean Magboltz (~40) and MUCH drier than the June Saclay gas (16–17 at this field).**
The drift column occupies samples **~9–11 → ~24–27** (p95–p99); the current 32-sample
window has ~4–5 samples of tail headroom **and** ~8 samples of empty pre-prompt baseline.

### det B/C/D — 800 V — measurement confounded (noise + real water gradient), MUST re-do
FEUs 1/2/5/6/7/8 carry heavy residual coherent noise in this run (e.g. det D_y shows
14–62 "strips"/event in *every* event; det C_x fires in all events at 2 strips). The
cluster filter (`v3`) recovers track-like samples but the deep-edge percentiles are
inflated by leftover noise and by an **apparent per-detector prompt offset** (cluster
earliest: A ~9–11, C ~15, B ~19). Cable length is NOT the cause (A/B/C all 1.5 m).
Indicative (noisy!) deep-edge / ceiling numbers, **do not size on these yet**:

| det | drift | latest p95 / p99 | at ceiling ≥31 | note |
|---|---:|---|---:|---|
| A | 600 V | 24 / 27 | 0.3–3 % | clean, driest |
| C | 800 V | 25 / 28 | 0.6–2.5 % | moderate noise |
| B | 800 V | 31 / 31 | 21–30 % | noisy AND/OR late |
| **D** | **800 V** | **31 / 31** | **7–20 %** | **wettest gas + noisy — prime truncation suspect** |

**Key reinterpretation given the daisy chain:** the fact that the 800 V detectors look
*slower / later* than det A@600 V is **no longer automatically an artifact.** Down-chain
detectors (esp. **D**) breathe the wettest gas, and June proved water can more than halve
low-field v and that attachment/water is per-detector. So the late deep edges on B/D are
plausibly **noise + a genuine water-induced slowdown superimposed.** Your job is to
**separate the two**: de-noise (spatial-cluster + per-strip significance, or proper
common-mode by connector), then measure each detector's true prompt t0 and deep-drift
edge. If det D's deep edge genuinely rides the sample-31 ceiling, **primaries are being
lost on det D right now.**

**Bottom line from data:** det A@600 V (driest, lowest field) is clean and fits with
margin, but it is the **best case, not the binding case.** The window must be sized for
the worst detector — likely **det D** — which is not yet cleanly measured.

---

## 5. Drift-time / sample budget (synthesis)

Full-gap (30 mm) drift time and 60 ns-sample count, three gas regimes:

| regime | 600 V (200 V/cm) | 700 V (233 V/cm) | 800 V (267 V/cm) |
|---|---|---|---|
| clean Magboltz 90/10 (§1, exact) | 741 ns → 12.4 smp | 704 ns → 11.7 smp | 681 ns → 11.3 smp |
| **nTOF measured det A, run_54 (§4)** | **~900–960 ns → ~15–16 smp** | ~800 ns → ~13 smp (scaled) | — |
| June Saclay wet 95/5 (§2) | ~1800 ns → ~30 smp (truncated) | 1389 ns → ~23 smp | ~1000 ns → ~17 smp |

Window budget = pre-prompt baseline + prompt + drift span + tail margin. **Sized for the
WORST detector in the chain, not det A.** With the **current latency 35** (prompt at
sample 11), det A@600 V occupies ≈ sample 11 → 27 (p99). The ~8 empty samples (0–10) are
reclaimable **only by lowering `latency`** (prompt sample = latency − 24 ⇒ latency 28 puts
prompt at sample 4).

**Illustrative minimum windows** (3 pre-prompt baseline + drift span + 3 tail margin),
*provisional — pending the clean det D/B measurement that sets the true worst-case span*:
- det A best case, measured gas @600 V: 3 + 16 + 3 ≈ **22 samples** (needs latency ≈ 28).
- @700 V: 3 + 13 + 3 ≈ **19–20 samples**.
- **Do NOT adopt these until det D (wettest) is cleanly measured** — if det D's span at
  800 V is inflated by water toward the June regime, the worst-case span could be larger
  than det A's and the safe window correspondingly longer.
- Floor is set by the **worst-detector drift span**; latency only removes the dead pre-roll.

**Why 700 V on det A** helps little on *length* (span 16→13 smp) but a lot on
**safety/gain**: more spark/gain margin, and it moves det A off the low-field corner
where a wet-gas excursion truncates (§2). Note det A is *first* in the chain (driest),
so its low-field risk is smallest; the down-chain 800 V detectors carry the wet-gas risk.

---

## 6. Open questions / what the deep analysis should resolve

1. **Find the worst-case detector (THE central task).** De-noise B/C/D (the coherent
   noise is per-connector — try common-mode by FEU/connector group, tighter per-strip
   significance, and spatial clustering as in `v3`) and measure each detector's true
   **prompt t0** and **deep-drift edge (latest sample)**. Then measure the **water
   gradient along the chain A→B→C→D**: expect v to fall down-chain. The binding
   constraint = detector with the latest deep edge = longest (drift + t0). Check
   explicitly whether **det D** already rides the sample-31 ceiling (= losing primaries
   now). Disentangle the real prompt-t0 offset (sets each detector's own budget) from
   the noise.
2. **Pin the gas water content per detector.** det A@600 V gives v ≈ 31–36 µm/ns ⇒ map to
   H2O via the Magboltz water grid (`garfield_sim/mm_water_grid_lxplus.py`; June fit
   1 % H2O + 1 % N2). Do it for each detector to quantify the A→D wetness gradient — is A
   near-dry (~0.3 %) and D approaching the 1 % fleet equilibrium? This sets per-detector
   tail margin and tells you if the series line needs attention (flow rate, moleculiar
   sieve/drier, leak).
3. **Full-gap vs recorded-column + attachment.** The span p95/p99 may under-read the
   fully-vertical 30 mm track. Estimate v from (deep-edge − prompt) for the longest-column
   tracks, and check attachment (deepest primaries attenuated before the window ends, as
   in June's ~23 mm recorded column). If deep charge is lost to attachment down-chain, the
   *physical* window requirement may be looser than the mechanical 30 mm — but that is a
   signal-loss you may not want to design around.
4. **Decide latency + n_samples jointly, for the whole stack.** Trimming n_samples without
   lowering latency barely helps (dead pre-roll dominates). Propose one (latency,
   n_samples) pair that covers the worst detector with a quantified margin for (a) a gas
   wetness excursion, (b) the det-to-det prompt-t0 spread, (c) the down-chain v gradient.
5. **Should det A go to 700 V?** Weigh spark risk (sparked at 800) vs gain/margin. det A is
   first/driest so its own truncation risk is low; 700 V buys ~3 samples and headroom.
   Consider also whether the *down-chain* detectors' wet-gas slowdown is the stronger
   argument for a drier/faster gas line or higher flow, independent of det A's voltage.

---

## 7. Reproduce / data pointers

- **Magboltz (theory):** `garfield_sim/mm_drift_9010_cern.py` →
  `results/drift_velocity_Ar_iC4H10_90_10_CERN.json`. Existing: `..._candidates2.json`
  (`Ar90_iso10`, Saclay P), `drift_velocity_Ar_iC4H10_95_5_Saclay.json`. Gas tables in
  `garfield_sim/gas_tables/` (the cached `.gas` only span ~7–83 V/cm — below operating
  field — hence the fresh run).
- **June measurement:** `mx_june_cosmic_qa/DET3_WEEKEND_ANALYSIS.md`,
  `14_/21_drift_velocity_scan.py`, `43_drift_window_truncation.py`,
  `report_det3_weekend/main.pdf`.
- **nTOF data (on `daq_lxplus`):**
  `/mnt/data/x17/beam_july/runs/run_54/cosmics_r560_dr800dA600_000/decoded_root/*_NN.root`
  (NN = FEU). Config: `.../run_54/run_config.json`; HV: `.../<subrun>/hv_monitor.csv`.
  Read with `~/PycharmProjects/nTof_x17/.venv/bin/python` (has uproot).
- **Drift-time extractor scripts:** scratchpad `drift_time_v2.py` (per-strip, fast),
  `drift_time_v3.py` (spatial-cluster filtered), `calib_noise.py` (noise/amp scale).
  Copied to `daq_lxplus:/tmp/`. Usage: `python drift_time_v2.py <decoded_dir> <out.png> 3=A_x_600 4=A_y_600 ...`.
- Repo memory: `[[july-beam-run224489]]`, `[[june-cosmic-qa-*]]`.
