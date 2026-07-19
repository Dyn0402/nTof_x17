# ZS threshold optimization from run_55 no-ZS data (2026-07-19)

**Goal: pick the zero-suppression configuration for the next beam runs —
threshold in σ, and whatever else the simulation shows is needed — using the
run_55 raw waveforms as the in-beam test bench.**

Scripts: `26_zs_sim_extract.py` (waveform-level simulation → cache) and
`26b_zs_analysis.py` (figures in `figures/26_zs/`, numbers in
`calib/26_zs_summary.json`). Inputs: run_55 `decoded_root` (full 32-sample
raw waveforms, all 4096 MM channels, 76k triggers across the resist scan) +
the pedestal/threshold run `pedestals_07-18-26_14-06-43` (taken 5 h before
run_55; per-channel ped and σ parsed from the `_thr.aux`).

## How DREAM ZS actually works (established from cfg + aux files)

- Thresholds are per channel: `thr_ch = ped_ch + N·σ_ch`, with σ_ch measured
  by a beam-off pedestal run and **N = `Sys PedRun Threshold`** (deployed
  default 5.00). Verified: aux `thr` = `Avr + 5.00·Std` to 0.5 counts.
- ZS mode runs with pedestal subtraction ON (baselines normalized to
  CmOffset = 256) and `ZsTyp=1` (tpc): sample-level readout,
  `ZsChkSmp = 4` extra samples per crossing.
- `Feu_RunCtrl_CM` (common-mode correction before ZS) is **currently 0**.
- Beam-off σ per channel is bimodal: ~86 % of channels at 3–6 ADC, ~14 %
  CM-dominated at 60–90 ADC.

## Finding 1 — CM correction is MANDATORY (the headline)

In-beam, the per-channel waveform wander is **10–20× the beam-off σ**
(median MAD 37–96 ADC across FEUs vs 3.4–5.7 beam-off). With thresholds from
a beam-off pedestal run and no CM correction, **even 5σ keeps 60–95 % of all
channels** — ZS would be nearly useless in beam (fig 01).

The wander is a common-mode baseline oscillation, >99 % coherent within each
Dream chip (64 ch): subtracting a per-Dream, per-sample median restores the
per-channel residual to **4.1 ADC ≈ the beam-off σ**. Per-FEU CM is NOT
enough (36 ADC residual — chips wander with different amplitudes).

**⇒ Enable `Feu_RunCtrl_CM = 1` in ZS mode.** All numbers below assume it.

Bonus: CM correction also absorbs the coherent +38 ADC post-flash baseline
shift (measured at 0–0.5 ms, decayed by ~6 ms), so flash windows do NOT
flood the ZS readout — volume at 0–2 ms is the same ~10 % as elsewhere.

## Finding 2 — volume vs threshold (CM on)

Medians over the beam gate (channel-level retention; sample-level volume for
tpc ZS incl. +4 samples/crossing):

| N [σ] | ch kept (b1) | sample volume (b1) | est. readout/event |
|------|------|------|------|
| 2.0 | 52 % | 17.4 % | 0.35 ms |
| 3.0 | 30 % | 11.0 % | 0.22 ms |
| **3.5** | **25 %** | **9.4 %** | **0.20 ms** |
| 4.0 | 22 % | 8.4 % | 0.17 ms |
| 5.0 | 17 % | 7.0 % | 0.14 ms |

(readout estimate: 2 ms/event measured no-ZS drain × volume, 2 % overhead
floor; retention sits far above the Gaussian floor because it is real
activity — captures, tracks, ringing — not noise.)

**~10× readout speedup at 3.5–5σ.** The FIFO comb that blanked 2–6 ms in
run_55 closes: at ~0.2 ms/event the DAQ sustains the trigger rate through
the whole 30 ms gate, so the next scan CAN measure the 1–8 ms region
(including the 3 ms target).

## Finding 3 — real track hits survive; losses are CM signal-bias, not
threshold

Survival of strips belonging to run_55 MIP-track clusters (>400 ADC hits,
x/y matched — 12k clusters):

| N [σ] | A | B | C | D |
|------|-----|-----|-----|-----|
| strips @3.5 | 99.7 % | 92.4 % | 93.6 % | 96.8 % |
| clusters ≥3 strips @3.5 | 99.8 % | 94.6 % | 95.7 % | 97.9 % |

- Det A: essentially lossless up to 5σ.
- B/C/D losses are **nearly threshold-independent** (B already loses 2.9 %
  of strips at 2.0σ): the CM median gets signal-contaminated in busy windows
  (capture blobs / B's wide ringing occupying a large fraction of a Dream
  chip), over-subtracting and pushing real strips below ANY threshold. The
  firmware CM will do the same thing — this is a property of ZS+CM in this
  radiation environment, not of the threshold choice.
- Going below 3σ therefore buys almost no survival while costing ~2× volume.
- Noisy channels (σ 60–90): 3.5σ ⇒ 210–320 ADC thresholds, still below the
  400 ADC analysis hit threshold — the offline analysis sees the same hits.

## Recommendation

1. **`Feu_RunCtrl_CM = 1`** — non-negotiable; without it ZS fails in beam.
2. **`Sys PedRun Threshold = 3.5`** (3.0–4.0 all defensible; 3.5 balances
   the volume knee against headroom for pedestal drift). Keep `ZsTyp=1`,
   `ZsChkSmp=4`.
3. Take the pedestal run close in time to physics (thresholds used the
   same-day pedestal run here and the σ scale was confirmed stable:
   post-CM in-beam residual ≈ beam-off σ).
4. First ZS run: include a short no-ZS reference subrun for a direct
   data-driven check of strip survival and CM bias.
5. If the firmware CM supports hit/outlier rejection, enable it — it would
   recover most of the 3–7 % busy-window strip loss on B/C/D. Worth checking
   the FEU documentation.

## Caveats

- Survival measured for strips above the 400 ADC analysis threshold;
  sub-threshold fringe strips are not tracked (ZS at 3.5σ keeps far more
  than the analysis uses, so no analysis-visible loss).
- Cluster amplitude loss is bounded by strip loss (the lost strips are the
  smallest); not separately tracked.
- Readout-time model is linear in volume with a 2 % floor — good enough for
  the ~10× conclusion, not for precise cadence prediction.
- b1 sits on the thermal peak: these volumes are the WORST case over the
  cycle; average volume will be lower.

## Figures (figures/26_zs/)

- `01_cm_mandatory.png` — retention vs time, no-CM vs CM (KEY)
- `02_volume_vs_thr.png` — channel + sample volume vs N by time batch
- `03_retention_det_hv.png` — per-det retention vs N and HV
- `04_strip_survival.png` — strip + cluster survival vs N (KEY)
- `05_survival_vs_hv.png` — survival vs HV at 3.5σ and 5σ
- `06_readout_estimate.png` — est. readout/event vs N
- `07_residual_sigma.png` — post-CM residual σ per channel/FEU
