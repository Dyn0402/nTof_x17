# SiPM top+bottom SUM threshold — run 224460 (newer run, plastics fixed)

**Question:** a good threshold on the SUM of a wall group's top + bottom SiPMs that
rejects background while collecting *all* MIPs.

## Method

Each wall group (4 bars, detn pairs (1,2)(3,4)(5,6)(7,8)) is read by a top and a
bottom SiPM. A through-going MIP fires *both* ends in time coincidence, so the
top-bottom coincidence (signal window centred on the group's dt peak, sideband-
subtracted) is a clean, plastic-independent **MIP tag**. For each tagged pair we form
`S = amp_top + amp_bottom` (this JOINT sum is new — the 07/09 caches only kept the
top/bottom marginals, so `build_sum.py` re-reads the run). Selection matches the
pipeline: beam-on & wall-active bunches, `tof - tflash > 0.1 ms` (late, off-flash).

- `build_sum.py <ARM>` — one arm per process (peak ~1.75 GB RAM), per-arm cache.
- `plot_threshold.py` — spectra, efficiency/background curves, recommendation.

## What the data says

1. **The top-bottom coincidence is ~99.8% pure MIPs.** Accidental (random) coincidences
   are ~1–2 per bunch vs ~735 real MIP coincidences per group per bunch. The MIP sum
   peaks at **~2000–2600 ADC ≈ 60–80 mV** (= 2× the single-channel MIP, ~31 mV) on all
   16 groups, arms A–D alike (the 224404 A/D deficit is gone).
2. **The late-spill wall flux is MIP-DOMINATED.** The inclusive single-SiPM spectrum
   itself peaks at the MIP (~31 mV); only ~14% of single hits fall below 5 mV (the
   sub-MIP soft-gamma / noise floor). **There is no MIP-vs-background valley** — the MIP
   Landau is broad and extends smoothly down to low sums (near-end hits, corner clips,
   straggling), with no separate background population sitting under it.
3. Consequently an amplitude threshold **cannot** reject the accidental fake-triggers
   (they are pairs of real MIP-scale hits, same amplitude as signal) — only a tighter
   *time* coincidence can. The threshold's real job is to sit above electronic noise.

## MIP efficiency vs sum threshold (min / mean over the 16 groups)

| threshold | min MIP eff | mean MIP eff | accidental fake-trig / bunch |
|-----------|-------------|--------------|------------------------------|
| 5 mV      | 100.0%      | 100.0%       | 1.95 |
| **8 mV**  | **99.5%**   | **99.8%**    | 1.68 |
| 10 mV     | 97.3%       | 98.6%        | 1.51 |
| 15 mV     | 88.4%       | 91.7%        | 1.30 |
| 20 mV     | 80.4%       | 84.7%        | 1.19 |
| 30 mV     | 67.5%       | 73.3%        | 1.04 |
| 40 mV     | 57.6%       | 63.6%        | 0.91 |

## Recommendation

**Set the top+bottom sum threshold at ~8 mV (~260 ADC) to collect ≥99.5% of MIPs on
every group** (10 mV is a fine round choice, ≥97% worst group / ~99% mean). This sits
~8–10× below the MIP peak (~60–80 mV), well above the ~1.2 mV per-channel recording
floor, and clears the sub-5 mV soft/noise floor. A single common value works: gains are
uniform in mV (per-arm 99% cuts cluster at 10–13 mV).

Raising the threshold to 15–20 mV buys almost no extra background rejection (the flux
is MIPs) while cutting real MIP efficiency to ~80–90% — not worth it under a
"collect all MIPs" priority. If low-amplitude background must be suppressed harder,
tighten the **time coincidence** (or require top∧bottom), not the amplitude.

Figures: `figures/sum_spectra_run224460.png`, `figures/threshold_tradeoff_run224460.png`.
