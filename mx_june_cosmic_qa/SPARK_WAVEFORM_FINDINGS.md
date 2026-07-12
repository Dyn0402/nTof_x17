# Spark waveform anatomy — what the detector does DURING a discharge (2026-07-12)

*Open-ended waveform-level follow-up to the reconstructed-hit spark study
(`det3_spark_analysis/`) and the no-dead-time result (`39_spark_deadtime.py`).
Script: `40_spark_waveforms.py sat_det3`. Uses the raw DREAM waveforms
(`decoded_root`: 512 ch × 32 samples × 60 ns, pedestal-subtracted, common mode
intact).*

**Why sat_det3 and not the headline g_det3_wknd spark run:** the p2 spark run has
`decoded_root` on disk only for its first ~13 k events, which end just BEFORE the
spark region (sparks start at eid 12986; local waveforms stop at 12975 — zero
overlap). `sat_det3` (490 V resist / 1000 V drift, the operating point) has full
decoded coverage over all 47 k events and 2878 sparks (9.1 % of firing events).
Same detector, gas, and operating regime — the morphology transfers. (To do this
on the p2 run, pull its remaining `decoded_root` files from lxplus AFS.)

## The headline picture

A spark is **not** a uniform flash and it is **not** a propagating streamer. At the
waveform level it is a **fast, global common-mode excursion**: the entire FEU's
baseline (all 512 channels) steps up together — on ~1/4–1/3 of sparks all the way
to ADC saturation — for a few hundred ns, then decays back to baseline within the
1.92 µs readout window. Riding on top of (and seeding) it is a **genuine, localised
avalanche charge concentrated at one edge** of the plane. The discharge couples to
every channel *simultaneously* (electrical common mode), while the genuine charge
stays spatially confined. This is the behaviour a resistive-strip Micromegas is
designed to produce — the resistive layer localises and quenches the discharge —
and it is what differentiates it from a bare-metallic Micromegas (propagating,
damaging sparks).

## The six measured facts (both planes, spark sample vs normal-muon control)

1. **Common mode IS the spark signature.** Median waveform over all 512 channels:
   normal muons ≈ 0 ADC (median CM peak 9); sparks reach a broad excursion with a
   saturation spike — **31 % (X) / 23 % (Y) of sparks drive the whole FEU to
   saturation** (CM peak > 1000 ADC). [panel a]

2. **The "50+ strips firing" that DEFINES a spark is inflated by the common mode.**
   Raw strips over threshold: median **93**. After common-noise subtraction
   (per-64-channel-chip median per sample), the genuine localised charge is median
   **40 (X) / 56 (Y)** strips — roughly half. Much of the multiplicity is every
   channel crossing threshold off the common-mode swing / cross-talk, not real
   charge. (The Y plane keeps more genuine strips — resistive strips run along y,
   so real charge spreads further, consistent with the sharing measurement.) [panel b]

3. **All at once, not propagating.** For the full-FEU sparks the high-ADC (2000 ADC)
   onset time is **flat across all 512 channels to ~1–2 samples** (median spread
   103 ns; the mode is one 60 ns sample). A charge front crossing the 40 cm plane at
   any physical speed would sweep visibly over many hundreds of ns to µs; it does
   not. The discharge is a simultaneous electrical common-mode step. [panel c]

4. **The genuine charge is drift-ordered, not streamer-ordered.** Within a spark the
   genuine-charge onset vs strip position is only weakly correlated (|corr| median
   **0.21**), and its time spread (~788 ns) ≈ the full drift window — i.e. that
   spread is ordinary micro-TPC drift time of the seed ionisation, not a propagating
   front. The genuine charge sits at **one edge** of the plane (see the gallery),
   consistent with the edge-dominated spark initiation found in the hit-level study. [panel d]

5. **Fast recovery.** The common mode is back below 500 ADC by the end of the
   1.92 µs window on **94 %** of sparks (median CM at window end ≈ 0). This is the
   waveform-level *cause* of the no-post-spark-dead-time result (script 39): the pad
   self-restores well before the next trigger (~270 ms away). [panel e]

6. **Common-mode pulse shape.** Aligned to the discharge step: flat baseline, a step
   to ~saturation in **< 1 sample**, then a decay over ~1 µs back toward baseline —
   a fast-rise / slow-recovery transient on the whole FEU. [panel f]

## Gallery (the "what does it look like")

`spark_waveforms_gallery.png` — raw (CM intact) vs CNS event images, both planes:
- **normal muon**: a single clean diagonal micro-TPC track; raw ≡ CNS (no common mode).
- **localised spark**: genuine charge saturated at one edge + a faint common-mode
  wash across all channels (removed by CNS).
- **full-FEU spark**: the entire array saturates after the step; CNS strips the
  common mode and reveals the edge seed charge underneath.

## Outputs

`<sat_det3>/…/alignment_tpc_veto50/spark_waveforms/`
- `spark_waveforms_gallery.png` — event-image gallery (3 event classes × 2 planes × raw/CNS)
- `spark_waveforms_analysis.png` — the 6 quantitative panels above
- `spark_waveforms.json` — all numbers per plane
- `wf_features.npy` — per-event waveform-feature cache (rerun with `--rebuild`)

## Caveats / how a referee reads it

- CNS uses the per-chip median, so on a fully-saturated chip it subtracts the
  saturated bulk — genuine charge that *also* saturates the whole chip is removed
  with the common mode. So the "genuine strip" count is a floor for the biggest
  events; it does not change the localisation conclusion (the seed is where the
  charge deviates from the chip median, which is exactly the edge cluster seen).
- The front-simultaneity spread (103 ns) is slightly inflated by the edge seed
  charge, which saturates from real avalanche a sample or two BEFORE the global
  step — that ordering is physical (seed → global discharge), and the bulk itself
  crosses within one sample.
- 60 ns sampling sets the timing floor: we can bound propagation only to "faster
  than ~1 sample across 40 cm" (≳ a few mm/ns), which already excludes any
  gas-physics streamer front. Finer timing would need a faster digitiser.

## Where this fits the paper

Topic 6 (HV + sparks). The spark section already has: Poisson-in-time,
muon-induced (4–6×), edge-dominated, non-propagating (hit-level), no post-spark
dead time (script 39). This adds the **waveform-level mechanism**: a self-quenching,
non-propagating, edge-seeded discharge that couples as a global common-mode step and
recovers within the readout window — the resistive-Micromegas spark-protection story,
shown directly in the raw data. Good candidate for one gallery figure + one
multi-panel figure, or folding into `det3_spark_analysis/main.tex`.
