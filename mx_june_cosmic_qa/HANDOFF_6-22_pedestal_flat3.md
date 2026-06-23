# Handoff — 6-22 overnight run QA + pedestal flat-threshold study

Run: **`mx17_det2_det3_overnight_6-22-26`** (Ar/Iso 95/5, non-ZS, `det_orientation.z=90`).
Detector slots moved *again* — always re-read `run_config` `detectors[]`:

| det | FEU (X,Y) | z [mm] | position |
|---|---|---|---|
| **mx17_3** | 3, 4 | 232 | bottom |
| **mx17_2** | 6, 8 | 702 | top (X is FEU **6**, not 7) |

`short_run` and `longer_run` share parameters → stats are combined (pooled). There is also
a `resist_525V_drift_1000V` pedthr HV point (not analysed here).

## Data / processing notes
- Pull only file-indices that have BOTH a `combined_hits` and a `rays` file — the DAQ PC
  processes one subrun at a time (decode → analyze_waveforms → combine), so there is a
  backlog while a run is live.
- **Combining subruns:** do NOT merge the `combined_hits_root`/`m3_tracking_root` dirs.
  `evn`/`eventId` restarts at 1 each subrun (globally unique only *within* a subrun), so a
  merged directory cross-pairs detector hits to the wrong M3 rays. Instead run each subrun
  through the full pipeline independently and **sum the per-category ray counts** from each
  `efficiency_breakdown.txt` (`combine_subrun_stats.py`).

## Headline result — efficiency was pedestal-limited, not detector-limited
The hit finder keeps a pulse only if `peak >= 5σ × per-channel pedestal RMS`, and the
pedestal RMS is computed **non-robustly** (no spark/saturation rejection, no common-noise
subtraction). The 6-22 pedestal runs are very noisy, so real micro-TPC signal was being
rejected as noise. The per-FEU **median pedestal RMS anti-correlates with efficiency**:

| det / subrun | median ped RMS (X,Y) | nominal eff |
|---|---|---|
| mx17_3 short | 19 / 22 | 48.0% |
| mx17_3 longer | 92 / 92 | 39.0% |
| mx17_2 short | 118 / 127 | 30.2% |
| mx17_2 longer | 61 / 70 | 48.2% |

### Flat-threshold reprocessing (3 × per-FEU median RMS)
Reprocessed locally with threshold = `3 × median(pedestal RMS)` per FEU (uniform), using
the rebuilt processor (`WFA_THRESHOLD_SIGMA=3` + `WFA_FLAT_SIGMA=<median>` env, see below).
Efficiency **~doubles with resolution preserved**:

| det | nominal (combined) | flat3 (combined) | core σ |
|---|---|---|---|
| **mx17_3** | 42.0% | **70.1%** | 0.65 → 0.65 mm |
| **mx17_2** | 42.2% | **65.7%** | 0.89 → 0.88 mm |

Mechanism: `no_hit` collapses (mx17_3 short 37%→2% — silent muons were sub-threshold) and
`hit_no_reco` drops (mx17_2 short 54%→21% — recovered strips let clusters reach ≥3 strips).
The unchanged core resolution shows this is real recovered signal, not noise.

**Caveat / proper fix:** a flat per-FEU value is blunt where the *median itself* is inflated
by coherent noise (mx17_2 short, median 118). The correct fix is a **robust per-channel RMS**
(MAD / iterative σ-clip) + saturation rejection + common-noise subtraction in
`computePedestals()`; flat-3×median is a clean, resolution-safe stop-gap.

## Files
QA scripts (this repo, `mx_june_cosmic_qa/`):
- `11_pedestal_qa.py <key>` — per-FEU pedestal mean/RMS + threshold plots (reads the
  `pedestals` tree from pulled `hits_root/`).
- `combine_subrun_stats.py` — pool short+longer breakdown counts (nominal).
- `compare_nominal_vs_flat3.py` — nominal vs flat3 efficiency comparison.
- `qa_config.py` keys: `o22_det3 / o22_det2 / o22_long_det3 / o22_long_det2` (nominal),
  `f3_det3 / f3_det2 / f3_long_det3 / f3_long_det2` (flat3).

Processor change (separate repo `~/CLionProjects/mm_strip_reconstruction`, NOT committed
here): added env override `WFA_THRESHOLD_SIGMA` in `WaveformAnalyzer` (alongside the
existing `WFA_FLAT_SIGMA`); default behaviour unchanged when unset. Rebuild:
`ninja analyze_waveforms` in `cmake-build-debug`.

Data / outputs (under `~/x17/cosmic_bench/`, NOT in the repo):
- nominal analysis: `Analysis/mx17_det2_det3_overnight_6-22-26/`
  (`combined_short_longer/` has the pooled figure).
- flat3 reprocessed data + analysis: `det2_det3/_flat3_reproc/` and
  `det2_det3/Analysis/mx17_det2_det3_overnight_6-22-26/` (`_compare_nominal_flat3/` has the
  comparison figure). Reprocess driver: `/tmp/reproc_flat3.py`.
