# TODO — investigate M3 reference degradation (6-23 det3_det4 run)

**Status: open, investigate next session.** Flagged 2026-06-24.

## Symptom
On the first **`mx17_det3_det4_overnight_6-23-26`** bench, the M3 reference tracker is
badly degraded — efficiency/alignment for BOTH mx17 detectors is blocked by it (not a
detector problem; the mx17_3/mx17_4 chambers are alive at raw level).

Measured on `longer_run` (indices 000+001), from the pulled `rays.root`:

| quantity | value | healthy bench |
|---|---|---|
| Chi2X median | **65** | ~0 |
| Chi2Y median | **97** | ~0 |
| frac Chi2X < 20 | 0.29 | ~0.9 |
| frac Chi2Y < 20 | 0.25 | ~0.9 |
| clean single tracks (both chi2<20) | **3.9%** | ~54% |
| good-track frac, pre→post chi2 cut | 41.8% → 3.9% | ~unchanged |

Consequence: with the wide M3 lever arm (stations at z = 24 / 144 / 1185 / 1302; rays use
Z_Down=24, Z_Up=1302) those poor fits project to garbage at the detector plane z=232/702
→ alignment starves (only ~34 M3-matched detector events, ≈ random-coincidence level;
median radial residual ~185 mm; no sub-mm convergence even with `Z_LO=150 Z_HI=1100`).

## Hypotheses to check
1. **Noisy-pedestal / threshold issue now hitting the M3 FEU** (same root cause as the 6-22
   run — non-robust pedestal RMS, no common-noise subtraction; see
   `HANDOFF_6-22_pedestal_flat3.md` and `processing-pedestal-threshold-suppression`). A
   noisy M3 FEU 1 → wrong hit picked per station → bad line fit → high chi2.
2. **M3 station alignment / geometry off** for this new det3_det4 bench (stations physically
   moved; the rays-producing tracker config may not have been re-aligned).
3. M3 detector HV / gas state this run (check `hv_monitor.csv`, `sub_runs[].hvs` FEU 0/1).

## Where to look
- `rays.root` chi2 already inspected (above). Next: the per-station M3 hit maps /
  multiplicity (`02_m3_reference_qa.py o23_long_det3` → `m3_up_down_positions.png`,
  `m3_chi2_distributions.png`, `m3_track_multiplicity.png`).
- The DAQ-side M3 tracking config that produces `rays.root` (chi2 is computed upstream, not
  in this repo) — `dream_config/CosmicTb_MX17.cfg`, M3 pedestals/alignment.
- Compare against a known-good bench's `rays.root` chi2 (e.g. day_det1 / a 6-22 subrun).

Data: `~/x17/cosmic_bench/det3_det4/mx17_det3_det4_overnight_6-23-26/longer_run/`.
qa_config keys: `o23_long_det3`, `o23_long_det4`.
