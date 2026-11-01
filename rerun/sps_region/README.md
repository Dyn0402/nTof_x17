# det4 at SPS H4 on one small working region (2026-11-01)

**Answer.** In region **R = det4 X 150–160 mm × Y 222–246 mm** (run 63 flat700, Ar/CF4/iso 88/10/2, resist 700 V,
243 V/cm, ZS 64), with the telescope subtracted:

- **σ_X = 138 µm** (p_mid; f range 112–153, stat ±7);
- **σ_Y = 120 µm** (wft fit p0; f range 89–136, stat ±9; centroid 112, p_mid 104).

Model-free upper bounds (telescope back-plane cluster of 2 strips): **X ≤ 153 ± 6 µm** (front 2-strip too),
**Y ≤ 145 ± 6 µm**.

- Efficiency: **89.9 ± 0.4 %** in both views at once (X 94.6 %, Y 92.0 %). This is a lower bound, because the
  telescope's purity limits it.
- Footprint without zero suppression (run 71 raw700): **3.1 strips (X) and 3.5 strips (Y)**; neighbour ÷ lead 0.41 / 0.36.
- Per-view timing on RAW, (X − Y)/√2: **107 ± 19 ns** (edge) and **69 ± 17 ns** (constant fraction 20 %). It is set by
  the charge: at equal charge R and the whole spot agree, and above ~2500 ADC they reach the bench's 20–30 ns.
- The trigger contributes ≤ 19 ns: the uRWELL's own hit time has s68 = 18.5 ns against it.
- The event time is not measured.

Slide note: <https://dylan-neff.web.cern.ch/notes/mx17-sps-region.html> (generator `make_deck.py`, output
`deck/mx17-sps-region.html`).

## Why a region

The whole-spot numbers (`rerun/sps_audit`) average over three things:

- det4's dead amplification stripes;
- live bands of different gain (lead-strip charge ~330 against ~210 ADC);
- the telescope back plane's three pitch zones.

On det4 X each zone falls on a different live band (zone 1.0 = X 123–184, 1.5 = 185–230, 0.5 = 231–246 mm). So
the free 3-zone fit there compares bands, not zones, and fails (χ²/1 = 7–88).

R is chosen on the maps (deck slides 2–3):

- X lies inside the widest, highest-gain live band (148–162 mm), ≥ 2 mm from both edges.
- Y ≥ 222 mm puts both telescope coordinates in the back plane's 1.0 mm zone.
- R avoids the oscillating Y channels (ch 372–376 at 154.6–158.7 mm, ch 510 at 250.4 mm) and sits in the densest part
  of the spot.
- R2 (X 202–213 mm, same Y) applies the same recipe to the second band. It is a cross-check only, and it is worse there
  (lower gain).

## Method

- **Inputs** (copied to `$SPS_REGION_DATA`, default `$WORK_DATA/sps_region`):
  - the final SPS pass telescope tables `x17rerun/sps_audit/tel/F{53,63,66,71}_*` (code 3ac0e79), slimmed;
  - `sps_audit/footprint/*.parquet`;
  - `paper1_timing/sps/*.npz` (code 48244778);
  - the g_det4 pass bundle (for the timing template);
  - `x17rerun/sps_region/eff/E_*.parquet` (this package, `eff_job.py`).
- **Positions.** Every single-cluster four-view uRWELL track is projected to det4 with the seeded robust affine of
  `sps_audit` (`common.align`). Its predicted det4 X, Y selects R, so tracks that cross a dead stripe are placed where
  they crossed.
- **Resolution** (`region_analysis.resolution`):
  - The affine is refitted on R's own tracks.
  - Cross-view fiducial ±1.5 mm.
  - Core σ of a Gaussian in ±1.2 mm (the `sps_audit` definition).
  - det4 = √(σ²_res − P²), with P the zone-1.0 pointing at z = 1120 mm.
- **Telescope f.** R has one zone, so f is measured on det4 Y over a whole X band, where the 1.0 and 1.5 mm zones are both
  populated. The 0.5 mm zone has only 50–270 tracks at the beam edge, so it is not used. That gives two zones and two
  unknowns, assuming det4 Y is uniform along Y in the band (the gain is flat there).
  - The six fits (3 estimators × 2 bands) give f = 0.254–0.284; the mean is **0.266**. The range is the systematic.
  - Check: the f model predicts the 2-strip subsample's pointing at 82–131 µm in both views. The X/Y difference is
    therefore det4's, not the telescope's.
- **Upper bound.** The residual of tracks whose back-plane cluster (in the measured coordinate) has 2 strips is det4 ⊕
  a smaller telescope term.
- **Efficiency** (`eff_job.py`, 9 run-63 groups, 147 279 tracks). For every single-cluster telescope track in the plateau,
  det4 must have a local maximum ≥ 50 ADC within ±2 mm of the prediction in X and in Y.
  - The window dependence is 86 / 90 / 91 % at ±1 / 2 / 3 mm.
  - det4 has some signal ≥ 50 ADC in both views in 97.8 % of these events.
- **Footprint.** `sps_cluster` quantities per event: run 63 (telescope-matched in R) and run 71 RAW (det4 position in R,
  117 events).
- **Timing.**
  - `paper1_timing` estimators (edge 25 % with the g_det4 template; CF 20 %) on the strip sums, both views ≥ 150 ADC,
    R by det4's own position.
  - (X − Y)/√2 measures the per-view part only. X and Y see the same avalanches, so the particle time, the primary
    clusters and the trigger cancel.
  - `trigger_ref.py` measures the uRWELL lead-hit time against the trigger in run 63 (9 files): s68 16.7–20.3 ns.

## Results (`results/region.json`)

| run / region | estimator | X res. | X 2-strip | X det4 (f range) | Y res. | Y 2-strip | Y det4 (f range) |
|---|---|---|---|---|---|---|---|
| run 63 R | p0 | 343 | 291 | 259 (247–267) | 254 | 145 | **120** (89–136) |
| run 63 R | p_mid | 263 | 176 (b2f2 153) | **138** (112–153) | 247 | 143 | 104 (66–123) |
| run 63 R | centroid | 352 | 302 | 272 (260–280) | 251 | 154 | 112 (78–130) |
| run 63 R2 | p_mid / p0 | 374 | 248 | 173 (124–199) | 283 | 221 | 172 (152–184) |
| run 66 r755+r780 R | p_mid / p0 | 298 | 223 | 196 (179–206) | 257 | – | 126 (97–142) |
| run 53 R | centroid | 347 | 293 | 265 (252–273) | 256 | 192 | 123 (93–139) |

All values in µm. The residual is uniform across R: four Y quarters agree within ±5 µm. The head-on fit angle in R has
θ s68 of 1.5° (X) and 0.7° (Y).

## What this does not rule out

- The subtracted numbers depend on the telescope model and on f. The upper bounds do not.
- det4 X is worse than Y with every estimator. Its charge centroid is worse even for 3-strip clusters. The cause is open.
  The X ±1 neighbour asymmetry is a candidate; it is not demonstrated.
- The **event time** at SPS (the common part) is not measured. The RAW strip sums cover only ~17 s per plateau, so fewer
  than 100 of their events have a telescope match. The (X+Y)/2 spread against the trigger (140–250 ns) contains untagged
  pile-up.
- **Next step, if wanted:** extract the strip sums for the run-71 telescope events and use the uRWELL hit time as the
  reference.
- The patch covers one gas, one gain and 10 × 24 mm². It is not n_TOF's gas, and it does not describe the whole chamber.

## Files

- `common.py`: loading and the telescope→det4 map.
- `region_analysis.py`: maps, f, resolution, footprint, timing, efficiency → `results/region.json`.
- `eff_job.py` with `condor/eff_job.sh`, `eff.sub`, `eff_jobs.txt`: the efficiency denominator table. It ran on lxplus
  (about 2 min per group); the output is in `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/sps_region/eff/`.
- `trigger_ref.py` → `results/trigger_ref.json`.
- `make_deck.py` → `deck/mx17-sps-region.html`.

Reproduce:
1. `SPS_REGION_DATA=<dir> python region_analysis.py`
2. `python make_deck.py`

The data dir holds `tel_run{53,63,66,71}.parquet` (slim copies of the F* tables, on EOS in `x17rerun/sps_region/`), `footprint/`, `timing/`, `bundle/`
and `eff/`.
