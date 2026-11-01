# paper1_spsfigs — the SPS figures of Papers I and II on the final SPS chain (2026-11-01)

**Verdict.** All five figures that were pending the SPS final pass are now built from it and marked current in the
`figures.toml` manifests.
- **sps_resolution:** only the **Y** view gives an intrinsic resolution: run63 p0 **190 ± 10 µm** (zone fit χ²/1 0.3).
  The **X** view has no measurement in the 3-zone sense: its zone fits fail (χ²/1 7–88), so "X 209 µm" is a
  model-dependent intercept.
- **known_angle:** at a known 25.4° stage angle, the inclined view reads **0.981 ± 0.03** of the stage tan once the drift
  velocity is taken from the column duration. With the registry v it reads 0.927, falling to ~0.5 at 110 V/cm.
- **kernel_forms:** the head-on sharing kernel is **field-independent** from 96 to 243 V/cm, so it is not diffusion. Its
  Y ±2/±1 is 0.26, not the 0.42–0.47 of the August stacks.
- **angled_kernel:** the X kernel is unchanged by a 25.4° rotation.
- **attachment:** draws the SPS gas fit against the bench.
- **det2 edge-T tension:** closed.

## Figures (`figures/`, copied into `papers/paper1|paper2/figures/` by `papers/make_figures.py`)

| figure | paper | input | key numbers |
|---|---|---|---|
| `sps_resolution` | I §8 | `rerun/sps_junk/results/final_rule.json` (telescope tables on the final oscillator rules) | run63 Y: p0 190 ± 10 (χ²/1 0.28), p_mid 217 ± 12 (6.8), centroid 199 ± 10 (11.9). run63 X: p0 314 ± 8 (7.3), p_mid 209 ± 11 (20.3), centroid 280 ± 6 (88.5). run53 Y: p0 262 ± 12 (0.31), hits 176 ± 10 (1.26), centroid 155 ± 7 (2.7), p_mid 292 ± 17 (8.4) |
| `kernel_forms` | II §2 | `kernel_job.py` on run71 raw700/450/275 (condor 4605384) | ±1/lead X 0.377, Y 0.390; ±2/±1 X 0.19, Y 0.26 (243 V/cm), the same within 0.01 at 156 and 96 V/cm; X ±1 asymmetry −0.07…−0.085 (low side larger), Y ≤ 0.01; n ≈ 2100–2200 per view |
| `angled_kernel` | II §2 | `kernel_job.py` on run63 flat700, run63_rot25 d425, d325 | X ±1 0.427/0.360 flat vs 0.430/0.353 (d425) vs 0.429/0.336 (d325); Y ±2 0.09–0.10 flat vs 0.12–0.13 at d425 (the geometry) |
| `known_angle` | II §8 | `tan_known.py` on the pass tables + `vbox.py` (19 new rows) → `results/known_angle.json` | 25.4°: column v 0.981 ± 0.03 (13 rows ≥ 180 V/cm), registry v 0.927 ± 0.04 (6 rows ≥ 190 V/cm); 15.465°: registry v 0.80 (3 rows), no column v |
| `attachment` | I §6 | `rerun/sps_gasfit/results/verdict.json` (4127f9ea) + `rerun/paper1_gas/results/summary.json` | SPS CF₄ water 1.68–1.85 %, O₂ 129–258 ppm; CO₂ water 1.9–2.1 %, O₂ 322–443 ppm (gap 27.9 mm); bench 0.42–1.00 % water, attachment-free at steady state |

`results/figs_summary.json` holds every number above, written by `make_figs.py`.

## The X zone fits (sps_resolution)

- **The model:** σ_res² = σ_det4² + b·pitch², fitted over the three back-uRWELL pitch zones (0.5, 1.0, 1.5 mm). On run63,
  for each estimator:

  | estimator | χ²/1, Y | χ²/1, X |
  |---|---|---|
  | p0 | 0.3 | 7.3 |
  | p_mid | 6.8 | 20.3 |
  | centroid | 11.9 | 88.5 |

- **Why X fails:** the zones are det4 regions too. det4's X view has real dead/weak amplification stripes inside the beam
  spot, while its Y view has no dead band under the telescope (`rerun/sps_audit/README.md`). So in X each zone samples a
  different det4 resolution, and the intercept is not det4's alone.
- **The per-zone values after subtracting the binary pointing** are non-monotonic in pitch: p0 338/290/282 µm,
  p_mid 239/153/159, centroid 222/282/196.
- **Status: documented, not resolved.** Resolving it would need a per-stripe det4 X resolution model, which this data
  does not constrain.
- **For the paper:** quote Y as the measurement (run63 p0, or run53 hits as the cross-check), and either drop X or give
  it as a per-zone range. This is folded into decisions A/B in `rerun/PAPER1_NUMBERS_2026-10-31.md`.

## The known angle

- **Method:** tan = w / v, so a row fitted with a wrong v reads tan × v_fit / v_true.
  - The registry v of the CF₄ rotated ladders (11.5–14 µm/ns) is the retired low-field estimate
    (`rerun/sps_followup/README.md`).
  - The column-duration v (`vbox.py`: v = 28.8 mm / T of the stacked strip sum, defined where the box and flat-box
    fits agree to 8 %) needs the column inside the 64-sample window, so roughly ≥ 180 V/cm.
- **Result:** at those fields, the 13 rows on the 25.4° mount read 0.467–0.496 against tan 25.4° = 0.475. The stage
  angle is the measured yaw (DET4_URW_MAPPING); the config said 25.64°.
- **Gap:** det4's geometric gap is 28.6 ± 0.5 mm (337f7fd1), not 28.8. That lowers every column v by 0.7 % and moves
  the ratio to 0.988.
- **The 15.465° rows (run61):** 32-sample windows, so no column v, and the stage angle comes from the config, not a
  measurement. With the registry v they read 0.80 of tan 15.465°, so the angle there is not established.
- **Sign:** only |tan| is defined at SPS (COORDINATES.md §4b).

## The det2 edge-T tension (rerun/consistency) — closed

- **The tension:** S4's head-on stack gives T(edge) − T(interior) = −86 ± 33 ns; the edge study's column duration
  gives −4 ns.
- **Why they differ:** they average different bands of a strongly non-monotonic T(d) profile:
  - T ≈ 584 ns in the outer 0–5 mm, where the column runs off the strips;
  - 915–950 ns at 5–15 mm, the fringe;
  - 800–860 ns further in.

  S4's out30 is the M3 reference < 30 mm from **any** side. 40 % of that sample sits in the outer 0–5 mm, and 69 % is
  nearest a y side.
- **Test:** weighting the edge study's own column-duration profile with S4's distance distribution predicts
  −78.5 ns, against S4's −86 ± 33 (`det2_edge_check.py`, `results/det2_edge_check.json`; the S4 head-on cache read
  from the job's tmp).
- **Rule:** quote edge T only with its band definition.

## Limitations

- **Gap (attachment):** the SPS compositions are quoted at the gas fit's 27.9 mm gap. det4's geometric 28.6 mm moves
  CF₄ water by about −0.1 %, inside its error (see the GASFIT row of the numbers sheet).
- **Gap (bench det4):** the bench det4 water (0.42 %) still uses the old 30.48 mm gap, and needs §6 to redo it.
- **ZS selection:** `angled_kernel` uses ZS rows with the pass selection (peak ≥ 300 ADC, ≥ 4 strips with samples).
  That biases the ±2/±3 responses upward through the zero-suppression floor. Flat and rotated rows are biased alike, so
  the comparison holds; the absolute values do not. Use `kernel_forms` (RAW) for absolute numbers.
- **Kernel definition:** `kernel_forms` takes lead amplitudes of 400–3000 ADC. Below that, the ±2/±3 responses sink
  into the noise.
- **Missing inputs:** the August `angled_kernel` (`kernel_ab.png`) read staging npz files that are no longer on disk,
  so it could not be byte-compared. It is remade here on the pass cache instead.

## Files

- `make_figs.py`: all five figures, and writes `results/figs_summary.json`.
- `attach_fig.py`: the attachment figure.
- `kernel_job.py`: the per-row kernel on the final cache builder (condor).
- `tan_known.py`: the known-angle table (run on lxplus over the pass tables).
- `det2_edge_check.py`: the det2 edge-T check.
- `condor/`: `job.sh`, `job.sub`, `jobs.txt` (cluster 4605384, 25 jobs, all done).
- **Outputs on EOS:** `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_spsfigs/` (`kernel/`, `vbox/`, code
  `code_d9dd97fc`). The kernel npz files (12 MB) are gitignored.
