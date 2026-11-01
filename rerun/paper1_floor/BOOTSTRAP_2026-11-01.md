# Model-free residual bootstrap of the clean-event angle remainder (2026-11-01)

The pass and the reconstruction model are unchanged: this is a study on top of pass 1. The saturation decision stays locked.

## Verdict: the data's own fit residuals carry the remainder, and no single named structure does

1. **Data residuals move the angle 1.5–2.0× more than the toy's own residuals.**
   - Each data plane's fit residual was moved onto a toy event's pass model and refitted with the production `fit_plane`.
   - Same machinery, same toy truths: det2 / det3 / det4 X give 0.243 / 0.241 / 0.416° against 0.123 / 0.162 / 0.283°.
2. **Calibrated, the excess explains about half to all of the remainder.**
   - The method only sees what the donor's own fit did not absorb. Its efficiency, measured with toy residuals on toy truths, is ε = 0.29–0.37.
   - Scaled by ε, the excess is **105 ± 17 % (det2), 59 ± 10 % (det3) and 54 ± 11 % (det4)** of the clean-event remainder² (U_cond, `README.md`).
3. **Cross-check on data truths.** Data residuals on data planes reproduce the measured clean resolution with the same efficiency as the toy's residuals reproduce the toy's: dD / ε / measured = 0.99 / 0.81 / 1.12 / 1.12 on det2 / det3 / det4 / det7 X. The Y view gives the same dD / measured ratio as X (0.23–0.35 against 0.30–0.35).
4. **The dominant structure is not any of the four named candidates.**
   - Every structure carries the same share of the residual's angle effect in data residuals as in toy residuals (table below).
   - **What carries the angle,** in both: the part that is coherent in time *and* across strips (shuffling either destroys ~80 %), and antisymmetric about the track (~40 %).
   - **Ruled out as the data-specific term:**
     - the chip-coherent baseline (~0);
     - the mean template-shape mismatch (removing it makes things slightly worse);
     - a per-plane t0/shape jitter (≤ 0.14);
     - a distinct ±1-strip asymmetry (the same share as the toy's).
5. **So the remainder is more of the same.** It is the event-to-event, fluctuation-like charge pattern the toy already makes (ionisation, avalanche and diffusion fluctuations about the smooth model), only larger in the data. The refit χ²/dof says the same: 26–55 with data residuals against 3.5–17 with toy residuals (the data planes themselves: 10–83).
   - **For the paper:** the remainder is the toy under-generating per-event charge-pattern fluctuations, not a definable model error, which is why no model term found so far removes it.
   - **Untested candidates** (simulation only): the Polya θ, the δ-ray rate and range, the cluster-size distribution, and per-event resistive spread.

## Numbers (calibration rows, clean events = the §8 'topo' sample; s68 in degrees, FD convention, the data's angle mix)

| | clean s68 / U_cond | toy L8 | T (toy res.) | N (noise only) | D (data res.) | D/T | ε = T/L8 | (U_boot/ε)² / U_cond² | χ²/dof data / D / T | data truth dD / measured, / ε |
|---|---|---|---|---|---|---|---|---|---|---|
| det2 X | 0.730 / 0.621 | 0.374 | 0.123 | 0.008 | 0.243 | 1.97 | 0.33 | **1.05 ± 0.17** | 83 / 55 / 17 | 0.33, 0.99 |
| det3 X | 0.761 / 0.630 | 0.444 | 0.162 | 0.012 | 0.241 | 1.48 | 0.37 | **0.59 ± 0.10** | 52 / 26 / 12 | 0.30, 0.81 |
| det4 X | 1.486 / 1.420 | 0.973 | 0.283 | 0.036 | 0.416 | 1.47 | 0.29 | **0.54 ± 0.11** | 10 / 6.0 / 3.5 | 0.33, 1.12 |
| det7 X | 0.722 / 0.562 | 0.585 | 0.183 † | 0.004 | 0.128 | — | — | — † | 54 / 29 / 60 † | 0.35, 1.12 |
| Y views | 0.68–1.02 | 0.47–0.80 | 0.31–0.33 † | ≤ 0.013 | 0.17–0.26 | — | — | — † | | 0.23–0.35 |

- † **Toy-truth comparison not valid** where the toy's residual is already larger than the data's (refit χ²/dof T ≥ D):
  - **Y views:** the Y toy is X physics fitted as Y (`README.md`, Limitations);
  - **det7 X:** the toy's det7 residual χ² is 60 against the data's 28.

  These rows rest on the data-truth arm only.
- The shares are (D² − T²) L8² / (T² U_cond²). The errors propagate the bootstrap errors of L8, T and D.
- **Electronics noise alone (N) moves the angle by ≤ 0.04°.** The toy's 0.37–0.97° floor is almost entirely the event's own charge-pattern fluctuations.

**Each structure's share of the residual's own angle effect**, 1 − s68(with it removed)² / s68², on toy truths, X:

| structure removed | det2 data / toy | det3 data / toy | det4 data / toy |
|---|---|---|---|
| chip-coherent baseline (quiet-strip per-sample median) | +0.01 / +0.11 | −0.17 / +0.10 | +0.05 / +0.08 |
| per-plane time shape ({M, dM/dt, (t − t_pk) dM/dt} projection) | +0.13 / −0.21 | −0.02 / +0.11 | +0.14 / +0.03 |
| antisymmetric ±1…3 strips about the peak strip | **+0.40 / +0.38** | +0.31 / +0.49 | **+0.41 / +0.42** |
| samples shuffled within a strip | +0.83 / +0.79 | +0.79 / +0.84 | +0.81 / +0.82 |
| strips shuffled within a sample | +0.84 / +0.87 | +0.85 / +0.87 | +0.83 / +0.87 |
| the mean fractional pattern (data only) | −0.10 | −0.12 | −0.06 |

- **Data truths** give the same pattern on every chamber and view: antisymmetric 0.19–0.63, time shuffle 0.57–0.90, chip baseline −0.67…+0.31 (noisy on det7 Y's 198 planes), time shape −0.47…+0.47, mean −1.16…+0.32.
- Relative to the excess D² − T² alone, the removals read 0.53–0.76 (antisymmetric) and > 1 (shuffles). That only says removing them hits the excess and the toy-like part alike. The like-with-like table above is the discriminating one.

## Method

- **Clean events** (`boot_ids.py`): the §8 ladder's 'topo' step on the frames (unsaturated, amplitude ranks 10–80 %, one candidate, one track, no band strip, strip-count and multiplicity excess below p90). It reproduces `floor_budget.json`'s topo s68 to the printed digit. Planes per row and view: det2 980 / 1150, det3 3045 / 3074, det4 3034 / 3411, det7 177 / 198.
- **Data residuals** (`boot_extract.py`, run through `rerun/stage_job.py --no-sync`, so nothing is written to Analysis_pass1):
  - the production reconstruction is replayed for those events: the same hits, seeds, windows, `fit_plane_candidates` and bundle t0 prior;
  - stored per plane: the gain-corrected window W, the pass model M on it, R = W − M, noise, censored rows, gain, positions and channels;
  - **replay closure:** w reproduces the table's on **100.0 %** of 2129 / 6119 / 6388 / 375 planes.
- **Toy truths** (`boot_run.py toy`):
  - the FD toy's full L8 events (FD_BASIS=pass: pass-1 bundles, interior placement, the row's q50 and v(t)), 600 per angle point (3 / 9 / 16°), fitted with the production fit;
  - the truth is the production model at that fit, exactly representable, so a refit of the model alone returns its angle;
  - the s68s are pooled with the data's angle mix.
- **Moving a residual** (`transplant`):
  - the donor is matched in angle bin and model peak (ratio 0.75–1.33), nearest in t0 phase among up to 15;
  - lean-oriented, aligned on the peak strip and peak sample (integer samples), in ADC;
  - uncovered or censored cells are filled with Gaussian noise at the truth window's noise;
  - refit with `wft.reco.fit_plane` and the bundle's t0-prior width, centred on the truth's t0;
  - **identity check:** a plane's own residual, unmoved, gives d = 0 to 10⁻⁴°.
- **Ablations** (`ablate`), on the donor's residual in its own frame before the move: the definitions are in `boot_run.py`'s docstring.
- **Data truths** (`boot_run.py data`): the same arms with up to 600 clean data planes as truths and donors from the same row (the truth plane excluded).
- **Campaign:**
  - code `d5df8f04` (extraction and main arms), 92 nodes; `ec93b75d` (toy-residual ablations), 9 nodes; both DAGs exited 0;
  - raw outputs: `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/floor_boot/raw/`;
  - data pickles in `.../floor_boot/data/`;
  - analysis: `boot_analyze.py` → `results/bootstrap.json`, `boot_summary.py` → `results/bootstrap_summary.json`.

## Limitations

- **A residual bootstrap only sees what the donor's fit did not absorb.**
  - The part of any error that looks like a change of angle (or of q in a depth bin) is invisible in R. Hence ε ≈ 0.3.
  - The calibrated shares assume the data's excess transplants with the toy residuals' efficiency. The equal structure shares support this; they don't prove it.
  - The raw, uncalibrated shares are 5–11 %.
- **Y and det7 rest on data truths only.** Their toy residuals already exceed the data's (Y: the X-generation Y toy; det7: toy χ² 60 against 28).
- **"Antisymmetric about the peak strip" is crude for a tilted track.** The charge moves across strips with depth, so the strip about which the pattern is antisymmetric changes along the column.
- **Alignment approximations:** integer-sample alignment (t0 phase matched, not interpolated), and ADC not rescaled within the ±33 % peak window. Both weaken coherent structures slightly and apply to D and T alike.
- **Sample:** in-sample calibration rows only, and det7's pools are small (177 / 198 planes).
- **Not tested here:** which fluctuation the toy under-generates. Separating Polya θ, δ-rays, cluster sizes and per-event spread would be a toy study; no new data is needed or possible.
