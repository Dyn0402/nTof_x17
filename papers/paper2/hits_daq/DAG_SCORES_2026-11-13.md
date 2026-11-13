# Scores of the 11-09 analysis DAGs (4646000, 4647206) — 2026-11-13

Both DAGs ended 11-11 with no failures: 4646000 120/120 nodes (`001` = `005` = 120, all exit 0, no evict/hold),
4647206 22/22 (same). Logs `~/x17rerun/hits_daq/ana_2026-11-09{,e}/log/ana.log` on lxplus. Tables on EOS
`cosmic_bench/x17rerun/hits_daq/results/`; laptop copy `~/work_data/paper2_hits/results1111/`.

## 1. SPS chamber E (det4), H2c with the row's own v and gap (`h2crow_sps_*`)

Scored with `sps_score.py --tables <dir with tiers_sps_* and h2crow renamed h2c_sps_*>` →
`results/sps_h2crow_score.json`. Old bench-v H2c on the same planes: same command on the original `h2c_sps_*`.
The 25.4° stage view is Y (inclined); v on these rows is set from W (v_stage = median|w_W| / tan 25.4°), so the ratio
column is the hits ladder against W on the same planes, and W's own accuracy is Paper I's known-angle check.
3000 planes per row and view.

Inclined view (Y), s68 [deg] and median u / w_W:

| row | W s68 | V0 ratio / s68 | V0r ratio / s68 | H2c bench v: ratio / s68 | H2c row v: ratio / s68 | H2c-V6 ideal, row v |
|---|---|---|---|---|---|---|
| run57 CO2 d550 | 1.10 | 1.129 / 1.47 | 1.046 / 1.12 | 1.010 / 2.38 | 0.983 / 1.99 | 1.014 / 1.53 |
| run57 CO2 d600 | 1.02 | 1.099 / 1.40 | 1.039 / 1.07 | 1.001 / 2.60 | 0.976 / 1.73 | 1.007 / 1.45 |
| run57 CO2 d650 | 1.03 | 1.091 / 1.44 | 1.034 / 1.02 | 1.023 / 2.52 | 0.978 / 1.60 | 1.007 / 1.37 |
| run57 CO2 d700 | 1.03 | 1.082 / 1.45 | 1.030 / 1.05 | 1.046 / 2.41 | 0.985 / 1.57 | 1.006 / 1.36 |
| run59 CO2      | 1.03 | 1.074 / 1.49 | 1.028 / 1.05 | 1.049 / 2.48 | 0.986 / 1.50 | 1.015 / 1.27 |

Findings:
- **The compression at a known angle:** the DREAM hits ladder reads 7–13 % steeper than W at 25.4° (V0 ratio
  1.07–1.13), falling with drift field (d550 → d700). This is IIb's known-angle point (no M3, no v involved).
- **The row's v fixes H2c's scale, not its spread.** Ratio 1.00–1.05 → 0.98–0.99 and s68 2.4–2.6° → 1.5–2.0°, but H2c
  still has 30–35 % of planes outside ±5° (in5 0.66–0.69 against W 0.77) and loses to the plain robust line V0r
  (s68 1.02–1.12°, though V0r is biased steep by +0.6–1.0°). The 11-09 tuning finding stands: about a fifth of H2c fits
  run away, seeded by a hits line far off W. H2c on the DREAM hits is not a usable known-angle reconstruction here.
- **Normal-incidence view (X):** H2c s68 2.1–3.1° against V0 0.4–0.6° and W 1.3–1.6°. At tan ≈ 0 the hits ladder has
  a few strips at one time, the line returns ~0 by construction (coverage 0.85–0.90), so the V0 number is not a
  resolution; H2c runs away there. Quote only the inclined view.
- run55 (0° flat drift) has no `h2crow` table (not in the DAG); its old `h2c_sps_run55` exists.

## 2. det3, estimators V7 (charge-weighted time) and V8 (ToT midpoint), full sample (`tiers2_*`)

`tiers.py --from-tables tiers2_<k>_det3_p*of*.parquet --families cw,totmid` → `results/ana1109e_tiers2_{sat,p2}_det3.json`.
V0 from `results/ana1106_tiers_*`. s68 [deg] against M3.

| sample, view | n | W | V0 naive / H0 | V7 naive / H0 / H2a | V8 naive / H0 / H2a | implied v V7 / V8 |
|---|---|---|---|---|---|---|
| sat X | 6191 | 1.04 | 3.18 / 2.06 | 3.21 / 2.25 / 2.13 | 4.67 / 2.71 / 2.62 | 45 / 56 |
| sat Y | 6175 | 1.01 | 3.75 / 2.68 | 3.74 / 2.83 / 2.37 | 4.01 / 3.26 / 2.70 | 55 / 121 |
| p2 X  | 19202 | 1.04 | 3.02 / 2.11 | 3.09 / 2.27 / 2.16 | 4.04 / 2.78 / 2.63 | 46 / 57 |
| p2 Y  | 19162 | 1.01 | 3.68 / 2.61 | 3.68 / 2.83 / 2.36 | 3.83 / 3.32 / 2.68 | 57 / 127 |

Findings (confirm the 210-plane smoke and `remedies.py`):
- **Charge-weighted time does nothing for the compression** (naive identical to V0, H0 0.1–0.2° worse): a time
  per strip cannot remove charge of the wrong depth, whichever way it is weighted (IIb §remedies).
- **ToT midpoint is worse everywhere.** Its ladder is ~1.2× (X) and ~2.2× (Y) flatter in time than the rising edge
  (implied v 56 / 124 against 45 / 56 µm/ns): the fall time grows with amplitude and with the Y view's RC tail. Consistent
  with amplitude walk; a per-strip check needs the windows, not done.

## Use in the papers
- IIa: the H0 / H2c / W table row (D02). The SPS row gives H2c 1.5–2.0° at a known angle with the row's v, W 1.0–1.1°.
- IIb: known-angle compression ratio 1.07–1.13 (chamber E, CO2, 25.4°), and V7/V8 in the estimator table.
