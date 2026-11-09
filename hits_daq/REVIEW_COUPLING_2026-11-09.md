# Independent review of the capacitive-coupling study (2026-11-09)

Reviewer: a fresh session, on Dylan's request in `HANDOFF.md` ("have another model look at the situation before going
further"). Read: `report_coupling/report.html`, `reports/Capacitive coupling in strip microTPCs.md`, the five research notes,
`coupling/*.py`, `coupling_scan.py`, `P0_XBIAS_2026-11-05.md`, the two literature surveys, `RECONSTRUCTION_BASIS.md`,
`docs/PARAMETERS.md`, `wft/model.py`. New checks run on the det3 Run 1 + Run 2 windows ($WORK_DATA/paper2_hits/windows):
`coupling/depth_ratio.py`, `coupling/phase_closure.py`, `remedies.py`, `ghost_census.py`, `universal.py`, `toy_mixing.py`
(all in `hits_daq/`); outputs under `$WORK_DATA/paper2_hits/{depth,phasecl,remedies}/` and `results/`. Every number below is
in `papers/tpc_mixing_review/NUMBERS.md` (generated).

## Verdict in one paragraph

The study's conclusions hold, and two of them can be stated more strongly. The position-in-strip test survives a closure
with known truth and a refitted p0 (the fit's lean is 0.007-0.07 mm and makes a near/far asymmetry of 0.03 against the
data's 0.08); the floor it finds is real (X 0.060-0.067 from a quarter pitch outward, Y 0.10) and bounds a fixed coupling at
about half the published fractions. The "shares" question (Q2) dissolves once the bias is written as ratio - 1 = C/N^2 + b:
the end coefficient C is measured (7.0 X, 11.7 Y), the model reproduces it to 2 %, and its decomposition by arms (geometry
2.3, footprint +1.5, diffusion +3.1, RC +4.6) is additive and order-independent within 10 %, whereas the "share of coupling vs
footprint" is a parametrisation of the same variance and has no meaning. Fitted k1 against a closure k1 (Q3) means nothing
at chi2/dof 66, as the report says. The chip-boundary test (Q4) is suggestive for the -2 strip and should be quoted with its
errors; it does not isolate electronics for -1. The ~7-9 % floor (Q5) has no identified physics; the review adds that the
real detector has 0.8 more mesh-end ghost strips per plane than the model on X, which is the same floor seen in the hits,
and that the depth-resolved width growth prefers Dp = 0.021 (D_T 350) over the pass's 0.023. The unexplained items (Q6)
remain unexplained; one of them (data below every model at the strip centre) is confirmed on Y and not on X.

## The six questions

**Q1. Is "pure coupling cannot replace sigma0" sound, given that the phase uses the fitted p0?** Yes.
`coupling/phase_closure.py` regenerated 8001 near-normal planes (|tan| < 0.034) under two truths (noK: sigma0 0.318/0.276,
k = 0; pureK: sigma0 0.033, k1 0.131/0.111, k2 0.010/0.020), with the plane's own residual or with fresh white noise, refitted
(p0, w, t0) with the pass fitter (pass hyper, k = 0) and recomputed the early-window near/far ratios from the refitted p0.
Refit minus truth: p0 MAD 0.007 mm (noK) and 0.070 mm (pureK) on X, 0.010/0.032 on Y. Near/far at |phase| 0-0.1 on X: data
0.205/0.124; pureK with true p0 0.160/0.144 (symmetric as it must be), with refit p0 0.174/0.138; noK with white noise
0.242/0.167. The lean exists but is a third of the data's asymmetry. Far side at |phase| 0.4-0.5: data 0.067, noK 0.040, pureK
0.127. So the data have a floor that a Gaussian of the fitted width lacks, at half the level of the pure-coupling replacement:
"pure coupling rejected, a same-time floor of ~6 % present" is right on X. On Y: data far 0.099 at the edge against noK 0.068
and pureK 0.132 (floor ~10 %), but the centre bin (0.216/0.159) is BELOW the Gaussian's 0.279/0.209: the Y footprint core is
narrower than 0.276 mm. Caveat that stays: the 'own-residual' arms (M_truth + (W - M_pass)) reproduce the data trivially
because the residual carries the unmodelled floor; only the white-noise arms are a closure. The report's phase figure
should be re-drawn from `phasecl/full/summary.json` (figure `phase_closure_x.png` there).

**Q2. Is any split of the same-time sharing into coupling and footprint meaningful?** No, and the paper should not offer
one. The two add to the cluster variance alike and the compression is invariant along the chi2 valley (the report's own
table). What is meaningful is the decomposition of the END coefficient by physical arm, which is additive:
`universal.py` on the resim tables gives a = 0.0018 (geometry), 0.0029 (+ footprint, no diffusion), 0.0042 (metallic:
diffusion + floor footprint), 0.0053 (model), 0.0070 (footprint 0.45); Y: 0.0056 (RC off), 0.0089 (model), 0.0130 (2x RC).
In C = a (L/p)^2: 2.3, 3.8, 5.4, 6.9, 9.1; Y 7.3, 11.5, 16.8. Data: 0.0054/0.0056 (X, Runs 1/2), 0.0090/0.0088 (Y) -> C 7.0-7.3
and 11.4-11.7. The white-noise toy gives the same increments with a constant offset of ~1 (its C: 1.3, 2.8, 4.3, 6.0, 8.0), and
C = 1.1 + 11 (sigma_rms/p)^2 with sigma_rms^2 = sigma0^2 + Dp^2 T/2 fits all 13 toy configurations to 0.06. Replace "shares"
by this decomposition in the paper.

**Q3. Does fitted k1 = 0.13/0.11 against 0.10/0.09 from a no-coupling closure mean anything?** No. The closure itself shows
the fit returns k1 = 0.10 for a Gaussian truth; the excess 0.03 is below any uncertainty one could assign at chi2/dof 66. The
waveform chi2 constrains the second moment of the same-time spread, not its shape; only the phase test (Q1) and the depth
stacks constrain the shape. Drop k1 as evidence; keep it as a fit result.

**Q4. Does the chip-boundary test isolate electronics?** For the -2 strip it is the only handle and it is suggestive (X
1.34 +- 0.04 % inside vs 0.45 +- 0.11 % across, 33+33 planes; Y 1.69 +- 0.03 vs 0.86 +- 0.22). Two caveats to state: (i) a
boundary strip's -2 neighbour in channel space is some other strip, so the "across" value is board coupling + nothing, while
"inside" is board + electronics: the difference, not the across value, is the electronics; (ii) the depth stacks show the
data's second-neighbour share at t0 to be 2.4 % per side on BOTH views (sum of both sides 0.049 +- 0.004 X, 0.047 +- 0.001 Y)
against 0.3-0.8 % for any Gaussian arm, and at depth (1000 ns) 0.27 vs 0.24 for the model on X: the tails are heavier than
Gaussian at the mesh AND at the cathode, so not all of the -2 excess can be a same-time electronic copy (an electronic copy
would not grow with depth beyond the model's growth). Quote: "mostly electronic at the mesh end; heavier-than-Gaussian tails at
depth, origin not identified."

**Q5. What physics gives a 7-9 % same-time floor and a 0.25 mm width?** Nothing identified; the review did not find it
either. What the review adds: (a) the depth-resolved width growth (`depth_ratio.py`, 569 X / 744 Y planes with |tan| <
0.008) gives d(sigma_eff^2)/dt = 3.5e-4 mm^2/ns on X against 4.2 for the pass (Dp 0.023), 3.2 for Dp 0.020 -> Dp ~ 0.021,
D_T ~ 350 um/sqrt(cm), 8 % below the pass; on Y 11.9 vs 12.6 (the RC term dominates); (b) the intercept (same-time width) is
0.113 mm^2 in the data against 0.071 for the pass on X, i.e. the mesh-end footprint on X is wider than 0.30 (0.33-0.37
equivalent) while on Y it is narrower (the tight sample's intercept is ~0); this is the strip-width equalisation acting on X
only and the pass's single sigma0 for both views is a compromise; (c) the ghost census (`ghost_census.py`, Run 1): data have
3.39 ghost strips per plane on X at |tan| 0.08-0.15 against 2.57 for model + noise, all of the excess at the MESH end (1.57
vs 0.82; cathode 1.82 vs 1.75); Y agrees within 0.3. The floor is therefore not an artefact of the early window: the same
strips fire as hits in the data and not in the model. Candidates the review considered and could not confirm: the
0.80/0.78 mm beat (the report's Rayleigh test rules out pinning at 0.80; the spec's 0.80 is "from memory" and should be
checked against the board drawing), direct comb-to-comb capacitance (derivative-shaped, bounded by the report), electronics
(-1 does not change across boundaries). The paper should present the floor as measured and unexplained.

**Q6. The unexplained items.** (a) The -1 ratio falling with amplitude (8.2 -> 5.4 %) is what an additive constant on the
neighbour does: (kA + c)/A; c ~ 10 ADC in the early window would explain it with a true ratio of ~5 %. Worth one check of the
baseline/common-mode residual on the neighbour strips in the early window before quoting the trend. (b) "Data below every
model near the strip centre": confirmed on Y (far 0.159 at |phase| 0-0.1 against the Gaussian's 0.209 and pureK's 0.144; near
0.216 against 0.279) and NOT on X in the depth stacks at t0 (data 0.374/0.218, pass 0.377/0.210). On Y the core is narrower
than the fitted sigma0. (c) "Inside the track the data shift half as much as any arm with diffusion" (p0_depth: inner slope
-30 ns/column vs -55 model): still unexplained; the toy gives an interior compression of only 0.5 % (b = +0.005) for a
leading edge with this diffusion, the data's universal fit gives b = -0.003/+0.017 (consistent with ~0), the resim arms give
b = -0.01 to -0.02 (the 60 ns depth-quantisation artefact at steep angles). The model's interior compression of 5 % in
p0_depth is probably that artefact rather than physics; the data's 1.7 % is near the toy's expectation.

## Added later the same day: the other chambers (`chambers_C.py` -> `results/chambers_C.json`)

Same statistic on the clean bootstrap windows of B (det2), A (det3), E (det4), D (det7), truth = the fit's w, L = 28.08 for
all, model arms regenerated on the window's own strips (no padding: absolute C ~10-15 % below the padded resim arms and
the full-sample det3 values, so compare chambers relatively). Data C_X: 5.2+-0.2, 6.1+-0.1, 8.0+-0.3, 7.2+-0.8 (B, A, E, D)
vs model 5.5, 5.8, 7.2, 5.1; geometry-only 0.9-1.4; metallic 4.2-5.2. Data C_Y: 10.1+-0.3, 9.7+-0.2, 9.1+-0.3 (B, A, E; D's
198 Y planes do not fill the bins) vs model 10.4, 10.4, 8.2; the model's RC increment (C_Y minus RC-off) 4.9, 4.5, 2.1 orders
with D_RC (5.1, 3.8, 2.2e-4). P0's inclined slope ratios order all four with D_RC on Y (1.27, 1.37, 1.42, 1.53 for E, A, B, D).
Chamber E's X value (8.0 vs toy law 5.8) is the outlier, as in P0 (its model over-predicts the hit bias at large angles;
dead/low-gain bands; column 2.5 mm short). Chamber C (det6) has no bootstrap extraction (92 % saturated, clean sample
unrepresentative) and is not included.

## What the paper can say (replacing the report's section 6 wording)

"Single-time-per-strip reconstruction reads tracks too steep because the strips at the two ends of the drift column receive
charge of the wrong depth: the partial end strips by geometry, the strips beyond the column through any lateral spread. The
ratio tan_rec/tan_true - 1 is C/N^2 with N the strips in the column; on chamber A, C = 7.0 (X) and 11.7 (Y) against the
reference, reproduced by the forward model and decomposed into strip geometry (2.3), the same-time footprint (+1.5), drift
diffusion (+3.1) and resistive spreading (+4.6, Y only). A same-time neighbour share of about 6 % (X) to 10 % (Y) is present,
has a flat position-in-strip dependence and a second-neighbour part that follows the channel map; its label does not change
the compression. The published attribution to capacitive coupling of 15-30 % is not supported on this chamber, and a toy
without any coupling reproduces the published biases of the 5-8 mm chambers."

## Things to change in existing documents

- `LITERATURE_MIXING.md` Sec. 1 point 2 ("no paper states that diffusion alone compresses the ladder"): now add Fang et al.
  (MPGD2024 poster, first-electron diffusion mechanism for the midpoint) and Herrmann 2019 (strip-count statement); see
  `literature_survey/gapfill_2026-11-09.md`.
- `report_coupling/report.html` section 3 ("data lie below every model near the strip centre"): true on Y, not on X (depth
  stacks at t0); section 5 (shares): replace by the C decomposition; add the closure figure.
- `OUTLINE_PLAN_2026-11-05.md` claim paragraph: "delayed copies" and "whatever the estimator" as already flagged; add "C/N^2".
- `docs/PARAMETERS.md`: note that the depth-resolved stacks prefer Dp 0.021 (D_T 350) and a per-view sigma0 (X wider than Y);
  not a change of the pass, a recorded tension.

## Files

- `coupling/depth_ratio.py` -> `$WORK_DATA/paper2_hits/depth/full/{summary.json,planes.parquet,stacks.npz,figures/}`
- `coupling/phase_closure.py` -> `$WORK_DATA/paper2_hits/phasecl/full/{summary.json,rows.parquet,figures/}`
- `remedies.py` -> `$WORK_DATA/paper2_hits/remedies/{run1,run2}.{json,parquet}`
- `ghost_census.py` -> `results/ghost_census_sat_det3.json`
- `universal.py` -> `results/universal.json`, `report_coupling/figures/universal.png`
- `toy_mixing.py` -> `results/toy_{arms,pitch,estimators,detectors}.json`
- the review paper draft: `papers/tpc_mixing_review/` (main.tex, sections/, figures/, NUMBERS.md, scripts/)
