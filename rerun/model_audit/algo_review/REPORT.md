# Algorithm review: is the voxel-profile + sliding-line fit the right algorithm? (2026-10-28)

Dylan's ask, before publication: "we are kind of splitting into voxels and sliding our track model around. Is this
the best way or is there something more elegant?" Independent read of `wft/model.py`, `wft/reco.py`, `wft/seed.py`,
the audit ledger and the paper-2 sections, plus a numerical check on 800 det3 bench events (M3 reference in the cache).

## Verdict

**The structure is right, and it has a name: variable projection.** The fit is a separable nonlinear least-squares
problem. The K depth-bin charges are *linear* parameters with a non-negativity bound; the three *nonlinear* parameters
(p0, w, t0) are what the line is. Eliminating the linear part at every nonlinear point (the NNLS inside `chi2_plane`)
and optimising only the three outer parameters is exactly the variable-projection method of Golub and Pereyra
(SIAM J. Numer. Anal. 10 (1973) 413), with Kaufman's simplification (BIT 15 (1975) 49); the bound-constrained form is
Sima and Van Huffel (J. Comput. Appl. Math. 203 (2007) 264, verify before citing). So "voxels + sliding the line" is
not an ad-hoc construction, it is the textbook way to fit a straight track whose emission profile along the track is
unknown and non-negative, through a known response. The paper should say so in one sentence; it makes the method
look like what it is.

**The search is not what limits the result.** A derivative-based outer solver (Levenberg–Marquardt on the
variable-projection residual, prototyped here) reaches the same angle as production in 99 % of planes at one fifth of
the cost, and the resolution against M3 is identical to 0.01°. A joint x–y fit with one shared charge profile (the one
structurally different formulation, `wft.model.fit_joint`, unused in production) changes s68 by 2 %, within the
statistical error. Cutting the depth grid to the drift gap changes nothing. Every alternative lands on the same 1.0°,
because 1.0° is set by the model (det3 χ²/dof 70 against 4–40 for a perfect model in the ws_floor toy; floor
0.30–0.35°). The elegance gains below are for the write-up, for throughput, and for the per-event errors; none of
them moves the headline numbers. **Decision (Dylan, 10-28): implement the LM solver and the covariance errors inside
B1, before the pass**, since everything is rerun from scratch anyway; spec in HANDOFF.md ("Fitter: LM
variable-projection outer solver"). The "not before the paper" caution in an earlier draft of this report is
withdrawn: it only applied to numbers already quoted, and none are.

## What was measured (det3, 800 events, both views, phys kernel hypers of `rerun/precheck_rcv/out/arm_det3_rcvk24.json` on the r06 det3 bundle's templates, v 36.6, K 24, PIVOT_START, no t0 prior)

### 1. Production search vs a gradient solver

`prod` = `wft.reco.fit_plane` as shipped (pivot scan, then `fit_plane_raw`: 105-point grid + Nelder–Mead, then six
curvature evaluations). `lm` = the same pivot scan and the same 7-point coarse t0 ladder for the start, then
Levenberg–Marquardt on (p0, w, t0) with the Kaufman projected Jacobian (finite differences of `build_matrix`, no extra
NNLS), covariance from JᵀJ.

| | prod X / Y | lm X / Y |
|---|---|---|
| s68 vs M3 (°) | 1.079 / 0.999 | 1.087 / 1.010 |
| far (> 5°) | 4.0 % / 2.5 % | 3.9 % / 2.5 % |
| NNLS solves per plane (median) | 316 | 59 |
| wall per plane (median) | 0.62 s | 0.12 s |
| same w within 1 µm/ns | — | 99.1 % |

The prototype LM stops looser than production in (p0, t0) (42 % of planes end > 1 χ² unit above production, 8 %
below): the χ² surface is piecewise smooth (NNLS active-set changes, the 5 ns tensor interpolation) and a
finite-difference LM without a polish is not the final word. The point is that the angle is the same wherever either
converges. With analytic derivatives (the erf footprint's derivative is a Gaussian, the t0 derivative is the template
derivative) and a short Nelder–Mead polish this is a drop-in replacement; the gain is cost and a proper covariance.

**Errors.** The shipped per-event errors are 1-D curvatures that ignore the p0–t0 correlation (the code says 20 %
optimistic). The full 3×3 covariance says ×2: p0 0.008 → 0.017 mm, t0 1.9 → 3.8 ns, w 0.026 → 0.053 µm/ns (medians);
|ρ(p0, t0)| reaches 0.75 at the 90th percentile. Both are far below the measured resolution, which is why F3 replaced
them with a per-chamber floor; the covariance is the honest statistical part if it is ever used.

### 2. The 60 ns t0 basins

Shifting the production minimum by one depth bin, (p0 ∓ 60 w, w, t0 ∓ 60), re-solving the profile and refining:

| | |
|---|---|
| raw shifted point, Δχ²/χ² (median) | +13 % (the lattice shift is NOT a symmetry of this kernel) |
| refinement walks back to production's basin | 84 % |
| refinement stays in the other basin **and is deeper** | **10.3 %** (X 13 %, Y 7 %) |
| … deeper by (median) | 25 units of χ²/dof, t0 later by 57 ns |
| … angle there | the same: median \|Δw\| 0.07 µm/ns |

So production misses a deeper t0 basin on one plane in ten, always one bin later, never at a different angle. On the
bench t0 is now pinned at the DAQ latency (F1/F6), which removes it. Off the bench it is a search defect worth two
extra refinements (start at t0 ± 60 as well). It is **not** the x–y t0 inconsistency of F5: taking the deeper basin
wherever it exists moves the x–y t0 width only from 67 to 63 ns rσ. That width is the model's per-plane t0
definition, not the optimiser.

### 3. Depth grid pinned to the drift gap (K = 13 instead of 24): null

χ² +0.13 %, w the same in 99.4 % of planes, t0 moved 0.0 ± 0.9 ns, q_sum ratio 0.995, x–y t0 width 58–69 vs 60–78 ns.
The far end of the column is soft (attachment: the arrival current falls with depth) so the grid end never bites;
this is the k30 null of F13 seen from the other side. The idea was to replace the plausibility window on q_uend and
the column guard with the gap length in the model; the data say there is nothing there to gain.

### 4. Joint x–y fit with one shared charge profile (`fit_joint`, seeded from production, t0 tied by the measured per-class dt_xy)

| | right pairs (400) | wrong pairs (x of i with y of i+1, 400) |
|---|---|---|
| cost of sharing, (χ²_joint − χ²_separate) / (χ²/dof), median | 200 | 1850 |
| relative cost | +18 % | +167 % |
| per-plane scale α, rσ | 0.048 (= the ln qx/qy width) | 0.45 |
| wrong above the right 95th percentile | | 83 % (AUC 0.95) |
| s68 X / Y (°) | 0.998 / 0.967 (prod 1.020 / 0.982 on the same rows) | |
| wall per pair | 2.5 s | |

As a pairing statistic it works but is weaker than det3's charge balance (96–100 %, satstudy), and it buys 2 % in
s68, inside the error. The +18 % χ² on genuine pairs says the two views' profiles and tied t0 are not consistent at
the χ² level under the present model (the stale dt_xy and its 60–70 ns width are part of it). It is the right
*formulation* of a 3-D track (five parameters, one profile) and would absorb the pairing gates, but it is not worth
the swap at this model fidelity. Kept as the candidate for a future unified reco, not for the paper.

## Where the headroom actually is (none of it is the fitter)

Measured 1.0–1.9° against a 0.30–0.35° floor; data χ²/dof 60–700 against 4–40 for a perfect model. The known
model gaps, all already on the ledger: the one-sided X copy (board, unmodelled; −0.05° mean, ∓0.19° head-on),
the Y template carrying RC undershoot, the brightness-dependent shallow reading on det6/det7 (5–10 %, OPEN), the
60 ns charge bins for steep tracks (−0.5° at 40°, 20 ns fixes it in the toy, 3× cost, untested on data), and
the residual 2 % angle compression after the bow map. A better optimiser finds the same minimum of the same
wrong model.

## Recommendations, in order

1. **Paper 2, §4:** present the fit as variable projection. One sentence: "The charge profile enters linearly and is
   eliminated at each step by non-negative least squares (variable projection, Golub and Pereyra 1973); the outer
   problem is three-dimensional." Cite Kaufman 1975 for the Jacobian form if a derivative solver is adopted, and the
   T2K ERAM analysis as the simultaneous-waveform-fit precedent (already in REFERENCES.md).
2. **B1, before the pass:** replace `fit_plane_raw`'s grid + Nelder–Mead with LM on the
   variable-projection residual with analytic derivatives plus a short polish; keep the pivot scan and the coarse t0
   ladder as the global start; add t0 ± 60 ns to the start set. ~5× fewer solves per plane, continuous t0, full
   covariance. Gate it the way every change is gated (fleet gate on the golden rows); expect bit-identical angles.
3. **Per-event errors:** if anything downstream weights by them, use the covariance (×2 the curvature errors) or the
   per-chamber floor; never the 1-D curvatures.
4. **Not worth doing:** gap-pinned grid (null); joint shared-profile fit as a resolution lever (null); any further
   search-strategy work on the bench (pivot + pinned t0 is already at the model limit).

## Limits

det3 only, one bundle (r06 templates with the precheck phys hypers, not the promoted `final_phys` bundle), no dead
list, no t0 prior, X-view resolution numbers with a ladder scale refitted per arm rather than the bundle's kw/w0.
The LM arm is a finite-difference prototype (`algo_review.py: lm_refine`), not production code. s68 on 400–800
events carries ~3–5 % statistical uncertainty, which is why 2 % differences are called null.

## Files

`algo_review.py` (prod / lm / shift / kgap arms, 14 workers, ~2 min for 800 events), `algo_summary.py` →
`results/rows800_summary.json`; `joint_review.py` (right and wrong pairs), `joint_summary.py` →
`results/joint_rows_summary.json`. Row pickles are in the job's scratch directory, not kept.
