# The sharing kernel, measured

Measures the MX17 resistive sharing kernel's **shape** from head-on beam data
without deconvolving anything, decides between the candidate forms, and turns
what survives into a change to `wft`.

Read `report.html` first (`make_report.py` builds it from the JSON products, so
re-running the analysis moves the numbers, tables and verdict together).

## Why

Every shipped calibration bundle carries `c2 > c1` — the ±2 strip receiving
more than the ±1 strip (det3 1.14, det2 1.53, det7 1.75, det4 2.12). No lateral
transport can do that. The 2026-08-17 audit showed it is **not** a bound
artefact: the ref-pinned cosmic χ² is genuinely flat in that direction, so the
fit walks there and stays. The cure is an external measurement.

## The method

At normal incidence every strip is driven by the same signal `C(t)` through a
lateral transfer function, so

    W_d = n_d * C    =>    n_0 * W_d  ==  n_d * W_0

`C` cancels identically and **both sides are measured data convolved with a
model filter** — no inversion, no Wiener filter, no regularisation parameter.
Causality also kills the truncation problem: `(n*W)[i]` only reaches back to
`W[<=i]`, and the pre-pulse region really is empty (1.7 % of peak), so the tail
past the window is never needed. This replaces the regularised deconvolution of
2026-08-17, whose λ moved the answer by ±10 % in area and 250→400 ns in τ.

The cancellation holds **only** head-on. That is why the flat runs can do this
and the angled ones cannot.

## What it found

| | |
|---|---|
| the shipped `delay` form | 4.2 % cross-relation residual |
| a cascade of one-poles (`lp`) | **2.1 %** |
| no lateral sharing at all | 16.5 % |

- `c2/c1 = 0.45 ± 0.02` on the beam **in the shipped form**, at all three
  drift fields; `0.63 ± 0.09` on near-vertical det3 bench cosmics. Nothing
  wants > 1.
- The ladder constraint `c2 = c1²` holds to 7 % (`c2/c² = 0.93 ± 0.01`).
- Y's constants sit still to 4 % over 95 → 243 V/cm. **X is not head-on** —
  `q(+1)/q(-1)` = 0.6–0.7, the flat mount's known 0.2–0.4° tilt — and gives no
  stable constant.

## What it could not deliver

**The absolute τ is not a constant.** It walks 629 → 1040 ns as the fit window
is lengthened from 600 to 1800 ns, monotonically: the measured tail is heavier
than one exponential, so the lumped cascade is the best of the forms tried, not
the right one. A diffusive RC-continuum kernel (`t^-3/2` tail, one parameter)
is the obvious next candidate and has not been tried.

Consequently the RC form is **not adoptable yet**. Transplanted onto det3 it
costs σ_θ(Y) 1.14 → 1.54°. The 1.92 µs bench window cannot see the tail that
sets the constants, which is also why det3's own fit gives τ = 375 ± 198 ns.

A trap that cost a round: at a **matched** window the two chambers agree
(det4 c = 0.51–0.53, det3 c = 0.42 ± 0.05). The raw ±1/centre peak ratios,
0.31 vs 0.50, look like a factor-1.6 disagreement and are not — the peak ratio
folds in the shaping and the window. Compare fitted constants at a matched
window, never peak ratios.

## What ships

`wft/model.py::build_matrix` gained one gated branch: a `c2_over_c1` hyper that
slaves `c2 = r*c1` before the per-plane `kY`/`cX` scaling. No existing bundle
carries the key, so nothing changes silently. Covered by
`wft/tests/test_c2_ratio.py`.

Checked against the frozen MPGD26 manifest, **only det3 and det7 were ever
affected** — det2 (0.74), det4 (0.67) and det6 (0.82) already ship physical
bundles. Both have been refit at r = 0.45/0.6/0.8 and scored on held-out
cosmics against the M3 reference (paired bootstrap):

| | det3 prod | det3 r=0.6 | det7 prod | det7 r=0.6 |
|---|---|---|---|---|
| σ_θ X | 0.996° | 0.996° | 1.354° | 1.404° |
| σ_θ Y | 1.143° | 1.145° | 1.458° | 1.471° |
| Δσ_θ Y paired | — | +0.028 ± 0.062 (0.5σ) | — | +0.023 ± 0.080 (0.3σ) |
| raw Y slope | 0.9876 | 1.0012 | 1.0150 | 1.0260 |
| held-out χ² | — | −1.1 % | — | +0.8 % |
| c2/c1 | 1.14 | 0.60 | 1.75 | 0.60 |

**It is free in resolution** — under 0.6σ in every plane of both detectors —
and it removes a free hyper. Ratios anywhere in the measured 0.45–0.8 are
indistinguishable, so the ratio needs to be *below 1*, not precisely pinned.

**Do not sell it as an angle-scale fix.** The constraint shifts the raw
`w → tan` slope by a near-constant **+1.5 %** on both detectors, which lands
det3 on 1.0012 and pushes det7 from 1.0150 to 1.0260. That is a fixed shift,
not a correction toward truth; det3 simply happened to be under-reading by
about the same amount. And a per-plane `kw` (det3 Y: 1.024) absorbs a global
slope anyway. `w0`/`kw` are measured from an existing reco, so the new bundles
carry the old pair and are stamped `w0_kw_stale`.

Bundles written: `calib_bundle_r06` beside each production bundle
(`20_make_ratio_bundle.py`). **Nothing is re-frozen** — the manifest still
points at the old bundles. det7 additionally runs kY to 4.7–6.0 against a bound
of 6, which is its long-standing v ↔ sharing degeneracy and is not addressed
here.

## Files

| | |
|---|---|
| `stacks.py` | per-event peak-aligned neighbour stacks, all three run_71 RAW drift plateaus, both views → `stacks_run71_raw.npz` (25 MB, not in git) |
| `forms.py` | the candidate kernels and the cross-relation residual |
| `fit_kernel.py` | the form comparison + paired bootstrap → `fit_kernel.json` |
| `systematics.py` | window / basis / gate / alignment → `systematics.json` |
| `bench_kernel.py` | the same measurement on near-vertical det3 cosmics → `bench_kernel_y.json` |
| `make_figures.py`, `make_report.py` | `figures/*.png`, `report.html` |

Bench arms live in `mx_june_wft/`: `17_ladder_recal.py` (RC kernel),
`19_ratio_recal.py` (the c2 slave), `18_ladder_bench.py` (scores them all).

## Next

1. **Re-freeze.** det3 and det7 are refit and their `calib_bundle_r06` written,
   but nothing points at them yet. Re-freezing means: swap the manifest, re-run
   the reco, then re-measure `w0`/`kw` with `bench/set_w0.py` (they are stale by
   construction), then re-run the downstream digest. That is a condor campaign,
   not an inline step.
2. **The continuum kernel.** Fit `G_d(f) = exp(-|d| sqrt(i ω τ_D))` — the
   semi-infinite RC line — and see whether it holds τ_D fixed across fit
   windows where the lumped cascade does not.
3. **A head-on X measurement.** Needs a mount alignment the flat runs did not
   have, or a tilt term in the model.
