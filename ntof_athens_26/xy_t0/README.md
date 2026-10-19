# xy_t0 — what the two planes agree on, and what `dt_xy` is

Follows [`HANDOFF_T0_PRIOR.md`](../chi2_bimodality/HANDOFF_T0_PRIOR.md), which
measured `x_t0 − y_t0` on the published sample, found a 66–77 ns half-width with
a >60 ns tail on ~43 % of tracks, read it as the **60 ns depth-bin degeneracy**,
and asked for four things: count the `dt_xy` fallback, measure `dt_xy` in situ,
build a t0 prior, judge it on that residual.

This does the first two. Doing them shows the third and fourth were resting on a
misreading.

```bash
X17_ROOT=D:/x17 ../../.venv/Scripts/python.exe xy_t0.py              # tables, ~2 min
X17_ROOT=D:/x17 ../../.venv/Scripts/python.exe make_xy_t0_figures.py # four figures
X17_ROOT=D:/x17 ../../.venv/Scripts/python.exe make_xy_t0_report.py  # figures/report.html
```

`xy_t0.py` writes to `<out>/xy_t0/`; the two `make_*` scripts only draw and only
report. Nothing here is re-reconstructed — every number comes from the stage-3
track table, so nothing in this package can change a fit.

---

## 1 · The answer, short

**The handoff's measurement stands. Its diagnosis does not, and its proposed
test cannot be run as written.**

| | |
|---|---|
| the planes really do disagree | 66–77 ns half-width on the published sample, reproduced exactly |
| **but not at 60 ns** | a periodicity test that sees the known 5 ns `T0_STEP` snap at 8–19 σ finds 60 ns at \|σ\| ≤ 1.0 |
| **and not from geometry** | the lowest-inclination quintile already carries 65–70 ns; `tan θ` adds ~10 |
| what it is | the per-plane t0 is *smoothly* uncertain at ~55–60 ns, against a fitted error of a few ns |

Same scale as one depth bin — which is why the bin was a tempting explanation —
but a broad uncertainty, not a two-minimum ambiguity, and it needs a different
fix from a prior that picks between two minima.

## 2 · The census, exactly (handoff §3 step 1)

The handoff expected "~100 % miss on A and C" and asked for it to be confirmed
rather than trusted. It is not approximately 100 %.

| arm | tracks | used the measured `dt_xy` | bundle keys | `ftst_diff` in data |
|---|---:|---:|---|---|
| A | 454 180 | **0** | −5, 1 | −4, −2, 0, 2, 4 |
| B | 360 869 | **0** | −5, 1 | −4, −2, 0, 2, 4 |
| C | 632 629 | **0** | −1, 5 | −4, −2, 0, 2, 4 |
| D | 663 484 | 229 532 (34.6 %) | −3, 3 | −5, −3, −1, 1, 3, 5 |

`ftst` is a 6-phase counter. A, B and C produce only **even** `ftst_diff` and
their keys are **odd**, so the intersection is empty by construction — no
statistics involved. **10.9 % of the campaign used a measured offset**; the rest
ran on the hardcoded −18.8 ns, unflagged and uncounted.

## 3 · `dt_xy` is a readout-clock effect, and needs no bench (§3 step 2)

It is not a per-chamber constant. The **whole** `x_t0 − y_t0` distribution
translates linearly with `ftst_diff`, by the same amount on all four arms:

| arm | measured ns per `ftst` unit | max residual to the line |
|---|---:|---:|
| A | −8.77 | 3.0 ns |
| B | −8.35 | 0.6 ns |
| C | −7.67 | 5.4 ns |
| D | −8.31 | 12.1 ns |

`ftst` has 6 phases over a 60 ns sample, so one unit is **10 ns** of readout
phase. The estimator is biased low in a known direction — whatever accidental
pedestal survives subtraction does not shift with `ftst` and pulls the
correlation peak toward zero lag — so this is consistent with the quantum
without proving equality. Either way there is **no chamber physics in `dt_xy`**,
and the beam data measures it for every class that actually occurs.

> ⚠ `shift_ns` and `dt_insitu` are not equally trustworthy. The **shape** is good
> to a few ns. The **absolute anchor** is a median on a 200 ns-wide peak and the
> whole column can move ~10 ns together. No acceptance or efficiency number is
> re-derived from it here, deliberately.

## 4 · The two mechanism tests

**Not the depth bin.** Two near-degenerate minima 60 ns apart would put a comb in
the residual at 60 ns. The Rayleigh statistic finds +1.0, −0.5, −0.9 σ against a
null of arbitrary periods — while the same statistic finds the known 5 ns
`T0_STEP` snap at 8.3, 18.9, 8.6 σ. **The test has demonstrated sensitivity and
sees nothing at the predicted period.**

**Not the plane separation.** The handoff called this "the first thing to check
and it could explain a large part of the table". A separation crossed by an
inclined track must vanish at normal incidence. It does not:

| arm | half-width at lowest `tan θ` quintile | at highest |
|---|---:|---:|
| A | 65.7 ns | 74.8 ns |
| C | 70.3 ns | 79.4 ns |
| D | 68.1 ns | 75.1 ns |

Geometry is present and is ~10 ns of a 66–77 ns spread.

## 5 · The gate is the metric — §1 is truncated, §3 step 4 is circular

`wft.reco.select_tracks` sets `gated` from `|(t0x − t0y) − dt| ≤ 120 ns` **and**
both planes plausible. **Zero of 2 111 162 gated tracks lie outside that
window** — the identity is exact, not approximate.

So the handoff measured the x/y residual on a sample already cut on that very
quantity, at ±120 ns around a centre wrong by up to 84 ns. The §1 numbers are not
wrong; they are a truncated view:

| arm | gated (what §1 measured) | all stage-3 | ungated |
|---|---:|---:|---:|
| A | 69.2 ns | 106.9 | 184.5 |
| C | 78.0 ns | 133.4 | 272.9 |
| D | 76.8 ns | 141.8 | 330.3 |

12–15 % of gated tracks sit within 20 ns of the cut edge.

**This retires §3 step 4 as written.** A prior that moved t0 would move which
tracks pass the cut, so the residual on the survivors would improve partly by
construction. Declare the target on the unselected population instead.

## 6 · What the coincidence test is actually worth

Against a scrambled pairing (same arm, same `ftst` class, another track's
`y_t0`) the ±120 ns window keeps 67–73 % of true pairs and 29–38 % of accidental
ones — an enhancement of **1.9–2.4×**, bought for ~30 % of the real tracks.
Useful; not the discriminator `select_pair`'s docstring describes when it claims
information "that single-plane selection cannot use".

Background-subtracted, the agreement peak is **78–85 ns** half-width on the
difference, so **55–60 ns per plane**, against a fitted `x_t0_err` medianing
3.8–10.6 ns on the full population and 2.6–7.4 ns on the selected one. **The
reported t0 error is understated by 5× to 21×**, which is its own finding and
affects anything that propagates it.

## 7 · Figures

| figure | what it is |
|---|---|
| `xy_agreement` | **the answer.** `x_t0 − y_t0` per `ftst` class, with the scrambled pairing under it. The shape slides — that is `dt_xy`. Its width is the resolution. |
| `dt_xy_law` | the linear law on four arms against the 10 ns phase quantum, and where the bench keys sit relative to the classes the beam produces. |
| `mechanism` | the two tests: the 60 ns comb with its 5 ns positive control, and the inclination dependence. |
| `gate` | the gated sample against the distribution it was cut from, and true-vs-scrambled acceptance. |

`figures/report.html` is the write-up; the DAQ Analysis tab serves it.

## 8 · What this does not settle

- **The in-situ t0 prior is not built** (handoff §3 step 3). This supplies the
  `dt_xy` law it would need and removes the reason to expect it to collapse the
  residual — a ~60 ns per-plane uncertainty is not something a better centre
  repairs.
- **Why the per-plane t0 is uncertain at one bin is not answered.** That it is
  smooth rather than combed rules out the two-minimum picture; it does not say
  what replaces it. That is a waveform-path question.
- **Nothing is re-reconstructed.** Whether fixing `dt_xy` in the bundles changes
  any physics result needs a refit, which is not this.
- **The absolute `dt_xy` level is provisional** — see the warning in §3.
