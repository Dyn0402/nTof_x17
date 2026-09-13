# HANDOFF — the fitted t0 is unpinned campaign-wide, and the two planes prove it

**Written 2026-09-12, out of `chi2_bimodality/`.** Local work only; nothing
re-reconstructed.

Companions: [`RUN145_R06_2026-08-19.md`](../../ntof_tracking/RUN145_R06_2026-08-19.md) §1
(why the prior was dropped), `wft/model.py` module docstring (the degeneracy),
[`README.md`](README.md).

---

> ## ANSWERED 2026-09-12 — read [`xy_t0/`](../xy_t0/README.md) before acting on this
>
> Steps 1 and 2 of §3 are done and they changed the picture. **The measurement
> in §1 stands; the mechanism named for it does not, and the test proposed in
> §3 step 4 cannot be run as written.**
>
> | this handoff says | measured since |
> |---|---|
> | "expect ~100 % miss on A and C — confirm it" | **exact**: 0 of 454 180 A tracks, 0 of 360 869 B, 0 of 632 629 C. `ftst` is a 6-phase counter, A/B/C emit only even `ftst_diff`, their keys are odd — empty by construction. 10.9 % of the campaign used a measured `dt_xy`. |
> | "measure `dt_xy` in situ, per run condition" | done, and it is **not a per-chamber constant**: the whole distribution translates linearly with `ftst_diff` at −7.7…−8.8 ns per unit on all four arms, against the 10 ns one-phase quantum. A readout-clock effect with no chamber physics in it. |
> | "that is the degeneracy landing in different minima" (§1) | **not supported.** No 60 ns comb: \|σ\| ≤ 1.0 against a null of arbitrary periods, while the same test finds the known 5 ns `T0_STEP` snap at 8–19 σ. |
> | "bound the geometry first — it could explain a large part of §1" (§4) | done, and it explains ~10 ns of a 66–77 ns spread. The **lowest-inclination quintile already carries 65–70 ns**, and a plane separation must vanish at normal incidence. |
> | "judge it on the x/y residual in §1" (§3 step 4) | **circular.** `gated` *is* `\|(t0x−t0y) − dt\| ≤ 120 ns` and both plausible — 0 of 2.11 M gated tracks lie outside. §1 measured a distribution already cut on that quantity, around a centre wrong by up to 84 ns. Unselected the half-widths are 107/133/142 ns, not 66/76/77. |
>
> **What it actually is:** the per-plane t0 is *smoothly* uncertain at ~55–60 ns
> — background-subtracted the agreement peak is 78–85 ns wide on the difference
> — against a fitted `x_t0_err` that medians a few ns. Same scale as one depth
> bin, which is why the bin was a tempting explanation, but a broad uncertainty
> rather than a two-minimum ambiguity, and it needs a different fix.
>
> §3 step 3 (build the prior) is still open, and `xy_t0/` supplies the `dt_xy`
> law it would need. But **stop expecting a prior to collapse §1's table**: a
> better centre does not repair a 60 ns per-plane spread, and the target must be
> declared on the unselected population or the improvement is partly by
> construction.

---

## 0 · Correcting the thing that started this

An earlier note out of `chi2_bimodality` said chamber D's t0 prior was
"disabled by a falsy zero" while A's and B's were on, and offered that as a
candidate explanation for D's χ². **That was wrong, and the correction matters
more than the original claim.**

`ntof_tracking.wft_beam.make_bundle` drops `t0_abs` and `t0_prior_sigma` when it
seeds a beam bundle **from every arm alike**, deliberately, and the condor job
re-asserts it after seeding (commit `11ce347`). The reason is good: the bench
`t0_abs` is an *absolute* arrival time in the readout window measured against
the bench's own scintillator trigger and DAQ latency. An n_TOF-triggered run has
a different trigger and a different latency, so the bench table is a wrong answer
stated to ±5 ns — and at σ = 5 ns it is effectively a hard pin to the wrong
place. Measured on run_145 tag 004: with the bench prior the fitted t0 sat at
236 ± 80 ns, pulled onto the bench's per-bin table (222–305); without it the free
fit medians 30 ns.

So: **the t0 prior is off for A, B, C and D in every campaign product, by
design.** It explains no chamber-to-chamber difference in χ², and D's bench-side
`t0_prior_sigma = 0.0` sits upstream of a step that zeroes all four anyway.

The real issue is the one that decision leaves open, and nobody has measured it
until now.

## 1 · What being unpinned costs, measured

`wft/model.py` says the χ² surface has **near-degenerate minima 60 ns apart —
one depth bin** — because the charge profile can shift a bin while `p0` slides by
`w·60`, and that only ~35 % of free fits land in the physical one.

There is an independent check on this in the data that needs no prior and no
simulation. **The x and y planes of one chamber see the same primary charge at
the same instant**, so `x_t0 − y_t0` must equal the FEU clock offset and nothing
else. Selected campaign tracks (gated, angle-calibrated, DCA < 30 mm, paired),
after subtracting the offset the reconstruction itself used:

| arm | n | residual half-width (p16–p84)/2 | **\|res\| > 30 ns** (half a bin) | **\|res\| > 60 ns** (a full bin) |
|---|---:|---:|---:|---:|
| A | 16 877 | 66 ns | 67.6 % | **40.2 %** |
| C | 15 063 | 76 ns | 72.6 % | **46.5 %** |
| D | 13 481 | 77 ns | 72.8 % | **46.7 %** |

The per-track fitted error `x_t0_err` medians **2.6 ns (A), 3.4 (C), 7.4 (D)**.
The two planes of the same chamber disagree by **more than a full depth bin on
four tracks in ten**, while each plane reports a few-nanosecond error. That is
the degeneracy landing in different minima on the two planes, and it is happening
at scale.

Note it is nearly the same on all three chambers (66/76/77 ns), so **this is not
the explanation for the χ² ordering either.** It is a separate, campaign-wide
defect that the χ² investigation happened to walk into.

## 2 · A second problem found on the way: `dt_xy` never applies on A and C

`wft.reco.select_pair` picks the x/y cluster pair whose `t0x − t0y` matches the
measured FEU offset, looked up as:

```python
dt = cal.dt_xy.get(int(ftst_diff), -18.8)
```

The bench bundles carry **two** `ftst_diff` keys each. The beam runs' actual
`ftst_diff` values do not overlap them, and on A and C they cannot, by parity:

| arm | bundle `dt_xy` keys | `ftst_diff` seen in beam data | overlap |
|---|---|---|---|
| A | 1, −5 (odd) | 0, ±2, ±4 (**all even**) | **never** |
| C | −1, 5 (odd) | 0, ±2, ±4 (**all even**) | **never** |
| D | 3, −3 | ±1, ±3, ±5 (odd) | 34 % of tracks |
| B | 1, −5 | — | not in this sample |

So for **every** A and C track and two thirds of D's, the x/y coincidence test
that is supposed to reject coherent noise in one plane runs against a **hardcoded
−18.8 ns**, not a measured offset. The fallback is not flagged, not counted, and
not in any provenance.

That −18.8 is also doing real work: it is the only thing keeping `select_pair`'s
coincidence test honest, and `select_pair` is the step whose docstring says it
uses information "that single-plane selection cannot use".

## 3 · What to build

1. **Count the fallback first.** One line in `select_pair` recording whether
   `dt_xy` hit or missed, aggregated per run per arm into stage-3 provenance.
   Everything below is guesswork until it is known how often the measured offset
   was actually used. Expect ~100 % miss on A and C from the parity argument —
   confirm it rather than trust this table.
2. **Measure `dt_xy` in situ, per run condition.** It is an FEU clock offset
   keyed by `ftst_diff`; it does not need the bench. The beam data measures it
   directly: the mode of `x_t0 − y_t0` per `ftst_diff` class, on tracks where
   both planes are unambiguous (one candidate each, so `select_pair` never
   chose). That is a self-consistent measurement from the same runs and it
   covers the `ftst_diff` classes that actually occur.
3. **Then build an in-situ t0 prior** — the thing `make_bundle`'s docstring says
   to wait for (`keep_t0_prior=True` "only once an in-situ t0 calibration for
   THIS run exists"). The n_TOF trigger gives an absolute time reference the
   bench never had: `t_since_flash_ns` is per trigger and the scintillator tags
   exist for the arm that fired. Build the per-`ftst` prediction from the beam
   data itself, per run condition, and check it against the 23 July boundary
   (CLAUDE.md) before pooling across it.
4. **Judge it on the x/y residual in §1, not on χ².** That table is the right
   metric: it needs no model of what t0 should be, only that two planes of one
   chamber agree. A correct prior should collapse the > 60 ns fraction from ~43 %
   toward the few-nanosecond fitted errors. **Declare that target before
   re-running.**

## 4 · What would make this wrong

- **If the 60 ns disagreement is physical rather than a fit failure.** The
  x and y strips sit at slightly different depths in the stack, and a genuinely
  inclined track crosses them at different times. Bound this from the geometry
  before blaming the fitter — an x/y plane separation of a few mm at the drift
  velocities in use (26–40 µm/ns) is tens of nanoseconds, which is the same order
  as the effect. **This is the first thing to check and it could explain a large
  part of the table in §1.**
- If `ftst_diff` is not stable within a run, a per-`ftst_diff` offset is the
  wrong parameterisation and step 2 fits noise.
- If the free fit's t0 is genuinely bimodal per plane rather than scattered, the
  half-width in §1 is the wrong summary and the fraction beyond one bin is the
  only meaningful number there.

## 5 · What this does *not* justify

Turning the bench prior back on. `keep_t0_prior=True` with the bench `t0_abs`
would pin the beam fits to a bench clock at ±5 ns, which is the failure mode
`RUN145_R06_2026-08-19.md` §1 was written about. The prior is only worth having
once step 3 exists.
