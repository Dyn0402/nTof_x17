# KERNEL_ARMS_2026-08-12 — per-plane τ and the X-sharing attribution (T2.1 + T1.2 arms)

Follow-on from `T0_PRIOR_2026-08-11.md` (the σ=5 trigger prior is ON in every
arm below — it is the recommended configuration) and
`SHARING_DEPTH_2026-08-11.md` (the model-free mechanism measurement).

## 1 · The naive kTauY restoration is REJECTED

Context: the 2026-08-06 merge dropped the code that read `kTauY` (per-plane
RC constant, measured τ_X 230 / τ_Y 410 ns) while the bundles kept the value;
the RC-ladder production line used it (`ANALYSIS_STATE §2.1`). Restoring the
*code path* and letting the shipped model read lp2's `kTauY = 1.78`:

| scan half, σ=5 prior | within5 | far | core σ | σθ X/Y | vsp X/Y | sNV Y |
|---|---|---|---|---|---|---|
| kTauY patched to 1.0 (shipped behaviour) | 92.58 | 2.68 | 0.437 | 1.14/1.14 | 1.0/1.5 | 1.23 |
| kTauY = 1.78 (naive restoration) | 92.58 | 2.68 | 0.461 | 1.14/**1.57** | 1.0/**5.0** | **1.51** |

**Y regresses badly on every Y metric.** Reading: lp2's hyper set is
self-consistent *without* a per-plane τ under the shipped `share_lp` kernel
(zero-padded cascaded one-pole) — the archived RC-ladder kernel in which
kTauY = 1.78 was fitted is a different representation (6τ truncation,
`tau2_fac_y`, `sigma_sY`, model-error weights). This is the F19 lesson a
second time: **RC constants are representation-dependent; porting one between
kernel forms is the arm-B beam-transplant mistake.**

Consequences:
- `wft/model.py` reads the per-plane factor under a **new key `tau_y_fac`**
  that no existing bundle carries — nothing changes silently; the inert
  `kTauY` in old bundles stays inert.
- T2.1 is not a merge, it is a **recalibration**: fit `tau_y_fac` (with `kY`,
  which co-moves) under the shipped kernel — `11_tauy_refit.py`.

## 2 · The refits (ref-pinned, 180 train events, rest pinned at lp2)

**`11_tauy_refit.py`** — free (tau_y_fac, kY):

    tau_y_fac = 1.129   kY = 1.957 (lp2 carried 2.875)
    chi2 1.4522e8 (lp2 exact) -> 1.1154e8   (−23 %)

The shipped representation wants only a *slightly* slower Y copy (×1.13, not
the RC-ladder's ×1.78) — but a much smaller kY. **Since lp2's kY = 2.875 was
optimal only under the RC-ladder kernel, the post-merge shipped chain has been
running with a Y sharing amplitude ~47 % too strong.** Every post-merge
shipped-code result inherits that (the reference `events.parquet` itself was
produced pre-merge and is fine).

**`10_cx0_refit.py`** — cX = 0, free (sigma_p0, Dp):

    sigma_p0 0.4087 -> 0.4413   Dp 0.01342 -> 0.01337 (unchanged)
    chi2 with cX=0 at lp2 values: 1.4482e8 vs 1.4522e8 with cX=1  (−0.3 %)
    after refit: 1.1260e8

Killing X's discrete sharing costs ~nothing at fixed hypers (c1 = 0.051 was
already the C1_MIN floor — "the data want very little discrete X sharing"),
and σ_p0 absorbs it with a +8 % inflation. Note both arms land at ~1.12e8:
the bulk of both gains is the same underlying Y misfit being absorbed through
different knobs (kY vs the shared σ_p0). chi2 cannot separate them — the
bench geometry metrics (implied-v flatness above all) are the judge.

## 3 · Bench arms (scan half, σ=5 prior)

| arm | within5 | core σ | σθ X/Y | vsp X/Y | cmp14 X/Y | s14 X/Y | sNV X/Y | s/fit |
|---|---|---|---|---|---|---|---|---|
| reference (`t0scan_s5`) | 92.58 | 0.437 | 1.14/1.14 | 1.0/1.5 | +0.03/−0.27 | 1.09/1.09 | 1.15/1.23 | 1.06 |
| `tauy` (fac 1.129, kY 1.957) | 92.58 | 0.447 | 1.14/1.15 | 1.0/1.7 | +0.03/**−0.11** | 1.09/**1.05** | 1.15/1.32 | **0.70** |
| `cx0` (σ_p0 0.441) | 92.61 | 0.452 | 1.13/1.14 | **1.5**/1.5 | −0.09/−0.34 | 1.06/1.08 | 1.21/1.21 | 0.58 |

**Verdicts:**

- **cX = 0 is DISFAVOURED.** X's implied-v spread degrades 1.0 → 1.5 and an
  X bias appears at |tan| > 0.14. The discrete X copy carries real geometric
  information — consistent with `SHARING_DEPTH_2026-08-11.md` (X's copy is
  delayed ~+70 ns, which prompt diffusion cannot produce). F6's strong form
  fails the model test as it failed the waveform measurement: X's ±1 sharing
  is NOT just diffusion booked twice. T1.2 answered.
- **tauy is a genuine trade, not a win**: it halves the long-documented Y
  angle-compression bias at |tan| > 0.14 (cmp14 −0.27 → −0.11, s14 1.09 →
  1.05 — the §14.3 pathology responding to exactly the predicted knob) and is
  34 % faster, but near-vertical Y worsens (1.23 → 1.32) and core σ ticks up.
  Not adoptable alone; it localises where the per-plane kernel matters and is
  the natural starting point for a *full* recalibration (all hypers free with
  tau_y_fac) rather than this 2-parameter refit.
- **Both arms' ~23 % ref-pinned chi2 gains did NOT translate into geometry.**
  §35's warning verbatim: chi2 is degenerate along the v↔kernel valley — a
  better optimiser finds the wrong bottom faster. Implied-v flatness remains
  the only trustworthy judge.

## 4 · T2.4 shear arm — also rejected as implemented

`p0shear` (stage-2 scan evaluated at p0 − w·u_mid, u_mid = half the 30 mm
column): within5 92.32 (−0.26), core 0.449 (+0.012), vsp X 1.4 (vs 1.0),
sNV 1.18/1.24. Slightly worse everywhere.

**The lever measurement (8-12, `12_shear_lever.py`) falsifies the first
explanation** ("the amp-weighted centroid already sits near the mesh, so
u_mid overshoots"). Measured on the bench cache, u_eff = (p_c − p0_ref)/w_ref
on inclined reference tracks:

    X: +391 ns (robust σ 131, n=4231)    Y: +377 ns (robust σ 139, n=3660)
    angle dependence ≤ 15 ns across |tan| 0.08–0.45

i.e. **the lever v1 assumed (u_mid = 410 ns) was essentially correct** and is
a genuine, angle-independent geometric constant. So the rejection is of the
*approach*, not the number: shifting the scan by w·u_eff centres the grid on
exactly the level-set the chi2 constrains best (the charge centroid), so
every w along a sheared p0 row sits in the p0–w valley and the scan's w
discrimination degrades to noise. Without the shear, a steep track's true
(p0, w) is up to ~8 mm outside the ±2.5 mm scan — yet the NM refinement
recovers, and the crossing grid retains w contrast.

The half-lever test confirms it: `p0shear200` (200 ns, scan half, lp2_t0p
bundle) lands *between* no-shear and full-shear and still loses to no-shear on
every metric that moved — within5 92.55 / core 0.447 / vsp 1.3/1.6 /
sNV 1.18/1.26, against 92.58 / 0.437 / 1.0/1.5 / 1.15/1.23 unsheared. The
degradation is monotonic in the shear strength. **T2.4 is closed**: the lever
is real and measured, and *any* rotation of the scan grid toward the valley
direction costs w discrimination faster than the basin coverage pays. A §21.1
fix, if one is ever needed, must score w along the sheared line with something
other than the raw chi2 minimum (the profile-end/t0 features, or a p0
mini-profile per w). `P0_SHEAR` stays default-off (now accepts a numeric lever
in ns for such arms).

## 4.5 · T2.1 full recalibration (8-12, `13_full_recal.py`)

All seven kernel/geometry hypers + `tau_y_fac` free, v pinned at 36.6,
ref-pinned on the 180-event calibration cache, warm-started at the 2-param
optimum. Two lessons before the result:

- A first run **without the c1 floor** slid straight to c1 = 0.028 / kY = 3.9
  — the det7 collapse valley (WFT §17.2) is reachable even with v pinned; the
  floor (`calibrate.C1_MIN = 0.05`) is needed, and the converged c1 sits
  exactly on it (as lp2's did).
- The optimiser **abandoned the 2-param warm start**: it walked kY back up to
  2.90 (lp2's value) and bought its chi2 through `tau_y_fac = 1.39` with a
  faster `tau_s` (145.5 → 116.3) and a halved `c2` (0.058 → 0.031):

      c1 0.0505  c2 0.0313  kY 2.899  tau_s 116.3  sigma_s 11.57
      sigma_p0 0.419  Dp 0.0139  tau_y_fac 1.394
      chi2 1.4387e8 -> 1.1009e8 (−23.5 %, 719 evals)

So the (kY, tau_y_fac) plane has at least two basins of equal chi2 (~1.10 vs
~1.12 e8): "weaker but same-speed copies" (2-param: kY 1.96, fac 1.13) and
"same-strength but slower Y copies with a faster shared tau_s" (full: kY 2.90,
fac 1.39). chi2 CANNOT separate them — §35 again — the bench arm
(`kascan_full`) is the judge.

**Bench verdict (scan half, σ=5 prior): a trade again, not a win.**

| arm | within5 | core σ | σθ X/Y | vsp X/Y | cmp14 X/Y | sNV X/Y | s/fit |
|---|---|---|---|---|---|---|---|
| reference (`t0scan_s5`) | 92.58 | **0.437** | 1.14/1.14 | 1.0/**1.5** | +0.03/−0.27 | 1.15/**1.23** | 1.06 |
| `kascan_full` | **92.64** | 0.448 | 1.13/1.13 | **0.9**/1.6 | −0.09/−0.18 | 1.16/1.35 | **0.39** |

Better: within5 (+0.06), far (2.63), best-yet X implied-v flatness (0.9), Y
deep-angle compression halved-ish (−0.18), and **2.7× faster**. Worse: core σ
(+0.011), near-vertical Y (1.23 → 1.35, worst of all arms), a small X
deep-angle bias appears (−0.09), Y implied-v spread 1.5 → 1.6. The same
signature as the 2-param arm: the per-plane τ genuinely fixes the §14.3
deep-angle Y pathology and genuinely breaks near-vertical Y in exchange.

**Reading: the Y deficiency is not an amplitude/speed calibration problem.**
Both basins of a full recalibration land on the same trade, so what Y is
missing is kernel *shape* (F6/F12 — a proper transmission-line response
instead of a scaled one-pole cascade), which no recalibration of the current
form can supply. That is the real post-MPGD26 project. If raw throughput ever
matters, `kascan_full`'s config is a defensible speed variant (same within5,
better X flatness, 2.7× faster) — but lp2 + σ=5 prior remains the
recommended det3 production configuration.

## 5 · Where this leaves the queue

- T1.2: **answered** (X's discrete kernel stays; its sharing is not diffusion
  booked twice).
- T2.1: **closed as far as recalibration can take it** (8-12). Naive port
  rejected; 2-param refit a trade; full recalibration (§4.5) a trade in a
  different basin. The per-plane τ knob is real but cannot win under the
  current kernel form — the remaining work is a Y kernel *shape* (F6/F12),
  post-MPGD26.
- T2.4: **closed** (§4). Lever measured (≈ u_mid, angle-independent); the
  shear approach itself is what fails, monotonically in strength. §21.1 open.
- T1.3: **done** (bundle `dead` mask + censoring + `14_dead_mask.py`; det3
  clean, matters for det2/det4). T1.4: **done** (`EXPLAIN_2026-08-12.md` +
  `wft/explain/` figures; doc prose in §9/§12/§19.1/§23).
- The σ=5 trigger prior remains the one validated, adoptable improvement
  (see `T0_PRIOR_2026-08-11.md` §8; production bundle `calib_bundle_lp2_t0p`
  written and smoke-verified 8-12).
