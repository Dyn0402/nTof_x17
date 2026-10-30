# Bow transfer gate on the pass-1 tables: PASS (2026-10-30)

PASS_PLAN E2 rule: one bow block per chamber (from its golden row), every other row re-measures only w0/kw; no row's
angle or position s68 may worsen beyond its error (rows ≥ 2000 events). Skeptical rows (uncalibrated angle scale) are not in the gate.

**Result: every row improves, in both views, in angle and in position.** The production arm ("pass": the golden block plus the row's
own w0/kw):
- angle s68 changes by −0.06° to −0.53°, and every change is at least 2.3σ (the smallest, det7 short_run X, is −0.070 ± 0.031°);
- position s68 changes by −0.008 to −0.051 mm.

The table is `gate_table.md`, the numbers are in `bow_finalize.json`, and `manifest.json` lists the inputs.

| chamber | rows (n events) | angle Δ, pass arm, X / Y (°) |
|---|---|---|
| det2 | golden 6-22 longer (3.7k, in-sample), 6-22 long (19k), 6-22 short (2.0k) | −0.09…−0.11 / −0.30…−0.34 |
| det3 | golden 6-27 (7.1k, in-sample), 6-28 P2 night (22k), **6-25 bottom slot (8.9k)** | −0.10…−0.17 / −0.14…−0.23 |
| det4 | golden 6-24 (8.1k, in-sample), 6-23 night (6.6k) | −0.12…−0.13 / −0.43…−0.47 |
| det6 | golden 6-26 (8.9k, in-sample), longer (4.4k), short (1.9k, information only) | −0.34…−0.53 / −0.31…−0.50 |
| det7 | golden 6-26 (8.8k, in-sample), longer (4.5k), short (2.4k) | −0.06…−0.10 / −0.30…−0.31 |

- **The strict arm** (golden w0/kw too, nothing measured on the row) is worse on rows of another day or later in the night: det3 6-25 +2.2° X,
  det4 6-23 +0.51° X, det6 longer +0.44° X, det7 short +0.28° X. The rigid tilt and scale move between runs (gas, mount), so they must stay per row,
  as the pass does. The bow shape transfers; w0/kw do not.
- **det3 6-25 (bottom slot)** improves with the chamber block (−0.17 / −0.23°): no per-mount block is needed.
- **Inputs:**
  - The block re-derived from each golden row's final pass table matches the production block (measured on the CAL3 table) to
    ≤ 0.0005 in tan and ≤ 0.1 mm in h (`blocks_rowcheck/`).
  - The tables were built by `make_pass_dag.py bowgate` (bow_node.py --table-out, code a72b889) into
    `Analysis_pass1/<row>/wft/bow_gate_table.parquet`.
  - The no-bow arm uses the golden calib_bundle with `bow` removed.
- **Found while running it:** `wft.reco.bow_UW` took U/W from both views' own fits, so an absurd fit in one view set the other view's bow
  tangent and mesh height. This affects 0–0.2 % of events per row; the angle shift is mostly 0.01–0.06 in tan, and 0.5–0.85 on 9 events (det3 6-25, det4 6-23).
  It is now guarded (an off-plate coordinate counts as a missing view, `BOW_UW_MAX`), and so is the gate's scorer
  (`gate_bow.prep`), which had turned one such event into a 86 mm position s68 on det7 longer Y. The pass-1 tables keep the old behaviour
  for those ≤ 0.2 % of events (a stated effect, not re-reconstructed).
- **For the det4 Y ablation (next):** the bow alone is worth −0.43° on det4 Y (in-sample). The audit fe bundle had no tan-form
  bow block, so the bow is the first suspect for the −0.245° G1 gain.
