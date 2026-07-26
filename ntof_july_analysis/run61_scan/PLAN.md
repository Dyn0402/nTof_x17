# run_61 singles+PS 2-D drift × resist scan — tracking / drift / optimization

**Run:** `run_61`, 2026-07-20 17:56 → 2026-07-21 04:32, **finished normally,
60/60 sub-runs complete**. From `run_config.json`: RAW **singles + PS** trigger
(`trigger_mode.py scint --singles --ps-pickup`; M4.C = or_veto(Singles, lemo0)
gated by the 30 ms N93B window, M4.D = OR(lemo0 = PS/γ-flash, lemo1 = C-out)).
**Ar/Iso 90/10, ³He target, neutrons, no Pb filter.** Full readout
(zero-suppress OFF), **64 smp × 60 ns = 3.84 µs**, IPD 90, latency 33.

Same trigger recipe and same DAQ config as `run_58` — this is the direct,
higher-statistics repeat of that scan. Kept as a separate package so run_58's
published numbers stay frozen.

## Scan grid (subruns `sngPS_dr{drift}_r{resist}_{seq}`)
- drift OUTER (all 4 dets): **700, 600, 500, 400, 300, 200 V** (6 pts)
- resist INNER A/B/C: **560 → 515 V** (−5 V effective, 10 pts, taken as two
  interleaved −10 V passes: 560→520, then 555→515)
- det D resist held **10 V below** the A/B/C setpoint.
- 6 × 10 = **60 sub-runs × 10 min**, all complete.

vs run_58 (drift 700→150, resist 580→540): the drift range is comparable, but
the **resist window sits 20 V lower** and overlaps run_58's only on 560…540 V.
So run_58 is the authority above 560 V, run_61 below 540 V, and 560–540 is the
cross-check region.

## What had to change vs run_58 — the comb timing
run_58's probe classes were hardcoded ms windows `(0,1) / (8,18) / (20,33)`.
run_61's comb is measured (identical at `sngPS_dr700_r560_000` and
`sngPS_dr200_r515_059`):

| slot | dt | events/spill | raw hits | what it is |
|---|---|---|---|---|
| 0 | 0 | 1 | — | the **γ-flash trigger** (`n_big` ~85 k, 100 % of spills) |
| 1–4 | **~4.1 ms** | 4 | ~6 | front-end **BLIND**, ~14 % still saturated |
| 5–6 | ~13.5 ms | 2 | ~96 | partially recovered |
| 7+ | 27.2 / 41.0 / 55.3 / 69.1 ms | 2 each | ~500 | **recovered** |

⇒ ~14.9 events/spill, dead cycle ~13.6 ms.

Applying run_58's windows unchanged would have (a) left `early` **empty** — its
0–1 ms slot holds only the flash trigger, which is excluded from reco — (b)
silently dropped the 4.1 ms blind batch, and (c) kept only the 27 ms pair out of
four recovered teeth. The edges are therefore re-derived
(`early` = 1–8, `mid` = 8–20, `late` = 20–95 ms) while the *semantics* are
unchanged, so the late-probe efficiency stays comparable with run_58 cell for
cell. Verified against `idx_in_burst`: the two agree to <1 %.

**Net gain: ~1300 late-probe events per HV cell vs run_58's ~390.**

One other deliberate difference: run_58 dropped every `n_big > 150` event from
reco. In run_61 that would remove ~14 % of the 4.1 ms `early` batch — exactly
the flash-saturated events whose loss *is* the post-flash inefficiency being
measured — biasing the early yield upward. Here they are reconstructed and kept
in the denominator; only pathological pile-up (> `RECO_MAX_HITS`) is zero-filled
and flagged `reco_skipped`. This makes `early` not directly comparable with
run_58's; `mid`/`late` (~0 % saturated) are unaffected.

## ⚠ run_61 has severe FEU dropouts — read this before using any number

Found while sanity-checking the first pass, whose optimum landed on the grid
corner (drift 700 / resist 560) for every detector with single-cell plateaus.
The drift marginal alternated (500 and 700 high; 600, 400, 200 low) — and those
groups are exactly run_61's **coarse-first interleave order**, i.e. a time
trend masquerading as a drift dependence.

Cause: **in most sub-runs only a subset of the 8 FEUs participates**, typically
for the first ~90 % of the sub-run, with the rest joining in the last decile.
`sngPS_dr700_r525_018` ran with **only FEU 6** until its last tenth. The DAQ log
records nothing — 122 lines, no error. **run_58 is clean in all 63 sub-runs**
(max zero-hit fraction 0.00 per detector), so this is a run_61 regression, not a
standing feature.

Scale (`feu_presence.py`, per-event census of all 60 sub-runs):
- only **23.4 %** of events have all 8 FEUs;
- only **14 of 60** sub-runs exceed 50 % all-8, and they are all drift 300/500/700
  (the first interleave group);
- drift **600 / 400 / 200** — the entire second and third groups, sub-runs 30-59 —
  sit at **5-10 %**;
- per-detector live fractions overall: A 0.44, B 0.41, C 0.42, D 0.34
  (a detector only needs its own 2 FEUs, so this is better than the all-8 rate).

FEU → detector map (from `io.build_channel_lut`): **A = (3,4), B = (5,6),
C = (7,8), D = (1,2)** — x-plane first. A detector needs BOTH to form a 3-D pair,
which is why sub-run 018 showed Det B "hits" (FEU 6 = B_y) but no B pairs.

**Correction applied.** `feu_presence.py` builds a per-event FEU-presence table
from the `(eventId, feu)` branches only — cheap, no reco — and joins onto the
existing cache by `(subrun, eventId)`. Every per-detector rate is then evaluated
on that detector's live events only, so a dropout leaves the DENOMINATOR instead
of being scored as tracking inefficiency. In RAW full-readout mode a
participating FEU always yields ~550 hits/event, so presence is unambiguous.

Effect: the drift dependence becomes monotonic and physical (Det A 0.010 →
0.043 from 200 → 700 V, saturating ~500-700 V) instead of alternating, and
Det A's efficiency roughly doubles (0.025 → 0.067 at resist 560). Residual
damage: cell statistics are very uneven — ~1500 late events in surviving cells
vs ~150 in dropout-hit ones (±25 %), and a few cells are empty outright.

## Deliverables

### A1 `analyze_tracks.py` — yield vs time-since-flash and HV, per detector
**`yield_vs_hv_early.png` / `_mid.png` / `_late.png`** — the three time groups
plotted SEPARATELY (4 ms / 13 ms / 27+ ms integrated), each showing yield vs
resist and vs drift for all four detectors. These are the primary figures; do
not pool the groups, their HV dependence is opposite.
Also `time_recovery.png` (3-class ladder), `slot_recovery.png` (recovery at full
comb resolution — 6 sampled times), `gain_vs_hv.png`, `liveness.png`,
`vs_run58.png` (late-probe resist curve overlaid on run_58's), and
`per_cell_stats.csv` (all three classes × every HV cell).

### A2 `analyze_drift.py` — v_drift, effective gap, efficiency vs drift & resist
Same estimators (t0_daq = P5 of segment `t0_ns`; T_max = P95 of micro-TPC track
`tspan_ns`; v = 30 mm / T_max; D_eff against the dry Garfield 90/10 curve). Full
6-point drift axis → a clean repeat of run_58's velocity curve.

### A3 `optimize.py` — best drift/resist operating point per detector
Kernel-smoothed efficiency surface (85 V × 7 V Gaussian) + 1σ plateau, run
**once per time group**: `eff_smooth_heatmap_{early,mid,late}.png`,
`profiles_{…}.png`, `recommendation_{…}.md`, `best_points_{…}.csv`. Read the one
matching your window. Caveat: `suggest_setpoint` inherits run_58's rule of
pulling drift up to the v-saturation knee — defensible for the late window,
questionable for the 4 ms one where efficiency falls with drift, so there the
"efficiency optimum" line is the honest number, not the "suggested setpoint".

## Headline result — the optimum MOVES with time since the flash

**There is no single HV optimum.** The resist dependence *reverses* across the
comb, which is exactly the "too much HV → longer post-flash saturation, too
little → inefficient" trade-off, resolved in time. Det A, P(3D pair) ×1000,
drift-pooled:

| resist V | 515 | 520 | 525 | 530 | 535 | 540 | 545 | 550 | 555 | 560 |
|---|---|---|---|---|---|---|---|---|---|---|
| **4 ms**  | 38 | 36 | **49** | 33 | 33 | 21 | 10 | 13 | 7 | 4 |
| **13 ms** | 15 | 18 | 21 | 13 | 23 | **39** | 35 | 29 | 17 | 10 |
| **27+ ms**| 14 | 10 | 15 | 24 | 22 | 35 | 44 | 39 | 45 | **67** |

The peak walks upward with recovery time: **≤525 V at 4 ms → ~540-545 V at
13 ms → ≥560 V at 27+ ms.** Higher gain buys efficiency once recovered but costs
you the early window, because the front end is still saturated there.

Robustness (the FEU dropouts correlate with resist within a pass, so this was
checked three ways) — Det A 4 ms falls 49→4 on all data, 30→0 on drift-700 only
(n≈650/cell), 31→2 on the 14 clean sub-runs. Not an artifact.

Which end you want depends on the window you care about; the 4 ms group is the
operationally important one (earliest the front end can be read at all). Note
the 4 ms curve is **flat-to-rising at the bottom of the scanned range**, so that
optimum is at or below 515 V and is *not bracketed* — as with run_58 above
560 V, the scan stops before the turnover.

For the late window only, run_61 and run_58 agree where they overlap (540 V:
0.0340 vs 0.0345; 550 V: 0.0383 vs 0.0387) and the stitched Det A curve runs
0.014 @515 V → 0.067 @560 V → 0.059 @580 V. Drift saturates ~500-600 V for the
late window; for the 4 ms window efficiency falls with drift too (Det A ~41 @
200 V → 20 @ 700 V), so the v-saturation argument and the early-window optimum
pull in opposite directions.

### Det A alone, un-smoothed — `detA_2d.py`
`detA/detA_2d_raw.png` (raw 2-D grid per time group, every cell annotated with
p ± err **and n**) and `detA/detA_profiles_raw.png` (raw 1-D slices with error
bars, both directions). No kernel smoothing anywhere.

⚠ **Read the raw grid with the n values.** The dropouts leave cell counts
bimodal — ~650-770 events when live, ~30-80 when hit — so on a raw grid the
dropout-hit cells give the largest *apparent* efficiencies from small
denominators. For Det A / 4 ms the top raw cell is drift 700 / resist 525 at
96 ± 41 per mille, which is **k = 5 pairs out of n = 52**. Cells with n < 200 are
hatched and excluded from the argmax. Best *reliable* cells:

| group | best reliable cell | value ×1000 |
|---|---|---|
| 4 ms | drift **200** V, resist **525** V | 57 ± 8 (n=768) |
| 13 ms | drift 700 V, resist 545 V | 52 ± 12 (n=346) |
| 27+ ms | drift 700 V, resist 560 V | 79 ± 7 (n=1295) |

Only 27 of 58 Det A cells survive the n ≥ 200 cut in the 4 ms group.

**Time-decorrelation checks** (the interleave is only partial — drift 200 is the
last block acquired, so a drift trend could still be a time trend):
- *Resist trend is solid.* At drift 700 the two interleaved passes each fall
  independently: pass 1 (seq 0-4) 560:0 → 520:30, pass 2 (seq 15-19) 555:8 →
  535:23. Pass 2 restarts low instead of continuing pass 1's endpoint, so this
  tracks resist, not time.
- *Drift trend is real but mild.* The resist-535 row is complete across all six
  drifts with mixed acquisition order (seq 57/27/47/22/42/17): 37.8, 39.1, 37.7,
  31.2, 29.7, 23.1 — a ~1.6× fall over 200→700 V, versus the ~12× resist effect.

## Layout
- `scan_lib.py` — subrun parse, burst/flash/dt model, reco caching, drift spectra
- `feu_presence.py` — per-event FEU-presence table + `attach()` (the dropout fix)
- `detA_2d.py` — Det A raw (un-smoothed) 2-D grid + profiles
- `process.py`  — parallel cache builder (`--jobs N`; the machine has 6 cores and
  is usually also running DAQ + decoding, so 3 is the safe default; ~100 s per
  sub-run ⇒ ~35–40 min for all 60)
- `analyze_tracks.py` / `analyze_drift.py` / `optimize.py`
Output → `<ANALYSIS_DIR>/July_HV_Scan/run61_scan/`.

## Sibling packages
- `run58_scan/` — the original (singles+PS, resist 580→540).
- `run62_scan/` — the singles-ONLY (no PS leg) counterpart taken right after
  run_61, same DAQ config, truncated at 3 h. Useful as the no-flash-trigger
  control: dropping the PS leg hands the flash's accept slot to physics.
