# October 2026 — findings, and the work that has to happen before the re-pass

**Written 2026-09-14.** The decision of this session: **the campaign re-pass is
pushed to October.** Nothing is re-reconstructed in September.

This file is the standing October list. It carries what was established about
two-track reconstruction, what is still unknown, and the ordered work. `PLAN.md`
§9 holds the D1–D15 register this maps onto; `STATUS.md` stays the live log.

> **The headline for October: two tracks in one chamber are not reconstructed as
> two, and the fixes for that are written but unvalidated and unshipped.** The
> re-pass is expensive (~8–12 h of condor plus a day of chain), so it should
> happen **once**, after the two-track work is tested — not before it.

---

## 1 · What September established

### 1.1 The production reconstruction loses close pairs, and now we know why

A waveform-overlay truth bench (`intra_bench.py`, run_145 stat090_0000, A and C)
sums two clean single-track triggers of one chamber and re-runs the production
seeder and fit. Truth is each donor's frozen single-track fit.

| separation | both tracks found and correctly paired (production) |
|---|---|
| ≥ 24 mm | **47 % (A) / 39 % (C)** |
| 12–24 mm | 18 % / 17 % |
| < 12 mm | ~0 |

Three causes, measured separately:

| cause | where | size |
|---|---|---|
| **x/y swaps** of time-degenerate tracks | `select_tracks` paired by summed dchi2, so the strongest x went with the strongest y | 15–26 % of tracks ≥ 24 mm; ~75 % when the planes rank the tracks differently |
| **plane-wide significance floor** | a strip is kept at ≥ 10 % of the brightest strip *of the whole plane*, so a brighter partner anywhere erases a fainter track | ~16 % of tracks ≥ 24 mm, at any separation |
| **merging at the 12 mm seed gap** | `seed_candidates` | 98–100 % below 12 mm, 62–65 % at 12–24 mm |

**Once a separated track is found, it fits like a single track** (robust σ of p0
against its donor ≲ 0.1 mm, same strip count). So H2b — a window or fit spanning
both tracks — is *not* what widens two-track fits on data. That is H3: two-track
events are genuinely busier. The reconstruction is the lever for *finding* the
tracks, not for the resolution of busy events.

### 1.2 Two fixes exist, are validated, and are not switched on

Both opt-in; with them off the output is production's, candidate for candidate.

| fix | bench, ≥ 24 mm | single-track check |
|---|---|---|
| x/y pairing by charge (`select_tracks(pairing=)`, bundle `xy_pairing`) | swaps 24 → 14 % (A), 23 → 7 % (C) | cannot touch a single-track event by construction |
| rescue-mode local significance floor (`WFT_SIG_FLOOR_LOCAL_MM=16`) | — | **0 production tracks lost**, 99.6 % bit-identical, +279 / +406 triggers gain a track |
| **both together** | **71 % (A) / 66 % (C)** | as above |

**Rejected by that same check**, despite helping on the bench: a local floor
*replacing* the plane-wide one (loses 1.4–7.3 % of production tracks) and
splitting seed clusters at 6/8 mm (11–27 %). Real clusters have holes; the clean
bench donors did not show it.

Full record: `../wft/MULTITRACK_2026-09-14.md`. Shipping steps: its §5.

### 1.3 What is still lost, and what it costs the physics

Below 12 mm nothing recovers: the two tracks share one seed cluster and need a
joint two-track fit, which does not exist. This is **D1**, and it is the one
deferred item with a direct effect on the signal region:

- **Small-opening-angle pairs land close together in the chamber.** Conversions
  and low-angle IPC are exactly the population that merges.
- On data (`det_a_intra`, 296 218 intra-A pairs): **0 real pairs below 20 mm of
  radial separation, 56 below 40 mm where ~12 400 are expected.** The per-view
  efficiency is 0.005 at 0–20 mm, and the loss is the **product** over the two
  views — a pair 12 mm apart in u and 300 mm apart in v is lost as completely as
  one close in both.
- **`acceptance.py` does not model this at all.** It treats the two legs as
  independently reconstructed and accepts a pair 10 mm apart at full efficiency.
  The measured loss has been folded in by hand once (capsule folded median 19.0 →
  24.7°), but it is not in the code.
- Same-chamber pairs are ~a third of the two-track sample (19 707 of ~62 700 at
  the published 30 mm cut) and are the natural place to ask whether two tracks
  share a vertex — `PLAN.md` §S4 also uses intra as the background normalisation.

### 1.4 Same-chamber vertex tests are blind today

All three event-mixed-controlled tests come out consistent with mixing
(`pair_vertex_imaging/intra_vertex.py`): Δy robust σ 152 mm real against 168 mm
mixed in A–A, 230 vs 231 in C–C; depth agreement 9.5 % vs 7.4 %; Spearman ρ
between the two views' depths ~0. A second track in the chamber makes **both**
tracks 3–4× worse in y (A 41 → 139 mm, C 48 → 170 mm). One ~2σ hint survives.
**These tests are not re-runnable as evidence until the reconstruction improves**,
because their sensitivity is set by the leg resolution, not by the statistics.

---

## 2 · What is NOT established

- **The joint two-track fit is written and measured (2026-09-16) but not shipped,
  and not yet run on data.** Design and acceptance criteria:
  [`HANDOFF_JOINT_TWO_TRACK_FIT.md`](HANDOFF_JOINT_TWO_TRACK_FIT.md); what it does
  and what it cannot do: [`TWO_TRACK_FIT_LOG.md`](TWO_TRACK_FIT_LOG.md). Two
  parallel tracks on the same strips are a model degeneracy and stay lost at any
  time offset — that has to be carried as inefficiency, not fixed.
- **Everything above is one sub-run and two chambers** (run_145 stat090_0000, A
  and C). B and D are untested, and D's hot-channel seeding has never been
  combined with the rescue floor.
- **Bench donors are clean.** Real two-track events are busier, so the gain on
  data will be smaller than 71 % / 66 %.
- **The A/B check compares to the frozen pass, not to truth.** It shows nothing
  moved; it does not show the added tracks are real.
- **Calibration is per detector and per run condition.** `xy_pairing` comes from
  run_145 — post-23-July noise, post-27-July access. The other side of each
  boundary is unmeasured.
- **The angle scale is still the standing blocker for the opening angle** (`k`
  varies 24–29 % run to run on C and D; the pointing estimators and the geometric
  bound disagree in a fixed direction). Nothing in the two-track work touches it.

---

## 3 · The October list, in order

### O1 — Implement and test the joint two-track fit ⭐ **the gating item**

**Written and measured 2026-09-16** — off by default, not in the production
chain. Record: `../wft/TWO_TRACK_FIT_2026-09-16.md`; working log with every
measurement that shaped it: `TWO_TRACK_FIT_LOG.md`; report:
`<out>/two_track/report.html`. What remains of O1 is the campaign-side work
below (steps 3–5) and the decisions in the handoff's §0.

On the overlay bench (coincident, on top of pairing + rescue) it recovers pairs
below 12 mm for the first time — **18 % (A) / 37 % (C), from 0** — and lifts
12–24 mm to 43 / 45 % without moving ≥ 24 mm; clean single muons split at
≤ 0.66 %. **Compute is not a constraint (Dylan, 2026-09-16):** a re-pass may take
a month, and the remaining O1 work is about effectiveness — first, dropping the
per-plane trigger, which on synthetics discards two thirds of what the fit
recovers at 0–6 mm. Ordered list: `TWO_TRACK_FIT_LOG.md` → *To do*.

`HANDOFF_JOINT_TWO_TRACK_FIT.md` is the design: a 2K-column NNLS with 6 outer
parameters (5 with tied t0), a trigger set so it runs only on candidates that
look merged, and a calibrated 1-vs-2 model-selection threshold. **This is D1.**

Build in this order — each step is a gate, not a formality:

1. **Synthetic** two-track planes from `wft/model.py` under the production
   bundle. Tests recovery, label ordering, crossings, and that one-track
   synthetics do not split.
2. **Bench variant** (`intra_bench build --two-track`) on top of
   `pairing_rescue16_ranked`, with separation bins refined below 12 mm.
3. **Single-track A/B** on every production trigger where a split is attempted,
   matched against the frozen pass. *The rescue floor passed this; replace-mode
   and split seeding died on it. Assume any new change dies here until it does
   not.*
4. **Cost benchmark** — trigger rate × per-attempt cost. *Done as a record
   (+115 % A / +223 % C); no longer a gate — compute is not a constraint.*
5. Only then, data.

**Two decisions to settle first** (handoff §0): the single-track contract has to
be restated, because a split necessarily replaces a track where it acts; and
whether this ships in the same re-pass as the pairing and rescue floor
(recommended: yes, one pass).

### O2 — Split seeding as *additive* candidates

The rejected split-at-6-mm variant separated 12–24 mm pairs well (12–18 mm 8 →
48 % A, 5 → 44 % C) and failed only because it *replaced* production clusters.
Offering the parts as extra candidates ranked below production pairs is the same
move that made the rescue floor safe. Cheaper than O1 and may cover 12–24 mm on
its own; O1 is still needed below 12 mm.

### O3 — Make the fixes shippable

- Per-run `xy_pairing` calibrations (36 runs, from each run's own stage-3 tracks;
  `intra_bench calib-pairing`), and a **merge step in `wft_beam.make_bundle`** —
  it has none, and this is the same shape as `run_beam_job.py --hot`.
- One run on each side of the 23 July noise boundary and the 27 July access, so
  the calibration is not silently run_145-only.
- Env vars into `condor/stage2_fullpass.sub`'s `environment` line.
- **A new smoke gate.** The existing one requires event counts identical to the
  August pass; this change adds tracks by design. Gate on "production candidates
  reproduced exactly, extra tracks counted".

### O4 — The single campaign re-pass

Only after O1–O3 pass. Sizing and the step-by-step timing are in
`HANDOFF_JOINT_TWO_TRACK_FIT.md` §10. Summary:

| | condor | end to end |
|---|---|---|
| measured, 2026-09-09 full pass | 8 h | ~13 h |
| pairing + rescue only | ~8–9 h | ~14 h |
| **+ joint fit (estimate, unmeasured)** | **~10–12 h** | ~15–17 h |
| A and C only | ~4–5 h | mixes reco versions per arm; stamp it |

Archive `<out>/reco_fullpass` (28 GB) and `stage3_fullpass` (11.4 GB) or write to
new trees. Keep the current pass reproducible, the way the full pass kept the
allowlist pass.

### O5 — Re-run the downstream analyses that the re-pass changes

Every one of these rests on a reconstruction that loses close pairs:

- `intra_vertex --multiplicity` (the two-track rows should approach the
  one-track rows, and two-track chambers should appear below 12 mm) and the three
  vertex tests, which only then become sensitive;
- `det_a_intra` — the real/mixed two-track resolution map, and whether the
  4.4 σ beam-axis convergence survives;
- **`acceptance.py`: fold the measured two-view efficiency product in properly**
  (§1.3). It is currently absent from the code, and it moves folded medians;
- the campaign opening-angle spectra and the published notes built on them.

### O6 — The angle scale (the standing blocker, independent of all the above)

Per-detector recalibration, deferred from September: `k` differs 24–29 % between
run_86 and run_145 on C and D; the run-to-run spread is one contiguous 48-hour
block (runs 128–147) where all three arms rise together, and run_145 — whose `k`
the current table borrows — sits at its peak; and the pointing estimators want a
*smaller* `k` on A and C while the geometric bound wants a *larger* one. **No
opening-angle spectrum on a borrowed `k` for C or D.** This is the item that
decides whether an angle can be quoted at all, so it should run in parallel with
O1 rather than after it.

### O7 — Simulate the double-track finding efficiency (D2)

Geant4 pairs through the response sim and the full chain, efficiency by opening
angle and by track separation. `PLAN.md` calls it *"the single most valuable
October item"*: it turns every "we probably lose some" into a number. **It is
also how O1 gets an efficiency curve that is not conditioned on our own bench's
clean donors** — so run it against the post-O1 reconstruction, not the current one.

### O8 — The rest of the register

Unchanged from `PLAN.md` §9 and not re-argued here: **D3** (x↔y pairing via
`dca_target` and charge profile — partly addressed by `xy_pairing`, the
sample-pointing half is untouched), **D4** (chambers B and D — the biggest single
limit on the signal region), **D9** (in-situ v_drift and diffusion per run
condition), **D10** (per-channel gain and dead-strip maps), **D13** (stage-1
filter efficiency), **D16** (track t0 against scintillator timing), plus the
efficiency-denominator question in `efficiency.py` (PCB conversions inflate the
denominator, so the current numbers are lower bounds).

---

## 4 · Sequencing

```
O6 angle scale  ─────────────────────────────┐ (parallel, independent)
O1 joint fit → O2 additive split → O3 ship ──┴→ O4 re-pass → O5 downstream → O7 sim
```

The re-pass is the join point. Anything that wants to ride it (a seeding change,
a hot-channel wildcard, a bundle change) must be validated **before** O4 or wait
for the next one.

---

## 5 · Traps that have already cost time

- **Geometry from waveforms, never hit times** (`CLAUDE.md`). Hits seed; they
  never set a position, angle or depth.
- **A seeding change can quietly lose single tracks.** The D hot-channel wildcard
  lost 16 152 of 33 393 good fits on its first tuning. A/B on single-track gated
  counts first, always.
- **χ²/dof is not 1 here** (clean single tracks: A 1.4–2.2, C 4.4–6.8; full-pass
  medians 5–12). Any Δχ² threshold has to be calibrated empirically.
- **Bundles must have c2 < c1**; read c2 with `wft.calib.effective_c2`, never
  `hyper['c2']`.
- **`k_arm` has no `--out`** and overwrites `kcal/k_arm_<run>.json`. Copy aside first.
- **`<out>/fullpass` is the allowlist pass**; the full pass is `<out>/reco_fullpass`.
- **`pandas.Series.to_numpy()` can be a view** — use `copy=True` before in-place
  boolean ops.
- **Laptop and condor differ at the 4 µm level** on unchanged code; bit-identity
  checks are laptop against laptop.

---

## 6 · Documents

| | |
|---|---|
| [`HANDOFF_JOINT_TWO_TRACK_FIT.md`](HANDOFF_JOINT_TWO_TRACK_FIT.md) | **O1.** The design, the validation plan, acceptance criteria, condor cost |
| [`HANDOFF_INTRA_TWO_TRACK_RECO.md`](HANDOFF_INTRA_TWO_TRACK_RECO.md) | the problem statement, the measured symptoms, and §10's progress record |
| [`../wft/MULTITRACK_2026-09-14.md`](../wft/MULTITRACK_2026-09-14.md) | the two shipped-but-off fixes, their validation, and how to switch them on |
| [`../wft/MULTITRACK_2026-08-12.md`](../wft/MULTITRACK_2026-08-12.md) | the three tiers of multi-track work; tier 3 is D1 |
| [`PLAN.md`](PLAN.md) §9 | the D1–D15 deferred register |
| [`STATUS.md`](STATUS.md) | the live log |
| `intra_bench.py`, `make_intra_bench_report.py` | the bench; reports at `<out>/intra_bench/<variant>/report.html` |
| board | <https://dylan-neff.web.cern.ch/x17/analysis.html> |
