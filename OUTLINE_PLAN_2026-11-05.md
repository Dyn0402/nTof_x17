# Paper II outline and plan (2026-11-05)

Companion to `papers/paper1`, target NIM A (short, focused). Figure inventory: `collection/INDEX.md`
(`collect_figures.py` rebuilds it). Slide-note version: `make_collection_deck.py`.

## The claim the paper makes

> On resistive-strip micro-TPCs a per-strip hit time is an aggregate of the strip's own charge and delayed copies of its
> neighbours'. That compresses the drift-time ladder by 20–30 % whatever the estimator, so no threshold or timing choice
> fixes it. A forward model fitted to the waveforms removes the bias and gives ~1° angles; a hit-only readout cannot,
> unless it records what the model needs.

On pass 1, same interior_good events, five chambers (`rerun/paper1_floor/figures/fleet.pdf`): angle s68 **H0 2.0–2.6° vs W
0.73–1.56°**, position core σ 550–805 vs 430–620 µm, efficiency within 5 mm within 1–5 points. H0's implied v is 46 against
36 µm/ns on det3.

## What changed since the existing `.tex` skeleton

- **The physics floor claim is withdrawn** (model audit F3, 2026-10-27). The "0.02° electronics, ~1° centroid jitter"
  story needed 0.30 mm of jitter against a physical 0.185 mm. The floor is **0.30–0.35°, set by diffusion** (`ws_floor`);
  measured is 3–6× that; pass 1 leaves a 0.56–0.65° remainder on det2/3/7 X that neither the reference, the X copy nor
  position structure explains. §4 is re-framed from "the floor is physics" to "how far from the floor, and what the rest is".
- **The kernel numbers moved.** Head-on SPS: ±1/lead 0.377 X / 0.390 Y, ±2/±1 0.19 X / 0.26 Y, field-independent. The old
  Y c2/c1 0.42–0.47 is superseded. The `.tex` `\tbw` text still carries the old values.
- **Fleet numbers.** `\requote{1.15–2.51°}` is the r06 table. Pass-1 headline s68 X/Y (°): det2 1.045/1.125, det3 0.991/0.901,
  det4 2.412/2.076, det6 1.256/1.506, det7 1.232/1.461 (`rerun/paper1_res/PASS1_RESOLUTION_2026-10-31.md`).
- **No new data** (CLAUDE.md, 2026-10-16). Everything open below is computation on existing waveforms, simulation, or a stated limitation.

## Proposed structure (NIM A, ~10 pages, ~10 figures)

| § | section | main figure(s) | state | note |
|---|---|---|---|---|
| 1 | Introduction | the claim in one schematic | write | short detector summary, cite Paper I |
| 2 | Signal formation: the sharing kernel | `kernel_forms`, `angled_kernel`; schematic | ready | state τ, c2 as bounds (tail outlasts the 3.84 µs window); X vs Y asymmetry as a limitation |
| 3 | Why hit times fail | event display (strip × time) + **ladder compression, five chambers, four estimators** | **remake** | the central diagnostic; det3 exists only inside the July report PDF |
| 4 | The forward model and its calibration | **forward-model diagram**, `templates_perplane`, `share_decompose`/`event_fit` | partial | diagram on the production physical-RC model; calibration protocol and traps (R9) |
| 5 | How far from the floor | `floor_vs_angle_det3`, `floor_breakdown_det3`, measured overlay | partial | the 60 ns depth-bin bias on steep tracks is a finding in itself |
| 6 | Hits against waveforms, same events | **tier benchmark** H0, H1, H2a, H2b, H2c, W; `fleet` | core, **open** | only H0 vs W exists on pass 1 |
| 7 | What a hit readout must record | **threshold × neighbours** scan | open | software only: re-extract hits from existing waveforms. VMM emulation stays deferred |
| 8 | Known angle in the beam | `known_angle` + a real geometry schematic | partial | X (normal) view only; 15.5° has no column v |
| 9 | Fleet performance, cost, outlook | `tracking_perf_all`, cost table | partial | outlook: in-situ calibration at n_TOF, two-track fits, VMM |
| 10 | Conclusions | | | |

Merge candidates if NIM A wants it shorter: §7 into §6 as a subsection; §8 into §9. Suggested: keep §7 separate because it is
the question a detector builder asks.

## Plan

Order by what unblocks the most. Sizes as in `mx_june_cosmic_qa/PAPER_PLAN.md`.

1. **R1 + ladder figure (M).** Hit tiers H0/H1 on pass 1 for five chambers, with CFD, leading edge, peak and matched filter.
   Gives `ladder_compression` and the H1 row. Starts from `rerun/paper1_floor/hits_arm.py` (already builds H0 on the same events)
   and `waveform_first_threading/05_estimator_compare.py` (the July det3 version).
2. **R2 (M).** H2a slope remap and H2b unsharing hybrid refitted on pass 1 (scripts 16, 26–34, 36 in the July study; the hybrid's
   constants were fitted on hits, so refit rather than reuse).
3. **R3 (L, new).** H2c: the forward model fitted to hit lists (one time + amplitude per strip). The key new result: how much of
   the gap a sharing-aware fit recovers without waveforms. If it closes most of the gap, the paper's message changes from
   "waveforms are needed" to "the model is needed".
4. **R5 (M).** The tier benchmark figure: s68 vs angle, bias, implied-v flatness, threading, position, efficiency, same events.
5. **R4 (M).** Threshold 3–7σ, ±1/±2 neighbour readout, amplitude on/off, as re-extractions from the stored waveforms.
6. **R6 (S).** Floor figure in paper style from `ws_floor` and the pass-1 budget (`rerun/paper1_floor`); write the withdrawn-claim
   correction into §5.
7. **Diagrams (S).** Forward-model diagram; known-angle geometry; kernel/sharing schematic that shows c2 < c1.
8. **R9, R10, X1 (S).** Calibration protocol and traps, cost table, X-asymmetry as the model's stated mismatch.
9. **R7 (S).** Known angle: add a hits-arm point so it is a hits-vs-waveforms comparison, not only a waveform check.

## Stated limitations to write down rather than chase

- τ and c2 are bounds (the tail outlasts the readout window).
- X-width excess and the X ±1 asymmetry are unexplained; the symmetric model carries them as a ±0.19° head-on systematic.
- 0.56–0.65° of angle remainder on det2/3/7 X is unexplained; the toy under-generates per-event charge fluctuation.
- Known angle is |tan| of the normal-incidence-referenced view at one mount angle (25.4°); the 15.5° rows have no column v.
- Hits tiers are built from the DREAM waveforms, so the VMM comparison is an emulation of the front-end, not a measurement.

## Decisions for Dylan

1. Is H2c (R3) in scope for this paper? It is the one new-result item and the largest piece of work.
2. Merge §7 into §6, and §8 into §9, for NIM A length?
3. Which chambers carry the headline: all five, or the calibration-row subset used in `fleet.pdf`?
4. Floor framing: report the 0.30–0.35° diffusion floor with the 3–6× gap, or leave the floor out of Paper II and keep it in Paper I?
5. Is the 60 ns depth-bin bias worth a paragraph? It matters for n_TOF steep tracks.
6. One mount angle for the text: 25.4° (pass-1 figure), 25.6° (tex) or 25.64° (README) are all in circulation.
