# Handoff

> **Read first:** `sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md`. This branch
> is one of two threads on same-chamber pairs; the other (single-track angle
> truth, the seeder, v) lives on `beam-off-cosmics`.

## Two-track limit: relative angle of the pairs — updated 2026-10-07 (dylan-MS-7C84)

**Goal:** answer Dylan's question about the two-track slide note: what relative angle do the modelled pairs have, does angle help to separate them, and what angle do real pairs have? Answer it from data, and add slides to `notes/two-track-limit.html`.

**Done:**
- `sept26_prelim_analysis/pair_angle.py` (new):
  - `asimov`: R5, the noise-free Δχ² on a grid of mesh separation d × divergence D over the drift column. Done; output `r5_asimov.csv`.
  - `oracle`: the same grid with noise; running, see below.
  - `data`: donor pointing, R3 pairs with their divergence, event-bench divergence, det_a_intra real vs mixed pairs, and an IPC point-source toy. Done.
  - Outputs go in `~/x17/sept26_prelim/two_track_limit/pair_angle/`.
- `sept26_prelim_analysis/make_pair_angle_slides.py` (new): six slides, wired into `make_two_track_deck.build` before the closing slide. The oracle slide says "pending" until `r5_oracle.parquet` exists.
- The deck is built and checked visually (render in the session scratchpad), but **not yet republished**.
- Results:
  - A common-vertex pair diverges over the 30 mm gap by only 30/L = 0.13 of its separation, so "close" means "parallel".
  - Angle helps a lot. Oracle partial: at d = 0, D = 1 mm resolves 52 % and D = 2 mm resolves 100 %. R3 in chamber A, ideal fit, under 1 mm: 24 % of near-parallel pairs against 90 % of pairs diverging 3 mm or more.
  - Bench donors come from different triggers. In y, their slopes scatter ±0.14 about the pointing line (in x, ±0.05), so 80 % of coincident bench overlays closer than 12 mm diverge by ≥ 3 mm in one view. Below ~3 mm the bench is optimistic for genuine pairs.
  - Real same-trigger pairs in chamber A look like mixed pairs: median opening angle 31–49°, and 0 of 296 k pairs closer than 24 mm, where mixing predicts 1.6 %.
  - IPC toy: 5.2 % of same-chamber M1 pairs (0.4 % of E0) are within 12 mm.
  - Side finding: the fixed chain resolves only 76 % of diverging chamber-A pairs at 3–12 mm, against ~97 % for the ideal fit. Suspected cause: the grid search seeds only parallel line pairs. Not tested.

**Update 2026-10-07:**
- The oracle finished (2 Oct 19:43): `r5_oracle.parquet`; the angle-oracle slide is filled in.
- F-rescan cluster 4348153 merged (A and C, 56/56 shards each). The ladder reproduces the fixed runs exactly at A 1200 / C 2400.
- Result: pick = matched in both chambers. A 1000 fails (0.83 %); C 2000 fails (0.76 %). So C's threshold is not too strict.
- The operating slide's title and callout are now computed from the ladder ("Real triggers confirm the matched thresholds").
- Deck and report rebuilt. **Not yet republished.**

**Next steps:**
1. ~~Republish~~ done 2026-10-07 16:10: the deck now also carries the same-chamber context slides (`make_same_chamber_slides.py`: origin, timeline, chain map, bench progression, pairing, the T2 thread, the join), and its beam-angle-scale slides say the gap is explained (electron scattering, Geant4; cosmic in-situ scale stands). Live at https://dylan-neff.web.cern.ch/notes/two-track-limit.html.
2. Optionally, add a relative-tan dimension to the fixed chain's grid search (`WFT_TWO_TRACK_SEARCH=grid` in `wft/reco.py`) and re-run the R3 split. This targets the 76 % vs ~97 % gap on diverging A pairs at 3–12 mm.
3. Ship decision (Dylan): bundles with profc pairing, fixed-chain env + F on condor, rescue floor, full re-pass.

**Gotchas / decisions:**
- Divergence on real pairs is 30 mm × |Δtan| from the two donors' fits. It carries about ±1.9 mm of measurement error, so the <1 and 1–3 mm classes are blurred.
- The fitted tan is compressed (pointing slope 1/294 in A x and 1/387 in C x, against the geometric 1/235). The common-vertex line uses the geometric value.
- The oracle threshold is R2's own 99th percentile on its tan 0.3, view-x singles (A 25.1, C 29.1), so the new curves are directly comparable with R2.
- The IPC toy ignores capsule size, the pinwheel offsets, dead strips and efficiency. It is a prior, not a measurement.
- This work sits in the git worktree `~/PycharmProjects/nTof_x17_tt`. The main checkout is on `beam-off-cosmics`.

**Key files & commands:**
- `sept26_prelim_analysis/pair_angle.py`: the analysis (`asimov | oracle | data | table`).
- `sept26_prelim_analysis/make_pair_angle_slides.py`: the slides.
- `~/PycharmProjects/dylan-cern-site/scripts/slidedoc.py`: gained `Plot.scatter`.
