# Same-chamber coincident pairs — the map of the work

**Written 2026-10-07. This file is on both branches, with the same content, on purpose.**
Read it first if you are working on reconstructing two tracks in one chamber,
on either branch. It says what each thread has established, where its record
is, and how the threads constrain each other. Update it on whichever branch you
are on. When the branches merge, keep the newer text of each section.

## The goal

Reconstruct two tracks from one vertex in **one** chamber, with correct angles.
Same-chamber pairs are about a third of the two-track sample, and the only
vertex test that needs a single chamber calibration. Two problems stand in the
way. They are being worked on separately:

| | thread | the question | branch / location |
|---|---|---|---|
| **T1** | two-track separation | does the reco find **two** tracks when two are there, and only then? | `two-track-joint-fit`, worktree `~/PycharmProjects/nTof_x17_tt` |
| **T2** | single-track angle truth | is each track's **angle** right, near normal included? | `beam-off-cosmics`, main checkout `~/PycharmProjects/nTof_x17` |
| T3 | late-t0 / NNLS runaway | is the **depth profile** observable at all for late tracks? | `beam-off-cosmics` (`qsum_runaway.py`, OCTOBER O10) |

Merge-base: `8ca9ed7` (2026-09-15). The branches share **no code**. They overlap
only in `HANDOFF.md`, `sept26_prelim_analysis/STATUS.md` and
`sept26_prelim_analysis/OCTOBER_2026.md`, so a merge is a docs-only conflict
(keep both sides' entries).

## T1 — two-track separation (`two-track-joint-fit`)

**Where to read:** `sept26_prelim_analysis/TWO_TRACK_LIMIT_RESUME.md` (the
handoff), `TWO_TRACK_FIT_LOG.md` (every measurement),
`HANDOFF_INTRA_TWO_TRACK_RECO.md` (the problem), `wft/TWO_TRACK_FIT_2026-09-16.md`
(the joint fit), `wft/MULTITRACK_2026-09-14.md` (pairing + rescue floor). Report
`<out>/two_track_limit/report/report.html`; slide note
<https://dylan-neff.web.cern.ch/notes/two-track-limit.html>.

**State (2026-10-07):**
- The physical limit is about one strip pitch. The real-track limit is about
  2–3 mm, set by forward-model mismatch.
- The fixed chain plus profile x/y pairing passes the split-ab contract on real
  triggers, A and C, all seven tags of run_145 stat090_0000.
  - Coincident pairs on the overlay bench: A < 12 mm 19 → 58 %; C 36 → 49 %.
  - Clean single muons split: 0.50 % (A) and 0.47 % (C). No event loses a track.
- Operating threshold, from the F rescan on real triggers (cluster 4348153,
  merged 2026-10-07): **A F = 1200, C F = 2400**, the lowest F that meets the
  contract. One step lower fails (A 1000: 0.83 %, C 2000: 0.76 %).
- A common-vertex pair diverges by only 0.13 × its separation across the gap,
  so close pairs are parallel pairs (`pair_angle.py`). In the noise oracle,
  2 mm of divergence resolves 100 % at any separation.
- All of it is opt-in and off in production. Shipping it is Dylan's decision.

## T2 — single-track angle truth (`beam-off-cosmics`)

**Where to read:** `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §7–10 (the full
record), `ntof_cosmics/README.md`. Pooled report
`ntof_cosmics/results/tracking/pooled/report.html`. Truth: the line joining
chambers A and C on run_149 beam-off through-going cosmics.

**State (2026-10-07 morning):**
- **The bulk angle-scale error is a calibration substitution, not physics.**
  `wft_beam.make_bundle` keeps the bench kernel but replaces the bench v with
  the 42.6 prior. Only the scale is affected: the fitted speed w does not
  depend on the bundle's v.
- **The in-situ recipe closes on cosmic truth.** Chamber A: production kernel,
  per-plane robust kw (`insitu_calib.py kwmed`), seeder 3. On held-out events
  y is flat at 1.00 for |tan| 0.15–0.6, x is within ±3 %, and the resolution
  at large angle halves. Chamber C: the r06 kernels (det7 marginally best)
  improve linearity a little (x 1.07 → 0.92 across |tan| 0.08–0.6, from
  1.12 → 0.89), but C's 8–13 % core tails are **not** from the kernel.
- **Seeder minimum 3 (opt-in `WFT_BEAM_MIN_STRIPS=3`)** recovers head-on
  tracks: on cosmics, near-normal ×5 and σ_tan 0.2 → 0.03–0.06. On beam
  (run_145, `ntof_cosmics/results/seed_beam/report.html`):
  - scintillator-confirmed tracks +45 % (A), +52 % (C), +40 % (D);
  - no particle lost, but 5–13 % of production x/y pairings are **re-paired**
    in busy events (a T1 `xy_pairing` question);
  - no close fake pairs made.
- **Beam angles: the capsule estimator is what fails.** The SiPM-wall
  boundaries, binned in strip position (no dilution, no capsule), give true
  tan = 0.89 × production raw for A. They also give an effective source
  distance of ≈ 330 mm instead of 234.6 mm, which is exactly the factor
  k_arm's capsule assumption carries. The capsule-pointing k (1.27, applied in
  stage 3) is therefore not trustworthy. **Still open:** beam 0.89 vs cosmics
  1.11. The t0 100–300 ns class reads 1.05, which points at the t0–p0 trade.
  Tracking handoff §10c.

## T3 — late tracks (`beam-off-cosmics`)

Tracks with t0 > 300 ns have deep depth bins that the 20-sample window cannot
see, and the unregularised NNLS fills them. Their geometry is unreliable: 17 %
of gated tracks. Record: STATUS 2026-10-02, OCTOBER O10, `qsum_runaway.py`.

## How the threads constrain each other

1. **The seeder is shared.** T2 changes `MIN_STRIPS_BEAM` (5 → 3). T1's rescue
   floor, split seeding and `N_CANDIDATES_BEAM = 5` all act on the same
   candidate list.
   - With min 3, small clusters compete for the five candidate slots, which can
     displace a real column.
   - `split_seeds` is called with `min_strips`, so its children get smaller too.
   - **Run the seeder change and T1's fixes together through the split-ab
     contract** (0.66 %, no lost track) before either ships.
2. **Bundle changes move T1's thresholds.** T1's Δχ² statistic and F were
   calibrated on the production bundles (v = 42.6, old C kernel). T2 changes v
   (and C's kernel). The forward-model mismatch that sets T1's 2–3 mm
   real-track limit may be partly the same mismatch, which is a hypothesis to
   test. **After any bundle change, re-run T1's bench and F ladder**; do not
   carry A 1200 / C 2400 across.
3. **Near-normal pairs need both threads.** T1's `pair_angle` oracle and bench
   sit at tan ≈ 0.3. Capsule pairs near the axis are near-normal, where T2 shows
   that production loses the track before any pair logic runs. Add a
   near-normal case to the oracle, and re-measure the bench yield with min 3.
4. **Angle errors T1 should use:** σ(|tan|) from T2's `resolution.csv`, not the
   constant `tan_err` (pulls 1.3 in the core, 9–16 near normal). Also,
   `slope_reliable` fails near normal (T2 §7d), and `det_a_intra`'s `slope`
   selection rests on it.
5. **Late tracks (T3)** contaminate both: flag t0 > 300 ns before quoting a
   pair angle or a two-track efficiency.
6. **Possible shared truth.** run_149 through-goers with the joined A–C line
   could serve as donors for T1's intra bench: real angles with truth, near
   normal included. The waveforms are in `~/scratch/ntof_insitu/beam/`.

## The join: one re-pass

OCTOBER_2026.md §4 makes the campaign re-pass (O4) the single join point.
Everything below must be validated **before** it, and it all rides together:

| change | thread | validated by | state |
|---|---|---|---|
| seeder min 3 (`WFT_BEAM_MIN_STRIPS=3`) | T2 | cosmic truth; beam purity on run_145 | **passed** (+40–52 % confirmed tracks); re-pairing → validate with T1's xy_pairing |
| in-situ v + robust kw per plane | T2 | held-out cosmic truth | **A closes** (`is2_A`); C built (`is2_C`, r06k7); B, D need another truth |
| C kernel | T2 | free-fit closure vs truth | r06k7 adopted; tails not kernel-driven |
| beam angle scale | T2 | scintillator edges (u-binned) vs cosmics | capsule k refuted (D_eff 330 mm); beam 0.89 vs cosmic 1.11 **open** |
| xy pairing + rescue floor | T1 | split-ab, bench | passed |
| fixed two-track chain, F per chamber | T1 | split-ab + F ladder | passed at A 1200 / C 2400 on old bundles |
| late-t0 depth grid fix | T3 | refit vs external pointing | proposed |

Order: merge the branches → T2 bundles → T1 recalibration on them → one
combined split-ab contract → re-pass → downstream.

## Open questions, in priority order (2026-10-07)

1. **Beam (0.89) vs cosmics (1.11) for the same chamber.** The capsule number
   (1.27) is explained: its point-source assumption fails (D_eff ≈ 330 mm).
   The scintillators, binned in u, are dilution-free. What remains is a 25 %
   beam/cosmic difference, with a t0 dependence (100–300 ns class 1.05). Until
   it is settled, no beam opening angle carries a calibrated uncertainty.
   Tests: tracking handoff §10c.
2. **Which x/y pairing is right in busy events?** Min 3 re-pairs 5–13 % of
   production's tracks, and timing and charge cannot arbitrate. Run T1's
   `xy_pairing` together with min 3 through split-ab.
3. **Chamber C's core tails** (8–13 %) are not from the kernel. Candidates are
   the x/y noise, dead or hot strips, and the outward-going sign effect (§7c).
4. **Near-normal reconstructed tracks confirm at only 6–23 %** on the
   scintillators. They may be mostly mis-measured angles, or the pointing may
   miss.
