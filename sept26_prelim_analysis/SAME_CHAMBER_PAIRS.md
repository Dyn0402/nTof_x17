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

**State (2026-10-07):**
- **The bulk angle-scale error is a calibration substitution, not physics.**
  `wft_beam.make_bundle` keeps the bench kernel but replaces the bench drift
  velocity (fitted together with the kernel) with the 42.6 µm/ns prior.
  The geometric in-situ v is A ≈ 38.6 / 37.7 and C ≈ 28 µm/ns.
- **Head-on tracks were lost by the beam seeder.** `MIN_STRIPS_BEAM = 5`, but at
  n_TOF S/N a near-normal track has only 3–4 strips over threshold.
  - With min 3 on cosmics: near-normal tracks ×5, σ_tan 0.2 → 0.03–0.06.
  - Beam purity of min 3: being measured on run_145 stat090_0000
    (`ntof_cosmics/seed_beam_test.py`; §10 of the tracking handoff).
- Still open:
  - chamber C runs the old det6 lp kernel: non-linear response with 10–17 % core tails;
  - A x is mildly S-shaped;
  - the cosmic angle response does not transfer to beam (§7c).

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
| seeder min 3 | T2 | cosmic truth (done); beam purity on run_145 | running |
| geometric v per plane | T2 | free fits vs the A–C line | measured A, C; B, D need another truth |
| C kernel (in situ, or r06) | T2 | free-fit closure vs truth | open |
| xy pairing + rescue floor | T1 | split-ab, bench | passed |
| fixed two-track chain, F per chamber | T1 | split-ab + F ladder | passed at A 1200 / C 2400 on old bundles |
| late-t0 depth grid fix | T3 | refit vs external pointing | proposed |

Order: merge the branches → T2 bundles → T1 recalibration on them → one
combined split-ab contract → re-pass → downstream.
