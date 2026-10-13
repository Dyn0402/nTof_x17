# Same-chamber coincident pairs — the map of the work

**Written 2026-10-07; both branches merged into main 2026-10-12 (`69c83b6`).**
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
  With the in-situ bundles, A and C agree on beam (capsule k 1.19/1.14 and
  1.19/1.09, from 1.29/1.24 and 1.77/1.54): the chamber-to-chamber differences
  were calibration.
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
  stage 3) is therefore not trustworthy.
- **The beam/cosmic gap is electron scattering (2026-10-07 pm).** Beam reads
  20–33 % shallower at the walls than cosmics, in A and C alike. Three tests
  settle why (tracking handoff §10d–10g):
  - cosmics that fire A's wall read **1.15** × raw (bootstrap 1.12–1.19), the
    A–C line's 1.11 (`ntof_cosmics/cosmic_wall_scale.py`);
  - A–C muons inside beam runs read within 2–5 % of beam-off ones, at ~1/3
    lower gain (`inbeam_through_goers.py`): not gain, noise or space charge;
  - Geant4 with an **ideal** reconstruction (`ntof_cosmics/g4_angle/`): the
    wall estimator reads 1.00 on muons but 0.59 on capture electrons (0.92
    above 4 MeV). The wall sees where a scattered electron lands, not its
    direction in the gap.
  So neither the beam wall scale nor the capsule k is angle truth for beam
  electrons; **the cosmic in-situ scale stands**. Adopting it for beam
  (stage 3 / `k_arm`) is the pending decision.

## T3 — late tracks (`beam-off-cosmics`)

Tracks with t0 > 300 ns have deep depth bins that the 20-sample window cannot
see, and the unregularised NNLS fills them. Their geometry is unreliable: 17 %
of gated tracks. Record: STATUS 2026-10-02, OCTOBER O10, `qsum_runaway.py`.

## Results on the is2_v2 re-pass (2026-10-12)

**Report:** `<out>/same_chamber/report.html` (`make_same_chamber_report.py`). Inputs are
`det_a_intra --arm A|C --t0-max 300 --axis measured --wall` (the re-pass also
`--tan-max-true 0.6`), outputs in `~/x17_local/same_chamber/{prod,cf,is2}_{A,C}`.

**The k used (Dylan, 2026-10-12): the capsule-focus k.** `kcal_is2_v2_cf` = kcal_is2_v2 ×
1.12 on A and C (HANDOFF_TRACKING §22, s = 1.12 with the source offset fitted). The stage-3
table is `~/x17_local/stage3_is2_v2_cf` (root disk: the data disk had 3.8 GB free). The 12 %
tension with the in-beam muons stays **open**: every angle is also shown at the muon k
(`stage3_is2_v2`).

- **Most of production's same-chamber pairs are not two particles.** Their legs confirm on
  the SiPM wall 7 % (A) / 9 % (C) of the time, against 44 % / 33 % for its tracks in
  general. The re-pass keeps 41 k / 6.7 k pairs against production's 182 k / 147 k.
- **One cut is required before any pair work: t0 ≤ 300 ns.** A fifth of gated is2_v2 A
  tracks are flat (tan ≈ 0) with t0 on the 60 ns grid (450/510/570). Two in one trigger
  make a 0° pair with Δt0 = 0 exactly. Uncut, they were 41 % of the re-pass pairs (T3).
- **Wall "both legs confirmed" does not certify two particles.** The wall is four 100 mm
  groups, so a junk leg beside a real one confirms in the same group. `wall2` (legs in
  different groups, each clear of a boundary, both fired) does. It counts the same in both
  chains: A 1,166 re-pass (33 runs) vs 1,095 production (31), C 748 vs 755 (33 vs 28).
  Off-time accidentals 11 / 0. **The re-pass removes junk and keeps the real pairs.**
- **Common vertex on the axis, re-pass only:** lines within 30 mm of each other and 20 mm of
  the measured axis, real over event-mixed, all pairs **A ×2.14 ± 0.19, C ×1.67 ± 0.18**;
  wall2 ×2.51 ± 0.46 in both. Production's all-pairs sample shows none in A (×0.94). It is
  not duplicates: no vertexed leg pair is within 5 mm in either view or under 5°. It grows
  with opening angle (A ×1.6 at 10–25°, ×3.3 above 40°). The muon k gives the same.
- **The in-chamber Δt0 does not separate real pairs.** With the late class cut, all pairs
  and wall2 pairs share the prompt core (A 51 % vs 56 % within ±75 ns).
- **Two-track separation on the re-pass:** the loss is now confined below ~20 mm in either
  view (production: below ~40 mm). The T1 fixes were off in this pass.
- Median opening angle of wall2 pairs at the capsule-focus k: A 29°, C 28° (mixed null 27°
  / 29°). No folded acceptance for wall2 yet, so no physics model is compared.

- **Physics (`same_chamber_physics.py`, report section "are there aluminium e⁺e⁻ pairs?"):**
  the vertexed excess sits over the capsule (A: 73 of 91 excess pairs at y 0–75 mm, capsule
  +3…+83). Folded through the detector, its opening angle fits multiple-scattered ²⁷Al
  conversion pairs (A χ² 6.0/4) and two unrelated leptons (3.6/4) alike; the preference flips
  with k. Unscattered conversion (Geant4 truth angle) is excluded (47/4). So: common-vertex
  lepton pairs from the capsule, consistent with aluminium, but e⁺e⁻ vs two Compton electrons
  of one capture is not separable without energy or charge. True common-vertex pairs are
  ~0.7 % (A) / ~4 % (C) of all same-chamber pairs.

- **CORRECTION 2026-10-13 — the vertex excess above is an artefact.** 77 % of A's pairs
  are in the first 12 ms (flash/epithermal junk), and the event-mixed pool comes from the
  same triggers. With tracks ≥ 12 ms and late-only mixing (`det_a_intra --tof-min-ms 12`):
  A ×1.00 ± 0.11, C ×1.38 ± 0.16 (muon k: 1.19 / 1.23). Geant4 says why the vertex test is
  weak: wall scattering leaves real e⁺e⁻ passing it only 1.4–5 % of the time, chance 2–3 %.

## Physics: data vs Geant4 (2026-10-13, overnight)

**Records:** `same_chamber_g4.py` (Geant4 truth → reconstruction response),
`same_chamber_compare.py` (data vs sim), report "Physics: did we see the aluminium e⁺e⁻
pairs?". Geant4 truth: `MX17_Full_Geant/scripts/same_arm_pairs.py` over
neutrons_thermal_trig_2cm_nose (10⁹ n), condor 4450421, parts
`/eos/user/d/dneff/x17/same_arm_pairs/parts/` (local `~/x17_local/g4_same_arm/parts`).

- **Correlated pairs exist:** after 12 ms the fraction of tracked events with a second
  same-chamber track is flat while the rate falls 3.5×. Fit f = a + b·rate over 12–75 ms:
  a = 0.154 ± 0.012 % (A), 0.157 ± 0.013 % (C), ~2,050 / 1,660 pairs; accidentals ~37 %.
- **Geant4 says they are e⁺e⁻:** of reconstructable same-chamber pairs in triggered
  events, 78 % (A) / 75 % (C) are one photon's conversion pair; two Compton electrons of
  two photons 3–9 %; lepton + own delta 9–11 %.
- **Rate: data ≈ 1/8 of Geant4.** Digitised Geant4 through the real reconstruction
  (g4_digi, is2 bundles, one-sided): P(2 trk | 2 leptons) 0.16/0.14, P(≥1 | 1) 0.54/0.45.
  Predicted 1.19 % (A) / 1.26 % (C); data/G4 = 0.13 / 0.12. Even total late pairs
  (0.26 %) are 4.6× below. Efficiency brackets give 0.12–0.52. Unresolved.
- **Shape:** A's correlated opening angle prefers Geant4 e⁺e⁻ (χ² 6.6/6) over unrelated
  tracks (26/6), mostly from the 0–10° and 40–50° bins; C does not discriminate.
- **Wall double ratio** (energy) 0.55 data vs 0.58–0.64 Geant4 e⁺e⁻ — but event-mixed
  pairs give ~0.6 too, so not a discriminant.

**Next for the factor ~8:** check Geant4's capsule (nose Al thickness, wrap) against the
real capsule; rerun the digitised pair sample with is2_v2's two-sided search and the data
cuts (t0 ≤ 300, |tan| < 0.6) applied to both tracks; look for a trigger/readout effect
specific to two-lepton events (per-trigger rates by ToF).

**Next:** a Geant4 truth reduction of same-arm lepton pairs by origin (conversion / two
Compton / Compton + Møller) to turn the excess into an e⁺e⁻ statement; fold the wall2 acceptance and compare models; T1 on the in-situ bundles (re-derive
F, then split-ab) so the < 20 mm pairs come back; the vertexed pairs' opening-angle spectrum
once there are more of them.

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
| beam angle scale | T2 | cosmic A–C line; cosmic wall test; Geant4 | gap = electron scattering; **adopt the cosmic in-situ scale** (needs a decision) |
| xy pairing + rescue floor | T1 | split-ab, bench | passed |
| fixed two-track chain, F per chamber | T1 | split-ab + F ladder | passed at A 1200 / C 2400 on old bundles |
| late-t0 depth grid fix | T3 | refit vs external pointing | proposed |

Order: merge the branches → T2 bundles → T1 recalibration on them → one
combined split-ab contract → re-pass → downstream.

## Open questions, in priority order (2026-10-07)

1. **Adopt the cosmic in-situ scale for beam?** The beam (0.89) vs cosmic
   (1.11) gap is explained: electron scattering between the gap and the wall
   (Geant4, ideal reco). Check what stage 3 / `k_arm` apply on the campaign
   and what moves downstream (opening angles, the 170° cut, T1), then decide.
   Related: the ideal gap line reads electrons shallower than their emission
   (gap/gun 0.73 at 5 MeV), so the X17 opening-angle spectrum must come from
   Geant4 hits, not truth directions. Tracking handoff §11.
2. **Which x/y pairing is right in busy events?** Min 3 re-pairs 5–13 % of
   production's tracks, and timing and charge cannot arbitrate. Run T1's
   `xy_pairing` together with min 3 through split-ab.
3. **Chamber C's core tails** (8–13 %) are not from the kernel. Candidates are
   the x/y noise, dead or hot strips, and the outward-going sign effect (§7c).
4. **Near-normal reconstructed tracks confirm at only 6–23 %** on the
   scintillators. They may be mostly mis-measured angles, or the pointing may
   miss.

## Follow-up 2026-10-13 (day): time window, vertexing, capsule position

**Slide note:** <https://dylan-neff.web.cern.ch/notes/same-chamber-pairs.html>
(`make_same_chamber_deck.py`). Code: `capsule_tof.py` (beam / g4 / tof / tilt →
`<out>/capsule_tof/`), `same_chamber_scatter.py` (Geant4 truth → `<out>/same_chamber/scatter_*.csv,
vertex_truthline.csv, sep_loss.csv`).

- **The < 12 ms cut is not where the pairs went.** EAR2 flux × 1/v puts 87 % of the beam's own captures
  before 12 ms (peak 5–6 ms), but 92 % (A) / 91 % (C) of the tracks that point at the capsule are after
  12 ms, half after 30 ms. That late population is delayed neutrons, not direct beam. Before 10 ms only 6–25 %
  of tracks point at the capsule (late: 46 % / 42 %), and the own wall fires for 25–75 % of them. The recorded
  rate is roughly flat at 4–30 ms (DAQ throughput). The late data are at least as capsule-pure as Geant4 through the
  reconstruction (31 % / 34 %), so the factor 8 is **not** non-capsule tracks in the denominator.
- **Al e⁺e⁻ cannot vertex.** Geant4 truth: p·β·θ = 0.40 MeV·rad from 0.5 to 6 MeV (wall + ~23 cm of air).
  The median conversion leg (2 MeV) is bent 9° and its line misses the vertex by 33 mm. With **perfect** gap lines
  only 2.3 % of real same-chamber pairs pass the vertex test (chance 2–3 %); the midpoint lands 75 mm off.
  X17 legs (8–10 MeV): 2.3–2.9°, 9–11 mm; 67–82 % have both legs within 20 mm.
- **The expected count already includes opening angle and close-pair loss** (digitised Geant4 through the
  real reco). 44 % / 43 % of Geant4 e⁺e⁻ are < 20 mm apart in a view; 20 % / 27 % more are at 20–40 mm, lost by
  the one-sided pass that the digitised response used but kept by is2_v2. Re-digitising with is2_v2 raises the
  prediction (data/G4 gets worse).
- **Capsule position.** Source at x −9.3, z −3.8 mm (one capsule radius, toward B, a little toward C). A–C show a
  ~3° x-vs-height shear that Geant4's centred capsule does not (tilt or common in-plane rotation). Height: the
  data's y band crossing is +31 (A) / +35 (C) mm; Geant4's nose by the same estimator is −28.
- **OPEN, important: the y-strip orientation.** `build_tracks.IN_PLANE_SIGN_Y = −1` (2026-09-07) flipped
  positions to make corr(y, tan y) > 0; flipping tans instead does the same and differs by a global vertical
  mirror. The July hits analysis (no flip) had the source "3 cm low"; Geant4 says the nose (−28); liquid B (PMT
  on top per survey) brightens toward reconstructed −y. All three favour a flipped y. Angles between tracks do not
  change under the mirror; absolute heights do (`target_y_mm`, `CAPSULE_V`, the 32.2 mm G4 shift, scint v-maps).
  Needs a hardware check (y connector up or down) before anything height-dependent is quoted.

- **UPDATE later 10-13: y mirror confirmed.** Dylan: y strips wired low FEU channel at the bottom. The strip map gives
  y = 0.78 × channel and local y = global +y, so `IN_PLANE_SIGN_Y = −1` mirrors y; the 09-07 fix should have flipped
  tan y. Corrected source height −31 (A) / −35 (C) mm (Geant4 nose −28). Stage-3 fix pending Dylan. The capsule pose in
  the beam is NOT settled: the beam is narrow (50 % within 2.5 mm), so a capsule 10 mm off a centred beam would put
  ~half the beam off it and ~11 % grazing the barrel wall. Handoff: `HANDOFF_CAPSULE_POSE_FIT.md`.
- **y mirror: independent confirmations (later 10-13, `ntof_cosmics/y_orientation_check.py`).**
  (1) Beam-off run_149 cosmics, 74 straight A–C muons with both walls on the n_TOF clock: the stored-higher arm is hit
  *later* (88 % vs 7 %, p = 4×10⁻¹¹, step ±2.3 ns = 66 cm at c). Muons go down, so stored y points down.
  (2) Plastic time − wall time vs stored v: all 8 bars later at stored-high v (+0.15…+0.47 ns/100 mm, 10–40σ), flat in
  u; with the plastic PMTs on top that is the far end. (3) Liquid B firing ×2 toward stored −v (PMT up); horizontal A/D
  answer 10–20× at their +u PMT side (control). Liquid timing is too noisy to use. Plus the wiring, Geant4's nose and the
  July hits result. Nothing points the other way.

- **Capsule pose fit (10-13 evening, `capsule_pose.py`, report `<out>/capsule_pose/report.html`, deck slide `pose`).**
  The beam meets the capsule on the **nose** (late y crossings −31.3 A / −34.6 C, flat 20–75 ms; toy at the Geant4
  pose −28.6 / −29.4). The x crossing measures the beam: x −10.3, z −3.6 mm. The beam–capsule offset is degenerate
  with the unmeasured capsule height (untilted d = 10 needs the capsule ~25 mm low) and with tilt. If the capsule
  is within ±5 mm of the Geant4 placement, ≥ 87 % of the beam enters the ³He and ≤ 4 % of Al captures are on the
  barrel. **The pose does not explain the 1 : 8**: d = 10 keeps Al captures per neutron at 0.95×. The x-vs-height
  step (~4 mm) is a 3.6σ, library-limited hint of a second, higher source. Needed: the capsule tip height
  (survey), the stage-3 y fix.

