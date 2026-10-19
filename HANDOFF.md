# Handoff

## Future-X17 meeting deck (19 Oct) — updated 2026-10-19 (dylan-MS-7C84)

**Resume:** deck live; next: Dylan's feedback → trim main deck to ≤ 11 slides, rebuild, republish.

**Goal:** a short slide note for the informal 19 Oct collaborator meeting: can the MX17 apparatus search for X17
elsewhere (LNL first)? Detectors → n_TOF → where next → papers.

**Done:**
- `talks/future_oct26/make_deck.py` (slidedoc) → `/media/dylan/data/x17/future_oct26/mx17-future-x17.html`,
  live at https://dylan-neff.web.cern.ch/notes/mx17-future-x17.html (site commit ae63480).
- 13 main slides (cover, setups + timeline, best chamber, fleet maps, n_TOF expectation, scintillators/liquids,
  singles, singles vs pairs, energy, LNL, big plastics, neutron facilities + ILL sketch, papers) + backup B1–B7.
- Every chart reads upstream outputs: bench `<key>_rays.csv`, `mpgd26/data/angle_resolution.json`, SPS
  `spatial_resolution/results.json`, scint_stack figures CSVs, capsule_pose `image.csv`, calorimetry `wp/` CSVs,
  facility-search `lnl/out/talk/numbers.json` and `lnl/out/plastics/days_vs_size.csv`.

**Next steps:**
1. Dylan reviews; candidates to move to backup: slides 5, 6, 8 (or fold 6 into 9) — he asked for < 10–11.
2. Rebuild + republish (commands below); commit the site repo after add-note.

**Gotchas / decisions:**
- Calorimetry framing: "scintillator energy scale works (14 Oct wall_plastic), n_TOF 2 cm stack is a range
  telescope; 4–5 cm plates make it a calorimeter, resolution to be measured" — reconciles the 8 Oct "closed".
- Liquid maps are from the 6 Oct scint-stack pass (before the y fix): v may be mirrored; PMT marked only on A/D (u side).
- Fleet numbers are FLEET_DIGEST r06 (constants in the script); budget-page chamber efficiencies are hits-based.
- Paper III/IV split is a proposal, not in PAPER_PLAN.md.
- Layout checked by headless Chrome screenshots cropped per frame (frames found by scanning for non-page colour).

**Key files & commands:**
- `.venv/bin/python talks/future_oct26/make_deck.py`
- `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py /media/dylan/data/x17/future_oct26/mx17-future-x17.html --slug mx17-future-x17 --force --deploy`

## Detector model basics: dead strips, gap, saturation, X footprint — updated 2026-10-19 (dylan-MS-7C84)

**Resume:** FINDINGS §35–38 (cloud_basics), all benches done. Kernel freeze is ON HOLD (Dylan 10-19: not until
everything else is solid). Next: model the X sharing asymmetry (step 3), then masking in the toy (step 2).

**Goal:** get the basics (dead areas, gain structure, saturation, drift gap, X footprint) to the best state the
existing data allow. NO NEW DATA.

**Done 2026-10-19:**
- §35 `strip_gain_audit.py`: strip-scale gain maps + dead/weak/hot strips per chamber + per-event flags.
  det4's amplitude-dependent X width (§32–34) = its dead/weak strips (55 % of head-on windows); clean windows
  are linear (r 0.43–0.48 = toy) and cut-stable at fixed σ0. det6 has 14 dead X channels in a connector
  pattern; det7 4. Nothing in cloud_basics (or the SPS run_71 stacks) masks them yet.
- §36 `gap_extent.py` + toy: timing-free drift gap from inclined tracks' charge extent. det3 28.10 ± 0.23,
  det2 31.45 ± 0.38, det4 30.67 ± 0.41, det7 ~29.9 ± 1.0; det6 not measurable (window too short). One table
  `gaps.py`, now used by unsat_width/bench_fit/driftscan_fit/iso_fit/xy_isotropy/make_xwidth_report.
  det3's excess unchanged; det7 now on Magboltz.
- §37 `sat_residual.py`, `chi2_split.py`: per-event censoring handles saturation (no recovery/neighbour
  anomaly); saturated events' big χ² is brightness. The stack studies are what can't use saturated events.
- §38: the RC model's worse X χ² is its footprint: σ0_x 0.25 gives production's χ² (det3 56 vs 57),
  resolution-neutral overall, head-on +0.04 ± 0.03°; residual left is the X sharing asymmetry.

**Next steps:**
1. (done) s0 benches, all five: σ0_x 0.25 recovers most of the X χ², overall resolution neutral, det4's X
   deficit +0.164 → +0.061°; head-on slightly prefers the wide footprint (det7, det3).
2. Masking: done in rc_diffusion.stack (`unsat_width.py --mask`); det4 still cut-dependent with masking alone,
   needs clean windows. Still to do: the xw_sim toy, and det6/det7 masked widths (det7 unsat sample too small here;
   its HV-pool caches are on the laptop).
3. Add an asymmetric X footprint (or c1_asym-like term) to build_matrix_rc and see if the stack σ0 (0.35)
   and the per-event σ0 (0.20–0.25) converge.
4. Per-event censored width fit (forward model with σ0/Dd as hypers) so saturated events are used
   instead of dropped (70–85 % on det2/6/7).
5. SPS run_71: map det4's dead/weak strips into the beam channel numbering and mask; requote beam σ0/X width.
6. (done, weak) gap_extent on the det3 drift scan: 900/1100 V consistent with 28.1 at ±0.7–1.1 mm.

**Gotchas:**
- gap_extent: use the window *integral* half-height (toy-closed); peak reads +0.6…+2.3 high. Events need
  ≥ 85 % charge in the column and total-charge normalisation, else a few mis-referenced events dominate.
- gap_extent_toy's template peaks 510 ns after arrival: t0 = −30 ns matches the data's peak sample.
- chi2_split per-event JSONs are MBs: they go to `~/x17/cosmic_bench/cloud_basics/chi2_split/`, not results/.
- / (nvme) is at 100 % (2 GB free): keep outputs on /media/dylan/data (= ~/x17/cosmic_bench).

**Key files:** `sps_beam_test_26/analysis/cloud_basics/{strip_gain_audit,xw_clean,gap_extent,gap_extent_toy,gaps,
chi2_split,sat_residual}.py`; arms `~/x17/cosmic_bench/cloud_basics/arms/arm_<det>_rcms{20,25,30,35}.json`;
bench driver `~/x17/cosmic_bench/cloud_basics/run_bench_s0.sh`.

## Detector spec page for Stephan's review — updated 2026-10-16 (dylan-Yoga)

**Resume:** page live with comments on; next: Dylan posts one test comment, sends Stephan the link, fold answers into spec.py + Geant4.

**Goal (Dylan):** before publication, a visual, clickable page with every detail of the MX17 module as we model it,
so Stephan can check it and answer the open questions directly on the site.

**Done:**
- Live: <https://dylan-neff.web.cern.ch/notes/mx17-detector-spec.html> (unlisted). Source `docs/detector_spec/`
  (README there): side view of the full Geant4 stack, top view with zoom insets, the amplification region to scale,
  17 open questions ranked by impact, all 97 numbers with a status (9 unknown, 31 assumed), the four chambers,
  a glossary (calendering defined).
- Questions asked as Dylan requested: resistive strip width/pitch (550/800 µm from memory, vs the 780 µm readout
  pitch), strip thickness (10 µm?), sheet resistance, kapton + glue under the strips, mesh spec (19/48 µm is a
  P2 placeholder) and calendering, window mylar 60 µm single/double aluminised and nm of Al, PCB stack-up,
  field cage, the ~19 mm Y dead bands.
- Comments: the budget page's Apps Script (`x17_facility_search/budget/feedback_apps_script.gs`) now routes
  `page: "detector"` to a "Detector" tab of the v2 budget feedback sheet. Dylan redeployed it (same URL) on
  2026-10-16; verified `?page=detector` answers and the budget GET is unchanged.

**Next steps:**
1. Dylan posts one test comment from the page, then deletes that row in the sheet's "Detector" tab; sends Stephan the link.
2. When answers arrive: read the "Detector" tab (Drive `read_file_content`), update `docs/detector_spec/spec.py`
   AND `MX17_Geant/shared/MX17ModuleGeometry.hh` (+ `design/NEEDED_INPUTS.md`), then
   `python3 docs/detector_spec/build_page.py --publish`. Mark handled rows `status` = done.

**Gotchas / decisions:**
- Flagged on the page: Geant4 puts the L3 guard ring + 264 µm FR4 between the kapton and the pads; the response
  model laminates kapton straight onto the pads. The FR4 thickness is a residual of the CAD's 1.70 mm board.
- The page only POSTs after a GET returns `page: "detector"`; otherwise it falls back to mailto, so detector
  comments can never land in the budget tab.
- Not tested: a live POST (would write a real sheet row). Mock-backend test passed.
- Side-view band heights are log-scaled; a true-scale bar sits beside it.

**Key files & commands:**
- `docs/detector_spec/spec.py` — all data; `template.html` — the page; `build_page.py [--publish]`.
- `DETECTOR_FEEDBACK_ENDPOINT=http://127.0.0.1:8766/mock python3 docs/detector_spec/build_page.py` — mock test.

## UCLA disk rescue + reuse — updated 2026-10-17 (dylan-MS-7C84)

**Resume:** Dylan runs `sudo bash /home/dylan/x17_local/rescue/reuse_disk.sh test` (erase + full write/read test), then read `reuse_disk.log`; if the media-error count didn't move, run `... format`.

**Goal:** get the 26 unreadable files off the failing UCLA NVMe (ADATA SX8200PNP, `/dev/nvme1n1p2`, NTFS),
then decide whether the disk can be reused as scratch space for analysis files that have a copy elsewhere.
Not X17 analysis; the notes live here because this repo is the one reachable from every machine.

**Done:**
- 26 files (26.67 GB) rescued, **1,028,096 B unreadable** (99.996 % recovered). Method: each file's NTFS cluster
  runs (`ntfsinfo`) → a read-only device-mapper device → `ddrescue -d` (pass `-n`, then `-r3`).
- Copied to `/eos/experiment/ntof/data/x17/research_old/ucla/<relpath>`. **All 26 match by name, size and md5.**
- Records + scripts on EOS in `research_old/ucla/_rescue_2026-10-16/`: `damage_report.txt` (what each hole hit),
  ddrescue `maps/`, `files.tsv`, `smart_before.txt`, the scripts, and the rescue's own `HANDOFF.md`.
- Damage: all ROOT files open. Each hole costs whole compressed blocks (baskets), which wipes 1–2 branches over an
  entry range: about 2.4–3 % of entries in the big Cooper-Frye trees, 22–34 % in the proton-only ones, and all of
  one branch in the two small AMPT files. Dca_xy_QA files lose 1 and 5 objects of about 2,000. The text files lose 23 (11.5 GB CSV)
  to 456 lines. Dylan declined partial-basket recovery (2026-10-16).

**Done 2026-10-17:**
- Whole disk compared with EOS `research_old/` by name and size: x17match (18,292 files), Saclay (91,464),
  x17_resp (16), Research leftovers (31, = `ucla/`) all match; mx17_response_sim matches except the 13 tiny
  `results_meshfield_nc1_20261016` logs Dylan said to drop. Other top-level items: chkdsk `found.000`, an old
  mirror of `~/upgrade-kit` (the home copy is newer), Windows junk.
- Deleted (Dylan's OK): staging `/media/dylan/data/ucla_rescue/` (data disk now 31 GB free) and the disk's
  `android/avd` (pixel_9_api34) + `gradle/` cache. `~/.bashrc` still points `ANDROID_AVD_HOME`/`GRADLE_USER_HOME`
  at `/media/ucla`: Gradle rebuilds its cache on the next build; recreate the AVD in Android Studio.
  Android builds fail while the disk is unmounted for the test.
- Decisions: erase from Linux, reformat **NTFS** (Windows access). Firmware: Dylan wants it updated (2026-10-17).
  Do it on Windows with ADATA SSD ToolBox **before** `reuse_disk.sh test`, so the test runs on the new firmware.
  The disk holds nothing needed now, so a failed flash loses no data. Record the new version here.
- Wrote `~/x17_local/rescue/reuse_disk.sh` (checks the drive serial 2K34291HANDC first):
  `test` = SMART → umount → blkdiscard → `badblocks -wsv -t random` → blkdiscard → SMART;
  `format` = GPT + one NTFS partition "ucla" with the old serial `CE90B08890B07913`, so the fstab UUID still
  matches; it also changes the fstab line to `defaults,nofail 0 0` (no boot hang if the disk dies). Log: `reuse_disk.log`.

**In progress / where it stopped:**
- Dylan is switching to Windows now to do the firmware (said at wrap-up 2026-10-17). On Windows the only step is
  ADATA SSD ToolBox; `~/x17_local/rescue/` (script, logs) is on the Ubuntu partition and not visible there.
- Next: Dylan boots Windows, updates firmware in ADATA SSD ToolBox (currently 42B2S7JA), then back on Ubuntu runs `test` (sudo, tmux, hours long). Not started.

**Next steps:**
1. After `test`: compare "Media and Data Integrity Errors" in `smart_pre_erase.txt` vs `smart_post_test.txt`
   and check `badblocks_found.txt` is empty. Unchanged + empty → `sudo bash .../reuse_disk.sh format`. Otherwise retire it.
2. Use it only for data with a copy elsewhere. Monthly: read everything back + check SMART; retire if errors on new data.
3. When the laptop is on: move `x17/sept26_fullpass` + `x17/sept26_slim` back from `yoga:/media/dylan/data/x17_hold/`
   to `/media/dylan/data/x17/` (29 GB; 31 GB free), verify names/sizes, then remove from the laptop.

**Gotchas / decisions:**
- Diagnosis: wear 2 %, spare 100 %, all logged errors are unrecovered reads (status 0x281), and the bad files date
  from 2022–23 → most likely charge loss on data left unused for years, not worn flash. An SSD replaces a bad spot only when it is rewritten,
  so an erase should clear it. The error count rose during the rescue because of our own retries, not because the drive got worse.
- Holes are 4 KiB-aligned and zero-filled in the rescued files; text holes collapse many lines into one NUL line.

**Key files & commands:**
- `~/x17_local/rescue/` (Ubuntu only; copy on EOS above): `rescue.sh pass1|pass2|finalize|status`, `parse_runs.py`,
  `damage_report.py`, `HANDOFF.md`.
- `sudo smartctl -a /dev/nvme1n1` — health; compare the media-error count.

## Track passport (tracking + MM dE/dx + scintillators per track) — updated 2026-10-15 (dylan-Yoga)

**Resume:** published 2026-10-15 as a 26-slide pedagogical note (principle → simulation → real detector): https://dylan-neff.web.cern.ch/notes/track-passport.html (`ntof_calorimetry/make_passport_deck.py`). The long report is `/media/dylan/data/x17/calorimetry/passport/report.html`.

**Goal:** Dylan asked for as much information as possible about each track, from tracking and energy together. It is a demonstrator of the set-up, not X17 physics.

**Done (results in `ntof_calorimetry/MM_EDEP.md`, "Track passport"):**
- `mm_edep_ana` ran for A+C. The beam MM calibration is fixed (beam-sim k; sim bias −0.46 → 0.00), giving per-track σ = 0.16 (A) and 0.22 (C). Stage 4 was rerun at σ 0.16.
- `track_passport`: 15,444 run_145 tracks (12,818 with an MM deposit). Cosmic path length: MM sec^0.97/1.03 (Heed 1.07), wall sec^1.0/1.3. The MM tags tracks with several particles in the gap (AUC 0.84); the wall cannot (0.47). Data and Geant4 agree that MM and wall are uncorrelated. The track angle improves the plastic energy for steep tracks (1.40 → 1.07 MeV) and for multi-particle tracks (1.70 → 1.08 MeV). Capsule-vs-elsewhere AUC from the geometry is 0.75. Example cards for 12 data tracks are in the report.

**Next steps:**
1. (done 2026-10-15: logged on the x17 board as a log entry and an output.)
2. Add time information: MM t0 vs scintillator pulse times per track (stack `*_dt_on`), as a fourth timing measurement.
3. Get a Geant4 reduction with the liquid (`stack_edep.py` on branch `lnl`). Geant4 "through the plastic" is T > 6 MeV for now.
4. The 2026-10-14 slow-tag AUCs (0.50/0.59/0.63) did not reproduce (now 0.74/0.66/0.80). Trace why.

**Gotchas / decisions:**
- The desktop data are reachable from the laptop through Windows on the same disk: `ssh desktop-win` → `D:\x17\...` (Ubuntu `/media/dylan/data`). Copied to the laptop so far: `calorimetry/`, `sept26_prelim/{kcal,stage3_fullpass/tracks_run_145_*}`, `scint_stack/{tracks/stack_run_145,ana,geometry.csv}` and `ntof_cosmics/g4_angle/neutrons_nose`.
- The desktop's pickles are numpy 2; the laptop venv has numpy 1.26 (the shim segfaults). The laptop copies were re-saved through parquet in a numpy-2 env, and the originals are kept as `*.pkl.np2`.
- Beam electrons scatter between the gap and the wall, so the clean path-length test is the cosmic one.
- Data counter energies are moved onto Geant4's scale (wall_plastic scale fits) before the kNN estimators are applied.

**Key commands:** `PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.track_passport` (~10 min), then `-m ntof_calorimetry.make_passport_report` and `-m ntof_calorimetry.make_passport_deck`; republish with `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py /media/dylan/data/x17/calorimetry/passport/deck/track-passport.html --slug track-passport --force --deploy`. The deck caches the beam-calibration histograms in `passport/deck_cache_beamcal.pkl`; delete it after rerunning the analysis.
- Site repo: pull before any add-note --deploy. A deploy from a stale tree pushes stale pages (the 15 Oct pair-display note had been deployed from the laptop uncommitted while origin was 11 commits ahead; committed now).

## Pair event displays — updated 2026-10-15 (dylan-Yoga)

**Resume:** 7 3D pair displays done and published; next: render more triggers / other runs if wanted (needs desktop stage 3).

**Goal:** 3D GIF event displays of selected pair candidates for the 15 Oct meeting. Each shows TPC charge, tracks, the fired scintillators and the capsule at its inferred pose.

**Done:**
- `sept26_prelim_analysis/pair_event_display.py` (main d55b293), four steps:
  - `candidates`: rebuilds run_145 stat090_0000 stage 3 from the is2_v2 reco, then pairs via `det_a_scint.match_run` + `det_a_intra.pairs` (both unchanged) plus two-arm A–C pairs;
  - `render`: GIF + MP4 + PNG per trigger;
  - `report`: report.html gallery;
  - `note`: self-contained note version.
- Rendered (in `/media/dylan/data/x17/sept26_prelim/pair_displays/gif/`):
  - same-chamber `wall2`: A 24576, A 52903, C 4748, C 49942;
  - two-arm: 38414, 23580, 18841.
- Published: https://dylan-neff.web.cern.ch/notes/pair-event-displays.html (md5 checked).
- 24576 for the slides: https://dylan-neff.web.cern.ch/files/x17_pair_24576.{gif,mp4,png}. The same files are in CERNBox at `/eos/user/d/dneff/x17/pair_displays/`.
- A Google Doc of links, "X17 pair event display 24576 — GIF download links", is in My Drive. Drive got a link doc rather than the GIF because the connector only takes inline base64 and this laptop has no rclone or Drive mount.

**Next steps:**
1. For more events or runs, run on the desktop. Point the code at the real `stage3_is2_v2_cf` and the slim parquet (the script currently rebuilds stage 3 locally, for this one sub-run only).
2. Waveforms for new triggers: `lxplus_cut_script()` in the module makes the lxplus cut script (LCG_105). Copy the npz files into `pair_displays/wf/`.

**Gotchas / decisions:**
- k here is the capsule-focus value: kcal_is2_v1 × (1.009, 1.003) × 1.12, giving A 1.103, C 1.081. The kcal JSONs are only on the desktop. Muon-scale angles are about 12 % smaller.
- TPC charge comes from decoded_root, not hits. Depth = (t − 140 ns − t0)·v/k, where 140 ns is the bundle template peak. It lines up with the fitted tracks on all 7 events.
- None of the 7 passes the vertex cut. Two-arm pairs at ≥170° are probably single through-going particles. C 4748 has Δt0 −281 ns. The page lists these per event.
- pyvista off-screen needs `pl.render()` before `screenshot()`, or every frame is identical. A per-frame GIF palette makes the legend flicker, so the code builds one shared palette.
- `eosexperiment.cern.ch` does not resolve off-site. Cut the files on lxplus instead.

**Key files & commands:**
- `PYTHONPATH=. .venv/bin/python -m sept26_prelim_analysis.pair_event_display render --events s:24576,t:38414` (s: = same-chamber, t: = two-arm)
- outputs in `/media/dylan/data/x17/sept26_prelim/pair_displays/` (wf/ = cut waveforms, 111 MB)

## Repo merges and cleanup — updated 2026-10-14 (dylan-MS-7C84)

**Resolved 2026-10-15:** `cut_scan.csv` was truncated by `yield_gate.py scan --runs run_86` (it rewrote the table with run_86 only). The committed table is correct: a full re-scan reproduces it exactly. The working copy was restored from git; `yield_gate.py` now keeps the other runs' rows on a subset run (branch `yield-gate-table-merge`). The cosmic analysis this table feeds is the is2_v1 check (see `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §15), superseded by is2_v2 (§19–20).

**Goal:** get the x17 repo's branches merged into main and pushed, and clean up the leftover branches.

**Done:**
- `worktree-sps-attachment-transfer` merged into main (e9dac9d, pushed). The HANDOFF.md conflict was resolved by keeping both logs; the branch's 10-08 to 10-11 sections sit before the 10-09 cosmic-bench section.
- `bench-pin-rc` and `origin/mx17-paper-status` merged into main (7a98ab7, pushed). HANDOFF.md: the 10-12 X-width section is at the top; the 10-12 cloud_basics section replaces 10-11.
- Garfield test fix and drift-consistency scripts committed (46aeaad, pushed).
- Local branches that were merged into main were deleted (beam-off-cosmics, same-chamber-oct, y-fix-regen, among others).

**In progress / where it stopped:**
- Still attached and merged: bench-pin-rc, is2v2-compare, mx17-paper-status, two-track-joint-fit, worktree-capsule-scale-kernel, worktree-sps-attachment-transfer. The same branches exist in the `nTof_x17_paper` and `nTof_x17_tt` clones, which are checked out on them.

**Next steps:**
1. Clean up the attached worktree branches once Dylan confirms (see the workspace handoff for the same question).

**Gotchas / decisions:**
- `beam-off-cosmics`, `same-chamber-oct` and `y-fix-regen` disappeared from the `nTof_x17_paper` and `nTof_x17_tt` clones during the sweep. The cause was not found. All three are ancestors of `main` in this repo, so no commits were lost.
- `main` merges were done without review, as Dylan asked.

**Key files & commands:**
- `HANDOFF.md` — the two merged logs
- `git worktree list` — attached branches

## X width excess: done with the data we have — updated 2026-10-17 (dylan-Yoga)

**Resume:** superseded by the 2026-10-19 section above (gap table fixed via gap_extent; kernel freeze on hold).

**Goal:** explain the X drift-width excess so the detector-model kernel can be frozen. NO NEW DATA: the detectors are gone.

**Done 2026-10-16/17** (§30 desktop, §31–32 laptop, §33 lxplus, §34 laptop):
- §30–31: saturation biases v low and D_T high (cut at 3500 ADC); `invert` water-branch bug fixed; det3 gap 27.9 mm.
- §32: det3 excess at every field, cut and |tan| cut; det4/det7 D_T falls ~100 with the amplitude cut.
- §33: closure toy (xw_sim.py, condor 4474928): the estimator has no amplitude bias. Unsat closure D_T +3…+17.
  Centring on the true track inflates by +90…+135.
- §34: signed-tan scan: det3 shows no field tilt (δ ≈ +0.02 everywhere, floor still ~350 against the toy's 298); the
  dish would need a 0.05 rad tilt spread, about 4× the map's steepest gradient. Amplitude terciles: det3 +57…+84 in
  every tercile. det4/det7 classes differ in both directions; det7's faint tercile is edge/empty events. No residual
  baseline. Model-free neighbour/peak ratio: on det4/det7 the core is narrower than linear below ~2000 ADC; det3
  follows the toy. No compression below the ceiling.

- Slide note (14 slides, open consistency questions + ideas on slide 13): https://dylan-neff.web.cern.ch/notes/x-width-excess.html,
  built by `cloud_basics/make_xwidth_deck.py` from results/*.json; rebuild + `add-note.py … --slug x-width-excess --force --deploy`.

**Next steps:**
0. Cheapest consistency fix first: unsat_width.GAP_STUDY uses 30 mm for det4/det6, but GAP_STUDY (K = 22) measured det6 27.85
   (det4 25.55, amplitude-dominated). Put one gap table in one place, requote det6 (v −7 %, D_T +4 %), rebuild the deck.
   The other open questions (det2 vs det3 at 347 V/cm, water per chamber/night, run_71 unsat cut, δ vs alignment) are on slide 13.
1. Freeze the kernel: Dd per chamber from the unsaturated stack at the 3500 cut, with +100 µm/√cm (det3, unexplained)
   and ±100 µm/√cm (det4/det6/det7, amplitude-dependent footprint) as stated systematics. Then the kernel arms on
   unsaturated stacks, and the gate.
2. Update make_model_deck.py slides 4 & 18 (saturation dominant; det3 excess unexplained; det4/7 amplitude-dependent
   footprint), rebuild, republish.
3. Optional, from existing data: does the wft forward model (whose neighbour copies scale linearly) fit det4's
   faint events worse than its bright ones? That would show whether the non-linearity is in the copies or in σ_p0.

**Gotchas:**
- unsat_width.py writes one JSON: never run two at once. Keys: `<key>[@sat<N>][@ref][@tan<T>]`; `--recompare` re-does
  only the Magboltz comparison. `--tan` sets x_width.TAN before the pool forks.
- x_width.det_of / drift_hv resolve `<key>__<subrun>` caches (from headon_cache.py --subrun).
- Any v → water inversion at ≳300 V/cm must use `invert(...)['branches']`.
- `pkill -f` with a pattern that also matches your own shell's command line kills that shell: use pgrep + kill PIDs.
- `o22_long_det3`/`o22_det3` fits sit at v 22.69 ± 0.00 (stuck at a bound): ignore them.
- xw_sim: `prep` rebuilds inputs.pkl and wipes the tune, so run `tune --merge` after it (the tune_*.json are committed).
  On lxplus a replica takes ~3 s. If `.bashrc: Permission denied` shows up there, the AFS token has expired: run `aklog`.
- xw_tan/xw_amp/xw_compress run on the laptop's 8 cores in minutes (inputs.pkl + the head-on caches). Not on lxplus:
  the caches are local.

**Key files:** `sps_beam_test_26/analysis/cloud_basics/` — `unsat_width.py`, `x_width.py`, `rc_diffusion.py` (CB_CENTRE),
`bench_water_e.py`, `headon_cache.py`. Caches: `~/x17/cosmic_bench/cloud_basics/xwidth/caches/` on both the desktop
(original set) and now the laptop (the §32 set; det2/det6 not built here, they need event caches or ref-only).

## X width excess (UNSOLVED, superseded by the section above) — updated 2026-10-12 (dylan-MS-7C84)

**Resume:** OPEN PROBLEM. Bench X widens ~dry-like while v says wet; no gas fits. Next: det3 Y joint fit, Magboltz cross-check, contaminants, beam X template. NO NEW DATA (detectors gone).

**Goal:** explain why the X cluster width grows with drift depth faster than Magboltz allows. Until it is explained the
detector model's drift diffusion is empirical (Dd fitted from X widths), and the re-pass stays on hold (Dylan, 10-09).

**Established (FINDINGS §29; report `~/x17/cosmic_bench/cloud_basics/xwidth/report.html`):**
- Every bench dataset shows it (5 chambers, 8 runs, det3 drift scan at 6 fields and 2 gain scans). X D_T is 400–520 µm/√cm.
  Ar/iso 95/5 with the water each run's v needs (0.6–1.0 %) gives 270–325. The extra is +300–430 in quadrature.
- It does not depend on gain (det3, 425–505 V). It is present over the whole active area, with no hot spot. Only the
  early (prompt) width depends on position: ×~2 in ⟨x²⟩ at the edges. That is a separate edge/footprint effect.
- It lands at the anode (no X undershoot, rc_D_x → 0, §17). It is not track inclination (§28), and δ-rays do not
  grow with depth.
- No Ar/iso/water mixture fits det3's v(E) and D_T(E) together: joint χ² 80/36 (grid iso 0.5–8 %, water 0–1.5 %).
  An iso-poor gas (~1 %) can fit a single-field chamber (det4, det7), but not det6, and not det3's scan.
- The beam has it too. det4 run_71 X, template-free fit: +118 / +143 / +286 in quadrature at 243 / 150 / 92 V/cm,
  above the thermal floor at 92. This is indicative only: at that field the drift time is 3× the window.
- It is closer to per-depth than per-time (det6/det7 ratio 0.86 ± 0.11 vs 1.16 per-time, ~2.7σ). Bench and beam
  per-time differ ×20.

**Hypotheses left, not separable with data in hand:**
1. Magboltz's water model: water slows electrons as predicted but cools them transversely less. Both setups are wet.
2. A non-gas lateral term linear in drift depth, not in any current model.
3. (weaker) A contaminant other than water (N2, CO2, …), or isobutane depletion beyond what det3 allows.

**Next steps (follow-up):**
0. **No new data can be taken: the detectors are gone (2026-10-16).** A dry-gas run is not possible; work with what we have.
2. Cheap: det3 drift-scan **Y** view (caches already hold Y). A joint X+Y fit per field: does Y's Dd agree with X's?
3. Magboltz cross-check: the same mixtures in another code or another cross-section set (Biagi water vs LXCat), or
   published Ar/iso/H2O D_T data.
4. Beam: get a beam X template (re-extract the run_71 bundle) and redo `x_width_beam.py` with the template model.
   Use the 25.6° runs (run_62/63) for a known-angle cross-check.
5. Other contaminants: add N2 / CO2 / air columns to `make_iso_jobs.py` (no O2 on the bench, §21).
6. The edge early-width effect is separate: worth a look on its own (frame/field distortion, footprint, M3 at the edges).

**Decision for the kernel meanwhile:** take Dd from the X stack (≈ dry), not from v (§28). Never carry a v-derived water
fraction into D_T.

**Key files & commands** (`sps_beam_test_26/analysis/cloud_basics/`, run with `../../../.venv/bin/python`):
- `headon_cache.py <key> [--subrun S --base DIR]` — head-on caches → `~/x17/cosmic_bench/cloud_basics/xwidth/caches/`
  (kept). The det3 scan sub-run *waveforms* were only in job scratch (gone). Re-pull with
  `rsync -a --exclude='long_run*' --include='*/' --include='decoded_root/***' --include='m3_tracking_root/***' --exclude='*'
  lxplus:/eos/experiment/ntof/data/x17/cosmic_bench/june_tests/Run/mx17_det3_saturday_scan_6-27-26/ <dir>/mx17_det3_saturday_scan_6-27-26/`
  and copy run_config.json in. (/media is nearly full, so use the nvme.)
- `x_width.py`, `x_width_map.py`, `x_width_beam.py` → `results/x_width*.json`
- `make_iso_jobs.py [--ext]` + lxplus `~/cloud_basics_condor/att/condor_mb_iso{,ext}.sub` (condor 4440949, 4440956) →
  `results/iso/`; `iso_fit.py` → `results/iso_fit.json`; `make_xwidth_report.py` → report.
- Refresh after the last 5 Magboltz points (dry, near-pure argon) land:
  `scp 'lxplus:cloud_basics_condor/att/magboltz_benchi*' results/iso/ && python iso_fit.py && python make_xwidth_report.py`

**Gotchas:**
- At ≳300 V/cm, v is not monotonic in water for 95/5. Inverting v → water there is ambiguous, so take the closest grid value.
- det3 gap 27.9 mm (dished cathode): scale v ×27.9/30 and D_T ×√(30/27.9) against fits made with GAP = 30.
- Stack v (GAP/T) runs ~10 % below bundle v on det2/4/7. D_T is ~insensitive (σ_drift/√gap), but the water inference is not.
- Branch bench-pin-rc is 40+ commits ahead of main (the whole cloud_basics line) and has never been merged: Dylan to decide.

## MM energy deposit per track (dE/dx) — updated 2026-10-14 (dylan-MS-7C84)

**Resume (2026-10-15):** steps 1–3 below are done (A+C ana, beam k fixed, combo at σ 0.16). The combination continues in the "Track passport" section above.

**Goal:** Dylan's ask: how accurately a reconstructed track's energy deposit in the MM gap is measured, first in
sim then on n_TOF data, and whether it helps the scintillators identify the track's energy. (The old M2 note only
said 1 vs 2 MIPs don't separate; that was not the question.)

**Done:** full record and numbers in `ntof_calorimetry/MM_EDEP.md`. Headlines: sim accuracy 10 % (A, contained) /
16 % (C); data add a 1D-along-x gain structure (removable with an in-situ map) plus a random per-track gain jitter
(σ 0.22–0.27 cosmics, 0.09 beam after map, from the truth-free halves test) → ~13–17 % beam, ~25 % cosmics.
Cosmic map does not transfer to beam. MM unusable < ~9 ms after the flash. Data: tracks firing no scintillator
carry +29 % MM charge (Geant4 predicts +15 %).

**In progress / where it stopped:**
- All four sims are on disk (`sim_{cosmic,beam}_{A,C}.parquet`); beam C finished at wrap-up, not yet analysed.
- `mm_edep_ana.chamber('A')` ran (→ `<calo>/mm_edep/results/chamber_A.json`); `main()` (A+C) not yet run.
- No report yet.

**Next steps:**
1. `PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.mm_edep_ana` (A+C).
2. Fix the beam calibration: use the beam sim's own `window_curve` / k for beam (see MM_EDEP.md "Known issues").
3. `python -m ntof_calorimetry.mm_edep_combo <σ>` with the measured σ (≈0.15 beam); add data category table
   (code for it is in the session, re-derive: stack_run_145 pulses vs WALL_THR/PLAS_THR, pointing at plastic).
4. Investigate the data "none" excess being 2× Geant4 (second particles? out-of-time tracks?).
5. `make_report` (report.html in `<calo>/mm_edep/`) and a slide note (slide-note skill), log on the x17 board.

**Gotchas / decisions:**
- Heed runs only on lxplus (desktop PyROOT is built for py3.10, system is 3.12). Reach lxplus only as `lxplus`.
- Bundles' per-channel gains are all 1.0. χ² tracks charge, not quality. Don't `pkill -f` (kills own shell).
- Disks: / and /media/dylan/data each have ~3.5 GB free.

**Key files & commands:**
- `ntof_calorimetry/{heed_edep,mm_edep_sim,mm_edep_data,mm_edep_ana,mm_edep_combo}.py`, `MM_EDEP.md`
- outputs `/media/dylan/data/x17/calorimetry/{heed,mm_edep}/`

## Stage-3 y fix — updated 2026-10-14 (dylan-MS-7C84)

**Resume:** DONE. Regenerated, audited and republished overnight 10-13→14. Remaining items are optional cleanup (below).

**Goal:** HANDOFF_Y_FLIP_FIX.md: fix the 09-07 y mirror at the source, guard it, regenerate everything, audit,
correct the record.

**Done 2026-10-14 (desktop):**
- `Y_FIX_REGEN.sh` ran end to end. Every live stage-3 set is `y_convention: "up"` and was proved an exact y
  mirror row by row: fullpass, is2_v2, is2_v2_cf (292 each), campaign (287, `--allow-drift`: its ToF columns
  predated the 09-11 trigtime rebuild), cosmic k_run_147/150. All guards pass; cosmic verdict "UP"; tests 10/10.
- Data disk full (3.8 GB): `regen_stage3 --work` stages and backs up on / (`~/x17_local/yfix_work/`,
  `<dir>/pre_yfix` symlinks). Largest set first.
- Grids that had been sized on the mirrored heights, found and fixed: `capsule_tof.Y_EDGES`
  (now −120…+60) and `capsule_y.Y0_GRID` (now −90…+90).
- The Athens side view compared the crossing with the gas centroid. It now uses the Geant4 nose
  (`CAD_CROSSING_Y = −27.8`), so the CAD shift is −3.5 mm.
- Added to the chain: late same-chamber pair sets (`same_chamber_late`, needed by `same_chamber_compare`) and
  `tracking_qa`.
- Results: capsule −31.3 / −34.6 (band crossing); ray trace −32.3 (A+C, campaign); G4 −28 recovered.
  Same-chamber excess mirrored, counts/angles unchanged. Liquids B, C now rise toward their PMT. Plastics 0/8.
  Full table: `Y_FLIP_AUDIT.md` §3–4.
- Republished: y-orientation (new "fixed" slide; evidence slides read the frozen `pre_fix/` inputs),
  same-chamber-pairs, capsule-pose, scint-stack; pair-vertex-y erratum; track-QA shards + tiers; board log;
  question closed.

**Optional next steps:**
1. Delete the backups (`~/x17_local/yfix_work/*/pre_yfix`, ~30 GB on /, plus the `pre_yfix` symlinks) once
   satisfied. Each old table is exactly recoverable by negating the five y columns.
2. `stage3_is2_v1_pilot` is still mirrored (superseded, no reader). Regenerate it if space allows:
   `.venv/bin/python -m sept26_prelim_analysis.regen_stage3 <out>/stage3_is2_v1_pilot --work ~/x17_local/yfix_work --combine [--allow-drift]`.
3. `mx_june_wft/drift_consistency/` (untracked) and the `garfield_test.py` / `cut_scan.csv` edits predate
   this session and were left alone.

**Gotchas / decisions:**
- The y-orientation deck's evidence slides must keep reading `ntof_cosmics/results/y_orientation/pre_fix/`.
  The regenerated files show the fixed frame.
- The band crossing images the capsule NOSE, not the gas centroid. Compare it with the Geant4 crossing (−28),
  never with 0.8.
- `same_chamber_compare` reads `~/x17_local/same_chamber_late/` (ToF ≥ 12 ms), not `same_chamber/`.

**Key files:** `Y_FIX_REGEN.sh`, `Y_FLIP_AUDIT.md`, `COORDINATES.md`, `sept26_prelim_analysis/regen_stage3.py`.

## Capsule pose in the beam — updated 2026-10-15 (dylan-MS-7C84)

**Resume:** biased, gas-fixed Geant4 library running (cluster 4467046); next: steps+reduce, digitise, pool, refit, rebuild deck.

**Goal:** where the ³He capsule sits relative to the beam and the chambers (sets the ³He/IPC/X17 yield; feeds the 1 : 8
pair-rate question). Full physics state: Correction box of `sept26_prelim_analysis/HANDOFF_CAPSULE_POSE_FIT.md`.

**Where the result stands:**
- Real EAR2 beam since 14 Oct (Lambda2D was the resolution function). Capsule ~1 cm below drawn; leans with its top
  toward B and A (φ ≈ 135°, θ ~7–15°); sideways beam/capsule degenerate (³He sees 20–53 % of the beam).
- z lean confirmed without D (15 Oct): A/C band-slope depth, `capsule_pose data_depth` (+4.6 mm toward A from h −80 to
  −20; D +4.5); stable to ~0.4 mm across ToF slices / run halves / wall tracks.
- Caveat: z moves below the source height, x above it; one rigid tilt fits neither shape well. Unmodelled hardware:
  the thin Al hanger tube above the capsule bends toward the A–D corner (Dylan: likely sets the tilt). Not to be modelled
  unless asked; check it first when a shape doesn't fit.
- Deck (published): https://dylan-neff.web.cern.ch/notes/capsule-pose.html (`make_capsule_pose_deck.py`).

**Running / where it stopped:**
- **Cluster 4467046** on lxplus: 200 x 5M EAR2 neutrons, 19.5 m, `--bias-capsule 20`, built from MX17_Full_Geant e3def59
  (gas primary-ionization fix + capsule-Al capture bias) in `/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_v2`.
  ~25 min/job, ~4 GB/file (~0.8 TB) -> `/eos/experiment/ntof/data/x17/full_sim/neutrons_thermal_ear2_L19p5_bcap20/`.
  Submit dir `~/condor/mx17_neutrons_ear2_L19p5_bcap20`. Worth ~1700 analog jobs for the library.
- Analog extension (cluster 4465059, old gas) REMOVED at Dylan's request. Its 400 partial files were moved (not deleted)
  to `full_sim/neutrons_thermal_ear2_L19p5/_killed_ext_2026-10-15/` (362 GB); delete only with Dylan's OK.
- Analog EAR2 libraries done: 19.5 m digitised (`g4_digi/g4_{A,C}_ear2_L19p5/`); 19.95 m steps/reduce on EOS
  (`full_sim/angle_scale/{steps,neutrons}_ear2_L19p95`), not pulled/digitised (old gas; low priority now).
- Local background waiters die with this session; nothing local is running.

**Next steps:**
1. When 4467046 is done (`condor_q`; check holds): on lxplus `cd ~/condor && bash make_ear2_subs.sh
   neutrons_thermal_ear2_L19p5_bcap20 ear2_L19p5_bcap20 && condor_submit steps_ear2_L19p5_bcap20.sub && condor_submit
   reduce_ear2_L19p5_bcap20.sub`. Reduce peaked ~12 GB on 1.2 GB files; these are ~4 GB: test one job's memory first or
   raise MEM in make_ear2_subs.sh. `lxstore status` before/after.
2. Pull `angle_scale/steps_ear2_L19p5_bcap20` -> `/media/dylan/data/x17/ntof_cosmics/g4_digi/` and
   `neutrons_ear2_L19p5_bcap20` -> `.../g4_angle/` (local disk has only ~3 GB free: check sizes; steps were 0.5 GB/100
   analog files, biased will be larger), then `bash ntof_cosmics/g4_digi/run_ear2.sh ear2_L19p5_bcap20`.
3. Pool: `POSE2_LIB=nose,ear2_L19p5,ear2_L19p5_bcap20`; rerun `capsule_pose shear`, `pose2` / `pose2z` / `pose2d`
   (families: `POSE2_XB=0 POSE2_XC=-16,-18,-20`; `-5 / -10,-15`; `-10 / -10,-15`) into a new out dir with copies of
   data*.csv (as `capsule_pose_ear2_pool/`); regenerate `make_pose2_report.py` and the deck (point POSE2 at the new dir).
4. If the toy is still the limit, extend the biased run (more jobs, same sub with new seeds).
5. Still needed from Dylan: collimator, capsule tip height/holder, survey/photo.

**Gotchas / decisions:**
- Library size: the toy uses capsule tracks captured within r < 8 mm; the real beam puts only 61 % there (93 % for
  Lambda2D). Pooling libraries is valid (the toy translates each track to its own capture point); `library()` resolves
  the reco tan signs per part (raw y-tan sign differs between digitiser vintages; both focus equally).
- `run_digi` needs the run's own reduction (`G4DIGI_REDUCED`; `run_ear2.sh` sets it) or it silently digitises ~1 event.
  ADC/e now comes from that reduction's median track (was fixed 247 e-); ear2_L19p5 was digitised with 247 (~3 % high
  amplitude, harmless for pointing). Gas-fixed libraries will give ~10 % fewer e-.
- `pose2d` (toy prediction of the A/C depth curve) is library-limited (pooled toy sd 1.2–1.4 mm vs data 0.15); the deck
  uses the depth data reading, not its χ². Pooled x-shear toy sd 0.3–0.6 mm.
- Geant4 11.2 (lxplus) records LS showers in ~10x more steps than 11.3: biased files are big. Not a port bug (cosmics
  identical old/new).
- uproot can't read EventTree's `capture_vol` char arrays; use PyROOT (`PYTHONPATH=~/Software/root_6_40/lib`,
  `bytes(e.capture_vol)`), or the reduce parquet.
- Never quote the capture source as the beam position; D z only from wall-coincident tracks ≥ 25 ms.

**Key files & commands:**
- `sept26_prelim_analysis/capsule_pose.py` — steps `pose2`, `data_z`, `pose2z`, `data_depth`, `pose2d`; env
  `POSE2_XB/XC/THETA/PHI/ALPHA/L/LIB` (LIB comma-list pools).
- `sept26_prelim_analysis/make_capsule_pose_deck.py` → `<out>/capsule_pose/deck/capsule-pose.html`; publish with
  `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <html> --slug capsule-pose --force --deploy` (pull the
  site repo first).
- `ntof_cosmics/g4_digi/condor/make_ear2_subs.sh <simdir> <tag> [first_job]`, `ntof_cosmics/g4_digi/run_ear2.sh <tag>`.
- MX17_Full_Geant: `--bias-capsule <f>`; its HANDOFF.md has the validation numbers.

## Same-chamber pairs + the y mirror — updated 2026-10-13 (dylan-MS-7C84)

**Resume:** y mirror fixed AND regenerated (10-14, see the section above); pairs 1:8 still open.

**Goal:** same-chamber (intra) pair physics against Geant4 for Dylan's Thu 2026-10-15 talk. Along the way we found
that stored global y is upside down.

**Done (10-12 → 10-13):**
- Pair results on the is2_v2 re-pass at the capsule-focus k (`kcal_is2_v2_cf`, provisional):
  - correlated pairs are 0.15 % of late events;
  - Geant4 says ~78 % of them are Al e⁺e⁻;
  - data ≈ 1/8 of Geant4;
  - the 10-12 vertex excess was retracted.
  - Record: `sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md`; report `<out>/same_chamber/report.html`.
- 10-13 follow-up (`capsule_tof.py`, `same_chamber_scatter.py`, deck `make_same_chamber_deck.py` →
  notes/same-chamber-pairs.html):
  - **Time window.** The < 12 ms cut is not where the pairs went. 87 % of beam captures fall before 12 ms, but 92 %
    of capsule-pointing tracks come after it (delayed neutrons). Late data are as capsule-pure as Geant4.
  - **Vertexing.** Al e⁺e⁻ cannot vertex: p·β·θ = 0.40 MeV·rad, a 33 mm miss, and only 2.3 % pass with perfect
    tracks. X17 legs miss by 9–11 mm.
  - **Expectation.** The expected count already includes the opening angle and the < 20 mm loss.
- **The y mirror.** `build_tracks.IN_PLANE_SIGN_Y` flips the y position; it should flip tan y.
  - Proven by cosmic-muon timing (run_149, p = 4e-11), 8/8 plastic PMT timing, liquid B, the capsule nose and the
    wiring.
  - `COORDINATES.md` (new) holds the frame and the rule. CLAUDE.md has a warning, build_tracks a warning comment.
  - Slide note: notes/y-orientation.html (`ntof_cosmics/make_y_orientation_deck.py`); checks in
    `ntof_cosmics/y_orientation_check.py`.
- The capsule pose fit was done by another session (315a50d, `HANDOFF_CAPSULE_POSE_FIT.md` Outcome):
  - one compact capture source on the nose, at frame x −10.3, z −3.6, at every ToF;
  - revised 10-13 night: that is NOT established as the beam position. Either the beam/frame is ~10 mm off the
    alignment, or the capsule is far off a centred beam and lower than drawn (³He yield 1–15 %); needs a survey;
  - neither reading explains 1 : 8 alone.

**Next steps:**
1. The y fix is its own section above: done 10-14.
2. The 1 : 8 pair rate:
   - re-digitise the Geant4 pairs with is2_v2;
   - Geant4 with delayed thermal neutrons on the capsule;
   - check for a busy-event two-track loss.
3. Talk on Thursday: present the pair fraction, Geant4's breakdown, chamber A's opening angle, and the 1 : 8 as open.
   No vertex excess. Heights are quotable again (fixed 10-14).

**Gotchas / decisions:**
- Read stage 3 through `build_tracks.y_up(df, path)` (flag-aware); `capsule_pose`'s own mirror is gone.
- Pair work: t0 ≤ 300 and ToF ≥ 12 ms. Early-time mixing made the 10-12 artefact.
- In-beam A–C straight lines are not muons (no timing direction). Use beam-off run_149 for anything that needs muons.
- The wall mean-of-ends time is position-independent. Wall-group cable offsets contaminate u slopes, so fit with
  per-group offsets.

**Key files & commands:**
- `python -m sept26_prelim_analysis.capsule_tof [beam|g4|tof|tilt]` → `<out>/capsule_tof/`
- `python -m sept26_prelim_analysis.same_chamber_scatter` → `<out>/same_chamber/scatter_*.csv`
- `python ntof_cosmics/y_orientation_check.py cosmic|pmt` → `ntof_cosmics/results/y_orientation/`
- Decks: `python -m sept26_prelim_analysis.make_same_chamber_deck`, `python ntof_cosmics/make_y_orientation_deck.py`,
  then `add-note.py <html> --slug same-chamber-pairs|y-orientation --force --deploy`

## Large q_sum tracks (NNLS runaway) — updated 2026-10-02 (dylan-MS-7C84)

**Goal:** explain the ~25–30 % of gated tracks with q_sum > 1e6 ADC (STATUS
2026-09-10), and find out whether they corrupt geometry as well as charge.

**Done:**
- Mechanism found. The NNLS depth-charge profile in `wft/model.py:chi2_plane`
  is unregularised. Depth bins whose column is ~0 inside the data window get
  unbounded q. Two routes:
  - time-censored (71 %): window 20×60 ns, last sample 1140 ns, while the grid
    spans 1080 ns after t0;
  - space-censored (29 %): the slope walks deep bins off the strip window.
- Campaign census, 14.26 M gated tracks:
  - 30.3 % have a runaway plane;
  - 17.0 % are in the late class (t0 > 300 ns), whose geometry is wrong;
  - 9.8 % of scint-coincident tracks are in the late class.
- Refit, run_145 tag 000, all four arms, 3 181 fits: production against a
  "guarded" fit that drops columns below 1 % of the largest.
  - Time-censored runaways: median |Δtan| 0.20, t0 moved in 82 %, flat fits 37 → 8 %.
  - Late normal fits move too (47 % beyond 0.05), so the class boundary is t0, not q_sum.
- Written up in STATUS.md (2026-10-02 entry) and as OCTOBER_2026.md item O10.
- Report: `<out>/qsum_runaway/report.html`. Slide note:
  https://dylan-neff.web.cern.ch/notes/qsum-runaway.html

**In progress / where it stopped:** nothing half-done. Follow-ups were
proposed, but Dylan has not picked any yet.

**Next steps (proposed, in order):**
1. Arbitrate production against the guarded geometry on late arm-A tracks, using
   scintillator pointing (`det_a_scint` / `scint_stack` extrapolation). Measure
   which fit points at the wall group or plastic bar that fired.
2. Quantify how many late-class tracks feed the pair and opening-angle tables,
   and whether a t0 cut changes the spectra or the perpendicular excess.
3. Prototype the real fix for O10:
   - only offer depth bins whose pulse peaks inside the window, or add a depth-continuity prior;
   - require early fits to stay unchanged and late fits to show no χ² loss;
   - store `q_obs`.
4. Work out what the late tracks physically are (out-of-time particles?), and
   what fraction of the space class is coherent ringing.

**Gotchas / decisions:**
- The guarded fit is a probe, not a fix. It is worse on χ² for late tracks.
  It is a monkeypatch of `wft.model.chi2_plane` inside `qsum_runaway` workers only.
- The run_145 data window is 20 samples, not 32. `wm.NSAMP` defaults to 32 in
  a fresh process and is only set inside `fit_plane`. Diagnostics outside a fit
  must use the window's own `W.shape[1]`. This cost one wrong classification here.
- An event can have several candidate windows per plane. Always key examples on
  (arm, event_id, plane, win).
- Don't classify by summing q across fits: a few 1e22 values swamp everything.
  Classify per fit by its dominant bin.
- A χ² explained-fraction does not separate noise windows from tracks, because
  window noise dominates every class.

**Key files & commands:**
- `sept26_prelim_analysis/qsum_runaway.py`:
  - `census` (~2 min; reads stage3_fullpass/tracks_campaign.parquet);
  - `refit --n 300 --jobs 14` (~12 min on 16 cores; writes `<out>/qsum_runaway/`).
- `python -m sept26_prelim_analysis.make_qsum_runaway_figures`, then
  `python -m sept26_prelim_analysis.make_qsum_runaway_report`.
- `python -m sept26_prelim_analysis.make_qsum_runaway_deck`, then
  `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out>/qsum_runaway/deck/qsum-runaway.html --slug qsum-runaway --force --deploy`.
- `<out>` = `/media/dylan/data/x17/sept26_prelim` (`python -m sept26_prelim_analysis.paths`).

## Beam-off cosmics: angle response + in-situ reco — updated 2026-10-12 (dylan-MS-7C84)

**Resume:** muons say is2 scale 1.00, capsule focus 1.12 (A and C). Next: measure in-beam muons directly in is2_v2.

**Goal:** a reconstruction that measures angles correctly at n_TOF, head-on included, so same-chamber pairs can be
reconstructed.

**Read first:** `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §22 (latest), then §20 (is2_v2 full pass vs production,
report `results/fullpass_is2_v2/report.html`) and §21 (capsule focus, 0.8 cut rejected). Earlier: §13–§19.

**Done:**
- 10-10/11 (another session, no wrap-up at the time): is2_v2 condor 4410513 finished (6,464/6,466; the missing
  run_104 stat090_0016 A/C have empty input, and production has no A/C there). Stage 3 is in
  `/media/dylan/data/x17/sept26_prelim/stage3_is2_v2` (stores the 0.8 superset as `gated`; the analysis cut is 0.6).
  Wall-confirmed vs production: A ×1.49, C ×1.82. Merged to main (f0c80cd).
- 10-12 (§22):
  - **Geometry ruled out for the capsule scale.** Distance errors cancel or push A and C oppositely. The source sits
    at global x −9.5 mm (joint (s,dx) fit = imaging −9.3), but with it fitted s = 1.12 on both arms.
  - **Depth origin ruled out:** median charge depth is at mid-gap. Note: a p0/t0 slide does NOT cancel against the
    A–C truth.
  - **Population ruled out:** s 1.08–1.14 across time window, capsule height, wall, χ², charge and |tan y| splits.
  - **Physical RC kernel (cloud_basics) on the run_149 A–C truth:** A y resolution −7 %, x S-shape a third flatter.
    C gets no gain (det7 constants make y +10 %). Raw scale moves < 2 %.
  - The opt-in RC kernel files (`wft/model.py`, `wft/calib.py` + tests) were brought from `mx17-paper-status` into
    main.
- 10-12 (later): slide note updated with §19–22: https://dylan-neff.web.cern.ch/notes/ac-insitu-angles.html
  (`ntof_cosmics/make_insitu_deck.py`, slides 13–21: is2_v2 pass, focus, steep tracks, scale by population,
  the muon-vs-capsule tension, RC kernel, problems, next steps). c549bcf on main.

**Next steps:**
0. (Cheapest, proposed to Dylan, not started) Measure the in-beam A–C muons' scale directly in the is2_v2
   stage-2/3 output instead of inferring it through production (the pooled norm in
   `/media/dylan/data/x17/ntof_cosmics/inbeam_through_goers/pooled_norm.csv` was measured with the production bundle).
   Muons at 1.00 ⇒ the 12 % is real; ~1.1 ⇒ it was the inference.
1. Run `focus_offset` logic on the g4_digi sim chain (`g4_*_is2_full`) with the data's dx. Then vary the simulated
   source (capsule walls, material behind the axis) to see what turns muon-correct angles into s 1.12. Or find an
   electron-specific angle truth.
2. Decide whether stage 3 `pointing()` should use the measured axis (x ≈ −9.5 mm) instead of 0. Today
   `dca_axis_mm` / `in_bore` are off-centre by about a capsule radius. Check who cuts on them before changing it.
3. Physical kernel at n_TOF, if pursued: n_TOF-gas Dp (Magboltz 90/10 at 243/278 V/cm), det6's own template for C,
   §28 joint Dd, then a refit. Only A's y and head-on resolution is likely to gain.
4. Still open: D in-situ bundle (v ≈ 28); B's HV in run_149; adopting is2_v2 as production (Dylan).

**Gotchas / decisions:**
- `capsule_focus.py` assumes the axis at x = 0. Its tan-sign halves are dominated by the −9.5 mm offset; use
  `focus_offset.py`. Exclude 1–10 ms (flash junk, miss 58–82 mm).
- The physical kernel was compared at each label's own TRAIN scale (`insitu_paired.py`); raw scale shifts are in the
  `cmp` v_scale column.
- Bench chamber map at n_TOF: A = det3, B = det2, C = det6 (runs det7's r06 kernel in is2_C), D = det7.
- Raw-tan constants (TAN_MAX etc.) change meaning with v. Never use the capsule band as the beam scale.

**Key files & commands:**
- `ntof_cosmics/focus_split.py`, `focus_offset.py` → `ntof_cosmics/results/focus_split/` (CSVs + logs).
- `ntof_cosmics/insitu_paired.py`, `~/scratch/ntof_insitu/recipe_rc.sh` (copy in `results/insitu_rc/`, with the
  paired CSVs). Bundles `~/scratch/ntof_insitu/bundles/rcm{3_A,7_C,6_C}`, reco tables `reco_rcm*ts_s3_*.parquet`.
- `python -m ntof_cosmics.focus_offset --src /media/dylan/data/x17/sept26_prelim/stage3_is2_v2` (~15 min, 1 core).

## Scintillator stack mapped by the MM tracks — updated 2026-10-06 (dylan-MS-7C84)

**Resume:** scint-stack maps done (whole-face, PMTs drawn); next: why vertical liquids don't answer near their PMT, arm D's extra unbiased tracks.

**Goal:** characterise every scintillator (efficiency and response heat maps), on the single-track imaging calibration.

**Done:**
- Code in `ntof_scint_stack/`; data `/media/dylan/data/x17/scint_stack/` (`paths.spell('scint')`). Calibration: per-run imaging `k` (tanx = k·tan_raw), extrapolation `u + L(α a + λ k tan) − δ` fitted on the wall group edges; α ≈ 0, λ 0.59–0.76. README has it.
- **Statistics (this session):** coverage was always the whole full pass (34 runs, 13.3 M gated tracks; run_79/81 excluded). The thin sample was the *unbiased* tag (another arm triggered): ~99 % of triggers fire one arm. Main maps now `*_full` layers on `all_late` (every late trigger, whole face, boundary-tolerant `_tol` probes, 25 mm): 134k–934k tracks/arm vs 2–9k unbiased. Unbiased shape is its own slide (100 mm).
- `checks.py`: trigger thresholds MEASURED (0.5 % low edge of each arm's own triggers, flat to ~1 mV campaign-wide; now in `extract.WALL_THR/PLAS_THR`; run_79 read-back had plastic 3–6 mV high). Time-cut scan: LATE_MS stays 10 ms (earlier = <15 % more stats, efficiency not converged).
- `ana` writes `funnel` (cut flow per arm); deck has a samples slide.
- **PMTs drawn on the maps.** Liquids from Geant4 (`MX17_Full_Geant/include/SimConfig.hh` `ls_rot_deg`, 17–18 July survey): A, D horizontal with PMT at +u; B, C vertical, PMT up. Dylan recalled B right / D top; Geant AND the data (A/D answer 12–23× more on the +u half, B/C no u gradient) disagree — told Dylan; switch is `make_deck.LIQ_PMT`. Plastic PMTs on top (Dylan's report; not in Geant, not testable).
- Slide note republished: https://dylan-neff.web.cern.ch/notes/scint-stack.html; long report regenerated.

**Next steps:**
1. ~~Why the vertical liquids (B, C) show no rise toward their PMT (C: 0.07 % top vs 0.34 % bottom) while A, D do.~~ **Answered 2026-10-13: the stage-3 y mirror.** In true y C is 0.34 % top vs 0.07 % bottom, a rise toward the PMT like A and D (`Y_FLIP_AUDIT.md` #3; confirm on the regenerated scint stack).
2. Arm D: a third of its late tracks sit on other-arm triggers (A–C 4–18 %) — real particles or chamber pick-up? D's unbiased maps rely on them.
3. Join run_149 cosmic tracks to the slim → MIP-clean efficiencies.
4. Plastic bar placement / D wall group-0/2 cabling against the as-built drawings.

**Gotchas / decisions:**
- Self-triggered maps read near 1 by construction (trigger = wall-sum AND plastic); they show holes, not levels. Quote unbiased numbers.
- Dead end: loosening LATE_MS below 10 ms (`checks.late_scan`): not worth it.

**Key commands:** extract → ana → checks → make_figures → make_report → make_deck → add-note (see `ntof_scint_stack/README.md`).

## Calorimetry feasibility + MM dE/dx — updated 2026-10-14 (dylan-MS-7C84)

**Resume:** scintillator proof of concept done 2026-10-14 (wall + plastic, muon-calibrated, vs beam e⁻ and Geant4): https://dylan-neff.web.cern.ch/notes/scint-calorimetry.html (`ntof_calorimetry/make_wp_deck.py`). Earlier feasibility note: https://dylan-neff.web.cern.ch/notes/calorimetry.html (`make_deck.py`). Open: C1 +15 % electron offset, PMT bases for LNL.

**Wall + plastic result (2026-10-14, `wall_plastic.py`, record in `PLAN.md` "Reopened"):** data ÷ Geant4 at ≥ 40 ms: plastic A1/A2/C2 1.05/1.03/1.02, **C1 1.15** (pure gain offset, cause unknown), wall A/C 0.95/0.96. Plastic is linear 3.4→8 MeV and is 20 mm thick; the wall measures dE/dx (~0.5 MeV flat), a 1-vs-2-particle tag. Plastic PMTs lose 3–8 % gain 12–25 ms after the flash (LNL: rate-proof bases or a pulser). Run-to-run stability 0.5–0.8 %. Geant4 side: `MX17_Full_Geant/scripts/stack_edep.py` on branch `lnl` (737e9a2, deliberately not merged to its main). Rebuild: `python -m ntof_calorimetry.wall_plastic`, then `python -m ntof_calorimetry.make_wp_deck`, then `add-note.py … --slug scint-calorimetry --force --deploy`. Outputs `/media/dylan/data/x17/calorimetry/wp/` (disk ~3.6 GB free).

**Goal:** find what energy information the n_TOF data can give, and test MM dE/dx as a concept.

**Done (all in `ntof_calorimetry/`, report `/media/dylan/data/x17/calorimetry/report.html`):**
- C1 plastic: the srccal keVee line reads the cosmic MIP at 0.76–0.97 of Bichsel 3.41 MeV (Geant4
  agrees: 3.375). New MIP-anchored scale `c1/calib_plastic_e.json` (4–7 %). Trigger threshold
  2.1–2.9 MeVee. Linear to 1.45 MIP.
- C4 liquids: MIP = 20–28 mV against a 16–18 mV n_TOF ZS threshold, so the liquids are threshold-limited.
  Their source scale is ×5–7 wrong. Usable: A u≥0 at 65–86 %, D u≥50 at 50–70 %.
- In-beam A–C through-goers at ≥10 ms reach the liquid at 0.10 of the cosmic expectation, so they are
  mostly not penetrating.
- M1/M2: raw road charge (`mm_charge.py`). FWHM/MPV ~1, and a 2-MIP flag reaches only 40 % at 10 %
  mistag, so this line is KILLED.
- C2 (condor 4409644): 1.2 MeV is lost before the plastic, and half of 4 MeV electrons reach it.
- C3 (condor 4409645, 10^7 pair sim, IPC reweighted to ipc_born): plastic energy gives a Z² gain over
  θ_open of 1.05–1.08, so KILLED. The true soft-leg T would give ×2.5–4: a design lesson.

**In progress / where it stopped:** nothing half-done. All condor jobs finished; outputs are on EOS
`full_sim/calorimetry/` and pulled to `<calo>/c2`, `<calo>/c3`.

**Next steps (optional):**
1. Measure a plastic bar (20 vs 25 mm). 25 mm would move the MIP scale by 21 %.
2. C1.4 gain vs time since flash: needs an in-beam energy reference (H-capture Compton edge?).
3. Tell the through-going-background owner that beam through-goers are not penetrating.

**Gotchas / decisions:**
- Beam through-goers are not a MIP sample. Use run_149 cosmics (clock-matched) for scintillator MIPs.
- Plastic MIP fits need per-event truncation at the trigger threshold (A's threshold is at 0.85–0.9 of
  the peak) and a σ/MPV prior.
- C3 metric: fine 2D bins on the IPC MC fake gains. Use coarse bins, pool bins with <10 MC events, and
  check the split halves.
- An apparent attachment fall in C was a selection artefact (per-track normalisation on in-window
  tracks).
- `pkill -f <pattern>` and `pgrep` loops match their own shell command line: don't.

**Key files & commands:**
- `ntof_calorimetry/README.md` — run order; `PLAN.md` — the plan plus "Results so far"
- `ntof_calorimetry/condor/` — C2/C3 submitters and reducer (deployed at
  `/afs/cern.ch/user/d/dneff/condor/calorimetry/scripts`)
- `PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.make_report`

## Detector model basics (cloud_basics) — updated 2026-10-12 (dylan-MS-7C84)

**Resume (pm):** see the "X width excess (UNSOLVED)" section above. It is the open blocker for freezing the kernel.

**Resume (am):** det4 pinned-D_rc arm DONE (FINDINGS §28): beam 2.2e-4 costs nothing on the bench (stack rms +0.3 %,
held-out Y +0.01–0.03 ± 0.02–0.03°); §27's 1.5–2.7× was rc_drift's assumed D_T — the joint X+Y fit pins Dd (X has no
RC) at ≈ dry on det2/3/4/7. Bench drift fields were wrong in bench_pinned (all 250 V/cm; really 347/208/243) —
`bench_water_e.py`; still 0.4–0.6 % water from v vs dry widths. Tight |tan|<0.01 brings bench σ0 to the beam 0.37.
Next: decide the bench D_T source (Magboltz water D_T, or a non-diffusion X growth term); then freeze the kernel
(det4 D_rc from beam, others bench; Dd from joint fit; σ0 from tight-|tan| stack) and run the full gate.
Older queue: det4 X vs amplification stripes, det7 bench plateau.

**10-12 (dylan-MS-7C84):** `bench_pinrc.py` (joint fit with pins + `--profile`), `bench_tanscan.py`, `bench_water_e.py`,
`compare_bench.py --base`. bench_attach's bundle path now resolves next to each plane_ratio cache (old one was session
scratch). Bench outputs `/home/dylan/x17/cosmic_bench/cloud_basics/bench_pin/` (arms in `arms/arm_det4_rc{f,p,pw,pd}.json`).

**10-11 (dylan-MS-7C84):** `attach_transfer.py` injection toy (SPS measurement code on known events, run_71 loss on/off,
paired): footprint/D_rc/c2-c1 shift ≤ 7 %, decompose b1 −27 %. Toy realistic at 243 V/cm only. `rc_drift.py` (RC fit +
Magboltz drift diffusion): beam det4 Y D_rc 2.08/2.08/1.94e-4 over 3 fields → 2.2e-4 ±10 % loss-corrected. Bench D_rc moves
by a third with the assumed D_T (σ0 with it): degenerate; det4 bench/beam 1.5–2.7× unexplained. `width_tan_split.py` inconclusive.

**10-10 pm (dylan-Yoga):** slide-note version published, notes/mx17-attachment-slides (24 slides, built by
`cloud_basics/make_attachment_deck.py`, commit 8fd794a): observable + four-mechanism fingerprints first, then a beam/bench
consistency section (shared-vs-differing setup table, bench drift scan, model-free overlay, loss rate vs field, ledger).
Honest caveat found while building it: the bench's model-free late/early ratio moves −4..+2 % because its own no-air model
falls 3–5 % (response tail + diffusion), so the bench O2 limit rests on the composition fits, not that ratio. The long-form
report notes/mx17-beam-attachment is unchanged. Slide cross-refs are `§§<id>§§` tokens resolved at build.

**10-10 am (dylan-Yoga):** DONE: gas compositions as model curves (FINDINGS §24-26, report F9-F13, notes/mx17-beam-attachment).
Beam run_71 1.51 % water + 0.078 % air (163 ppm O2), all 3 fields x 2 views fit on the high-stat grid (eta smoothed in E);
run_63 blocks 230 -> 147 ppm O2 over the night; CO2 period ~290 ppm; bench det3 6-field drift scan 0.95 % water, no air (<~10 ppm).
One physics model, setup-specific effective shapers. X: snapping rejected, footprint tails real but resolution-neutral (§23).
Next: det4 X deficit vs amplification stripes; det7 bench rising plateau; three-body O2 attachment scale (ppm is model-dependent).
Data note: re-extracted run_71 cache lives only in the laptop job tmp (regenerate: extract_det4_only.py run71_raw --cm masked --keep 12);
det3 drift-scan decoded_root pulled to ~/x17/cosmic_bench/det3/mx17_det3_saturday_scan_6-27-26/.

**Goal:** Dylan (10-09): get a physically sound, consistent detector model *before* publishing or re-passing; solve as many
consistency problems as possible and converge on one model over the coming weeks. Re-pass is ON HOLD until then.

**Done (narrative + numbers in `sps_beam_test_26/analysis/cloud_basics/FINDINGS.md` §1–12):**
- Prompt footprint σ0 ≈ 0.4 mm is readout, not gas (avalanche ~25 µm); same on bench and beam (leading edge, unbiased stacks).
- Bench drift diffusion = dry Magboltz on 4/5 chambers (det6 low). Moisture barely changes D_T.
- Y spreads as a 1-D RC line (σ² += 2·D_rc·t, D_rc 2–9e-4 mm²/ns); X does not. Y template carries the RC undershoot → use X template for both.
- NO bench attachment — my earlier claim was a max-normalised stacking artefact (retracted; use unbiased fixed-threshold stacks).
- Beam run_71: late charge decline is common to X and Y (±4 sums) and field-dependent (−5/−9/−20 % by 600 ns at 243/150/92 V/cm) → gas attachment in wet beam gas, not electronics.
- `wft.model.build_matrix_rc` (opt-in via `rc_D_y` key; production untouched; tests `wft/tests/test_rc_kernel.py`).
- Held-out bench, no refit: Y better on all 5 chambers (det2 −0.15/−0.36, det3 −0.09/−0.14, det4 −0.15/−0.20,
  det6 −0.55/−0.90, det7 −0.12/−0.28 °, all/head-on). X mixed: det6/det7 better, det2/3 neutral, det4 +0.16 worse; X χ² worse everywhere.

**In progress (lxplus condor, all 21 jobs were still IDLE at wrap-up — queue is clogged by my own 6164 is2_v2 jobs, cluster 4410513):**
- 4410533: Magboltz attachment (beam_w1p7 ± 0.05/0.1/0.2 % O2, bench_w0p5 ± O2) → `lxplus:~/cloud_basics_condor/att/magboltz_*.json`.
- 4410535: RC refits, arms rcf3 (σp0_x, σp0_y, rc_D_y), rcf4x (+rc_D_x), rcfD (+Dp) × 5 chambers → `lxplus:~/cloud_basics_condor/rc/out/arm_<det>_<arm>.json`.

**Next steps:**
1. Copy att JSONs to `cloud_basics/results/`; does 1.7 % H2O + some O2 reproduce size AND field dependence of the beam loss?
2. Bench the refit arms (`plane_bench.py`, then `compare_bench.py --start blind`) vs production and rcm. Does X want its own σ0 / an RC term?
3. X deficit hypothesis: charge on a 550 µm resistive strip snaps X to resistive-strip centres (0.80 vs 0.78 mm pitch) — a Gaussian footprint can't do that. Model it if refits don't close X.
4. Template-free beam charge-vs-depth on 25.6° runs (run_63/run_62 ZS; re-pull from EOS, mind ZS censoring).
5. Untested: leading spike = primary ionisation inside the amplification gap.
6. Then full gate (gate.sh) for the physical kernel; only then revisit the re-pass. Also make_report.py/report.html for cloud_basics.

**Gotchas / decisions:**
- Judge kernels by scale-corrected σ68 (divide each arm's tan by its own slope) with paired bootstrap; rc arms read angles 3–8 % flat.
- Stack unbiased (`CB_STACK` default): fixed threshold, no per-event normalisation.
- Bench template does not close a forward fit on beam; quote beam σ0 from the leading edge only.
- Magboltz on condor (desktop ROOT is py3.10 vs system 3.12); `v_um_ns` in magboltz JSON is ×10 µm/ns — use `v_true_um_ns`.
- plane_bench float-casts hypers: `rc_tmpl` is numeric (1.0 = X template).
- Housekeeping: offload lxplus `~/plane_ratio_condor/bench` (134 MB) with `lxstore offload`.

**Key files:** `sps_beam_test_26/analysis/cloud_basics/` (FINDINGS.md, make_rc_arms.py, compare_bench.py, beam_xy_pulse.py, …);
outputs `/home/dylan/x17/cosmic_bench/cloud_basics/` (arms/, bench/); RC payload in lxplus `~/cloud_basics_condor/rc/`.

## Per-view sharing kernel (PAPER_PLAN C2) — updated 2026-10-09 05:00 (overnight, dylan-MS-7C84)

**Resume:** ON HOLD (Dylan 10-09) — likely superseded by the physical RC kernel (see Detector model basics). Per-view Y ratio validated (gate: Y −0.04…−0.08°, head-on −0.10…−0.13°, det2/3/7); re-pass staged in `condor_campaign_pvy95/`, NOT submitted — needs Dylan's go.

**Done:**
- C1: r06 reproduces FLEET_DIGEST exactly (waveform-fit cells, five keys).
- `wft` gained opt-in per-view keys (`c2_over_c1_<view>`, `sigma_p0_<view>`, `c1_asym_<view>`) and `WFT_MODEL_FRAC`; absent = bit-identical (41 tests).
- Study tools + log in `sps_beam_test_26/analysis/plane_ratio/` (FINDINGS.md is the narrative); report `/home/dylan/x17/cosmic_bench/plane_ratio/report.html` (make_report.py).
- Candidate bundles `calib_bundle_pvy95` (+ controls `calib_bundle_prodt0`) in each golden key's wft dir; gate products `events_/alignment_/angles_/efficiency_{pvy95,prodt0,prodref}`.

**Decisions for Dylan:**
1. Submit the re-pass (`/home/dylan/x17/cosmic_bench/condor_campaign_pvy95/README_SUBMIT.md`). Mind the promote trap written there.
2. c2 < c1 gate: applied to the hyper it binds on Y (det2/3/7 improve further to y = 1.6). Hyper or observable?
3. det4/det6: their candidates help Y only against w0/kw-refreshed controls; adopting them = also the R06_GATE §4 w0/kw refresh.

**Gotchas:**
- Refitting the per-view ratio in the bench calibration does NOT work (det3 got worse); pin it.
- `wft.calibrate._event_chi2` is path-dependent (warm t0); recal.py uses a deterministic cold objective. The local calib cache is not r06's training cache (its 1.10e8 chi2 is not reproducible here).
- condor: LCG_105 numpy cannot unpickle numpy-2 caches — use LCG_108. `TAG=$(test && echo)` under set -e kills scripts.
- Condor working dir lxplus `~/plane_ratio_condor` (small JSON only), inputs on `/eos/user/d/dneff/plane_ratio/`.

## MX17 papers (two-paper plan) — updated 2026-10-19 (dylan-MS-7C84)

**Resume:** LaTeX outlines of Papers I+II are in `papers/`; next: Dylan sends 2 Overleaf git URLs, run `overleaf.sh link`.

**Goal:** get two companion papers written: I, the MX17 chambers and their performance (June cosmic bench + det4 in SPS H4); II, the waveform forward-model reconstruction and how much a hit-based readout loses. Stephan edits in Overleaf with tracked changes (production/design input for I).

**Done:**
- 10-08: plan as tables in `mx_june_cosmic_qa/PAPER_PLAN.md` (outlines, work list C/R/D/V, n_TOF tripwires); slide note https://dylan-neff.web.cern.ch/notes/mx17-detector-paper-status.html via `make_paper_plan_deck.py`. C1 (reproduce r06) was done 10-09.
- 10-19 (commit e6448d5, on main): `papers/paper1/`, `papers/paper2/` — LaTeX outlines, one file per section with a state box, intent, placeholder text (`\tbw`), `\askStephan`/`\askDylan` boxes, `\analysis{id}` margin notes, `\requote` numbers. Paper I has a new §2 "Design and construction" (incl. production & QA, Stephan's) seeded with the 17 spec-page questions. Both compile (12 / 8 pages, pdflatex+bibtex).
- `papers/make_figures.py` + `figures.toml` per paper: figures/<id> is a copy (current), a faded "SUPERSEDED" stamp (hits-based: edge turn-on, timing) or a grey placeholder. 11 real figures, rest placeholders.
- `papers/overleaf.sh link|pull|push|sync <paper>` (git subtree). Tested both ways on a local stand-in remote: link, Stephan-edit pull, local push, refusal when Overleaf is ahead. NOT yet run against real Overleaf.
- Workflow documented in `papers/README.md`; pointers in CLAUDE.md layout and PAPER_PLAN.md.

**In progress / where it stopped:**
- Waiting on Dylan to create the two blank Overleaf projects and store a git token (Dylan has Overleaf premium).

**Next steps:**
1. Dylan: create blank projects "MX17 Paper I"/"MX17 Paper II", copy git URLs (Menu → Integrations → Git), make a token (Account Settings → Git integration), store it: `git config --global credential.https://git.overleaf.com.helper store` then one `git ls-remote <url>` (user `git`, password token).
2. `papers/overleaf.sh link paper1 <url>` and `... paper2 <url>`; commit + push origin main (the link adds a merge commit). Then Dylan shares with Stephan as Editor, turns Track changes on, sets main.tex / pdfLaTeX.
3. Dylan: send Stephan the spec page link and the Overleaf link (spec-page section of this file).
4. Fill placeholders as figures appear: edit `figures.toml` (source + state = "current"), `make_figures.py`, commit, `overleaf.sh push`. Candidates needing data not on the desktop: sps_resolution (residuals.npz), kernel_forms (stacks_run71_raw.npz), sps_setup (mpgd26 inputs).
5. Open decisions still blocking text: C2 (freeze r06 vs per-plane c2/c1, pvy95 re-pass staged), C4 (journal, author list; then switch document class from plain article).
6. Analysis items as in PAPER_PLAN: D1 requote, D2, D3, D4, R1–R7, R9, R10.

**Gotchas / decisions:**
- Overleaf git carries plain text, not tracked changes/comments: accept/reject Stephan's changes in Overleaf before pulling; never push repo edits into a section with pending tracked changes. Figures are always safe to push.
- overleaf.sh `link` joins histories with a commit carrying git-subtree trailers (a plain `git subtree merge` refuses unrelated histories — tried, failed). Never forces.
- Paper figures are tracked despite the global `*.png` ignore (`!papers/*/figures/*.png` in .gitignore).
- Unconfirmed in the text: gas mixtures per setup (bench Ar/iC4H10 95/5? SPS Ar/CF4/iC4H10 88/10/2?) — left as `\askDylan`. refs.bib has 5 entries from memory; replace with INSPIRE BibTeX. D_RC value in II §2 has no unit.
- Section state boxes duplicate PAPER_PLAN states: change both.
- Unrelated: uncommitted cloud_basics changes in the tree belong to another session; not touched.

**Key files & commands:**
- `papers/README.md` — the workflow
- `.venv/bin/python papers/make_figures.py all [--list|--run|--only id]`
- `cd papers/paper1 && latexmk -pdf main.tex` (or pdflatex/bibtex)
- `papers/overleaf.sh sync paper1`
- `mx_june_cosmic_qa/PAPER_PLAN.md` — states and work list; `make_paper_plan_deck.py` + add-note.py to republish the status note

## Cosmic-bench efficiency + gain maps (A–E) — updated 2026-10-09 (dylan-MS-7C84)

**Resume:** both PDFs built and accepted "good enough"; nothing pending — rerun scripts only if a colleague wants a different kernel.

**Goal:** per-chamber efficiency and relative-gain maps of the June cosmic bench, one page each, in the MPGD26 slide-27 format, for a colleague.

**Done:**
- `mx_june_wft/report/make_bench_effmaps.py` → `~/x17/cosmic_bench/Analysis/bench_efficiency_maps/cosmic_bench_efficiency_maps_s4mm.pdf`.
  Gaussian σ = 4 mm (my reading of "4 mm kernel"; hard r = 4 mm not tried), 0.5 mm step, viridis 0–100 %, within 5 mm of M3 ref.
  Detector-local frame (x horizontal, y vertical; inverse of alignment θ ≈ 90°), fixed axes −20…420 mm, fixed page size, method text in footer.
- `mx_june_wft/report/make_bench_gainmaps.py` → `…/bench_efficiency_maps/gain/cosmic_bench_gain_maps_s8mm.pdf`.
  Per-muon Q = √(x_q_sum·y_q_sum)/√(1+tx²+ty²) from wft events; Gaussian mean of ln Q, σ = 8 mm, /chamber median, log colour 0.3–3.
- Datasets: A det3 `g_det3_wknd` (21 953 rays, 93.1 %), B det2 `g_det2` = 6-22 `long_run` (19 045, 91.3 %), C det6 `g_det6_long` (9 628, 74.9 %), D det7 `g_det7_long` (9 428, 57.1 %), E det4 `g_det4` (12 259, 41.6 %).
- Options not used: det3 `sat_det3` (7.1 k rays, 93.3 %); det2 `o22_long_det2` = `longer_run` (3.7 k, 92.0 %, ~3 muons/kernel).

**Next steps (only if asked):**
1. Hard-disc r = 4 mm variant if that is what "4 mm kernel" meant; larger σ for det6/7/4.
2. Investigate the ×2.5 gain rise toward the edges (seen in both wft Q and summed hit integral; probably edge fringe field, cf. script 32).

**Gotchas / decisions:**
- Data-dir changes (outside repo): 6-22 `long_run` (det2_det3) now has `m3_tracking_root_v2` + EOS reprocessed hits as `combined_hits_root`; old hits kept as `combined_hits_root_june_orig`. Re-ran `01_alignment`/`02_efficiency` for `g_det3_wknd` (its 08-12 outputs were stale vs the 08-21 r06 reco), `g_det2`, and `o23_long_det4` (alignment only; that key is a tiny `longer_run`, efficiency fails — hits lack `significance`; irrelevant).
- Peak-strip amplitude is NOT a usable gain estimator for det3/det2 at operating HV (~40 % of events saturated); raw waveforms for g_det3_wknd/g_det2 are not local.
- Gain is measured on detected muons only → survivor-biased where efficiency collapses (det4, det7). Gain is relative per chamber, not comparable across chambers.
- Per-ray CSV caches in the output dir (`<key>_rays.csv`, `<key>_box.json`) — delete to force recompute.

**Key files & commands:**
- `.venv/bin/python mx_june_wft/report/make_bench_effmaps.py [--sigma 4]`
- `.venv/bin/python mx_june_wft/report/make_bench_gainmaps.py [--sigma 8]`

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
