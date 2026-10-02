# HANDOFF — track the beam-off cosmics (run_149 first)

**Written 2026-10-02.** Companions: [`README.md`](README.md) (the entry point:
inventory, clock match, framing), the live note
<https://dylan-neff.web.cern.ch/notes/beam-off-cosmics.html>, and
`sept26_prelim_analysis/OCTOBER_2026.md` item O9.

---

## 0 · Where we are, and the decision

The clock match without beam works on **one sub-run only**: run_149
`cosbounce_cos_0000` against n_TOF 224678 (15 min). 2,467 triggers are matched
within ±50 ns, which is 92.2 % of the in-window triggers (beam-on gives 96 %),
with a 10.6 ns core MAD and accidentals too small to measure. See
README §"Clock match without beam".

The README's next list has three threads: match the rest, track the cosmic
runs, and the three handles. **Dylan's decision (2026-10-02): start the
tracking now, and scale the clock match up alongside it.** The reasoning:

- Two of the three handles need **no n_TOF at all**:
  - **collinearity**, i.e. setting the 170° `BACK_TO_BACK_DEG` cut from data;
  - **capsule DCA**, the closest approach of the through-going line to the
    capsule.
  Both can use all 57.6 h of production-point cosmics, not just the ~16 %
  that n_TOF saw. They also tighten the opening-angle background soonest.
- Only the third handle, the **sign of the arm-to-arm Δt**, needs the match.
  It is a 2–3 ns lag, below the ~7 ns per-event Δt resolution, so it only pays
  off on a large matched two-arm sample. That sample does not exist until
  tracking does.

**If you do one thing: §2 (track run_149/cos_0000 in full), then §3.**

---

## 1 · Two places the beam-on pipeline assumes a beam — check before running

Both are from reading the code, not from running it on a cosmic sub-run yet.

**(a) Stage 1 needs the n_TOF slim, and cosmic runs have none.**
`sept26_prelim_analysis/candidate_filter.py` takes its arm flags from the
n_TOF slim (`arm_flags`, `ACCEPT_NS`). The 46 runs in 79–162 with no slim are
exactly the beam-off cosmic and pulser runs (STATUS.md, re-slim table). With
no slim, line ~410 warns and the n_TOF columns come out empty, so its
classification is not meaningful here. It is also the wrong tool anyway: the
2026-09-10 full pass exists because stage 1 kept only 12.8 % of the real
two-track events (`HANDOFF_FULLPASS_2026-09-10.md` §1).
→ **Skip stage 1 and run a blind full pass** (`make_stage2_campaign.py
--full-pass`, possibly with `--subset`). The cost: the run_145 pass took
9.3 cpu-h per ~47 k triggers. That makes cos_0000 (22,760 triggers) about
4.5 cpu-h, and all of run_149 (21.8 h × ~25 Hz ≈ 2 M triggers) about
400 cpu-h. Check how the job list and allowlists are built: the script reads
the file tags from the stage-1 candidate table (`tags_from_stage1`), so a
cosmic sub-run may need its tags listed another way.

**(b) The angle scale k cannot be measured from cosmics with `k_arm`.**
`k_arm.py` measures k by assuming the tracks come from the capsule
(tan = (u − foot_x)/d_perp). Through-going cosmics break that assumption on
purpose. So cosmic tracks have to **borrow k** from the nearest beam runs.
That matters because k is not stable:

- the 3–5 August block, runs 128–147, rises on all three calibrated arms at
  once (A +2.3 %, C +7.8 %, D +7.3 %), and both of its edges fall in beam-off
  gaps;
- run_149 sits just after that block.

Borrow k from the nearest beam runs on both sides, and quote the opening
angle under both choices. If the near-180° peak moves when k changes, that
shift is the systematic.

**An opportunity, not a claim yet.** A single straight particle crossing two
arms is one line: the two chambers' reconstructed angles have to agree with
the line through their two crossing points. That gives an angle-scale
constraint with **no capsule assumption**. It could check `k_arm`
independently, and it is a candidate for the deferred "Cosmic-bench
cross-calibration". It also bears on the open question about the 128–147
block: run_133 and run_134 are cosmic runs inside it. Worth one look once
two-arm tracks exist.

---

## 2 · First step: run_149/cos_0000 through reco and into tracks

This is the same sub-run the clock match already covers, so every track
can carry a scintillator time from the start.

1. A full-pass stage-2 reco of `run_149/cosbounce_cos_0000`. Use the same
   bundles and v_drift pin (42.6 µm/ns) as the campaign, so the cosmic tracks
   and the beam-on tracks are like for like.
2. Merge, then `campaign_tracks --fullpass <reco>`, with `--k-from` pointing
   at the neighbouring beam runs (§1b). `campaign_tracks` has `--out`; use a
   separate output directory under `ntof_cosmics/results/` and never write
   into the campaign track table. Note that `k_arm` has **no `--out`** and
   overwrites `kcal/k_arm_<run>.json` silently
   (`fullpass_chain_2026-09-10.sh` step 1), so do not run it on cosmic runs.
3. Join to `results/clock_match/pairs_run_149_cosbounce_cos_0000_224678.csv`
   on the DREAM trigger. That gives each track its n_TOF arm and time where
   one exists.

**The first numbers to report:**

- the fraction of the 22,760 triggers with ≥1 track;
- the fraction with a track in two arms, split by arm pair (A–C and B–D
  oppose; the others are perpendicular);
- the two-arm opening-angle distribution;
- the capsule DCA of the joined line, using pointing estimators, not shape
  fits (CLAUDE.md, "Take positions from pointing").

Before trusting the sub-run, read `hv_monitor.csv` for trips. Positions and
angles come from the waveforms only, never from `combined_hits` times
(CLAUDE.md, reconstruction basis).

---

## 3 · Then, in parallel

**Tracking, scaled up.** Run all of run_149 on condor (~400 cpu-h), then
run_133, run_89, run_103 and run_134. Together these hold 41 of the 57.6 h.
Compare against the beam-on `back_to_back` sample (`tight_coincidence.py`)
on the same axes: opening angle, capsule DCA, and arm pair.

**The clock match, scaled up** (README §Next):

- a smooth drift model, which absorbs the slow δ_b wander and lets stage 4
  shrink from ±300 µs;
- several n_TOF runs per sub-run (cos_0001 straddles 224678/224679), so key
  by (run, bunch);
- recover the psTime = 0 bunches from the 0.5009 s grid;
- show that k and κ transfer between sub-runs;
- then all of run_149 and run_103, about 13 h with n_TOF recording.

**The handles, as their inputs arrive:**

- the 170° cut, set from the cosmic opening-angle distribution;
- the through-goer capsule-DCA shape, which should be roughly flat at the
  geometric acceptance against signal peaked on the capsule;
- the mean arm-to-arm Δt, ⟨t1 − t2⟩, on matched two-arm cosmics, tested for
  a shift.

**Not blocking:**

- the 92 % vs 96 % match-efficiency gap (candidates: the unfitted per-bunch
  rate, and the 418 single-trigger bunches);
- the 7 supercycle mislocks in the old `cache_pulse_match` (prior only).

---

## 4 · Where results go

- The numbers and figures go in this directory's `results/`, plus an HTML
  report (CLAUDE.md, "Reporting results").
- The note grows by adding slide functions to `make_deck.py` and rerunning
  the deck and publish commands in README §Reproduce.
- The X17 board (`x17-board` skill) has no entry for this work yet. Log the
  first tracking numbers there, probably under `background`, and consider a
  question about whether the 170° cut should be set from cosmics.

---

## 5 · Progress — 2026-10-02 evening

**§2 done for run_149/cos_0000.** Report: `results/tracking/report.html`
(`make_tracking_report.py`). Code: `cosmic_tracks.py` (`fetch` / `build` /
`analyse`); `make_stage2_campaign.py` gained `--tags-json` (sub-runs and tags
given directly, for runs with no stage 1).

- **Reco:** condor 4354841, 12 jobs, none held, campaign full-pass config
  unchanged (bundles A/B/D r06, C lp; v pinned 42.6; no hot mask). Output in
  `/eos/user/d/dneff/x17/cosmics_fullpass`, apart from the campaign's. HV clean.
- **Yield is low:** 2,289 of 22,760 triggers (10.1 %) have a gated track, 136
  (0.6 %) in two arms. The gate costs more on cosmics than on capsule tracks
  (A y-plausibility 55 %, D quality ~77 %). Not yet split from scintillator
  coverage outside the chambers.
- **Two-arm pairs are A–C** (64 triggers); B–D is empty (needs a horizontal
  cosmic). Perpendicular pairs: AD 20, CD 18–19, AB 19, BC 8, BD 7.
- **170° cut:** of the ~40 opposing pairs whose lines meet within 20 mm,
  **74–77 % are above 170°**; clean 16 % quantile 162–167°. The 64-pair
  opposing sample including the non-meeting tail: 51–56 %.
- **Capsule-free angle scale (§1b's opportunity) — it bites.** On clean A–C
  through-goers, track slope ÷ joined-line slope (median): **A 1.06–1.11,
  C 0.85–0.96**, the same under both borrowed k. I.e. the beam-derived k reads
  A ~8 % too shallow and C ~12 % too steep. ~40 events; corr 0.99 for A,
  0.7–0.9 for C (C has outliers). Part of the 170° inefficiency is this.
- **Clock join consistent:** 18–20 two-arm triggers have an n_TOF match; in
  94–95 % the matched n_TOF arm is one of the two tracked chambers.
- Pointing preselection (both tracks < 30 mm from the axis): 11 pairs, 8 > 170°.

**Running:** the other 86 sub-runs of run_149, condor cluster **4355060**,
1,028 jobs, same package (scratchpad `pkg_run149`; lxplus
`~/cosmics_stage2_run149`), same EOS dir. When done:

    S=<scratch dir off /media>
    for s in $(ssh lxplus 'ls /eos/user/d/dneff/x17/cosmics_fullpass' | sed -nE 's/^run_149_(cosbounce_cos_[0-9]{4})_beam_.*/\1/p' | sort -u); do
      .venv/bin/python ntof_cosmics/cosmic_tracks.py fetch --subrun $s --tarballs $S/tarballs
      .venv/bin/python -W ignore ntof_cosmics/cosmic_tracks.py build --subrun $s
      .venv/bin/python -W ignore ntof_cosmics/cosmic_tracks.py analyse --subrun $s
    done

then pool the pairs across sub-runs (not written yet — `analyse` and the report
are per sub-run) and redo the three numbers with ~3 500 clean A–C events.

**Next, in order:**
1. Pool run_149; re-measure the slope ratios per arm and axis with real
   errors. If they hold, that is a cosmic k for A and C (and for B/D from the
   perpendicular pairs, with a slope check written for them), and the beam-on
   `k_arm` values for the 128–147 block have an independent check.
2. Redo the 170° efficiency with the cosmic k. Only then propose a cut.
3. Split the track-yield loss: gate vs. geometry.
4. Understand the non-meeting tail (lines miss by cm, open angle 120–160°).

**Disk:** nothing in this work writes to `/media/dylan/data` (`~/x17` is a
symlink to it, and every `paths.out` default resolves there).
`cosmic_tracks.py` refuses any output path under `/media`; products are in
`results/tracking/` (parquet gitignored, ~7 MB per sub-run).
