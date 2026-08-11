# n_TOF reprocessing: current state

**Keep this file current.** It is the resume point if a session drops.

---

## 2026-08-11 evening — n_TOF merged 27 runs; we match them bit for bit

[`FINDINGS_2026-08-11_official_ledger.md`](FINDINGS_2026-08-11_official_ledger.md),
ledger CSV [`campaign_qa/results/ledger_2026-08-11.csv`](campaign_qa/results/ledger_2026-08-11.csv),
tools `campaign_qa/official_ledger.py` + `campaign_qa/compare_identity.py`.

**Bookkeeping, in the form that matters** (`campaign_qa/completed_ledger.py`,
`results/completed_ledger_2026-08-11.csv`). Forget the merge — it carries no
information about usability. The question is whether `completed/<run>/` **covers
the run**, judged from the `index` tree (replicated in full in every partial) plus
the last partial's own hits: **401 of 445 runs are complete from n_TOF's
partials**, 3 more only from their merged file (224526, 224566, 224569 — partials
cleaned up post-merge), **30 only from ours**, 1 only as an off-recipe copy
(224576), and 10 are complete nowhere right now (9 that n_TOF is rewriting, plus
our 224709). A `prod_v11` product counts as `OFF_RECIPE`, **not** as coverage.

**224576 cannot be reprocessed by us — its raw is gone.**
`DAQ/.../X17_measurement/224576/stream1/` is empty *in the EOS namespace*, and X17
raw carries no tape replica (`eos fileinfo` → `d2::t0`), so there is nothing to
recall. n_TOF still had the input on 08-11 (their reprocessing wrote 35 partials
before the directory was emptied again), so the ask is theirs: finish it, or send
us the raw. **Meanwhile do not delete `reproc/prod_v11/224576/`** — with
`official/completed/224576/` empty and no merged file, it is the only complete
product of that run in existence.

**Do not count partials against `ceil(raw/4)`.** n_TOF used **two split sizes** —
10 raw files per job before 07-08, 4 after — so that rule mis-flags the older half
of the campaign, and the raw has aged off disk for 309 of 445 runs anyway.

**The official pass moved.** Against the 08-10 inventory: **27 unmerged runs are
now MERGED**, **24 are being reprocessed from scratch right now** (`completed/`
emptied and refilling — do not read that as data loss; `official_ledger.py` calls
it IN_FLIGHT off the directory mtime), 3 are still PARTIALS_ONLY (224451-224453)
and 2 still zero-byte (224405, 224667). **224560 was already merged and got wiped
and re-queued anyway — its 31.8 GB merged file is gone for now.**

**Two of the newly merged runs are ours** (224573, 224577), so eight runs now sit
in both processings — and the direct test is finally possible.

**Given the same UserInput we reproduce n_TOF bit for bit.** On 224572 (ours
`v12_liqpileup` vs their `v4`) every hit of WAL A-D, PSS A-D, SILI and PKUP matches
on all 22 columns — `tof`, `amp`, `area`, `chi2`, every one. The liquids match too
except `afast` on 3-6 hits in ~85 000 (0.00-0.02 %), a numerically unstable
integral on pathological pulses. On 224574/224577 our `prod_v11` is bit-identical
to official on WAL/PSS/SILI/PKUP and differs **only** in LIQ hit count (official
+17 to +21 %) — exactly the documented v11→v12 liquid step, nothing else. That
measures, rather than infers, the warning in `SLIM_FEASIBILITY_2026-08-08.md` § (c).

**DREAM run_79:** 224573 and 224577 are now official v12; 224576 is mid-reprocess.
When it lands, run_79 is fully covered officially and prod_v11 need not be mixed in.

**224688-224718 stays ours.** All 31 are still RAW_ONLY officially; the pass went
past the block and on to 224719+, which is a different experiment
(`UserInput_2026_EAR2_STAR_commissioning_v0.h`). We hold 30 of them, 831 partials,
674 GB. **224709** is the exception — its last job (partial 0023, one condor
eviction and a retry) was still running at 16:07, so the driver called it stalled;
finish and harvest it by hand.

**Physics widened to all 26 beam runs of the block** (was 13): every tree still
overlaps official in rate and amplitude, 0.00 % off-flash, modal `tflash` in the
same bin. **One run is different — 224708**: PSSD ×39, PSSA ×6.2, PSSB ×1.3
against its neighbour 224707, confirmed over *every* partial (135 beam bunches).
PSSC, all walls and all liquids are normal, and the median amplitude *falls* on
exactly the trees that gained hits (PSSA 124→96, PSSD 131→97 ADC) — a
low-amplitude population, so **the detector, not the processing**. Held out of the
aggregate ranges; `campaign_qa/results/compare_224708.json`. Why is untested.

**The campaign driver has exited:** 15 moved, 3 flagged. The two `COPY FAILED`
runs (224705, 224711) both verify **clean on the ntof disk** — full contiguous
partial set, 0 unreadable, 0 gaps — so that was a `cp -r` exit code, not a bad
transfer; their staging copies can be dropped.

---

## 2026-08-11 — our products check out against n_TOF's own processing

`campaign_qa/` ([`FINDINGS_2026-08-11_campaign_qa.md`](FINDINGS_2026-08-11_campaign_qa.md),
[`report.html`](campaign_qa/results/report.html)). The 17 runs the
campaign has moved to `/eos/experiment/ntof/data/x17/reproc/prod_v12/` were checked
against the runs n_TOF processed themselves. **They are good.**

**Configuration is identical, not merely similar.** n_TOF's production
`UserInput_2026_EAR2_X17_v4.h` **is** our v12_liqpileup — they adopted it after the
July handoff. Every parameter column matches and all 26 referenced pulse-shape
templates are byte-identical (md5). The only differing lines in `history` are the
header file name and the directory the templates are read from (our AFS staging vs
their EOS `shapes_X17_v4`). So this is an equivalence check against the official
product, **not** an absolute validation: a defect in the shared recipe is invisible
to it.

**Structure: 494 partials over 17 runs, 384 GB, 0 unreadable, 0 bunch gaps**, every
run contiguous at `ceil(raw/4)` and every partial actually opened and read (not
sampled) — `verify_transferred.py`.

**Physics, on the 13 runs with beam** (`compare_campaign.py`, one partial each,
against official 224660-224676): intensity-normalised hit rates overlap the official
range on all 12 trees (WALA ours 1483-1556 vs official 1470-1558 hits per 1e12 p);
the modal `tflash` lands in the **same 10 ns bin as official on every tree**;
**0.00 %** of beam bunches are off-flash (broken July processing: 37-85 % on PSS);
arm offsets match, including the PSSC ~33 ns and PSSD ~27 ns features that **the
official runs show in the same trees** — a channel property under this recipe, not
something we introduced. Hit *quality* (`quality_metrics.py`, ours 224691 vs
official 224672) agrees within a few percent: T1 6.66 vs 6.69 ns, MIP peak 1057 vs
1047 ADC, relative width equal.

**Two traps, both paid for, both now handled by the tooling:**

* **Never use 224678-224687 as the control** — those official runs have **no beam**
  (zero PulseIntensity, zero PKUP amplitude). Comparing our beam runs to 224687
  makes our output look 400x too busy. It is not.
* **Gate on protons.** The first partial of 224692 is 75 % empty PS pulses, which
  have no flash, so tflash is 0 and every flash check flags them. Whole-run, 224692
  is 98.0 % beam and clean. `beam_state.py` reads the whole-run beam state from the
  `index` tree (replicated in full in every partial) with one open per run — run it
  before any comparison.

**224706, 224716, 224717, 224718 have no beam at all** (0 of 2615/4/8/16 bunches
with protons). They processed correctly and are simply quiet — a few MB per partial
instead of ~800 MB. Expect more of these in 224701-224718, and note that the
acceptance test for them can only be structural.

**Not yet done:** the 14 runs still in flight; no DREAM slim has been run over the
new block, so the association efficiency and clock QA on it are still open.

---

## 2026-08-10 — 55 runs were processed but never merged; full availability pass

`skip_diagnosis/` ([`README.md`](skip_diagnosis/README.md)). Setting up to
reprocess one of the 41 runs from
[`NTOF_REPROCESSING_REQUEST_2026-08-08.md`](NTOF_REPROCESSING_REQUEST_2026-08-08.md),
we found them already sitting in
`/eos/experiment/ntof/processing/official/completed/<run>/`. **The
reconstruction finished; only the MERGE is missing**, and the pass publishes
only merged files, so `done/` looked empty.

**Campaign-wide inventory of all 445 runs** (`inventory.sh`,
`inputs/inventory_2026-08-10.csv`): 359 MERGED, **53 PARTIALS_ONLY**, 2
MERGE_EMPTY (zero-byte `done/` file), 31 RAW_ONLY. The 55 unmerged carry 2 223
partials and 1 816 GB.

**Availability in beam time** (`availability.py`), over 289.0 h and 342 DREAM
sub-runs — data taking ended 08-10, DREAM now runs to run_162:

| | hours | |
|---|---|---|
| AVAILABLE (merged **or** partials) | **223.3** | **77 %** |
| NEEDS PROCESSING (raw staged, nothing done) | 60.9 | 21 % |
| no n_TOF live | 4.7 | 2 % |

The split is clean in time: every DREAM run through **run_147 is 88-100 %
available**, every run from **run_150 on is 0 %**. That boundary is the pass
stopping at 08-07 19:56, not a data problem. The unprocessed n_TOF runs are the
contiguous block **224688-224718** (31 runs, 12.76 TB raw, all staged).

**The partials are not second-class, proven end to end.** run_116/stat090_0001
slimmed straight off the 224632 partials: **PASS on all 19 clock-QA checks, 0
warnings**, efficiency 94.23 % (held-out 94.16 %), accidental 0.0498 %, residual
RMS 6.87 ns, 1 146/1 146 bunches fitted. Its per-arm offsets land within 0.6 ns
of the locked run_79 v12 values (A -17.6 vs -16.81, B +7.7 vs +7.55, C +1.2 vs
+1.62, D -1.2 vs -0.83) — independent confirmation from the physics, not just
the `history` checksum.

**Is the no-merge deliberate?** No marker or lock file exists anywhere (`ls -A`
on the run directories, `official/`, `done/`), and merged runs keep their
partials too. Size correlates hard — **275 of 275 runs under 20 GB of output
merged, zero failures**, then 76/74/57/25/17 % as size climbs — but the
populations overlap (merged 20.5-42.0 GB, unmerged 20.3-39.5 GB), so it is not a
deterministic rule. **Measured, not guessed:** our own 224688 merge node (44
partials, 34.0 GB) died in FIVE minutes on condor's `max total download bytes
exceeded (max=1024 MB)` plus `disk usage exceeded allowed max` (3 MB requested,
58 GB used) — the transfer cap and disk request, NOT the 1 h wall. That is the
failure `ntof_io.py` recorded in July. It cannot be n_TOF's whole story though,
since they merged 66 runs in the 1-20 GB band that also exceed 1 GB of transfer,
so their merge is invoked differently; we cannot see their config.

Also established: **the `index` tree is replicated IN FULL in every partial**
(224632 partials 1/32/63 all carry bunches 1..4966 with identical Date/Time), so
one open per run is enough and the merge adds nothing structurally.

**224649/224650 have no DAQ directory at all** — drop them from the request; the
"recall from tape" ask was based on a bracketed guess.

**The merge is the failing step, and it left proof:** `done/run224405.root` and
`done/run224667.root` are **zero bytes**, dated 08-05. A failed merge leaves an
empty file rather than nothing, so `exists()` is not a usable test.
`slim_pipeline/config.ntof_files()` now (a) falls back to `completed/<run>/`
when `done/` has no merged file and (b) requires the merged file to be
non-empty first — without (b) those two runs resolve to an empty file while a
complete partial set sits next door. This matches what `ntof_io.py` has said
since July: the merge node dies on condor's 1024 MB transfer cap — now confirmed
directly on our own 224688 merge (§ above).

**Corrected in passing:** a first reading blamed the 2 h `longlunch` wall on the
*processing* jobs. That is measured and real — three of our 78 July jobs were
killed by `SYSTEM_PERIODIC_REMOVE`, absorbed by `RETRY 3` — but it is **not**
why these runs have no output, because their processing completed. Kept in
`skip_diagnosis/walltime_diagnosis.py` as a separate finding.

**Next:** ask n_TOF to re-run the merge only (not 90 h of reconstruction that is
already on their disk), and mention the two zero-byte files. Slim campaign can
re-run over the 28 now that `ntof_files()` finds them.

---

## 2026-08-09 — the plastics really do ring: after-pulses in the PSS hit stream

`pss_ringing/` ([`report.html`](pss_ringing/report.html)). Chasing the long tail
seen after the DREAM/PSS match: **every large plastic pulse is followed by a
train of real secondary pulses in the raw trace**, and the PSA reports them.
~4.4 excess hits per large pulse over 18–1000 ns, against 0.007 on the SiPM
walls in the same run — a factor ~650. Two components: a broad sporadic
population peaking at 32–40 ns and decaying through a microsecond, plus a
2 ns-wide **echo at 81–82 ns identical on all four plastics** (a reflection;
~8 m of cable if it is one bounce at 0.66 c).

Established on run 224572 (v12 hits + local raw chunks) by four independent
checks: an event-mixed accidental control, a time-reversal control (4.13 forward
vs 0.90 backward), the walls as a same-beam control with a 3× wider pulse, and
raw traces conditioned on whether the PSA gave the 81 ns hit. **The PSA is not
inventing these** — the secondary pulses are visible one event at a time.

**The after-pulses are the DREAM/PSS late tail, and there is a cut for it**
([`report_veto.html`](pss_ringing/report_veto.html)). Measured on the reference
pair slimmed locally at ±3 µs: the plastic excess at 150–1000 ns is 122,133 hits
against a core of 47,292.

- **Per-hit flag:** `amp_0 < 0.05 × max(amp_0 on the same channel in the previous
  1000 ns)`. Removes **99.5 %** of the 150–1000 ns excess and 94.8 % of the
  25–150 ns excess, for **10.4 %** of the core — all of it small-amplitude.
  Must be computed on the **full n_TOF stream** (a parent just outside the slim
  window is the case a slim-only recomputation gets wrong). Store `shadow` and
  `dt_prev` as floats rather than the boolean, so R and T stay re-tunable.
- **Cheap fallback, no new branch, works on today's slims:** `amp_0 > 250`
  removes 95.7 % of the late tail for the same core cost.
- **Per-trigger metric:** per (trigger, arm) take the **largest-amplitude**
  plastic hit — not the earliest — and cut on its residual. On the trigger's own
  arm **89.5 % land within ±25 ns**, median −5.6 ns. "Earliest" gives a median of
  −589 ns, because in a µs-wide window the earliest hit is an unrelated single.

This also refines `slim_pipeline/config.py`: the late side is *not* featureless.
At 1 ns binning there is a bump at 70–90 ns where the 81 ns echo lands.

Not done: tuned on one segment of one run. The flag's cost scales with the
singles rate, so R and T want re-checking on a high- and a low-rate segment
before a campaign-wide number is quoted.

---

## 2026-08-08 — the DREAM-keyed slim, and what n_TOF still owes us

Two things landed today. Detail in
[`SLIM_FEASIBILITY_2026-08-08.md`](SLIM_FEASIBILITY_2026-08-08.md) (§8 = what
exists, §9 = what is left).

**1. `slim_pipeline/` — n_TOF hits keyed to DREAM event IDs. Built, validated,
run on condor.** Per (DREAM sub-run × n_TOF run) segment: join → candidates →
fit the clock from scratch → keep every scintillator hit within **±150 ns** of
the fully corrected prediction, plus the same width at +100 µs as an accidental
control. ~33 MB per DREAM sub-run, ~10 min, one core, 3.2 GB RSS. Output goes to
`<eos>/july_beam/runs/<run>/<subrun>/ntof_hits/`.

It reproduces **from the slim alone** the published match (95.89 % / 0.046 %)
and the published liquid same-arm diagonal, on two disjoint hours each against
its own published numbers. Efficiency is identical to four decimals whether a
segment ran locally or on a worker. `slim_pipeline/validate.py` is the check;
`segments.py` says there are **206 ready segments over 60 n_TOF runs**.

*Not done:* the campaign has not been submitted, and the three validated outputs
sit unpublished in `~/x17slim/out` on lxplus.

**2. The n_TOF pass is incomplete — 41 runs to ask for.** The 5–7 August pass
uses our v12 UserInput (verified: identical on all 14 detector rows and all 26
templates, under n_TOF's own name `UserInput_2026_EAR2_X17_v4.h` — **do not
identify it by filename**, it collides with our own `v4_walshapes`). It then
stopped, and 41 runs have **no processed output of any kind**, blocking 117 h —
48 % of the campaign. Request written and published:
[`NTOF_REPROCESSING_REQUEST_2026-08-08.md`](NTOF_REPROCESSING_REQUEST_2026-08-08.md)
and <https://dylan-neff.web.cern.ch/notes/ntof-reprocessing-request.html>.

Two mechanisms, separated: runs after 224687 are missing because the pass
stopped; the in-range gaps correlate with **size** — 0 of 63 skipped below
0.35 TB, 30 of 72 above, rising with size (`slim_study/why_skipped.py`).

**Traps found the hard way, now in code:** the `index` tree's Date/Time are
LOCAL not UTC (a flat 7200 s error against anything else); our own `prod_v11`
runs really are v11, and differ from v12 on the four LIQ rows only — mixing them
inside one DREAM run is safe for the trigger legs and a 14–21 % liquid yield step.

---

## Earlier: the reprocessing itself

Last updated: 2026-07-30 (evening). **The n_TOF side is closed; the analysis has
started, and the DREAM<->n_TOF time calibration is now locked.**

> **Superseded in one respect, 2026-08-08:** "the n_TOF side is closed" was true
> of the *UserInput* and still is — n_TOF adopted v12 unchanged. It is not true
> of *coverage*: 41 runs have no processed output at all. See the section above.

> **The authority on the match is
> `../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`.** Constants, the per-run
> re-derivation recipe, what transfers and what does not. Tooling and slides in
> `../ntof_dream_merge/match_study/`; machine-readable constants exported to
> `../../nTof_x17_DAQ/calibrations/dream_ntof/`. Everything earlier is retired
> into `archive/` (see `archive/README.md`) -- do not build on it.

Headline, on the complete reference pair (2061 bunches, 213 420 triggers):

- **accept window ±25 ns, one band.** 95.84 % efficiency at a **measured**
  0.049 % accidental rate, two-arm ambiguity 0.15 %. The old
  ±150 ns + [250,450] ns window bought +0.17 points of efficiency for 7× the
  background; the satellite band carries no signal on v12 at all.
- **the window was never a resolution.** The DREAM timestamp clock drifts
  ~1 ppm bunch to bunch, smearing the residual in proportion to time since flash
  (9 ns at 1 ms, 37 ns at 40-80 ms). Fitted per bunch and cross-validated the
  residual is **flat at 6 ns** over the whole 80 ms. This corrects
  `time_align.py`'s "36 ns of DREAM trigger jitter".
- **the per-bunch fit was audited for self-fulfilment** and passes five tests:
  107 triggers per bunch for 2 parameters, split-half rho = +0.996 (0.92 ppm of
  real drift against 0.06 ppm of fit noise), a 3-5 % in-sample-vs-cross-validated
  gap, wide-window efficiency **identical to five decimals**, and a wrong-bunch
  parameter swap that makes things worse. `match_study/scripts/bias_check.py`.
- **nothing transfers between processings.** K and T0 refit on v12 to
  1.103724e-4 (+1.35 %) and -253.64 ns, plus per-arm trigger-path offsets
  A -16.81 / B +7.55 / C +1.62 / D -0.83 ns. The wall top/bottom "cable offsets"
  are ±32-39 ns on the official file and within ±5.5 ns on v12 -- a flash-finder
  artifact, not cabling; `dream_trigger.py`'s stored table is flagged.
- **do NOT run the tflash repair on v12**: it would shift LIQC/D by 15 ns and add
  25 ns RMS on PSSC, and the stored time base already has the liquids within 1 ns
  of the walls.
- **the reprocessing checks out in situ**: the v12 liquid flash times reproduce
  the divert-off `flash_timing` calibration to 0.1-0.5 ns; walls spread 4.0 ns;
  per wall channel RMS 2.3 ns; liquid-vs-wall -0.8..+0.2 ns. No internal offsets
  are needed.
- coverage (accidental-subtracted): 96.00 % of triggers get a wall AND plastic
  partner, 98.59 % wall-only -- the plastic leg still costs 2.58 %.
- new: `ntof_dream_merge/fast_singles.py`, a vectorised `dream_trigger`
  (validated bit-identical) -- the original is O(N_hits x N_bunches) and cannot
  run on 2061 bunches.

Earlier the same day: **the handoff package now carries the corrected
saturation story.** `ntof_handoff/README.md` §8b had been left on the retracted
ADC-wrap text (its last commit predated the signed-int16 finding): (a) now says
`satuflag` is reliable on the liquids and structurally absent on the walls, and
recommends cutting `satuflag` **or** `amp` > ~63 800 -- neither alone is complete
(satuflag misses ~9 % of over-ceiling hits; the amp cut misses flagged hits whose
extrapolated amp lands back in range); (b) now says the ADC clips at its rails
with no wrap, gives the ±950 mV baseline-offset table, and adds that the wall
**front end** limits at ~34 600 counts (~half of ADC full scale) where no rail
test can see it. The two `adc_wrap_*.png` figures were withdrawn from the package
and replaced with `sat_examples_liq.png` / `sat_population_liq.png`. The stale
question we were about to ask n_TOF about a per-channel `start` offset is
retracted in the README too -- it was our parser's 259 pre-samples.
**Do not re-issue the old "cut amp > 31 000" advice**: on LIQA that cut removes
2 099 hits of which 1 561 (74 %) are ordinary half-scale pulses.
**New today, and they answer the saturation question end to end** —
`FINDINGS_2026-07-30_saturation_walls_plastics.md` and
`FINDINGS_2026-07-30_liquid_leg_fullpair.md`:

- **The walls have a hard ceiling BELOW the readout limit.** Reported `amp`
  terminates at 43 220-44 915 on all four and never reaches 63 800; the raw
  excursion limits at 32 888-34 635 counts. So it is analogue, and no rail test
  can see it — **cut WAL `amp` > 34 600 in post-processing**. Fires only in the
  flash (physics-time wall amplitudes stop below ~25 000). Now in the handoff.
- **The plastics are the opposite**: PSSA/B/C do reach the ADC rail, so
  `satuflag` fires on them. **PSSD does not** — it is analogue-limited at 44 806
  (70 % of range), which is the real reason it never sets the flag.
- **`amp > 63 800` on a plastic is a FIT-QUALITY flag, not a saturation flag.**
  Correcting what this file said earlier: about half the over-ceiling PSSA/PSSC
  hits are *unflagged because they never clipped* — measured peak 58-62 k against
  a 63 568 rail — and the fit merely overshoots (1.45x on PSSD up to 22-80x on
  genuinely clipped hits). `satuflag` is right about both halves.
- **Flag implemented**: `ntof_io.saturated(tree, amp, satuflag)` +
  `saturation_ceiling(tree)` — one definition, per-family ceilings (WAL 34 600,
  LIQ/PSS 63 800, SILI 59 100, PKUP 59 400). `liq_coincidence.py` and
  `liq_saturated_study.py` now call it.
- **`area` is proportional to `amp` BY CONSTRUCTION, and the PSA guide says so.**
  With AMPLITUDE OPTION=2 "both the final amplitude and area will be determined
  from the fitted pulse", so area = amp x integral(shape): `area/amp` takes
  exactly one value per `pulseshape`, matching the per-shape counts to the hit.
  **The measured pair is `amp_0` (pre-fit maximum) and `area_0` (pre-fit
  integration)** — use those for a real integral or an un-extrapolated amplitude.
  `amp`/`amp_0` is the best saturation diagnostic in the file: ~1.0 clean,
  1.24-1.30 wall flash artifacts, 1.45 PSSD overshoot, 22-80 clipped plastics.
- Figures: `liq_study/pss_over_ceiling_PSSC.png` (the flash plunge to the rail),
  `liq_study/wal_front_end_WALB.png` (the divert step parked at ADC zero),
  `liq_study/wal_pss_saturation.png` (spectra + width vs amplitude).
- **Method** (`liq_study/sat_curve.py`): width-vs-amplitude plateau departure,
  calibrated on the liquids where `satuflag` is truth. The liquids stay flat to
  0.1 ns up to the ceiling *even inside the flash*, so a departure below the
  ceiling is real and not a flash artifact. Do **not** use the automatic
  "1.2x the plateau" rule — it mis-called both LIQ and PSS; read the table.
- **A physics-time clipped liquid pulse keeps its time** (`clipped_timing_check.py`):
  dt to a fixed-depth raw crossing is 3.5-3.8 ns against 3.6-3.7 for unclipped
  controls, so saturated hits are usable as TIME hits with `amp` as a lower
  bound of 63 800. The 114-129 ns mistiming tail is entirely flash-region.
  `area` cannot help recover amplitude — it is exactly proportional to `amp`.

The §8b(a) table is now a real whole-run census (it had been a single partial,
3.4 M LIQA hits, labelled as the run: the run has 51 M) — reproduce it with
`liq_study/amp_ceiling_census.py`, output kept as
`liq_study/amp_ceiling_census_v12_224572.json`.
`ntof_dream_merge/liq_coincidence.py` carried the same wrong cut and now uses
`ntof_io.saturated()`. **Re-run on the whole of `stat090_0000`** (1012 bunches,
100 083 exclusively-matched events, 3.3× the old sample) it reproduces the 07-29
table cell by cell — same-arm diagonal at −5…−25 ns, excess **3.6-5.9×** over the
shifted control. **`stat090_0001` replicates it** on a disjoint hour — diagonal
cells 0.164/0.146/0.016/0.092 against 0000's 0.165/0.151/0.018/0.094, same
−5…−25 ns offsets. Restate the excess as **2.7-5.9×** over both sub-runs, not the
"5-7×" from 300 bunches (LIQC is the weak cell, 2.7-3.6×, on a better-measured
floor with an unchanged signal cell).

**The merge tooling is ~46× faster as of 2026-07-30** and bit-identical: the
per-bunch/per-event Python double loop in `liq_coincidence.py` is now one
`window_residuals()` call (bunch and time packed into a sorted float64 key, two
`searchsorted`, one ragged gather), and `ntof_io.variant_cache()` replaces the
`tempfile.mkdtemp()` sandbox in all five scripts with a persistent directory
fingerprinted on the file set — same isolation, but the bunch index is not
rebuilt every run. **1 h 52 min → 2 min 25 s** cold, **1 min 55 s** warm, with all
33 residual histograms identical bin-for-bin (363 038 entries both ways).

§3 of `archive/FINDINGS_2026-07-29_dream_crosscheck.md` (300 bunches, wrong cut) is now
**superseded** by `FINDINGS_2026-07-30_liquid_leg_fullpair.md` — both sub-runs,
correct cut. Read the 07-30 file for any liquid number.

Earlier, 2026-07-29 (evening): **the DREAM cross-check has run on the
FULL reference pair** -- see `archive/FINDINGS_2026-07-29_dream_crosscheck.md`
(retired: its matcher numbers predate the re-derived time map, and its MM
cross-check needs re-running at ±25 ns). On all
2061 bunches / 213k DREAM events of both run_79 sub-runs: **v12 95.7 % / 0.5 %
on its own tflash**, vs official+repair 92.4 % and official-alone 12.2 %.
Both sub-runs agree to 0.0 points. First physics through the merge both pass:
MM chamber activity concentrates in the matched arm, and the liquids show
5-7x same-arm coincidence excesses at -5..-25 ns offset (the v12 LIQ time
base is wall-aligned). **Nothing found motivates another UserInput variant --
ship the campaign on v12.**

Earlier the same day: the pre-ship tests (`FINDINGS_2026-07-29_pre_ship_tests.md`);
headline confirmed at 252 bunches, T2/T3 pass. ~~Two output-integrity problems --
ADC wrap-around and an unusable `satuflag` -- go in the handoff~~ **both
retracted the same evening**: the raw samples are signed int16 and the tooling
read them unsigned, so there is no wrap, and `satuflag` is verified good on the
liquids (119/123 clipped runs matched per pulse). T4's per-hit liquid check is
no longer blocked either -- the raw-to-`tof` offset is a constant 259 samples.
See `FINDINGS_2026-07-29_signed_decoding.md`.

**Open, for a separate session:** the shared raw parser
`nTof_x17_DAQ/stream1_monitor/ntof_raw.py:163` still decodes samples as `<u2`.
The full write-up — evidence, the fix, the two operational consumers to re-check
(`stream1_size_controller.py`, `wall_probe.py`), the `0x8000` fill collision and
the 259-pre-sample offset — is in that repo as
`stream1_monitor/SIGNED_DECODE_FIX_NOTE.md`. Nothing there has been changed yet.

**Auditing this?** Start at `REVIEW.md` -- it maps every claim to the tool that
produced it, says what is reproducible and what is ephemeral, and lists the
mistakes I made and corrected so you know where the error modes were.

---

## Where things are

| | |
|---|---|
| variant studies (run 224572) | `/eos/experiment/ntof/data/x17/reproc/<variant>/completed/224572/` |
| production runs (224573-224579) | `/eos/experiment/ntof/data/x17/reproc/prod_v11/<run>/completed/<run>/` |
| processing scratch (must be /afs) | `/afs/cern.ch/work/d/dneff/x17_reproc/` |
| UserInputs, staged | `/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/<variant>/` |
| UserInputs, source | `ntof_processing/userinputs/<variant>/` |
| package for n_TOF | `ntof_handoff/` |
| DREAM-vs-reprocessed entry point | `../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md` |
| micromegas channels in the n_TOF DAQ (MMA/MMB, MGAS) | `NTOF_MICROMEGAS_SIGNALS.md` + `mm_signals/` |
| the flash charge measured on one strip, and vs the HV current | `mm_flash/` (`report.html`; open question in `mm_flash/HANDOFF_CHARGE_COMPARISON_2026-08-11.md`) |
| retired documentation | `archive/` -- do not build on it |
| local copy of 224572 v12 | `/media/dylan/data/x17/ntof_reproc/v12_liqpileup/` |

Note on 224572: it has no directory under `prod_v11/` because it is the
reference run every variant was built on -- its production-configuration output
is `reproc/v12_liqpileup/completed/224572/`. Nothing is missing; the naming just
differs. The user EOS (`/eos/user/d/dneff/x17/reproc/`) is kept empty: every
output is copied to the ntof disk and verified file-by-file before the source
is removed.

`ssh -K` is mandatory. Without delegated credentials there is no AFS token and
no condor auth, and `/eos/user/d/dneff` looks like it does not exist.

**Never merge a run.** The official merge node cannot (condor file-transfer cap
at 1024 MB), hadd over EOS dies and leaves a truncated file that still opens.
`ntof_io.ntof_paths()` chains `run<run>.parts/run<run>_NNNN.root` instead.

## Division of labour, settled

- **Time base** comes from `flash_timing/` -- `t_flash = tof_PKUP + C`, ~1 ns.
  Not from stored `tflash`, even reprocessed.
- **Hit content** comes from the reprocessing. That is what these variants are
  optimising, and it is what the calibration cannot give.

## Variants

| variant | change vs its base | singles eff / false | verdict |
|---|---|---|---|
| v1_flash | G-FLASH THRESHOLD only | - | superseded |
| v2_elim | + PSS/LIQ AREA/AMP 1..60, PSS amp thr 50 | - | superseded |
| v3_shapes | + 551 ns WAL and LIQ templates | - | LIQ part rejected |
| v4_walshapes | v2 + WALL templates only | 95.2 / 0.6 | superseded by v8 |
| v5_liqshort | v4 + 81 ns LIQ templates | - | rejected |
| v6_lowthr | v4 + PSS/LIQ amp thr 25 | 95.2 / 0.6 | **no effect on the matcher** |
| v7_step | v4 + STEP SIZE WAL 5/5, PSS 2/3, LIQ 2/3 | 95.1 / 0.6 | neutral/slightly worse |
| v8_pssfit | v4 + PSS shape fitting, 101 ns templates | 96.4 / 0.6 | the big win; superseded by v12 |
| v9_liqaug | v4 + LIQ shipped pair + a measured third | 95.2 / 0.6 | neutral |
| v10_pssfit_step | v8 + PSS STEP SIZE 2/3 | 96.4 / 0.6 | equal to v8 |
| v11_pssfit_width | v8 + PSS SIGNAL WIDTH LOW 4 ns | 96.4 / 0.6 | superseded by v12 |
| **v12_liqpileup** | v11 + LIQ STEP SIZE 1/3, fast/slow boundary | **96.4 / 0.6** | **production, shipped to n_TOF** |
| v13_liqexpand | v12 + LIQ EXPAND PULSES 1, 150 ns width | - | rejected, -17..-28 % liquid hits |

### The liquids, settled

`liq_study/FINDINGS_liquids.md` has the detail. In short:

- **not templates**: every template we built was measured on the isolated
  minority of pulses and none transferred. Note the "8-24 % isolated" figure
  uses a 200 ns window, so it measures TAIL overlap -- the fast components are
  mostly resolvable (24-30 ns median gap vs 6 ns FWHM)
- **photon statistics floor**: fit residual scales as sqrt(A), flat at
  0.61-0.67 over a factor 25 in amplitude, so no template basis can absorb it.
  **07-29: true of LIQA/LIQC/LIQD, NOT of LIQB** (residual/sqrt(A) 0.62 -> 1.59;
  an amplitude-binned basis cuts it 24 % held-out). And the "saturation breaks
  the scaling at the rail" line was measuring the ADC wrap -- with wrapped
  pulses dropped it does not break
- **not two pulse classes**: tail/total is one band at 0.21 above 3000 ADC
- **v12 works**: LIQ `STEP SIZE` 2/4 -> 1/3 gives **+14 to +21 % yield**, chi2
  neutral-to-better, pileup flag +50 %, walls and plastics bit-identical
- **PSD is not obtainable from the PSA**: `afast`/`aslow` are 0 % filled;
  setting the boundary fills `afast` but leaves `aslow` at zero because the
  slow component lies outside the reconstructed pulse boundary, and expansion
  to reach it costs more than it gains. **The reported liquid `area` has
  therefore always been missing its slow component** -- in the official
  processing too.
- **raw waveforms would NOT help**: an iterative deconvolution on the raw data
  finds 0.67x the PSA's hits, not more (`deconv_vs_psa.py`). And 67-76 % of
  pulses have a neighbour inside their own 150 ns tail, so a custom fitter
  faces the same overlap. This is a rate limitation, not a software one.

### The processing has hit its floor at 96.4 %

**Note (07-29):** every figure in this section is from the original 100-bunch
grading, which is the sample all the variants were compared on and so remains
the right basis for comparing them *to each other*. On the larger 252-bunch
sample the absolute numbers are v12 96.3 % / 0.5 % against v4 95.3 % / 0.5 % --
same gap, slightly lower false rate. Quote the 252-bunch numbers outwardly.

v8, v10 and v11 give **identical** matcher efficiency (96.4 % / 0.6 %) and
identical leg breakdowns, despite reconstructing the plastics very differently:

```
              PSS chi2 p50   PSS amp p50   PSS hits   hits with amp>2000
  v8            baseline       baseline     baseline        2834 / 4757
  v10_step      +1.6..+4.0%    +20..+44%    -13..-17%       2822 / 4746
  v11_width     -2.6..-13.2%   -16..-25%    +28..+34%       2845 / 4767
```

The last column is why: **above ~2000 ADC the three are identical to <1 %**.
Once shape fitting is on, the trigger-relevant plastic population is fixed and
the remaining knobs only shuffle small pulses, which the discriminator never
sees. Three very different reconstructions giving the same efficiency is strong
evidence the limit is no longer in the PSA.

So the remaining 2.5 % plastic-leg loss is **not a pulse-recognition problem**.
If it is worth chasing, it is analysis-side: the per-arm discriminator
threshold model (`thr['plastic']`), the D_PMTS channel selection, plastic dead
time, or genuine detector inefficiency. Do not spend more UserInput variants
on it.

**v11 chosen over v8** on the tiebreakers: same efficiency, same timing and
amplitude quality (every metric within 0.8 %), same large-pulse yield, but
3-13 % better plastic fit chi2 and 28-34 % more plastic hits available to
non-trigger analyses. **v12 then adds the liquid fix on top of v11 and is what
ships**; the plastic and wall configuration is identical between them.

### What the second sweep established

- **v8_pssfit wins on the headline**: 95.2 -> 96.4 % efficient at the same
  0.6 % false, 93.4 -> 95.5 % in the hardest 1-3 ms bin.
- **and it does so with FEWER plastic hits** -- 0.72-0.99 of v4 at every
  amplitude cut -- while producing MORE valid candidates (103,816 vs 101,809).
  The gain is plastic TIMING in pileup, not plastic yield. Hit count and
  quality point opposite ways here; that is what the scorecard is for.
- **v6_lowthr changes nothing for the matcher** despite +15-25 % more plastic
  and +29-47 % more liquid hits: the trigger emulation thresholds the plastic
  leg in mV at the discriminator, so 25-50 ADC hits never enter it. Those hits
  are real and remain available for non-trigger analyses, but they do not
  belong in production.
- **The plastic leg is still what limits the AND.** Wall-only efficiency is
  98.9 % and flat in time; the AND is 96.4 %. The plastic costs 2.5 % overall
  and 3.4-3.7 % at 1-10 ms -- a pileup signature. v10/v11 target exactly that.
- The liquids have now resisted three template treatments. Stop there.

## Scorecard, and the v4 baseline to beat

Efficiency is the headline; timing and amplitude are the guards. Accept a gain
only if the guards do not degrade.

```
efficiency  dream_regression.py   singles matcher   95.2 % eff / 0.6 % false
                                  (1-3 ms bin)      93.4 % / 2.6 %
timing      quality_metrics.py    T1 wall top<->bot sigma      6.65 ns
                                  T2 wall<->plastic sigma      6.46 ns
                                  T2 centre (wall vs plastic)  8.75 ns
                                  T3 walk over amp deciles     1.38 ns
amplitude                         A1 MIP peak                  1081 ADC
                                  A1 FWHM/peak                 1.22
                                  A2 log(top/bot) resid        0.362
                                  A2 sqrt(top*bot) flatness   +9.1 %
content     grade_candidate.py    flash-id bad bunches         0.0 %
                                  per-arm offsets      +1.5/+2.0/+1.0/-2.0 ns
            compare_fits.py       WAL chi2 p50           0.85-1.06
```

Two things to remember when reading these:

- **Accidental subtraction is not optional.** Both trees are high-rate; without
  an off-time sideband subtracted, T2 reads 38.8 ns instead of 6.46 and the MIP
  peak does not exist. Any coincidence width quoted elsewhere in this project
  without subtraction is inflated.
- **`match_window`'s efficiency is not evidence at early times** -- its own
  false-match probability is ~100 % at 1-3 ms. Quote the singles matcher.
- **These sigmas come from a background-subtracted second moment.** An earlier
  FWHM/2.355 estimator with 2.5 ns bins reported 3.18 ns for every variant and
  could not discriminate at all -- `tof` is quantised to 1 ns. If you find
  3.18 ns quoted anywhere, it is the stale estimator.

## The loop

```bash
# 1. generate + stage + submit
python ntof_processing/make_variants.py vX_name
./ntof_processing/deploy_userinput.sh vX_name <local> /afs/cern.ch/work/d/dneff/x17_reproc/userinputs
rsync -a -e "ssh -K" <local>/ dneff@lxplus:/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/
ssh -K dneff@lxplus  # then RunProcessing.sh -y 2026 -a EAR2 -c X17_measurement -r 224572 \
                     #   -p .../userinputs/vX/UserInput.h -o /eos/user/d/dneff/x17/reproc/vX

# 2. grade -- partials land ~25 min in, no need to wait for all 16
xrdcp partials 0001,0002 down       # bunches 1-397; the DREAM pair needs 146-245
python ntof_processing/dream_regression.py <dir>     # efficiency
python ntof_processing/quality_metrics.py  v=<files> # timing + amplitude
python ntof_processing/grade_candidate.py  v=<files> # flash id + offsets + counts
python ntof_processing/compare_fits.py     a=.. b=.. # fit chi2

# 3. move off the user quota when a variant is kept
#    (verify per-file sizes BEFORE deleting the source)
```

## Which runs to reprocess, and what is actually still on EOS

Measured 2026-07-28 (`scratchpad/eos_inventory.txt`, regenerate with the loop in
the git log of this file): **156 of the 329 run directories still carry
stream1**, spanning **2026-07-02 to 07-28**. The missing 181 are scattered
through the range, not a clean age cutoff -- so retention is NOT the simple
"~2 weeks" the earlier handoff assumed, and it cannot be predicted per run.
Check before planning, do not extrapolate.

The two DREAM runs staged locally, and the n_TOF runs that cover them by
wall-clock:

| DREAM run | window | n_TOF runs | on EOS? |
|---|---|---|---|
| run_79 | 07-26 18:07 -> 07-27 10:00 | 224572 (done), 224573-224579, 224580 | all present |
| run_55 | 07-18 19:11 -> 23:53 | 224498, 224499 | both present, 165 / 156 files |

So **9 runs remain to reprocess** for the two DREAM runs we hold, and all of
their raw data is still on disk. Several are short (224575 has 17 raw files,
224579 has 1, 224578 has 71) -- fine, just quick.

## Production status

run_79's n_TOF coverage is **reprocessed and complete**. Verified that partial
count equals job-list count for every run, so nothing failed silently:

| run | raw files | job lists | partials |
|---|---|---|---|
| 224572 | 152 | 16 | 16 (the reference, many variants) |
| 224573 | 156 | 16 | 16 |
| 224574 | 152 | 16 | 16 |
| 224575 | 17 | 2 | 2 |
| 224576 | 150 | 15 | 15 |
| 224577 | 166 | 17 | 17 |
| 224578 | 71 | 8 | 8 |
| 224579 | 1 | 1 | 1 |

`RunProcessing.sh` splits by ~10 raw files per job, which is why the short runs
have few partials -- not a failure. The merge node aborts on every long run
(the 1024 MB condor transfer cap); that is expected and bypassed by reading
partials. **run_55 was dropped deliberately** -- n_TOF will reprocess the
campaign from our UserInput instead.

## Next

0. ~~Run `archive/PRE_SHIP_TESTS.md`~~ -- **done 07-29**, results in
   `FINDINGS_2026-07-29_pre_ship_tests.md`. T1 green on 2.5x the sample
   (v12 96.3 % vs v4 95.3 %, gap preserved), T3 green, T5 says keep the
   boundary but document it hard, T6 holds except on LIQB. **T4 is not closed**
   -- the per-hit raw classification could not be made trustworthy.
1. Send `ntof_handoff/` to n_TOF (UserInput v12, 26 templates, README, report),
   **with three additions to the README** that came out of the tests:
   - `satuflag` (**rewritten 2026-07-29 evening**): it is reliable on the
     liquids -- verified against the raw waveforms on 119 of 123 clipped runs,
     including every physics-time clip -- and is **never set on the walls**,
     because wall saturation is an undershoot outside the detected pulse
     window. A flagged hit must be **cut, not used**: its `amp` is a fit
     extrapolation (66 k-832 k against a physical ceiling of 63 800). The old
     advice to cut on `amp` above ~31 000 was based on a decoding error and
     would throw away ordinary half-scale pulses.
   - `aslow` is always zero and `(area - afast)/area` is **not** an n/gamma
     discriminant: its per-pulse spread is 4-9x the physical band and it drifts
     a factor two with amplitude. Usable in aggregate only.
   - the liquid `area` is missing its slow component (already known).
2. Raise with n_TOF separately, because these are PSA/DAQ properties and affect
   the official processing too:
   - the liquid `area` / slow-component issue;
   - ~~ADC under-range wrap-around~~ **withdrawn: there is no wrap** (our
     decoding error, not a DAQ property);
   - `satuflag` not being set for the walls at all -- still true, and now
     understood: wall saturation is a negative undershoot, opposite to the
     pulse direction, and `AnalyseSaturation` only scans inside found pulses.
3. **Optional, if the liquid yield claim has to be airtight:** T4's per-hit
   question needed the raw-vs-reconstructed time alignment, and that is now
   **solved** (2026-07-29 evening): `tof = start + j - 259` for zero-suppressed
   blocks, constant to +-0.6 ns over 220 isolated pulses, because the ACQC
   block `start` is the zero-suppression trigger sample and the payload begins
   259 pre-samples earlier. The flash block starts at 0 and needs no offset.
   Nothing to ask n_TOF; T4 can simply be redone.
3. If liquid PSD is wanted, request stream1 raw for the runs of interest
   (~2.7 GB per 70 s chunk, ~150 files per run); reader and extraction tooling
   already exist in this repo.
