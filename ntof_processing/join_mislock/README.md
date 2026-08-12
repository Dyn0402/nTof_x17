# join_mislock — the 2026-08-11/12 matching-failure investigation

**The n_TOF↔DREAM matching failures (25.7 % of attempted beam, 107 of 291
segments) are TWO bugs of ours, both silent, both recoverable. The data is
fine.**

1. **Whole hours** — `pulse_match` takes a supercycle-shifted wall-clock lock
   on a count tie. Demonstrated recovered at 95.47 % efficiency on a formerly
   total-loss hour.
2. **Boundary slivers** — `bunch_join`'s offset bootstrap takes its median
   over *all* bursts including those with no pulse in the run, walking the
   offset off a correct lock by however far the sub-run overhangs. This is
   the "mystery class". Demonstrated recovered at **95.56 %** on the exemplar
   that had defeated every previous probe. See the solved section below.

Neither shows up in any join statistic: both produce ~100 % matched bursts at
a few ms rms, because a wrong lock on the 1.2 s PS grid is still a lock.

Narrative and full evidence chain: the campaign report
[`../SLIM_CAMPAIGN_2026-08-12.md`](../SLIM_CAMPAIGN_2026-08-12.md) and the
published note <https://dylan-neff.web.cern.ch/notes/ntof-dream-join-mislock.html>.
This directory holds the scripts that established it, in the order they ran,
plus the key result files. Everything else (inventory, shift table, margin
CSV, all diag outputs) is on lxplus in `/afs/cern.ch/work/d/dneff/x17slim/`.

## The mechanism, one paragraph

`pulse_match.match_subrun` aligns each DREAM sub-run to the beam record by
scanning a ±120 s offset and keeping the one that matches the most trigger
clusters. Both the pulse timing and the parasitic/dedicated intensity schedule
repeat with the accelerator supercycle (39.6 s or 43.2 s depending on the
hour), so locks whole cycles apart match ~100 % of clusters with ~5 ms rms —
the objective cannot tell them apart. On a tie, `n > best_n` keeps the first
(most negative) offset scanned; ±120 s admits exactly ±3 cycles, so corrupt
hours sit at −3 supercycles (measured: bunch shifts +26 and +20 = 118.8 s at
two different spacings; +41 = 129.6 s = 3×43.2 on the other schedule). Which
hour fails is Poisson noise on which lock catches one more cluster — the
margin study found 35 of 41 failures at count-margin exactly 0, and **14
accepted segments at margin ≤ 2** (healthy by coin flip; listed for
verification). Boundary slivers are the same count degeneracy in
`bunch_join`'s ±3 s δ-scan against a truncated pulse list (minority side lost
26 of 26 straddling pairs). A mis-locked segment looks healthy in every join
statistic and simply writes no file — the QA never sees it.

## Scripts (chronological)

| script | question it answered |
|---|---|
| `marginal_closure_test.py` | is the failure a statistics floor? (no — 14-min truncations of correctly-joined data fit easily; also proved the −0.98 ms wide-scan feature sits under healthy fits) |
| `flashref_subpop_test.py` | is the −0.98 ms structure a mis-tagged-flash subpopulation? (no — carried by ordinary matched events; it is the 0.9927 ms post-flash hold-off edge) |
| `dream_forensics.py` | is anything wrong with DREAM in the corrupt hours? (no — flash leads 100 % of bursts, multiplicity and hold-off identical to clean hours) |
| `perbunch_lag.py` | does the coincidence exist per bunch in a failed segment? (yes, everywhere — though the per-bunch lags it reports for a mis-joined segment are artifact+noise; see the kill-list) |
| `recovery_shift.py` | does the standard chain recover a mis-joined hour once the bunch shift is applied? (**yes: 95.47 % / cv 95.43 %, S/N 1319, accidental 0.065 %** — `recovery_shift_run_96_stat090_0001_224597.json`) |
| `arbitration_floor.py` | is there a segment length below which arbitration cannot work? (yes, and it is structural: the count margin grows ~linearly with clusters when schedules differ — run_79/0001 wins by count at 25 clusters — but the intensity discriminant is a STEP function: it is carried by sparse schedule-break events ~one per 10–40 min, so run_96/0001 has r-separation 0.001 through 150 clusters then 4.2σ at 200, and run_86/0001 stays ambiguous to 400. A 5-minute sub-run typically contains no discriminating event at all; the scan route is the standard path for short segments, and the AmbiguousLock message now says so below 200 clusters) |
| `sliver_census.py` | which "sliver" failures are really sliver-class? (36 of 78 have no fitted sibling and are whole-sub-run mislocks in disguise — recoverable; the mystery class is the 42 with a fitted sibling, all ≤ 402 bunches; the run-START orientation hypothesis died 33/33) |
| `margin_study.py` | what separates failed from fitted sub-runs? (the pulse_match count margin: failures 35/41 at 0, all ≤ 8; fitted median 23; **14 fitted at ≤ 2**; run_102/0003's dead tie −69.3 s r=0.508 chosen vs +60.3 s r=0.925 true validated the intensity-fluctuation discriminator in the wild) |
| `cross_bunch_matrix.py` | where did the mystery class's true partners go? (every DREAM bunch × every n_TOF bunch; found the off-diagonal ridge at −280. `--dump`/`--load` splits extraction from the FFT so the heavy stage runs on any box with numpy. **Its `z` ranking is degenerate — believe only the 20 ns sharpness**) |
| `shift_ridge.py` | how much of the ridge is real? (**129 of 130 eligible bunches, 99 %, sharp at −280 ± 1**, 45 counts over a floor of 0 at −250 ns — the measurement that closed the mystery class) |

The wide bunch-shift scans that found the mechanism used the existing
`slim_pipeline/segment_diagnose.py` with `--span 200` (condor wrapper
`diag3_*` on lxplus). Eight of eight scanned failures show exactly one sharp
peak (S/N 541–1822).

Scripts here run standalone next to the repo packages (they `sys.path` their
own directory; run them from a checkout root or a condor sandbox that
transfers `ntof_processing`, `ntof_dream_merge`, `ntof_july_analysis`,
`common`, `mx_july_beam_qa`). `X17_BEAM_JULY` must point at a beam_july tree.

## The fix (design agreed 08-12 ~03:00, NOT yet implemented — Dylan's call)

1. `pulse_match`: score locks by count **+ intensity-fluctuation correlation**
   (the schedule repeats, the fluctuations don't: r 0.925 vs 0.508 picked the
   true lock outright) against the per-window grid measured from the pulse
   timestamps; check offset continuity across a DREAM run; **fail loudly on
   an ambiguous winner** — the silent tie-break is the actual bug.
2. `clock_qa` absolute check 'pulse_match margin adequate': proposed
   WARN < 10, FAIL ≤ 2 unless scan-verified; test set must include a
   margin-0-but-correct segment.
3. Provenance before any bulk re-slim: `calibration.json` gains `join_shift`,
   lock margin, r-score.

## Recovery recipe

Whole-hour class (41 segments): with the 2026-08-12 pulse_match fix
(below), **just re-run the slim** — the join locks correctly on its own and
the standard chain runs unmodified. Acceptance test: run_96/0001×224597
end-to-end through the fixed chain with zero manual input →
**eff 95.4773 %, accidental 0.065 %, full product written**, provenance
`chosen_by: intensity, margin 0, r_sig 3.2` in calibration.json. Sub-runs
the fix declares AmbiguousLock need one shift scan
(`segment_diagnose --span 200`), then re-run with
`match_subrun(..., accept_offset_s=<scanned offset>)`.

## ✅ SOLVED 2026-08-12: the sliver class was OUR bug, one line

**Everything in the section below headed "the sliver class is NOT recovered"
was wrong, and it was wrong for one reason: the join was never correct. The
δ-scan locked correctly and the bootstrap that followed threw the lock away.
The exemplar now fits at 95.56 % efficiency.** The retracted reasoning is
kept below the fix because the way it failed is the lesson.

### The bug

`dream_event_to_bunch` bootstraps the final offset from the scan's lock:

```python
delta = float(np.median(epoch - ps[assign(best_delta)]))      # WRONG
```

`assign()` returns the *nearest* pulse for every burst, **clipped to the
first or last pulse of the list**. A burst whose pulse is not in this n_TOF
run at all — every burst of the sub-run that falls outside the run, which on
a boundary sliver is the MAJORITY — still gets one, and contributes
`epoch − ps[0]` instead of the offset. The median over all bursts therefore
walks the offset by however far the sub-run overhangs the run. The docstring
already said "the offset is DEFINED by the matched pairs"; the code never
applied the mask.

Measured on run_79/stat090_0002 × 224573 (77 % overhang):

| quantity | value |
|---|---|
| δ-scan lock (correct) | **+0.790 s** → 247 bursts, bunches 1–247 |
| bursts with no pulse in this run | 806 of 1053 (77 %) |
| median over ALL bursts (**shipped**) | **−957.971 s** |
| median over MATCHED only (correct) | **+0.837 s** |
| what the corrupted δ produced | 277 bursts, 277 bunches, range 1–527, resid 8 ms |

That last row reproduces the shipped campaign log bit for bit. Every burst
was paired with a pulse **~280 later** than the one it belongs to.

### Why it hid for a whole campaign

Both sides sit on the 1.2 s PS grid, so a wrong lock is still a lock:
residuals stayed at 8 ms, 277 of 277 bursts landed on the grid, and the
"PulseIntensity matches the CSV at r = 1.0000" check was **circular** — it
compares n_TOF against the CSV at the same wrong bunch, and never compares
DREAM against n_TOF. Probe (c) below is that circularity, written up as
proof of innocence.

`delta_hint_s` did **not** protect against it: the hint constrains the scan
window, and the bootstrap then discards the scan's answer.

### How it was caught

`cross_bunch_matrix.py` — correlate every DREAM bunch against every n_TOF
bunch of the run (1 µs-bin FFT over ±80 ms), then test whether the best
match sharpens to 20 ns. **129 of the 130 DREAM bunches whose partner exists
in the file (99 %) show a sharp coincidence at n_TOF bunch b−280**: 45 counts
in a 20 ns bin over a floor of 0, residual −250 ns ± 20. The 10 bunches whose
partner would be bunch < 1 match nothing, exactly as they must. The n_TOF
hits were never missing — they sat under a bunch index 280 lower than the one
we asked for.

Note the matrix's own `z` column is saturated garbage (MAD = 0 on sparse
rows, so `z = pk/1e-9` ~ 1e8) and its argmax picks an arbitrary row among
ties; that is why the first pass reported only 26 sharp matches. The verdict
rests on the 20 ns sharpness test, never on z. `shift_ridge.py` re-tests the
shift window directly and is the number to quote.

### The failure file is clean — checked directly

v11 partials of 224573: 200 bunches per file, contiguous global numbering
1–1000, `index` tree carrying the whole run's 3118 bunches in every partial,
PKUP 1000 bunches all exactly on the 1.2 s grid, zero bad psTime,
PulseIntensity identical to the index tree. Nothing wrong on the n_TOF side.

### The fix

```python
k0 = assign(best_delta)
sel = np.abs((epoch - best_delta) - ps[k0]) < MATCH_TOL_S
if not sel.any(): raise RuntimeError(...)     # refuse; do not invent an offset
delta = float(np.median((epoch - ps[k0])[sel]))
if abs(delta - best_delta) > PS_SPACING_S: raise RuntimeError(...)
```

The second guard is the general lesson: **the bootstrap refines the scan's
lock, it may not move it.** Anything past one pulse spacing is a different
lock arrived at silently — the shape this bug had, and the only reason it
survived a campaign. It fires on all 48 affected segments.

### Blast radius

The bug can only bite when the overhang exceeds **half the sub-run's BURSTS**,
because that is when the median falls into the clipped population. Measured
by construction in `overhang_threshold.py`: the threshold is sharp at 50 %
and identical for start-side, end-side and both-side overhang, and the
corruption grows continuously from zero as it is crossed.

**The fraction is of bursts, not of time**, and the two diverge across a beam
gap, a DAQ pause or a parasitic-only stretch. Thinning pre-run bursts while
holding the wall-clock extent fixed moves the corruption while time-overhang
stays at 76.6 %: 62.0 % bursts corrupt, 52.2 % corrupt, 45.1 % clean. So the
`overlap_frac` table below is a wall-clock **proxy** — a good one where
density is roughly uniform, and wrong for exactly the segments that straddle
a beam gap. **Measured on `run_81/stat090_0001 × 224581`:**

| | overhang | overlap |
|---|---|---|
| wall clock (`overlap_frac`) | 0.614 | 0.386 |
| **bursts (what the median sees)** | **0.335** | **0.665** |

349 bursts in the sub-run, 232 matched by this n_TOF run; 12.3 bursts/min
inside the run against 3.9/min over the overhanging stretch, **3.2× sparser**.
Measured from the sub-run's own burst list at a lock matching 98.0 % of
bursts to the beam record, so it does not depend on the join under test.

**The proxy does not merely blur the classification, it can INVERT it.** This
segment is listed at 0.386 overlap — nominally deep in the bites-region — and
is actually at 0.665 in the units that matter, i.e. on the other side of the
line. Any segment straddling a beam gap can move that far. That is the case
for taking the tables from the fixed campaign rather than patching the
existing ones.

Do **not** recompute the tables in bursts from the first campaign — on a
mislocked segment the matched-burst count is itself the corrupted number
(run_81/0001 × 224580 reporting 44 bunches is exactly that trap). The fixed
campaign computes matched-against-total bursts as part of every join, so the
burst fractions fall out of it with no separate reconstruction; take them
from there and drop this caveat. A segment wholly
inside its n_TOF run (`overlap_frac` 1.000) has `sel` all-true and the two
medians are identical — the whole-hour class was never exposed, and neither
were the two recoveries made during the 08-12 campaign. Against the campaign
inventory, `overlap_frac < 0.5` predicts the failures:

| | FAILED | OK |
|---|---|---|
| overlap_frac < 0.5 | **48** | 1 |
| overlap_frac ≥ 0.5 | 18 | 54 |

**The 18 above 0.5 are not a third mechanism — they are bug 1.** Crossing the
66 sliver failures against the census's sibling test:

| | no fitted sibling | sibling_ok |
|---|---|---|
| overlap < 0.5 | 15 | 33 |
| overlap ≥ 0.5 | **17** | 1 |

17 of the 18 have no fitted sibling, i.e. they are the census's whole-sub-run
`pulse_match` mislocks filed as slivers because the sub-run straddles a
boundary. Not merely a correlation: two of the census's *demonstrated* cases
sit in this list (run_132/0007 × 224663 at overlap 0.846, run_139/0007 ×
224668 at 0.560, both pinned by cached v1 offsets), as do both dark-run
families (run_128/0000 at 0.881; run_156/0009 and /0011). They are covered by
`ce8ced7`, not by this fix.

**Exactly one segment in the campaign is unexplained: `run_81/stat090_0001 ×
224580`** — sibling_ok (so the pulse_match offset is pinned), nominal overlap
0.689 (so the bootstrap median should stay in the matched population), yet it
joined only 44 bunches / 4,249 events with 17 dropped pulses and the 'dropped
pulses look like no beam' guard FIRING. Treat its 0.689 with suspicion: it
rests on the same file-count duration extrapolation shown above to run short
for this very sub-run, so the "should not have been bitten" argument is only
as good as a number we already caught being wrong. Do not build a third
mechanism on it before the fixed campaign reports.

**No wrong product shipped. This bug destroys data; it does not corrupt it.**
A bug-bitten segment pairs its triggers with the wrong bunches, so the clock
fit finds nothing and the segment FAILS loudly, writing no file. All 48 did.
The campaign's 170 OK products are sound, and no re-validation of them is
needed.

Structurally, the corrupted δ lands either grid-aligned (a full-looking join
on wrong bunches → no coincidence → the clock fit fails) or off-grid (nothing
clears `MATCH_TOL_S` → an empty join). Neither path writes a passing product.
**Tested at the one point where it was load-bearing for a specific shipped
file**: `run_81/stat090_0001 × 224581`, the segment briefly accused below,
re-run under the fix with a cold cache → **0.94812, identical to the shipped
value to five decimals.**

A short-lived claim that `run_81/stat090_0001 × 224581` was a silently-wrong
product is **withdrawn**. It rested on `joined_bunches (232) ≤
overlap_min·60/1.2 (75)`, and `overlap_min` is derived from an *estimated*
sub-run duration (file count × 47.1 s when no stop time is recorded), so the
test converts an estimator error into a data-quality verdict. The measurement
that settles it is the product's own efficiency: **94.81 % of physics triggers
matched at an accidental rate of 0.052 %**. A wrong-bunch join lands at the
accidental rate — ~7e-4 for ~1,000 candidates over a 75 ms burst, a factor
~1,400 below what that segment shows. Those 232 bunches are the bunches its
triggers belong to; the duration estimate was short.

**Generalise that, because it is the whole lesson twice over:** efficiency
and the +100 µs control ARE the join-correctness test, and they are the same
instrument as `shift_ridge.py` — a ±25 ns coincidence against a control. Any
proposed structural gate (bunch counts, overlap arithmetic, orientation)
either agrees with them or is wrong, and the ones built on estimated
quantities are wrong in the dangerous direction. Two bad inferences this
night came from structural reasoning over an estimated or mislocked index
(the "+41 shift", the "206 s anomaly"); a third nearly condemned good data.

### Retracted below

The section that follows argued the sliver class survived a *proven* join.
The join was not proven; it was corrupted after a correct lock, by code that
ran between the lock and the product. The transferred-δ test "confirmed" a δ
the bootstrap then overwrote. Every probe below is sound as a measurement and
useless as an inference, because all of them conditioned on the same bad
bunch mapping.

## ⚠ RETRACTED — The sliver class is NOT recovered, and it is NOT a join error

Tested 2026-08-12 morning: run_79/stat090_0002 × 224573 re-slimmed with the
majority side's δ transferred (`delta_hint_s = 0.829`, the value its own
×224572 segment fitted at margin 507; the sliver's δ-scan accepts it
cleanly) — **and the clock fit still finds no sharp coincidence**. Since the
same sub-run's pulse_match offset is proven right by the ×224572 fit, the δ
is right, and the bunch mapping follows from both, the sliver failure
survives a fully correct join. Corollary: the per-bunch scan's "random
−1…−24 ms lags" on this segment were measured on CORRECT bunch pairings.

*(Retracted: the δ was right and was then discarded by the bootstrap. The
bunch mapping did not follow from it.)*

**Census (2026-08-12, `sliver_census.py` on the campaign inventory) — the
78 non-OK "sliver" segments are two populations:**

- **42 with a fitted sibling** (the same DREAM sub-run fitted on its other
  side, pinning the shared pulse_match offset): the true mystery class.
  All small — ≤ 402 joined bunches, median 158, overlap median 0.26. The
  pulse_match fix does nothing for these.
- **36 with no fitted sibling** (on 24 sub-runs): the offset was never
  verified, and these include every *large* "sliver" failure (up to 1101
  bunches / 55.6 min) plus all four dark DREAM runs appearing among
  slivers (run_126/128/150/156). Cached v1 offsets betray supercycle
  mislocks: run_132/0007 locked −68.08 s where its run's OK sub-run locked
  +48.32 (−68.08 + 3×39.6 = +50.7); run_139/0007 locked −48.88 vs 0003's
  +67.5 (118.8 apart) — both with the trademark 100 % match / ~4 ms rms.
  **These are whole-sub-run mislocks that happened to straddle a boundary
  and got filed as slivers. They belong in the plain re-run recovery with
  the whole-hour 41.**

The "failed slivers contain n_TOF run STARTS" observation did not survive
the census: mystery-class failures split run_END vs run_START almost
exactly evenly (and OK slivers exist in both orientations). Orientation
carries no information; withdrawn.

Probe results on the mystery class (2026-08-12, run_79/0002 × 224573):

- **(a) cross-processing: FAILS IDENTICALLY on prod_v11** — same wide-scan
  −0.9830 ms z 32, same 2 µs non-sharpening hump, matching bootstrap
  counts. The processing recipe and infrastructure are exonerated; v11 and
  the v12 lineage derive the same hits from the same raw data.
- **(b) per-bunch scan at the verified join: NEGATIVE** — all 277 bunches
  show only envelope artifacts (1 µs-bin peaks at scattered ms lags,
  median −5.1 ms, bounded above by the −0.98 ms hold-off edge); the 49
  "sharpening" bunches are 5 counts over a floor of 0 at scattered ±17 µs
  positions = Poisson flukes. A genuine per-bunch coincidence would be
  ~90 counts in one 20 ns bin (~90σ). **There is no per-bunch time-base
  offset to find; per-bunch recovery is impossible on this axis.**
- **(c) n_TOF bunch bookkeeping: IMMACULATE in the failing window** —
  every one of bunches 1–527 sits exactly on the beam-CSV pulse grid
  (psTime + 0.829 s, rms 0 ms) and the recorded PulseIntensity matches the
  CSV at r = 1.0000, indistinguishable from healthy windows of 224572/3.
  Bunch b really is pulse b. (Trap for the tester: pick the nearest pulse
  to psTime + δ, not psTime — without the 0.829 s offset half the picks
  are wrong everywhere and the test has no power.)

Net: join right, processing right, bunch record right, no per-bunch lag —
DREAM's events in the window have no counterpart in their own bunches' n_TOF
hits at ANY lag within ±80 ms. Last suspect standing: the n_TOF *payload*
(waveform buffers) misassociated with bunch headers at the run transition,
with a possibly DRIFTING shift K(b) (a constant K is excluded by the flat
±200 shift scans; a drifting one would smear them flat).
`cross_bunch_matrix.py` tests this: every DREAM bunch × every n_TOF bunch,
looking for a sharp off-diagonal ridge.

## The fix — IMPLEMENTED 2026-08-12 (this commit)

- `ntof_july_analysis/pulse_match.py`: `select_lock()` enumerates every
  candidate lock, arbitrates near-ties (count margin < 10) by the
  intensity-fluctuation correlation (Fisher-z ≥ 3σ), and raises
  `AmbiguousLock`/`NoLock` instead of silently returning a winner. Caches
  are versioned (v2); pre-fix caches rebuild automatically.
  `accept_offset_s=` is the scan-verified override (recorded as
  `chosen_by: 'verified'`; refuses to override a confident contradicting
  selection). Tests: `ntof_july_analysis/test_pulse_match_locks.py`
  (5 synthetic cases incl. the margin-0-but-correct case, which must RAISE,
  never coin-flip).
- `ntof_dream_merge/bunch_join.py`: `delta_hint_s=` (majority-side δ
  transfer within a sub-run), δ-scan ambiguity guard (margin < 3 without a
  hint raises), join provenance in `DataFrame.attrs`.
- `slim_pipeline/slim.py`: `Segment.delta_hint_s`; `calibration.json` gains
  the `join` block (pulse_match offset / margin / chosen_by / r_sig, δ,
  δ-margin, hint) — recovered and originally-clean segments are permanently
  distinguishable.
- Validation: healthy sub-runs bit-identical (run_79/0001 offset +51.918,
  margin 222, by count); both corrupt hours lock correctly (run_96/0001
  +50.722 by intensity 3.2σ; run_86/0001 +42.32 at 12.1σ = the scan's +20
  pulses); margin-5 run_86/0000 correctly raises pending verification;
  `tests/test_clock_qa.py` 29/29 unchanged.
- Still with the matching-QA session: the `clock_qa` absolute check
  'pulse_match margin adequate' (WARN < 10 / FAIL ≤ 2 unless verified),
  reading the new `calibration.json:join` block.

## Traps recorded on the way

- Consecutive `BunchNumber`s are n_TOF-**recorded** bunches, not PS pulses —
  bunch-index and wall-clock arithmetic diverge; only direct timestamp
  measurement is trustworthy (this trap fired twice in one hour).
- `PSTime` in official merged files can be corrupt (denormals/NaN in
  run224607); use `Time`.
- A nohup on lxplus does not survive the ssh session's 2FA expiry (systemd
  session cleanup); the failure mode is a 0-byte log. Use condor.
- The full kill-list (12 confident wrong models, each killed by a minutes-long
  measurement) is in the campaign report — it is the argument for the
  loud-failure requirement.
