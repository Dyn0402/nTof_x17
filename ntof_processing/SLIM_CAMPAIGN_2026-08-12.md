# n_TOF → DREAM slim campaign, 2026-08-12

**Verdict: the slim is healthy, a quarter of the beam is not in it, and most of
that quarter can be recovered.**
170 segments were fitted and all 170 pass every QA check with no population
outliers; 107 segments failed to fit, and **two independent bugs** were found
during the campaign, both in our own code, which between them account for the
failures: a silent tie-break in `pulse_match` (the whole-hour class) and an
unmasked median in `bunch_join`'s offset bootstrap (the sliver class). Both are
fixed. One lost hour re-joined at 95.47 % efficiency and one lost sliver at
95.56 %, both fleet-typical, through an otherwise unmodified chain.

Both bugs **destroy data rather than corrupting it** — a wrong bunch assignment
leaves no coincidence, so the segment fails loudly and writes nothing. The 170
shipped products are sound and need no re-validation. Nothing is re-slimmed in
production yet.

**And 14 of the 170 accepted segments were decided by a margin of two clusters
or fewer** — they pass every check and were a coin flip away from the same
failure. They should be verified before use.

A recovery campaign on the fixed code was started and stopped at 20 of 83 jobs
when the second bug was found; its partial results are reported under Recovery
below. It will be re-run once in full against a committed fix.

Supersedes the segment inventory of `SLIM_CAMPAIGN_2026-08-09.md`, and
**retracts that campaign's conclusion about the 54 unslimmable sub-runs**
(§5 below).

## What ran

| | |
|---|---|
| n_TOF runs | 79 settled + 4 addendum = 83 condor jobs |
| source | `official/done` merged files, `official/completed` partials for 224667 |
| segments attempted | 291 |
| fitted | 170 |
| failed to fit | 107 |
| skipped (no joined beam) | 14 |
| output | 12 GB, `/afs/cern.ch/work/d/dneff/x17slim/out_<run>/` |

The run list is every run in `campaign_qa/results/settled_runs_2026-08-11.txt`
that overlaps DREAM beam time, taking **FINISHED ∪ STABLE-BUT-UNMERGED** — large
runs never merge, and their complete partial sets are the same processing, so
the wrapper prefers a non-empty merged file and falls back to `completed/`.
All five unmerged runs were checked for completeness against their own `index`
tree before use (100 % of bunches, no internal gaps); only 224667 overlaps beam.

Four runs merged *after* the coverage index was built (224576 at 19:33, nine
minutes after the scan) and went in as an addendum once they had been byte-stable
for 90 minutes and verified complete. 224576 recovered run_79/0009, 0010 and
0011 at ~95.9 % efficiency each.

Both coverage inputs were regenerated first. The DREAM listing was stale at
Aug 8 (282 sub-runs → 342, now reaching Aug 10 09:04) and the n_TOF index times
were rebuilt from the `index` tree of all 669 merged `run224*.root`.

## QA

**170 segments: 170 pass, 0 warn, 0 fail, 0 population outliers.**

Efficiency median 94.94 %, range 93.58–97.37 %. This is the first campaign with
no warnings at all: the four `bunches fitted` warnings of 2026-08-09 were empty
PS pulses, and the empty-pulse filter now removes them at the join.

**Read that number with the failures next to it.** A mis-joined segment fails
its clock fit and writes no file, so QA never sees it. The QA layer is clean
*because* the broken segments are absent — the same shape as the bug described
below, a component that discards what it cannot handle and reports success on
the remainder.

Dashboard: `/media/dylan/data/x17/slim_campaign_2026-08-12/clock_dashboard.html`.

## Exposure — what is missing

Of 233 DREAM sub-runs attempted:

| | sub-runs | share |
|---|---|---|
| fully recovered | 132 | 56.7 % |
| partially recovered | 38 | 16.3 % |
| nothing recovered | 63 | 27.0 % |

In joined bunches — n_TOF-recorded bunches, which is a measurement rather than a
wall-clock proposal, and *not* the same thing as PS pulses (see the method note):
**148,412 matched, 51,331 lost = 25.7 % of attempted beam.** The rate is not
uniform in time — it was 12.3 % at the halfway mark and doubled over the late
beam period. Six DREAM runs have no fitted segment at all: run_126, run_128,
run_135, run_137, run_150, run_156.

Failures split 66 sliver (a sub-run's minority slice across an n_TOF run
boundary) and 41 whole-sub-run. Of 33 straddling sub-runs, **none fitted on both
sides**, 26 fitted on exactly one, and in all 26 the side with more joined
bunches won — so a straddling sub-run reliably loses its minority slice.

## Root cause 1 of 2 — the whole-hour class

Found jointly with the parallel investigation session; the mechanism is theirs,
the campaign-scale measurements are this one's. This accounts for the 41
whole-sub-run failures; the 66 boundary slivers have a second, unrelated cause,
described under Recovery.

`pulse_match.match_subrun` scans ±120 s for the burst-to-pulse offset and scores
candidates by matched-cluster count. Candidate locks one supercycle apart **tie**
on that count, and the strictly-greater tie-break keeps the first, which is the
most negative. Since the scan range is 120 s ≈ 3 × 39.6 s, the error is
consistently −3 supercycles. Two shift scans confirm it: run_96/0001 × 224597
locks +26 bunches off (S/N 1273), run_86/0001 × 224583 locks +20 (S/N 1708).
Both convert to the same 118.8 s through their own window's mean pulse spacing.

**The bug is the silence, not the tie.** The objective genuinely cannot
distinguish these alignments, and it returns one anyway with perfect-looking
match statistics (100 %, 5.7 ms rms) at the wrong offset. The fix is to score
the intensity *sequence* — which does discriminate — and, non-negotiably, to
fail loudly on an ambiguous winner.

### Some of the accepted data was correct by luck

The lock margin — best minus runner-up cluster count — was measured across 211
sub-runs after the campaign. Failures sit at margin 0 in 35 of 41 cases and
never above 8. Fitted segments run p10 = 3, median 23, max 380 — **but 14 of
them sit at margin ≤ 2, and two at margin 0.** Those passed every QA check and
are in the campaign's 170, and they were one cluster away from locking
elsewhere. They are not known to be wrong; they are known not to have been
decided by evidence.

**Recommendation: verify those 14 by shift scan before the products are used**,
and treat the margin as a first-class quality number rather than an internal
detail. The intensity-fluctuation term is validated in the wild by the 43.2 s
row above — r = 0.925 against 0.508 would have picked the truth outright — but
it cannot rank healthy locks, because for a correct lock the runner-up's r is
often nearly equal (the intensities repeat with the schedule). Count, plus the
r term, plus cross-sub-run continuity, plus loud failure, is the full design.

Proposed `clock_qa` gate, for the pipeline owner's decision: WARN below margin
10, FAIL at ≤ 2 unless the segment carries a shift-scan verification or an
r-term separation, and mandatory `join_shift` provenance for anything below the
WARN line. Margin alone cannot hard-FAIL, because two segments that fitted
correctly sit at 0.

### Constraints this campaign contributed

Each of these killed a candidate model:

- **No statistics floor.** Smallest fitted segment 34 joined bunches; largest
  failure 1,126 bunches at normal candidate density. 76 fitted segments have
  fewer bunches than the largest failure.
- **No excess triggers in failing hours.** Median excess over healthy peers of
  the same n_TOF run: −0.2 %. A noise epoch able to mis-tag ~100 % of bursts
  needs ≳470 Hz, which would show as tens of percent.
- **No pre-flash region.** A burst window begins at the flash and runs 75.5 ms;
  0 of 94,124 non-flash triggers precede their own flash, first physics trigger
  at a hard 0.993 ms edge.
- **No partially-corrupted segments.** 1 bunch below 5 % match efficiency out of
  73,118 across all fitted segments — corruption is all-or-nothing per segment.
- **No time clustering.** 37 same-outcome blocks against 32.9 expected under
  independence; adjacent sub-runs against the *same* n_TOF run alternate
  pass/fail within the hour.
- **The beam schedule does not select the failing hour.** run_86/0000 fitted at
  5.93 s spacing and periodicity 7.5; run_86/0001 failed at 5.91 s and 7.6.

**The universal −0.982 ms association is the burst's own structure.** The
flash-to-first-trigger edge measures 0.9927 ms in every healthy segment, and the
feature sits under successful fits at the same lag and significance. It is not
diagnostic of anything.

## §5 — retraction

`SLIM_CAMPAIGN_2026-08-09.md` concluded that 54 DREAM sub-runs (~23 % of beam)
"are not missing n_TOF data — whatever DREAM was triggering on is only loosely
associated," on the evidence that the −0.982 ms association was broad and did
not sharpen. **That conclusion is withdrawn.** The feature is present under
healthy fits, so its presence carries no information about a segment, and the
inference drawn from it read a universal artifact as a per-segment property. The
width measurement stands as a fact about the feature; nothing follows from it
about the affected hours.

Those hours are not loosely associated. Their coincidence is sharp and sits at a
wrong bunch assignment. Three sub-runs the Aug-9 campaign could not do are
already recovered here purely because more n_TOF runs are merged — some of the
"54" were never unmatchable, only uncovered.

## Recovery

### Pre-flight on both fixes, with predictions registered in advance

A first recovery run on the `pulse_match` fix alone (cluster 17840308) was
stopped at 20 of 83 jobs when the `bunch_join` bug came to light — finishing
would have produced sliver results that had to be redone. It is superseded by a
five-run pre-flight on both committed fixes, whose purpose was to catch a
systematic failure before spending a campaign, **not** to estimate a recovery
rate.

Each target's burst overhang was computed and sent to the parallel session
*before* its job ran, so the threshold could not be placed afterwards:

| segment | burst overhang | predicted | result |
|---|---|---|---|
| run_79/0005 × 224574 | 0.714 | recover | **OK, 95.76 %** |
| run_79/0008 × 224576 | 0.939 | recover | **OK, 96.01 %** |
| run_79/0011 × 224577 | 0.567 | recover | **OK, 96.12 %** |
| run_81/0001 × 224580 | 0.691 | recover | **OK, 94.72 %** |
| run_81/0001 × 224581 | 0.335 | clean, reproduce | **OK, 0.94812 = shipped** |

Five for five. run_79/0011 is the one that mattered: at 0.567 it is the
thinnest margin over the threshold, and it recovered.

Over the 23 comparable segments:

| transition | n | |
|---|---|---|
| OK → OK | 10 | efficiency delta **+0.00000** on every one |
| **FAILED → OK** | **5** | recovered, no manual input |
| FAILED → AmbiguousLock | 5 | honest refusals of already-lost segments |
| OK → AmbiguousLock | 3 | all run_82 short scan sub-runs |
| OK → FAILED | 0 | no regressions |

Ten exact reproductions beside five recoveries **in the same jobs** is what
separates a correct fix from a permissive one — a fix that merely joined more
aggressively would have moved the reproductions too.

One recovery was not predicted and is the most telling: **run_79/stat090_0014 ×
224577 at 94.64 %**. That is the sub-run whose *both* sides failed in the first
campaign, the 57-minute hole originally offered as evidence that the sliver
class was an n_TOF-side problem. It was our own bootstrap.

**The three OK → AmbiguousLock are not losses.** Those segments had products,
and by the efficiency argument above a product at fleet-typical efficiency has
a correct join. The guard is not saying they were wrong; it is saying it cannot
self-verify them, which is true and is what it was built for. All three are
run_82 short scan sub-runs, below the ~200-cluster floor where arbitration has
no power in principle and a bunch-shift scan is the standard route. They are to
be scanned, re-issued with `accept_offset_s` so the provenance records
`chosen_by = 'verified'`, and counted on their own line as recovered-via-scan —
not netted out of the headline. They must **not** be reinstated on the strength
of their old efficiency alone: validating a lock by the self-consistency of the
join it produced is the exact circularity that cost this campaign.

### The whole-hour recovery, demonstrated

**Demonstrated, at fleet quality.** run_96/stat090_0001 × 224597 — a whole hour
that this campaign lost completely — was re-joined with bunch shift +26 and then
run through the **unmodified** standard chain: bootstrap S/N 1319,
K = 1.121347e-4, T0 = −279.66 ns, arm offsets −15.81/+8.69/+2.18/−0.49 ns,
787 of 788 bunches individually fitted, da RMS 8.36 ns, **efficiency 95.47 %
(held-out 95.43 %), accidental 0.065 %, purity 99.93 %**. Against this
campaign's fitted band (median 94.94 %, range 93.58–97.37 %) the recovered hour
is indistinguishable from a healthy one. One integer, no pipeline changes.

That certifies the **whole-hour class — 41 segments** — as recoverable, and each
segment's shift is **measurable by a single unambiguous scan, demonstrated on
8 of 8 scanned failures**: every one shows at most one peak within ±200, at
S/N 541–1822 where a peak exists at all.

**Scan; do not look the shift up.** Five verification scans tested the predicted
table. The three rows with a validated 39.6 s supercycle came back within one
count (predicted 20.4 / 23.4 / 23.1, measured +21 / +24 / +23). The 43.2 s row
did not: run_102/0003 × 224607 was predicted +24 and measured **+41**.

The prediction failed for a reason worth stating, because it was not the
harmonic effect the table warned about — the comb detector's 43.2 s was the
correct period. The formula was wrong. It assumed the error is the most negative
schedule multiple that fits inside the ±120 s scan, i.e. `floor(120/u) × u`.
In fact the scan range constrains each *lock*, not the distance between them:
that segment's true lock is at +60.28 s and the chosen one at −69.32 s, both
comfortably inside ±120 s, and the error between them is 129.6 s — larger than
the formula can ever return. Add the unreliable bunches-to-seconds conversion on
top and the table is a rough guide at best. Which multiple ties is a property of
the count landscape, not of the scan edge, and nothing short of the scan
predicts it.

**Recommended re-slim recipe**: per failed whole-hour segment, run the shift
scan — it is cheap, unambiguous and self-verifying — apply the found shift at
the join, record `join_shift` and the lock margin in the products, then run the
standard chain. Use `shift_predictions.txt` as a cross-check on the scan, never
as the source of truth. 41 scans is about one condor evening.

**The 43.2 s family resolves to the same mechanism.** Reading that segment's
`pulse_match` lock structure directly, run_102/0003 × 224607 is a dead
607-against-607 tie between −69.32 s (intensity-fluctuation r = 0.508, the one
chosen, because it is more negative) and +60.28 s (r = 0.925, the truth). The
error is −129.6 s = **3 × 43.2 s**. So the statement holds in its general form:
every scanned failure is `pulse_match` taking a more-negative schedule-multiple
lock on a count tie, three units of whatever that window's recorded-bunch
periodicity is — 118.8 s in a 39.6 s window, 129.6 s in a 43.2 s one.

An intermediate measurement of 206 s for this segment was withdrawn: it was
taken across the *mislocked* bunch range, which is the bunch-index-versus-time
trap described in the method note, hit a second time within the hour.

### The sliver class was a second bug, in our own join

The **66 sliver failures** were first attributed to a separate, unexplained
mechanism — their shift scans are flat within ±200, so no integer correction
applies. That was wrong, and the cause was found on 2026-08-12: a one-line
defect in `bunch_join.dream_event_to_bunch`.

The offset bootstrap took the median of `epoch − psTime` over **all** bursts.
A burst whose pulse is not in this n_TOF run still gets assigned the nearest
one, clipped to the first or last in the list, and contributes garbage. On a
segment that overhangs its n_TOF run — which is what a boundary sliver *is* —
those bursts are the majority, so the median walks the offset by roughly the
overhang. The docstring already said the offset is defined by the matched
pairs; the code never applied the mask. Measured on run_79/stat090_0002 ×
224573 with 77 % overhang: the grid scan locks correctly at +0.790 s, the
median returns **−957.971 s**, and the join ships every burst paired with a
pulse ~280 later than its own.

It survived a whole campaign because it is invisible to every consistency
check: both sides sit on the 1.2 s pulse grid, so residuals stay at 8 ms, every
bunch lands on the grid, and the intensity cross-check is circular (it compares
n_TOF against the beam CSV at the same wrong bunch, never DREAM against n_TOF).
It was found by correlating every DREAM bunch against every n_TOF bunch, which
showed a sharp 20 ns ridge at bunch *b*−280 in 129 of 130 cases.

With the fix, the exemplar goes from no peak at all to **95.56 % efficiency,
247 of 247 bunches fitted, S/N 1769** — an ordinary healthy segment.

**This bug destroys data rather than corrupting it**, and that is the load-
bearing consequence. A wrong bunch assignment means no coincidence, so the
clock fit finds nothing and the segment fails loudly and writes no file. All 48
affected segments did exactly that. **The 170 shipped products are therefore
sound and need no re-validation** — there is no silently-corrupted class to
hunt. The one product briefly suspected of it, run_81/stat090_0001 × 224581,
was cleared by the same instrument: it matches 94.81 % of triggers at a
0.052 % accidental rate, and a mis-paired segment sits at the accidental rate,
a factor of ~1,400 lower. Its apparent anomaly was the coverage map's
file-count duration estimate being short, not its join being wrong.

**The same defect shape was searched for elsewhere and not found.** Every
clipped nearest-neighbour lookup feeding a median or a fit across
`ntof_dream_merge/`, `pulse_match` and `slim_pipeline` was checked:
`pulse_match._refine` already medians over its matched mask and guards on at
least three survivors, and `bunch_join`'s other median (`resid_mad_s`) was
always masked by `ok`. The pattern is written correctly everywhere else it
appears, so this was a single-site defect rather than a habit.

Predictor for what the fix recovers: the bug can only bite when the overhang
exceeds half the sub-run. Of the attempted segments, 48 failures sit below
overlap fraction 0.5 against a single OK, while above 0.5 the split is 18
failures to 54 OK. So those **48 are the candidates for recovery**; the 18
failures above 0.5 are a different problem and are not promised.

**A wrong shift cannot fake a good recovery, for this class.** The ±200 scans
find exactly one peak — S/N 1273 and 1708 at the true shift, ≤2.5 everywhere
else including every other schedule multiple. The bunch-shift degeneracy and the
time-lock degeneracy are different things, and only the latter ties. So a
re-join at a wrong integer fails loudly rather than producing a plausible clock.

That is a bound on this failure class, not a general guarantee, so **before any
bulk re-slim the slim should record its provenance**: `join_shift` (0 for
originally-clean segments), the `pulse_match` lock margin, and the winning
lock's intensity-fluctuation score, written into `calibration.json`. Today a
recovered segment and an originally-clean one are indistinguishable in the
products, and retrofitting provenance after a bulk re-slim is precisely the kind
of silent gap this campaign was about.

`shift_predictions.txt` holds the per-segment period, predicted offset and shift
for all 35 whole-hour failures. Treat it as a cross-check, not a lookup: the
five 39.6 s rows agree with their scans to within one count, and the one row
tested outside that family did not, for the reason above. Sliver failures are a
separate instantiation — their shift scans are flat within ±200 — and will not
come back from a shift correction.

## How to tell a complete table from a truncated one

Every number in this report comes from a **single read** of the campaign
products, taken after the drain was confirmed by **counting completed job
summaries (83 of 83), not by an empty `condor_q`**. Both halves matter.

An empty queue means the schedd is down about as often as it means the jobs
finished — this project walked into that on 2026-08-09 and it is in the condor
notes. If it happens mid-wait, the aggregation runs over whatever summaries
exist and produces a partial table that looks final and sums correctly. That is
the same shape as both bugs this campaign was about: a component reporting
success over the subset it can see.

Interim figures quoted anywhere carry their job count ("36 jobs / 122
segments") because a running campaign is a moving target — two numbers read
minutes apart will disagree without either being wrong, which happened once
here and looked like an arithmetic error until the two reads were separated.

**"0 held" is not evidence that staging is healthy**, and it was quoted as such
twice during this campaign. `max_retries = 0` means a slow or failed 30 GB
`xrdcp` from EOS exits nonzero and the job is FAILED, never HELD — so the held
count can only ever exonerate condor-side resource violations, which are not
the part of the job most likely to break. The check that does work is the same
one that verifies the drain: a wrapper dying at the copy writes no
`slim_summary`, so **summaries against job directories** catches it. At 81
completed jobs that ratio was 81/81, which is the first actual evidence of
healthy staging in the campaign rather than an inference from a queue column
that structurally could not have shown otherwise.

The pattern behind all three of these — drain by summary count, staging by
summary count, duration by measurement rather than by the coverage map's
extrapolation — is one rule: **count what the work produced, not what a
bookkeeper says about it.** Every silent failure in this campaign, in the code
and in the operations both, was a component reporting success over the subset
it could see.

Operational note for the next campaign: `max_materialize = 10` in `slim.sub`
cost roughly 5.5 h for 83 jobs while every materialised job matched instantly
(0 idle at every check), so the pool had capacity throughout. The cap cannot be
raised on a running cluster — it is frozen in the spooled submit digest, and
`condor_qedit JobMaterializeLimit` reports success and updates the displayed
LIMIT while the factory ignores it. Set it from measured EOS staging bandwidth,
or drop it; it is not needed for condor-side resources, which are matched per
job and never metered as a fleet total.

## Artifacts

- `inventory.csv` — one row per attempted segment (status, efficiency, overlap
  fraction, joined bunches/events, dropped pulses, guard, sliver/whole)
- `shift_predictions.txt` — per-segment supercycle, predicted offset and shift
- `clockqa_final.log`, `clock_dashboard.html`
- analysis scripts on `/afs/cern.ch/work/d/dneff/x17slim`: `inventory.py`,
  `exposure.py`, `join_axis.py`, `floor_axis.py`, `rate_test.py`,
  `gate_shape.py`, `perbunch_match.py`, `guard_defined.py`, `straddle.py`,
  `spacing_risk.py`, `schedule_period.py`, `window_grid.py`, `predict_shift.py`,
  `status_audit.py`

## Method note

Four of this session's own analyses produced confident wrong numbers before
being caught, and all four were classification or parsing errors rather than
physics:

1. `"does not overlap" in error` matched the *explanatory sentence* of the
   standard failure message, filing 69 of 75 Aug-9 failures as skips.
2. A log-text pass/fail rule let a later "efficiency"/"wrote" line re-pass a
   segment after its RuntimeError, marking 17 failures as successes.
3. An autocorrelation in pulse *index* found nothing, because a 39.6 s
   supercycle is 9.08 pulses — the wrong variable, not a null result.
4. A supercycle detector scoring "fraction of peaks that are integer multiples"
   returned nothing for the two windows with known answers and small spurious
   units elsewhere.
5. Recorded bunches were called PS pulses throughout, so every spacing and
   period here is the schedule as n_TOF recorded it (see the traps below).
6. The shift formula `floor(120/u) × u` assumed the scan range bounds the
   *distance between* two locks; it bounds each lock separately. It cannot
   return the 129.6 s error it was asked to predict, and the failure was
   initially misread as the harmonic effect the table did warn about.

Status comes from the slim summaries now, never from log text; `status_audit.py`
checks the two agree. Every one of these died to a measurement that took
minutes, which is the same argument as the loud-failure requirement above: a
component that cannot distinguish must say so rather than return its best guess.

Two traps for anyone repeating this work:

- **Consecutive `BunchNumber`s are n_TOF-recorded bunches, not consecutive PS
  pulses.** A "mean spacing" computed from them mixes in the gaps where the
  beam went to other users, so bunch-index arithmetic and wall-clock arithmetic
  diverge — which is exactly how the +41 prediction went wrong. Only a direct
  elapsed-time measurement between the bunches concerned is trustworthy. Every
  spacing and supercycle number in this report is measured over recorded
  bunches and inherits that caveat; the 39.6 s ladder survives it because it
  was confirmed by scan, and the other families did not.
- **`PSTime` is corrupt in at least one official merged file.** In
  `done/run224607.root` the finite entries are denormal (~1e-38) and the rest
  NaN. Use `Date`/`Time` instead, as everything in this campaign does.
