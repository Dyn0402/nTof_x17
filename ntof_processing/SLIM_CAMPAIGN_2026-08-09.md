# The first full slim campaign — 2026-08-09

**119 of 202 segments slimmed, covering 116 of 170 DREAM sub-runs. The other 54
are not missing n_TOF data: they contain a real but ~6 µs wide association that
cannot support a ±25 ns match, and that is a DAQ question rather than a fit
parameter.**

Design and feasibility: [`SLIM_FEASIBILITY_2026-08-08.md`](SLIM_FEASIBILITY_2026-08-08.md).
Pipeline: [`slim_pipeline/README.md`](slim_pipeline/README.md).
Live QA: <https://dylan-neff.web.cern.ch/notes/ntof-dream-clock-qa.html>.

---

## 1. What ran

| | |
|---|---|
| condor cluster | 3919396, 59 jobs, 10 concurrent, 2.5 h wall |
| input | n_TOF `official/done`, the 5–7 Aug pass (our v12 content under the name v4) |
| slim window | ±1 µs (widened from ±150 ns the same day, §3) |
| output | `~/x17slim/out_<ntofrun>/runs/<run>/<subrun>/ntof_hits/` on lxplus |
| products | 7.1 GB, 209,666,036 n_TOF hits against 8,448,205 DREAM triggers |

Every job exited 0. Nothing was lost; the failures below are all *segments*
inside successful jobs.

## 2. Results

| outcome | segments |
|---|---|
| OK | 119 |
| FAILED | 74 |
| SKIPPED_LOW_JOIN | 9 |

Segments are the wrong unit for judging coverage, because one DREAM sub-run is
deliberately proposed against every n_TOF run it might overlap. In sub-runs:

| | |
|---|---|
| covered by ≥ 1 successful segment | **116 of 170** |
| not covered at all | **54** (23 % of beam minutes) |
| failed pairings that were covered elsewhere | 16 (harmless — the generous proposal working) |

Quality of the 119, from `clock_qa.py`: **0 FAIL**, median efficiency 95.07 %
(93.58–97.31), median residual RMS 6.60 ns, median accidental 0.05 %. One fleet
outlier (`run_78/stat090_lat051_c0_0005`, arm C offset −10.3 ns against a fleet
median of +1.5, z 5.9) on a 3-minute lat-scan segment.

**The single most useful number:** coarse-search S/N is ≥ 801 on every successful
segment and ≤ 3.3 on every failure. There is no grey zone, so "did the clock fit
work" is a binary question and not a judgement call.

## 3. Three defects the campaign exposed, in ascending order of quietness

Each was invisible to the checks that existed before it. That is the argument
for the QA layer, not a claim that the QA layer is now complete.

### 3.1 The clock fit was seeded, and validated by luck

`fit_global` started from a hard-coded `T0 = −250 ns` and only accepts
candidates within ±250 ns of where it is looking. Right for run_79
(fitted −252.6), wrong for run_77 (+109.5). Seven of nine run_77 segments died;
the two that lived started from 312 candidates against a hard floor of 200.

Fixed by `clockfit.bootstrap`: histogram every candidate within ±50 µs, take the
peak, require significance over the floor beside it. Reference reproduces to 4
decimals (K to 1e-13 relative); 224570 went 0/3 → 3/3 and 224571 1/9 → 7/11.

**Lesson worth keeping:** the pipeline was validated on one pair, and one pair
cannot exercise a per-pair constant. A fit seeded near its answer validates
beautifully and proves nothing.

### 3.2 ±150 ns truncated a quarter of the plastic yield

The inherited window check asked whether the kept `dt` was still *rising* at the
edge. A coincidence peak *wider* than the window falls away slowly and passes
it: on the reference it returned 0.94, "flat, window is wide enough", while 23 %
of the plastic was being cut.

`slim_study/pss_tail_probe.py` measured it properly at ±10 µs: the plastic tail
is real, one-sided late by 22×, smooth with no discrete echoes, and runs to
microseconds. Window widened to **±1 µs**, which holds 93 % of the PSS excess
lying within ±2 µs at 2.24× the hits. At that window the plastics are contained
(edge/peak 0.024, from 0.234).

Corrected in passing: the liquids do **not** need a wide window — their apparent
tail is symmetric (3,701 early against 2,224 late), i.e. subtraction noise. An
earlier reading of the integral scan called it real.

### 3.3 A ~1 ms association hid 54 sub-runs

28 failures had full 60-minute overlap, a healthy join (~108 k events, ~1,250
bunches, ~1.1 M candidates at a normal rate) and a flat residual histogram. Ruled
out in order: too-few-events (they are full length), wrong pairing (bunch ranges
match exactly and index timestamps are contiguous to 29 s), bunch-assignment
offset (`segment_diagnose.py` scanned shifts −5…+5: flat at every one).

`segment_diagnose.py` then cross-correlated DREAM triggers against n_TOF
candidates by FFT over the whole 80 ms burst — the only way to see every lag at
once — and found the same answer on two independent pairs three days apart:

| | lag | robust z |
|---|---|---|
| run_118/stat090_0003 × 224642 | −0.9830 ms | 22.9 |
| run_132/stat090_0005 × 224662 | −0.9820 ms | 21.2 |

**This was first reported as a recoverable flash-reference offset. That was
wrong, and the refinement ladder is what corrected it.** A fixed offset sharpens
when you re-centre on it; this does not:

```
2000 ns bins   excess 3,363   34.7 sigma   PASS
 500 ns bins   excess   271    5.0 sigma   fail
```

Subdividing 2 µs into 500 ns quarters would give ~840 per bin if the excess were
contained. It gives 271, so the excess is spread over ~12 such bins — **about
6 µs wide**, against ~6 ns for a real coincidence.

**Conclusion:** those hours are not missing n_TOF data. Whatever DREAM was
triggering on is only loosely associated with the n_TOF hits. No window and no
refinement recovers a nanosecond match from a microsecond-wide association.

The failures cluster in time, which is the clue to follow:

```
08-01 23:30 -> 08-02 11:35   13 sub-runs   (run_118)
08-03 23:34 -> 08-04 03:36    5 sub-runs   (run_132)
08-05 03:04 -> 08-05 06:06    4 sub-runs   (run_139)
```

plus scattered singletons. A 12-hour block starting at 23:30 has the shape of an
overnight intervention. **Next step is the DAQ logs for those windows, not more
searching.**

## 4. What guards this now

| tool | what it answers |
|---|---|
| `slim_pipeline/clock_qa.py` | 13 absolute checks on one segment; NA rather than a silent pass on old files |
| `slim_pipeline/dashboard/` | robust z against the fleet — catches a segment that passes every absolute check but sits 380 ns from its peers |
| `slim_pipeline/tests/test_clock_qa.py` | 19 injected defects, asserting each check fires |
| `slim_pipeline/segment_diagnose.py` | separates too-few-events / wrong pairing / bunch offset / offset-out-of-range, and measures the correlation width |
| `slim_pipeline/lxplus/campaign_status.py` | segment coverage against the proposal, and refuses to treat a mixed-window tree as one dataset |

The test file has already earned its place twice: it caught `uproot.arrays()`
returning a dict for some files and a structured array for others, and it
exposed the useless window check in §3.2.

Two acceptance criteria were wrong on their own terms and are now fixed:

* **bootstrap** tested peak/floor *ratio*. A ratio only means anything for a peak
  narrower than the bin — the ms-offset correlation gives ratio 1.4 and 35 σ at
  once, and the ratio test discarded it. Now significance (≥ 8 σ, against ~4 σ
  expected from the tallest of 5,000 noise bins).
* **containment** flagged 39 % of segments on a LIQ edge/peak ratio whose core
  excess is 37× smaller than PSS's, i.e. mostly Poisson noise. Now requires the
  edge excess itself to be > 3 σ.

## 5. Condor lessons, all learned the hard way

* `-queue` on the command line is a hard parse error against an in-file `queue`
  statement. The submit file now ends without one.
* `transfer_input_files = dir/` with a trailing slash flattens the package tree.
* A trailing `# comment` on a value line is parsed as part of the expression.
* Each job must remap `out` to `out_<run>`; 59 jobs merging into one directory is
  a needless race.
* `max_retries = 0`. Every failure here is deterministic, and a retry costs
  another 30 GB copy to reach the identical error — this is what dropped a stale
  ±150 ns file into a ±1 µs tree mid-campaign.
* **An empty `condor_q` means the schedd is down, not that the jobs finished.**
  bigbird27 died at 13:16 for 1.7 h and a poller read the silence as success.
  Only believe an empty queue when `condor_q` also exits 0.

## 6. State, and what is next

**Done:** pipeline, QA, tests, dashboard published, 119 segments validated on
lxplus.

**Not done:**

1. **Publish the 119 to EOS.** They sit in `~/x17slim/out_*`; `./publish_to_eos.sh
   out_*` pushes them. Recommended now rather than waiting for the 54 — the
   dashboard states the gap plainly, and holding them back does not bring the
   rest closer.
2. **The 54 sub-runs** need the DAQ-side answer for the three time blocks above.
3. **The plastic late tail** is contained by the ±1 µs window but still
   unexplained. **Do not quote a plastic hit yield until it is.**
4. **The 41 runs n_TOF still owes us** —
   [`NTOF_REPROCESSING_REQUEST_2026-08-08.md`](NTOF_REPROCESSING_REQUEST_2026-08-08.md),
   unrelated to the 54 above.
5. `STATUS.md` has no entry for this campaign: another session held uncommitted
   changes to it at the time and it was not mine to touch.

## 7. Addendum, later on 2026-08-09: the QA sweep of the 119

Aggregating the campaign's `clock_qa.json` records (0 FAIL, 46 WARN) resolved
all three open threads and reworked the QA around the plastic ringing:

* **The 45 containment WARNs were a check artifact, all LIQ.** Fleet-total
  edge excess: early +14,491 vs late +15,623 — symmetric, which no truncated
  coincidence is. It is the +100 µs control mis-stating the local floor
  slightly (a flat pedestal, common to both edges), passing 3 σ on big
  segments. The check is now sided: it flags edge *asymmetry*, which a
  pedestal cancels out of. (PSS: early −1,253 vs late +659,840 — the ringing,
  one-sided as measured.)
* **The one fleet outlier (run_78/stat090_lat051_c0_0005, arm C) was a fit
  defect, now fixed.** The per-arm offsets were the intercepts of free
  per-arm line fits, extrapolated to t = 0 from data starting at 0.1 ms; on a
  3-minute segment the slope noise displaced them by up to 12 ns, and the
  matched residuals sat unimodally at +12 ns (C) / −8 ns (D) to prove it. On
  lat051-sized slices of the reference the old estimator scatters 2–5 ns RMS
  (worst 9–18 ns); the refined median that replaced it, 1.3–1.8 ns. A new
  check ('per-arm residuals centred') FAILs the segment retroactively —
  nothing global could see it. 224571 re-slimmed with the fix: all 7 of its
  segments now PASS every check (lat051 arm C −10.3 → +0.7 ns, residuals
  centred to ±1.8 ns).
* **The ringing (`pss_ringing/`, section 3 above's "unexplained tail") is now
  in the QA and the format**: `shadow_amp`/`shadow_dt` branches in new slims,
  'PSS late tail is ringing' and 'plastic primary within accept' checks in
  `clock_qa.py`, and the fleet re-judged with them. The ±25 ns slice is safe:
  the largest plastic pulse per trigger lands inside it for 91.2–93.8 % of
  matched triggers on every segment, and the late tail is ringing at
  99.2–104.4 %, not mis-handled yield.

**Fleet after the rework and the 224571 refit: 112 PASS, 4 WARN (the known
bunches-fitted quartet), 0 FAIL, 0 outliers.** Dashboard republished. Note
for the pending EOS publication (§6 item 1): the 116 on lxplus predate the
`shadow_amp`/`shadow_dt` branches (QA covers them via the in-window
fallback); re-slimming the campaign (~2.5 h wall) would bake the branches in,
or publish as-is and let analyses recompute in-window.
