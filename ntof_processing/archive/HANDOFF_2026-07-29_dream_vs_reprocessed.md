> # ⛔ RETIRED — do not build on this
>
> **Superseded by `../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`.** Archived 2026-07-30.
>
> The how-to for running the DREAM comparison against our own processing. It was executed, and both its recipe and its numbers are superseded: the pipeline is now `ntof_dream_merge/match_study/`, and its matcher figures (252 bunches, ±150 ns window, old K and T0) do not describe the current calibration.
>
> The traps it documented — cache isolation per variant, `repair_tflash` off, never `hadd` a run — are still live and are carried in `../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md` §6.
>
> **Read that file.**

---

# Handoff: comparing DREAM against our reprocessed n_TOF data

**STATUS 2026-07-29 (evening): EXECUTED.** Section 9's deliverable and more is
done -- see `FINDINGS_2026-07-29_dream_crosscheck.md` (full-pair matcher
numbers, MM-chamber cross-check, liquid coincidences). This file remains the
how-to; the numbers below marked "252 bunches" are superseded by the full-pair
ones.

**Written 2026-07-29 for a fresh session, human or model, starting cold.**
You do not need to have run anything before. Everything here was executed and
observed in the session that wrote it unless explicitly marked otherwise.

The job: **compare the DREAM Micromegas stream against our own best n_TOF
processing**, rather than against the official n_TOF processing that every
earlier result used.

---

## 1. State in one paragraph

We reprocess n_TOF ourselves. The final candidate UserInput is `v12_liqpileup`;
it has been graded against the official processing and against every intermediate
variant, and it wins on the number that matters -- the DREAM coincidence matcher
goes from **95.3 % to 96.3 % efficient at 0.5 % false** on 252 bunches. The
machinery for joining DREAM to n_TOF already exists and is closed at 100 % on the
bunch join; it was built against the *official* file and needs one flag flipped
and its caches sandboxed to run against ours. **The reference pair is fully
self-contained: n_TOF run 224572 alone covers both real DREAM sub-runs of
run_79.** Nothing is blocking this analysis.

## 2. Read these, in this order

1. **`ntof_dream_merge/HANDOFF_2026-07-27_dream_ntof_matching.md`** -- the
   DREAM<->n_TOF clock chain, the accept bands, the n_TOF internal timing, and
   the gotchas. This is the authority on the *matching*. Its Section 5 is marked
   resolved at the top; believe the marking.
2. **`ntof_processing/STATUS.md`** -- where the reprocessing is, which variant is
   which, and what runs exist where.
3. **`ntof_processing/FINDINGS_2026-07-29_pre_ship_tests.md`** -- the final round
   of tests. Read at minimum its "Headline" table and the three `NEW` sections;
   two of them affect anyone reading the output.
4. **`ntof_processing/REVIEW.md`** Sections 5-6 -- the mistakes that have already
   been made in this repo, so they are not made again. Section 5 is now four
   entries long and three of them are *alignment* bugs. Take that seriously.
5. `ntof_processing/liq_study/FINDINGS_liquids.md` -- only if you care about the
   liquid scintillators. It opens with a WARNING block listing which of its own
   claims have since been corrected.

## 3. What is on the laptop right now

**All of the DREAM side. Most of the n_TOF side.**

| what | where | size | complete? |
|---|---|---|---|
| DREAM run_79 | `/media/dylan/data/x17/beam_july/runs/run_79/` | 2.4 GB | **yes** |
| n_TOF 224572, official | `/media/dylan/data/x17/beam_july/ntof_data/run224572.root` | 26 GB | yes, merged, byte-exact vs EOS |
| n_TOF 224572, **v12** | `/media/dylan/data/x17/ntof_reproc/v12_liqpileup/` | 32 GB when done | **downloading, started 07-29** |
| n_TOF 224572, v11 | `/media/dylan/data/x17/ntof_reproc/v11_pssfit_width/` | 3.8 GB | partials 0001-0002 only |
| n_TOF 224572, v4 | `/media/dylan/data/x17/ntof_reproc/v4_walshapes/` | 3.8 GB | partials 0001-0002 only |
| raw stream1 chunks | `/media/dylan/data/x17/ntof_raw_224572/head_*.bin` | 3.1 GB | 7 chunks, bunches 161-163 and others |

**run_79 has sixteen sub-run directories and only two of them contain data**:
`stat090_0000` (starts 2026-07-26 18:07) and `stat090_0001` (19:07), 1.2 GB each.
`stat090_0002` through `_0015` are 160 kB stubs with empty `combined_hits_root`
and `decoded_root`. Do not go looking for the missing 14 -- there is nothing to
find.

**Both of them live inside n_TOF run 224572**, verified by running the join:

```
stat090_0000 <-> 224572 : 106 127 events, 1012 bunches, range  146-1157
stat090_0001 <-> 224572 : 109 354 events, 1049 bunches, range 1165-2213
```

So **you need exactly one n_TOF run**, and 224572 is the one variant study we
have the most processings of. Runs 224573-224580 are *not* needed for run_79.

### If the download did not finish

16 partials, `run224572_0001.root` .. `_0016.root`, 32.2 GB total:

```bash
cd /media/dylan/data/x17/ntof_reproc/v12_liqpileup
for p in $(seq -w 1 16); do
  [ -s run224572_00$p.root ] && continue
  xrdcp -f root://eospublic.cern.ch//eos/experiment/ntof/data/x17/reproc/\
v12_liqpileup/completed/224572/run224572_00$p.root .
done
```

It needs a valid Kerberos ticket (`kinit dneff@CERN.CH`); `xrdfs ... ls` failing
with "No such file or directory" usually means the ticket expired, not that the
path is wrong. Throughput is ~8 MB/s, so a full run is about 70 minutes.

**Never `hadd` the partials.** See Section 6.

## 4. Which processing to compare against

**Use `v12_liqpileup` for 224572.** For run_79 that is the whole story.

If you ever extend to other runs: 224573-224579 exist only as `prod_v11`
(`/eos/experiment/ntof/data/x17/reproc/prod_v11/<run>/completed/<run>/`,
~31 GB each, 224579 is nearly empty). **That is not a problem for a
wall/plastic analysis**, because v11 and v12 differ *only* in the liquid
configuration -- verified by hit count, identical to the entry in all eight wall
and plastic trees:

```
WALA v12 1 894 802  v11 1 894 802   PSSC v12 9 043 427  v11 9 043 427   (etc.)
```

So mixing v12/224572 with prod_v11/other-runs is safe for anything that does not
touch `LIQ*`. It is *not* safe if you use the liquids. 224580 was never
reprocessed at all.

## 5. How to run the comparison

The matching code defaults to the *official* file and to our laptop-side `tflash`
repair. Both must be changed. `dream_regression.py` already does all of this and
is the working example -- **read it before writing anything new**; the pattern is
about 15 lines:

```python
# 1. Build the DREAM<->bunch join FIRST, against the run as a whole.
from ntof_dream_merge.bunch_join import dream_event_to_bunch
ev = dream_event_to_bunch('run_79', 'stat090_0000', 224572)

# 2. THEN point the reader at the candidate, and sandbox the caches.
import ntof_dream_merge.ntof_io as ntof_io, ntof_dream_merge.tflash_repair as rep
files = sorted(Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
               .glob('run224572_[0-9]*.root'))
ntof_io.ntof_paths = lambda r: files
ntof_io.ntof_path  = lambda r: files[0]
rep.CACHE_DIR = ntof_io.CACHE_DIR = Path(tempfile.mkdtemp())
ntof_io._TFLASH_FIX_CACHE.clear()

# 3. Turn the repair OFF. The whole point is to test the FILE.
import functools
mw.read_bunches = functools.partial(ntof_io.read_bunches, repair_tflash=False)
```

Order matters and the sandboxing is not optional -- see Section 6.

Existing tools, all in `ntof_processing/`:

| tool | what it answers |
|---|---|
| `dream_regression.py <dir> [dream_run] [subrun] [nbunches]` | the matcher efficiency, the false rate, and which leg limits it |
| `quality_metrics.py label=files` | timing and amplitude quality; `--side-lo/--coinc/--late` for robustness |
| `grade_candidate.py`, `compare_fits.py` | flash id, per-arm offsets, fit chi2 (`--max-entries` to cap memory) |
| `liq_study/*.py` | the liquid scintillators specifically |

The run that produced the headline:

```bash
.venv/bin/python ntof_processing/dream_regression.py \
    /media/dylan/data/x17/ntof_reproc/v12_liqpileup run_79 stat090_0000 400
```

With all 16 partials present you can raise `400` and use `stat090_0001` too;
between them the two sub-runs give 2068 bunches, against the 252 the headline
currently rests on.

## 6. Traps, every one of which has already cost someone a session

- **`ssh -K` is mandatory** on lxplus. Without delegated credentials there is no
  AFS token, no condor auth, and `/eos/user/d/dneff` appears not to exist.
- **Never merge a run.** The official merge node dies on the condor 1024 MB
  file-transfer cap, and `hadd` over EOS produces a truncated file *that still
  opens*. Chain the partials.
- **Caches are keyed by run number only.** A reprocessed run224572 read through
  the normal paths silently reuses the official file's bunch index and you get
  plausible, wrong answers. Sandbox `CACHE_DIR`.
- **Build the DREAM<->bunch join BEFORE pointing the reader at a candidate.** The
  join runs off PKUP and the index tree for the *whole* run; a candidate may be a
  few partials. Getting this backwards silently reports "covers none of the
  bunches".
- **`match_window`'s efficiency is not evidence at early times** -- its own
  false-match probability is ~100 % at 1-3 ms. Quote the singles matcher.
- **`repair_tflash` defaults to True.** Against a reprocessed file that is
  testing your repair, not the processing.
- **Heredocs with f-strings** break over ssh on backslash-in-f-string. Write the
  script to a file and rsync it.

## 7. Things about the OUTPUT that will bite an analysis

These are properties of the PSA and the DAQ, not of our UserInput, so they apply
to the official file too. Full numbers in
`FINDINGS_2026-07-29_pre_ship_tests.md`.

- **`satuflag`** (**corrected 2026-07-29 evening** — the paragraph this replaces
  was based on reading the raw samples as unsigned; see
  `FINDINGS_2026-07-29_signed_decoding.md`). On the **liquids it is reliable**:
  matched against the raw waveforms, 119 of 123 clipped runs carry a flagged hit
  within 100 ns, including every physics-time clip. On the **walls it is never
  set**, because wall saturation is a negative undershoot, opposite to the pulse
  direction, and the PSA only tests for rail contact inside a found pulse.
  **A flagged hit must be cut, not used** — its `amp` is a fit extrapolation
  (66 k–832 k against a physical ceiling of 63 800). Do **not** cut on `amp`
  above ~31 000: that was the artifact rail, and it discards ordinary
  half-scale pulses.
- **The ADC wraps under-range rather than clipping.** Baselines sit near the top
  of the unsigned-16-bit range, so a pulse bigger than the baseline reappears
  near 65 535 as a full-scale positive spike. No flat top, so a clipping test
  finds nothing. Sub-percent; for walls and plastics it only happens inside the
  flash.
- **`aslow` is always zero** and `(area - afast)/area` is **not** an n/gamma
  discriminant -- its per-pulse spread is 4-9x the physical band and it drifts a
  factor two with amplitude. Aggregate use only.
- **Liquid `area` is missing its slow component**, in every processing including
  the official one.

## 8. Open questions, and how much they matter

### 8.1 The raw-to-reconstructed time offset  [open, and interesting]

**Not a blocker for this analysis** -- it only affects comparing reconstructed
hits to raw stream1 waveforms.

The PSA guide (`~/x17/ntof_processing/PSA_Guide_20240704.pdf`, "Timing
properties") answers what the branches mean:

- **`tof` is a 30 % constant-fraction arrival time**, not the peak;
- **`peak_tof` is the peak moment** (first highest point, parabola vertex, or
  fitted-Pulse-Shape peak, depending on `AMPLITUDE OPTION`);
- the guide even gives the conversion, `arrival = peak - dt`.

Measured, this checks out internally: `peak_tof - tof` is 1.3 ns median
(p16-p84 = 0.6-2.6 ns), exactly right for a 30 % crossing on a 6 ns FWHM pulse.

**But it does not explain the observation.** Stacking the raw trace on every hit,
the raw pulse peak sits **+26 ns after `peak_tof` on LIQA and +19 ns on LIQD**
(and +28 / +21 against `tof`). The lag is per-detector, stable (p16-p84 = 27-30 ns
on LIQA above 4000 ADC), constant in absolute time -- so it is not a sampling-rate
mismatch -- and there is no per-detector time-offset parameter in the UserInput to
explain it. The bunch identification is certain (raw bunch 161 scores 20 % of its
large isolated peaks against PSA bunch 161, versus a 1.5 % background over all
197 candidates).

So the question for n_TOF is no longer "what does `tof` mark" -- the guide answers
that -- but: **why does the reconstructed time sit ~20-28 ns before the raw
sample-index peak, by a per-detector constant?** Leading guess: the ACQC block
`start` in stream1 and the sample origin the PSA uses are not the same, by a
per-channel amount.

Consequence today: **count-based raw comparisons are fine; per-hit ones are not.**
`deconv_vs_psa.py`'s 0.67 ratio is count-vs-count and survives.
`new_hits_vs_raw.py` is per-hit, does not work, and says so at the top.

### 8.2 The liquid yield  [open, leaning positive]

v12's `STEP SIZE` change gives +14-21 % liquid hits. 95 % of the gain is resolved
shoulders on existing pulses. Everything measurable points to them being real
(rate profile matches, chi2 p50 neutral and p90 better, count still under the raw
resolvable-pulse ceiling), but the per-hit confirmation needs 8.1 first. Quote the
yield with that caveat, not as a confirmed number.

### 8.3 The residual 2.5 % plastic-leg inefficiency  [open, not pulse recognition]

Three very different plastic reconstructions (v8/v10/v11) give identical matcher
efficiency, so this is not a pulse-recognition problem. If it is worth chasing it
is analysis-side: the per-arm discriminator threshold model (`thr['plastic']`),
the `D_PMTS` channel selection, plastic dead time, or genuine detector
inefficiency. **Do not spend more UserInput variants on it.**

### 8.4 LIQB  [open, low priority]

LIQB does not follow the photon-statistics floor the other three liquids do; its
small pulses are narrower and nearly tail-free. Does not affect the shipped
configuration. See the T6 section of the findings file.

## 9. What "done" would look like

The obvious first deliverable is the thing 8.2 and Section 5 set up: **the
matcher, graded on all 2068 bunches of both DREAM sub-runs, official vs v12**,
with the per-leg and per-time-bin breakdown `dream_regression.py` already prints.
That turns the headline from a 252-bunch number into the whole reference pair,
and it needs no new code -- only the finished download.

Beyond that, `ntof_dream_merge/PLAN.md` Phase 5 is the standing plan for what the
merged record is *for*.
