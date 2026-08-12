# waveform_pull — raw n_TOF waveforms around every DREAM trigger

**The slim says where to look; this keeps the raw samples there.** One product
per DREAM sub-run, beside the slim it was built from, holding every
zero-suppressed block of every scintillator channel that overlaps **±5 µs** of a
corrected DREAM prediction — plus the same windows at +100 µs as the accidental
control, plus PKUP.

Sizing, and why the window is wide:
[`../WAVEFORM_PULL_ESTIMATE_2026-08-12.md`](../WAVEFORM_PULL_ESTIMATE_2026-08-12.md).
The time calibration authority remains
[`../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`](../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md);
nothing here changes it.

```bash
# one run, local raw, scratch output
python -m ntof_processing.waveform_pull.pull_run 224572 \
    --slim-base /media/dylan/data/x17/slim_campaign_2026-08-12 \
    --raw-dir /media/dylan/data/x17/ntof_raw_224572 \
    --out /tmp/wf

# does it hold every hit the slim found?
python -m ntof_processing.waveform_pull.verify /tmp/wf/runs/*/*/ntof_wf/*.root
```

## Why this exists at all

The raw stream1 is the only place the samples live — the processed n_TOF output,
ours and n_TOF's, is PSA results only. **The EOS disk copy of the raw expires
about two weeks after the run**, and 27 of the campaign's 83 runs had already
expired by 2026-08-12. Everything is still on CTA tape, but a recall is hours to
days, so this is a one-shot: pull generously now or pay the tape again later.

## What it keeps, and why block-driven

For each DREAM trigger, for each of the twelve scintillator detectors, the
window is

    [t_pred_ns + tflash(det, detn, bunch) - W,  ... + W]

in the PSA's own `tof` space, and **every ZS block that overlaps it is kept
whole, whether or not the PSA found a hit in it**. That is the difference
between this and a hit-driven pull, and it is the whole point: a hit-driven pull
can only ever show you what the PSA already told you about. This one can show
you what it missed.

It costs almost nothing to be block-driven — blocks are ~930 samples and already
~1 µs wide, so at ±25 ns the two are within 30 % of each other — and the product
is then simply *the raw, restricted in time*, with no physics in the cut.

## The time bridge, which is the part that can be silently wrong

`t_pred_ns` is in **t_since_flash** space; a raw block is in **tof** space. The
bridge is `tflash`, per **channel** and per bunch, and it is taken **unrepaired**,
because that is what the slim matched on (`slim.py` reads with
`repair_tflash=False`). Two independent sources are used and compared:

| source | coverage | |
|---|---|---|
| the processed file's `tflash` branch | every (det, detn, bunch) with any hit | authoritative |
| back-solved from the slim, `tof - dt_ns - t_pred` | wherever the slim kept a hit | exact |

**Per channel, not per detector, and that distinction is the whole reason the
cross-check exists.** tflash differs between the eight channels of one wall
detector by up to 23 ns while being constant to 0.00 ns within a channel
(measured on det0/bunch398 of the reference pair). Keyed per detector, the two
sources each pool those eight differently and disagree by 1-15 ns -- which is
exactly how this was found, on run 224615, by a cross-check that had never been
run before the campaign was about to start.

The cross-check reads a SAMPLE: five contiguous stretches of ten bunches
(`--tflash-sample-blocks`). Reading every bunch of all twelve trees costs ~15 min
per segment, as much as the raw read it protects, and a systematic offset shows
up in fifty bunches as plainly as in a thousand. The full read was done once, on
224615: **29,583 channel-bunches cross-checked, zero disagreement**; the sample
reproduces ~2,190 per segment with the same result.

They must agree to 1 ns or the run stops. Whatever neither covers is filled from
the **nearest bunch on the same channel** (marked `source = 2` in the output,
and counted in the log) — not from the run median, which would inherit LIQA/LIQB's
~120 ns outliers and PSS's known ~350 ns per-bunch mis-tags where the neighbour
inherits only the ~4 ns bunch-to-bunch step.

Control hits count towards the back-solve, with their +100 µs removed. **Not**
counting them is how LIQC lost bunches 398 and 399 of the reference pair
entirely, during development: its only hits there are control hits, so it got no
flash time, no window and no waveforms, and the sole symptom was four uncovered
hits in the closure check.

## Output

`<eos>/july_beam/runs/<run>/<subrun>/ntof_wf/`

| file | |
|---|---|
| `ntof_wf_<run>_<subrun>_<ntofrun>.root` | trees `blocks`, `events`, `tflash` |
| `ntof_wf_<...>_provenance.json` | window, raw files read, coverage, counts |

`blocks` — `bunch`, `det` (0-11 as the slim, plus PKUP), `detn` (**the raw ACQC
channel id; the slim's `detn` is the same number**), `tof0`, `n`, `samples`.
Sample *j* of a block sits at `tof = tof0 + j`, and `tof0 = start - 259` for a
zero-suppressed block, `0` for the always-kept flash block. Samples are
**signed** int16.

`events` and `tflash` are copied in from the slim, so a window can be recomputed
from this file alone with no reference back.

Two things the file does not carry, deliberately: the zero-suppression fill code
(`-32768`) is bit-identical to the negative rail, so a filled gap and a genuine
clip differ only by context; and `PRE_SAMPLES = 259` was measured on LIQA alone,
so a per-detector refinement of a nanosecond or two is still open — the product
holds what is needed to measure it (see below).

## Verification

`verify.py` re-derives the answer without reusing anything that built the file:
**every slim hit inside the pulled window must sit inside a kept block.** On the
reference pair that is 100.000 %.

It separates three things that look alike and are not:

- **missing** — no block covers the hit, and the arithmetic is wrong. Fails.
- **edge** — the hit is 1-2 ns past the end of the block holding its pulse. The
  waveform *is* there; this is the PRE_SAMPLES convention, measured at exactly
  2.0 ns on PSSD with zero spread over 3/3. Reported, does not fail.
- **absent bunch** — the raw file set was short. Fails, loudly and separately,
  because a bunch with no raw looks exactly like a quiet detector downstream.

## Running the campaign

`lxplus/` holds the batch machinery. One job = one n_TOF run = every DREAM
sub-run that overlaps it.

```bash
./lxplus/stage.sh                                    # push code, no data
ssh -K lxplus && cd x17wf                            # -K: no token, no EOS
./raw_inventory.sh runs.txt > inventory.csv          # disk vs tape, TODAY
awk -F, 'NR>1 && $2==0 {print $1}' inventory.csv > recall.txt
./recall.sh request recall.txt                       # fire the tape recalls
awk -F, 'NR>1 && $2>0  {print $1}' inventory.csv > ondisk.txt
myschedd bump
condor_submit pull.sub -a 'runs=ondisk.txt'          # start on what is here
./recall.sh check recall.txt                         # poll; then submit those
```

`raw_inventory.sh` must be re-run, not remembered: the disk copies expire
day by day, and a run that was on disk last week is a tape recall today. A
**partial** disk copy is treated as gone — a short file list is exactly how
224526 came to be processed at 13 % coverage.

## Cost

| | |
|---|---|
| raw to read | 0.4-0.9 TB per n_TOF run, 34.9 TB for all 83 |
| decode | ~500 MB/s per core — the pass is **I/O-bound, not CPU-bound** |
| output | ~4 GB per DREAM sub-run at ±5 µs with control, ~390 GB campaign |
| ratio | ~1 % of the raw it is cut from |

## Traps, all paid for

Performance:

- `ak.Array(list_of_ndarrays)` is **60 s per 10 k blocks**. `ak.unflatten` on a
  flat buffer plus counts is **0.001 s** for the same data, and it is also the
  only spelling that works on both awkward versions in play.
- `np.searchsorted` on `tof0` alone is wrong: `tof0` rises within a channel, not
  across the file, and searching the non-monotonic array returns nonsense
  silently — it read 2.9 % hit coverage on a file that was in fact complete.
  The search has to be lexicographic on (channel, tof0).

The worker environment differs from the laptop in five ways, each of which
fails only AFTER the raw read unless it is preflighted:

| | |
|---|---|
| `common` package | `ntof_io` imports `common.beam_july_paths` at module level |
| `$X17_BEAM_JULY` | that module resolves the DREAM tree at import and raises without it |
| `ntof_io.ntof_paths` | resolves against the local staging tree, not EOS — use `slim._bind_ntof`, which also fingerprints the cache per file set |
| **awkward 1.10 / uproot 4.3** on LCG_105 | `ak.contents` does not exist and the `mktree` type APIs disagree with awkward 2; use `ak.unflatten` and create the tree from its first batch |
| `set -u` | LCG's own `setup.sh` references an unbound `COMPILER` and dies instantly |

And two about the batch system:

- **Workers CAN write to EOS**, with `MY.SendCredential = true` — verified
  2026-08-12, cluster 20191171: the worker gets a `dneff@CERN.CH` ticket and both
  xrdcp and FUSE cp succeed. The slim's wrapper says otherwise; that was true of
  its submit file, not of the batch service. It matters because the output
  (~4 GB per sub-run) is far past condor's ~1 GB file-transfer cap, so
  publishing from the worker is the only route open.
- The forwarded credential is valid ~19 h **from submit, not from start**. A job
  that queues a long time wakes with a dead ticket and would fail to publish
  after the whole read, so the wrapper checks `klist -s` in preflight.
