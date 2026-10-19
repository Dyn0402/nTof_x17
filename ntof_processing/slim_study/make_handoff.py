#!/usr/bin/env python3
"""Build the n_TOF-facing request: which runs are missing, and what they block.

Emits
  ../NTOF_REPROCESSING_REQUEST_<date>.md   the readable handoff
  ntof_reprocessing_request.html           the same, as a standalone page for
                                           the site (handoff_html.py)
  missing_runs_<date>.csv                  the same list, machine-readable

Inputs are the cached listings under `coverage_inputs/` (see coverage_map.py)
plus `needed_runs_raw.txt` (run, n_raw_files, raw_bytes), which is
    ssh lxplus 'D=/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement;
                for r in <runs>; do echo "$r $(ls $D/$r/stream1 | wc -l) \
                $(du -sb $D/$r/stream1 | cut -f1)"; done'
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import coverage_map as cm
import handoff_html
import why_skipped

HERE = Path(__file__).resolve().parent
DATA = HERE / 'coverage_inputs'
TODAY = '2026-08-08'

# The reprocessing pass covered this contiguous run range; anything inside it
# that is absent from done/ is a gap. Derived, never typed -- an earlier
# hand-copied list of the raw-gone gaps was one run out and made the arithmetic
# in the handoff disagree with itself.
PASS_LO, PASS_HI = 224300, 224687


def main() -> int:
    v12 = cm._spans(DATA / 'ntof_index_times.txt', shift_s=cm.INDEX_LOCAL_SHIFT_S)
    raw = cm._spans(DATA / 'ntof_raw_times.txt', skip_short=False,
                    shift_s=-cm.RAW_WRITE_LAG_S)
    subs, _ = cm.load_dream(DATA / 'dream_eos_subruns.txt',
                            DATA / 'dream_daq_subruns.txt')
    v12set = {r for r, _, _ in v12}
    rawspan = {r: (a, b) for r, a, b in raw}

    sizes = {}
    p = DATA / 'needed_runs_raw.txt'
    if p.exists():
        for ln in p.read_text().splitlines():
            f = ln.split()
            if len(f) == 3:
                sizes[int(f[0])] = (int(f[1]), int(f[2]))

    # Every run that is NOT in done/ needs a time window before we can say what
    # it blocks. Three sources, in order of trust:
    #   index  -- exact, but only exists for a run that WAS processed
    #   raw    -- stream1 mtimes on the EOS disk, ~1 min late, good enough
    #   bracket-- neither: stream1 has already been cleaned off disk. n_TOF run
    #             numbers are strictly time-ordered, so the run must lie between
    #             its nearest measurable neighbours. Coarse, but it is the only
    #             way to see that such a run overlaps beam at all -- 224649 and
    #             224650 were invisible until this was added.
    known = dict(rawspan)
    for r, a, b in v12:
        known.setdefault(r, (a, b))
    ks = sorted(known)

    def window(r):
        if r in known:
            return known[r], 'measured'
        lo = [q for q in ks if q < r]
        hi = [q for q in ks if q > r]
        if not lo or not hi:
            return None, 'unknown'
        t0, t1 = known[lo[-1]][1], known[hi[0]][0]
        return ((t0, t1), 'bracketed') if t1 > t0 else (None, 'unknown')

    candidates = [r for r in range(PASS_LO, PASS_HI + 1) if r not in v12set]
    candidates += [r for r, _, _ in raw if r > PASS_HI]

    blocks, spans, how = {}, {}, {}
    for r in sorted(set(candidates)):
        w, kind = window(r)
        if w is None:
            continue
        spans[r], how[r] = w, kind
        a, b = w
        for (run, sub), (t0, t1, n) in subs.items():
            if min(b, t1) - max(a, t0) > 60:
                blocks.setdefault(r, []).append(
                    (run, sub, min(b, t1) - max(a, t0)))

    rows = []
    for r in sorted(blocks):
        a, b = spans[r]
        nf, nb = sizes.get(r, (0, 0))
        bl = sorted(blocks[r])
        rows.append(dict(
            ntof_run=r,
            start_utc=datetime.fromtimestamp(a, timezone.utc).strftime('%Y-%m-%d %H:%M'),
            hours=round((b - a) / 3600, 2),
            window=how[r],
            stream1_on_disk=('yes' if r in rawspan else 'no -- recall from tape'),
            raw_files=nf,
            raw_TB=round(nb / 1e12, 2),
            dream_runs=' '.join(sorted({x[0] for x in bl},
                                       key=lambda s: int(s.split('_')[1]))),
            dream_subruns=len(bl),
            beam_hours_blocked=round(sum(x[2] for x in bl) / 3600, 2)))

    out_csv = HERE / f'missing_runs_{TODAY}.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Campaign totals, computed the same way coverage_map.py does them, so the
    # handoff cannot drift from the coverage table.
    cov = {cm.READY: 0.0, cm.RECOV: 0.0, cm.LOST: 0.0}
    for (run, sub), (t0, t1, n) in subs.items():
        acc, _ = cm.classify(t0, t1, v12, raw, v12set)
        for k in cov:
            cov[k] += acc[k]
    beam_h = sum(cov.values()) / 3600
    ready_h = cov[cm.READY] / 3600
    after_h = (cov[cm.READY] + cov[cm.RECOV]) / 3600

    gaps = [r for r in range(PASS_LO, PASS_HI + 1) if r not in v12set]
    lost_runs = [r for r in gaps if r not in {q for q, _, _ in raw}]
    recov_runs = [r for r in gaps if r in {q for q, _, _ in raw}]

    tot_h = sum(r['beam_hours_blocked'] for r in rows)
    tot_tb = sum(r['raw_TB'] for r in rows)
    tot_f = sum(r['raw_files'] for r in rows)
    past_end = [r for r in rows if r['ntof_run'] > 224687]
    inside = [r for r in rows if r['ntof_run'] <= 224687]

    # Skip rate vs raw size -- the one thing that distinguishes the runs the
    # pass left behind. why_skipped.py has the full analysis and the ruled-out
    # alternatives; this recomputes the binning so the page cannot drift.
    import numpy as np
    _raw, _out = why_skipped.load()
    _runs = sorted(r for r in _raw if PASS_LO <= r <= PASS_HI)
    _tb = np.array([_raw[r][1] / 1e12 for r in _runs])
    _ok = np.array([r in _out for r in _runs])
    _e = [0, .05, .15, .25, .35, .45, .55, .65, .75, 1.0]
    skip_bins = [(lo, hi, int(m.sum()), int((~_ok[m]).sum()))
                 for lo, hi in zip(_e, _e[1:])
                 for m in [(_tb >= lo) & (_tb < hi)] if m.sum()]
    _big = _tb >= why_skipped.FLOOR_TB
    skip_small_n, skip_small_k = int((~_big).sum()), int((~_ok[~_big]).sum())
    skip_big_n, skip_big_k = int(_big.sum()), int((~_ok[_big]).sum())

    n_past = len(past_end)
    past_lo, past_hi = past_end[0]['ntof_run'], past_end[-1]['ntof_run']
    on_disk = [r for r in rows if r['stream1_on_disk'] == 'yes']
    on_tape = [r for r in rows if r['stream1_on_disk'] != 'yes']
    unproc_h = beam_h - ready_h
    skip_table = '\n'.join(
        f'| {lo:.2f}–{hi:.2f} | {n} | {k} | {k/n:.0%} |'
        for lo, hi, n, k in skip_bins)
    small_post = ', '.join(str(r['ntof_run']) for r in rows
                           if r['ntof_run'] > PASS_HI and r['raw_TB'] < 0.35)
    md = [f"""# Request to n_TOF: {len(rows)} X17 runs still to process

**From the X17 / DREAM group (Dylan Neff), {TODAY}.** Contact: dneff@cern.ch.

Thank you — the pass you ran on 2026-08-05 to 08-07 is **exactly right**, and we
have verified it: every X17 file in
`/eos/experiment/ntof/processing/official/done/` carries the UserInput we
proposed, parameter for parameter and template for template. We diffed the
`history` string in `run224572.root` against our own copy: identical on all 14
detector rows and all 26 pulse-shape filenames.

**325 runs are done. {len(rows)} are still missing, and we need them.** They
block **{tot_h:.0f} hours** of X17 beam time — {tot_h/beam_h:.0%} of our
campaign, none of which has any processed output at all.

---

## What happened, as far as we can see

The pass ran from **5 August** and the last file landed **7 August at 19:56**.
Nothing has been written since — about a day as we write this. That stop cleanly
explains the tail: **{n_past} runs ({past_lo}–{past_hi}) are simply after the
point where `done/` ends**, and they cover our last two days of data taking
(DREAM runs 150–156).

What it does **not** explain is the rest. There are **{len(gaps)} runs missing
from inside {PASS_LO}–{PASS_HI}**, scattered through the range rather than
clustered at either end, and **{len(inside)}** of those overlap X17 beam time.
We cannot see a reason for them from the outside. One partial correlation: 
{len(lost_runs)} of the {len(gaps)} in-range gaps no longer have their stream1
staged on the EOS disk, which would explain a skip if the pass reads from disk —
but the other {len(recov_runs)} do still have it and were skipped anyway.

**If you know why those were passed over, we would like to hear it** — it is the
one piece we cannot reconstruct, and it would tell us whether re-running them is
straightforward or whether something about them is broken.

## A clue: only large runs were skipped

We looked for anything that distinguishes them. Directory structure is identical
on both sets — `stream0` + `stream1`, every file `.finished`, no stragglers. An
output-size cap does not fit: it would have to sit below 21 GB, and 42 processed
runs exceed that. Position in the run range says nothing; the gaps are scattered.

**Raw size fits, and not subtly.** Of the 135 in-range runs whose stream1 is
still staged:

| raw TB | runs | skipped | rate |
|---|---|---|---|
{skip_table}

**Below 0.35 TB nothing was ever skipped — 0 of {skip_small_n}.** At or above it
{skip_big_k} of {skip_big_n} were, and the rate keeps climbing with size. That is
the shape of a resource limit a large job sometimes misses and sometimes makes —
wall clock, memory or scratch — rather than a rule that rejects a run outright.
If it were deterministic the big runs would all have failed; they did not.

The control: of the {n_past} runs missing from *after* {PASS_HI}, three
({small_post}) are below 0.35 TB, a band in which the pass never skipped
anything. So those are missing because the pass stopped, not for this reason.
Two mechanisms, cleanly separated.

We cannot see your job configuration, so this is an association, not a
diagnosis — but if there is a per-job limit worth raising for the re-run, that
size distribution is where we would look.

## What we need

| | |
|---|---|
| runs | the {len(rows)} listed below |
| UserInput | the same one already used — `UserInput_2026_EAR2_X17_v4.h` |
| output | the same place, `/eos/experiment/ntof/processing/official/done/` |
| raw | {len(on_disk)} still have stream1 staged on disk under `/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/<run>/stream1/`; {len(on_tape)} ({', '.join(str(r['ntof_run']) for r in on_tape)}) will need a recall from tape |
| order | **whatever suits your queue** — we want all of them |

### These runs have no processed output at all

Not just no v12 — **nothing**. There is no file for any of them anywhere under
`/eos/experiment/ntof/processing/`, and none under the earlier `v2` processing
either. `done/` keeps older output (files back to April 2025, including 141 from
July 2026), but in {PASS_LO}–{PASS_HI} every one of the 325 files present is
dated 5–7 August. A run processed under v2 and then skipped by this pass would
still be sitting there with its old timestamp; none is.

### The one naming point that will confuse anyone checking

The file is called `UserInput_2026_EAR2_X17_v4.h` and its content is what our
group tracks internally as **v12_liqpileup**. Both names refer to the same
thing; the header comment inside the file says so. We mention it only because
our own repository also has a *different* file called `v4`, and we do not want
anyone to reconcile the two by filename.

---

## The {len(rows)} runs

`beam h blocked` is how much X17 DREAM beam time depends on that n_TOF run.
A `bracketed` window means the run has neither a processed file nor staged
stream1, so we placed it between its nearest measurable neighbours by run
number — coarse, but enough to show it overlaps beam.

| n_TOF run | start (UTC) | hours | window | stream1 | raw files | raw TB | DREAM runs affected | beam h blocked |
|---|---|---|---|---|---|---|---|---|"""]
    for r in rows:
        md.append(f"| **{r['ntof_run']}** | {r['start_utc']} | {r['hours']:.1f} | "
                  f"{r['window']} | {r['stream1_on_disk']} | "
                  f"{r['raw_files'] or '—'} | {r['raw_TB']:.2f} | "
                  f"{r['dream_runs']} | {r['beam_hours_blocked']:.1f} |")
    md.append(f"| | | | | | **{tot_f}** | **{tot_tb:.1f}** | | **{tot_h:.0f}** |")

    skipped = sorted(set(gaps) - {r['ntof_run'] for r in rows})
    skip_block = '\n'.join(' '.join(f'{q}' for q in skipped[k:k + 10])
                            for k in range(0, len(skipped), 10))

    md.append(f"""
Machine-readable: `missing_runs_{TODAY}.csv`.

## Where the campaign stands

| | |
|---|---|
| processed and verified | 325 runs, {PASS_LO}–{PASS_HI} |
| still needed | {len(inside)} inside that range, {n_past} after its end |
| X17 campaign | DREAM runs 77–156, 2026-07-26 to 08-08, 282 beam sub-runs |
| processed today | {ready_h:.0f} h of {beam_h:.0f} h ({ready_h/beam_h:.0%}) |
| not processed | {unproc_h:.0f} h ({unproc_h/beam_h:.0%}) |
| after these {len(rows)} | ~{beam_h:.0f} h (essentially all of it) |

## The {len(skipped)} in-range gaps we are NOT asking about

Of the {len(gaps)} runs missing from {PASS_LO}–{PASS_HI}, we are asking for the
{len(inside)} that overlap X17 beam time. The other {len(skipped)} were live
while DREAM was not, so they block nothing for us:

```
{skip_block}
```

We mention them only because they are part of the same unexplained set — if they
were skipped for a reason that also applies to the ones we are asking for, that
would be worth knowing.

## Why it matters to us

We key every n_TOF hit to a DREAM trigger through a time calibration that is
fitted **per (DREAM run, n_TOF processing) pair** and does not transfer between
processings. Mixing a v12 run with an older processing inside one DREAM run is
not an option: the plastic flash identification alone differs by 37–85 % of
bunches, and our own v11 differs from v12 by 14–21 % in liquid hit yield. So a
DREAM run is either fully covered by this processing or it waits.
""")

    out_md = HERE.parent / f'NTOF_REPROCESSING_REQUEST_{TODAY}.md'
    out_md.write_text('\n'.join(md))

    out_html = HERE / 'ntof_reprocessing_request.html'
    out_html.write_text(handoff_html.render(
        rows, tot_f, tot_tb, tot_h, beam_h, ready_h,
        past_end, inside, gaps, lost_runs, recov_runs, skipped,
        on_disk, on_tape, skip_bins, TODAY))

    print(f'{out_md}\n{out_html}\n{out_csv}')
    print(f'{len(rows)} runs, {tot_f} raw files, {tot_tb:.1f} TB, '
          f'{tot_h:.0f} beam hours blocked')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
