# Cosmic-bench reprocessing on the 2026-07-24 waveform analyzer

> **2026-07-28:** the reprocessing described here is still the current hit
> generation on disk and is still worth having (more hits, better low-amplitude
> recovery, `significance` branch). What changed is what hits are *for*: they
> are a cluster-finding and QA product, not the basis for position, angle or
> depth. See `../RECONSTRUCTION_BASIS.md`. The waveform-first chain reads
> `decoded_root` directly — 145 subruns / 209 GB are local, so no reprocessing
> is needed to move to it.

> **SUPERSEDED SAME DAY by the matched-filter rework (`a1cce79`).** The
> `33e132b` (unified-trigger) reprocessing documented below was completed on the
> whole local set, then the analyzer was reworked again to trigger on
> low-amplitude signals (a **matched-filter boxcar gate**, now the default, width
> from the Dream shaping time; adds a per-hit `significance` branch). The entire
> local bench was re-run a final time on `a1cce79` — that is the current
> generation on disk. See "Final generation: a1cce79" at the bottom. Everything
> between here and there is the intermediate `33e132b` pass, kept for its
> old-vs-new numbers and the pedestal/pull/consistency machinery, which the
> a1cce79 pass reuses unchanged.


Driver: `reprocess_cosmic_bench.py` (plan/execute), `reprocess_compare.py` (old-vs-new).
Logs + pre-overwrite hit-count snapshots: `reprocess_logs/`.
Analyzer: `mm_strip_reconstruction` @ `33e132b`, **Release** build
(`cmake-build-release/`) — all processing before this date used the Debug `-O0`
binary.

## What changed in the chain

New unified pulse trigger (the old derivative trigger dropped 22–46 % of genuine
hits), median local baseline with a 2-sample guard gap, full-span integral
(the old between-crossings sum kept ~52 % of the charge of a 5–10 σ pulse),
interpolated TOT/crossings, `trunc_left`/`trunc_right` branches, ZS densify
before CNS, and — the big one for old data — **common-noise subtraction ON by
default with pedestal RMS measured post-CNS** (the old default was CNS off).

## Scope

Everything on this machine, in place (`hits_root/`, `combined_hits_root/`
overwritten). Done in three passes: passes 1–2 = the 122 subruns that already
had decoded data locally; pass 3 = the 87 subruns whose `decoded_root` had to be
pulled from lxplus first (see below). **209 subruns total, ~825 FEU files,
~198 GB decoded input, 0 analyzer failures.** Decoding was not re-run —
`decoded_root` is the input.

### Completing the set: 87 subruns pulled from lxplus

87 subruns had `combined_hits_root` (old-generation) but no local
`decoded_root` — outputs kept, inputs left on EOS. Pulled 824 files (429
decoded + 395 pedestal, 82 GB) from
`lxplus:~dneff/x17/cosmic_bench/june_tests/<run>/<subrun>/{decoded_root,raw_daq_data}`
and placed them into the matching local `<bench>/<run>/<subrun>` trees (824/824
placed, every subrun matched an existing local dir). The machine now holds the
complete decoded set.

### Analyzer pinning (why pass 3 uses a worktree build)

Mid-session the shared analyzer repo
(`~/CLionProjects/mm_strip_reconstruction`) advanced two commits —
`781c5b6` (low-gain matched-filter mode) then `018f575` (**makes the
matched-filter gate the default**, a different pulse-finding algorithm) — and
its Release binary was rebuilt at 16:11, mid-run, killing the first pass-3
attempt with a `PermissionError` on the exec'd binary. Finishing on `018f575`
would have made these 87 subruns a *different generation* than the 122 done on
`33e132b`. Fix: built `33e132b` in an isolated `git worktree` (does not touch
the other session's tree), validated it reproduces the pass-1 binary's output
exactly (1,481,146 hits on a check file, identical), and pointed the driver at
it via the `REPROC_SOFT` env override. **The entire fleet is one generation:
`33e132b`.** The active `018f575`/low-gain work is a separate future
reprocessing decision, not mixed in here.

Pedestals pulled from lxplus (`~dneff/x17/cosmic_bench/june_tests/...`, 29 MB/s)
for det2/det3 6-22 `longer_run` (its own 20H20 set, FEU 06/08), det6/det7 6-26
(00H43, FEU 03/04/06/08) and det4 6-24 (FEU 03). The six 6-27 drift-scan points
have no pedestal of their own anywhere — they share the run's single 16H35 set,
which the driver resolves explicitly ("sibling subrun") rather than guessing.

25 GB of `decoded_root` was also pulled for det3-p2 6-27 and det3/det4 6-23,
whose `combined_hits_root` held 22 files whose decoded input was no longer
local. Without it those two runs would have ended up half-new/half-old in one
directory.

## Result: two populations, both expected

**June campaign runs — hits UP**, exactly as the analyzer validation predicted:

| run | old hits | new hits | ratio |
|---|---|---|---|
| det4 6-24 long_run | 733,233 | 3,537,619 | **4.82×** |
| det6/det7 6-26 long_run | 7,849,239 | 23,889,843 | **3.04×** |
| det3 6-16 ArIso (ZS) | 3,534,729 | 6,746,796 | 1.91× |
| det3/det4 6-23 long_run | 1,789,456 | 5,645,312 | 3.15× |
| det3-p2 6-27 sanity check | 2,878,269 | 4,848,303 | 1.68× |
| det3 6-27 saturday long_run | 899,317 | 1,456,505 | 1.62× |
| det3 6-27 drift scan (6 pts) | — | — | 1.57–2.08× |
| det2/det3 6-22 longer_run | 1,548,021 | 1,332,095 | 0.86× |

The saturday long_run's 1.62× matches the standalone validation on that run's
FEU08 (186,984 new vs 119,026 derivative = 1.57×) — an independent confirmation
that the fleet reprocessing reproduces the validated behaviour.

**det4's 4.82× is the headline.** The June conclusion "det4 is gain-limited,
hybrid not measurable at this operating point" was an artifact of the old
trigger discarding 50–90 % of its 5–20 σ strips. det4 must be re-examined
before that verdict stands (`REPROCESSING_PLAN.md` §3.2 step-2 gate).

**Jan/Apr/May runs — hits DOWN to 0.29–0.54×.** This is the CNS default flip,
not a regression. Direct test on `det3_HV_Scan_5-5-26/resist_490V_drift_900V`
FEU 01, same binary, same pedestal:

| | hits | hits/event (median, 512 ch) | amp median |
|---|---|---|---|
| CNS **off** (old default) | 4,881,912 | 188 | 110.5 |
| CNS **on** (as reprocessed) | 1,412,004 | 53 | 133.5 |

0.29× — matching the observed ratio for that subrun exactly. 188 hits/event on
a single 512-channel board is noise-dominated; the removed population is the
coherent common-mode these boards are known for (FEU 6/8 raw σ~115 vs CNS
σ~10). The surviving hits have a *higher* median amplitude.

## Consequences to handle before quoting anything

1. **Every `cache/` veto50 pickle in the Analysis tree is now stale** relative to
   the hits underneath it. Nothing downstream has been re-run.
2. **"Hits" now means CNS-subtracted hits** for every run. Historical efficiency
   / resolution numbers were computed on non-CNS production hits; the June
   waveform scripts (24, 26–28) always did their own pedestal+CNS, so those are
   consistent, but the hit-level chain (08/09/12) is not directly comparable to
   its own past output.
3. The 6-22 "flat3" pedestal workaround (`HANDOFF_6-22_pedestal_flat3.md`) exists
   because that run's pedestal RMS was computed pre-CNS on spark-inflated
   pedestals. Post-CNS pedestal RMS addresses that at the source — the flat3
   reprocessing tree is superseded in principle and should be re-checked
   (`_flat3_reproc/` is no longer on this machine anyway).
## What is still old-generation

Bench-wide audit of every `combined_hits_root` file (does it carry `trunc_right`
and was it written today): **122 new-generation, 144 old-generation.**

The 144 sit in **87 subruns that have combined hits but no `decoded_root` on this
machine** — outputs kept, inputs left on lxplus. Mostly HV-scan points plus some
substantial runs: det2/det3 6-22 `long_run` (16 GB), det6/det7 6-26
`longer_run`/`short_run`, det1/det2 6-17 (all 10 subruns), det3/det4 6-23
`longer_run`/`short_run`, det3 6-25 `long_run`, det3 6-26 quick, and the 6-27
`hv_scan`/`hv_scan2` points. All 87 exist on lxplus: **72.8 GB total**
(~40 min at the measured 29 MB/s, ~50 min to process).

Until those are done, `combined_hits_root` across the bench is mixed-generation.
Per-subrun it is not mixed — every subrun is wholly old or wholly new — so
anything keyed on a single subrun is self-consistent. Check the generation with
`'trunc_right' in uproot.open(f)['hits'].keys()`.

Also found, unrelated to the reprocess: the det4 6-24 run has **three** copies of
its outputs — `det4_day/` (full tree, decoded present, reprocessed, and what
`qa_config.g_det4` points at), plus `det4/` and `det_4day/` holding only stale
`combined_hits_root` + `m3_tracking_root`. The two stale copies were left alone;
they are a wrong-tree hazard for anything not going through `qa_config`.

---

# Final generation: a1cce79 (matched-filter, low-amplitude rework)

**This is what is on disk now.** After the `33e132b` pass above, the analyzer was
reworked to trigger on low-amplitude signals and the entire local bench was
re-run one final time on commit **`a1cce79`**.

## The change
A **matched-filter (boxcar) gate** is now the default pulse finder: the waveform
is smoothed by a boxcar whose width matches the shaped pulse before the 5σ gate,
so low-amplitude slow risers that the raw-sample gate missed now trigger.
Threshold stays 5σ; a per-hit `significance` branch is added. The gate width is
set per-FEU from the Dream shaping time: `--mf = max(3, round(1.7 × peaking_ns /
tps))`. On the cosmic bench (tps = 60 ns, peaking 180 ns) that is `--mf 5`;
older runs whose `run_config.json` lacks a sample-period field fall to the
analyzer's AUTO width, which is **byte-identical** to explicit `--mf 5` at 60 ns
(verified). Every subrun therefore uses the same effective gate.

## Faithfulness to the DAQ pipeline
The reprocess driver imports `process_run.py`'s own cfg helpers
(`run_sample_period_ns`, `find_dream_cfg`, `parse_dream_peaking`) and passes the
**identical** `--tps/--mf` the official pipeline computes — validated to match on
a det6/det7 file (`--tps 60 --mf 5`).

## Build isolation (learned the hard way)
The shared analyzer checkout advanced *twice* mid-session and a rebuild broke a
running pass with a `PermissionError`. Both the `33e132b` and `a1cce79` passes
were therefore run from **dedicated `git worktree` builds** (in the session
scratchpad) pointed to by the `REPROC_SOFT` env var, so another session's edits
or rebuilds cannot corrupt an in-flight run. To reproduce:
`git worktree add --detach <dir> a1cce79 && cmake -S <dir> -B <dir>/cmake-build-release -DCMAKE_BUILD_TYPE=Release && cmake --build ...`,
then `REPROC_SOFT=<dir>/cmake-build-release ../.venv/bin/python reprocess_cosmic_bench.py --jobs 6`.

## Result
825 FEU files, **0 failures**, 245 combined files, 100.7 min wall.
Bench-wide audit: **245 combined files carry `significance` (a1cce79); 0 live
outputs are older-generation.** The only non-new combined files are the two
stale *duplicate* det4 trees (`det4/`, `det_4day/` — no decoded data, not
`qa_config.g_det4`) and one `_unmatched` combiner leftover.

Old(≈33e132b)→new(a1cce79) hit yield, **+40 % overall** (189.0 M → 264.2 M),
concentrated where the low-amplitude gain matters most — the low-gain 700 V
det6/det7 running:

| run | 33e132b | a1cce79 | ratio |
|---|---|---|---|
| det6/det7 6-26 HV scan (400–500 V) | — | — | **2.8–4.5×** |
| det6/det7 6-26 overnight long_run | 23.9 M | 32.4 M | 1.36× |
| det3/det4 6-23 short/longer_run | — | — | **7–13×** |
| det2/det3 6-22 long_run | 7.9 M | 10.1 M | 1.27× |
| det4 6-24 long_run | 3.54 M | 5.25 M | 1.48× |
| det1 (He/ArIso, high-gain) | — | — | ~1.09× |

(High-gain runs barely move — their pulses already cleared the raw gate; the
matched filter buys the most on low-gain/low-field operating points.)

## Still open (unchanged by this pass)
1. All `cache/` veto50 pickles are stale; nothing downstream re-run.
2. det4 6-24 stale duplicate trees (`det4/`, `det_4day/`) still present.

**2026-07-30 update:** the lxplus/EOS upload (previously listed here as open,
parked pending an explicit decision) is done — see
`HANDOFF_EOS_HITS_UPLOAD_2026-07-30.md` for the run→EOS-path resolution and
verification record. All 278 reprocessed directories (1,070 files, 22.38 GB)
are live on EOS with 0 failures.
