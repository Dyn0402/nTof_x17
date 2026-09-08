# HANDOFF — the hot-channel wildcard is built and wired in, and its first
# tuning made D worse. Tune it locally, event by event, before touching
# condor again.

**Written 2026-09-08. Local work on run_145/D only.**

Companions: [`HANDOFF_D_NOISY_CHANNELS.md`](HANDOFF_D_NOISY_CHANNELS.md) (the
original spec this implements), [`STATUS.md`](STATUS.md),
`compare_hotmasked_rerun.py`.

---

## 1 · Where this stands in one paragraph

`HANDOFF_D_NOISY_CHANNELS.md`'s wildcard spec (never seed on a flagged
channel, keep it in the fit down-weighted, cap its influence, record it) is
**built, unit-tested, and wired into both reconstruction drivers** (bench and
beam) and the condor packaging. It was then **actually run** — a real
lxplus/condor re-reconstruction of D/run_145/stat090_0000, all 7 tags, full
reco, no allowlist — and the result **fails the spec's own success
criterion**: gated-fit rate should go up, it went from 72.6 % to 33.4 %, and
of the events common to both runs **zero gained a fit that the frozen run
didn't have; 16 152 lost one it did have**. This is not a code bug. It is two
unvalidated constants, both flagged as "first cut, not tuned" when they were
written, both now confirmed to matter a lot. Fixing them needs looking at
real event windows directly, which needs no condor at all — the raw data is
already local.

---

## 2 · What is built (do not redo this part)

| piece | file | what |
|---|---|---|
| per-strip dead/hot/noisy classifier | `noisy_channels.py` | built from raw `combined_hits`, matches `k_robustness`'s independent measurement to ~1 point per chamber. `hot` class only is wired downstream (`noisy` is threshold-fragile, empty on D-x — see its own docstring) |
| bundle field | `wft/calib.py` `CalibrationBundle.hot` | mirrors `dead`, save/load round-trips |
| fit down-weighting | `wft/model.py` `HOT`, `HOT_NOISE_INFLATION`, `prep_plane` | a hot row's noise is inflated (NOT censored like `dead` — `sat` untouched, row keeps its dof) |
| seeding admission | `wft/seed.py` `seed_candidates` | ranks/admits by CLEAN (non-hot) strip count; **the exact admission rule is the thing to revisit, see §4.2** |
| provenance | `wft/reco.py` `PlaneFit.n_flagged_strips` | dead+hot strips in the fit window, per plane, every row |
| condor plumbing | `ntof_tracking/wft_beam.py`, `condor/{run_beam_job,make_beam_package,beam_reco.sub}` | `--hot <json>` ships a per-arm wildcard file the same way `--allow` ships a stage-2 allowlist |
| bundle builder | `apply_hot_wildcards.py` | re-derives `calib_bundle_prelim` (verified byte-for-byte against the frozen run's own `events_prelim.meta.json`) and attaches `hot`; `--export-json` makes the condor-shippable file |
| tests | `wft/tests/test_hot_mask.py` | 4 checks, synthetic bundle, no data needed. Full suite 23/23 passing |
| comparison tool | `compare_hotmasked_rerun.py` | frozen vs. a re-run's `events_prelim.parquet`, the table in §3 |

Git commit `a77c262` has everything above except this handoff and the
comparison script. **`wft`/`ntof_tracking` must be committed before any
`make_beam_package.py` run** — it ships `git archive HEAD`, so uncommitted
changes there silently do not reach the worker.

---

## 3 · What was measured

Cluster 4141149 (lxplus condor, 7 jobs, D/run_145/stat090_0000, `--hot`
carrying 42 x / 57 y hot channels, `--v-drift 42.6` verified to match the
frozen bundle exactly). All 7 completed, no holds. Merged with
`ntof_tracking/condor/merge_beam_tags.py` into
`/home/dylan/x17/wft_beam145_hotmasked/analysis/` — **a directory dedicated
to this comparison, never the frozen products' own location.**

```
python -m sept26_prelim_analysis.compare_hotmasked_rerun \
    --new /home/dylan/x17/wft_beam145_hotmasked/analysis/run_145/stat090_0000/mx17_D/events_prelim.parquet
```

| | frozen (no wildcards) | hot-masked |
|---|---:|---:|
| events attempted | 46 218 | 33 393 (-28 %) |
| both-plane fit converges | 72.6 % | 33.4 % |
| quality_ok (both planes) | 67.4 % | 29.4 % |
| median x_chi2/dof | 10.7 | 41.9 |
| median x_n_strips | 42 | 32 |

Of the 33 393 events common to both tables: **0 gained a good fit, 16 152
lost one**. Splitting the hot-masked table by whether a window touches a
flagged strip (42-59 % of attempted windows do, per plane — far more than
the raw per-channel hot fraction, because a typical cluster is wide enough to
cross one somewhere in D): flagged windows both_ok 36.2 %, clean windows
17.7 % (clean windows are ALSO down from baseline — see §4.2, this is not
purely a fit-weighting effect).

---

## 4 · Two candidate causes — both are constants that were guessed

### 4.1 `HOT_NOISE_INFLATION = 10.0` (`wft/model.py`) is probably too weak

Windows that touch a flagged strip show chi2/dof 45.6 against 6.6 for windows
that do not (`compare_hotmasked_rerun.py`'s split). A hot channel's actual
amplitude excursion is evidently large enough that inflating its noise by
only 10x still lets it pull real chi2 weight — recall from
`HANDOFF_D_NOISY_CHANNELS.md` §2.1 that these are NOT small signals, they are
wide, dilute deposits with the SAME median charge as a normal strip (0.94x),
just spread over more strips. A single flagged sample's raw residual can
plausibly be many times a normal strip's even after a 10x haircut.

**How to find the right number, locally, without condor**: pull the raw
window for a handful of hot-touching events (see §5.1) and directly compare,
at a few candidate `HOT_NOISE_INFLATION` values, how much of the chi2 each
hot row contributes vs. a clean row in the same window. `wm.chi2_plane`
already returns the profiled `q`; `wm.model_waveforms(...)` gives the
per-strip model prediction, so `(W - model) / noise` per row is directly
inspectable. Scan `HOT_NOISE_INFLATION` (try 10, 30, 100, 300) against a
FIXED small set of real windows and look for where the hot rows' residual
stops dominating chi2, rather than guessing another round number blind.

### 4.2 The seeding admission rule may be discarding real tracks

`seed_candidates` currently rejects a candidate cluster when its CLEAN strip
count is below `min_strips` (5 for beam). But `HANDOFF_D_NOISY_CHANNELS.md`
item 1 says "never seed on a flagged channel" — read most naturally as: a
cluster that exists ONLY because of flagged strips should not seed. A real
5-6 strip cluster that merely grazes 1-2 hot strips now loses its seed
ENTIRELY once bumped below `min_strips` on the clean count alone, rather than
being admitted (with its hot strips still present as members, still
down-weighted in the fit per §4.1) — which is arguably the OPPOSITE of "a
track should never be lost because it crossed a bad channel." This is the
more likely driver of the 46 218 -> 33 393 drop in attempted events, and
probably also explains why even CLEAN windows (no flagged strip at all) show
a lower both_ok than baseline: a real track's x-plane cluster can lose its
seed to this rule while its y-plane cluster (reported with
`x_n_flagged_strips`/`y_n_flagged_strips` == 0, since THAT plane's cluster
had no hot strips) never gets attempted at all because the whole event never
made it into `wanted`.

**A concrete, ready-to-try fix** (`wft/seed.py::seed_candidates`, in the loop
building `out`):

```python
# current:
if clean_counts[c] < min_strips:
    continue
# candidate: reject only a cluster that is ENTIRELY flagged strips
if clean_counts[c] < 1:
    continue
```

This still satisfies "never seed on a flagged channel ALONE" (an all-hot
blob has `clean_counts == 0` and is still rejected) while no longer
penalizing a real cluster for merely including a hot strip among several
clean ones. **Test this locally against the SAME small event set as §4.1
before touching condor** — it changes which events even reach the fit, so it
needs its own check independent of the noise-inflation tuning.

---

## 5 · Order of work for the next session

**Do not iterate via condor again until a local pass has a plausible
setting.** A condor cycle is ~15 min of wall-clock per iteration; a local
check on a handful of events is seconds. Dylan's own words on this: test
event by event, locally.

### 5.1 Build a few real windows locally, no condor needed

The raw data run_145/D needs is already staged locally — `combined_hits_root`
AND `decoded_root` both exist under
`/media/dylan/data/x17/beam_july/runs/run_145/stat090_0000/`. Nothing here
needs EOS or lxplus:

```python
from ntof_tracking import wft_beam as wb
from wft.calib import CalibrationBundle
from wft import model as wm

cal = CalibrationBundle.load(
    '/media/dylan/data/x17/beam_july/analysis/wft/run_145/stat090_0000/'
    'mx17_D/calib_bundle_hotmasked')   # already built, from apply_hot_wildcards.py
wm.use_calibration(cal)

cfg = wb.beam_config('D', 'run_145', 'stat090_0000')
cfg.file_tags = ['260805_14H06_000']   # one tag is plenty for this
# reconstruct_subrun (or its internals -- see wft_beam._stream_windows /
# wft.reco._stream_windows) yields (event_id, windows-dict, ...) per event;
# pick a few event_ids known to have flagged strips (compare_hotmasked_rerun's
# merged table already has x_n_flagged_strips/y_n_flagged_strips per event --
# load the already-merged hot-masked table and filter on those columns to
# find good candidates FAST, then re-extract just those events' windows).
```

`ntof_tracking/wft_beam.py`'s `--limit-per-tag` (also on `run_beam_job.py`,
built for exactly this: "smoke-testing the stack on a login node") is the
other route — a local `reconstruct_subrun` call over ~50-200 events finishes
in well under a minute and gives a fresh `events_prelim.parquet` to compare
against the merged one already at
`/home/dylan/x17/wft_beam145_hotmasked/analysis/run_145/stat090_0000/mx17_D/events_prelim.parquet`
for the SAME event_ids — no condor needed for that comparison either, since
that table already has every event this local subset would also cover.

### 5.2 Tune §4.1 and §4.2 against that small local set

Scan `HOT_NOISE_INFLATION` and try the `seed.py` admission change from §4.2,
independently and together, checking against the local sample's
both_ok/chi2-per-dof (`compare_hotmasked_rerun.py`'s `summarize()` works on
any `events_prelim.parquet`, condor-produced or local).

### 5.3 Only once local numbers look right: re-run the full condor cluster

Same recipe as this session: `apply_hot_wildcards.py` (rebuild the bundle
with the new `HOT_NOISE_INFLATION`/admission code — note `HOT_NOISE_INFLATION`
is a module constant in `wft/model.py`, not a bundle field, so a code change
+ recommit is needed, not just a bundle rebuild), `--export-json`,
`make_beam_package.py --hot ... `, rsync, `condor_submit`, `merge_beam_tags.py`,
`compare_hotmasked_rerun.py`. Compare against §3.3's full criteria list
(angle-scale spread, head-on excess, cluster width, A/C unchanged, gated
count up) — this session only checked convergence rate and chi2/dof, which
was enough to catch the regression but is not the full HANDOFF_D_NOISY_
CHANNELS.md §3.3 acceptance bar.

---

## 6 · Where everything is

| | |
|---|---|
| git commit with the wildcard mechanism | `a77c262` |
| the frozen baseline (untouched) | `/media/dylan/data/x17/sept26_prelim/fullpass/run_145/stat090_0000/mx17_D/events_prelim.parquet` |
| the re-derived baseline bundle (verified match) | `/media/dylan/data/x17/beam_july/analysis/wft/run_145/stat090_0000/mx17_D/calib_bundle_prelim` |
| the hot-masked bundle used for the condor run | `.../mx17_D/calib_bundle_hotmasked` |
| condor package (local copy) | `/home/dylan/x17/wft_beam145_hotmasked/` |
| condor package (on lxplus, still there) | `lxplus:~/wft_beam145_hotmasked/` |
| raw per-tag condor outputs, pulled back | `/home/dylan/x17/wft_beam145_hotmasked/results/*.tar.gz` |
| the merged hot-masked table from this session's run | `/home/dylan/x17/wft_beam145_hotmasked/analysis/run_145/stat090_0000/mx17_D/events_prelim.parquet` |
| local raw data (combined_hits + decoded_root) | `/media/dylan/data/x17/beam_july/runs/run_145/stat090_0000/` |
| hot-channel classification (source of truth) | `<out>/noisy_channels/noisy_channels_run_145.csv` |
