# The track QA page — <https://dylan-neff.web.cern.ch/x17/qa-tracks.html>

Built 2026-09-10. The per-track tracking QA, browsable from the campaign down
to a single track, on static CERN web hosting with no server behind it.

## Why it exists

`tracking_qa` reduces 29 M tracks to quantiles, and a quantile is where a lead
stops. It says arm D's `chi2/dof` p50 moved in run_116; it cannot say which
tracks moved it. Answering that has meant pulling the 11.4 GB campaign table
onto a machine that has it. This makes the same table answerable from a
browser, from the control room, with nothing installed.

## The four levels

| level | what it is | where it comes from | cost |
|---|---|---|---|
| campaign + 36 runs | quantiles, gate rates, pathology fractions | `data/x17-trackqa.json` | 217 kB, with the page |
| a run's file tags | the same, per tag — a real time series | `data/trackqa/<run>.json.gz` | 3–210 kB, on click |
| a sub-run's tracks | the tracks themselves | one parquet shard | **a few MB of a 10–20 MB file** |
| one track | every column it has | one row of that shard | ~0 after the first |

Levels 1–2 are frozen from the `tracking_qa` CSVs. **Level 3 is computed in
the browser, from the same columns that script profiled** — so the two can be
checked against each other, and the live histogram draws the frozen run median
as a tick for exactly that reason.

## How level 3 works without a server

CERN's Apache honours HTTP range requests (`206 Partial Content`, correct
`content-range` — measured 2026-09-10). Parquet keeps each column in its own
contiguous byte range. So the page reads the footer, works out which ranges
hold the columns being plotted, and asks for those. Measured on one 121 k-track
sub-run:

| | first 7 columns | each further variable |
|---|---|---|
| local server | 205 ms | 24–42 ms |
| **the deployed site** | **2.4 s** | **0.3 s** |

Almost all of the difference is per-request latency, not transfer: CERN serves
HTTP/1.1 and a column scan is a few dozen small ranges. Changing a *cut* after
the load costs nothing — the columns are already in memory as typed arrays.

Three things this rests on:

1. **Range requests.** If they stop being honoured nothing errors; every
   sub-run click silently becomes a 10–20 MB download.
2. **SNAPPY.** The browser reader (hyparquet) ships snappy and nothing else.
   `trackqa_shards.py` writes snappy for that reason alone — it costs ~11 %
   over zstd and saves a wasm dependency.
3. **Row groups of 16 384.** Small groups make one track cheap and a column
   scan chatty; large ones do the reverse. The measured table is in
   `trackqa_shards.py`.

## Rebuilding it

```bash
# 1. the shards -- ~35 min, 292 files, 4.6 GB, on the machine with the track table
python -m sept26_prelim_analysis.trackqa_shards

# 2. the two summary tiers, into the site repo
python3 ~/PycharmProjects/dylan-cern-site/scripts/freeze_x17_trackqa.py \
    --qa    /media/dylan/data/x17/sept26_prelim/tracking_qa_fullpass \
    --index /media/dylan/data/x17/sept26_prelim/trackqa_web/shard_index.json

# 3. push -- the site and the shards go separately, and both are needed
cd ~/PycharmProjects/dylan-cern-site
./scripts/deploy-eos.sh          # page, JS, both summary tiers
./scripts/deploy-trackqa.sh      # the 4.6 GB of parquet
```

`trackqa_shards` skips shards that already exist (re-indexing them rather than
rebuilding), so an interrupted build resumes. Pass `--force` to rebuild.

`deploy-trackqa.sh` takes run names to push a subset:
`./scripts/deploy-trackqa.sh run_145 run_86`.

## Rebuilding it — the time base

`t_since_flash_ns` and `e_neutron_keV` are filled from the slim's `events`
tree, which carries the trigger's flash time as `t_pred_ns`. That product is
per sub-run and lives in `<out>/trigtime/`; step 1 above reads it through
`build_tracks`. If it is missing, extract it where the slims are:

```bash
# on lxplus, ~90 s for all 292 sub-runs, 380 MB
python -m sept26_prelim_analysis.trigger_time extract --all
# then rsync it home, and backfill any stage-3 table built before it existed
python -m sept26_prelim_analysis.trigger_time backfill --campaign
```

## Three things worth knowing about the data

**Rejected tracks are in the shards.** All 29.16 M, not the 14.26 M that pass
the 3D gate. Shipping only the survivors would have made the one question the
gate raises — what does it throw away, and did that change? — the one question
the page could not answer. It is why the shards are 4.6 GB and not 2.3.

**14.5 % of tracks have no time since the flash, and it is not this
column's fault.** 24 923 996 of the 29 159 045 carry one; the other
4 235 049 are exactly — in all 292 sub-runs, with zero exceptions either way —
the tracks whose **`bunch` is also null**, i.e. the triggers stage 1 never
joined to an n_TOF bunch at all. A sub-run straddling an n_TOF run boundary
keeps only the segment the slim was built on: `run_114/stat090_0000` joins
events 2–14 098 and nothing after. So the coverage is the DREAM↔n_TOF join's,
it predates this column, and `bunch` is the one field to select on if a plot
needs the joined sample.

**Four sub-runs have no `select_reason`** — `run_79/stat090_0009`, `_0010`,
`_0012` and `run_154/stat090_0006` — because they were built before
`build_tracks` recorded it. The shard writes the column as all-null, records
the fact in its own parquet metadata, and the page shows *"not recorded"*
rather than an em dash, so a missing column is never mistaken for a missing
value. Every other column in `COLUMNS` is required and its absence is fatal.

## What it is not

Not a hit display. The reconstruction stores the fitted lines and the
candidates it rejected, **not the strip charges they were fitted to**, so the
sketch on an expanded track is the fit's answer and not the event. A real hit
display needs the waveforms re-read from `decoded_root` on EOS — a curated few
hundred events would be tractable, the whole campaign is not.

And per `RECONSTRUCTION_BASIS.md`: everything here is waveform-forward-model
geometry. Nothing on this page comes from `combined_hits` times.

## Files

| file | repo | what |
|---|---|---|
| `trackqa_shards.py` | `nTof_x17` | stage-3 tables → 292 web shards + index |
| `scripts/freeze_x17_trackqa.py` | `dylan-cern-site` | CSVs → tiers 1 and 2 |
| `pages/x17/qa-tracks.html` | `dylan-cern-site` | the page |
| `js/x17-trackqa.js` | `dylan-cern-site` | the four levels |
| `js/x17-trackqa-worker.js` | `dylan-cern-site` | parquet reading, off the main thread |
| `js/lib/hyparquet/` | `dylan-cern-site` | vendored reader — the site's only dependency |
| `scripts/deploy-trackqa.sh` | `dylan-cern-site` | the shard push |
