# July beam HV-scan quick-look analysis

Scripts that turn a July beam run into web-browsable HV-scan plots. Output goes
to `/mnt/data/x17/beam_july/analysis/July_HV_Scan/<...>/`, which the DAQ
flask_app **"Analysis" tab** browses (`GENERAL_ANALYSIS_DIR = {BASE_DATA_DIR}analysis`).

> **Always run with the repo venv** — `.venv/bin/python` (pandas/uproot are only
> in the venv, not system python3). Run from the repo root
> (`~/PycharmProjects/nTof_x17`).

## The three scripts

| Script | Use when | Output subdir |
|---|---|---|
| `july_hv_scan.py [run]` | one run that is a drift×resistdrop grid (`scan_drift<D>_resistdrop<D>`), e.g. run_3/6/7 | `July_HV_Scan/<run>/` |
| `run9_mesh_scan.py [run]` | one run packing several scans, each a mesh config (`scan<NN>_dr<D>_A<HV>_<i>`), e.g. run_9 | `July_HV_Scan/<run>_mesh/` |
| `compare_scans.py` | overlay arbitrary scans, **across runs** (edit `SERIES` at top) | `July_HV_Scan/<OUT_LABEL>/` |

```bash
.venv/bin/python ntof_july_analysis/july_hv_scan.py 7      # run_7 (or "run_7", or omit for newest)
.venv/bin/python ntof_july_analysis/run9_mesh_scan.py 9
.venv/bin/python ntof_july_analysis/compare_scans.py       # edit SERIES first
```

All three are **safe to run on a live run**: unprocessed / pedestal-only subruns
are skipped (see "Gotchas").

## Data conventions (July / `beam_july`)

- **4 detectors** mx17_A/B/C/D → FEUs A:[3,4] B:[5,6] C:[7,8] D:[1,2].
- **HV channels** (from `run_config.json` `detectors[].hv_channels`): resist =
  card 5 ch 1–4 (A–D), drift = card 9 ch 0–3. Per-subrun HV values are in
  `sub_runs[].hvs["<card>"]["<ch>"]`. The `A<HV>` token in a subrun name is only
  mx17_A's resist HV — **read each detector's own resist HV from the config**,
  which every script does.
- **Hits** live in `<subrun>/combined_hits_root/*_feu-combined_hits.root`, tree
  `hits`, columns `eventId, feu, channel, time, amplitude`. `time` is already ns.
- **Total events** = max `eventId` across `<subrun>/decoded_root/*.root` (`nt` tree).

### Metrics & knobs (top of `july_hv_scan.py`)
- `AMP_THRESHOLD = 400` ADC hit threshold.
- `TIME_WINDOWS` — coarse time-of-arrival bins, edges `[0,800,1150,3500,7000,10000] ns`
  (matches `../plot_hits_vs_time_hv_scan.py`). The **1.15–3.5 µs** bin is the
  "mid-window turn-off" (post-flash signal that drops precipitously with HV).
- `detector_window_metrics()` → per-window `hits_per_event` and `mean_amplitude`.
  `WINDOW_METRICS` maps those keys to axis labels; adding a metric there flows to
  all the window plots.

### Plot types (all use independent y per panel so shape, not absolute level, reads)
- **heatmap_* / lines_*** (`july_hv_scan.py`): collapsed single-window drift×drop.
- **timewin_*** : metric vs resist HV, one line per time window, panel per detector.
- **timewin_grid_*** : rows = time window, cols = detector, one line per drift.
- **mesh_* / compare_*** : same grid but one line per **series** (mesh config, or
  arbitrary run/scan), plus a blown-up mid-window panel.

## How to add a cross-run / cross-scan comparison

Edit `SERIES` at the top of `compare_scans.py`. Each entry selects subruns of one
run by a **regex on the subrun name** (resist HV still comes from that run's
config, so mixed naming schemes mix freely):

```python
SERIES = [
    {'label': 'run_10 (mesh disconnected)', 'run': 'run_10', 'match': r'^dr800_'},
    {'label': 'run_9 scan01 (mesh off)',    'run': 'run_9',  'match': r'^scan01_'},
    {'label': 'run_9 scan02 (cfg 2)',       'run': 'run_9',  'match': r'^scan02_'},
]
OUT_LABEL = 'run10_vs_run9'   # -> July_HV_Scan/run10_vs_run9/
```

Restrict drift with the regex (`r'^dr800_'`) so you compare like with like. Then
`.venv/bin/python ntof_july_analysis/compare_scans.py`.

## Gotchas

- **Pedestal contamination**: each subrun's raw dir gets a *copy* of the shared
  pedestal acquisition named `Mx17_pedestals_datrun_...`, which shares the
  `_datrun_` token with real data. All loaders exclude `_pedestals_` files
  (`PEDESTAL_NAME_TOKEN`). If plots come out flat, the DAQ `processor_watcher`
  may have processed the pedestal instead of the beam data — see the DAQ-repo fix
  and reprocess.
- **Truncated ROOT files**: a live run may leave a combined-hits file with no
  `hits` tree; `load_hits` skips those.
- New naming scheme? Add a regex/parser like `SCAN_RE` (run9) — the metric and
  plotting helpers are naming-agnostic and reusable via import.
