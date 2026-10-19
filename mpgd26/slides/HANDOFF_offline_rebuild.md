# Rebuilding mpgd26 figures away from the bench

> ## Mostly resolved -- 2026-08-27, later the same day
>
> The bench sent a 563 MB payload on a flash drive
> (`D:\mpgd26_data_for_windows`, with its own `README.md` mapping every
> file to the script that reads it).  **19 of the 21 inputs below are now
> on this machine**, and every figure family except three rebuilds here.
> Slide 9 -- the live blocker -- is rendered and placed in the deck.
>
> Read [the reproduction record](#what-the-flash-drive-changed) at the
> bottom before using the tier tables: some of their rows are now stale,
> and one figure needs non-default flags that the tables do not mention.

**2026-08-27.** Written on the Windows laptop after slide 9's micro-TPC figure
turned out to be unrebuildable here. This is the inventory: what still builds
on a machine with no `/media/dylan` and no `~/CLionProjects`, what does not, and
**exactly which files to copy** to change that.

Every path below was checked on this box, not inferred from the docstrings.
Paths are **relative to the repository root** unless they start with `/` or `~`.

---

## The situation in one paragraph

`mpgd26/slides/mpgd26_talk.pptx` holds **104 pictures. 89 of them exist nowhere
else on this machine** — only 15 have a byte-exact copy in
`mpgd26/slides/assets/img/`. The repository's top-level `.gitignore`
blanket-excludes `*.png`, which silently catches `assets/img/` too, so a clone
arrives with almost none of the deck's images and no way to rebuild most of
them. **The .pptx is currently the archive as well as the deck**, and it is
git-ignored itself. Do not lose it.

`mpgd26/slides/NOTES.md` § *"Images are intentionally not tracked in git"*
already flags this and suggests carving `mpgd26/slides/assets/` out of the
ignore rule. This document is the argument for doing it.

### The 15 that are safe here

Byte-exact in `assets/img/` **and** regenerable from committed inputs:

| asset | slides |
|---|---|
| `ear2_onfig_{1_target,2_neutrons,3_collimation,5_station}.png` | 4, 5, 6, 7 |
| `x17_story_top_{1,2,3}_*.png` | 8, 9, 10 |
| `x17_story_bot_{1_boost,2_spectrum,3_detect}.png` | 11, 12, 13 |
| `campaign_overview_{timeline,highlight}.png` | 43, 44, 45 |
| `ear2_hall_photo.jpg` | 4–7, 61 |
| `photo_station_topdown.jpg`, `photo_arm_outside.jpg` | 40, 41 |

(Slides 11 and 12 match the reversed-boost renders, so that swap did land.)

---

## The environment here, before any of it

| | |
|---|---|
| `.venv` | **does not exist.** Every command in `mpgd26/README.md` and `NOTES.md` starts `../.venv/bin/python` — none run as written |
| Anaconda python | numpy / matplotlib / pandas / PIL present, **pyvista absent**. 8 make scripts import it (`make_{anim,bench,chamber,ear2,microtpc,ntof,sps,target}.py`, and everything through `style.py`) |
| PIL from Git Bash | needs anaconda's DLL dirs on `PATH` or it throws `ImportError: DLL load failed while importing _imaging`: `export PATH="$HOME/anaconda3:$HOME/anaconda3/Library/bin:$HOME/anaconda3/Library/mingw-w64/bin:$HOME/anaconda3/Scripts:$PATH"` |
| poppler | **present** via texlive (`pdftocairo`, `pdfimages`), so the `engineer_package/figures/*.pdf` route works |

**First job on this machine is a real venv with pyvista** (0.46.5 / VTK 9.5.2
renders every scene here correctly — proven on the EAR2 and X17 sets this
week). Until then the render scripts have no interpreter.

---

## Copy these — ranked by payoff per byte

Nothing in this table is on the laptop; all 16 paths were checked and all 16
are missing. Source paths are as the code names them; `/media/dylan/data/x17/…`
and `~/x17/…` are the same tree (data disk and its mirror), and `qa_config`
uses the `~/x17` form.

### Tier 1 — one small file, one figure back

| # | copy | to | unlocks |
|---|---|---|---|
| ~~1~~ | **DELIVERED** -- `repo_relative/mpgd26/data/mx17_impulse_response.npz`, exported already | `mpgd26/data/` | **slide 9**, `microtpc.png` -- done, and in the deck |
| 2 | `…/mx17_3/wft/efficiency/efficiency_breakdown.json` -- **STILL MISSING**, not on the drive | mirror the tree, or pass a path | `efficiency_breakdown.png`, `efficiency_residual_tail.png` |
| ~~3~~ | **STALE ROW** -- `make_efficiency_map.py` has read `ray_hit_miss_list.csv` since 56e05dc, and that **is** on the drive | -- | `efficiency_map_sliding.png` |
| ~~4~~ | **DELIVERED** -- `metrics_run_57_perdet.csv` *and* the run_32 `wf/*.root` | `~/.cache/mpgd26_status/` | all twelve `status_*.png` |

For 2–3 the run key is `sat_det3` →
`~/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/`

#### Recipe for #1

The file is **not** a straight copy of a bundle — `scenes_microtpc` reads
`grid_ns`, while `wft.calib.CalibrationBundle.save()` writes that array as
`grid`. Export it on the bench:

```python
import numpy as np, os
B = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
     'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/'
     'mx17_3/wft/calib_bundle_r06')
z = np.load(os.path.join(B, 'arrays.npz'))
np.savez_compressed('mx17_impulse_response.npz',
                    grid_ns=z['grid'], tmpl_x=z['tmpl_x'], tmpl_y=z['tmpl_y'])
```

Two 1-D arrays per plane — small enough to e-mail. Drop it in `mpgd26/data/`,
then

```
cd mpgd26 && ../.venv/bin/python make_microtpc.py --theme light --right waveforms
```

⚠️ `scenes_microtpc`'s docstring still names `calib_bundle_lp`. Take it from
**`calib_bundle_r06`** instead: `lp` is one of the pre-2026-08-21 bundles with
`c2 > c1` that `mx_june_wft/RETIRE_C2GTC1_2026-08-21.md` retired. The
single-electron template is a measured per-plane response and not part of the
kernel, and r06 is the same fit with the ratio pinned — so the array should be
identical, but r06 is the bundle that is not retired. **Record in the export
which bundle it came from.**

### Tier 2 — a directory, several figures

| # | copy | unlocks |
|---|---|---|
| ~~5~~ | **DELIVERED**, both | all four `share_*` rebuild pixel-identically (worked event 1663, `c2/c1 = 0.60`) |
| ~~6~~ | **DELIVERED** -- only `scripts/plot_geometry.py`, which is all that is imported.  Put at `~/CLionProjects/MX17_Full_Geant/scripts/`, the default, so no env var is needed | all ten rebuild |
| 7 | **STILL MISSING** -- a checkout of **MX17_Geant** | `mx17_board_peel{,_zoom,_slide}.png`.  `scenes_chamber.py` only *re-sourced* its numbers from that repo, so `make_chamber.py` is fine without it |

The event `.pkl` is the only bulky item in tier 2; the bundle itself is small.
Neither Geant repository needs its data — only the geometry/model scripts.

### Tier 3 — bulk, and probably not worth moving

| copy | unlocks | why it is awkward |
|---|---|---|
| `…/mx17_3/wft/events.parquet` + `…/wft/alignment/alignment.json` + `cfg.m3_tracking_dir` | `angle_correlation.png`, `angle_resolution.png` | M3 tracking is a directory of ROOT files |
| `…/g_det3_wknd/…/wft/efficiency/ray_hit_miss_list.csv` | the 0.5 mm efficiency map | one CSV, but a per-ray one |
| `/media/dylan/data/x17/beam_july/analysis/wft/run_79/stat090_0000/mx17_A/merged_prelim.parquet` + one run_32 `*flashOff*A500*.root` | the rest of `status_*.png` | `make_status_plots.py --data <dir>` takes a mirror, and **skips missing inputs with a message instead of crashing** — partial copies are fine |
| `/media/dylan/data/x17/ntof_mm_flash/mm_224709.npz` | `make_flash_slides.py` | its three committed reductions are already in the repo; only the digitiser dump is absent |
| `/media/dylan/data/x17/slim/out_224670/…` + run_145 `imaging_summary.json` | `target_pointing_*.png` | slim ROOT |

---

## What builds here today

All 26 `make_*.py` import cleanly under a pyvista venv except the three that
need MX17_Full_Geant, which exit at import with a clear message.

**Verified by actually rendering this week:** `make_x17.py` (all five story
frames), `make_ear2.py` (all five build frames, both label layouts),
`make_microtpc.py`'s 3-D render.

**Inputs all present, so they should run unchanged:** `make_campaign.py`
(reads `ntof_run_report/data/events_per_subrun.csv`), `make_timeline.py`,
`make_pair_kinematics.py` (committed 394-line reduction — the `--reduce` path
is the one that needs the 64 MB npz), `make_x17_rate.py`, `make_couplings.py`,
`make_photos.py` (both originals are in `mpgd26/photos/`),
`make_chamber.py`, `make_target.py`, `make_sps.py`, `make_report.py`.

> **Corrected 2026-08-28: `make_hv_window.py` was on that list and does not
> belong there.**  It was inferred from an import check, not from a render.
> `--numbers` works, and so does everything on the beam's clock, but the
> efficiency panel (and the `NN %` on the scoreboard) goes through
> `hv_tradeoff.bench_eff_on_ntof_axis`, which opens two files that are on
> **neither** mirror root and were never committed:
>
> | file | recoverable |
> |---|---|
> | `~/x17/response_sim/hv_slope/slopes.json` | **yes** — `ntof_july_analysis/hv_tradeoff/results.json` carries `bench.slope10` / `slope10_err`, which is exactly what `bench_gain_slope()` returns |
> | `…/mx17_det3_saturday_scan_6-27-26/hv_scan{,2}/mx17_3/efficiency_vs_hv_scan{,2}.csv` | **no** — two small CSVs, and the whole of that panel |
>
> **Resolved the same day: all three were on `F:`.**  That is a *different*
> drive from `D:\mpgd26_data_for_windows`, and it carries a much fuller
> `x17` tree — `F:\x17\response_sim\hv_slope\` and
> `F:\x17\cosmic_bench\Analysis\mx17_det3_saturday_scan_6-27-26\hv_scan{,2}\mx17_3\`.
> Copied to `~/x17/response_sim/hv_slope/` and, for the bench tree, to the
> same path under **both** `C:\media\dylan\data\x17` and `~/x17`, since
> different modules name different roots (see *Where it was unpacked*).
> Verified against the repository before use: `bench_gain_slope()` returns the
> committed `results.json` values to the last digit.  Slides 50–55 were
> rebuilt and swapped — NOTES.md, *“Slide 21's build fills the slide now”*.
>
> **Check `F:` before declaring anything unbuildable.**  This file's whole
> *“No regeneration path anywhere”* section was written against `D:`, and at
> least one of its rows may not survive a look at `F:\x17`.
>
> The lesson stands, though, and it is the one already in this file's *“Two
> ways this fails quietly”*: **an import is not a render.**  Nothing else on
> the list above has been rendered here either.

The `pdftocairo` figures (`charge_sharing_schematic`, `unsharing_depth_bias`,
`event_display_3d`, `angular_resolution`, `spatial_residuals`,
`time_resolution`) rebuild here too — their source PDFs in
`mx_june_cosmic_qa/engineer_package/figures/` **are** tracked, and poppler is
installed. They are all 2026-07-14 hit-chain vintage; read NOTES.md before
putting any of them back on a slide.

---

## No regeneration path anywhere

| picture | why | where the only copy is |
|---|---|---|
| `atomki_angular_correlations.png` | `pdfimages` of `~/Downloads/Neff n_TOF Analysis Meeting X17 Update 3_24.pdf`, **which is not on this box** | inside the .pptx |
| `atomki_spectrometer_schematic.png` | crop of Fig. 4 of arXiv:1504.00489 — re-derivable, but by hand from the paper | inside the .pptx |
| `ear1_ear2_photo.jpg` → `ear2_hall_photo.jpg` | the source `.jpg` **is** tracked; the crop command is in NOTES.md | both present ✓ |

---

## Two ways this fails quietly — read before trusting a rebuild

1. **`make_bench.py` and `make_figures.py` do not fail without the disk.**
   `geometry.bench_reference_paths()` returns `None` and they fall back to
   *nominal chamber positions and sampled muons*, printing one line. The figure
   builds and looks right; it is simply no longer the measured geometry with
   real reconstructed tracks. Rebuild the bench figures off the bench and the
   deck quietly loses that claim.

2. **`make_microtpc.py` used to write a figure with an empty right half.**
   `scenes_microtpc.strip_waveforms()` returns `None` when the impulse response
   is absent, `draw_waveforms()` returned `False`, and `compose()` ignored it.
   Fixed 2026-08-27: it is now a `SystemExit` naming the missing file. Recorded
   here because the same shape — an optional input, a boolean nobody checks —
   is worth looking for elsewhere.

---

## Re-checking after a copy

Run from the repository root:

```python
import os, sys
sys.path.insert(0, 'mx_june_cosmic_qa')
from qa_config import get_config
sat = get_config('sat_det3').OUT_BASE
A = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
     'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3')
H = os.path.expanduser('~')
for name, p in [
        ('microtpc  impulse response', 'mpgd26/data/mx17_impulse_response.npz'),
        ('share_*   bundle',           A + '/wft/calib_bundle_r06/arrays.npz'),
        ('share_*   worked event',     A + '/wft/calib_work/calib_cache.pkl'),
        ('effic.    breakdown json',   sat + '/wft/efficiency/efficiency_breakdown.json'),
        ('effic.    r2mm map csv',     sat + '/wft/maps/Plot_Data/efficiency_r2mm_cut.csv'),
        ('resol.    events.parquet',   sat + '/wft/events.parquet'),
        ('status    recovery csv',     H + '/.cache/mpgd26_status/metrics_run_57_perdet.csv'),
        ('ntof3d    MX17_Full_Geant',  H + '/CLionProjects/MX17_Full_Geant/scripts/plot_geometry.py'),
        ('peel      MX17_Geant',       H + '/CLionProjects/MX17_Geant/scripts/model/plot_mx17_model.py')]:
    print('%-28s %-8s %s' % (name, 'OK' if os.path.exists(p) else 'MISSING', p))
```

---

## What the flash drive changed

**2026-08-27, later the same day.**  `D:\mpgd26_data_for_windows` (563 MB)
arrived with its own `README.md` mapping every file to the script that reads
it.  This section records what was done with it, what still cannot be built,
and -- the part that matters most -- **which figures were verified against the
deck rather than merely re-rendered**.

### Where it was unpacked

The scripts hold *absolute Linux* paths.  On Windows a path beginning `/` is
drive-relative, so `/media/dylan/...` resolves to `C:\media\dylan\...` for
any process whose cwd is on C:.  That makes the "no edits" option work here
too:

| from the drive | to |
|---|---|
| `home/dylan/x17` | `C:\Users\Dyn04\x17` (`~/x17`) |
| `home/dylan/.cache/mpgd26_status` | `C:\Users\Dyn04\.cache\mpgd26_status` |
| `media/dylan/data` | `C:\media\dylan\data` |
| `MX17_Full_Geant_scripts/plot_geometry.py` | `C:\Users\Dyn04\CLionProjects\MX17_Full_Geant\scripts\` |
| `repo_relative/mpgd26/data/mx17_impulse_response.npz` | `mpgd26/data/` |

Three scripts disagree with the drive's own layout about where the *same*
tree lives, so it is mirrored at all three roots (it is only ~10 MB):

* `make_share.py` wants `/media/dylan/data/x17/cosmic_bench/Analysis/...`
* `make_efficiency_breakdown.py` wants `/home/dylan/x17/cosmic_bench/...`
* `common/beam_july_paths.py` wants `$X17_BEAM_JULY`, `/mnt/data/x17/beam_july`
  or `~/x17/beam_july` -- **none** of which is where the drive puts
  `beam_july`.  Export `X17_BEAM_JULY=C:/media/dylan/data/x17/beam_july`.

`repo_relative/sps_beam_test_26/.../results.json` was **not** copied: the
tracked file already in the repo is semantically identical (same JSON, only
key order and float formatting differ).  Do not overwrite a tracked file with
it.

### Two environment notes

* **`pyarrow` is required** and was in neither interpreter.
  `make_resolution.py` and `make_status_plots.py` read `.parquet`.
* **`PYTHONIOENCODING=utf-8` is required.**  `make_resolution.py` prints a
  `theta` through `cosmic_micro_tpc_analysis.load_alignment` and dies on the
  cp1252 console with `UnicodeEncodeError` -- a pure console-encoding failure
  that looks like a data failure.

### Verified against the deck, not just re-rendered

Every rebuilt asset was compared pixel-by-pixel with the media part the deck
is actually using.  **18 of 20 came back pixel-identical**, which is the real
result here: fonts, matplotlib and the numerics all agree, so this laptop is
a faithful build environment and not merely a working one.

| | |
|---|---|
| pixel-identical to the deck | `efficiency_map_sliding`, all four `share_*`, `angle_correlation`, `angle_resolution`, and 11 of the 12 `status_*` |
| intentionally new | `microtpc.png` -- the slide-9 re-fit, now byte-identical to the deck because it was placed there |
| **differs** | `status_track_rate` -- see below |

`make_resolution.py` also rewrites the tracked `mpgd26/data/angle_resolution.json`.
The rebuild reproduces every committed value except the 15th decimal of four
`s68_err_deg` entries (a different BLAS).  **Revert it** rather than commit
float noise.

### The one figure whose default flags do not reproduce the deck

`make_efficiency_map.py` run bare gives the **hard-disc** map.  The deck's
slide 27 uses the Gaussian one, and three of its four flags are non-default:

```
python make_efficiency_map.py --gaussian --sigma 3 --vmin 0 --min-rays 1
```

`--min-rays 1` is the one that is easy to miss and impossible to guess: at the
default 5 the map masks 18.7 % of the face instead of 1.2 %, and *looks*
plausible either way.  With all four it is pixel-identical to the deck.

(Its caption still says the sigma is "the same length as the auto-derived
hard-kernel radius".  With `--sigma 3` against an auto-derived 6.32 mm that
sentence is simply untrue -- a static string in `main()` that `--sigma` does
not update.  Left alone here; worth fixing when the figure is next touched.)

### `status_track_rate` -- the deck is on an older reconstruction

The rebuild has the same shape, the same annotations and the same two quoted
numbers as the deck (`no events at all before 0.99 ms`, `29 % ... lands in
3-8 ms`) but a y-axis **~4.4x higher**: the deck peaks near 850 tracks/ms, the
rebuild at 3 725.

The drive's `run_79/stat090_0000/mx17_A/merged_prelim.parquet` simply holds
more tracks than whatever built the deck copy.  It is not a selection
difference -- `x_ok`, `x_ok & x_quality_ok`, `x_ok & y_ok` and all three with
quality were tried and none lands near 850.

**No claim on the slide moves**, because both numbers on it are ratios and
positions rather than counts.  But the absolute normalisation is a real
difference, and which vintage belongs in the talk is a call for whoever knows
what was reprocessed -- so the deck was **left alone**.  Note also that
`CLAUDE.md`'s run_79 caveat applies to this figure: chamber A x-view
connector 8 was dead through run_79, so A-x counts there are masked-channel
counts.

### Still not buildable anywhere on this machine

| what | why |
|---|---|
| `efficiency_breakdown.png`, `efficiency_residual_tail.png` | `efficiency_breakdown.json` is not on the drive.  The script names its own fix: `mx_june_wft/02_efficiency.py g_det3_wknd --max-dropped -1` |
| `mx17_board_peel{,_zoom,_slide}.png` | needs an **MX17_Geant** checkout |
| `status_eff_recovery` (backup slide) | needs `.../mx17_det2_det3_overnight_6-22-26/hv_scan/mx17_{2,3}/efficiency_vs_hv.csv`, not on the drive.  `make_flash_slides.py` skips it with a message rather than failing |
| `atomki_angular_correlations.png` | unchanged -- the source PDF is still only in `~/Downloads` on the Linux box |
