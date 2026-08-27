# Rebuilding mpgd26 figures away from the bench

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
| 1 | the det3 impulse response, exported (recipe below) | `mpgd26/data/mx17_impulse_response.npz` | **slide 9**, `microtpc.png` — the live blocker |
| 2 | `…/mx17_3/wft/efficiency/efficiency_breakdown.json` | mirror the tree, or pass a path | `efficiency_breakdown.png`, `efficiency_residual_tail.png` |
| 3 | `…/mx17_3/wft/maps/Plot_Data/efficiency_r2mm_cut.csv` | same | `efficiency_map_2mm.png` |
| 4 | `~/.cache/mpgd26_status/metrics_run_57_perdet.csv` | `~/.cache/mpgd26_status/` | one of the six `status_*.png` |

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
| 5 | `…/mx17_3/wft/calib_bundle_r06/` + `…/mx17_3/wft/calib_work/calib_cache.pkl` | `share_cartoon`, `share_kernels`, `share_build`, `share_decompose` (`make_share.py`, worked event 1663) |
| 6 | a checkout of **MX17_Full_Geant** (or set `$MX17_FULL_GEANT`) | `setup3d_1…9.png` and `ntof_plan.png` — ten pictures, slides 30–41. `make_{ntof,ntof_plan,anim}.py` refuse to import without it |
| 7 | a checkout of **MX17_Geant** | `mx17_board_peel{,_zoom,_slide}.png` |

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
`make_hv_window.py`, `make_photos.py` (both originals are in `mpgd26/photos/`),
`make_chamber.py`, `make_target.py`, `make_sps.py`, `make_report.py`.

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
