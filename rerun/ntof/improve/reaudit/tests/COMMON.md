# Common rules for the n_TOF tracking re-audit test agents (session eb8200b3, night of 2026-11-23/24)

You are one of several test agents working for an orchestrating session. Read this file fully, then your own brief.

## Repo and data
- Repo: /home/dylan/PycharmProjects/nTof_x17 (python: `.venv/bin/python` from the repo root). Read CLAUDE.md there.
- THE DATASET IS FINAL. Never propose a new run, scan or hardware check. If something cannot be answered from the
  existing data, say so as a stated limitation.
- The n_TOF analysis view (the current baseline) is `$WORK_DATA/ntof_pass1/stage3_r2_final/` (WORK_DATA=/media/ucla/work):
  543 tables `tracks_run_<run>_<subrun>.parquet` (+ .meta.json), 192 columns, one row per track. It is the rebuilt stage 3
  with v(t) angles, the in-situ angle map (`rerun/ntof/stage3/angle_map/angle_map_r2.json`), the censored-window cut and
  quality flags. Key columns: `arm` (A/B/C/D), `gated` (the analysis selection: true |tan|<0.5 + cuts), `x_local`,
  `y_local` (mm, chamber frame), `tanx`, `tany` (mapped, analysis angles), `tanx_nomap`, `tany_nomap` (before the
  in-situ map), `tan_raw_x/y`, `x_w`, `y_w` (fitted transverse speed, mm/ns), `x_p0`, `y_p0`, `x_t0`, `y_t0`,
  `x_q_u50`, `x_q_u90`, `x_q_uend` (column time quantiles, ns), `drift_t_end_ns`, `x_q_sum`, `y_q_sum`,
  `t_since_flash_ns`, `p0_x/p0_y/p0_z` and `d_x/d_y/d_z` (global point and direction; `_nomap` twins), `dca_source_mm`,
  `wall_<arm>`, `plastic_<arm>`, `x_split_child`, `t0_railed`, `low_charge`, `d_faint`, `censored_window`,
  `x_bow_a`, `x_bow_h`, `v_t`, `ntof_run`, `event_class`.
- Helpers (read them before re-deriving anything): `sept26_prelim_analysis/pass1_tracks.py` (reader, `SOURCE_XZ_MM`,
  `cosmic_tags()/cosmic_filter()`, `nomap()`), `rerun/ntof/improve/track_quality.py` (pointing, `source_x`),
  `rerun/ntof/improve/line_residuals.py` (A-C cosmic lines), `rerun/ntof/improve/angle_map.py` (the map),
  `rerun/ntof/improve/bd_map.py` (D map from B-D lines), `rerun/ntof/stage3/map_view.py`, `COORDINATES.md`,
  `docs/PARAMETERS.md`. Findings so far: `rerun/ntof/improve/README.md` (findings 1-23) and
  `rerun/ntof/improve/AUDIT_FIX_ROADMAP.md`.
- Existing caches you may read (do not overwrite): `$WORK_DATA/ntof_pass1/ana_r2/track_quality/lines.parquet`
  (A-C cosmic lines on true beam-off runs, unmapped slopes), `.../ana_r2/track_quality/bd_lines.parquet`,
  `.../ana_r2/track_quality/angle_map.json`. Older caches under `$WORK_DATA/ntof_pass1/ana/` predate the stage-3 rebuild.
- +y is up (the beam direction). Never decide an axis sign from a self-consistency test. Chamber B has positions only
  (no angle). Beam-off (cosmic) data is selected by registry class (`pass1_tracks.cosmic_filter`), never by "time is null".
- Late tracks (t_since_flash >= 20 ms, better >= 25 ms) are the clean capsule sample; before ~12 ms most A/C tracks are
  ringing fakes.

## Working rules
- Write ALL your scripts and outputs under your own directory `$WORK_DATA/claude-jobs/eb8200b3/<your test id>/`
  (create it). Do NOT write into the repo, do NOT git add/commit/push, do NOT modify any existing product or cache.
  Another session is working in the repo tonight (it owns `rerun/paper1_gap/` and `$WORK_DATA/claude-jobs/e1a7a26e/`):
  do not touch either.
- The desktop is shared tonight. Use at most 3 worker processes, start heavy commands with `nice -n 10`, and read only
  the parquet columns you need (pyarrow `columns=`). The ucla disk is NTFS: reading all 543 tables takes minutes, so
  build one small cache of your own first and work from it.
- No condor submission and no ssh unless your brief explicitly says so. lxplus only through the literal alias
  `ssh -o BatchMode=yes lxplus` (never a node name).
- Figures: ordinary document figures (about 3:2, 10.5 pt), PNG, saved in your directory.

## What to report back (your final message)
Plain text, complete sentences, numbers with uncertainties, in this order:
1. The answer to each question of your brief, with the numbers.
2. How each number was obtained (sample, cuts, counts) so it can be re-derived, and the paths of scripts and outputs.
3. What you could not verify, what surprised you, and anything that contradicts the documents. Do not explain a
   surprise away: report it. Say explicitly which statements are measured and which are your inference.
Keep it under about 900 words; put long tables in a file and give the path.
