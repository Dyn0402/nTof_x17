# n_TOF pass 1: moving the downstream analysis onto it (NTOF_PASS_PLAN P7)

*Started 2026-11-20. One row per downstream study: what it now reads, the command, and its headline number on pass 1
next to the retired chain's. Chain script: `rerun/ntof/PASS1_DOWNSTREAM.sh`.*

## What the downstream reads now

| product | where (desktop) | built by | notes |
|---|---|---|---|
| stage 3 as on EOS | `$WORK_DATA/ntof_pass1/stage3/` | copy of EOS `Analysis_pass1/stage3/` (MD5SUMS) | frozen-v columns + `<col>_vt` twins |
| **analysis view** | `$WORK_DATA/ntof_pass1/stage3_vt/` | `rerun/ntof/stage3/vt_view.py` (2.5 min) | `<col>` = stage-3 `<col>_vt`, `<col>_frozen` = frozen; verified column by column against stage 3 |
| reco tree | `$WORK_DATA/ntof_pass1/reco_tree/` | `rerun/ntof/stage3/reco_tree.py` (1.5 min) | canonical stage 2 (ledger_canon.json) merged per (sub-run, arm), old `events_prelim.parquet` layout, `tag` column |
| downstream products | `$WORK_DATA/ntof_pass1/ana/<study>/` | the studies below | regenerable; reports go to EOS / the site when final |

Reader for new code: `sept26_prelim_analysis/pass1_tracks.py` (`read(columns, arms=, runs=, filter=)`, one schema over
all 543 sub-runs; beam-off sub-runs read null flash columns).

**The analysis angle is the v(t) one** (Dylan 2026-11-17): `tanx` in the view = `tanx_vt`, and `gated` = `gated_vt`,
which already holds true |tan| < 0.5. Every migrated study therefore cuts at 0.5, not the 0.6 of the is2_v2 era, and
has no k (k_arm = 1). Scripts that took a k from `kcal_*` or the capsule-focus scale (`stage3_is2_v2_cf`) now take
the angle as it is.

## Studies

(filled in as each one is migrated)

### Capsule pointing per run (`campaign_imaging`, scale-free band crossing) — migrated 11-20
`campaign_imaging --reco reco_tree --runs <production> --no-y` → `ana/imaging_campaign/`. 33 runs imaged (run_126 has too few
tracks, as before).

| | is2-era (`imaging_campaign`, prod reco) | pass 1 |
|---|---|---|
| source x (A, C mean) | −9.32 ± 0.30 (run-to-run sd) mm | **−9.09 ± 0.27 mm** |
| A–C half difference | 1.04 mm | 1.16 mm |
| D z crossing | −3.45 ± 0.83 mm | **−5.55 ± 0.76 mm** |
| stable across the campaign | yes | yes (largest block shift 0.35 mm) |

D moved 2.1 mm: D is now on the L1 latency pin and its in-situ bundle; the crossing is scale-free but not t0-free.
This file feeds the measured beam axis of the pair studies (`det_a_intra --axis measured`).

### Capsule focus (`ntof_cosmics/capsule_focus.py`, x view, nominal axis) — migrated 11-20
`capsule_focus --src stage3_vt --label pass1 --tan-max 0.5` → `ntof_cosmics/results/capsule_focus/focus_pass1.csv`.
Sharpest scale s per |tan| bin (s = 1 means the angles focus the capsule as they are):

| |tan| | A is2_v2 | A pass 1 | C is2_v2 | C pass 1 |
|---|---|---|---|---|
| 0.1–0.2 | 1.050 | 0.925 | 1.075 | 0.950 |
| 0.2–0.3 | 1.050 | 0.975 | 1.125 | 0.950 |
| 0.3–0.4 | 1.075 | 1.025 | 1.075 | 0.950 |
| 0.4–0.5 | 1.075 | 1.000 | 1.050 | 0.925 |

On the v(t) angles the focus sits at or a little below 1 on both arms, where is2_v2 needed 5–12 % more. The median
miss is larger on pass 1 (A 0.1–0.4: 19.8–20.2 mm against 16.2–18.8): pass 1 keeps ~40 % more tracks (seeder min 3,
two-track), and the added ones point at the capsule less often. This is the nominal-axis scan; N-G1 G5 (run_145,
measured source offset, frozen v) is the calibrated version and is not replaced by this table.

### Capsule pose, data side (`capsule_pose data / data_z / data_depth`) — migrated 11-20
`load_data` and `data_z_step` read `'pass1'` (A, C; B, D) instead of `stage3_is2_v2_cf` / `stage3_fullpass` →
`ana/capsule_pose/data*.csv`. Against `capsule_pose_ear2_pool/data.csv` (is2_v2_cf):

| arm | window | tracks is2 → pass 1 | x crossing | y crossing (height) | in-focus fraction |
|---|---|---|---|---|---|
| A | late 20–75 ms | 1.05 M → 1.38 M | −10.12 → **−9.32** | −31.33 → **−30.37** | 0.58 → 0.58 |
| A | mid 12–20 | 0.29 → 0.40 M | −10.34 → −8.54 | −32.39 → −31.52 | 0.51 → 0.50 |
| A | beam 4–12 | 0.27 → 0.78 M | −14.0 → −28.3 | −40.7 → −38.9 | 0.27 → 0.13 |
| C | late | 0.82 → 1.09 M | −10.65 → **−9.99** | −34.64 → **−32.25** | 0.53 → 0.55 |
| C | mid | 0.24 → 0.33 M | −10.93 → −9.90 | −36.64 → −34.07 | 0.45 → 0.47 |
| C | beam | 0.23 → 0.59 M | −5.0 → −10.2 | −47.8 → −42.8 | 0.25 → 0.18 |

The late windows (the capsule-pure sample) keep their purity with 32 % more tracks, the source moves ~0.7–0.8 mm
toward the frame centre in x, and the A–C height difference shrinks from 3.3 to 1.9 mm (the N-G1 C3 finding, now on
the whole campaign). The beam window has 2.5–2.9× more tracks and half the in-focus fraction: the added early tracks are
mostly not from the capsule, so the early-window crossings are background-dominated (use `beam_bgsub`).
**Not redone:** the pose fits (`pose2*`). Their toy is built from Geant4 tracks digitised and reconstructed with the
is2 bundles (`g4_digi g4_<arm>_is2_full`, `ear2_*`); fitting pass-1 data with it would mix chains. It needs the
digitised libraries re-reconstructed with the pass-1 code (plan P5 test 8).

### Capsule focus, split by population (`ntof_cosmics/focus_split.py`) — migrated 11-20
`focus_split --src pass1 --label pass1 --tan-max 0.5` → `ntof_cosmics/results/focus_split/focus_split_pass1.csv`
(5.45 M gated A/C tracks, |tan x| 0.1–0.5, 25 min).

| split | A is2_v2 | A pass 1 | C is2_v2 | C pass 1 |
|---|---|---|---|---|
| all | 1.106 | **1.020 ± 0.002** | 1.088 | **0.974 ± 0.001** |
| 20–80 ms | 1.108 | 1.024 | 1.077 | 0.961 |
| wall fired | 1.101 | 1.017 | 1.086 | 0.961 |
| χ² < median | 1.099 | 1.021 | 1.090 | 0.966 |
| tan x > 0 / < 0 | 0.978 / 1.266 | 0.910 / 1.149 | 1.257 / 0.995 | 1.116 / 0.876 |

The script's own reading: a scale error common to A and C is not geometry (distance errors and axis offsets are
antisymmetric between the two). **On is2_v2 the common part was +9.7 %; on pass 1 it is −0.3 %** ((1.020 + 0.974)/2).
What is left is antisymmetric, A − C = 4.6 %, which is the geometric kind; the sign split is the nominal axis
(the source sits at x ≈ −9 mm). So the v(t) scale with κ removes the common focus tension the old chain had; the
G5 residual (N-G1, run_145, frozen v, A +8.1 / C +4.5 %) should be re-read on the v(t) angles before it is quoted.
The populations still split as before: early (< 10 ms), wall-not-fired, χ² > median and other-arms-fired tracks
do not focus (miss 50–90 mm), i.e. they are not capsule tracks.

### Where the tracks come from in time (`capsule_tof tof / tilt`) — migrated 11-20
`capsule_tof tof --chains pass1,is2_v2_cf`, `tilt --chains pass1` → `ana/capsule_tof/`. Capsule-pointing excess
(fraction of lines within 15 mm of the source, minus a shuffled-angle null) and track rate per ToF bin:

| ToF (ms) | A rate/ms is2 → p1 | A excess is2 → p1 | C rate/ms is2 → p1 | C excess is2 → p1 |
|---|---|---|---|---|
| 4–5 | 47 k → 154 k | 0.073 → 0.031 | 25 k → 93 k | 0.109 → 0.034 |
| 8–9 | 29 k → 76 k | 0.198 → 0.099 | 30 k → 67 k | 0.160 → 0.096 |
| 12–15 | 36 k → 52 k | 0.367 → 0.334 | 31 k → 42 k | 0.303 → 0.308 |
| 20–25 | 34 k → 45 k | 0.447 → 0.447 | 27 k → 36 k | 0.398 → 0.408 |
| 40–50 | 19 k → 25 k | 0.463 → 0.464 | 15 k → 19 k | 0.419 → 0.429 |

After 12 ms pass 1 has a third more tracks at the same purity. Before 10 ms it has 2.5–5× more, almost all of them
not pointing at the capsule; the capsule-pointing rate itself (rate × excess) is about the same or a little higher.
So the early additions are the busy-window background, and the < 12 ms veto of the pair studies stays.
Height (y band crossing, ≥ 20 ms): A −30.4, C −32.1 mm (is2_v2_cf −31.3, −34.6). Source over the whole late sample
(tilt, all heights): A x −9.34 ± 0.01, C x −10.03 ± 0.02, D z −1.7 ± 0.3 mm.

### Same-chamber pairs (`det_a_intra`, `same_chamber_physics`, `same_chamber_compare`) — migrated 11-20, NOT final
Chain options (`same_chamber_compare --chain pass1 --compare cf`, `same_chamber_physics --chain pass1 --compare cf`;
`events --chain pass1`). det_a_intra: `--src stage3_vt --reco reco_tree --imaging ana/imaging_campaign/per_run.csv
--tan-max-true 0.5 --t0-max 300 --axis measured --wall`, all ToF → `ana/same_chamber/pass1_{A,C}`, ≥ 12 ms →
`ana/same_chamber_late/pass1_{A,C}`; comparison tables `ana/same_chamber/cmp_pass1/`. run_104 stat090_0016 (no
flash time) is skipped under the ToF cut (new guard in det_a_intra).

| ≥ 12 ms | A is2_v2_cf | A pass 1 | C is2_v2_cf | C pass 1 |
|---|---|---|---|---|
| tracks | 1.34 M | 1.78 M | 1.06 M | 1.41 M |
| same-chamber pairs (real) | 3,452 | **9,558** | 2,722 | **6,114** |
| pairs < 40 mm apart: real / event-mixed expectation | 15 / 106 | **1,386 / 547** | 51 / 143 | **748 / 353** |
| vertexed real / mixed ratio (all) | 1.00 ± 0.11 | **0.87 ± 0.07** | 1.38 ± 0.16 | **1.09 ± 0.10** |
| vertexed ratio, wall-confirmed | 0.90 ± 0.12 | 0.66 ± 0.07 | 1.28 ± 0.21 | 1.02 ± 0.12 |
| correlated pair fraction a (rate fit, per tracked event) | 0.154 ± 0.012 % | **0.23 ± 0.01 %** | 0.157 ± 0.013 % | **0.22 ± 0.01 %** |
| rate-fit χ²/ndf | 64 / 10 | **669 / 10** | 52 / 10 | 180 / 10 |
| data / Geant4 (digitised route) | 0.13 | 0.19 | 0.12 | 0.18 |
| opening-angle median, correlated | 25.5° | 19.1° | 24.8° | 21.5° |
| wall double ratio (data; G4 e+e− 0.58 / 0.64) | 0.54 | 0.53 | 0.55 | 0.59 |

What changed: pass 1 reconstructs two tracks in one plane (two-track on, 4-pitch minimum), so close pairs come back
(< 40 mm: 15 → 1,386 on A). What it does not settle:
- **The close pairs are not yet known to be real.** They exceed the event-mixed expectation by ~840 (A) / ~400 (C),
  pull the correlated opening angle down (A 25.5° → 19.1°, toward the mixed shape), and vertex less often than mixed
  pairs, which is what split single tracks would do. On the beam-off cosmics (run_149) 0.42 % (A) / 0.33 % (C) of
  single-muon events carry a second track within 40 mm. The is2-era two-track threshold (F: A 1200, C 2400) and the
  split-ab contract were measured on the old bundles; `SAME_CHAMBER_PAIRS.md` says to rerun them after any bundle
  change. Until that is done on pass 1, the pair fraction and its opening angle are upper bounds, not measurements.
- **The rate fit is poor** (χ²/ndf 67 on A): the pair fraction is not a + b·rate on pass 1, so `a` is a summary, not
  a clean correlated fraction.
- **The Geant4 side is the old reconstruction's response** (`DIGI` p1/p2 from g4_digi on the is2 bundles, `F_TWO` from
  the 09-14 overlay bench; flagged `stale` in `cmp.meta.json`). data/G4 0.19 is not a pass-1 number.
- The physics fold (`physics_compare.csv`) still uses the is2-era cosmic angle-resolution table
  (`ntof_cosmics/results/tracking/pooled/resolution.csv`).

Needed to close it: (1) the split-ab contract and F ladder on the pass-1 bundles (T1); (2) g4_digi re-reconstructed
with the pass-1 code (P5 test 8) for DIGI and the pose toy; (3) a pass-1 resolution table from the run_149 A–C line.

### Scintillator stack mapped by the tracks (`ntof_scint_stack extract / ana`) — migrated 11-20
`X17_SCINT_OUT=ana/scint_stack`; `extract --src stage3_vt --runs <production>`; `ana --arms ACD` (B has no angle in
pass 1, so its pointing calibration cannot run; `merge` now merges only the arms run). m = tan (v(t), k = 1).

| | A | C | D |
|---|---|---|---|
| tracks, is2-era fullpass → pass 1 | 2.87 M → 3.55 M | 2.87 M → 2.85 M | 5.65 M → 5.94 M |
| k used | 1.23 → 1 | 1.45 → 1 | 1.62 → 1 |
| λ (slope that the wall edges want, × m) | 0.636 → **0.847** | 0.627 → **0.840** | 0.587 → **0.847** |
| wall offset from survey (mm) | — → +0.9 | — → +3.4 | — → +7.1 |

λ < 1 means the wall edges are fitted best with a shallower slope than the chamber measures. On pass 1 it is
the same on all three chambers (0.84–0.85) where the is2-era k gave 0.59–0.64 with chamber-to-chamber spread: the
chamber-to-chamber part was calibration. The common 0.85 is the beam-electron effect already known from Geant4
(electrons scatter between the gap and the wall; beam/cosmic wall gap, HANDOFF_TRACKING §10–13), not an angle error.
Still from the is2 era: `ana.CAPSULE_XYZ = (−9.3, −31.3, −3.45)`; pass 1 gives x −9.1, y −30.4 / −32.1, z −5.5 (imaging) or −1.7 (tilt).
The report and deck (`make_report`, `make_deck`) are not regenerated yet.
