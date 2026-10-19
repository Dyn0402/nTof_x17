# HANDOFF — fix the stage-3 y mirror, and audit everything it touched

> **STATUS 2026-10-13 (later): §2 done — the code fix, the guard (§2.4), the loaders (§2.3), the digitiser, the
> tests. §3 is NOT done: the desktop, which holds every table, was offline. Run `bash Y_FIX_REGEN.sh` there; it
> regenerates each table from its own sidecar, proves each one is an exact y mirror of the old, runs both guards and
> every downstream rerun. §4 is `Y_FLIP_AUDIT.md` (table filled; "pending" entries come from that run). §6 errata are
> written in the affected docs; notes, board and the y-orientation slide still need the regenerated numbers.**

**Written 2026-10-13.** Read `COORDINATES.md` first; it has the frame, the sign conventions and the evidence. The slide
note is <https://dylan-neff.web.cern.ch/notes/y-orientation.html>.

## 0 · The defect, in three lines

- **What it does:** `sept26_prelim_analysis/build_tracks.py` sets `IN_PLANE_SIGN_Y = −1` and applies it to the y
  **position** (`yl = −(y_p0 − 199.29)`), leaving tan y unflipped.
- **What it should do:** the strip coordinate already points up (FEU channel 0 at the bottom), so the position must
  not be flipped, and **tan y must be flipped** (the reco's convention, as in x, is `tan_raw = −d(strip)/d(depth)`).
- **Effect:** stored global y is mirrored. Correct = −stored, for both `p0_y` and `d_y`. It has been in effect
  since 2026-09-07 (commit that introduced `IN_PLANE_SIGN_Y`; find it with
  `git log -S IN_PLANE_SIGN_Y -- sept26_prelim_analysis/build_tracks.py`).

**Evidence:** cosmic-muon timing (run_149, p = 4×10⁻¹¹), 8/8 plastic bars, liquid B, the capsule nose, and the
wiring. **This is settled; do not re-litigate it.** Do re-run the checks after the fix (§2.4).

## 1 · Goals

1. Fix the convention at the source, with a guard so a mirror cannot come back silently.
2. Regenerate every stage-3 table and every product that depends on absolute y.
3. Audit everything done since 2026-09-07, plus older work that used the same convention, and classify each
   result:
   - **unaffected:** angles between tracks, |tan|, x/z;
   - **cosmetic:** a figure axis or label is mirrored;
   - **numbers change:** a height, a y window, a Geant4/CAD comparison;
   - **conclusion changes.**
4. Correct the record: docs, reports, slide notes, the X17 board, and errata for talk material already shown.

## 2 · The fix

### 2.1 build_tracks

- `local_and_global`: `yl = +(y_p0 − STRIP_MAP_HALF)` and `ty = −y_tan_theta · k`.
- Keep x exactly as is.
- Replace the long 2026-09-07 comment with a short pointer to `COORDINATES.md`. Keep the history in `COORDINATES.md`.
- Record the convention in the stage-3 sidecar or schema: bump `SCHEMA` and add a `y_convention: "up"` field.
  Loaders that currently mirror y check this flag (§2.3).

### 2.2 Everything else that encodes the same choice

Search `IN_PLANE_SIGN_Y`, `STRIP_MAP_HALF -`, `y_p0 - ` and `HALF -` across the repo. Known so far:

- **`ntof_cosmics/g4_digi/digitise.py` (line ~280):** `coord = STRIP_MAP_HALF − v`. This maps simulated v onto the
  strips the way the (wrong) stage-3 convention expects. Real hardware is `STRIP_MAP_HALF + v` (x stays `HALF − u`).
  - After the build_tracks fix, the digitiser must change too, or the sim reco becomes mirrored.
  - Check that the digitised pipeline still reproduces the Geant4 truth y band crossing (−28 mm) afterwards.
- `ntof_tracking/run145_target_imaging.py` (line ~345–445) uses `IN_PLANE_SIGN` on `y_p0`: the older target-imaging
  path.
- `ntof_athens_26/make_overhead_figure.py`, `sept26_prelim_analysis/{capsule_y,hit_maps,chamber_b,pairs,
  beam_cache,make_figures,k_arm,det_a_intra}.py` mention `IN_PLANE_SIGN_Y`. Read each one.

### 2.3 Loaders that already mirror on load

These were added on 2026-10-13 to work around the bug. **After the fix they double-flip.**
- `sept26_prelim_analysis/capsule_pose.py` mirrors y on load.
- Check `capsule_tof.py`, and grep for `-p0_y`, `-d_y` and `mirror`.

Make them conditional on the stage-3 `y_convention` flag, or remove the mirror and require the new schema.

### 2.4 The guard

Add an automated check that fails if stored y is mirrored:
- **Primary:** `ntof_cosmics/y_orientation_check.py cosmic` re-run on the regenerated run_149 tables. The
  stored-higher arm must be hit **earlier** (median Δt sign test).
- **Cheap secondary, on any beam stage-3 table:** the late (≥ 20 ms) y band crossing of A and C must be negative
  (≈ −30 mm, the nose). `capsule_tof.crossing` on `(p0_y, hgt − p0_y)` does this.
- Put the secondary in the stage-3 build (warn loudly, or refuse to write) and the primary in a test or the chain
  script.
- Run the existing wft/stage tests.

## 3 · Regenerate

Stage-3 tables are built by `sept26_prelim_analysis.campaign_tracks` from the stage-2 full passes. Check
`HANDOFF.md`, `DATASET.md` and the `*.log` beside each table for the exact command and k.

| table | location | notes |
|---|---|---|
| `stage3_is2_v2_cf` | `~/x17_local/` (root disk; the data disk was full) | current analysis chain (capsule-focus k), A/C |
| `stage3_is2_v2` | `/media/dylan/data/x17/sept26_prelim/` | muon k |
| `stage3_fullpass` | same | production, all four arms (B/D) |
| `stage3_campaign`, `stage3_is2_v1_pilot`, `stage3` | same | older; regenerate only if something live reads them, otherwise mark retired |
| run_149 cosmic tracks | `ntof_cosmics/results/tracking/k_run_147/tracks_*` (and other `k_*`) | via `ntof_cosmics/cosmic_tracks.py` |
| scint stack | `/media/dylan/data/x17/scint_stack/tracks` | built from stage 3 (`ntof_scint_stack.extract`, `v_mm = y_local`) |
| g4_digi samples | `/media/dylan/data/x17/ntof_cosmics/g4_digi/*` | after the digitiser fix |

Check disk space first (`df -h`). The data disk has been near full; see the memory note on data-disk writes.
Compute is not a constraint.

## 4 · The audit: what has used absolute y since 2026-09-07 (and before)

Start from the consumer list. This grep gave ~130 files:

`grep -rlE "p0_y|d_y\b|target_y_mm|y_local|v_mm|CAPSULE_V|CAPSULE_SHIFT|angle_to_beam|tany|IN_PLANE_SIGN_Y|pred_(sipm|plastic|ls)|vy_mm|joined_y" --include=*.py`

The heaviest areas are `sept26_prelim_analysis` (38), `ntof_cosmics` (19), `ntof_athens_26` (16) and
`ntof_tracking` (10). Build a table in `Y_FLIP_AUDIT.md`: file/product, what y it uses, verdict class (§1.3), action.

**Known items, most consequential first:**

1. **Capsule height** (`capsule_y.py`, `HANDOFF_CAPSULE_Y.md`, `<out>/capsule_y/`). The 2026-09-14 conclusion "CAD
   height wrong by ~30 mm (y_s ≈ +33)" is the mirror. Corrected: −31/−35 mm, at the nose. Note that
   `capsule_pose.py` finds the capsule ~4 mm lower than the Geant4 placement. Update the handoff's Outcome and any
   "placement constant".
2. **Same-chamber pairs:**
   - `same_chamber_physics.CAPSULE_V = (−29.5 + 32.2, 50.7 + 32.2)` and `same_chamber_compare.CAPSULE_SHIFT_MM = 32.2`
     place the capsule at the valve end. The vertex-y windows, the "73 of 91 excess pairs over the capsule" statement
     and the Geant4 vertex-y comparison (`cmp_vertex_y.csv`) are mirrored.
   - The opening angles and the 1 : 8 rate are not affected.
   - Re-run `same_chamber_physics`, `same_chamber_compare` and `make_same_chamber_report`/`deck` after regeneration.
3. **Scintillator stack** (`ntof_scint_stack`):
   - All v-maps are mirrored. The liquid B/C "no rise toward the PMT" open question resolves to a rise toward the PMT.
   - Check `pred_sipm_*`, `pred_plastic`, `pred_ls` and any v acceptance cut. The geometry is nearly symmetric in v
     (`LS_OFFSET_V` ≤ 0.7 mm), so matching rates should barely move. Verify that.
   - Republish the scint-stack note.
4. **Capsule focus / angle scale** (`ntof_cosmics/focus_split.py`, `capsule_focus.py`, `focus_offset.py`, HANDOFF_TRACKING §21–22):
   - The "capsule height inside/outside" split used `CAPSULE_Y = (−29.5, 50.7)` in the stored frame, so it selected
     the wrong end.
   - The k results themselves are y-sign-free. Re-run the split for the record.
5. **Two-chamber pairs, vertex imaging, IPC/X17 opening angles** (`pairs.py`, `pair_qa.py`, `source_imaging.py`,
   `ntof_athens_26/pair_vertex_imaging/*`, `opening_angle.py`, `acceptance*`, `campaign_acceptance.py`, `fold`):
   - Angles are invariant.
   - Vertex y images, y windows and any acceptance computed with the source at +33 mm (`capsule_y` constant) need
     checking. The geometry is close to y-symmetric, so expect small changes, but verify.
6. **Tracking QA web page** (`trackqa_shards.py`, `TRACKQA_WEB.md`, `/x17/qa-tracks.html`): any y display or y cut.
   The shards need regenerating with stage 3.
7. **Hit maps and chamber B** (`hit_maps.py`, `chamber_b.py`): display orientation.
8. **Published material shown to others** (record errata; do not silently edit history):
   - Athens talk figures (`ntof_athens_26/make_overhead_figure.py --projection y`, the side view; `y_image.py`).
   - MPGD26 slides (`mpgd26/`): check for any side view or height claim.
   - Notes on the site: pair-vertex-y, pair-vertex-capsule-image, capsule-imaging, same-chamber-pairs,
     scint-stack, ac-insitu-angles, beam-off-cosmics.
   - The X17 board (`x17-board` skill): add a log entry and fix any height statements.
9. **Before 2026-09-07:** y positions kept the raw strip direction, but the tan was not flipped either, so y
   position and tan disagreed (corr < 0). Any 3D direction from that period had a mirrored vertical *direction* with
   correct positions. That is the build_tracks comment's "what it broke" list (`target_y_mm`, `angle_to_beam_deg`,
   LS/plastic crossings, opening angles between chambers). Products from that era using inter-chamber opening angles
   were already flagged then. Confirm they were redone after 09-07; after the fix they will be right for the first
   time in both position and direction.
10. **Geant4 comparisons that used the frame shift.** Anything that moved Geant4 by +32.2 mm to match data, or
    compared data heights with `HE3_GAS_Y`. The corrected data sit near Geant4's nominal placement (≈ −4 mm).

**For each "numbers change" item:** rerun, record before/after in `Y_FLIP_AUDIT.md`, and update its report or note.
**For each "conclusion changes" item:** flag it at the top of the relevant handoff/STATUS and tell Dylan.

## 5 · Coordination

- Another session ran the capsule pose fit (`capsule_pose.py`, commit 315a50d; `HANDOFF_CAPSULE_POSE_FIT.md`
  Outcome). It mirrors y on load and may still have uncommitted edits in the main checkout. Coordinate: rebase onto
  its work, and switch its loader to the schema flag.
- Branching: work on a branch. CLAUDE.md git rules: merge to main and push when verified, never force-push.
- Unrelated dirty files in the main checkout (`garfield_sim/garfield_test.py`, `ntof_cosmics/results/yield_gate/cut_scan.csv`,
  `mx_june_wft/drift_consistency/`) are not part of this. Leave them alone.

## 6 · Deliverables

- The fix plus the guard and test; regenerated stage-3 tables (sidecars say `y_convention: up`).
- `Y_FLIP_AUDIT.md` (repo root): the full table, before/after numbers.
- Updated `COORDINATES.md` (mark the fix done, with the commit), CLAUDE.md (shorten the warning to the rule once fixed),
  the affected handoffs/STATUS, and the reports and notes republished.
- A board log entry.
- A short slide added to the y-orientation note: "fixed on <date>, what moved".
