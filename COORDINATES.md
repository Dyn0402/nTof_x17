# Coordinates and signs — the n_TOF detector frame

**Read this before touching any sign, axis or strip-direction constant, and before quoting a height.**
Written 2026-10-13, after stage-3 global y was found to have been mirrored since 2026-09-07.

> **Status (2026-10-13): fixed in the code, branch `y-flip-fix` (see `Y_FLIP_AUDIT.md`).** New stage-3 tables
> say `y_convention: "up"` in their sidecar (schema `sept26_prelim/tracks/2`). Tables built before the fix are
> still mirrored until `Y_FIX_REGEN.sh` has run on the desktop. Read any stage-3 table through
> `build_tracks.y_up(df, path)`, which reads the flag and handles both.
Evidence: <https://dylan-neff.web.cern.ch/notes/y-orientation.html> (`ntof_cosmics/make_y_orientation_deck.py`).

## 1 · The global frame (EAR2)

- **+y is the beam direction, vertical, floor → ceiling.** The spallation target is below; the beam dump is above.
- The frame is right-handed. In the top-down view, +z points right (east) and +x points up (north)
  (`MX17_Full_Geant/GEOMETRY_COORDINATE_CONVENTION.md`).
- The arms: **A at +z, B at −x, C at −z, D at +x.**
- Origin: the nominal capsule position and the chamber-frame centre (the surveyed station centre). The capture
  source images at x ≈ −10, z ≈ −4 mm in this frame (`capsule_pose.py`). Whether that is the beam being off the frame
  or the capsule being off a centred beam is NOT settled; see `<out>/capsule_pose/report.html`.
- Each chamber has a local frame (u, v, w).
  - u is the x-view strip coordinate, v the y-view strip coordinate, w the drift depth.
  - `ntof_tracking/reco/geometry.py`: `U_HAT`, `W_HAT`, `DetTransform`.
  - `DetTransform.R` is a rotation about global y only, so **local v is global +y for every arm**.
- The capsule is mounted nose-first: its aluminium nose (tip) is at the bottom, facing the beam, and the valve is at
  the top.

## 2 · The strip coordinate and the two sign constants

`mx17_m1_map.csv`: `x_position_mm` and `y_position_mm` = 0.78 × FEU channel, from 0 to 398.58 mm.
The plane centre is at 199.29 mm (`STRIP_MAP_HALF`).

**x view.**
- The strip index runs along −u. `build_tracks.IN_PLANE_SIGN = −1` is applied to the **position only**.
- This was measured 2026-08-20 on the target image. The pinwheel puts each plane centre ~16 mm off the beam axis, so a
  wrong choice would split opposing arms' images. That is an external constraint.
- The tan is not flipped. This fixes the reconstruction's tan convention:
  **`tan_raw = −d(strip coordinate)/d(depth)`**.

**y view.**
- Hardware (Dylan, 2026-10-13): **low FEU channels are at the bottom of each chamber**, so the strip coordinate
  increases upward (+v = +y).
- With the tan convention above, the correct mapping is:
  - **position NOT flipped:** `y_local = +(y_p0 − 199.29)`;
  - **tan y flipped:** `tan_y = −tan_y_raw · k`.
- In code: `run145_target_imaging.IN_PLANE_SIGN_Y = +1`, `TAN_SIGN_Y = −1`, `local_y()`, `local_tan_y()`;
  `build_tracks` imports them. The Geant4 digitiser maps simulated v to strip `HALF + v`.
- 2026-09-07 → 2026-10-13 did the opposite (`IN_PLANE_SIGN_Y = −1` on the position, tan unflipped). That
  **mirrored global y**. Fixed 2026-10-13 (`HANDOFF_Y_FLIP_FIX.md`, `Y_FLIP_AUDIT.md`).
- Before 2026-09-07 neither was flipped: y positions were right but y directions were mirrored.

## 3 · Why the y error survived, and the rule that prevents a repeat

The 2026-09-07 test was corr(y_local, tan_y) > 0: a track from the source must move outward. That test only fixes the
**relative** sign of position and tan. Flipping the position or flipping the tan both pass it. The two choices differ
by a global mirror of the axis, and **every internal consistency check is blind to a global mirror.** These all pass
either way:
- angles between tracks;
- the A–C cosmic line;
- the band slope;
- track–scintillator matching in u.

**Rule: an axis orientation is decided only by an external, direction-carrying fact, and the fact is written down
next to the constant.** For y, the facts on record are:

| test | what it uses | result | where |
|---|---|---|---|
| cosmic-muon timing | muons go down; wall mean-of-ends time on the n_TOF clock, beam-off run_149 | stored-higher arm hit **later**, 88 % vs 7 %, p = 4×10⁻¹¹, step ±2.3 ns | `ntof_cosmics/y_orientation_check.py cosmic` |
| plastic PMT timing | plastic PMTs on top (Dylan) | 8/8 bars later at stored-high v, 5–40σ; flat in u | `y_orientation_check.py pmt` |
| liquid PMT position | B, C vertical with PMT on top (survey, Geant4); A, D horizontal with PMT at +u (control) | B fires ×2 toward stored −v; A, D 10–20× at +u | `ntof_scint_stack` eff_map |
| capsule nose | beam captures sit in the upstream (bottom) Al, Geant4 −28 mm | stored +31 / +35 | `capsule_tof.py`, `capsule_pose.py` |
| wiring + strip map | channel 0 at the bottom; y = 0.78 × channel | raw v points up | this file §2 |
| July hits analysis | no position flip | source "~3 cm low" | `mx_july_beam_qa/TRACKING_RUN55_ANALYSIS.md` |

**After any change to a sign, a strip map or a frame:**
1. Re-run `y_orientation_check.py cosmic` then `verdict`. The stored-higher arm must be hit **earlier**
   (`verdict` exits non-zero otherwise; `tests/test_y_convention.py` runs it where the data are staged).
2. The capsule y band crossing must sit at the nose (≈ −30 mm), not at the valve. This is automatic:
   `y_guard.late_y_crossing` runs in every `build_tracks.build` and on every combined table, and refuses to
   write a table that reads > +3σ.

## 4 · What a y mirror does and does not change

- **Unchanged:**
  - every angle between two tracks (opening angles, X17 and IPC);
  - |tan|;
  - x and z positions and the x/z imaging;
  - counts and efficiencies that do not cut on y.
- **Mirrored:**
  - `p0_y`, `d_y`, the sign of `tany`, `target_y_mm`, `angle_to_beam_deg`;
  - vertex heights and source/capsule heights;
  - scintillator v-predictions and v-maps;
  - every comparison of a height with Geant4 or CAD.
- **Possibly changed:** any cut on a y window (capsule y ranges, `in_bore` along y), and any acceptance or efficiency
  correction that is asymmetric in y.

## 4b · The cosmic bench and SPS (Paper I)

- **Bench: tan = +d(strip coordinate)/dz_up in both views** (`wft.reco.tan_of`, `docs/PARAMETERS.md` §3.1).
  - External facts: M3 z points up (ray files `Z_Up` = 1302 mm, `Z_Down` = 24 mm; L0 is the top plane, cosmic muons go
    down), and the alignment maps positions and the M3 tangents with the same rotation and handedness.
  - Measured on pass 1 (2026-10-31, `rerun/paper1_charge/`): w rises with the M3 tangent on 245/247 (row, view) pairs
    (both slots); the two exceptions are det2 6-17 (half-dead X, excluded). Later charge comes from higher up: the
    readout board is down and electrons drift down onto the mesh. The bow's h > 0 therefore means the mesh is higher.
- **SPS: the tangent sign is not pinned.** Positions at normal incidence carry no depth information, and the yaw of a
  rotated mount enters the position alignment only through cos φ. Only |tan| is quoted from SPS.

## 5 · Other conventions worth knowing

- Global x is common to A and C: `U_HAT` A = +x, C = −x. A shift of the source in global x moves A's and C's images
  in opposite local directions.
- The wall bars run along v and are read at both ends (`detn = 2g + {1, 2}`). The mean of the two end times is
  independent of position along the bar. Which end is the top is not documented yet; add it here when known.
- The plastic bars are split in u; their single PMT is at the top. The liquids are as in the table above.
