# Coordinates and signs — the n_TOF detector frame

**Read this before touching any sign, axis or strip-direction constant, and before quoting a height.**
Written 2026-10-13, after stage-3 global y was found to have been mirrored since 2026-09-07.
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
- 2026-09-07 did the opposite (`IN_PLANE_SIGN_Y = −1` on the position, tan unflipped). That **mirrors global y**.
  Fixing it in `build_tracks` and regenerating stage 3 is `HANDOFF_Y_FLIP_FIX.md`.

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
1. Re-run `y_orientation_check.py cosmic`. The stored-higher arm must be hit **earlier**.
2. Re-run the capsule y band crossing (`capsule_tof.py tof`). It must sit at the nose (≈ −30 mm), not at the valve.

The fix session should turn these into an automated check.

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

## 5 · Other conventions worth knowing

- Global x is common to A and C: `U_HAT` A = +x, C = −x. A shift of the source in global x moves A's and C's images
  in opposite local directions.
- The wall bars run along v and are read at both ends (`detn = 2g + {1, 2}`). The mean of the two end times is
  independent of position along the bar. Which end is the top is not documented yet; add it here when known.
- The plastic bars are split in u; their single PMT is at the top. The liquids are as in the table above.
