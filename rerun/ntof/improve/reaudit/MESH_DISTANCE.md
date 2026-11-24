# The distance of the n_TOF chambers from the beam axis (2026-11-24)

This note answers item 8 of `OPEN_QUESTIONS.md`. Two Geant4 builds place the mesh of each chamber at different
distances from the beam axis, 234.6 mm and 239.6 mm. The data analysis uses 234.6 mm. We looked at which value is
right, and what a wrong value does to the data and to the Geant4 comparison. This is a reading of documents, the CAD
and the code. Nothing here is a new measurement, and nothing in the products was changed.

## The answer

The mesh of A and C is most likely at **239.6 mm** from the beam axis, and the mesh of B and D at **239.1 mm**. The
analysis therefore places every chamber about 5 mm too close to the axis.

## The evidence

- **What was surveyed.** The survey of July 2026 measured the distance between the **mylar entrance-window faces** of
  opposite chambers: 40.9 cm for A–C and 40.8 cm for B–D, each halved
  (`~/CLionProjects/MX17_Full_Geant/GEOMETRY_COORDINATE_CONVENTION.md`, lines 60–78). The window is the only mylar
  face a tape measure could reach, since the kapton cathode is inside the gas. That last point is an inference.
- **Where the mesh is behind the window.** The CAD gives 35.01 mm from the window face to the readout surface. The
  as-built module in Geant4 (`MX17_Geant/shared/MX17ModuleGeometry.hh`, `flangeGap_mm = 5`) has a 5 mm gas gap
  between the window and the cathode, then the 30 mm drift gap.
- **Where 234.6 mm comes from.** 234.6 = 204.5 + 30.1. The value 30.1 is `W_STRIP` in `ntof_tracking/reco/geometry.py`,
  "mylar front → strip plane (40 µm mylar + cathode + 30 mm)". It assumes that the window lies on the cathode, as in
  the legacy Geant4 stack. The as-built module does not have this.
- **The two Geant4 builds.** The `nose` library was built at MX17_Full_Geant `3d97437` (2026-07-23) on the legacy
  stack: its drift gas runs from w = 0.1 to 30.1 mm. The `ear2_L19p5` library was built from `ear2-beam` (`19a6a71`),
  which contains the as-built module (`ae8313f`, 2026-08-06): its drift gas runs from w = 5.1 to 35.1 mm. In both,
  w = 0 is the window face at 204.5 mm.
- **A test in the data.** κ is the ratio of the angle velocity, measured on the A–C cosmic lines, to the timing
  velocity v(t) (`rerun/ntof/stage3/vt_score/vt_kappa.json`). v(t) is 30 mm divided by the drift-column duration, so it
  does not depend on the chamber distance. With 469.2 mm between the A and C strip planes, κ is 0.968 / 0.973 (A x / y)
  and 0.978 / 0.979 (C). With 479.2 mm, κ becomes 0.989 / 0.993 and 0.999 / 0.999. So κ comes within 1 % of 1 in both
  views of both chambers. This agrees with 239.6 mm. However, other biases of the column duration could also give a
  few per cent, so it is not a proof.
- **What remains unknown.** We do not know whether the window was bulged when it was measured. The modelled sag is
  8 mm at 3 mbar. If the survey caught a bulged window, the mesh is farther out still.

## What a 5 mm error does

- **The angle scale of the data.** The A–C cosmic lines are the scale reference of the spline map. Their true tangent
  is the difference of the two positions divided by the separation of the strip planes. A separation of 469.2 mm in
  place of 479.2 mm makes every true line tangent 2.1 % too large. Therefore the spline map calibrates the data
  tangents 2.1 % too steep. The B–D lines carry the same factor.
- **Pointing at the capsule.** Pointing does not change. Extrapolating back to the source multiplies a tangent that is
  2.1 % too large by a lever arm that is 2.1 % too short, so the two errors cancel. For the same reason the capsule
  position cannot pin the distance.
- **The opening angle.** If the opening angle is computed from the track directions, each leg is off by about 0.3° at
  tan 0.3 and 0.5° at tan 0.5.
- **Chamber D.** The B–D lines give D a scale of 0.997 at the current distance. At the new distance this becomes
  about 1.02, while A and C come to about 1.00. So D would read about 2 % steeper than A and C. We do not understand
  this.
- **Geant4.** The `nose` library has the chambers physically 5 mm too close, and it lacks the 5 mm gas gap and the
  window. Re-digitising cannot move a chamber, so the `nose` library cannot be repaired. The `ear2_L19p5` library
  already has the right geometry. It only needs the right mesh plane in the digitiser (`X17_G4_W_MESH=35.1`).

## What to do (Dylan decides)

1. **Data:** set the strip-plane distance to 239.6 mm (A, C) and 239.1 mm (B, D) in `ntof_tracking/reco/geometry.py`
   and `run_config.json`, then rebuild stage 3, refit κ and refit the spline map. The reconstruction works in the
   local chamber frame, so no re-reconstruction is needed. The natural moment is the stage 3 of a second pass. The
   phase-4 test campaign compares configurations on the same geometry, so it is not affected.
2. **Geant4:** retire the `nose` library and use `ear2_L19p5` only (`../phase4/CAMPAIGN.md` section 10).
3. **State as a limitation:** the distance is taken from the survey of the window faces and the CAD, not measured on
   the mesh. A bulged window at the time of the survey would move it farther out.
