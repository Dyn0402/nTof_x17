# n_TOF tracking: what we did not understand, and what we found when we looked (2026-11-24)

The re-audit of the morning (`report.html`) left a list of things that were unexplained or inconsistent. Dylan asked
for them to be written down and looked into before the data are re-reconstructed. This note gives each item with what
was measured in the afternoon, the conclusion, and what it means for the re-reconstruction. The tests are X1, X2 and X3
(`tests/X1_gap_puzzle`, `tests/X2_g4_digi`, `tests/X3_small_items`: scripts and `results.txt`). Everything uses the
existing data.

Status words: **explained** (a cause is measured), **partly explained** (causes are ruled out, one remains as an
inference), **open** (we do not know).

## Summary

| # | item | status | consequence for the re-reconstruction |
|---|---|---|---|
| 1 | The drift-time map did not predict the cosmic angle map | explained for A; partly for C and D | none for the fit; the second pass should store two more quantities (item 1) |
| 2 | Chamber C: rectangular block and large gradient in the drift time | partly explained | none; do not use C's drift-time map for its angle scale |
| 3 | Chamber C: the angle scale does not follow the drift time | open | none; the spline map measures the scale directly |
| 4 | Chamber A: the scale differs between the two views | explained | none |
| 5 | Chamber A: gap gradients from muons and from beam tracks disagree along y | explained | none |
| 6 | The beam pointing scale reads 0.95 to 0.99 | explained as an estimator effect, to about 5 % | none; the cosmic lines are the scale reference |
| 7 | Geant4 events reconstruct 3 to 10 % steeper than their truth | explained | fix the digitisation before it is re-run (two items) |
| 8 | The two Geant4 builds disagree on the chamber distance by 5 mm | answered (documents + κ test) | the data scale is 2.1 % steep: change the distance at the stage 3 of pass 2 (Dylan decides); retire the `nose` library |
| 9 | Chamber D with every switch on lost real tracks | explained: the sizing run used a wrong pin | the campaign uses the right bundle; the D comparison must be redone there |
| 10 | Chamber D loses tracks at x_local 60 to 120 mm in runs 135 to 150 | partly explained | none; state as a limitation |
| 11 | All chambers lose about 6 % of tracks in runs 139 to 147 | partly explained | none found; state as a limitation |
| 12 | The step at run_150 | mostly explained | none |
| 13 | The residuals of A and C on one cosmic line are anticorrelated | explained | none |
| 14 | Split children at the t0 limit: 27 % or 40 % | explained | none |
| 15 | Runaway fits flagged as good quality | open, small | a table-level sanity flag |
| 16 | The two-track unit test gives different results on repeated calls | open, small | none for the campaign |
| 17 | Items of earlier studies that no test addresses | open | listed for the campaign |

## 1. Why the drift time did not predict the cosmic angle map

**What was unclear.** On the cosmic bench, the duration of the drift-time column maps the drift gap, the angle scale
follows the gap, and the tilt of the drift lines follows the gradient of the gap. In the morning the same test at
n_TOF gave a weak result. The drift-time maps correlated with the cosmic scale at only −0.3 on chambers A and C. On C
and D the maps were too large to be a gap.

**What we measured.**
- The drift-time maps measure the length of the column, in every chamber. We compared the mean charge profile along
  the drift direction in the regions where the column is long with the regions where it is short. One profile is the
  other stretched in time (A: stretch 1.11 and 1.09, χ² 17/10 and 6.7/10, against 235 and 200 with no stretch). There
  is no slope along the profile, so electron attachment is not the cause (attachment below about 0.1 per cm).
- **For chamber A the gap picture holds in the interior.** With the smooth scale of the spline map, the part of the
  scale common to the two views follows the drift time with slope −0.33 ± 0.06 (correlation −0.63). A gap change
  predicts −0.34. The tilt follows the gradient of the gap with coefficient a = 0.53 ± 0.04, the bench value, and this
  explains 55 % of the tilt that a smooth mesh shape leaves.
- **Chamber A has a diagonal crease.** The spline map shows a ridge along x_local = −y_local. The drift-time map has a
  stripe at the same place: across the diagonal the column is 8 to 10 % shorter exactly where the tilt potential peaks
  (+0.8 mm) and the scale peaks. The correlations are −0.70 with the potential and −0.89 to −0.94 with the scale. Beam
  tracks and beam-off tracks agree (0.92). The bench found a diagonal crease of the cathode foil in one mounting.
- **The outer 50 mm is the fringe field.** There the tilt does not follow the gradient of the gap, and the scale falls
  to −0.28 (A) and −0.36 (C).

**Why the test of the morning failed.** There were five reasons.
1. The per-cell scales of the old 50 mm map were too noisy. The spline map removed this noise.
2. The edges were included. The fringe field is not a gap effect.
3. The test was made per view. The gap predicts only the part of the scale that is common to the two views.
4. On C and D the column runs into the end of the readout window. The n_TOF window is 20 samples of 60 ns, which is
   1200 ns. On C and D the last bin still holds 0.26 to 0.41 of the plateau, and 14 to 18 % of late tracks are cut
   (24 to 26 % inside C's block). The 90 % point is then compressed where the column is long.
5. The scale of C has a part that is not a drift-velocity effect (item 3).

**Conclusion.** For A the bench picture is confirmed at n_TOF: gap, tilt and scale are consistent in the interior. The
spline map already contains all of it and more (the fringe field), so nothing is gained by adding a drift-time term on
A. For C and D the drift time is real geometry but cannot be turned into an angle correction.

**For the second pass.** The fit does not change. Three stored quantities would make these maps exact: the depth profile
of every track (already a switch, `reco.STORE_PROFILES`, now on in every test configuration), an unquantised column
end, and the number of depth bins cut by the readout window. On C and D a drift-time map should use the 50 % point or
a fit of the whole profile, and leave out the outer 25 mm.

## 2. Chamber C: the rectangular block and the large gradient

The drift time of C is more than 15 % long in a rectangle at x_local −175 to −75 mm and y_local −100 to +25 mm, and it
falls by 10 % per 100 mm along both axes.
- The block appears only in the drift time. Charge, number of strips, χ², t0 and the fraction of flagged strips are
  normal inside it.
- Its edges are sharp (10 to 15 mm) and appear in both views. The x strips cross the y edges, so no effect of one
  view's electronics or masks can make them. Three of the four edges are in the middle of a connector.
- It does not follow the gas. Between the slowest and the fastest fifth of the runs in drift velocity, the block
  changes by 0.94 ± 0.05. A pattern of gas composition predicts 1.23.
- The large gradient grows by 1.13 to 1.20, between the value for a fixed gap (1.045) and the value for gas (1.44 if
  all of the velocity change is water). So part of the gradient may be the gas. This is an inference.

**Conclusion.** The block is geometry or field: a step of about 2.4 mm in the gap, or a region of weaker drift field.
We cannot separate the two. The depth profile inside the block rests on 44 tracks. Chamber D behaves like fixed
geometry too (its gradient grows by 1.06 to 1.10, a fixed gap predicts 1.08). Read as a gap, D spans 4 to 5 mm, which
no test excludes.

## 3. Chamber C: the angle scale does not follow the drift time

The common scale of C follows the drift time with slope −0.14 to −0.21. A gap change predicts −0.49. The scale
correlates with y (−0.65) and has a band of +0.10 to +0.14 at y_local about −150 mm. Along y the drift time and the
scale fall together, which no change of drift velocity can do. **We do not know what this is.** It is stable, the
x view shows it where the slope of the cosmic lines does not vary with y, and the beam pointing confirms the map.
The spline map corrects it whatever its cause.

## 4. Chamber A: the scale differs between the two views (explained)

Over the whole chamber the scales of the x and y views correlate at 0.28. In the interior they correlate at 0.85. The
x scale falls at the x edges and the y scale at the y edges. In the interior the difference follows the difference of
the two views' drift times with slope about −1, which is a stretch of one view's time axis and not a gap.

## 5. Chamber A: gap gradients from muons and from beam tracks (explained)

The bench audit found that the gap-gradient fields along y from beam-off muons and from late beam tracks do not
correlate (0.04). In the interior they agree (0.91, with a ceiling of 0.94 to 0.99 from half samples). The
disagreement is in the outermost 25 mm in y. There the drift time depends on which way the track leans: a track
leaning outward is shorter by 0.2 to 0.7 in ln T, because its column leaves the active volume. The two samples have
different angle mixes (6 % of late tracks lean outward, 14 % of beam-off tracks).

## 6. The beam pointing scale (explained as an estimator effect)

The slope of the measured against the expected tangent of capsule tracks is 0.990 (A), 0.965 (C) and 0.946 ± 0.009
(D). On Geant4 truth, with no detector, the same estimator gives 0.97 to 0.99 (nose library) and 0.92 to 0.99
(EAR2-beam library), mostly from scattering and from the cut |tan| < 0.5. The cosmic lines give D 0.997 ± 0.004.
We make no scale correction. Geant4 cannot give the bias to better than about 5 %, and its order of the chambers does
not match the data.

## 7. Geant4 events reconstruct steeper than their truth (explained)

The slope of the reconstructed on the true tangent is 1.03 to 1.05 for the nose library and 1.09 to 1.12 for the
EAR2-beam library. It does not depend on energy, charge, particle or the definition of the truth. There are two causes.
- **Both libraries: the reconstruction removes distortions that the simulation never contained.** The Geant4 events are
  reconstructed with the data bundles. These divide the tangent by kw (A 0.976 / 0.965) and apply the bench bow block.
  The digitiser simulates an ideal chamber. With the fitted transverse speed taken directly (tan = w / v_true), the
  nose library reads 0.996 to 0.999. Straight muons show the same: 1.027 / 1.034 through the bundle, 0.994 / 0.998
  directly.
- **EAR2-beam library only: a 5 mm depth error.** In that Geant4 build the drift gas runs from w = 5.1 to 35.1 mm.
  `digitise.py` had the mesh fixed at 30.1 mm, so the last 5 mm of every track arrived at once. Re-digitising 500
  events with the right depth gives 0.995 to 1.002.

**Fixed now:** the digitiser refuses steps beyond its mesh, and the mesh plane is set per library with
`$X17_G4_W_MESH` (35.1 for `ear2_L19p5`).

**To do before the digitisation is re-run** (not done):
- A bundle for the simulation with kw = 1, no bow block, no map and v equal to the simulated velocity.
- The muon closure gate on the median ratio of the direct tangent, 1.00 ± 0.005 per chamber and view. The "fitted/true
  1.001" quoted in `rerun/ntof/PASS1_DOWNSTREAM.md` came from a line fit pulled by outliers; the median ratio on the
  same 240 muons is 1.037 ± 0.005.

**Numbers that rest on the affected products:** the EAR2 pair response (p1 0.762 / 0.728, p2 0.398 / 0.433), the capsule
pose fits with `POSE2_LIB=ear2_L19p5`, the EAR2 rows of the audit's Geant4 tables, and the statement that data/G4 of
about 0.10 does not depend on the library. For the nose library, the scaled tangent changes the |tan| < 0.5 acceptance
by 5 to 6 %, so p1 is about 5 % low and p2 about 10 % low (an inference, not recomputed).

The "broken" y view of chamber C in the EAR2 product was one runaway fit that dominated a correlation computed without
a cut. With |tan| < 0.5 the correlation is 0.983.

## 8. The chamber distance in the two Geant4 builds (answered in the evening, `MESH_DISTANCE.md`)

The two Geant4 builds place the mesh at 234.6 mm and at 239.6 mm from the axis. The analysis assumes 234.6 mm.
- **The answer.** The survey measured the mylar window faces, and the as-built chamber has a 5 mm gas gap between the
  window and the cathode. Therefore the mesh is most likely at 239.6 mm (A, C) and 239.1 mm (B, D).
- **A check in the data.** κ comes within 1 % of 1 in all four views when the A–C separation is set to 479.2 mm.
- **The consequence.** The data angle scale is 2.1 % too steep, and pointing is unchanged. The fix is a geometry
  constant plus a refit of the map and of κ, with no re-reconstruction. Dylan decides.
- **For Geant4.** The `nose` library is retired. No new simulation is needed.

## 9. Chamber D with every switch on (explained: a wrong pin)

The morning comparison found that D lost 6.6 % of its real late tracks with every switch on, and that 90 % of the
matched tracks changed with a t0 shift of +15 ns. The sizing cluster ran D on `bundles_osc/D_E2_beam`, whose trigger
pin is the run_149 value (x +18.3, y +20.0 ns). Pass 1 uses the L1 pin (x −17.8, y −11.2 ns). For the 78 to 81 % of
candidates with identical seed, window and rank, t0 shifts by +36.0 / +31.1 ns, which is exactly the pin difference.
**The D numbers of the morning (the loss of real tracks, and "the oscillating-channel hot list is a net loss") are
therefore confounded and must be measured again in the campaign,** which uses the canon bundle `D_E2_beam_L1`.

## 10. Chamber D at x_local 60 to 120 mm in runs 135 to 150 (partly explained)

Late D tracks in this band fall from 49 to 51 per 1000 late triggers to 30 to 46, and recover from run_152. The band
is the x view of FEU 1, connectors 3 and 2. It is not a cut (all tracks are lost, the gated fraction does not change),
not a seed slot taken by hot channels (0.1 % of events reach the cap), and not a plain loss of hits (the hit occupancy
stays low after the tracks recover). 91 % of the band's tracks come with plastic bar 2 of D. In run_141 the trigger
rate of bar 2 falls by 17 % and the track yield per bar-2 trigger falls by 23 %. Why bar 2 and this band lose events in
beam only is not settled.

## 11. All chambers lose about 6 % of tracks in runs 139 to 147 (partly explained)

The loss starts sharply inside run_139, at sub-run 0007 (about 05:00 on 5 August). It has the same size in A, C and D
(−6.3 to −6.5 %), is uniform in position and angle, and sits in the high-charge tail (A: −15 % above 2400 ADC, +6 %
below 1200 ADC). The trigger mix is constant and the loss does not follow the beam intensity. It is intermittent in
run_147 and gone from run_150. A loss common to three chambers with separate readout points to the readout or the
reconstruction of high-charge events. We could not check the hits, because only run_145 has hit files on the desktop.

## 12. The step at run_150 (mostly explained)

Nothing changed in the run configurations. The drift velocity rose gradually through the 22-hour cosmic run_149, and
run_150 follows a 23-hour gap with no beam. The step of t0 relative to the pin (−2 to −6 ns) is 70 to 110 % what the
velocity change predicts from the slope measured inside runs (−0.47 to −0.90 ns per percent). The first sub-run has
12 % more charge because the beam had just returned: most runs start high after a beam stop (median +4 %), and run_150
followed the longest stop.

## 13. The residuals of A and C on one cosmic line are anticorrelated (explained)

The line is built from the two chambers' positions at the mesh. Moving both points into the gap along their tracks
removes the correlation, at 20 mm depth on the mapped slopes. The line referenced at the mesh mixes about 4 % of each
chamber's slope error into the line. It is not a gain: the residual widths grow by 3 to 5 % with the points moved.

## 14. Split children at the t0 limit (explained)

The 27 to 28 % of the t0 study is late children (from 25 ms) over the campaign: A 27.2 ± 0.4 %, C 28.2 ± 0.4 %. The
36 to 44 % of the morning is all times on 8 acquisitions, where 85 to 90 % of the children are early. Over the campaign
at all times the fractions are 34.6 % (A) and 35.5 % (C).

## 15. Runaway fits flagged as good quality (open, small)

Between 0.5 and 2 % of fitted views have an absurd result (a position of 10⁶ mm, a tangent of 10⁷) while `quality_ok`
is true. None of them is `gated`, so no analysis number is affected. A table-level sanity flag (position inside the
strip plane, |tan| below 5) would remove the trap for a study that reads ungated rows.

## 16. The two-track unit test (open, small)

`wft/tests/test_two_track.py::test_accepted_split_replaces_its_parent` failed once under load and passes otherwise.
Repeated calls of the fit on the same window in one process give different χ² (18919, 18287, 13330). The production
reconstruction is reproducible (the reference reproduces pass 1 to 10⁻⁶ mm on 20,251 tracks), so this is a property
of the test or of state carried between calls. We did not find its origin.

## 17. Items of earlier studies that no test of today addresses

- The second cause of a lost second track: both views are fitted, but the x–y association builds one track (about one
  third of the losses in the injection test). No switch addresses it.
- The excess of correlated same-chamber pairs of A at 250 to 400 ns in the column-time difference (t0 study; at most
  8 % of A's correlated pairs).
- The small curl of the tilt field above its noise (χ² 114 and 155 for 49 points).
- Whether the seed ranking by charge with the cap after the veto loses real tracks on D. Only the single
  configuration can show it.
