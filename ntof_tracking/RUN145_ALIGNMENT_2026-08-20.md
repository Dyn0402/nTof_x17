# run_145 imaging — the in-plane sign was mirrored, and the chambers are pinwheeled

**2026-08-20.** Dylan looked at the MPGD26 overhead slide and said the
alignment or the calibration had to be off for one or both detectors. Both
were, and they were two different things. This is what was wrong, how it was
measured, and what it changed.

The reconstruction is *not* implicated: `events_prelim.parquet` is unchanged.
Both defects were in the analysis that reads it
(`ntof_tracking/run145_target_imaging.py`) and in the figures built on it.

---

## 1 · The two defects

**(i) The in-plane sign was applied to the angle instead of the position.**
`reco/geometry.py` carries an explicit ALIGNMENT CAVEAT: the sign of the strip
coordinate within the plane is taken from the mapping convention and has never
been survey-verified. The imaging analysis handled it by flipping the *tan*,
by the fitted sign of the pointing slope, and leaving the *position* alone.
That combination is a **mirror of the track about the strip-plane centre**.

It would be harmless if the plane centre were on the beam axis. It is not —
the chambers are **pinwheeled**:

| arm | plane centre (global, mm) | pinwheel |
|---|---|---|
| A | (−16.35, 0, +234.6) | 16.35 mm |
| B | (−234.1, 0, −15.75) | 15.75 mm |
| C | (+17.30, 0, −234.6) | 17.30 mm |
| D | (+234.1, 0, +15.50) | 15.50 mm |

So the mirror displaces the reconstructed source by **2 × pinwheel**, and in
*opposite global directions* for opposing arms.

**(ii) The point-source relation used the wrong lever.** It was written
`tan θ = u / |centre|` with `u` measured from the plane centre. The correct
relation is `tan θ = (x_local − foot_x) / d_perp`, with `d_perp` the
perpendicular distance (234.6 mm, not 235.2) and `foot_x` the local x of the
foot of that perpendicular (= the pinwheel). The lever error is 0.24 % and
irrelevant; the **origin** error is 16 mm, i.e. a spurious 0.07 offset in
tan θ, which is a quarter of the plotted range.

## 2 · The measurement that settles it

Not the focus — the focus is a fitted quantity and would be a circular test.
**The zero crossing of the pointing band.** Where `tan θ = 0` the track is
perpendicular to the plane, so that `u` is the foot of the perpendicular from
the source. It contains no angle scale, no drift velocity and no part of the
bench transfer: it is scale-free.

Robust IRLS line fit over the linear range (30 < |lever| < 130 mm; beyond that
the window truncation flattens the band), pointing-coincident tracks, both
sub-runs:

| arm | measures | old convention | **corrected** | surveyed |
|---|---|---|---|---|
| A | global X | −21.8 mm | **−10.9 mm** | 0 |
| C | global X | **+41.7 mm** | **−7.1 mm** | 0 |
| B | global Z | −38.0 mm | +6.5 mm | 0 |
| D | global Z | +36.2 mm | −5.2 mm | 0 |

**Opposing arms have to see the same source.** Under the old convention A and
C disagree by 63.6 mm; the mirror predicts 2 × (P_A + P_C) = 67.3 mm, and the
residual is the real misalignment. Corrected, A and C agree to 3.8 mm and
every arm lands within ~11 mm of the beam axis.

That is the whole argument. Everything below is corroboration.

## 3 · What else moved (and what did not)

Same tracks, same selection, only the geometry changed — the sign scan in
`scratchpad/signscan2.py`, each convention at its own best k:

| | arm A | arm C |
|---|---|---|
| median axis-miss | 34.6 → **14.8 mm** | 52.5 → **18.1 mm** |
| fraction r < 10 mm | 18.5 → **31.4 %** | 14.6 → **25.0 %** |
| external confirmation | 51.2 → **54.0 %** | 38.9 → **40.4 %** |
| image focus scan k | 0.75 → **1.18** | 0.85 → **1.68** |
| per-track k | 1.18 → 1.17 | 1.22 → 1.31 |
| **in-situ v [µm/ns]** | **36.2 → 36.3** | 34.9 → 32.4 |

Three things to note.

**The two k estimators now agree on arm A** — 1.18 from the image focus scan
against 1.17 per-track. They used to disagree by a factor 1.6, and the focus
scan railed *below 1*, which is unphysical. It railed because the mirrored
geometry made the image unfocusable at any k.

**The headline did not move.** In-situ drift velocity 36.2 → 36.3 µm/ns. It
should not have: the defect was an *offset*, not a scale.

**The external confirmation rate went up.** That is the one check that never
touches the track's own angle scale — the SiPM wall sits 96 mm behind the
strip plane. The read-out order had been fixed empirically ("descending"), so
mirroring the frame flips the order with it and cancels; what does *not*
cancel is the 2 × pinwheel offset, and removing it is the gain.

## 4 · What changed in the code

- `run145_target_imaging.py` — `IN_PLANE_SIGN = -1.0` and `local_x()`;
  `plane_geometry(tr)` returns (d_perp, foot_x); `track_lines` applies the
  sign to the POSITION; the empirical `tan *= sign(slope)` flip is **gone**;
  the pointing fit, both k estimators and `pointing_coincidence` all use the
  perpendicular lever; wall segment order is now **ascending** and plastic
  detn 1 is the negative-u bar (the same empirical mapping, restated in the
  corrected frame). New: `pointing_x_coincident` (the scale-free source fit
  with a bootstrap error) and an `image_at_kphys_coincident` block.
- `run145_note_figs.py` — same fix; it had its own copy of the flip.
- `mpgd26/make_run145_pointing.py` — the frozen `SIGN = -1` is gone; the
  overhead figure is rebuilt (see §5).
- `run145_wall_segment_3d.py` — inherits the fix through `track_lines`.

Outputs are parked, not overwritten:
`<subrun>/pre_align_backup_20260820/imaging_fullcov/` and
`<subrun>/imaging/pre_align_backup_20260820/note_figs_fullcov/`.

## 5 · The deck figure

`run145_overhead_AC.png` is now two panels.

**Left** — the station from above with **both strip planes end to end**. The
old figure cropped to |X| < 130 mm of a ±199.29 mm plane, which is what Dylan
spotted: a third of each plane was outside the frame, arm C's off the top.

**Right** — the quantitative half: every confirmed track back-projected to the
target plane, histogrammed per arm. A and C are opposing and calibrated
independently and they land on the same place, **−9.6 and −8.9 mm, inside the
r = 10 mm bore**; 29 % / 24 % of confirmed tracks fall inside the bore itself.

The figure applies the per-arm in-situ angle scale (A 1.25, C 1.58), one
number per chamber, and says so. That makes "does it focus" partly circular —
but **where** the fans cross is the zero crossing, which is not.

## 6 · Still open

- **The angle scale is per chamber and the spread is large**: k = 1.25 (A),
  1.58 (C), 1.99 (B), 1.70 (D), i.e. in-situ v from 34 down to 21 µm/ns
  against a Magboltz prior of 42.6. A's is credible; the others are not
  obviously physical and no per-chamber drift-HV record was found in
  `run_145/`. Same gas, same nominal field — this needs explaining before any
  of these velocities is quoted as a measurement.
- **B and D's zero crossings are unstable** (−48 ± 13 mm on D in one sub-run).
  D's band has a dead region and a vertical stripe at u ≈ −135 mm; B is the
  known-anomalous chamber. Only A and C should be quoted.
- **The residual ~9 mm** common to A and C is either a real beam/target offset
  or a residual survey error, and this analysis cannot separate them.
- **run_79 has the same two defects** — it shares `run145_target_imaging`'s
  geometry conventions through `run79_merge_prelim`. Not yet redone.
