# Follow-up of the outstanding questions: items 2, 8, 10, 16, 17 and 18 (audit tag q, 2026-11-24)

This note answers six of the twenty outstanding questions of `AUDIT_2026-11-24.md` with the data on disk. Nothing in the
reconstruction, the bundles or the pass products was changed.

Scripts (`rerun/paper1_gap/audit/`): `q16_mountings.py` (item 16), `q2_amplitude.py` (item 2). The product of velocity and
drift time (items 10, 17) and the censoring census (item 8) were computed inline. Their numbers are in
`$WORK/audit/q16/vT_rows.csv` and in the tables below. Outputs: `$WORK/audit/q16/{q16.json, q2_amplitude.json, vT_rows.csv}`
with `WORK=/media/ucla/work/claude-jobs/e1a7a26e/gap`.

## Summary

1. **Item 8 (the 6-22 run of chamber A).** The drift time of this run is censored by the depth grid of the fit. The bundle
   has K = 24 depth bins (1440 ns), and 70 % of the tracks have charge in the last bin. Two things caused it. The logged
   drift voltage was normal: 1000.25 V at 0.337 µA, the same current as on 6-28. The gas in chamber A was wet that night,
   and the drift velocity was 24–26 µm/ns (water ledger: at least 1.6 % water). Chamber B ran in the top slot the same
   night at 39.6 µm/ns. Since q_u90 is capped near the end of the grid, the "nearly flat" 6-22 gap map is an artefact of
   the censoring. Therefore 6-22 is not evidence of a third cathode shape. The registry already classes these rows as
   skeptical ("micro-TPC angle unusable"), and the censoring is the likely cause.
2. **Item 17 (the mean gap).** The product of the ladder velocity and the median q_u90 is the same in every mounting of a
   chamber, even when the velocity changes by a large factor. On chamber A the gap is 28.2 mm on 6-25 (v = 22 µm/ns at
   500 V), against 28.1 and 28.2 mm in the golden mounting (v = 37 µm/ns at 1000 V). So the gap is a property of the
   chamber and not of a mounting. The 1.8 mm shortfall from the design is shared by chambers C and E (28.3 and 28.2–28.5 mm).
   It is therefore most likely a common construction height or a common scale of the estimator, and not something
   particular to chamber A.
3. **Item 10 (chamber C).** The audit had the order of C's rows reversed. In time order (short 00:58, longer 01:58, long
   06:03) the velocity *rose* from 16.8 to 18.9 to 25.0 µm/ns. Chamber D, on the same gas the same night, rose from 32.5
   to 35.3 to 38.1 µm/ns. The gas therefore dried out during the night, and the gain rose with it (saturated fraction 0, 33,
   then 92 %). In C's two early rows 21–22 % of the tracks reach the end of the depth grid, so their drift times are
   censored. For this reason the gap from velocity × drift time is low there (18.7 and 24.3 mm), while the last row gives
   28.3 mm. D gives 29.3–29.6 mm in all three rows.
4. **Item 16 (chamber E's two mountings).** E's gap map is the same in both mountings (correlation 0.97 ± 0.03; the
   split-half ceiling is 0.92 ± 0.04). By the same test chamber A's two mountings correlate only 0.39 ± 0.03 (ceiling 0.99).
   An angle map made in one E mounting works in the other: E 6-24 → 6-23 on Y gives 1.468° against 1.471° for the
   mounting's own cross-validated map. E was re-seated between the two days (offsets moved 1.6 and 7.8 mm, rotation 0.8°)
   and stayed in the top slot. So a re-seat does not by itself change the cathode.
5. **Item 18 (what moves the foil).** We have no direct measurement. The mounting history narrows it, however. Chamber A
   was not moved between 6-22 and 6-23 (offsets −216.5/−163.0 and −216.4/−163.6 mm). Before 6-25 it was re-seated in
   the bottom slot, shifted 27 mm in y. Before the 6-26 weekend scan (started 6-27 00:26) it was moved to the top slot. Its cathode changed shape at
   both moves. Chamber E was re-seated in place and its cathode kept its shape. We cannot say whether the size of the
   move, the slot, or the handling made the difference.
6. **Item 2 (the amplitude of the gap term).** Saturation lowers the fitted amplitude b in every saturated mounting. With
   unsaturated events only, A 6-25 goes from 0.22 to 0.33 ± 0.02 (0.38 ± 0.03 with the field also built from unsaturated
   events), B from 0.37 to 0.41, and the golden mounting from 0.46 to 0.48. b is quoted at β = 0.68. Rescaled to the β of
   the operating point ((1 + β)/1.68, with β 1.43 at 500 V and 0.87 at 1000 V from R4), the unsaturated 6-25 value is 0.48
   and the golden value is 0.53. These two agree within about 10 %. On chamber C the low b comes from the two early rows,
   whose drift time is censored and whose gas was changing (b 0.09 and 0.08). The last row gives 0.23. Its Y view
   (0.42) is normal, and only its X view (0.14) is low. Chamber E remains unexplained: b is 0.17 and 0.09 with no
   saturation and no censoring.

## Item 8: censoring by the depth grid

The fit represents the charge profile in K depth bins of 60 ns. The registry rule is K = 24 on the bench, or 30 at a
drift voltage of 500 V or less (`docs/PARAMETERS.md`). The rule assumes the velocity that the voltage gives in dry gas.
On a wet night the column is longer than the grid. Table 1 gives, for each gate row and the 6-22 rows, the end of the grid
and the fraction of tracks whose profile reaches its last bin ("rail").

**Table 1.** Grid end and rail fraction (single-track events, both views).

| row | grid end (ns) | rail | median q_u90 (ns) |
|---|---|---|---|
| A 6-22 long run (+ 16 mesh-scan sub-runs, 6-22 test run) | 1440 | 0.64–0.78 | 1210–1340 |
| A 6-25 | 1800 | 0.000 | 1059 |
| A golden (6-27 sat, 6-28 p2) | 1320, 1260 | 0.017, 0.019 | 617, 611 |
| A 6-27 drift scan 300 / 500 / 700 V | 1740 / 1740 / 1440 | 0.05 / 0.03 / 0.07 | 1382 / 1376 / 914 |
| B (3 rows) | 1200 | 0.014–0.017 | 640–650 |
| E 6-23 / 6-24 | 1320 / 1440 | 0.025 / 0.086 | 617 / 724 |
| C long / longer / short | 1440 | 0.17 / 0.27 / 0.21 | 927 / 1037 / 887 |
| D long / longer / short | 1260 / 1380 / 1440 | 0.03 / 0.04 / 0.14 | 643 / 701 / 754 |

The censoring on 6-22 is severe. The low-voltage sub-runs of the golden mounting are not censored by the grid (3–5 %).
So the grid does not explain item 7, the flattening below 700 V, and that item stays open (R4 also excluded the end of
the readout window there).

Since a censored q_u90 compresses every map built from it, the per-mounting work should check the rail fraction of
every row before a q_u90 field is used. The direct angle maps do not use q_u90 and are not affected in this way.

## Items 10 and 17: velocity × drift time

The ladder velocity v (mean of the two views) times the median q_u90 is proportional to the gap, as long as the drift
time is not censored. Table 2 normalises it to 28.2 mm on the golden mounting of chamber A.

**Table 2.** Gap from v × q_u90 per row, in time order within each mounting.

| row | start | drift V | v_x / v_y (µm/ns) | rail | saturated | gap (mm) |
|---|---|---|---|---|---|---|
| A 6-25 | 06-25 11:40 | 500 | 22.0 / 21.8 | 0.000 | 0.87 | 28.18 |
| A 6-27 sat | 06-27 22:49 | 1000 | 37.0 / 36.5 | 0.000 | 0.26 | 28.05 |
| A 6-28 p2 | 06-28 01:34 | 1000 | 37.7 / 37.1 | 0.000 | 0.29 | 28.25 |
| E 6-23 | 06-24 03:16 | 600 | 35.6 / 38.7 | 0.000 | 0.00 | 28.47 |
| E 6-24 | 06-24 12:22 | 600 | 30.7 / 32.6 | 0.101 | 0.00 | 28.17 |
| C short | 06-26 00:58 | 700 | 16.8 / 17.3 | 0.223 | 0.00 | 18.71 |
| C longer | 06-26 01:58 | 700 | 18.9 / 19.7 | 0.214 | 0.33 | 24.31 |
| C long | 06-26 06:03 | 700 | 25.0 / 26.3 | 0.114 | 0.92 | 28.32 |
| D short | 06-26 00:58 | 700 | 32.5 / 32.5 | 0.068 | 0.62 | 29.25 |
| D longer | 06-26 01:58 | 700 | 35.3 / 34.8 | 0.000 | 0.68 | 29.58 |
| D long | 06-26 06:03 | 700 | 38.1 / 38.3 | 0.000 | 0.75 | 29.64 |
| B short / longer / long | 06-22 19:18 – 06-23 06:58 | 1000 | 39.0–39.1 / 39.4–39.6 | 0.000 | 0.49–0.52 | 31.2–31.8 |

(The rail fractions in Table 2 use the analysis selection, so they differ slightly from Table 1.)

The product holds to 1 % across a factor 1.7 in velocity on chamber A, across 14 % on E and across 15 % on D. This is a
strong check that the ladder velocity and q_u90 measure the same drift. It also means the gap of chamber A is the same in
the 6-25 mounting as in the golden mounting. We have no absolute measurement of the gap of any chamber, so the
1.8 mm shortfall from the design value cannot be placed in the chamber or in the scale of the estimator. However, the
shortfall is not particular to chamber A.

## Item 16: chamber E's two mountings

`q16_mountings.py` compares the two mountings of E, with chamber A's two mountings as the control. It makes three
comparisons. The first is the gap map (the q_u90 field), as a full map and with its quadratic part removed (the "fine"
part). The second is the mesh height from the position route, fitted as a quadratic with a separate offset plane per
row. The third is an angle map (tilt only, the R3 "dir_tilt" arm) fitted on all of one mounting and applied to the other.

**Table 3.** Gap maps of the two mountings (correlation over the common coverage, bootstrap errors). The split-half value
is the correlation between two independent halves of the larger mounting. It is roughly the highest correlation the
noise allows. It uses half the events, so on E it lies below the full-statistics value. The rms of the difference of the
two full maps is 0.23 mm on E (each map 1.0 mm rms) and 1.01 mm on A (each map 0.8–1.0 mm rms).

| chamber | full map | fine part | split-half, full | split-half, fine |
|---|---|---|---|---|
| E (6-23 vs 6-24) | 0.97 ± 0.03 | 0.68 ± 0.21 | 0.92 ± 0.04 | 0.52 ± 0.26 |
| A (6-25 vs 6-27/28) | 0.39 ± 0.03 | 0.33 ± 0.05 | 0.99 ± 0.00 | 0.97 ± 0.02 |

**Table 4.** Angle-map transfer: s68 of θ − θ_ref (deg) in the target mounting.

| map → target | view | pass 1 | constant | transferred map | own CV map |
|---|---|---|---|---|---|
| E 6-23 → 6-24 | X | 2.141 | 2.284 | 2.203 ± 0.046 | 2.120 |
| E 6-23 → 6-24 | Y | 1.436 | 1.840 | 1.408 ± 0.040 | 1.324 |
| E 6-24 → 6-23 | X | 2.405 | 2.617 | 2.372 ± 0.078 | 2.406 |
| E 6-24 → 6-23 | Y | 1.627 | 2.150 | 1.468 ± 0.055 | 1.471 |
| A 6-25 → 6-27/28 | X | 0.923 | 1.042 | 0.884 ± 0.006 | 0.725 |
| A 6-25 → 6-27/28 | Y | 0.805 | 0.934 | 1.059 ± 0.008 | 0.621 |
| A 6-27/28 → 6-25 | X | 0.876 | 1.065 | 0.893 ± 0.011 | 0.742 |
| A 6-27/28 → 6-25 | Y | 1.030 | 1.305 | 1.068 ± 0.014 | 0.705 |

On E the transferred map recovers 84 % (6-23 → 6-24, Y) and 100 % (6-24 → 6-23, Y) of the gain of the mounting's own map.
On A it recovers about half of the gain on X and makes Y worse in one direction. The half that does transfer on A is the
static mesh term, as the audit concluded.

The mesh heights from the position route have large errors on E (0.3–0.9 mm on the shape coefficients) and agree within
2.6σ. On A the curvatures agree within 0.2 mm per (100 mm)², except kww on the X view (2.4σ). The linear terms of A
differ by 0.4–0.7 mm per 100 mm. A rigid tilt of the chamber in the telescope gives the same linear term, so those terms
measure the mounting and not the shape of the mesh.

**Verdict.** E's cathode kept its shape when E was re-seated in place. For E, one map for both days would have worked.
The re-finalization can still fit E per mounting, which is the safe default.

## Item 2: the amplitude b with and without saturation

`q2_amplitude.py` refits b exactly as R3's `fit_b` does, on subsets. The errors come from a bootstrap over events with
the field fixed.

**Table 5.** b (both views together; X and Y in brackets), quoted at β = 0.68.

| mounting | saturated | all events | unsaturated | unsaturated, field also unsaturated | saturated only |
|---|---|---|---|---|---|
| A 6-25 | 0.87 | 0.218 ± 0.007 (0.20, 0.22) | 0.334 ± 0.015 (0.31, 0.34) | 0.381 ± 0.026 (0.24, 0.49) | 0.202 ± 0.007 |
| A golden | 0.28 | 0.456 ± 0.006 (0.37, 0.51) | 0.476 ± 0.008 (0.40, 0.51) | 0.461 ± 0.009 | 0.416 ± 0.011 |
| B | 0.51 | 0.367 ± 0.008 (0.34, 0.38) | 0.410 ± 0.009 (0.41, 0.41) | 0.430 ± 0.009 | 0.322 ± 0.012 |
| D | 0.71 | 0.472 ± 0.016 (0.50, 0.46) | 0.415 ± 0.028 | 0.502 ± 0.034 | 0.499 ± 0.019 |
| C | 0.66 | 0.179 ± 0.010 (0.09, 0.36) | 0.107 ± 0.020 | 0.157 ± 0.032 | 0.211 ± 0.013 |
| E 6-23 | 0.00 | 0.174 ± 0.041 (0.41, 0.15) | same | same | — |
| E 6-24 | 0.00 | 0.094 ± 0.051 (0.27, 0.06) | same | same | — |

**Table 6.** Chamber C per row (time order).

| row | saturated | rail | b | b_x | b_y |
|---|---|---|---|---|---|
| short (00:58) | 0.00 | 0.35 | 0.075 ± 0.042 | −0.01 | 0.23 |
| longer (01:58) | 0.33 | 0.31 | 0.092 ± 0.020 | 0.00 | 0.27 |
| long (06:03) | 0.92 | 0.21 | 0.230 ± 0.012 | 0.14 | 0.42 |

On A, B and the golden mounting the saturated events give a lower b than the unsaturated ones. On D there is no clear
difference. For A 6-25 the effect is large, since 87 % of its events are saturated. The remaining difference between
6-25 and golden is the conversion from drift time to gap: b is defined per unit of the ln q_u90 field at β = 0.68, while
the field of a given gap shape grows as (1 + β). With R4's β(V), the unsaturated values become 0.33 × 2.43/1.68 = 0.48
(6-25) and 0.48 × 1.87/1.68 = 0.53 (golden). So on chamber A the amplitude is one number once saturation and β are
accounted for.

On C the low value comes from the two early rows. Their drift time is censored and their gas was changing, so their
q_u90 field does not describe the gap. In the last row, the Y view (0.42) agrees with the other chambers, and only the
X view (0.14) is low. Chamber C's construction record mentions "a column of waves in the mesh". A mesh structure along
one axis that the quadratic mesh term does not describe would act on one view only. This is a possible explanation, but
we have not tested it.

On E, b is low without saturation or censoring, and mostly on Y. E's pass-1 angle resolution is poor (1.4–2.4°). We have
no explanation.

**Consequence for the reference-free route.** Fit b on unsaturated events, and quote it at the β of the operating point.
The direct per-mounting maps do not use b and are not affected.

## Not answered here

Items 1, 3–7, 9 and 11–15, 19 and 20 are unchanged. Item 7 (flattening below 700 V) is not caused by the depth grid
(Table 1). Item 9 (drier gas on 6-25) is consistent with Table 2: the product v × q_u90 on 6-25 gives the golden gap, so
the low velocity on 6-25 is gas and field, not gap.
