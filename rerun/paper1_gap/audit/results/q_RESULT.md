# Follow-up of the outstanding questions: items 2, 8, 9, 10, 12, 16, 17, 18 and 19 (audit tag q, 2026-11-24)

This note answers or narrows nine of the twenty outstanding questions of `AUDIT_2026-11-24.md` with the data on disk. Nothing in the
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
   chamber and not of a mounting. *(Corrected in the third round: the statement first made here, that C and E share A's
   1.8 mm shortfall, came from v × q_u90 normalised to chamber A. The independent v_ang × T table gives A 28.2, C 28.7,
   D 29.7, E 29.5, B 31.1 mm, in agreement with the geometric gaps.)*
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
1.8 mm shortfall from the design value cannot be placed in the chamber or in the scale of the estimator. (The other
chambers' values in this table are relative to chamber A; the third round gives them independently.)

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

## Second round (same day): items 9, 12, 19, and the angles of censored tracks

Scripts: `q19_curl_cens.py`, `q12_gain.py`. Outputs: `$WORK/audit/q16/q19_curl.json`, `q12_gain.json`.

**Censored tracks have biased angles.** A track whose charge reaches the last depth bin is reconstructed shallower than
the other tracks of the same row. Table 7 compares the two groups at 0.1 < |tan θ_ref| < 0.5, where the median |tan| of
the two groups is the same. The signed residual is (θ − θ_ref) times the sign of tan θ_ref, so a negative value means
the track is reconstructed too flat.

**Table 7.** Censored ("railed") against uncensored tracks, pass-1 angles (deg).

| row | view | railed / other tracks | s68 railed / other | median signed residual, railed / other |
|---|---|---|---|---|
| C short | X | 94 / 360 | 1.19 ± 0.15 / 1.54 ± 0.08 | −0.30 / +0.02 |
| C longer | X | 229 / 962 | 1.43 ± 0.09 / 1.31 ± 0.04 | −0.55 / +0.12 |
| C longer | Y | 190 / 689 | 1.58 ± 0.12 / 1.56 ± 0.06 | −0.47 / +0.29 |
| C long | X | 72 / 2418 | 1.28 ± 0.16 / 1.26 ± 0.02 | −0.74 / +0.14 |
| C long | Y | 119 / 1742 | 1.45 ± 0.16 / 1.43 ± 0.03 | −0.80 / +0.38 |
| E 6-24 | X | 170 / 1628 | 4.55 ± 0.88 / 2.23 ± 0.08 | −0.76 / +0.07 |
| E 6-24 | Y | 128 / 1309 | 1.57 ± 0.44 / 1.50 ± 0.06 | +0.04 / +0.09 |

The railed tracks are 0.3 to 1.2° flatter than the others. Since the waveform fit does not see the charge beyond the
last bin, it sees a shorter column and therefore a smaller angle. On the rows of chambers A, B and D (except D short)
fewer than 3 % of the tracks are railed, so the effect is negligible there.

**Item 19 (curl test on C and D).** We repeated the curl test of audit D (planes through 6 × 6 cells of τ = w0/v) with
and without the railed tracks. On C's last row the cross difference is +1.80 ± 0.67 (1e-3 per 100 mm, 2.7σ) with all
tracks, which reproduces audit D, and +0.26 ± 0.91 (0.3σ) without the railed 21 %. The curl excess of chamber C
therefore comes from the censored tracks. These tracks are biased flatter, and they are more common where the gap is
longer, so their bias has a position dependence that need not be a gradient field. On D's longer row no track is
railed, so censoring does not explain its 3.3σ. However, with only 7 cells on the X view, our bootstrap of the same
test gives +12 ± 13 instead of audit D's +6.4 ± 1.9. The error depends on which cells survive the 40-event cut, so we
do not consider the D excess established.

**Item 12 (fine drift-time map against the gain map).** We tested whether saturation causes the correlation. Saturated
tracks carry an offset in ln q_u90, and their fraction follows the gain. On 8 × 8 cells of the golden mounting
(fine part = cell map minus a quadratic, cross-run slope with a noise-free denominator) the slope d ln T / d ln q is:

| events | slope | events in the cells |
|---|---|---|
| all | +0.067 ± 0.013 | 18,677 |
| unsaturated only | +0.164 ± 0.023 | 13,377 |

The hypothesis is contradicted: without the saturated tracks the slope is larger, not smaller, and it is 5 times the
event-level walk slope (about 0.03). The fine gain map is almost the same as the map of the saturated fraction
(cross-run r = 0.99), so the saturated tracks, whose charge is capped, dilute the correlation instead of making it. A
real fine-scale correlation between the drift time and the gain therefore exists at 7σ. Audit B found no correlation
between the full maps, so the link acts on the fine scale only. One candidate is the mesh transparency, which rises
where the drift field is lower, i.e. where the gap is longer. However, that would also act on the smooth gap map,
where no correlation is seen. We have not identified the mechanism. Until it is understood, the fine gap map is
uncertain at the level of the gain correction (audit B: about 10 % of the fine structure).

**Item 9 (the 6-25 gas).** The water ledger (`mx_june_wft/drift_consistency/water_ledger.csv`) gives at least 0.65 %
water on 6-25 and 0.90 % in the golden mounting, from the logged drift voltage and the Magboltz tables. This agrees
with the 0.68 % against 0.97 % of audit C. Table 2 adds that the gap is the same in the two mountings, so the lower
velocity on 6-25 is gas and field, and not gap. Item 9 is confirmed.

## Third round (same day, after Dylan's notes): gaps of all chambers, censored tracks, items 1, 3, 4, 5, 7, 12, 15

Scripts: `gaps_all/make_gaps.py` (its `report.html` is the internal gap table), `q12_gain.py` (second test),
`q12_views.py`, `q15_windows.py`, `q1_surfaces.py`, `q3_vx.py`, `q4_yamp.py`, `q5_curv.py`. Outputs in
`$WORK/audit/q16/`.

**Drift windows (Dylan, 11-24).** The drift windows could have been exchanged between chambers between runs, and there is
no record of it. A change of a gap map between two mountings of one chamber may therefore be a different window.

**The gap of every chamber** (`gaps_all/report.html`). We measured g = v_ang × T on every row that has a head-on stack,
the method of Paper I's figure two_velocities (c). Rows taken in the first hours after a chamber was installed are left
out, because attachment shortens the visible column there (C before 06:00 on 6-26, D's first row, E's first row on 6-23).

| chamber | rows | g = v_ang × T (mm) | χ² | geometric gap (mm) |
|---|---|---|---|---|
| A | 4 (drift scan) | 28.24 ± 0.41 | 4.8 | 28.08 ± 0.19 |
| B | 3 | 31.09 ± 0.29 | 1.3 | 31.12 ± 0.42 |
| C | 8 | 28.71 ± 0.37 | 7.3 | 28.0 ± 1.0 (adopted) |
| D | 7 | 29.70 ± 0.32 | 6.9 | 29.44 ± 1.12 |
| E | 2 | 29.54 ± 0.40 | 2.0 | 30.48 ± 0.44 |

The two independent methods agree for every chamber within 1 mm (E: 2σ). The chambers are therefore not all near 28 mm.
They range from 28.2 (A) to 31.1 mm (B), around the design value of 30 mm. In Table 2 above, v × q_u90 was normalised
to chamber A, so it only measured each chamber against A. Since the windows may have moved between chambers, the gap
belongs to a chamber and a window together, and we cannot say which part comes from which.

**Censored tracks: correction or exclusion.** The charge past the last depth bin was recorded. Censored tracks start
about 300 ns into the 1920 ns readout window, and the depth grid ends 1440 ns after the start, so about 180 ns (3 bins)
of recorded waveform are not used. A refit with K = 27 would use it, but that is a pass-level job. At table level, the
pass-1 angle of the censored tracks against the reference has a slope below 1 (C: 0.89 to 0.96). Rescaling their angle by a
cross-validated slope per row removes their median bias (C long Y: −0.80° → −0.10°, C longer X: −0.55° → +0.09°). It does not
change the row's s68 (≤ 0.02°). On E 6-24 X and D short, the censored tracks are weak tracks (charge about 3 times lower),
not long columns. There the slope is unstable (0.80 and 0.50) and the correction overshoots. Leaving them out improves E
6-24 X from 2.14° to 2.03°. Recommendation: rescale per row on C, and exclude on E and D.

**Item 12, second and third tests.** Three results:
1. The drift-time estimator does not respond to the gain. In the mesh-voltage scan of the golden mounting the gain rises
   by a factor of 6 from 445 to 495 V, while the median q_u90 of unsaturated tracks stays at 612–620 ns. Even at high gain
   the response is at most about 0.05 per unit ln q. Therefore the fine gap map does not need a gain correction.
2. The fine charge map follows the fine drift-time map with slope d ln q / d ln T = 1.30 ± 0.19 (unsaturated, cross-run,
   r = 0.46). Two known effects predict part of it. The charge of a track is proportional to its path in the gas, which
   is the gap: this gives 1 / (1 + β) ≈ 0.6. The drift field is lower where the gap is longer, and the drift scan gives
   d ln q / d ln E = −0.12: this adds about 0.07. Together they explain about half (0.60–0.67).
3. The link is the same in the two views: each view's charge map follows the other view's drift-time map as closely as its
   own (slopes 1.19–1.36, ± 0.2), and the fine charge maps of the two views agree (slope 0.95 ± 0.02). So it is a property
   of the position in the chamber, not of the electronics channels. The other half is a gain that is higher where the gap
   is longer. A change of the amplification gap of about 0.3 µm per mm of drift gap would be enough, but we cannot test
   this.

**Item 15 and Dylan's window question.** If a window moved between chambers, its cathode relief should reappear in the
other chamber's gap map, possibly turned by 90° or mirrored. We first removed the dependence of q_u90 on the track angle,
since the telescope acceptance makes the angle mix vary across the chamber in the same way for every chamber. Then we
compared every pair of mountings with one map turned in steps of 5°, with and without a mirror. The null distribution is
the correlation at angles that are not multiples of 90°, which has the same smoothness. The known same-window pair, E 6-23
and E 6-24, peaks at 0° with r = 0.73, but only z = 2.9 above its own null. Several pairs of different chambers reach
similar levels: A golden with C at 0° (z 2.7), A golden with B at 180° (2.5), A golden with E 6-23 at 0° (2.7) and with E
6-24 at 90° mirrored (2.9). The last two contradict each other, because E's two days match each other at 0°. The fine
maps therefore have too few independent features to identify a window. **A swap can be neither confirmed nor excluded
from the gap maps.** The similarity between chambers does not come from the angle mix, because it survives the
correction.

**Item 1 (the fine tilt not explained by the gap).** In the two-surface model the tilt potential is P = h + a g. So the
fine mesh height can be predicted from the tilt map and the drift time alone, h = P − a g. We compared it with the fine
mesh height of the position route (6 × 6 cells, golden mounting, quadratic removed, P from one run and h_pos from the
other). With a = 0.2–0.4 the correlation is r = 0.65–0.66 and the slope of h_pos on the prediction is 0.88–0.91 (the model
expects 1). The position-route map is noisy (run 1 against run 2: r = 0.31), so r = 0.65 is at its noise ceiling
(about √0.31 = 0.56). The fine gap alone does not correlate with h_pos (r = 0.1). Therefore the fine tilt that the gap
does not explain is a fine relief of the mesh, about 0.15 mm rms, and the position route sees it independently. The fine
gap structure is on the cathode (audit B). Both surfaces carry fine structure.

**Item 3 (X-view velocity map).** Saturation does not explain it: the angle scale of saturated tracks is 0.997 ± 0.003 of
that of the unsaturated ones on X (1.001 ± 0.002 on Y). The X map is reproducible (run to run r = 0.93, rms 3.2 %), larger
than Y (2.5 %), and the two correlate at r = 0.71. About 2 % of X-only structure remains. The X strips lie one layer
deeper and the X charge sharing is asymmetric on chamber A; we have no waveform-free way to test either.

**Item 4 (geometric against drift-time amplitude along y).** On the audit's 4 × 4 cells, with the same cells for every
estimate, the curvature of the gap along y is −1.02 ± 0.66 (run 1) and −1.26 ± 0.38 (run 2) mm per (100 mm)² for the
geometric X method, −1.30 ± 0.21 and −1.35 ± 0.13 for the drift time, and −0.9 to −1.3 for the β-free product v × T.
The ratio is 0.8–0.9 ± 0.3–0.5. The factor 1.4 is not reproduced, and the geometric errors are too large to support it.

**Item 5 (curvature along x).** The curvature of the gap along x in the golden mounting (event-level quadratic with |tan|
terms) is +0.43 ± 0.03–0.05 mm per (100 mm)² over the whole chamber. Without x < 80 mm it is +0.08 to +0.13 (± 0.04–0.09),
and over 80 < x < 320 mm it is +0.05 to +0.09. On the half x < 200 mm it is +1.7. So the gap is flat along x between 80
and 320 mm, and it rises toward the board edge at x < 80 mm. This is the drift-line slice that "bends the other way". A
single parabola along x does not describe it. This also explains item 6: a gradient or curvature depends on the region.

**Item 7 (flattening below 700 V).** Three causes are now excluded. The depth grid is not the cause (Table 1). The readout
window is not the cause (R4). Attachment is not the cause either, because the charge per unit path at 500 V (ln q 9.11)
equals that at 700–1100 V (9.06–9.11). At 300 V the charge does drop, by 0.6 in ln q. The drift-time estimator does not
respond to the gain (item 12). The flattening at 500 V remains open.

**Items 6, 11, 13, 14.** Item 6 is explained by item 5. Item 11 agrees with the water ledger (Run 2 is the drier run,
0.90 % against 0.90–1.05 % in the Saturday sub-runs). Item 13 is a property of the maps (they have fine structure that a
gradient fit does not describe), and no estimator in this audit depends on it. Item 14 is the per-sample charge centroid
of the drift-line test. A sample mixes charge from several depths, because the electronics response spans several
samples, so the centroid slope is compressed at large |tan|. This is the aggregate effect of `RECONSTRUCTION_BASIS.md`.
The drift-line test removes it by balancing the sign of tan. This explanation is an argument, not a new measurement.

## Not answered here

Only item 20 (n_TOF, the re-audit session's) is not treated here; item 7 and the second half of item 12 remain open. Item 7 (flattening below 700 V) is not caused by the depth grid
(Table 1). Items 9, 12 and 19 are in the second round above.
