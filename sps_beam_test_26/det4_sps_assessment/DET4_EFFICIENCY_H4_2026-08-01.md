# det4's efficiency map at H4, measured against the uRWELL

**2026-08-01.** The first beam-referenced efficiency map of det4 (mx17_E), from
runs 53 / 56 / 57 at three resist voltages. Uses the corrected strip map
(`DET4_URW_MAPPING_2026-08-01.md`), without which none of this is possible.
Figures: `det4_efficiency_summary.png` (the one to look at),
`det4_efficiency_map_run56.png`, `det4_efficiency_map_run57.png`.

**Three results.**

1. **The June cosmic band map is confirmed by a beam, and sharpened.** Between
   the bands det4 is at **0.1 %** — not low, *dead* — at every voltage tried.
   Inside them the beam resolves structure the cosmics could not: what
   `DET4_SPS_ASSESSMENT.md` §3b called a single 38 mm band at X 178–216 is
   really **two live strips with a dead notch at X 188–199** between them.
2. **In a good band det4 already performs like it did on the bench**: **80 %**
   within 5 mm across the full illuminated length of X 149–161, against 80.0 %
   for that band in June. It is not a damaged detector at H4.
3. **But the operating point is not there yet.** Averaged over the beam spot it
   is 26–30 %, because the spot is parked on X ≈ 200 — the band that turns out
   to be interrupted — and because efficiency is **still climbing at 670 V**,
   the highest point taken. On the bench this chamber ran at 495 V.

---

## 1. Method

For every uRWELL track with exactly one cluster in all four reference views, the
front→back track is extrapolated to det4 (z = 1120 mm) and mapped into
detector-local mm through the alignment fitted on that same run. det4's own
response is then classified with the June accounting, so the numbers mean the
same thing as `DET4_SPS_ASSESSMENT.md` §3:

- **fired at all** — any in-time hit;
- **reconstructed** — a cluster in both views;
- **within 5 mm** — reconstructed, and within 5 mm of the uRWELL prediction.
  This is the efficiency plotted;
- **discharge** — ≥ 6 clusters, or a cluster wider than 40 strips.

Two things have to be got right and are easy to miss:

- **Cluster in millimetres, not in channel index.** The inverted plugs put
  physically adjacent strips 127 FEU channels apart at three boundaries inside
  each view.
- **The drift gate is per run.** run 53 runs 600–1900 ns and runs 56/57 run
  600–3600, because the sampling configuration differs between them. `effmap.py`
  reads the gate off the hit-time spectrum rather than assuming either; hard-coding
  run 53's gate onto run 56 costs 5 points of efficiency outright.

The amplitude cut is irrelevant here — zero suppression already sits at ~38 ADC
in run 56, and moving the offline cut from 0 to 60 ADC changes the answer by
0.1 %.

## 2. The numbers

| resist [V] | 505–535 | 610–620 | 655–670 | June bench, 495 V |
|---|---|---|---|---|
| mount | flat | flat | rotated 25° | cosmic bench |
| clean reference tracks | 398,397 | 30,712 | 46,506 | 12.9 k rays |
| det4 fired at all | 27.6 % | 44.4 % | 49.8 % | 95.6 % |
| within 5 mm, whole beam spot | 10.9 % | 26.2 % | 30.4 % | 40.1 % (whole chamber) |
| within 5 mm, inside June bands | 15.6 % | 37.8 % | 44.5 % | 77.4 % |
| **within 5 mm, between bands** | **0.0 %** | **0.1 %** | **0.1 %** | 15.4 % |
| **within 5 mm, band X 149–161** | 21 % | **75 %** | **80 %** | **80.0 %** |
| best 10 mm window | — | 83 % @ 150–160 | 85 % @ 150–160 | 97 % @ 205–213 |
| discharge-flagged | 0.1 % | 0.7 % | 1.8 % | 8.2 % |

The alignment behind each column is independently fitted and each comes out a
proper rotation: run 53 +90.20° / det +1.005 / 0.48 mm median residual, run 56
+90.28° / +1.009 / 0.50 mm, run 57 +90.00° / +1.116 (the 25° yaw) / 2.67 mm.

**The between-band number is the headline of the map.** June measured 15.4 %
there and attributed it to muons crossing the boundary at an angle. With a beam
and a 0.5 mm reference it is 0.1 %. The dead area is completely dead, and no
voltage in the range scanned changes that — the 505 V and 670 V columns agree to
0.1 %.

## 3. What the beam sees that the cosmics could not

June's band edges are blurred by the muon's own excursion across the 30 mm drift
gap — `DET4_SPS_ASSESSMENT.md` §2d measures a 5–6 mm floor and could not tell
whether the true gain profile was sharper. It is. Resolving the profile in 1 mm
bins on a near-normal beam (run 56, flat mount):

| June said | the beam says |
|---|---|
| one band, X 146–164 | live X 149–161, 12 mm, **79 % mean / 91 % peak** — one clean band |
| one band, X 178–216 (38 mm, "the band to use") | **two** bands, X 182–187 and X 200–214, with a **dead notch at 188–199** |
| band X 228–234 | ~0 % at 610 V, 6.6 % at 670 V — marginal, do not use |

The notch is visible in the June charge profile in hindsight — the orange curve
in panel C dips by a factor 4 at X ≈ 195 — but it never crossed the live/dead
threshold, so the band-finder merged the two. This does not invalidate the June
work; it is what a 5 mm-resolution measurement looks like when you re-measure it
at 1 mm.

**Along the strips the bands are genuinely one-dimensional.** Inside X 149–161
the efficiency is flat in Y at 0.80 ± 0.10 over the illuminated 100 mm
(panel B). Measuring this needs the band cut — profiling efficiency against Y
over the whole spot mixes in the band structure through the beam's own X–Y
correlation and produces a spurious slope.

## 4. Two things to act on

**Move the spot to X ≈ 155, not X ≈ 200.** The beam currently sits on the
interrupted band. The 149–161 band is 12 mm of 80 %-efficient chamber over the
full strip length, and it is the best target on the instrumented part of the
detector — a ~45 mm move in the striped coordinate. That reverses the June
recommendation, which ranked X 177–215 first on the strength of its width.

**The gain curve has not plateaued at 670 V, and that needs explaining.** On the
bench this chamber ran at 495 V in Ar/iso 95/5 and was *discharge*-limited in
its good band (8.2 % discharges). At H4 it is at 670 V, still gaining, with
1.8 % discharges. That is ~175 V of shift in the same nominal gas. Candidates,
in order of how cheap they are to check:

1. **The gas det4 is actually flowed with at H4 is not Ar/iso 95/5.** The config
   default says it is; the comment at `run_config_beam.py:397` says to
   re-establish the point on whatever it actually runs on. Check the line.
2. **The chamber had been gassed for hours, not days.** It was installed the
   same morning; residual air/water suppresses gain and attaches. Re-take one
   point at a fixed voltage a day later — if it has moved, this is it.
3. Drift transparency: 700 V here vs 600 V on the bench, over the same 30 mm.
   Least likely to be worth 175 V, and the run 57 drift scan now running will
   settle it.

Until this is resolved, no efficiency, cluster-size or resolution number from
H4 should be compared with a June fleet number as if the chambers were at
equivalent operating points.

## 5. Caveats

- The beam only illuminates detector-local **X 140–235, Y 150–265**. Everything
  outside that is unmeasured here; the chamber's other bands (X 26–110, 292–304,
  356–398) are not on the instrumented connectors at all.
- The 505–535 V column is the whole of run 53, which was an HV scan — it is a
  mixture, not a single point, and is included only for the trend. The other two
  columns are cut to the logged dwell windows.
- Run 57 is the rotated mount, so its tracks cross ~14 mm of strips over the
  drift gap; its residual is 2.67 mm rather than 0.5 mm and its band edges are
  correspondingly softer. It is in the table because it is the highest voltage
  taken, not because it is the cleanest geometry.
- z = 1120 mm from `mapping_check/followups.py`; the residual minimum is shallow
  and the config placeholder 1155 works as well. Not a survey.
- "Discharge" here is a topology flag, not the June spark tagger.

## 6. Reproducing

```bash
# on banco -- writes the npz the rest consumes
python extract_pair.py run_56 meshscan_m60V

# locally
cd sps_beam_test_26/det4_sps_assessment/mapping_check
../../../.venv/bin/python effmap.py --out _run56
../../../.venv/bin/python effmap.py run_57_meshscan_m90V_v3 --out _run57 \
    --t0 16:49:19.957 \
    --windows "670:16:53:05-16:55:05,665:16:55:12-16:57:12,660:16:57:17-16:59:17,655:16:59:22-17:01:22" \
    --label "run 57, resist 655-670 V, rotated 25 deg"
../../../.venv/bin/python effmap_summary.py
```
