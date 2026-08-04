# Is det4 worth taking to the SPS?

**2026-07-31.** Analysis of the June cosmic data on the waveform-first basis
(`../../RECONSTRUCTION_BASIS.md`), asking whether det4 — the chamber left out of
the n_TOF experiment — is worth the effort of a 200 GeV muon characterisation.

**Short answer: det4 works, and it is not "low gain". 62 % of its area does not
amplify at all, in fixed stripes; the remaining 38 % is a normal detector
(77 % efficient, 0.59 mm, 2.1°). No voltage fixes the dead area, and it is not
the readout — the pedestals are flat across the dead bands.**

**As a beam target it comes down to one strip.** The band at detector-local
X 177–215 mm (reference Y −10 to +30 on the sliding map) is 38 × 360 mm and runs
at **82 % within 5 mm, 90 % excluding discharges**, with det3-like cluster sizes,
0.62 mm and ~2°, uniform along its whole length, and it tolerates the inclination
a micro-TPC scan needs. That is a usable target for an 8 cm beam (69 % averaged
over the spot, or 82 % collimated). What it cannot deliver is any statement about
the chamber as a whole — every full-area number would describe this bulk's defect,
not the MX17 design. §3b.**

---

## 1. What det4 does, whole-chamber

Run `g_det4` = `mx17_det4_day_6-24-26/long_run`, Ar/iso 95/5, resist 495 V,
drift 600 V, 12.9 k clean M3 rays in the true active area. Waveform-first
reconstruction, same accounting as `mx_june_wft/FLEET_2026-07-29.md`:

| | det4 | det3 | det2 | det6 | det7 |
|---|---|---|---|---|---|
| within 5 mm | **40.1 %** | 93.1 % | 91.1 % | 75.3 % | 53.6 % |
| detector fired at all | 95.6 % | 100 % | 100 % | 100 % | 100 % |
| fired but no reconstruction | **39.5 %** | 0.3 % | 0.6 % | 3.4 % | 8.6 % |
| discharges | 8.2 % | 2.7 % | 3.4 % | 17.9 % | 27.7 % |
| median summed charge per ray, X / Y plane [ADC] | **472 / 1046** | 8995 / 9179 | 12668 / 12068 | 17688 / 21226 | 16777 / 24614 |
| rays with ≥3 strips on **both** planes | **56.6 %** | 99.5 % | 98.9 % | 96.4 % | 89.0 % |

det4 is the only chamber in the fleet whose losses are *charge*, not discharges:
det6 and det7 are spark-limited, det4 collects 10–50× less charge than anything
else and simply cannot build a cluster. `01_uniformity.py`, `03_charge_vs_position.py`.

## 2. The loss is not uniform — it is stripes

Cell-by-cell (25 mm squares) det4's efficiency runs 0 → 93 %, with a spread
3.4× the binomial expectation. Nothing else in the fleet does this:

| | det4 | det3 | det2 | det6 | det7 |
|---|---|---|---|---|---|
| cell-efficiency rms | **0.234** | 0.045 | 0.061 | 0.183 | 0.268 |
| binomial expectation | 0.069 | 0.043 | 0.057 | 0.067 | 0.074 |
| excess dispersion | **0.224** | 0.012 | 0.020 | 0.170 | 0.258 |

The structure is one-dimensional: efficiency swings 0 → 98 % with detector-local
**X**, and is flat in Y (`02_bands.py`, `bands_g_det4.png`). Collected charge
follows it over two orders of magnitude — a 165× contrast between the peaks and
the troughs (`stripes_g_det4.png`).

**It is the amplification region, not the readout.** Local X is measured by the
X plane (FEU 6), so a dead group of X-plane channels would reproduce the
pattern. It does not: the *Y* plane's raw hit count — no reconstruction
involved, straight from `combined_hits` — falls from 6.7 to 2.9 strips per ray
in exactly the same X bands, correlation 0.89 between the two planes' profiles
(det3 0.72, det2 0.45, and with 5–10× smaller swings). Both planes lose charge
together, which only happens if there is no avalanche to share.

Pattern geometry, from a 2 mm-binned charge profile with the live/dead threshold
at the midpoint in log charge (`04_stripe_metrics.py`):

- 11–12 live bands, median width **12 mm** (2–38 mm), median spacing **35 mm**
- live fraction of the chamber width: **38 %**
- spacings are irregular (6–60 mm), so this is a defect distribution, not a
  periodic mechanical structure

The pattern is the chamber's own: correlating det4's charge-vs-X profile against
the other four chambers' gives 0.21 / 0.39 / −0.08 / 0.29, and det4's profile has
p90/p10 = 39 against 1.6–5.6 for the rest (`06_fleet_profiles.py`). It is also
stable in time — the same stripes at the same X in the 6-23 run 24 h earlier
(`07_crossrun_stability.py`, `crossrun_occ.png`).

## 2b. Could it just be the X-plane connectors? Four tests, all negative

The stripes live in the coordinate the X plane measures, so dead strips or a bad
Panasonic connector is the first thing to rule out. `08_connector_test.py`,
`09_periodicity.py`:

| test | expectation if it were readout | measured |
|---|---|---|
| **T1** dead channels | dead strips give *no* hits | **0 dead strips** on either plane in 95,803 events. Strips between the bands fire at 0.0012 hits/event (live bands 0.0087) with median amplitude 114 ADC (live 356). They are quiet, not dead. |
| **T2** band edges vs the 49.92 mm connector pitch | edges land on boundaries | mean edge-to-boundary distance **11.8 mm**, random expectation 12.5 mm; 1 of 23 edges within 3 mm; **5 of 12 bands straddle a boundary** |
| **T3** periodicity at the connector pitch | strong 49.92 mm modulation | folded modulation 0.51 dex vs a **median of 0.57 dex for random trial periods** — 70 % of random periods beat it. The connector pitch sits in a *dip* of the periodogram (strongest period is 115 mm, i.e. the broad envelope) |
| **T4** the other plane | a deaf X plane leaves Y untouched | the Y plane's cluster size falls **6.6 → 2.9 strips** and its charge **4823 → 496 ADC** in the same X bands, while the Y plane is perfectly smooth in its *own* coordinate (its strips run parallel to the stripes, so they average over them) |

T4 is the one that closes it: whatever removes the charge does so before either
readout sees it.

**There is a real connector-level anomaly, and it is not this.** FEU 6 connectors
7–8 (local X 300–399 mm) carry about half the noise of the others in the stored
pedestal (4.8 vs 8.2 ADC CNS) and pass ~2.5× more small hits at the same
significance. §2c identifies the cause — they were *unplugged when that pedestal
was taken* — and shows why it cannot be the band structure: it is a uniform
scale factor over a 50 mm block, not a 12 mm band pattern, and the two
connectors with the lowest effective threshold contain both the chamber's single
best band (X 355–398, 91 %) and one of its deadest stretches (X 300–350, live
fraction 0.09). If the pattern were a threshold effect these two would look
alike; they are opposites.

Figures: `efficiency_map_g_det4.png` (the map, with connector boundaries drawn),
`connector_test_g_det4.png` (T1/T2/T4), `periodicity_g_det4.png` (T3).

### Why this was missed for a month: the production kernel hides it

The standard sliding-kernel map (`12_efficiency_map_sliding.py`, default
**r = 25 mm**) smooths over a diameter of 50 mm — four times the median stripe
width. `efficiency_map_sliding_k25_g_det4.png` is the production map: a mild
horizontal banding on a uniform mid-efficiency field, which reads exactly like
"gain-limited everywhere". The same script at **r = 8 mm** on the same rays
(`efficiency_map_sliding_k8_g_det4.png`) resolves the stripes at full 0 → 1
contrast:

```bash
cd mx_june_cosmic_qa
../.venv/bin/python 12_efficiency_map_sliding.py g_det4 --kernel=8 --grid=250 --min=8
```

(That writes into the detector's `efficiency/` dir and overwrites the production
`efficiency_map_sliding.{png,json}` — back them up first, as was done here; the
r = 25 mm versions are the ones the PDF builder consumes.)

In the reference frame the stripes are **horizontal**, because this run's
alignment is θ ≈ 90°: detector-local X maps to reference Y. The `has_any` panel
stays uniformly ~1 across the whole chamber in both versions — the detector
fires everywhere, it just cannot build a cluster between the stripes.

The kernel series `efficiency_map_sliding_k{25,8,3,2,1}_g_det4.png` shows where
the information runs out. At 14,787 rays over the active area the ray density is
0.10 /mm², so a kernel of radius r holds ~0.32 r² rays: 20 at 8 mm, 2.8 at 3 mm,
0.3 at 1 mm. Below ~3 mm the map stops being an efficiency measurement and
becomes a scatter plot of individual rays — at r = 1 mm only 22,464 of 90,000
grid points hold a single ray. The bands stay visible anyway, which is itself
the point: they are visible in a *one-ray-per-pixel* map because the underlying
efficiency really is near 0 or near 1.

**There is a physics floor below ~5 mm that no kernel can beat.** A cosmic muon
crossing the 30 mm drift gap at a typical tan θ ≈ 0.2 moves ~6 mm in X between
entering and reaching the mesh, so each "ray position" smears the gain profile
over that distance. Measured on 1 mm bins (33 rays/bin), the 0 → 100 %
transitions take 9–24 mm (median 12 mm) — comparable to the band width itself,
so the bands are rounded peaks rather than sharp-edged holes. Part of that
softness is the track excursion; the true gain profile may be sharper. Either
way it is consistent with a continuously varying amplification gap — a mesh
height / lamination problem — rather than discrete dead patches.

**Worth checking on the rest of the fleet.** If a 25 mm kernel can hide a
structure this strong on det4, it can hide a weaker one elsewhere; det6 and
det7 both carry excess dispersion (0.17 / 0.26) that has never been looked at
below the 25 mm scale.

## 2c. The pedestals: the readout is fully connected, and we have a control

08 showed no *dead* channels, but a badly-seated connector is not a dead channel —
it is a strip that has lost contact with the preamp input. That has an
unmistakable pedestal signature and needs no beam: an unloaded input loses the
strip's capacitance, so its noise collapses. A *resistive* strip that has lost
its bias leaves the readout strip exactly where it was — same capacitance, same
pedestal. `10_pedestals.py`, on the pedestal run that this data was taken with
(`MX17_pedestals_pedthr_260623_18H43`, 400 frames, CNS per 64-channel block):

| FEU 6 (X plane) | live bands | between bands | ratio |
|---|---|---|---|
| CNS pedestal RMS | 7.97 ADC | 7.74 ADC | **0.97** |
| raw pedestal RMS | 166 ADC | 187 ADC | 1.13 |
| pedestal mean | 370 ADC | 384 ADC | 1.04 |

Flat. Zero channels below half the median noise on either plane. Every readout
strip in a dead band carries its full load.

**And the same file contains a positive control for what "disconnected" looks
like.** FEU 6 connectors 7–8 read raw RMS **5.1 / 5.3 ADC against 165–200** on
connectors 1–6 — a 35× collapse, with CNS RMS 4.8 vs 8.2. They were unplugged
when this pedestal was taken, which is exactly why they recorded **zero hits**
in the whole 6-23 run. On 6-22 the same connectors read 182 / 183 raw (normal),
and by the 6-24 run they were back and carrying the highest occupancy in the
chamber. So we know precisely what a disconnect looks like in these pedestals,
and nothing in connectors 1–6 resembles it.

**Consequence to remember:** the 6-24 run inherited this 6-23 pedestal
(`pedestals: latest`), so FEU 6 connectors 7–8 were processed against a noise
estimate ~1.7× too small — their significance is inflated, which is why they
show the chamber's highest raw occupancy at its lowest median amplitude
(34–42 ADC). This is why every band conclusion here is cross-checked with a
fixed **60 ADC amplitude cut**, which uses no pedestal at all: the bands survive
with a contrast of **55×** and the same positions (`11_strip_break_test.py`,
bottom panel of `strip_break_g_det4.png`).

## 2d. Broken resistive strips, or a varying gap? — the edges are ~5 mm wide

Two more tests, motivated by the arc visible at the bottom of the map.

**Do the columns die partway along Y?** A resistive strip that breaks mid-length
leaves the segment beyond the break unfed, so it should be live up to some Y and
dead beyond — at a different Y for each strip. Column by column (6 × 30 mm cells,
`strip_break_g_det4.png`):

- over the bulk of the chamber (X ≈ 90–330 mm) the columns are **uniform in Y**:
  median |low-Y − high-Y| step 0.19, and only 9 of 38 live columns exceed 0.30;
- the exceptions cluster in the **low-X corner, X ≲ 80 mm**, with steps of
  0.4–0.6, the dead side on **low Y in 8 of 9 cases**, and the transition around
  Y ≈ 228 mm. That is the arc in the map, and it *is* what a break partway along
  a strip looks like.

So the Y-structured corner is a genuinely different feature from the rest. Note
that "uniform in Y" does not rule out resistive strips elsewhere — a strip that
lost its feed *at the bus* is dead over its whole length, indistinguishable in Y
from a gap defect.

**How sharp is a band edge?** This does separate them. A broken resistive strip
gives a boundary one strip pitch wide (0.78 mm), blurred by charge sharing to
~1–2 strips. A varying amplification gap gives a ramp. The measurement is limited
by the muon's own excursion — a track crossing the 30 mm gap at tan θ moves
30·tan θ in X — so `12_edge_sharpness.py` measures the 10–90 % edge width in bins
of reference inclination:

| \|tan θ\| | X excursion | rays | 10–90 % edge width |
|---|---|---|---|
| 0.00–0.05 | 0.8 mm | 2,906 | **6 mm** (IQR 5–7) |
| 0.05–0.12 | 2.6 mm | 3,913 | 11 mm (8.5–15) |
| 0.12–0.25 | 5.6 mm | 4,194 | 16 mm |

Re-measured on the near-vertical sample at 1 mm binning: median **5 mm**
(individual edges 2–9 mm), unchanged when tightening to |tan θ| < 0.03. So it is
neither binning- nor blur-limited: **the chamber's own gain edge is ~5 mm, about
six strip pitches.** That is 3–6× wider than a discrete strip boundary could be,
and points at a continuously varying amplification gap — a mesh-height /
lamination problem — over most of the chamber, with the low-X corner as a
separate, more break-like feature.

Verdict on the connector question: **ruled out.** Verdict on broken resistive
strips: not ruled out in the low-X corner, but disfavoured over the bulk, where
the gain ramps over ~5 mm rather than stepping at a strip.

## 3. Inside the live stripes det4 is a working detector

Scored the way the fleet is scored, with the bands defined on even eventIds and
scored on odd ones so the selection cannot inflate the answer:

| | live stripes (38 % of area) | between stripes | whole chamber |
|---|---|---|---|
| within 5 mm | **77.4 %** | 15.4 % | 39.9 % |
| core σ | **0.59 mm** | 1.38 mm | 0.69 mm |
| σ_θ X / Y | **2.12° / 1.92°** | 7.8° / 6.0° | 3.05° / 2.51° |
| angle bias X / Y | −0.24° / −0.25° | +0.02° / −0.58° | −0.17° / −0.31° |
| mean strips X / Y | 6.2 / 6.6 | 3.9 / 2.9 | 4.8 / 4.4 |
| median charge X / Y [ADC] | 4710 / 4823 | 82 / 496 | 472 / 1041 |
| detector fired at all | 99.5 % | 93.0 % | 95.6 % |
| discharges | 7.2 % | 9.1 % | 8.3 % |

In-sample the live-stripe efficiency is 78.5 %, out-of-sample 77.4 % — the
band-finding costs about one point, so the number is real.

For a beam you would care about contiguous windows, not the total:

| window | best position (local X) | mean efficiency |
|---|---|---|
| 10 mm | 205–213 mm | 0.97 |
| 20 mm | 373–391 mm | 0.93 |
| 30 mm | 365–393 mm | **0.91** |
| 40 mm | 177–215 mm | 0.88 |
| 50 mm | 149–197 mm | 0.78 |

So a collimated beam parked at X ≈ 365–393 mm sees a ~91 % efficient
micro-TPC across the full 36 cm of Y. 17.7 % of the chamber width is ≥90 %
efficient, 26.8 % is ≥80 %.

## 3b. The X 177–215 mm band interrogated as a beam target

On the sliding map this is the strip at **reference Y ≈ −10 to +30 mm** (local X =
reference Y + 186 mm for this run's alignment). It is the widest live band, 38 mm
across and the full 360 mm of Y. `13_beam_window.py`:

| | band X 177–215 | whole chamber | det3 for scale |
|---|---|---|---|
| within 5 mm | **82.2 %** (n = 1655) | 40.1 % | 93.1 % |
| …excluding discharge events | **89.9 %** | 43.7 % | 95.7 % |
| detector did not fire | **0.1 %** | 4.5 % | 0 % |
| discharges | 8.6 % | 8.2 % | 2.7 % |
| core σ | 0.62 mm | 0.69 mm | 0.47 mm |
| σ_θ X / Y | 2.05° / 1.98° | 3.05° / 2.51° | 1.20° / 1.14° |
| mean strips X / Y | **7.1 / 7.0** | 4.8 / 4.4 | 6.7 / 7.2 |
| median charge X / Y | 6422 / 6443 | 472 / 1046 | 8995 / 9179 |

Inside this band det4 is not a marginal detector. Cluster sizes are *det3's*, the
charge is 70 % of det3's, it fires on 99.9 % of crossings, and **the leading loss
is no longer gain — it is discharges** (8.6 %). Take those out and 90 % of
crossings reconstruct within 5 mm. In this band det4 behaves like det6/det7
(spark-limited) rather than like a low-gain chamber, which also means the
operating point is worth re-optimising here: §4 shows 495 V sits ~15 % below the
gain peak at 505–510 V, but pushing up will cost sparks.

**Uniform along its length**, with one exception: efficiency is 0.76–0.90 across
the whole 360 mm of Y except a single weak patch at Y ≈ 290–320 (0.53 ± 0.04).

**What an 8 cm beam sees.** Sliding a disc-weighted 80 mm spot across the
efficiency profile, the best placement is centred at local **X ≈ 183 mm** and
averages **69 %** (chamber average 39 %) — the spot is necessarily wider than the
band and picks up the dead notch at X ≈ 166–177 and the edge beyond 215. Two
practical options: collimate to ~40 mm on the band and get 82 %, or take the full
spot and select on the telescope offline. The natural target for an 8 cm spot is
**X ≈ 146–215**, which is 69 mm of mostly-live chamber (two bands) with a ~10 mm
notch between them.

**You do not need to angle the chamber along the stripes.** The worry is real —
a track inclined across the stripes moves 30·tan θ in X over the drift gap — but
the band is 38 mm wide, so even tan θ = 0.4 (12 mm excursion) stays inside. Both
inclinations were measured on the existing cosmics, in-band:

| |tan θ| | across the stripes | along the stripes |
|---|---|---|
| 0.00–0.05 | 0.78 within 5 mm | 0.83 |
| 0.05–0.12 | 0.82 (σ_θX **1.31°**) | 0.80 (σ_θY **1.26°**) |
| 0.12–0.25 | 0.83 (σ_θX 1.84°) | 0.79 (σ_θY 2.02°) |
| 0.25–0.60 | **0.84** (σ_θX 3.14°) | (too few rays) |

Efficiency is flat to slightly *rising* with inclination across the stripes —
inclined tracks fire more strips, and that outweighs the excursion. So either tilt
works; the choice only decides which plane gets the micro-TPC lever arm. Note that
in each row the plane with no inclination has a meaningless σ_θ (no slope
information) — that is why one column of each pair is quoted.

**The other two candidate bands**, for completeness:

| band | width | within 5 mm | core σ | σ_θ X / Y | strips X / Y | charge X / Y |
|---|---|---|---|---|---|---|
| X 146–165 | 19 mm | 80.0 % | 0.56 mm | 1.91° / 1.77° | 7.1 / 7.3 | 9907 / 9525 |
| X 177–215 | 38 mm | 82.2 % | 0.62 mm | 2.05° / 1.98° | 7.1 / 7.0 | 6422 / 6443 |
| X 355–398 | 43 mm | 79.7 % | 0.60 mm | **4.29°** / 1.85° | **5.0** / 6.7 | 4261 / 4818 |

X 146–165 has the highest charge in the chamber (det3-level) and the best angles,
but is only 19 mm wide. **X 355–398 looks the widest and is the one to avoid**:
its X-plane angular resolution is 2× worse on 5.0 strips instead of 7.1, because
this is exactly the FEU 6 connector 7–8 region running on the stale pedestal
(§2c). Its efficiency is partly built on near-noise hits. X 177–215 is the band
to use.

## 3c. The band on the board — where to point the beam

`14_board_map.py` puts the 3 mm-kernel efficiency map onto a scale drawing of the
readout PCB (`board_map_g_det4.png`). Board geometry from the DFS3498A Gerbers in
`~/x17/mx17_gerbers/Gerber pcb readout`: outline −220…+250 mm in both axes
(470 × 470), the 399.36 mm metallised square centred on (0,0), four mezzanine
footprints (two per edge, centred at ±100 mm along their edge, four 64-channel
connectors each) and the M6 frame holes at ±214 mm every 55 mm.

**Coordinate conversion: `gerber = detector-local − 199.68 mm`** on both axes.

Connector blocks are drawn from the strip map, which is the authority: connector
k covers channels 64(k−1)…64k−1 = local 49.92(k−1) … +49.14 mm, monotonic on both
planes. The Gerber corroborates the grouping — the two mezzanines per edge sit at
local ~99 and ~299 mm, the centres of channels 0–255 and 256–511.

**Orientation caveat.** The figure is drawn as requested, X bank along the bottom
and Y bank on the right; the Gerber has both banks on its +X and +Y edges, so the
X bank is mirrored in Y here. Which physical edge carries which plane could not
be settled from the Gerbers — both `L3-TrackY` and `L4-TrackX` fan out to both
edges — so treat the *edge assignment* as a drawing convention and the *connector
numbering*, which is what you need to find a strip, as measured.

**The live band in hardware terms:** local X 177–215 mm = **strips 227–276**
= from **X-connector 4, channel 35** to **X-connector 5, channel 20**. It
straddles the X4/X5 boundary, which is worth knowing when probing it on the
bench.

**Beam placement.** The spot drawn bold on both figures is the one to use — it
is centred on the **Y4/Y5 interface** (local Y = 199.29 mm; Y4 ends at 198.90,
Y5 starts at 199.68) and on the **middle of the live band** in X, which lands it
on **X4+X5 and Y4+Y5 only — 256 channels, four connectors**:

| centre (local) | centre (gerber) | connectors lit | mean efficiency |
|---|---|---|---|
| **(196, 199) mm** ← proposed | (−4, −0) mm | X4, X5, Y4, Y5 | **0.64** (n = 680) |
| (184, 313) mm — free optimum | (−16, +113) mm | X3–X5, Y6–Y8 | 0.79 (n = 568) |
| (184, 199) mm | (−16, −0) mm | X3–X5, Y4, Y5 | 0.69 |
| chamber average | | | 0.39 |

The proposed centre is, to within 4 mm, the **geometric centre of the board** —
easy to set up and to check. Edge margins of the 80 mm spot inside its four
connectors: 6.2 mm at X4's low edge, 12.8 mm at X5's high edge, 9.5 mm each side
in Y. (Within the two-connector window the efficiency peaks at X = 190 mm with
0.66, but that leaves only 0.2 mm of margin at X4's edge — not worth it.)

The free optimum is kept on both figures as a faint dotted circle. Restricting
to four connectors costs 0.79 → 0.64: about 0.09 of that is moving Y from 313 to
the chamber centre, the rest is holding X inside the X4/X5 pair. All of these
remain far above the 0.39 a blind placement would give — **the X coordinate is
what matters**, and any of these placements has it right.

### Two views, and which one to mount

`14_board_map.py` writes both:

- **`board_map_g_det4.png`** — bench view. X bank at the bottom, bands vertical.
- **`board_map_g_det4_rot90ccw.png`** — the same drawing rotated **90°
  counter-clockwise**: X bank on the **right**, Y bank along the **top**, and the
  gain bands **horizontal**.

(`--views none,ccw,cw` also writes the clockwise version, which puts the X bank
left and the Y bank at the bottom; either rotation makes the bands horizontal,
they differ only in which edge the cables leave from.)

The rotated one is the mounting orientation for the beam. With the bands
horizontal, a **left–right yaw of the board runs the track along a band**, so
inclined tracks never leave the live stripe, and the inclination is seen by the
**Y plane (FEU 8)** — that is the plane that gets the micro-TPC lever arm in
this configuration. In-band on cosmics that reads σ_θY = 1.26° at |tan θ| 0.05–0.12
and 2.02° at 0.12–0.25 (§3b, "along the stripes").

Pitching up–down instead crosses the bands and gives the lever arm to the
**X plane (FEU 6)**; §3b measures that as fine too — efficiency actually rises
slightly with inclination, to 0.84 at |tan θ| 0.25–0.60, because the band is
38 mm and the excursion at tan θ = 0.4 is only 12 mm. So both micro-TPC axes are
available: yaw for the Y plane, pitch for the X plane, or roll the chamber 90°
about the beam axis between runs to swap which plane is being characterised.

## 4. There is no voltage that fixes it

The 6-23 overnight run stepped det4's resist voltage **465 → 525 V** in 5 V
steps (drift 600 V, 20 min each). Its efficiency was never published because
that run's M3 reference is degraded — but none of this needs a reference
(`05_hv_headroom.py`):

| resist HV | 465 | 480 | 495 | 510 | 525 |
|---|---|---|---|---|---|
| median hit amplitude, X [ADC] | 170 | 240 | 276 | **319** | 281 |
| live fraction of X strips | 0.237 | 0.237 | 0.237 | 0.229 | 0.232 |
| events with ≥3 strips on both planes | 0.178 | 0.168 | 0.162 | 0.166 | 0.178 |

Gain rises ×1.9 from 465 to 510 V and then rolls over — 505–510 V is the top of
det4's curve at this drift field, so the 6-24 run at 495 V was already within
~15 % of the best available. Over that whole 60 V range the live area does not
move (0.22–0.24) and the fraction of reconstructable events does not move
(0.16–0.19). **The dead stripes get slightly louder; they do not turn on.**

The scale of the problem: the troughs are ~60× below the peaks. The measured
gain slope is 1.4 %/V, so closing a 60× gap needs ≈ 290 V more — against ~15 V
of headroom before the chamber tops out and starts discharging. Even if the
threshold bias on the amplitude slope makes that estimate 2× pessimistic, it is
not reachable.

Untested knob: det4 has only ever run at **drift 600 V**, while det2/det3 ran at
1000 V. Higher drift field is worth trying for overall charge, but a uniform
field change cannot create stripes and cannot un-create them.

## 5. What 200 GeV muons would change

Not the thing that is broken.

- **Ionisation.** A 200 GeV muon (βγ ≈ 1900) sits on the Fermi plateau; the
  bench cosmic spectrum is a few GeV (βγ ≈ 20–40), just above minimum. Expect
  ~10–30 % more ionisation. On the measured gain slope that is worth **≈ 15 V of
  mesh voltage** — and we measured that 60 V changes nothing about the dead area.
- **Primary statistics are not the limit anyway.** ~29 clusters/cm in Ar/iso 95/5
  over the 30 mm drift gap is ~85 primary clusters per track. det4 is not short
  of ionisation; it is short of avalanche over 62 % of its area.
- **What a beam does add:** rate (10⁴–10⁶ tracks per spill vs ~1 Hz of cosmics),
  a controlled angle scan, a better telescope, and rate-capability tests. All of
  those are real — but they would be measured on the 38 % of det4 that works,
  and the answers would be properties of *this defective bulk*, not of the MX17
  design.

## 6. Recommendation

**Do not take det4 to the SPS as a characterisation target.** What the beam
would measure, det4's cosmic data already says: in its live stripes it is a
77 %-efficient, 0.59 mm, 2.1° micro-TPC; between them it does not amplify; no
operating point changes that. A test beam would spend a scarce slot to
re-measure a known chamber defect at higher statistics.

Two things that *are* worth doing, both cheap:

1. **Re-check it on the bench before writing it off permanently.** The stripe
   measurement is from June, on a chamber bulked 22 June and measured 23–24 June.
   A few hours on the cosmic bench (`01`/`04` here re-run on the new key) tells
   you whether the pattern is still there and whether it has grown. If the
   chamber was re-bulked or re-tensioned since, this is the only measurement
   that matters.
2. **Feed the pattern back to the bulking.** 11 irregular non-amplifying bands
   covering 62 % of the area, stable in position, identical on both planes, is a
   lamination/mesh-spacing failure map. Together with the run_config note
   ("Had a few bubbles, but appears that the pillars underneath were still
   there, so just no caps") that is a concrete QA datum for the construction
   team — arguably the most useful thing det4 has produced.

**If a beam slot is available anyway, the target is X 177–215 mm** (reference
Y −10 to +30 on the sliding map), and §3b shows that is a better proposition
than "one stripe of a broken chamber": 82 % within 5 mm, 90 % excluding
discharges, det3-like cluster sizes, 0.62 mm, ~2°, uniform over its full 360 mm
length, and it tolerates the inclination a micro-TPC scan needs without any need
to tilt the chamber along the stripes. An 8 cm spot centred at X ≈ 183 mm
averages 69 % and can be selected offline. Do **not** use X 355–398 despite it
being the widest band — that is the stale-pedestal connector region and its
X-plane angles are 2× worse. What such a run cannot deliver is any statement
about the *chamber*: every full-area efficiency or uniformity number would be a
property of this bulk's defect, not of the MX17 design.

## Reproduce

```bash
cd <repo>
.venv/bin/python sps_beam_test_26/det4_sps_assessment/01_uniformity.py \
    g_det4 sat_det3 o22_long_det2 g_det6_long g_det7_long
.venv/bin/python sps_beam_test_26/det4_sps_assessment/02_bands.py g_det4 sat_det3
.venv/bin/python sps_beam_test_26/det4_sps_assessment/03_charge_vs_position.py \
    g_det4 sat_det3 o22_long_det2 g_det6_long g_det7_long
.venv/bin/python sps_beam_test_26/det4_sps_assessment/04_stripe_metrics.py g_det4 --split
.venv/bin/python sps_beam_test_26/det4_sps_assessment/05_hv_headroom.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/06_fleet_profiles.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/07_crossrun_stability.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/08_connector_test.py g_det4
.venv/bin/python sps_beam_test_26/det4_sps_assessment/09_periodicity.py g_det4
.venv/bin/python sps_beam_test_26/det4_sps_assessment/10_pedestals.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/11_strip_break_test.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/12_edge_sharpness.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/13_beam_window.py
.venv/bin/python sps_beam_test_26/det4_sps_assessment/14_board_map.py
```

`08`–`11` read `stripes_g_det4.json`, so run `04` first.

Figures and json land next to the scripts. Note for whoever repeats the HV work:
FEU 6 connectors 7–8 (local X > 300 mm) were **not read out** in the 6-23 run —
they carry the highest occupancy on 6-24 — so they must be excluded from any
6-23 comparison rather than counted as dead chamber. `05` and `07` do this.
