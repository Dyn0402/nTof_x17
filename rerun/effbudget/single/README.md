# det3 single-track efficiency, X/Y pairing, matching radius (Paper I §5, 2026-11-02)

Three questions from Dylan on the det3 efficiency, answered on the pass-1 tables of the two det3 golden rows
(`sat_det3`, 6 454 interior M3 tracks; `c26 … 6-27 p2 long_run`, 20 088). Inputs are read from EOS through
`root://eospublic.cern.ch` (the job copied them to `$WORK_DATA/claude-jobs/17c92c46/`, same layout as EOS
`cosmic_bench/`, so `--root` is either). Results in `results/`.

| script | question | output |
|---|---|---|
| `single_track.py` | can a single-track efficiency be isolated without the chamber selecting itself? | `single_track.json`, per-event parquet |
| `xy_pairing.py` | with two particles, are X and Y segments paired correctly, on real cosmics? | `xy_pairing.json`, per-event parquet |
| `radius.py` | efficiency against the matching radius; a reason for R | `radius.json` |

## 1. Single-track efficiency: a self-selection bias, measured

A1's "multi-particle" flag is set by det3 itself (≥ 2 particle-like candidates in a view), so a chamber failure that
looks like two tracks would leave a "single-track" sample by its own failure.

- **An external tag does not work on the bench.** M3 sees a second straight 4-layer track in 2 of det3's 276
  multi-flagged events (22 with χ² < 10 mm², 44 with < 50): det3's second particles are local (soft shower
  electrons, secondaries from the material above, at large angles: tan spread 0.38), not through-going.
  The M3-clean sample therefore contains them, and its efficiency equals the overall one (95.4 %).
- **A two-view test does.** A real second particle is ionisation in the gas, so both views see it with the same
  charge (X/Y charge ratio spread 6 % per event on single tracks). The skeptical rule removes from the sample only
  events whose second particle is seen in **both** views, **charge-paired** (|ln q_x/q_y − med| < 5 σ), and
  **detached** from the muon (not within 3 mm of it anywhere in the gap, in both views: a δ from the muon would stay
  as a loss). Everything else stays in the sample as a loss: second candidates in one view only, spark-vetoed
  "shower" and "muon + unrelated deposit" events (M3 shows no excess activity in them: local), discharges, δ.

| det3, interior, R = 5 mm, POS_TAIL + EDGE out of the denominator | sat_det3 | p2 long |
|---|---|---|
| all tracks, credit (M3 among several reconstructed tracks counts) | 95.43 ± 0.26 | 95.15 ± 0.15 |
| det3-flagged single (biased upward) | 97.14 ± 0.21 | 96.97 ± 0.12 |
| **skeptical single** (all but an independent 2nd particle) | **95.45 ± 0.26** | **95.13 ± 0.15** |
| independent 2nd particle present (both views, paired, detached), credit | 94.5 ± 2.0 (n 127) | 96.1 ± 0.9 (n 440) |
| 2nd candidate in one view only (kept as loss in the skeptical sample) | 47 % credited (n 59) | 44 % (n 174) |
| spark-vetoed shower / muon + other (kept as loss) | 0 % (n 69) | 0 % (n 240) |

**Reading:** the single-track efficiency is **95.2 %** (pooled, ±0.13 stat) when nothing that might be the
chamber's own failure is excused, and at most **97.0 %** if every one-view second candidate and every spark-vetoed
shower were an outside particle. With an independent second particle in the chamber, the muon is still
reconstructed in 94.5–96.1 %: the chamber is as efficient with two particles as with one, once a multi-track event
counts when one of its tracks is the M3 track. No δ-like "attached" second particles were found among the two-view
events (second particles sit a median ~70 mm from the muon).

## 1b. The two loss groups looked at closely (`single_track_v2.py`, Dylan 2026-11-02 p.m.)

**One-view second deposits (233 events): mostly two real particles merged in one projection.** In the view with a
single candidate, that candidate is wider than a single track (median 15 strips on the muon, 25 when off it, against 12),
carries the charge of both particles, or sits 5–15 mm off the muon (between the two, inside the 12 mm seed gap).
195 events (84 %) are "merged in projection": a two-particle event at the two-track separation limit of one projection,
not a chamber malfunction; 41 % of them are still credited. 14 are a faint (⅓ of the muon), short deposit seen in one
view only: all 14 credited, so not losses. 24 unclear (kept in the sample).
**Joint two-track fit on these (2026-11-02, `MERGED_PROJECTION_2026-11-02.md`):** pass 1 refused every one, through
the beam-tuned `plausible` flag in its trigger and acceptance and a column-time coincidence that short secondaries fail.
The bench mode `xy_pairing['split_corrob'] = 'charge'` (corroboration by the other view's charge; one plausible child)
raises the credit on the 195 from 41.0 to 53.8 % and the correctly paired muon from 20.5 to 36.4 %. It adds no split on
clean single muons on any of the five chambers, and the single-track efficiency does not move. It takes effect at the
next bench re-pass, and n_TOF stays on `count`.

**Spark-vetoed deposits: local, i.e. δ-rays, not outside showers.** Extra M3 clusters per event, upper / lower doublet
(sat_det3): detected muons 0.56 / 0.72; independent second particle 1.55 / 1.46; **'shower' 0.49 / 0.47**;
'muon+other' 0.65 / 0.38; discharge 0.72 / 0.84. Nothing accompanies the 'shower' muons through the telescope, so the
deposit is made in det3 (gas, cathode, window). 54–80 strips over threshold at 0.7–0.9 % of muons: in order of
magnitude, knock-on electrons above ~0.1 MeV (~1–1.5 % of muons). Counted as losses with the other δ classes;
'muon+other' (a clean muon plus a separate deposit, vetoed together) is a two-particle event and leaves the sample.

**Revised single-track efficiency (pooled, 25 588 tracks): 96.0 ± 0.1 %** (sat_det3 96.24 ± 0.24, 6-27 long
95.87 ± 0.14); 96.8 % if the δ-ray deposits are excused too. Losses: δ 2.5 % (on track 0.9, saturated blob 0.9,
vetoed 0.7–0.9), discharges 0.5, noise bands 0.2, fit off 0.3, non-particle second deposit 0.2. Hardware alone
(discharges + bands): 0.7 %.

**Fiducial** (`results/fiducial_variants.json`, both rows): 30 mm from every side vs the current 30 x / 8 y interior:
production 92.31 vs 92.29 %, credit 95.01 vs 94.98 %, single-track 95.83 vs 95.96 %, −9 % tracks. Whole active area
91.50 / 94.13 %. Efficiency is insensitive to the y cut.

## 1c. Full active area, and the two-particle breakdown (`single_track_area.py`, Dylan 2026-11-02 evening)

Headline over the **full active area** (the edge cut moves efficiency ≤ 0.9 pt): production 91.81 ± 0.16 %, credit
94.45 ± 0.13 %, **single-track 95.14 ± 0.13 %** (interior 96.10, reproducing single_track_v2's 95.96 to 0.14 pt with a
candidate-only classification that is defined everywhere). Near the edges unclassified losses stay losses.
Two-particle events (3.2 %): separate in both views 95.5 % (620); merged in one projection 44 % (232); clean muon +
separate deposit tripping the discharge veto 1 % (93); all 73.7 %. Dylan: these are fair two-particle events and
count as losses when the muon is not recovered; anything we cannot classify counts as a loss.

**δ-rays are one population by size** (median strips over the 10 % floor): detected 12, saturated blob 19,
δ on track 34, vetoed 'shower' 67 (all > 50 by definition); same muon |tan| 0.17 in all; no M3 activity.
Discharges: 113 strips, 220–230 at SNR > 6.

**Overlap in one projection:** the joint two-track fit (cross-plane corroboration, F 120) is aware of it, but its
trigger and its acceptance both require beam-tuned `plausible` candidates/children: 65 % of the 195 interior merged
events never attempt a split, all 68 attempts refused (`guards_ok` False: both_plausible 44, column_shared 27,
distinguishable 20). Being fixed in a separate session: `MERGED_PROJECTION_2026-11-02.md` (pending).

## 2. X/Y pairing on real two-particle cosmics

552 events (both rows) with ≥ 2 candidates in each view where M3 marks exactly one X and one Y candidate (< 5 mm,
the next > 10 mm). Keep-or-swap with the brightest other candidate of each view:

| rule | correct | similar charge (\|ln q1/q2\| < 0.5, n 268) | different charge (n 284) |
|---|---|---|---|
| charge (\|ln q_x/q_y − med\|) | 95.7 ± 0.9 % | 91.0 % | 100 % |
| charge + charge-arrival time (u50) | **98.0 ± 0.6 %** | 95.9 % | 100 % |
| production track_id (pass 1 bench chain) | 80.1 % (15.9 % wrong, 4.0 % M3 particle in no track) | | |
| production, events with every candidate `plausible` (n 386) | 94.8 % | | |

With the X/Y charge share this balanced, charge matching is the same decision as brightest-with-brightest; the
time features resolve most of the similar-charge cases. **Finding (bench chain):** production's track selection
drops the second track when a candidate fails the beam-tuned `plausible` gate (False on ~16 % of bench winners,
PASS1_RESULT bug 4), and the single surviving pair then often joins the M3 segment of one view to the other
particle's segment in the other. The pairing cost is right; the plausibility gate should not apply on the bench.
**Fixed 2026-11-02** (`wft.reco.select_tracks` mode `cost`, bundle `xy_pairing['select']`): 96.7 % on these 552,
no change on single-candidate events, n_TOF unchanged. Mechanism, validation and what needs re-running:
`PAIRING_FIX_2026-11-02.md` (scripts `pairing_replay.py`, `pairing_m3_all.py`, `pairing_rereco.py`,
`pairing_rereco_compare.py`, `pairing_ntof_check.py`).

## 3. Matching radius

eff(R) has two regimes: the resolution core (core σ 0.55 mm per view at the mesh, M3 pointing + scattering), steep
to ~4 mm; then a slow rise of 0.1–0.2 pt/mm to 40 mm made of mis-measured tracks (A1's δ / saturated blob / bent M3)
and accidentals (event mixing: 0.07 % at 5 mm, 0.27 % at 10, 4.4 % at 40). **R = 5 mm is the knee: 9 core σ, the
Gaussian core exhausted, accidentals 0.07 % (< the statistical error, 70× below the inefficiency).**
R ∈ [4, 8] mm moves the efficiency −0.5 / +0.6 pt (systematic). The tracks found 5–15 mm off are counted as losses,
consistent with δ-pulled fits being losses.

| R [mm] | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| sat_det3 production | 87.3 | 90.9 | 92.2 | 92.6 | 92.9 | 93.2 | 93.5 | 94.6 | 95.3 |
| sat_det3 credit | 90.6 | 93.7 | 94.9 | 95.4 | 95.6 | 96.1 | 96.4 | 97.7 | 98.3 |
| accidental (upper bound) | 0.01 | 0.02 | 0.04 | 0.07 | 0.09 | 0.17 | 0.27 | 1.13 | 4.39 |

## What this does not rule out

- One-view second candidates may be real particles whose other view merged with the muon or fell under threshold;
  the skeptical number counts them as losses (that is the 95.2–97.0 bracket).
- The spark-vetoed showers have no reconstructed candidates, so the two-view test cannot be applied to them.
- Two rows of one chamber, one gas, one HV (490 V); the other chambers are not analysed (Paper I focuses on det3).
