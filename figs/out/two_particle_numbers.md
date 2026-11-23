# Chamber A two-particle and single-track numbers: one consistent account (2026-11-10)

Paths are relative to the repo root unless they start with `~`. Two reconstruction versions appear in the sources:

- **first pass (pass 1):** EOS `Analysis_pass1` tables, the original bench bundles (`select = gate`, `split_corrob = count`).
  Local copies of the study outputs on it: `~/work_data/claude-jobs/twotrack/out_p1/*.json`.
- **two-track re-pass:** EOS `Analysis_pass1_twotrack`, chamber A Runs 1 and 2 re-reconstructed from the waveforms with
  `select = cost` (pairing fix) and `split_corrob = charge` (merged-projection split). Outputs:
  `rerun/effbudget/single/results/*.json` (committed) and `~/work_data/claude-jobs/twotrack/out_tt/*.json` (identical for
  `single_track_area`; see the note on Table 2 below for the one small difference).

The paper's current text uses the re-pass. Every number below is Runs 1+2 pooled unless it says otherwise.
"Interior" = 30 mm from the x sides, 8 mm from the y sides; "full area" = the measured active area.

## 1. The two-particle sample (classification of `single_track_area.py`)

Source: `rerun/effbudget/single/results/single_track_area.json` (re-pass) and
`~/work_data/claude-jobs/twotrack/out_p1/single_track_area.json` (first pass). Credit = some reconstructed candidate within
5 mm of the reference ray in both views (`det_any_5`) or the production pick detected; nothing removed from any denominator.

| configuration | full area, re-pass | muon within 5 mm | interior, re-pass | muon within 5 mm |
|---|---|---|---|---|
| (i) separate in both views | 662 | 95.6 ± 0.8 % (633) | 619 | 95.8 ± 0.8 % |
| (ii) merged in one projection | 176 | 47.2 ± 3.8 % (83) | 159 | 46.5 ± 4.0 % |
| (iii) muon + separate deposit, > 50-strip veto | 93 | 1.1 ± 1.1 % (1) | 93 | 1.1 ± 1.1 % |
| all two-particle events | **931** (3.2 % of 29 398) | **77.0 ± 1.4 %** | 871 (of 26 542) | 76.7 ± 1.4 % |
| single-track (everything else) | 28 467 | 94.79 ± 0.13 % | 25 671 | 95.70 ± 0.13 % |

Class (iii) exists only in the interior (the discharge-veto waveform class is defined there; edge-zone veto rays stay in the
single sample as losses), so it is 93 in both columns. The text's 931 = 662 + 176 + 93 with 95.6 / 47 / 1 % and 77.0 % is
correct (full area, re-pass); `papers/paper1/figs/data/two_particle_eff.json` holds the same numbers.

**First pass, same script** (`out_p1/single_track_area.json`): full area 620 / 232 / 93 = 945, 95.5 / 44.4 / 1.1 %, all
73.7 %; single 28 453 at 94.81 %. Interior 579 / 213 / 93 = 885.

**Why "620 to 662, 232 to 176" does not conserve 852 → 838.** Matched event by event (out_p1 vs out_tt per-event parquet,
`single_track_area_events_*.parquet`): of the 232 first-pass merged events, 176 stay merged, **42 move to (i)** (the split
gives the merged view a second segment that is charge-matched to the other view) and **14 move to the single-track sample**
(the split succeeds but the resulting second pair fails the 5σ charge match or is "attached" within 3 mm, so the event no
longer qualifies as two-particle; 3 of the 14 were credited before, 7 after). Nothing else moves (single → two-particle: 0;
(i) and (iii) unchanged). So 945 → 931 and 28 453 → 28 467. The text should say "232 to 176 (42 to separate, 14 to single
tracks)".

### The other counts in the text and notes

| number | what it is | version | area | source |
|---|---|---|---|---|
| **276** | Run 1 only: events chamber A itself flags as multi-particle (A1 classes m3_picked, other_picked, m3_not_found, far:MULTI, spark:shower, spark:muon+other), POS_TAIL/EDGE out; the reference sees a second straight line in 2 of them. Runs 1+2: **1 192**. This is A1's self-tag, not the two-particle sample, and it includes δ-ray "showers". | either (identical) | interior | `results/single_track.json` → `sat_det3.selections["det3-tagged multi"]`, crosstab |
| **567** | pairing sample: ≥ 2 candidates in both views, reference marks exactly one X and one Y candidate (< 5 mm, next > 10 mm) | re-pass | interior | `results/xy_pairing.json` (126 Run 1 + 441 Run 2) |
| **552** | the same selection on the first-pass tables | first pass | interior | `out_p1/xy_pairing.json` (124 + 428); README §2, PAIRING_FIX |
| 567 vs 552 | all 552 are in the 567; the 15 extra are events where the re-pass split created the second candidate of the merged view | | | event match of the two `xy_pairing_events.parquet` |
| 605 | "independent second particle" (both views, charge-paired, detached) of `single_track.py` — a different classifier, run only on A1-flagged events, POS_TAIL/EDGE out; first pass 567 (a coincidence with the pairing 567) | re-pass | interior | `results/single_track.json` |
| **146** | A1 *loss* class `multi:m3_not_found` (merged in projection, only tracks *not detected* by the production pick), first pass; 121 of them stay in it on the re-pass, and 25 get a gated, reference-matched track (Run 1: 3/37) | first pass | interior | `rerun/effbudget/pass1/results/a1_table.json` → `merged_followup` |
| **195** | "merged in projection" of `single_track_v2.py` (one-view events among the A1-flagged ones: 195 merged + 14 faint + 24 unclear = 233), the MERGED_PROJECTION fixed set (credit 41.0 → 53.8 %, i.e. 80 → 105) | first pass | interior | `out_p1/single_track_v2.json` (49 + 146; the Run 2 count 146 is a second coincidence), `results/merged_effcheck.json` |
| 142 | the same classifier on the re-pass | re-pass | interior | `out_tt/single_track_v2.json` |
| **232** | merged class of `single_track_area.py` | first pass | full | `out_p1/single_track_area.json` |
| 213 | same | first pass | interior | same |
| **176** | same | re-pass | full | `results/single_track_area.json` |
| 159 | same | re-pass | interior | same |

So the merged class has three classifiers (A1 loss class, `single_track_v2` one-view classifier, `single_track_area`
candidate classifier), two versions and two areas. Only `single_track_area` on the re-pass (176 full / 159 interior) belongs
in the main text. The 195 is legitimate only as "the fixed set on which the split was validated (first-pass classes)".

## 2. Pairing efficiency of the standard reconstruction, chamber A

| number | what it is | source |
|---|---|---|
| **96.6 ± 0.8 %** (548 / 567; wrong 13, unpaired 6) | production `track_id` on the 567 events, two-track re-pass (cost selection + split) | `results/xy_pairing.json` → `pooled.all.production` |
| 96.7 ± 0.8 % (534 / 552) | the cost selection replayed on the first-pass tables, same golden-run events | `PAIRING_FIX_2026-11-02.md` §3; `results/pairing_replay.json` |
| 80.1 % (442 / 552) | first-pass `gate` selection (the "earlier selection") | same |
| **95.9 ± 0.5 %** (1 447 / 1 509) | cost selection replayed on the first-pass tables of **all 63 chamber A rows** (every HV/drift setting, not only Runs 1 and 2), interior | `results/pairing_m3_all.json` → `chambers.mx17_3[">=2 both"]` |
| B 96.8 ± 0.7 (614/634), C 92.2 ± 1.6 (273/296), D 90.1 ± 2.3 (146/162), E 94.0 ± 2.2 (109/116) | same, all rows of each chamber | same file |

The study rules on the 567 (not the production reconstruction): charge + time 97.7 ± 0.6 % (554), charge alone
95.6 ± 0.9 % (542); first pass 98.0 / 95.7 % on 552 (`out_p1/xy_pairing.json`). The "brightest X with brightest Y" rule
gives exactly the charge-only answer on these events (identical verdict in all 567).

**Recommendation.** Quote 96.6 ± 0.8 % for chamber A (it is the reconstruction and the events of the rest of the section).
Keep the per-chamber list only as a separate, labelled comparison ("all runs of each chamber, first-pass tables with the
pairing selection replayed: A 95.9, B 96.8, C 92.2, D 90.1, E 94.0 %, n = 1 509 / 634 / 296 / 162 / 116"), so that A's 95.9
is not read as a second measurement of the 96.6.

## 3. "With an independent second particle": 95.9 ± 0.8 against 95.6 ± 0.8

Area is **not** the only difference.

| | 95.9 ± 0.8 (Table 2) | 95.6 ± 0.8 (two_particle figure, text) |
|---|---|---|
| script / classifier | `single_track.py` (`indep_2nd`: both views, charge-paired, detached; evaluated only on A1-flagged events) | `single_track_area.py` class (i) (candidate tables, every event) |
| area | interior | full |
| denominator | POS_TAIL and EDGE A1 classes removed | nothing removed |
| credit | A1 class ∈ {detected, m3_picked, other_picked} | `det_any_5` or detected |
| n, k | 605, 580 | 662, 633 |
| source | `results/single_track.json` | `results/single_track_area.json` |

With area alone changed, `single_track_area` gives 95.8 ± 0.8 % (n 619, interior). The three numbers agree within errors, but
the paper should use one classifier: 95.6 ± 0.8 % (662, full area) or 95.8 ± 0.8 % (619, interior).

## 4. Single-track efficiency: the four numbers

| number | definition | n | source |
|---|---|---|---|
| **94.8 ± 0.1 %** (94.79) | `single_track_area.py`: full area, every reference track except the 931 two-particle events; credit `det_any_5`; nothing removed | 28 467 | `results/single_track_area.json` → `pooled["active area \| single-track"]` |
| **95.7 ± 0.1 %** (95.70) | same, interior | 25 671 | `pooled["interior 30x/8y \| single-track"]` |
| 95.94 ± 0.12 % (Table 2) | `single_track_v2.py` "v2": interior; removes POS_TAIL + EDGE (100 tracks, all lost) **and** the two-particle classes of `single_track.py` / `single_track_v2.py`; A1-class credit | 25 603 | `results/single_track_v2.json` → `single_track_v2.pooled["v2: (i)-(iii) removed"]` (`out_tt` gives 95.92: overlay read; `results/` is the direct read of the re-pass products) |
| 95.1 % (`eff_vs_radius` caption) | **all** interior reference tracks, two-particle events included, credit at R = 5 mm — not a single-track number | 26 542 | `results/radius.json` (95.24 × 6 454 + 95.02 × 20 088) = `single_track_area.json` `interior \| all rays: credit` 95.08 |

Decomposition of 95.70 → 95.94: removing POS_TAIL + EDGE from the `single_track_area` interior sample gives 96.07 %
(24 567 / 25 571); the other classifier and the A1-class credit then take it to 95.94 (−0.13 pt). The Table 2 caption's
26 442 is the interior after the same removal. The text's statement that "nothing is removed" is true of 94.8 / 95.7, not
of Table 2.

Table 2's other rows (95.31, 97.01, 95.30, 96.75, 95.9) are all on the `single_track.py` definitions (interior, POS_TAIL/EDGE
out, A1-class credit), `results/single_track.json` and `results/single_track_v2.json`.

**Recommendation.** One number per area everywhere: **94.8 ± 0.1 % (full area, n 28 467) and 95.7 ± 0.1 % (interior,
n 25 671)**, both `single_track_area.py`, nothing removed. Rebuild Table 2 on the same definitions (or label it explicitly
"POS_TAIL and EDGE removed, A1-class credit"); the eff_vs_radius caption should say "all reference tracks" (95.1 %).

## 5. The loss budget

**Hardware (discharges + noise bands).**

| number | definition | source |
|---|---|---|
| 0.57 % (discharges 0.47 + bands 0.10) | lost single tracks under the credit definition, full area, of 28 467 (133 + 28) | `results/single_track_area_events_*.parquet` (`credit == False`, recat), as drawn in `loss_budget` |
| 0.63 % (0.52 + 0.11) | same, interior single tracks, of 25 671 — the "0.6 %" of the result paragraph | same |
| **0.72 %** (bands 0.22 + discharges 0.50) | appendix A: A1 classes of **all** interior reference tracks (26 542, two-particle events included) lost by the **production pick** (strict detection): 58 band + 133 discharge | `rerun/effbudget/pass1/results/a1_table.json` |

The difference is almost all the noise bands: A1 counts a band track as lost when its view fails the competing-cluster
cut, but 30 of the 58 band tracks have a candidate within 5 mm in both views and are credited by the paper's definition.
Discharges are 133 in both (0.47 % of 28 467, 0.50 % of 26 542). Recommend: main text "discharges and noise bands cost
0.6 % of single tracks (0.47 + 0.10 % over the full area)"; appendix A says that its classes are of the production pick
over all interior tracks, so the band class (0.22 %) is about twice what the efficiency definition loses (0.10 %).

**The edges.**

| number | definition | source |
|---|---|---|
| 0.9 pt (0.91) | interior single-track efficiency minus full-area (95.70 − 94.79): the edge zone's losses (380 of 2 796 edge-zone single tracks, 1.33 % of 28 467) minus what those tracks would lose at the interior rate (120 tracks, 0.42 pt) | `results/single_track_area.json`; per-event parquet |
| 1.5 pt (1.52) | everything the loss_budget figure puts under "edges", as % of the full-area single tracks: all edge-zone losses (veto 1.01 + fit off 0.32; δ-rays and discharges in the edge zone are not classified and land here too) plus the interior class "edge of the other coordinate" (0.19) | `rerun/effbudget/single/make_figures.py` `GROUPS`, same parquet |

The 1.5 pt overstates what the edges cost: about 0.4 pt of it is the ordinary (δ, discharge, fit) loss rate of the tracks
that happen to be in the edge zone, and the 0.19 pt other-coordinate class is inside the interior number too. Recommend one
statement: "the edge zone costs 0.9 pt of the full-area efficiency (94.8 against 95.7 %); a further 0.2 % of tracks are lost
where they leave through the other coordinate's edge", and in the improvements section "edges, up to ≈ 1.1 pt" (0.9 + 0.2)
rather than 1.5.

## 6. Pairing figure numbers (for the caption; `papers/paper1_v2/figs/out/pairing.json`)

- 567 events, recomputed from the re-pass tables with `xy_pairing.py`'s selection and reproducing its stored verdicts
  exactly (closure in `pairing.json`).
- Charge only 95.6 %, charge + time 97.7 %. Charge ratio ≥ 1.65: 290 / 290 for both. Ratio < 1.65: 91.0 % (charge) and
  95.3 % (charge + time), n 277. Every failure has a ratio below 1.26 (charge) / 1.38 (charge + time); below 1.05:
  41 % / 81 % (n 32).
- Time handle: u50_X − u50_Y (difference between the views of the median arrival time of the fitted charge after t0),
  half-width 43 ns on single tracks, 48 ns on true pairs, 176 ns on swapped pairs. It rescues 18 charge-only failures and
  spoils 6 charge-only successes. The production cost uses ln(qX/qY) (per class), u50, u90 and the column time t0 + u50
  (bundle `xy_pairing.features`); the study rule uses ln(qX/qY) and u50 only.

## What the paper should say (minimal set)

1. **Single-track efficiency, chamber A, Runs 1+2: 94.8 ± 0.1 % over the full active area (28 467 tracks), 95.7 ± 0.1 % in
   the interior (25 671)** — any reconstructed track within 5 mm of the reference in both views; every reference track
   except the two-particle events; nothing else removed.
2. **Two-particle events: 931 of 29 398 (3.2 %), full area** — separate in both views 662 (muon found 95.6 ± 0.8 %),
   merged in one projection 176 (47 ± 4 %), muon + separate deposit tripping the 50-strip veto 93 (1 %); all 77.0 ± 1.4 %.
3. **The two-track re-pass moved 232 merged events: 42 to "separate", 14 to single tracks, 176 left** (first pass
   620 / 232 / 93 = 945).
4. **X/Y pairing, 567 interior two-particle events with a reference-tagged particle: standard reconstruction 96.6 ± 0.8 %;
   charge + time rule 97.7 ± 0.6 %, charge alone 95.6 ± 0.9 %; no failure when the two charges differ by more than a factor
   1.4 (1.65 is a safe statement), 95 % (charge + time) when they are within 1.65.** Per chamber, all runs, first-pass tables
   with the pairing selection replayed: A 95.9, B 96.8, C 92.2, D 90.1, E 94.0 %, clearly labelled as such.
5. **Hardware floor: discharges + noise bands = 0.6 % of single tracks** (0.47 + 0.10 % full area). Appendix A's 0.72 % is
   explicitly "production pick, all interior tracks".
6. **Edges: 0.9 pt** (full area against interior); recoverable by edge-aware reconstruction ≲ 1.1 pt including the
   other-coordinate edge class.
7. **Merged-projection split validation: on a fixed set of 195 interior first-pass merged events, the muon is credited in
   53.8 % (from 41.0 %)** — say it is the first-pass classifier's set. Drop "146" from the appendix or call it "146 lost
   tracks of the A1 merged class".
8. **"276" is Run 1's A1 self-tagged multi-particle events (1 192 for Runs 1+2)**, which the reference confirms as a second
   straight track in only 2; say "Run 1" in the text.
9. Table 2: either rebuild on the definitions of item 1 or caption it "POS_TAIL and EDGE classes removed, A1-class credit";
   the eff_vs_radius caption's 95.1 % is all interior tracks, not single tracks.
