# Pogo-field short check: method

**Question.** Each MX17 readout plane is two M1 cards. Each card reaches the readout board through a 10 × 30 field
of pads on a 1 mm pitch, pressed together by a JZ pogo-pin carrier. If a card is seated badly, pins can bridge
neighbouring pads, so two strips end up on one DREAM input. These cards are hard to align (det7 X1/Y1 "gave trouble
while tightening", logbook 06-29). This check asks, for any data set, whether a plane shows that fault.

**The trap.** The obvious test is "channels that are pogo neighbours are correlated". Within a Mec8 block, a pad's
grid neighbours are ±1 (radial), ±5 (along the edge) and ±4/±6 (diagonal) channels. With the linear strip map those
are also strips 1, 4, 5 and 6 apart on the detector, and every pair of strips 5 apart on the same connector *is* a
vertical pogo pair. Neighbouring strips share charge. Plain amplitude correlation falls from 0.6–0.8 at 1 strip to
0.14–0.23 at 5 strips on every bench plane (`figures/corr_vs_distance.png`). An inclined track also covers 5 strips
easily. So "pogo neighbours correlate" will fire on a healthy chamber. Comparing pogo neighbours with non-neighbours
at the same distance is impossible inside a connector, because at d = 5 there are no non-neighbours.

## The tests

Hits are used for QA only: which channels fire together and how big they are. No position, angle or depth is
reconstructed from hit times. Events with any saturated hit and events with > 40 hits are dropped. A channel
"fires" at amplitude > 50 ADC.

1. **Jump test (track-proof).** A cluster is a run of hit strips in strip order, broken by ≥ 2 quiet strips.
   J(a, b) counts the events in which a and b both fire **in different clusters**. Charge sharing and a track's own
   extent only ever make contiguous runs, so a firing across quiet strips is something the detector cannot do by
   itself. The exceptions are accidentals, a second track, a delta ray, and one inclined track split by
   under-threshold strips.
   - Rate: r(a→b) = J / F(a).
   - Background: the median of r over the *other card's* channels, i.e. b's own accidental and pickup level.
   - Excess: e = r − background, symmetrised as e_sym = min of the two directions. A link needs e_sym ≥ 0.10 and
     J ≥ 10.
2. **Same charge.** Over the jump events, ln A_a must follow ln A_b (ρ ≥ 0.5). The ratio must also be identical
   event by event: sd ln(A_a/A_b) < 0.30 and |mean| < 0.20. A shared node reads one charge (det7 X nets: sd ≈ 0.1).
   The two halves of a split track do not (det3 X: 42 such pairs, sd 0.3–0.7). A plain "|ln ratio| < tol" count is
   fooled by hits just above threshold, which all look alike. The regression is not.
3. **Excess correlation at the same distance.** C (cosine of amplitude products) is compared with the **same
   plane's** distribution at the **same strip distance**: Z = (C − median_d) / (1.4826 MAD_d). A link also needs
   C ≥ 0.8, Z ≥ 8 and d ≥ 3. This is the distance-matched null for the naive correlation. It catches nets whose
   strips are contiguous, where no quiet gap ever forms and test 1 is blind.
4. **Null that cannot be a pad short.** Pairs on *different* cards sit on two separate pogo fields. Anything that
   links them is pickup, crosstalk, accidentals or an edge effect. Their link rate is the plane's false-link rate,
   and channels with cross-card links are kept out of pogo claims.
5. **Layout.** Nets are the connected components of the within-card links.
   - **Pogo-shaped net:** its members are connected through links between pogo-grid neighbours alone, and none of
     them has a cross-card link.
   - **Enrichment:** the link rate on pogo-neighbour pairs divided by the link rate on non-neighbour pairs more than
     12 strips apart (beyond two pogo steps).
   - **Card verdicts:**
     - **POGO:** a pogo-shaped net of ≥ 3 channels, or two of any size, and enrichment ≥ 10.
     - **POGO?:** a single pogo-shaped pair.
     - **OTHER:** links the layout does not explain.
     - **CLEAN:** no links.
6. **Adjacent twins (d ≤ 2).** These cannot jump. Pairs whose ratio spread is < 0.25 over ≥ 30 co-fires are listed.
   On the M1 they are either pogo-horizontal or Mec8/cable neighbours, and the data cannot tell which.

**Optional confirmation, bench only.** With M3 tracks, a shorted channel must fire when the track crosses its
partner's strip (`rerun/det7_pogo/fire_offset.py`; det7 X: 0.446 vs 0.226). This is the strongest single test.
Steps 1–6 do not need it, so they run on beam data too.

## Running it

```
# bench: accumulate both planes of a chamber (~10 s per run), then judge, then the report
../../.venv/bin/python run_bench.py g_det7_long --feu-x 6 --feu-y 8 --subs long_run longer_run --out-dir out
../../.venv/bin/python pogo_check.py out/*.npz --out-dir results
../../.venv/bin/python make_report.py
```

- **Another data set:** feed `accumulate.PairStats.add(channels, amplitudes)` one event at a time for one FEU, with
  the strip index from that run's strip map (`strip_index_from_positions`), and save. `pogo_check.py` only needs the
  npz.
- **SPS det4:** `run_sps.py` (see "Checked" below).
- **n_TOF beam:** needs a reader for the beam hit cache. The track-proof tests still hold, but beam events are busier.
  Lower `MAX_HITS` or require a single cluster per plane before trusting the jump test there.

## Assumptions and limits

- **Layout.** The pad → Mec8 wiring comes from the M1 gerbers. The Mec8 → DREAM convention was chosen by the det7 X
  data (`rerun/det7_pogo/conventions.py`). Both are assumed identical for every card and every chamber, since the
  M1 is one design. A card wired differently would look like OTHER.
- **Noisy channels** (rms ≳ 60, global pickup) spread the amplitude ratio and can fail the same-charge test even when
  they are shorted. Their membership is under-reported unless the per-group common mode is removed first.
- **Contiguous nets** with no strong excess correlation (C < 0.8) are invisible to every test here.
- **The data cannot tell what caused a pogo-shaped net:** leaning pins, debris or a bowed card.
- **Checked:** one golden bench night per chamber, and det4 at H4 (run_71 RAW, `run_sps.py`: FEU 03 renumbered to M1
  channels, connectors 3–6, both cards; CLEAN, 0 cross-card links, but only the beam-lit channels are testable: 98/256
  X, 201/256 Y). Other nights can differ (det7's X1 was reseated and glued on 06-30).
