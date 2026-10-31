# SPS: the oscillation's neighbours (open items 1 and 3 of rerun/sps_audit)

**Verdict (2026-10-31): fixed at the root, in the builder.** The oscillating Y ch 373 couples in-phase copies onto its
neighbours (371/374/375/376/370), and those copies won the seed. Now, in the events where the oscillator is ON, they
are censored per event. Weak oscillators next to ch 372 are found by their ON rate, the hot probe runs in blocks, and it
compares a channel with its live neighbours only. Code archive `c36322ec`.

**Re-run:**
- The 30 ZS rows whose events change (census) were re-run, staged and promoted (`results/promoted_rows.txt`).
- The 41 others are unchanged under the rule, bit-identical in refs, hot set and event set, so they keep `3ac0e79`.
- The census of the three RAW run71 rows did not finish. Those rows were not re-run (see open items).

**Y-fit excess at the oscillation**, over the beam's own density (`compare.py`; 4.5 mm bins at 155.6 and 158.0 mm,
summed, before → after):

| row | before | after |
|---|---|---|
| run55 d700 | 19.7 % | ≤ 0 (−1.0 %) |
| run60 ov01 / ov03 / ov05 / ov09 | 1.5 / 1.6 / 1.7 / 1.4 % | 0.1 / 0.3 / 0.35 / 0.0 % |
| run57 d550 / d600 / d650 / d700 | 1.3 / 1.0 / 1.0 / 0.6 % | 0.4 / 0.55 / 0.0 / 0.0 % |
| run59 | 1.1 % | 0.0 % |
| run61 d210 (open item 3) | 1.5 % | ≤ 0 |
| run55 d400 | 0.5 % | ≤ 0 |

All 30 re-run rows sit at ≤ 0.55 % after (≤ 0 = no excess over the neighbouring bins).

The full per-row table is `results/compare.csv`. In the 154–160.5 mm zone the junk falls from 13.5 % to 0.8 % on run55
d700 and from 3–4 % to ~1 % (the beam's own share) on the others.

**Resolution sample: unchanged.**
- run63 flat700: its pass row is unchanged (census), and all 9 telescope files give the same hot set and no coupling
  under the new rules (`res_check.py`).
- run53: the telescope tables were made with X ch 173 (213.7 mm, a stripe-edge strip in the beam) censored as hot. The
  new rule would not flag it. That touches X only; run53's quoted resolution is Y. The tables were not remade.

**Other changes in the re-run rows:**
- run55 d500–d700: 360–391 X fits move, because X ch 172/173 are no longer censored as hot (the band-edge
  false positive).
- run61 d070/d140: X ch 149 (232.4 mm) is newly seed-hot in one 1000-event block, so 131–142 X fits move (see open
  items).
- Elsewhere ≤ 65 Y fits move outside the coupled events (newly censored 373).

## What the junk was

After the seed fix (`rerun/sps_audit`, `3ac0e79`) four CO2-epoch ZS rows kept 2–13 % of their Y fits at 155–160 mm
(run55 d700 13 % in a 154–160.5 mm zone, 7.9 % as an excess over the beam's density; run57 d550–d700, run59, run60
ov01/03/05/09/13 2–4 %), and run61 d210 1.3 %.

Probe of run55 d700 (`probe.py`, `look_probe.py`: the cache's 3000 events, Y channels 356–395):

- ch 373 (157.56 mm) is ON (peak > 300 ADC) in 36 % of the events, with a lobe every ~15 samples: an oscillation;
- in those events ch 375 (156.0 mm) exceeds 300 ADC 80 % of the time against 0.2 % otherwise (×385); 371 ×39; 370 and
  376 carry copies at 100 ADC (×43, ×22). Same phase as 373: electronic cross-talk, not charge;
- 389 of the 405 events whose Y seed sat within 4 strips of the oscillator had it ON.

So the copies win the seed whenever the event's real Y signal is weaker. On run60 ov03 the copy is on 371 and 374
(×32 and ×19 at 300 ADC); on run57/run59 on 371/374/377.

Two further facts set the rule:

- On run55 d400/d600, run57, run59, run60 ov13 ch 373 is ON in 4–11 % of the events at 8–21× its live neighbours, but
  seeds only 1–3 % of a block (the real track usually wins). The seed rule (3 %) misses it. A 1 % seed floor catches it
  but also flags X strips in the beam next to low-gain strips the H4 layer does not list (`lobes.py`/`look_lobes.py`: X 74/75
  at 190 mm, 148/149 at 232 mm, 57, 129, 251; 10–33× their live neighbours in ON rate, single pulses). In the X view a pure
  rate rule is unsafe.
- Periodicity (≥ 3 lobes) is not a usable signature: 373 is 88 % periodic on run59 but 1–5 % on run55 d400/d600 and
  run61 d210, while real CO2 pulses show ≥ 3 lobes in 5–20 % of ON events.

## The rule (`rerun/model_audit/sps_build_cache.py`)

Per plateau, a probe of the sub-run's plateau events (as many as the row keeps, at least 2000):

1. **Seed-hot** (`hot_in_blocks`, `hot_channels`): a channel that seeds > 3 % (`HOT_FRAC`) of a 1000-event block
   (`HOT_BLOCK`) and > 10× (`HOT_RATIO`) the median of its **live** ±4 neighbours (`live_neighbours`: not in the H4
   layer's `dead_stripes`, `STRIPE`), in any block. Changed from: the first 2000 events as one block, all ±4 neighbours.
   - Blocks: an oscillation that is on in part of the plateau (run61 d210) is caught.
   - Live neighbours: the last live strip at a stripe edge in a busy beam (X ch 172/173 at 214 mm on run55) is no longer
     censored as hot; it was, with the beam's edge, in run55 d500–d700.
2. **Near oscillators** (`near_oscillators`): a channel within 2 ranks (`OSC_NEAR`) of a SEED_BAD channel (372, 510) that
   is ON (> 300 ADC) in ≥ 1 % (`OSC_MIN`) of the probe's events at ≥ 5× (`OSC_RATIO`) its live neighbours, and is not a copy
   of one already found (≤ 50 % of its ON events shared, `OSC_COPY`). Targeted: the oscillation moves between the
   neighbouring channels of the known oscillator (372 → 373).
3. **Coupled channels** (`coupled_channels`): for each oscillator (1, 2 or SEED_BAD) that actually oscillates in the
   plateau (ON rate > 10× its live neighbours', `COUPLE_OSC_RATIO`), a channel within ±4 ranks is coupled when, at 100 or
   300 ADC (`COUPLE_LIT_ADC`), it is lit in ≥ 10 ON events and its ON/OFF lit-rate ratio exceeds 10× (`COUPLE_RATIO`) the
   same ratio of all strips away from every oscillator (the beam's own ON/OFF modulation).
   - **Per event:** in an event where the oscillator is ON, its coupled channels are kept out of the seed and the
     own-centroid ref, blanked in the window and censored in the fit (`censor_x/censor_y` in the record → `missing` rows
     in `sps_reco.one`, bridged in `sps_cluster`). In the other events they are ordinary strips.
   - A real strip is lit whether or not the oscillator is on. A SEED_BAD channel that is quiet in the plateau (ch 510 in
     the beam at 250 mm) is ON only when a track crosses it, so it does not count as oscillating and its neighbours'
     charge sharing is not taken for coupling.

The meta of every re-run row records `hot_seed_channels`, `coupled_channels` ({sub-run: {oscillator: [channels]}}) and
`n_events_coupled_censored`; each event carries `x/y_n_coupled`.

Tests: `wft/tests/test_sps_coupled.py`, `wft/tests/test_sps_seed_bad.py` (stripe edge, partial oscillation).

## Open items / limitations

- **RAW run71 rows not re-run.** Their census (reading 3000+ RAW events twice from ~600 MB files) did not finish in the
  session. They keep `3ac0e79`: seed fix + blanking, with a 0.27 % residual at 156–160 mm (rerun/sps_audit).
  - To close it: `condor_submit code=~/x17rerun/code_c36322ec census.sub` with only the three run71 keys.
  - Then `make_stage_dag.py` and `condor/promote_rows.sh` for those that change.
- **The rules are thresholds.**
  - Near-oscillator rule: 1 % ON at ≥ 5× live neighbours, and since `48244778` at least 20 ON events
    (`OSC_MIN_EVENTS`). Without the floor, a 315-event telescope file (run53 group 003) flagged Y ch 371.
    - The census on `48244778` against the promoted products shows the floor changes no pass row. On all 68 ZS rows the
      event sets, refs, hot sets and coupled sets are identical (`results/census_48244778.csv`).
    - So `c36322ec` and `48244778` are the same rule on every pass row, and the ledger accepts both.
  - Seed rule in blocks: it newly flags X ch 149 in run61 d070/d140. That is one block's fluctuation of a strip in the
    232 mm band, probably real. It is censored there: unknown charge, no bias, a little less information on 131–142 X
    fits.
- **The H4 layer's `dead_stripes` does not list every low-gain X strip.** That is why the rate rules stay at 3 % in X and
  the weak-oscillator extension is restricted to the neighbourhood of ch 372/510.
- **Telescope tables remade on the final rules (`48244778`, `condor/tel.sub`, tables in
  `/eos/user/d/dneff/x17rerun/sps_junk/telG/`).** `tel_job.py` now uses the same probe as the pass: blocked live-neighbour
  seed rule, near oscillators, coupled channels. On run53 no channel is hot any more (X ch 173 was before); on run63 none
  was and none is.
  - Re-measured with `analyze_final.py`, giving `results/final_rule.json`.
  - **run63 flat700 is unchanged** (≤ 1 µm in every estimator and view): X p_mid 209 ± 11, Y p0 190 ± 10.
  - **run53 Y moves ≤ 5 µm**: hits 177 → **176 ± 10**, centroid 154 → 155 ± 7, p0 261 → 262 ± 12, p_mid 287 → 292 ± 17.
    Its X residuals shrink by 13–20 µm (p0, p_mid, centroid; hits unchanged). That is the edge of the beam spot coming
    back.
  - `rerun/sps_audit/README.md` (table §2) and `rerun/PAPER1_NUMBERS_2026-10-31.md` are updated to these numbers.
- **Hash.** The products' meta records `c36322ec`, the deployed archive (a pre-rebase commit). The SPS-chain files on
  `main` are identical to it.

## Files

| file | what |
|---|---|
| `probe.py`, `look_probe.py`, `rule_on_probe.py` | the Y-region probe of a row and the coupling diagnostics |
| `lobes.py`, `look_lobes.py`, `rule_on_lobes.py` | per-event peaks + lobe counts of all channels; the rules on them |
| `census.py`, `census_table.py` | which rows change, without fitting (event set, hot set, refs, censored events, junk zone) |
| `make_stage_dag.py`, `compare.py` | the staged re-run of the changed rows; staged vs pass per row |
| `condor/` | census, lobes, promote_rows.sh |
| `results/` | census and compare tables |
