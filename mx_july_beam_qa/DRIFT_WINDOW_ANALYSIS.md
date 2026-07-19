# Drift window & drift voltage — ANALYSIS & RECOMMENDATION (2026-07-19)

Answers `DRIFT_WINDOW_HANDOFF.md`. Data: run_54 cosmics (2026-07-18, all 4 subruns)
**and run_55 beam** (2026-07-18 19:11, neutron beam ON, 3He, Mode-3 scint-doubles
trigger, r560 subruns c00/c01/c02) — both at drift A=600 V, B/C/D=800 V, resist 560 V,
32 smp × 60 ns, latency 35. Extractor: `24_drift_time_edges.py` (per-64ch-connector
common mode, robust MAD thresholds, hot-channel mask, spatial clusters ≥3 strips,
**monster-event cut** at >100 hit strips). Result JSONs: `daq_lxplus:/tmp/v5_*.json`,
`/tmp/v6_*.json` (v6 adds per-event tuples; box was unreachable when this was written —
numbers below are the v5 pass, printed in `/tmp/v5_cosmics.log`, `/tmp/v5_beam.log`).

## TL;DR recommendation

| knob | setting | why |
|---|---|---|
| det A drift | **700 V** (staged via 650) | not for speed today (A is dry, gains 0.7 smp) — for **wet-excursion immunity**: at 0.8 % H2O (= what B/C/D breathe *right now*) A@600 V needs 22.6 smp and truncates any window ≤30; A@700 V needs 17.7 and fits n=28 |
| B/C/D drift | 800 V (keep) | already fastest allowed; wetter gas needs the field |
| latency | **35 → 32** | prompt peak measured at sample 8–9 (identical in cosmic AND beam triggers, all 4 dets); 32 moves it to ~5–6, keeping 3–4 pre-signal baseline samples |
| n_samples | **32 → 28** | covers worst measured deep edge + pulse fall + a ~3-smp wet-excursion margin; saves 12.5 % event size / readout |
| aggressive floor | 26 (not lower) | eats the excursion margin; go only with per-run span monitoring. **16–20 samples is physically impossible** while B/C/D drift in 0.8 % H2O (their full-gap column alone is ~15 smp) |
| beam drift scan | **yes, small one** (A: 600→650→700, ~30 min each) | the real gate for 700 V is spark stability under beam, and span(600)/span(700) on det A pins the water model independent of thresholds; B/C/D stay untouched |

The single highest-leverage action is not DAQ at all: **dry the gas line**. If B/C/D
reached det A's dryness, every detector's column would be ~11–12 smp and
latency 32 / n_samples 22–24 would be safe. A dryer/higher flow is worth more than
any trimming below 28.

## 1. What the de-noised data actually shows

### 1a. The handoff's "late/slow B & D" was mostly two artifacts

1. **Flash/monster events** (~2 % of beam triggers, 129–152 of 9000 cosmic events on
   det D = sparks): 400+ strips saturated at ~4100 ADC, followed by **fixed channel
   blocks** (e.g. FEU2 ch 38–63 and 448–472 — one DREAM chip end + one full chip)
   ringing at samples 24–31. These blocks masqueraded as 25-strip "track clusters"
   peaking late. Waveform dumps confirm: every "late cluster" inspected was one of
   these blocks after a saturated event.
2. **Coherent noise on the y-planes / det B** in the cosmic run (beam run_55 B_x is
   clean and matches C/D). After the monster cut, det D_x ceiling fell 8.9 % → 1.5 %.

**No detector is currently losing drifting primaries** at 32 smp / latency 35:
cleaned x-plane ceilings (peak ≥ 31) are A 0.3 %, C 0.4 %, D 1.5 % (cosmics; beam
consistent). The remaining y-plane/B ceiling occupancy is ringing, not physics — a
trimmed window *cuts* it. (Electronics follow-up on the ringing blocks is separate.)

### 1b. Timing is common to all four detectors and both trigger modes

- First recorded signal (pulse rise above threshold): **sample 6–7** (p5).
- Prompt (near-mesh) pulse **peak: sample 8–9**, i.e. peak ≈ latency − 26.
  The handoff's "latency − 24" and the suspected per-detector t0 offsets (B~19, C~15)
  were artifacts of near-vertical cosmics (single broad strip peaks mid-column) and
  of the ringing clusters. Beam Mode-3 and cosmic-singles triggers give the *same*
  prompt sample → one latency serves both run types.

### 1c. Full-gap drift time and per-detector water content

Best estimator: span of peak samples across clusters with ≥6 strips (inclined
tracks), cosmics (best stats; beam agrees within thin statistics):

| det (plane) | V_drift | span p95 [smp] | T_full (span/0.89) | v [µm/ns] | H2O (Magboltz grid) |
|---|---:|---:|---:|---:|---|
| A (x) | 600 | 11 | 12.4 | ≈ 40 | **≈ 0 (dry, ≤0.2 %)** |
| C (x) | 800 | 13.5 | 15.2 | 33.0 | **≈ 0.85 %** |
| D (x) | 800 | 13.0 | 14.6 | 34.2 | **≈ 0.80 %** |
| B (x) | 800 | (18 cosmic — noise-inflated; beam ≈ C/D) | ~15 | ~33 | ~0.8 % |

(0.89 = recorded-column fraction anchored on det A = clean-Magboltz; June-style
threshold/attachment losses at the column ends. Deep-charge amplitude falls ×~0.35
over 24 mm → λ_eff ≈ 23–25 mm, milder than June's 16–18 mm.)

**The daisy-chain gradient is real but saturates immediately**: A (first, dry) →
B ≈ C ≈ D ≈ 0.8–0.9 % H2O, near June's ~1 % fleet equilibrium. The gas wets up in
the first hop; detectors B–D all pay ~+30 % drift time at 800 V.

## 2. Window budget (why latency 32 / n_samples 28)

All in 60 ns samples; prompt peak P moves with latency (P ≈ latency − 26.5).
Signal end ≈ P + T_full + 2 (CR-RC² fall to half). At **latency 32 → P ≈ 5.5**,
first-sig ≈ 3.5, leaving 3 clean pre-signal baseline samples:

| scenario | T_full | signal ends | fits n=28 (last=27)? | n=26? |
|---|---:|---:|---|---|
| A @600 dry (today) | 12.4 | 19.9 | yes | yes |
| A @700 dry | 11.7 | 19.2 | yes | yes |
| B/C/D @800, 0.8 % (today) | 14.6–15.2 | 22.1–22.7 | yes (+4 margin) | yes (+2) |
| B/C/D @800, excursion to 1.2 % | 20.3 | 27.8 | marginal (−1) | no |
| **A @600, wet to 0.8 %** | **22.6** | **30.1** | **no** | no |
| A @700, wet to 0.8 % | 17.7 | 25.2 | yes | no |

- **n_samples 28, latency 32**: covers today's worst measured deep edge
  (cosmic latest-peak p95 ≈ 25–27 at latency 35 → 22–24 at 32) with the same or
  better margin than the current 32/35 config, and survives a moderate wet excursion.
  Saves 12.5 % readout/size.
- **n_samples 26, latency 31–32**: fine for today's gas, zero excursion headroom.
  Only with per-run span monitoring (run the extractor on each new subrun).
- Below 24: not until the gas line is dried — the B/C/D column alone is ~15 smp.
- Keeping latency 35 makes trimming pointless (6 dead pre-roll samples): any
  n_samples cut comes straight out of the tail margin.

## 3. Det A drift voltage: 600 → 700 V

- **For**: (i) removes the one catastrophic corner — A@600 in gas as wet as its
  neighbors' truncates *any* affordable window (row 5 above; exactly the June 500 V
  failure mode); (ii) A is first in the gas chain so it *should* stay dry, but a flow
  drop / line disturbance / stack reshuffle would put it at 0.8 % overnight;
  (iii) small bonuses: −0.7 smp column, less attachment time, more margin from the
  rising v(E) branch.
- **Against**: sparked at 800 V (600 V stable at 0.14 µA). 700 V is untested — hence
  staged soak: one subrun at 650 watching `hv_monitor.csv` (trips/current), then 700.
- Physics cost ≈ nil (micro-TPC z-sampling changes by <6 %).

## 4. Proposed beam drift scan (do it, it's cheap)

During normal Mode-3 beam running, det A only (B/C/D untouched → their physics
continues): 600 V (reference exists) → 650 → 700, ≥30 min each (≈2000 triggers →
~100–200 clean A-tracks/plane/point). Read out with `24_drift_time_edges.py`.
Deliverables: (a) spark/current stability at 650/700 = the actual gate for §3;
(b) span(600)/span(700) ratio on the same gas → water content on A without the
recorded-fraction assumption (dry ratio 1.06, 0.8 %-wet ratio 1.28 — cleanly
separable even at ±1 smp); (c) confirms the predicted ~19-sample end point before
freezing latency 32 / n=28. A B/C/D point at 750 V would measure the wet-gas slope
but costs their margin while it runs — optional, low priority.

## 5. Verification still open (nice-to-have, not blocking)

- v6 per-event pass (adds prompt-anchored cluster cut: a through-going column must
  contain a prompt strip; kills residual ringing percentiles) was running on
  `daq_lxplus:/tmp` when the box dropped off the network — JSONs land as
  `/tmp/v6_{cosmics_run54,beam_run55}.json`. Expected effect: C_y/D_y/B ceilings
  collapse toward the x-plane values; no change to the recommendation.
- run_53 (beam, 03:11 same day) r560 subruns as a 16 h gas-stability cross-check.
- Electronics: locate/fix the post-saturation ringing blocks (per-FEU fixed channel
  ranges, e.g. FEU2 38–63 & 448–472); until then the monster cut + window trim
  handle it offline.
