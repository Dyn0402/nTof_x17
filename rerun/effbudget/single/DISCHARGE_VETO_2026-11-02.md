# A discharge-aware spark veto: design, validation, and why it ships OFF (2026-11-02)

**Verdict.** A veto that re-seeds a vetoed event from its clean parts (`wft.seed.SPARK_RESCUE`) is implemented, tested
and validated on the six bench golden rows. **It ships switched off.** On every chamber, the tracks it recovers are far
less often right than ordinary tracks are:

- right against M3 (≤ 5 mm): det3 75 % and 59 % (two rows), det2 57 %, det6 26 %, det7 17 %, det4 11 %;
- ordinary events on the same rows: 84–96 %.

Without the rescue gate the fitter does find the muon in ~19 % of det6 discharges (the GD "veto off" result), but it
finds something else 2.5 times as often.

- The one class that works is a **clean muon plus a separate deposit** ("muon+other") on det3/det2: 29 right, 4 wrong
  over the three rows.
- **Discharges and δ-ray showers** are where the wrong tracks come from.

Ordinary events are bit-identical with the switch on or off (438 events, every column). n_TOF and SPS never run this
code. The all-or-nothing veto (`SPARK_VETO_HITS = 50`) stays the algorithm, and nothing needs re-running.

## 1 · Design (`wft/seed.py`, `wft/reco.py`)

An event over `SPARK_VETO_HITS` is no longer dropped at seeding. In `reco._stream_windows`, after the band re-seed,
the event goes through the rescue:

1. **Ringing chips** (`seed.ringing_chips`, `SPARK_CHIP_CM_SIG`). A chip rings when the raw common mode of its 64
   channels swings by more than k × its median noise. A ringing chip loses its hits, and its strips are censored in the
   fit like dead strips (the window's `missing` mask). The default is off: see §3.
2. **Floor re-applied** on what remains, so the discharge no longer sets the 10 % scale.
3. **Cluster cap** (`SPARK_CLUSTER_MAX` = 40 strips). A wider cluster is a deposit, not a track, and is not offered to
   the fit. `n_dropped` counts only the competing track-sized clusters, so the multiplicity that matters is the
   footprint cluster's, not the plane's.
4. **Hard veto** if no candidate survives in either view.
5. **Rail censoring** (`SPARK_RAIL_CENSOR`). In rescued windows, every sample whose raw ADC is on a rail (≥ 4000 or
   ≤ 5) is censored. The fit's own saturation test reads the CM-subtracted sample, which a chip-wide swing pulls under
   `sat_adc`. 30 % / 42 % of chips in vetoed events have a high / low rail strip, against 5 % / 6 % in ordinary events.
6. **The rescue gate** (`reco.spark_rescue_gate`, `SPARK_RESCUE_LQ_NSIG` = 5): one particle in both views.
   - Both views fitted and `quality_ok`, with the χ²/dof medians of the *ordinary* events only, so ordinary flags
     cannot move.
   - Neither view isochronous.
   - |tc| ≤ the bundle's `tc_tol`.
   - |ln qx/qy − median| ≤ 5 × the class width of `xy_pairing['lq_class']`.

   Accepted events carry `spark = True` and `spark_rescued = True`. Every attempt, accepted or not, is written to
   `events.spark_rescue.parquet` with its gate columns. Rejected events leave the events and candidates tables (the
   hard veto).
7. **The meta** records `selection.spark_rescue` (switches and counts).

## 2 · Validation method

- **Event sets** (`dveto_events.py`, staged rr1 inputs):
  - `veto_m3`: every vetoed trigger with one M3 recipe ray (budget rays), sampled to 500–1000 on det4/det6/det7 and the
    det3 weekend row;
  - `veto_nom3`: vetoed triggers with no M3 ray of any quality;
  - `ord_m3`, `ord_nom3`: ordinary references.
- **Re-reconstruction** (`dveto_rereco.py`): the full production chain with the pass-1 bundles, `SPARK_RESCUE` on and
  off through `WFT_DIAG`.
- **Scoring** (`dveto_score.py`, `dveto_gate.py`, `dveto_summary.py` → `results/discharge_veto.json`):
  - real = winner within 5 mm of the M3 ray at the mesh (raw frame, as budget `r`);
  - fake = farther, or any track in a trigger with no M3 ray.

  The gate and its pieces are judged offline on the sidecar.
- **Classes:** the A1 waveform classes (`sparks_<det>.parquet`), on interior rays only.

**Truth caveat.** Triggers with no M3 ray are not empty. On ordinary M3-less triggers the reconstruction finds a track
in 87 % (det3) and 36 % (det6) of cases. M3-less "fakes" are an upper bound, and the M3-tagged purity is the number
that decides.

## 3 · Results

Rescue attempts on the M3-tagged vetoed sample (real / fake against M3):

| row | vetoed M3 (sample/row) | pass 1 already (band re-seed) | no gate | **rescue gate** | gate + `n_dropped` ≤ 2 | gate purity | ordinary purity |
|---|---|---|---|---|---|---|---|
| det3 sat | 220/220 | 8 | 34 / 108 | **9 / 3** | 0 / 2 | 75 % | 95.4 % |
| det3 weekend | 500/758 | 11 | 76 / 286 | **22 / 15** | 1 / 3 | 59 % | 95.2 % |
| det2 | 146/146 | 3 | 24 / 83 | **8 / 6** | 1 / 1 | 57 % | 94.9 % |
| det4 | 500/1208 | 3 | 30 / 364 | **17 / 131** | 0 / 1 | 11 % | 83.5 % |
| det6 | 1000/2072 | 24 | 189 / 481 | **37 / 103** | 20 / 26 | 26 % | 95.7 % |
| det7 | 500/3161 | 37 | 36 / 255 | **8 / 40** | 1 / 2 | 17 % | 90.1 % |

Ordinary purity: interior, under the veto, both views fitted, full pass-1 budget rays.

**By class** (interior, gate, real / fake):

| row | discharge | muon+other | shower |
|---|---|---|---|
| det3 sat | 2 / 0 | 7 / 1 | 0 / 0 |
| det3 weekend | 2 / 1 | 18 / 3 | 1 / 6 |
| det2 | 3 / 0 | 4 / 0 | 0 / 0 |
| det4 | 3 / 18 | 12 / 19 | 1 / 65 |
| det6 | 30 / 37 | 2 / 0 | 0 / 10 |
| det7 | 1 / 7 | 6 / 3 | 0 / 2 |

**M3-less vetoed triggers with an accepted rescue:**

| | det3 sat | det3 weekend | det2 | det4 | det6 | det7 |
|---|---|---|---|---|---|---|
| accepted / triggers | 30 / 230 | 13 / 200 | 52 / 300 | 64 / 200 | 68 / 300 | 45 / 200 |

On ordinary M3-less triggers the track rate is 87 % (det3) and 36 % (det6).

**Scaled to the whole row** (interior, gate): what the rescue would add, in points of interior tracks.

| row | recovered | fake tracks added |
|---|---|---|
| det3 sat | +0.14 pt (9) | 1 |
| det3 weekend | +0.16 pt | ~23 |
| det2 | +0.21 pt (7) | 0 |
| det4 | +0.34 pt | ~250 (2.1 pt) |
| det6 | +0.8 pt | ~100 (1.2 pt) |
| det7 | +0.5 pt | ~75 |

The ungated det6 figure (+4.3 pt recovered) agrees with the GD scan's veto-off +3.7 pt.

**Quality of the recovered tracks** (gate, real, s68 per view against M3):

| | residual [mm] | \|Δtan\| | ordinary residual [mm] | ordinary \|Δtan\| |
|---|---|---|---|---|
| det3 | 0.35–1.2 | 0.015–0.025 | 0.5–0.6 | 0.019 |
| det2 | 0.4–0.7 | 0.015–0.03 | | |
| det4 | 1.1 | 0.03–0.04 | | |
| det6 | 0.7–1.0 | 0.06–0.16 | 0.5 | 0.025 |
| det7 | 0.6–1.6 | 0.02–0.16 | | |

On det6/det7 the angles of recovered discharge tracks are 2–6× worse than ordinary ones.

**Ringing-chip mask.** On det6, 97 % of vetoed M3 events ring on all 8 chips of a view, and the M3 footprint's chip
rings in 92 %. The discharge starts within ±1 sample of the muon's own first signal: "the chip-wide blocks start at the
muon's crossing", as in the A1 audit. On det3, 103 of 220 vetoed M3 events ring on all 16 chips. A chip mask therefore
removes the muon with the discharge. The hits-only estimate (`explore`-stage, CM > 20σ):

| | det6 interior | det3 interior |
|---|---|---|
| footprint candidate in both views, chip mask on | 2 / 1 576 | 36 / 113 |
| the same, no mask | 944 / 1 576 | 70 / 113 |

The mask stays off, and no full chip-mask re-reconstruction was run.

**Bit-identity.** With `SPARK_RESCUE` on and off on the same event set (sat_det3, 438 ordinary events), all 61 columns
are identical. Against pass 1 the fits differ only at float noise (≤ 5e-12 mm in p0). The differing `quality_ok` flags
(3–4) come from table-level medians over a subset table, not from the rescue. The gate takes its medians from ordinary
rows only, so rescued events cannot move an ordinary flag.

## 4 · Why the gate cannot be rescued by tuning (what was tried)

Varying the gate offline on the sidecar:

- Quality and the charge balance are the strongest pieces. Dropping either doubles the fakes.
- Adding the production `n_dropped ≤ 2` rule improves det6 to 20 / 26, still 43 %. It kills the det3 muon+other
  recoveries, which by construction carry a second cluster.
- Tighter lq (3σ): det6 31 / 46.

The best det6 purity found is ~45–50 %, against 96 % ordinary. In M3-tagged discharge events the fakes are mostly fits
to the discharge's own structure inside the ringing chips. They are coincident and charge-balanced in x and y because
the discharge is too.

## 5 · n_TOF

The bench veto is not in the n_TOF chain. `ntof_tracking.wft_beam` drops a **plane** over 150 floored hits and keeps
the other view, plus a wide-isochronous cluster veto. On run_145 stat090_0000 (all 7 tags, `dveto_ntof.py`):

| arm | triggers | bench rule (> 50 x+y) would drop | busy-plane rule drops (any plane) | both planes busy | planes > 50 that are several separate clusters |
|---|---|---|---|---|---|
| A | 34 149 | 17.3 % | 1.9 % | 1.6 % | 83 % |
| B | 36 836 | 28.6 % | 4.2 % | 1.2 % | 91 % |
| C | 32 807 | 32.2 % | 2.7 % | 2.0 % | 92 % |
| D | 57 734 | 39.3 % | 1.1 % | 1.1 % | 92 % |

Busy beam planes are multi-particle, not discharges. The bench rule would be wrong there, and the beam rule is already
per plane. **n_TOF stays unchanged.** The rescue is a bench-driver path, and the n_TOF and SPS drivers do not call it.
This is the dataset-level switch, as GD found ("SPS: no veto needed").

## 6 · What needs re-running

Nothing. `SPARK_RESCUE = False` is bit-identical to pass 1's veto. If the switch is ever turned on:

- the bench rows need re-reconstruction;
- downstream code that classifies 'spark' from the hits count must instead read the table's `spark_rescued`
  (`budget.py`, `02_efficiency.py`, `paper1_res`, `dp_score`, `spark_timeline`, `peraxis_deconvolve`).

## 7 · For Paper I

> Events with more than 50 strips above the relative floor are rejected before the fit. We tested a discharge-aware
> alternative: the event is re-seeded without track-incompatible clusters (> 40 strips), with railed samples censored,
> and a recovered track is accepted only if both views are fitted with an ordinary χ²/dof and agree in column time and
> charge. Tested against the M3 reference, it recovers a clean muon accompanied by a separate deposit in most such
> events (det3/det2: 29 of 33 accepted tracks correct). In discharges and δ-ray showers, the accepted tracks are mostly
> not the muon: 11–26 % correct on det4/det6/det7, against 84–96 % for ordinary events. The discharge typically begins
> at the muon's own crossing and spans every chip of the plane, so the muon's charge is not separable from it. We
> therefore keep the event-level veto and count vetoed muons as inefficiency: a discharge is a chamber loss, and a
> vetoed muon + deposit is a two-particle event lost at this multiplicity.

## Files

- **Code:** `wft/seed.py` (`SPARK_RESCUE`, `SPARK_CLUSTER_MAX`, `SPARK_CHIP_CM_SIG`, `SPARK_RAIL_*`, `ringing_chips`,
  `spark_rescue_seeds`) and `wft/reco.py` (`spark_rescue`, `spark_censor`, `spark_rescue_gate`,
  `SPARK_RESCUE_LQ_NSIG`; `reconstruct_run`, `_stream_windows`).
- **Tests:** `wft/tests/test_spark_rescue.py` (8).
- **Parameters:** `docs/PARAMETERS.md` §5.
- **Study scripts:** `dveto_events.py`, `dveto_rereco.py`, `dveto_score.py`, `dveto_gate.py`, `dveto_summary.py`,
  `dveto_ntof.py`.
- **Numbers:** `results/discharge_veto.json`, `results/discharge_veto_ntof.json`.

## What this does not rule out

- A gate built from different information (a discharge model subtracted per chip, the drift-column shape before the
  onset, a learned classifier) might separate muon from discharge. None was found here.
- A **muon+other-only** rescue looks sound on det3/det2 (+0.1–0.2 pt each). It needs a classifier that does not use
  M3 to tell muon+other from showers; on det4 the same class is 12 / 19.
- The samples are 150–1000 events per set. The purity figures have ±5–10 pt statistical errors, which does not change
  the ordering.
- M3 itself is degraded in sparky events, so some "fakes" may be right tracks with a bad reference. This was not
  quantified, and the det6 angle widths argue against it being large.
