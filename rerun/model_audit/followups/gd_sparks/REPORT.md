# GD — Spark detection after the FE unification (round 3, item 7, 2026-10-28)

## Verdict: keep the veto as it is (event dropped at > 50 seeding hits) — det4 confirms (2026-10-29)

**det4 (scored 10-29, 11372 interior rays, detection 40.8 %):** no veto +0.40 pt detected but +1.67 pt wrong fits;
> 100 hits +0.40 / +1.46; > 40 −0.17 / −0.36. Same shape as det2/3/6/7: 50 is on the best curve, the verdict stands.
**The criterion.** `SPARK_VETO_HITS = 50` counts floored seeding hits over x + y, after the relative floor, the flat-hit
drop and the band re-seed.

**It sits on the best curve on every finished chamber (det2/3/6/7).** It was scanned against:
- the busiest plane's count, and both planes required;
- strips on the ADC rail, saturated hits, strips above SNR 6;
- the common-mode step and negative ringing;
- combinations of these.

Every waveform discharge signature does worse at the same purity.

**Above 50 hits, few accepted fits are right.** Only 0–25 % agree with M3 within 5 mm and 5° in both views. Loosening the
veto buys about 1 good track per 2–6 wrong ones.

**Tightening to 30–40 cuts wrong fits by 30–45 % at a cost in detection:**

| chamber | wrong fits, 50 → 30 hits | detection lost (5 mm) |
|---|---|---|
| det7 | 1196 → 822 | 5.1 pt |
| det6 | 402 → 221 | 2.6 pt |
| det2 | 141 → 109 | 0.5 pt |
| det3 | 244 → 181 | 0.1 pt |

40 is defensible if purity is the priority. 50 is kept as the efficiency-side choice; the result is not sensitive near it.

**The band re-seed of vetoed events (FE risk item) is validated.** The events it brings back are mostly good:

| chamber | events back | good | wrong | not accepted |
|---|---|---|---|---|
| det7 | 273 | 103 | 64 | 106 |
| det6 | 73 | 28 | 17 | 28 |
| det3 | 9 | 6 | 0 | 3 |
| det2 | 4 | 1 | 1 | 2 |

This beats any band of the hit count above 30.

**Bug fixed (`wft/seed.py`).** `seeds_from_hits` bound `SIG_REL_FLOOR`, `SPARK_VETO_HITS` and `N_CANDIDATES` as defaults at
import, so a `WFT_DIAG` override was recorded in the metadata but never applied. They are now read at call time.
Production is bit-identical: same defaults, and no call site passes them. Tests: `wft/tests/test_spark_veto.py`; suite
110/110.

## 1. Criteria in the chain
| stage | criterion | bench | SPS | n_TOF |
|---|---|---|---|---|
| decode | incomplete windows skipped | yes | yes | yes |
| hits | no event veto (saturated flag, block CM, matched-filter gate) | yes | yes | yes |
| seeding | > 50 floored hits x + y → event not fitted; flat hits dropped first | yes | — | — |
| band re-seed | vetoed events re-counted without band strips (windows ≥ 1800 ns) | yes | — | off |
| wft_beam | > 150 hits on a plane → that plane dropped, other view kept; wide-isochronous cluster veto | — | — | yes |
| SPS reco | none (self-seeded, peak ≥ 300 ADC) | — | yes | — |
| χ²/dof | quality flag only (`quality_flags`) | yes | yes | yes |

## 2. Bench: threshold scan
**Method** (`lx_sparks.py` on lxplus with the fe code, fe bundles, rr1_cm0 hits, decoded_root_fa80c06):
- every M3-matched event of the fe gate range with > 30 hits is refitted with the veto off;
- identity with the fe table is checked (Δ = 0), and the fe detection is reproduced exactly;
- scoring is against the interior M3 rays.

**What the 50-hit veto drops, by waveform class.** Discharge = > 20 strips on the rail, > 150 strips above SNR 6, or
> 100 strips below −10σ.

| chamber | vetoed | discharge, lost | discharge, found without the veto | no discharge, found | no discharge, wrong / competing clusters |
|---|---|---|---|---|---|
| det2 | 83 | 38 | 12 | 1 | 32 |
| det3 | 104 | 33 | 2 | 3 | 66 |
| det6 | 1304 | 966 | 275 | 7 | 56 |
| det7 | 1983 | 1425 | 361 | 23 | 174 |

**Scan** (detected % / good % / wrong among accepted %):

| chamber | 30 | 40 | **50** | 60 | 80 | off |
|---|---|---|---|---|---|---|
| det2 | 89.92/88.65/3.50 | 90.18/88.80/4.02 | **90.45/88.89/4.47** | 90.62/88.94/4.80 | 90.74/88.94/5.27 | 90.83/88.97/5.63 |
| det3 | 92.58/91.49/2.98 | 92.64/91.51/3.61 | **92.68/91.52/3.97** | 92.73/91.52/4.17 | 92.76/91.54/4.43 | 92.76/91.54/4.56 |
| det6 | 72.47/70.48/3.99 | 73.62/71.29/5.13 | **75.09/72.02/6.88** | 76.95/72.92/9.36 | 78.43/73.60/12.77 | 78.83/73.78/15.64 |
| det7 | 57.54/51.52/15.66 | 60.50/53.05/18.43 | **62.68/54.20/20.43** | 64.30/55.03/22.14 | 66.26/55.82/25.23 | 67.15/56.17/28.41 |

- **Veto off:** det6 +3.7 pt detected but only +1.8 pt good, wrong fits 402 → 1033. det7 +4.5 / +2.0 pt, wrong 1196 → 1916.
  Angle s68 gets worse (det7 X 1.81 → 1.99°).
- **Alternatives** (`results/roc_<det>.csv`, `figures/roc.png`): all sit on or below the hit-count curve.
  - The busiest plane's count is equivalent; requiring both planes is worse.
  - Rail > 5 strips costs 15–19 pt on det6/det7.
  - Hits > 50 AND a discharge signature: ≤ +0.27 pt for +0.5 pp wrong.
- **χ²/dof does not rescue events above 50 hits:** AUC 0.49–0.67.
- **A threshold scaled per chamber** (by median track size) is the same as a fixed one.

## 3. SPS and n_TOF
- **SPS det4:** 0.04 % of events exceed 50 hits. Those are beam pile-up (multi-cluster), not discharges. No veto is needed.
- **n_TOF run_79 tag 000** (beam seeder's 150-hit busy-plane rule):
  - busy planes are 0.8–4.5 % of events per arm and view;
  - 53–100 % of them are one contiguous region over 150 strips or railed, i.e. flash or discharge;
  - multi-particle-like busy planes are 0–0.94 % of events;
  - only the busy plane is dropped; the other view is kept;
  - **not established:** whether the multi-particle-like planes hold fittable tracks (not refitted).

## Limitations
- **det4**: done 10-29 (see the verdict).
- **Good/wrong is judged against M3**, which is itself degraded in sparky events (cross-talk).
- **Waveform criteria** are scored on the candidates plus 400 random controls.
- **n_TOF:** one file tag, no refit. **SPS:** one run.

## Files
- **Code** (`rerun/model_audit/followups/gd_sparks/`): `lx_sparks.py`, `gd_score.py`, `gd_figs.py`, `ntof_busy.py`,
  `pull.sh`.
- **Results:** `results/{roc_<det>.csv,score.json,strict_scan.txt}`.
- **Outputs:** EOS `/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe/<row>/wft/gd_sparks/`. Local copy and figures:
  `~/x17/cosmic_bench/model_audit/gd_sparks/`.
- **DAG on lxplus:** `~/x17rerun/gd/`.
