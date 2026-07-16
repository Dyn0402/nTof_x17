# Investigation brief — det3 "near-miss" (reco_far) tail, on the current v2 basis

**For:** a follow-up session picking up the detector-3 reco_far study.
**Written:** 2026-07-13. **Owner asked for this explicitly.**

---
## ✅ DONE (2026-07-13) — see `report_v2.pdf` (supersedes `main.pdf`)

The owner's three questions are answered on the v2 basis. New code:
`edge_chi2_extract.py [KEY]` → `edge_chi2_plots.py [KEY]`
(caches `edge_chi2_data_<KEY>.npz`, numbers `edge_chi2_meta_<KEY>.json`).
Verified on `g_det3_wknd` (headline) **and** `sat_det3` (490V/1000V) — story transfers.

1. **Reference tracker's fault? YES, ~half.** |r| rises monotonically with M3 track
   Chi2 *even inside the χ²<5 cut*; top-decile M3 χ² misses >2mm 50% vs 18%. Of >2mm
   misses: **49% bad-M3-only** (χ²ₘₐₓ≥1), 12% discharge-only (mult≥25), 8% both,
   32% ordinary resolution. Chamber's **intrinsic** within-5mm accuracy is **98.5%**
   (clean reference + no discharge) vs 93.2% raw. → `fig1_...`, `fig2_...`
2. **Edge correlation? NO (in v2).** >2mm and >5mm miss rates are FLAT vs distance to
   nearest active edge. The prior "low-X/left-edge" concentration was a pre-v2 (χ²<20)
   discharge artifact; v2 removed it. → `fig3_...d`
3. **Passivated ~2cm Y strip: measured & already handled.** In detector-local frame,
   efficiency is dead for Y<18mm and Y>380mm (**18.0 / 18.7 mm passivated**, ~1.8cm each,
   Y only; X has sharp geometric edges at 0/399, no passivation). Edges *identical* on
   both runs → physical, not alignment. The 09 percentile box already self-clips to the
   active Y (reco only happens there), so **88.8% is a true active-area efficiency, NOT
   diluted** by passivation. Nominal 40×40cm would wrongly read ~80%. Measured active
   area **≈36.2×40.4 cm**. Recommend documenting this explicitly in 09 / engineer pkg.
   → `fig3_...a-c`

**Follow-on finding (owner-requested):** because the miss is reference-limited, scanned the
M3 χ² cut (`m3_cut_scan.py`, `fig4_...`) — det3 core σ falls 0.63→0.46 mm with NO plateau
inside χ²<5, so the reference tracker limits us across the whole band; intrinsic σ ≲0.3–0.45 mm.
**Recommended reprocessing cut: χ²<1.0 + NClus=4** (σ→0.47, eff 88.8→91.5 %, reco_far 6.5→4.0 %,
43 % stats). True active area recorded permanently in `common/mx17_active_area.py`
(X∈[0,398.6], Y∈[18,380] mm). **Two action items for another instance** — reprocess everything
on the stricter cut, and draw 40×40 + true-active outlines on all 2-D plots: see
`M3_CUT_AND_ACTIVE_AREA_NOTE.md`.

**Still open (from the list below, not done):** waveform-level discharge confirmation
(Q2 original) and the tighter-veto trade-off (Q3 original) — lower priority now that v2
shows the tail is half reference-tracker and spatially uniform.

---

## What reco_far is

In the efficiency breakdown (`09_efficiency_breakdown.py`), every M3 reference muon
crossing the active area is put in one bucket. `reco_far` = the detector **did**
reconstruct a valid X+Y point, but it landed **> 5 mm** from the telescope track.
It is *not* a detection failure (the chamber saw the muon) and it is *not* a spark
(> 50 strips is pulled into `spark` first). It is a **position** miss. On det3 it is
the single largest "loss" off the 88.8 % efficiency, so it is worth understanding
what it actually is and whether any of it is recoverable.

## Current numbers (M3 v2, headline run) — the target of this study

Run `g_det3_wknd` = `mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check`,
from `.../mx17_3/efficiency/efficiency_breakdown.txt`:

| bucket | % of active-area crossings |
|---|---|
| reco_near (≤5 mm) | 88.8 |
| **reco_far (>5 mm)** | **6.5** |
| spark (>50 strips) | 4.4 |
| hit_no_reco | 0.1 |
| no_hit (silent) | 0.2 |

Of reconstructed hits, 93.2 % are within 5 mm; core σ(|r|<15) = 0.63 mm, median |r| = 1.01 mm.

## What already exists here (READ BEFORE STARTING — but note it is STALE)

`det3_recofar_analysis/` (this directory): a committed 4-page report (`main.pdf`),
`extract_recofar.py` → `make_plots.py`, `recofar_data.npz`, `recofar_meta.json`.
Prior conclusions (still qualitatively believable):
- reco_far is **two overlapping populations**: (i) a benign **5–10 mm near-miss
  shoulder** (~40 % of the tail) = the resolution/angle tail spilling just past the
  5 mm cut; (ii) a genuinely-wrong **>20 mm tail** (~38 %) from **elevated-multiplicity
  "sub-veto" discharges** (20–50 strips — *below* the 50-strip spark cut, long
  multi-pulse clusters) that plant a displaced competing cluster the micro-TPC fit
  sometimes picks. Concentrated on the **low-X / left edge**.
- Ruled out: multi-ray matching (multiray = 0 %), single-plane readout fault (both
  planes contribute).

### ⚠️ Why it must be re-run, not trusted as-is
The prior pass used **`chi2_cut=20.0`** (see `extract_recofar.py:47`), i.e. the
**pre-v2 ray recipe**, and predates the file-000/003 recovery. Its headline was
**reco_far = 13.7 %** and **no_hit = 2.4 %** (`recofar_meta.json`: n_reco_far 7271,
n_no_hit 1296 of 52995). On the current **M3 v2** recipe (`chi2 < 5`, NClus ≥ 3) plus
the recovered files, the same run is **6.5 % / 0.2 %** — v2 alone roughly **halved**
reco_far, which is exactly the benign near-miss shoulder tightening up (v2 also moved
within-10 mm match 85.6 → 95.6 %). **So the prior magnitudes are superseded; redo the
whole characterisation on v2 rays before drawing any number.** The interesting question
is what the ~6.5 % that *survives* v2 is made of.

## The actual questions to answer (on v2)

1. **Re-derive the split on v2.** Of the surviving 6.5 %, how much is 5–10 mm
   near-miss (recoverable — the efficiency is ~95 % at a 10 mm match), 10–20 mm, and
   >20 mm genuine tail? Is the >20 mm tail still elevated-multiplicity and still
   low-X/left-edge, or did v2 clean that too?
2. **Waveform-level confirmation of the discharge hypothesis** (prior follow-up #3,
   never done). Go to the raw DREAM waveforms and check that the >20 mm events carry a
   **discharge / afterpulse** signature (20–50 strips, long duration, multi-pulse)
   rather than a clean muon cluster. Reuse the machinery in `40_spark_waveforms.py`
   (per-64ch-chip common-noise subtraction, saturation/onset features) — it already
   knows how to read `decoded_root` and separate genuine localised charge from common
   mode. `quality` fields in the old meta already show the tail's `durx_far_mean`
   ≈ 3888 ns vs near ≈ 581 ns — a long-duration signature to confirm at the sample level.
3. **Does a tighter discharge veto recover efficiency, and at what cost?** (prior
   follow-up #1). Try a ~30–40-strip cut, or a cluster-duration / multi-pulse veto,
   and measure how much of reco_far moves into a `spark`-like category **and** how much
   true efficiency it costs. Goal: quantify a cleaner operating definition, not just
   relabel losses.
4. **Low-X / left-edge inspection** (prior follow-up #2): is that edge sparking /
   common-mode noisy? Cross-reference the edge/spark findings (see memory
   `june-spark-and-recofar-analysis` and `40_spark_waveforms.py` — sparks are
   edge-seeded on det3).

## Data / infra pointers

- **Ray recipe = v2**: `M3RefTracking(..., chi2_cut=5.0)`, NClus ≥ 3. `qa_config`
  already prefers `m3_tracking_root_v2/` when pulled. **Change `extract_recofar.py:47`
  from 20.0 → 5.0** (and confirm it pulls v2 rays) as step one.
- **Which run for what:**
  - `g_det3_wknd` (above) = the headline **efficiency** run — do the categorisation /
    split here so numbers match the package. **But** its `decoded_root` is on disk
    only for the first ~13k eids (which *end right before* the spark/tail region), so
    it is **not** waveform-friendly past there — pull the rest from lxplus AFS
    (`~dneff/x17/cosmic_bench/june_tests/<run>/<subrun>/decoded_root/`, rsync the
    `*_<FEU>.root` you need) OR:
  - `sat_det3` = `mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V`
    (490 V / 1000 V operating point) has **full local `decoded_root` coverage** and is
    what `40_spark_waveforms.py` was run on — best run for the **waveform** part
    (question 2). Check reco_far there too and confirm the story transfers.
- **Categorisation must match `09_efficiency_breakdown.py`**: same alignment, same
  active box (5 mm margin), spark pulled out at >50 strips *before* the reco/hit split.
- venv: `../../.venv/bin/python`. Reproduce prior: `extract_recofar.py [KEY]` then
  `make_plots.py`; caches (`*.npz`) are gitignored.

## Deliverable

Refresh `det3_recofar_analysis/` on the v2 basis: updated `recofar_meta.json`,
figures, and the report `main.tex`/`main.pdf`, with the v2 split, the waveform
confirmation, and the veto trade-off. If a cleaner operating definition emerges,
note whether/how it should feed back into `09_efficiency_breakdown.py` and the
engineer package's `21-det3A-efficiency-breakdown` (which currently, correctly,
reports the honest 6.5 % as a near-miss/edge position tail — see
`engineer_package/make_efficiency_breakdown.py`).

Related memory: `june-spark-and-recofar-analysis`, `micro-tpc-angle-bias-and-vdrift`,
`june-cosmic-qa-fleet-summary`.
