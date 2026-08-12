# June continuity 2026-08-13 — "the new numbers don't look like june_grand_qa.pdf"

Trigger: the remade fleet report contrasted sharply with `june_grand_qa.pdf`
(2026-07-25) — Detector A there was a 22k-ray, well-aligned run at 82.3 %
efficiency; the campaign report showed det3 with 7k rays and different numbers
everywhere. Full audit run overnight 8-12/13, with the June PDF as the
known-answer reference and an independent reproduction on the desktop.

**Verdict: the processing is sound and reproduces the June answers on the
June run.** Every difference the report showed is either (a) a different run,
(b) a deliberate definitional change made since June, or (c) the basis change
hits→waveform-first. One new campaign bug was found and fixed (§3). The fleet
report is remade in the june_grand_qa.pdf format with the June best runs
restored (§6).

## 1 · The single biggest difference: the report keyed det3 on the wrong run

The campaign digest/fleet keys used `sat_det3` (saturday scan long run, 7,049
rays) for det3 — the wft calibration golden run. The June PDF's Detector A
page is `g_det3_wknd` = `mx17_det3_p2_det1_overnight_6-27-26/
long_run_p2_det1_sanity_check` (22,417 clean rays). The campaign DID
reconstruct that run (tier A, promoted), the report just never featured it.
The remade report keys Detector A on `g_det3_wknd`.

## 2 · Validation against the known answer (hits chain, same run)

Today's hits chain through the frozen accounting (`02_efficiency --source
hits g_det3_wknd`) vs the June PDF page:

| quantity | June PDF (7-25) | today | |
|---|---|---|---|
| within 5 mm | 82.3 % | 82.78 % | ✔ |
| has_any | 99.5 % | 99.50 % | ✔ |
| core σ | 0.66 mm | 0.667 mm | ✔ |
| align θ / z | 89.45° / 714 mm | 89.45° / 714 mm | ✔ |
| offsets | −236.83 / −209.14 | −236.84 / −209.14 | ✔ (10 µm) |
| spark (crossing) | 10.6 % | 2.91 % | definitional, §4 |
| reco-at-all | 88.8 % | 96.51 % | consequence of §4 |
| rays | 22,417 | 22,417 (pre-box) | ✔ |

## 3 · New bug found: false-start segment in the campaign reco (FIXED here)

The run's `decoded_root` contains the false-start acquisition
`datrun_260628_01H29_000` whose event ids restart at 0 and collide with the
first main (01H34) segment. The M3 chain had quarantined it
(`m3_tracking_root/_false_start_01H29/`); the condor reco job consumed all of
decoded_root, so `--matched-only` matched false-start detector events against
main-segment M3 rays: **620 duplicate-id rows / 310 colliding ids** in the
promoted table, and a crash in `03_angles` (duplicate index → ambiguous
`.loc`). A scan of all 153 staged campaign parquets shows this run is the
ONLY one affected.

Fix: both copies of every colliding id dropped (segment of origin is not
recoverable from the table); originals parked in
`wft/pre_clean_falsestart/` with a README; staging copy untouched.
**Post-freeze queue:** `run_reco_job.py` must restrict decoded_root to the
datrun prefixes present at the top level of the M3 tracking dir.

## 4 · Definitional shifts since June (deliberate, not regressions)

- **Spark veto** — the 2026-07-25 relative-significance floor
  ([[june-rerun-matched-filter-reco-regression]]) tags multiplicity on
  floor-filtered strips. Crossings the old veto binned as `spark` are now reco
  attempts: det3-p2 10.6→2.9 %, det6 43.5→17.9 %, det7 66.5→27.0 %. On the
  hits chain those events mostly land in `reco_far` (6.5→13.7 % on det3-p2);
  the wft fit recovers them (reco_far 3.8 %). This is where det6/det7 "gain"
  20–40 efficiency points vs the June PDF.
- **Active box** — each accounting draws its own 0.5–99.5 percentile box from
  its own reco footprint; ray denominators differ by ±2–7 % between chains
  (det6 wft 9,628 vs hits 10,340 vs June 10,366). Not lost data.
- **θ resolution** — June's bar was the hybrid-hits estimator (script 34
  σ68); the report now quotes the waveform-first σθ with w0/kw applied,
  roughly 2× better fleet-wide.

## 5 · Detector A on the waveform-first basis (the new page-A numbers)

22,197 rays · within 5 mm **87.10 %** · reco-at-all 90.86 % · has_any
99.99 % · spark 2.88 % · core σ **0.435 mm** · median |r| 0.694 mm ·
σθ X/Y **1.19 / 1.12°**, bias −0.06 / −0.01° (w0/kw-corrected; frozen-code
Y bias −0.26° matches the arctan(w0_y/v) prediction, INVESTIGATION §4).
Bundle: `calib_bundle_lp2_t0p` (same detector/conditions as sat_det3; copied
into this run's wft tree so downstream tools resolve it).

**Independent desktop reproduction:** inputs synced to the desktop's own
Analysis tree, `02_efficiency` run there from the fleetcheck worktree —
category counts identical to the event (19,333 / 835 / 639 / 1,387 / 3).
Together with the earlier 400-event bit-identical re-reco from the desktop's
own raw copy (verify.log), the chain is machine-independent end to end.

## 6 · The remade report

`mx_june_wft/report/make_grand_report.py` now emits
`Analysis/fleet_report/report.html` in the june_grand_qa.pdf layout: fleet
summary (four bar charts, efficiency/spark vs resist HV, MM layout, fleet
table + a June→tonight continuity table), then Detector A–E sections with the
June stat cards, info box and figure slots. Figures built by
`mx_june_wft/report/make_june_figs.py` (per-ray CSV → sliding trio, hit/miss
scatter, wide breakdown, position/angle correlation densities) on the wft
basis. Self-contained copy `note_selfcontained.html`; the 8-12 tabbed report
is archived in `fleet_report/archive_tabbed_2026-08-12/`.
