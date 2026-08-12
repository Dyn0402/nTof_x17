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

## 5b · Follow-up (same night): the 5.3 % wft deficit is a job artifact, and
## the angle "hole" is a self-inflicted mask

Second look after the 87.1 % detA number raised "how did A get less
efficient":

- **The campaign job silently dropped ~5.3 % of matched events on this run.**
  1,175 hit_no_reco rays are simply absent from the job's table; local re-reco
  of a 300-event sample at the frozen code+bundle recovers **300/300** with
  valid X+Y fits. Losses are random per event (gap statistics = pure
  thinning), inputs are byte-identical EOS↔local, `wft/` is unchanged since
  freeze `effef73`. The job's own M3 event list is provably different: 36,745
  events vs 26,670 recipe-passing locally, 97.8 % inside the χ²-only
  superset, 5,943 recipe events missing → `M3RefTracking` resolves
  differently under LCG_105 (python 3.9, uproot 4.3, awkward 1.10) **on v1
  rays files** — and this run is the only tier-A run without
  `m3_tracking_root_v2` (audit: det6 lost 0 events, det7 lost 3).
  Fix: full local re-reconstruction (frozen code+bundle, local matched list),
  campaign table parked in `wft/campaign_lxplus_reco/`. Post-freeze queue:
  repro the v1-file read on LCG_105; run future campaigns' M3 matching on
  v2 everywhere or pin a modern stack.
- **The head-on "hole" in the angle correlation is the `slope_reliable`
  gate, not the fit.** The gate (|tan| ≥ 0.08) is a hits-chain inheritance —
  June needed the 33/34 signature-hybrid because the time-ladder angle has no
  lever arm head-on. The forward fit does not have that problem: with w0/kw
  applied, detA's head-on bands are unbiased (|bias| ≤ 0.15°) at σ68
  1.0–1.7° (same as inclined bands), with 88–97 % sign fidelity down to
  1–3°. The gate was masking **37 % (X) / 44 % (Y) of reconstructed planes**
  out of the angle accounting and plots. The report now uses full coverage
  (`angles_w0corr/angles_fullcoverage.json`, June's |θ|<5° σ68 convention);
  the frozen `03_angles` output is untouched; retiring the gate there is a
  post-freeze item. The genuine head-on weakness is tiny and different:
  46 in-table X-plane fit failures skew head-on (0.2 % of rays).
- False-start decoded files quarantined
  (`decoded_root/_false_start_01H29/`), mirroring the M3 quarantine, so no
  future reco consumes them.

## 5c · Resolution (early 8-13): detA lands at ~93 %, fixes propagated

Full local re-reconstruction of `g_det3_wknd` at the frozen code+bundle:
**within 5 mm 93.0 %, reco-at-all 97.2 %, core σ 0.444 mm** — matching
sat_det3 (93.3 %) as predicted; the "less efficient detector A" was entirely
the campaign-job artifact. w0/kw-corrected angles: bias −0.06/+0.02°,
σθ 1.17/1.14°, |θ|<5° σ68 1.33/1.26° at full coverage (June hybrid: 1.63°).
(First re-reco pre-dated the decoded false-start quarantine and carried 175
colliding ids; an event-level dedup mistakenly clobbered its candidates
sidecar, so the final table is a clean re-reco with the quarantine in place —
no collisions by construction.)

Fixes made everywhere per Dylan's instruction (commit `0f35a3c`):
`03_angles.py` full-coverage accounting (+ `s68_lt5_deg`, `*_relonly`
continuity fields, loud dup-id handling; implied-v keeps the gated basis);
`run_reco_job.py` restricts decoded fetches to M3-covered acquisitions,
records `n_matched`/`m3_has_nclus`, and FATALs on NClus-less M3 reads.
Fleet rerun with these fixes requested from the `cosmic_reprocess_12-8-26`
session; the n_TOF reconstruction session briefed (w0/kw stopgap, slope gate,
LCG matching gotcha). Still queued post-freeze: restore w0/kw in `plane_fit`
itself; head-on X-plane fit fragility (46 events, 0.2 %).

## 5d · Arm-D (det7) beam-side sign mirror: bench records checked, clean

The run_145 beam session reports arm D fitting with a mirrored x-plane sign
vs arms A/B/C (det4-SPS-style connector inversion suspected). June bench
records exonerate the map: mx17_7's `dream_feus` ordering is identical to
det2/3/4, and all five June alignments (hits and wft) converge at
`ref_x_sign=+1`, θ 89.2–90.1°, positive-slope correlations — a mirrored map
cannot produce det7's 0.62 mm core σ. The suspect is beam-side: July install
re-cabling, arm-D mounting parity (D pairs against B across the target), or
the pointing fit's `sign_z` for that arm. Handed back to the beam session
with a discriminating test (externally-confirmed multi-arm track through D).

Second round: the beam session's wall test is sign-blind on D (96 mm lever ×
small angles) but surfaced a real POSITION-frame anomaly — per-segment wall
crossings compressed toward centre (+22/+29/−80/−108 mm vs bands ±75/±175;
~36 % in-band vs arm A's 51–71 %). Bench closes its side: det7's position
scale vs M3 is 0.9997 (X) / 1.0012 (Y), offsets ≤ 0.1 mm — fleet spread
[0.9989, 1.0016] — and no 8-channel group discontinuity (a re-plug offset
would show one). Map, parity, pitch, ordering all exact. Remaining suspects
are beam-side only: arm-D geometry description (position/rotation/lever) or a
position-only reflection (strip origin from the wrong side, u→L−u with tan
untouched — invisible to the fan-sign test, maximal in the wall test);
suggested re-running D's per-segment medians under u→L−u / u→−u as the
no-new-data verdict.

Third round, closing the thread: the beam session ran the reflection test —
**no reflection** (both mirrors make D sharply worse, 36→17–19 % in-band,
ordering anti-correlates; control arm A 60→9–13 % shows the instrument has
teeth). D's frame convention matches A, consistent with the exact bench
scale/parity. The residual anomaly is ONE-SIDED (positive-u segment bands
compressed, negative-u nearly correct), which no offset/scale/reflection can
produce — leading candidates are wall-side instrumentation (the
[[sipm-readout-window-side]] record's dead-bar asymmetry, never verified
against data, per-arm orientation unconfirmed; or dead/hot channels in D's
positive-u wall pairs). Testable from slim occupancy alone; on the beam
session's queue. Bench involvement CLOSED; published arm-A/fleet v numbers
unaffected.

Fourth round (beam session, occupancy test): wall-side hypotheses
DISFAVORED — D's +u pairs have fleet-normal rates (0.17/0.31 vs A's
0.18/0.29) and amplitudes; the one-sided compression must live in the
beam-side geometry description of D or D's reconstruction on that half.
Multi-arm confirmed-track test is now the primary instrument. Byproduct for
the record: **WALD detn 7 (pair 3, −u side) runs at half the fleet-typical
median amplitude (666 vs ~1050–1300) — weak SiPM channel**, unrelated to the
compression but relevant to any WALD hit-efficiency accounting.

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
