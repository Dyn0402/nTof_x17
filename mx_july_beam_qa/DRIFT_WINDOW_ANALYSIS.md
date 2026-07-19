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
| det A drift | **700 V** (staged via 650) | not for speed today (A is dry, gains 0.7 smp) — for **wet-excursion immunity**: at 0.8 % H2O (= what B/C/D breathe *right now*) A@600 V needs a ~22.6 smp column and truncates any affordable window; A@700 V needs ~17.7 and fits with the fall inside n=30 |
| B/C/D drift | 800 V (keep) | already fastest allowed; wetter gas needs the field |
| latency | **35 → 32** now, → **30** after clearing B | prompt peak = latency−26.5, pulse rises 4 smp before peak (§2b). Latency 32 keeps the rise at ~sample 5–6 with ~5 median baseline; latency 30 (median baseline 3) is the aggressive target. |
| n_samples | **32 → 29** now, → **24** after clearing B | scaled loss table §2d: n=29 loses **0 % of primaries, 0.05 % of charge** on clean dets and keeps det B's sample-31 bin; n=24 cuts **8 smp (25 %)** for still **0 % primaries lost, 0.56 % mean charge loss** — but drops B's tail, so needs B cleared first. |
| the gate | **det B** | B's deep edge piles at the sample-31 ceiling (noise vs real-drift-already-truncating, unresolved). It is what separates "cut 3" from "cut 8"; resolve it (§5) to unlock max-trim. |
| beam drift scan | **yes, small one** (A: 600→650→700, ~30 min each) | the real gate for 700 V is spark stability under beam; span(600)/span(700) pins the water model; B/C/D stay untouched |

> **How much can we safely cut?** On clean detectors (A/C/D) the loss is nearly flat down
> to **n=24**: **0 % of drifting primaries lost** (peaks always captured — only the far
> charge *tail* is clipped) and mean charge loss **<0.6 %** (§2d, `fig_loss_curve.pdf`).
> So the aggressive target — **latency 30, n=24 (cut 25 %)** — is defensible on physics;
> the only blocker is **det B**. Until B's sample-31 pile-up is confirmed noise, keep
> that bin: **latency 32, n=29 now** (cut 3 for ~zero loss), then n=24 once B is cleared.
> *(An earlier draft's "latency 32/n=28" was sized on peaks and ignored the +7-sample
> fall; loss here is computed from the measured pulse shape folded onto the deep-edge
> distribution — the honest way to size it.)*

The single highest-leverage action is not DAQ at all: **dry the gas line**. If B/C/D
reached det A's dryness, every detector's column would be ~11–12 smp and a deeper trim
(n≈26, full containment) would become safe. A dryer/higher flow is worth more than any
window trimming.

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

## 2. Window budget

### 2b. The pulse has width — measure the rise and the fall, not just the peak

Peak positions alone under-size the window. From the peak-aligned mean pulse (cosmic
hit strips, `slides_drift_window/fig_pulse_shape.pdf`; A_x and C_x identical):

| edge | offset from peak | meaning |
|---|---:|---|
| rise onset (5 %) | **−4 smp** | pulse first climbs off baseline |
| rise 50 % | −2 to −3 | |
| fall 50 % | +3 | |
| fall 20 % | **+5** | pulse essentially complete |
| fall 5 % (≈ baseline) | **+7** | fully returned; then a small CR-RC² undershoot |

So a single primary occupies **peak−4 → peak+7** (≈ 11 samples wide). The window must
hold: [earliest prompt rise − ~3 baseline]  …  [latest deep peak + fall]. The earlier
draft used "+2" for the fall — wrong by ~3–5 samples.

### 2c. Budget with the real footprint (prompt-anchored, worst clean detector, p99)

Worst-clean anchored (latency-35 frame): earliest peak p5 = 8, latest peak p99 = 25.
Rise = peak−4; baseline wanted = 3 before rise; fall = peak+5 (to 20 %) / +7 (baseline).
Lowering latency by Δ shifts everything Δ earlier:

| latency | baseline before rise | n for detect+time | n for fall→20 % | n for fall→baseline |
|---:|---:|---:|---:|---:|
| 35 (now) | 4 | 29 | 31 | 33 (>32 → clips deepest 1 % now) |
| **34 (rec.)** | **4** | 28 | **30** | 32 |
| 33 | 2–3 | 27 | 29 | 31 |
| 32 | 2 (too few) | 26 | 28 | 30 |

- **latency 34, n=30 (recommended):** 3–4 baseline before the rise, full pulse to the
  20 % level for ≥99 % of clean-detector primaries and to baseline for ≥95 %. Only the
  deepest ~1 % have their <20 % tail clipped (still detected, timed, ~80 % of charge).
  Net trim = 2 samples / 6 % vs the current 32.
- **latency 34, n=32 (conservative):** keep n, just fix latency — full baseline-return
  on the deepest 1 % and headroom for a wet-gas excursion or det B (§5). Reclaims the
  pre-roll only.
- **latency 33, n=28 (aggressive):** only if the requirement is detect + drift-time,
  NOT full charge. This is what the earlier "latency 32/n=28" really was.
- **det B caveat (§5):** B's fall runs to the sample-31 ceiling even prompt-anchored;
  if that is real drift rather than residual noise, B wants n≥32 or resolution first.
- Full containment leaves little to trim: the honest saving is ~2 samples, and it comes
  from the fall budget, not from imaginary pre-roll. Drying the gas is the real lever.

### 2d. Scaled options — how aggressive can we go, and what does it cost?

§2c sized for *full pulse containment of every primary*. If instead we accept the
tiny, quantified loss from (a) opening ~1 sample before the rise and (b) clipping the
far end of the deepest tails — exactly the ask — the trim is much larger. Loss computed
by folding the measured pulse shape (charge captured vs where the window ends relative
to each strip's peak) onto the per-track deep-edge distribution, prompt-anchored,
**clean detectors A/C/D** (curve: `slides_drift_window/fig_loss_curve.pdf`):

| option | latency | n | cut | primaries lost | mean charge loss | tracks losing >5% of a strip | baseline (med) | keeps B's smp-31? |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| conservative | 34 | 32 | 0 | 0.0% | 0.02% | 0.2% | 7 | yes |
| **balanced (now)** | **32** | **29** | **3 (9%)** | **0.0%** | **0.05%** | **0.6%** | 5 | **yes** |
| aggressive | 31 | 28 | 4 (12%) | 0.0% | 0.05% | 0.6% | 4 | yes |
| **max-trim** | **30** | **24** | **8 (25%)** | **0.0%** | **0.56%** | **3.1%** | 3 | **no** |

Two loss channels, both small:
- **Deep primaries lost (peak clipped)** — the "losing a primary" metric — is **0.0 %
  on clean detectors down to n=24**, because the deepest anchored peaks sit at frame-35
  sample 25 (p99); you never clip a peak, only the charge *tail* past it. Below n≈23 it
  rises steeply (the knee).
- **Charge loss** is a mean over tracks; it stays <0.6 % to n=24 because only the
  deepest few % of tracks have any tail outside the window (the deepest track at n=24
  keeps ~87 % of its deepest strip's charge; the mean is dominated by shallow tracks
  that lose nothing).
- **Early side:** opening within ~1 sample of the rise costs a fixed **~1.1–1.4 % of
  tracks** their leading edge — but these are anomalously early risers (frame-35 rise
  ≤ 3, well before the prompt at 8), i.e. likely not clean prompt tracks. Median
  baseline stays ≥3 samples down to latency 30.

**The one gate is det B.** All options except max-trim keep the sample-31 bin, so if
B's late pile-up is real drift you preserve whatever it has there; **max-trim (n=24)
drops B's 29–31 coverage** and is only safe once B is cleared as residual noise (§5 —
a short common-mode-by-connector check, or the drift-scan stats). If B is real drift it
is *already* truncating at the current n=32, so trimming the others doesn't worsen it —
but you would not want to trim B's own window further.

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

## 5. v6 prompt-anchored cross-check — DONE 2026-07-19 (locally, not on the DAQ)

Ran the v6 per-event pass **locally** (decoded ROOT rsync'd off-box at 15 MB/s, idle
I/O priority; processed on a 62 GB workstation; ROOT deleted after — see §6 for why
never on the DAQ). The cut requires each track cluster to contain a **prompt strip**
(earliest peak ≤ sample 12) — a real through-going drift column must, so this removes
residual coherent-noise/ringing clusters that lack one. run_54 cosmics seg000, deep-edge
ceiling (fraction of tracks with latest peak ≥ sample 31), raw vs anchored
(`figures: slides_drift_window/fig_v6_crosscheck.pdf`; JSONs `v6_{cosmics,beam}.json`):

| plane | raw ceil | anchored ceil | anchored deep-edge p95 | verdict |
|---|---:|---:|---:|---|
| A_x / A_y | 0.3 / 1.0% | **0.0 / 0.0%** | 22 / 19 | clean |
| C_x / C_y | 0.5 / 2.3% | **0.0 / 0.0%** | 24 / 20 | clean (C_y ringing removed) |
| D_x / D_y | 2.5 / 8.4% | **0.0 / 2.6%** | 22 / 20 | **COLLAPSED — D_y late edge was ringing, as predicted** |
| B_x / B_y | 20.7 / 25.7% | **6.0 / 9.3%** | 31 / 31 | **STILL HIGH — does not collapse** |

**Result: confirms the recommendation for A, C, D** — their true (prompt-anchored) deep
edges sit at samples 19–24, comfortably inside n=28, and the scary y-plane/D_y ceilings
of the first look were ringing (removed by the anchor cut, 54–84% of raw clusters were
non-anchored junk). **New finding: det B is a genuine outlier** — even prompt-anchored,
B_x/B_y keep ~6–9% of tracks at the sample-31 ceiling with deep-edge p95 = 31 and
nstr≥6 span ≈ 20 smp (→ v ≈ 25 µm/ns ≈ 1.2% H2O). Caveats: small counts (≈4–6 anchored
ceiling events/plane), cosmic-only (beam run_55 c00 gave too few anchored tracks per
plane, <20, to confirm), and B had the worst coherent noise. So B is **either** a real
slow-drift population (B wetter than C/D — odd for its 2nd-in-chain position) **or**
residual coherent noise that co-clusters with a prompt hit; this pass cannot separate
them.

**Impact on the recommendation:** for A/C/D unchanged (latency 32 / n=28 safe). For
**B it is marginal**: if B's late population is real drift, lowering latency 35→32
shifts B's ceiling edge from sample 31 to ~28, i.e. right at the n=28 boundary, so the
deepest few % on B could clip. Mitigations, in order: (1) **resolve B first** — take a
beam subrun with more stats and/or a B-specific coherent-noise study (common-mode by
connector) before freezing the trim; (2) if B stays suspect, set **latency 31** (prompt
→ sample 5, B edge → ~27, fits n=28 with 1 smp to spare) rather than 32; (3) accept a
quantified few-% deep-charge loss on B only. This does **not** change the A→700 V call
or the n=28 headline; it adds one gate ("clear B") before the most aggressive latency.

Still open: run_53 (03:11 same day) as a 16 h gas-stability cross-check; electronics fix
for the post-saturation ringing blocks (per-FEU fixed channel ranges, e.g. FEU2 38–63 &
448–472). Both to be done **locally / off the DAQ**.

## 6. Incident note (2026-07-18/19) — the analysis OOM-crashed the DAQ machine

During this study, parallel `uproot library='np'` extraction jobs (each loading a full
decoded ROOT file into RAM, ~1–1.5 GB) were launched in the background on `daq_lxplus`
across several ssh drops. Each drop hid the still-running job, so jobs were relaunched
rather than cleaned up: **8 concurrent python jobs (~8.5 GB) accumulated**. The machine
has **15 GB RAM and no swap** and was simultaneously taking beam data (run_55). uid-1000
RSS reached 14.87/15 GB; the OOM killer cascaded from 23:32 (starving `RunCtrl`, the
DREAM `java` DAQ, and `CAENGECO2020` HV control), the box hung ~00:23 and stayed down
until a **manual reboot at 09:06 (~9 h downtime; tail of run_55 lost)**. Evidence:
`journalctl --boot=-1 | grep -i oom` (task dump shows PIDs 3171574/3171781 = the two
confirmed jobs + the 3176xxx cluster). **Rule going forward: do NOT run heavy analysis
on `daq_lxplus`.** Copy decoded ROOT off-box and process on a workstation; if a quick
on-box peek is unavoidable, one job, hard `entry_stop` cap (≤1000 evts), never
backgrounded, and `ps -u mx17 | grep python` to confirm no orphans before and after.
