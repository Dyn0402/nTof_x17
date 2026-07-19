# run_55 resist-HV scan — MM tracking vs time-since-flash (2026-07-19)

**Goal: find the optimal resist HV per detector for data-taking at ≥3 ms after
the gamma flash, balancing MIP efficiency against post-flash saturation.**

Data: `~/x17/beam_july/runs/run_55/` (combined_hits + decoded pulled from EOS;
raw_daq_data/hits_root still on EOS only). Cyclical scan r560→r520 in −5 V
steps, all four dets together, 10 min/subrun; drift fixed 800 V B/C/D, 600 V A;
gas Ar/iso 90/10; scint-DOUBLES trigger (≥2-of-4 wall×plastic, MM-independent)
in a 30 ms beam gate; ³He target; no beam filter. 24 subruns survive (3 cycles
for r535–560, 2 for r520–530; DAQ died mid `c02_024`). 76 214 triggers.

Scripts: `25_hv_scan_extract.py` (per-event top-5 clusters/plane → cache) and
`25b_hv_scan_analysis.py` (figures in `figures/25_hv_scan/`, summary json in
`calib/25_hv_scan_summary.json`).

## How the time axis works (and its big limitation)

- Triggers come in bursts, one per beam pulse (~3 s apart). First trigger of
  each burst = gate open = gamma flash (t₀). In ~30 % of bursts the MM records
  the flash as ~2600 low-amp garbage hits ("partial saturation" class); in
  ~70 % the flash event is EMPTY in the MM (electronics railed flat, or lower
  intensity pulse) — but the burst time structure is identical, so t₀ = first
  trigger is a valid flash anchor for all bursts.
- **The DAQ FIFO samples the gate as a comb**: ~4–5 events at 0–0.5 ms, then
  ~8 ms readout dead time, a batch at ~8–12 ms ("b1"), another at ~16–28 ms
  ("b2"). **2–6 ms after flash — including the 3 ms target — is unsampled.**
  (No-ZS events are ~1.6 MB; readout ~8 ms/batch. A ZS or short-window run
  would fill the gap.)
- Flight-path physics: 19 m / thermal (2200 m/s) ≈ 8.6 ms — **b1 sits exactly
  on the thermal peak**; b2 is sub-thermal.

## The ³He-capture flood (dominant systematic)

n+³He→p+t (764 keV, heavily ionizing, mm–cm range) floods the MMs at thermal
times. Det D sees a capture-scale cluster in ≳90 % of b1/b2 windows (A/B/C
much less — geometry). Consequences:

- "Track candidate = any x/y clusters" is contaminated by captures; wide/high
  charge blobs also mask the trigger track (top-5 clusters/plane in the
  extraction mitigates this).
- MIP-like selection (3–20 strips, ≤25 mm, x/y sample overlap) suppresses the
  blobs but is gain-biased: at high HV real MIP clusters widen past the cuts;
  at low HV capture blobs shrink INTO the cuts (det D's "MIP" rate at 520 V is
  ~13 % — capture fragments, not MIPs; A/B/C sit at 1–2 % there).
- No clean drift-time discriminator exists (micro-TPC arrival spans the whole
  window), so per-trigger MIP rates are an efficiency PROXY, cleanest for A,
  usable for B/C, capture-contaminated for D below ~540 V.

## Q1 — when do tracks turn on after the flash?

1. **0–0.5 ms: everything is dead at every HV.** 0 tracks in 44 686 triggers;
   alive fraction (any MM pulse above threshold) 0–7 % (D ~12 %, its capture
   flood already visible). The scint trigger itself is unaffected (rates
   uniform across the gate) — it is purely an MM outage.
2. **2–6 ms: no data** (DAQ comb). Turn-on completes somewhere in 0.5–8 ms.
3. **By ~8 ms: recovered — except the highest HVs on C and D.** Alive
   fraction and MIP rate are flat b1→b2 at ≤550 V (all dets). At 555–560 V:
   - **C: MIP rate 0.5–1.2 % at b1 vs 5.9–7.8 % at b2** (occupancy 24 %
     rising to 36–44 % through the gate) — still gain-sagged at 10 ms,
     recovered by ~20 ms.
   - **D: same signature, milder** (occupancy 83 % → 98 % at 560 V).
   - Saturation time grows with HV, as suspected; at 560 V it exceeds 10 ms
     for C and D.
4. **Early-track amplitude**: median amp of b1 tracks is ≥ b2 (ratio up to
   1.3–1.4 where sag is present) — survivor bias: under sagged gain only the
   Landau tail clears threshold. So yes, early tracks are effectively
   suppressed; it shows up as rate loss + biased-high medians, not low medians.
5. Det A shows the OPPOSITE in-gate trend (b1 13.6 % → b2 6.1 % at 555):
   efficiency degrades through the gate. Cause open — candidate: space-charge
   accumulation in A's low drift field (600 V) under the capture/thermal load,
   or trigger-mix evolution. Worth a dedicated look.

## Q2 — efficiency and amplitude vs HV at fixed time (b1, 8–12 ms)

MIP-track rate per trigger (b1) — the geometric ceiling per arm is ~50 %
(doubles trigger, unknown which 2 arms):

| HV  |  A   |  B  |  C  |  D*  |
|-----|------|-----|-----|------|
| 520 |  1.5 | 0.8 | 2.1 | 12.9 |
| 530 |  2.0 | 2.2 | 2.5 |  8.9 |
| 540 |  3.1 | 2.8 | 3.2 |  9.4 |
| 550 |  8.1 | 3.1 | 2.1 | 13.9 |
| 555 | 13.6 | 2.5 | 1.2 | 13.2 |
| 560 | 12.3 | 2.1 | 0.5 | 13.4 |

*D contaminated by capture fragments, esp. ≤540 V. C/B ≥550 V suppressed by
long saturation (their b2 curves keep rising to 555 V — the gain is there,
the detector just isn't recovered at 10 ms).

- **A**: clean exponential-ish rise, no plateau by 560 V, no saturation sag
  even at 560 V. A is simply under-biased (dry gas, drift 600 V).
- **B**: weak HV dependence 2–3 %, b2 rises to ~5 % at 555–560 V. B also has a
  population of very wide (~200 mm) low-amp clusters at all HVs (the known
  flash-ringing/coherent junk) that complicates its clustering. Inconclusive —
  needs the ringing understood first.
- **C**: true gain curve peaks ≥555 V (b2), but the long-saturation penalty
  at ≥555 V makes early-gate operation much worse than 540–550 V.
- **D**: alive everywhere (captures visible even at 520 V); plateau-ish MIP
  rates 545–560 V; saturation sag appears at 560 V.
- Median MIP amplitudes are threshold-biased (only tracks above detection
  enter) — they sit at 8–12k ADC (x+y) everywhere; not a gain measurement.
  The resist-current curve of D (1.0→5.4 µA, 520→560 V) doubles every ~17 V —
  a real gain slope; A's current is leakage-dominated (2.7→3.7 µA only).

## Recommendation (for ≥3 ms operation, this gas/config)

| det | operating point | why |
|-----|----------------|-----|
| A | **≥560 V** (and revisit drift 600 V) — not yet efficient; scan higher before freezing | rate still rising at 560, NO saturation penalty visible even at 560 |
| B | **550–555 V** tentative | b2 gain still rising to 555; no strong sag seen; blocked on the wide-cluster/ringing issue |
| C | **545–550 V** | gain wants ≥555 but long saturation (>10 ms) at ≥555 V; 545–550 = best early-gate efficiency |
| D | **540–550 V** | plateau by ~545; sag at 560; lowest HV with full capture visibility and near-ceiling MIP rate |

**Caveats**: (i) 3 ms itself was never sampled — at 555+ V, C/D are provably
NOT recovered at 8–12 ms, so the 3 ms goal likely needs the lower operating
points or a saturation fix; (ii) all efficiencies are per-trigger proxies with
unknown per-arm geometric participation; (iii) B is unresolved.

## Follow-up wanted

1. **A run that samples 1–8 ms**: zero-suppression ON (or short window /
   fewer samples) to shrink readout dead time, or a delayed trigger gate
   stepping through 1–8 ms. This is the only way to answer "alive at 3 ms".
2. Investigate det A's in-gate efficiency decline (drift 600 V space charge?).
3. Det B wide-cluster/ringing diagnosis (waveform-level, needs raw/decoded).
4. Higher-HV point for A (565–575) to find its plateau.
5. If absolute efficiency is wanted: tag-and-probe across arms (doubles
   trigger gives a correlated pair; solvable for per-arm ε with the 16-pattern
   likelihood) — machinery exists in `ntof_tracking/`.

## Figures (figures/25_hv_scan/)

- `01_time_structure.png` — trigger comb + per-det track rate vs time
- `02_turnon_vs_time.png` — MIP rate vs t per HV (4 dets)
- `03_eff_vs_hv.png` — b1 vs b2 rate vs HV, per-cycle dots (KEY)
- `04_amp_vs_hv.png` — median amp vs HV (threshold-biased, see text)
- `05_amp_vs_time.png` — amp vs t per HV
- `06_amp_spectra_b1.png` — b1 amplitude spectra
- `07_blob_activity.png` — capture-blob probability vs time (pile-up map)
- `08_alive_recovery.png` — occupancy vs time per HV (KEY: saturation
  recovery; rising curves = still saturated in-gate)
- `09_early_vs_late.png` — b1 vs b2 scatter (above diagonal = sagged at 10 ms)
