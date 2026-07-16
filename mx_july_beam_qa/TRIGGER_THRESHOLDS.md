# SiPM-wall trigger thresholds from run224460 (top+bottom sum, one per wall)

**Date:** 2026-07-16 · **Analysis:** `18_trigger_threshold.py` + `18b_trigger_figs.py`
(cache `cache/18_trigsum_run224460.npz`, figures `figures/run224460/18_trigger/`)
· **Slides:** `slides/run224460_slides.pdf` (trigger frames near the end)

## Setup

- Trigger observable: analog sum of each 4-bar group's top+bottom SiPMs; the
  hardware will apply **one threshold per wall** (4 groups share it).
- Trigger candidates: prompt top-bottom pairs (|dt − μ_g| ≤ 8 ns), late TOF
  (>0.1 ms after flash), **duplication-vetoed** (rule of `17_duplication_veto.py`:
  same-side neighbor within ±4 ns, amp ratio ⅓–3).
- MIP (signal) sample: candidates in true plastic coincidence (|dt′| ≤ 8 ns,
  +20..+120 ns sideband subtracted).
- Purity by tag-and-probe: MIP content = tagged/ε_p with ε_p (plastic tag
  efficiency, 0.13–0.22) measured per group in the MIP-dominated high-sum region
  (1.3–2.5× peak).

## MIP-sum peaks per group [mV]

| wall | g1 | g2 | g3 | g4 |
|---|---|---|---|---|
| WALA | 46 | 48 | 48 | 44 |
| WALB | 62 | 66 | 68 | 68 |
| WALC | 66 | 66 | 64 | 70 |
| WALD | 72 | 66 | 72 | **44** |

WALA (whole wall ~30% low gain) and WALD g4 (weak channels 7/8) set each wall's
compromise.

## Recommended thresholds

Rule: highest threshold keeping the **weakest group** of the wall at ≥95% MIP
efficiency.

| wall | **threshold** | eff g1–g4 @thr | purity g1–g4 @thr | pairs/bunch | if duplication unfixed |
|---|---|---|---|---|---|
| WALA | **12 mV** | 0.96 / 0.99 / 0.98 / 0.98 | 0.92 / 0.93 / 0.93 / 0.94 | 2454 | 2878 (+17%) |
| WALB | **14 mV** | 0.97 / 0.96 / 0.96 / 1.00 | 0.90 / 0.89 / 0.88 / 0.91 | 2916 | 2981 (+2%) |
| WALC | **12 mV** | 0.96 / 0.96 / 0.97 / 0.98 | 0.86 / 0.86 / 0.88 / 0.84 | 3184 | 3293 (+3%) |
| WALD | **14 mV** | 0.99 / 0.96 / 0.97 / 0.97 | 0.88 / 0.85 / 0.86 / 0.94 | 2526 | 3168 (+25%) |

## Figures

- `figures/run224460/18_trigger/trigsum_spectra_linear.png` — **the headline view**:
  per-group sum spectra (junk peak ~10–15 mV, valley, MIP bump at 44–72 mV) with the
  recommended wall threshold as a dashed vertical line on every panel.
- `trigsum_spectra.png` — same on log-y; `purity_vs_eff.png` — trade-off curves;
  `threshold_scan.png` — per-wall eff/purity/rate vs threshold.

## Notes / caveats

1. Thresholds are ~20–30% of each wall's lowest MIP-sum peak — safe against
   modest gain drift; purity already 85–95%. Raising the threshold buys little
   purity until ~25–30 mV, where WALA / WALD-g4 efficiency slides below ~90%.
2. **Bandwidth-limited alternative:** ~25 mV keeps strong groups ≥90% but drops
   WALA/WALD-g4 to ~85–90%, for ~40% less rate — full curves in
   `threshold_scan.png` / `purity_vs_eff.png`.
3. **The duplication short cannot be vetoed in hardware**: until the analog paths
   (WALA 5↔7, WALD 2↔4, 5↔7) are fixed, WALD triggers ~25% and WALA ~17% above
   the table's rates, mostly duplicated junk.
4. Rates are per bunch for the late-TOF sample — relative between walls, NOT an
   absolute in-spill bandwidth estimate (near-flash rates are much higher).
5. Assumes current HV. After the recommended WALA (+WALD g4) gain trim, re-run
   `python 18_trigger_threshold.py <run>` + `18b` — walls should then converge
   to a common ~14–16 mV threshold.
