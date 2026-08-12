# Overnight investigation 2026-08-12/13 — det3 "something looks wrong"

Trigger: (1) the fleet-report efficiency maps looked off-centre and the track
counts low; (2) the HV-scan tab's det3 fit-quality fraction collapses with
resist HV. Both chased to ground on det3, verified on the desktop against an
independent data copy, and compared to the combined-hits analysis throughout.

**Verdict up front: the reconstruction, the alignment and the campaign
numbers are sound** — bit-identical on re-reconstruction from a second copy
of the raw data on a second machine. What was actually wrong, in decreasing
order of importance:

1. the frozen code silently **stopped applying the calibrated w→angle
   constants (w0/kw)** — this is the fleet-wide Y angle bias, now measured,
   explained, and corrected through the standard accounting;
2. the per-plane `quality_ok` flag is an **amplitude cut in disguise** and
   collapses with HV by construction — it gates nothing in the headline
   accounting, and the HV tab now plots a gain-normalized metric instead;
3. the maps *look* off-centre because the detector *is* off-centre — same
   footprint, bin-for-bin, in the hits chain.

---

## 1 · Alignment and centring: nothing is wrong

- wft vs hits-chain alignment for golden det3: **identical to 10 µm / 0.05°**
  (offsets −236.84/−209.10 both chains; θ 89.40 vs 89.45; z 714 vs 714/715).
- The efficiency footprint (X ≈ [−215, +145] mm in reference frame) matches
  the hits-chain `map_within_5mm.png` bin-for-bin, including the dead column
  at X > 145. That column is the detector-local **Y-passivation band**
  (active Y = [18, 380] mm, `common/mx17_active_area.py`) mapped to +X by the
  ~90° strip rotation, plus a ~35 mm mounting offset from the telescope axis.
  Real geometry, present in every previous analysis.

## 2 · Track counts: nothing is wrong

Golden det3 stage by stage (numbers identical on desktop):
47,452 M3-readout events → 41,455 raw tracks → recipe **[χ²<1 & NClus≥4]**
keeps 8,479 events with a good track (17.9 %) → active-area/box cut →
**7,049 rays** in the wft denominator (hits-through-same-accounting: 7,119;
pre-campaign baseline: 7,130 — the ~1 % differences are the alignment box
edge). The "missing" tracks are the frozen M3 recipe, unchanged from every
previous analysis. Relaxing it is the known +25 % from the drop-layer study —
a deliberate, separate decision, not a campaign regression.

## 3 · The quality flag: an amplitude cut in disguise

`quality_ok = chi2/dof < 300` (absolute, `wft/reco.py`). The fit's χ² is
weighted by **pedestal noise only**, so any shape mismatch proportional to
signal gives χ²/dof ∝ amplitude². Measured on golden det3:

- scaling exponent d(log χ²/dof)/d(log q_sum) = **2.03 (X) / 1.95 (Y)**;
- median χ²/dof 22 → 812 across amplitude octiles *within one run at fixed HV*;
- `p0_err` pinned at its 0.330 mm floor in **every** octile — fit precision
  does not degrade with amplitude;
- quality-FAIL events still reconstruct at **93.1 % within 5 mm** (median r
  0.776 mm vs 0.674 for PASS);
- across the two drift-1000 HV scans, (χ²/dof)/(q/1000)² is **flat**
  (≈1 saturday run, ≈2–3 6-22 run) from 445–505 V: the entire
  quality-vs-HV collapse is gain², nothing else;
- genuine saturation exists but is small and confined to the top decile:
  dof/strip 32.0 → 30.4 (≈5 % of samples censored at `SAT=3550`), where
  within-5 mm drops to 81 % — δ-ray/shower-rich events, physics not
  pathology.

**Impact: none on headline numbers** — `compat.load_table(quality_only=False)`
everywhere in the accounting; the flag is informational. The HV tab now
plots the gain-normalized shape χ², which is flat where conditions are good,
≈7–10 on the 6-23 drift-600 series (real off-conditions v mismatch) and ~216
on the near-dead 6-26 quick run — a metric that flags real breakdowns.

Post-freeze fix queued: make `quality_ok` amplitude-aware (threshold on the
gain-normalized χ², or scale the noise model with signal).

## 4 · The fleet-wide Y angle bias: unapplied calibration constants

`9dd7d6e` introduced per-plane angle constants and calibrated them into every
bundle: `tan = (w·1e3 − w0[plane]) / (kw[plane] · v)`. **`f9e18d2`'s
`plane_fit` rewrite silently reverted the formula to `tan = w·1e3/v`** — the
frozen campaign computes angles without the constants that every bundle
carries. Prediction arctan(w0/v) vs the campaign digest bias:

| det | pred X | meas X | pred Y | meas Y |
|---|---|---|---|---|
| det3 | −0.01 | −0.07 | −0.27 | −0.28 |
| det2 | −0.33 | −0.36 | −0.32 | −0.39 |
| det4 | −0.19 | −0.16 | −0.30 | −0.19 |
| det6 | +0.14 | +0.03 | **−1.14** | **−1.04** |
| det7 | +0.05 | −0.09 | −0.53 | −0.32 |

The Y-heavy pattern, the det6 outlier, and det2's both-plane bias are all
reproduced. Applying the constants post-hoc (exact 9dd7d6e formula; script
`corrected_angles.py`, standard 03_angles accounting, results in
`<OUT_BASE>/wft/angles_w0corr/`) collapses every |bias| to ≤ 0.27°:

| det | frozen bias X/Y | corrected bias X/Y | corrected σθ X/Y |
|---|---|---|---|
| det3 | −0.07 / −0.28 | −0.07 / −0.04 | 1.15 / 1.14 |
| det2 | −0.36 / −0.39 | −0.07 / −0.08 | 1.25 / 1.51 |
| det4 | −0.16 / −0.19 | +0.03 / +0.08 | 2.39 / 2.46 |
| det6 | +0.03 / −1.04 | −0.10 / +0.22 | 2.39 / 2.59 |
| det7 | −0.09 / −0.32 | −0.13 / +0.27 | 1.94 / 1.67 |

kw (up to 1.089 on det6/7 Y) also rescales slopes: σθY improves a few % when
applied; implied-v spread is computed from w directly and is unchanged.
This closes the "fleet-wide angle bias" open item — it is a **code
regression at the freeze**, not a detector or calibration change. The n_TOF
tilt concern (constant δθ read as v-tilt) is bounded by these corrected
numbers. Post-freeze fix queued: restore the two lines in `plane_fit`;
until then any quoted angle bias must come from `angles_w0corr/`.

## 5 · Independent reproduction (desktop)

400 golden det3 events re-reconstructed on the desktop (its own copy of
decoded_root + M3, worktree at the campaign HEAD, shipped bundle):
**p0/w/t0 bit-identical** (|Δ| = 0), χ² within 4e-9, all ok/quality/n_tracks
flags 100 % identical, M3 matching reproduces 8,479 events exactly.
`verify_rereco_desktop.py`; log `desktop:~/fleetcheck_data/verify.log`.

## 6 · Loose ends recorded

- ~2 % of golden det3 events carry absurd `q_sum` (> 1e6 up to 1e20; NNLS
  charge blow-ups). Median-based metrics are immune; flagged for the
  post-freeze queue.
- det6's bundle remains under suspicion (σ_s 165.9 ns vs det3's 12.1; v 26.7
  vs det7's 36.6 in the same run — see memory note); its **Y bias is now
  explained** by w0, but v/σ_s degeneracy is a separate open item. The
  report's det6 text no longer presents 26.7 µm/ns as a settled gas fact.
- The 6-23 det3 series (drift 600 V) has reco fraction ~0.4–0.6 and elevated
  shape χ² — the expected off-conditions v mismatch, now visible as such.
