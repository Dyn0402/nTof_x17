# PLAN 40 — Drift-gap / attachment story: three skeptic-hardening tests

**Paper point 7. Priority 4. The conclusion "30 mm mechanical but only ~23 mm recorded,
because H₂O-laden gas (with ~0.2 % O₂) attaches electrons from the top of the drift"
is defensible but rests on internal closure (campaign frozen, no gas probe). These
three tests are the cheapest remaining ways to close the holes a referee will find.
They are independent — do them in any order; 40a is the most important.**

Shared context: the evidence chain and its weak points are summarized in
`../PAPER_STATUS.md` topic 7. Key existing numbers: λ_att(1000 V) = 15.4 mm (det3),
recorded column v·T_sat ≈ 23.4 mm flat for ≥700 V, v_geom(1000 V) = 34.30 ± 0.28 µm/ns,
det2 λ ≈ 40 mm with the SAME v ≈ 34–35. Framing to defend: **water sets the velocity
(shared gas line), O₂ sets the attachment (per-detector budget).**

---

## 40a — Diffusion-only toy bound (kills the gain/attachment degeneracy argument)

**Referee attack:** "Your amplitude-vs-depth decay could be diffusion spreading charge
below threshold at long drift, not electron capture."

**Tool:** `25_signal_formation_toy.py` — standalone MC, no data needed. Verified knobs
(parsed from sys.argv as `--flag=value`): `--lam=` (LAMBDA_ATT, mm; attachment applied
as survival exp(−z/λ) per cluster, default 15.0), `--xt1=`, `--xt2=` (sharing),
`--dt=` (DT_DIFF, diffusion), `--vtrue=`, `--unshare`. The data-matched closure config
is `--xt1=0.35 --xt2=0.12 --lam=10` (runbook §7 last line).

**Method:**
1. Reproduce the baseline: run the closure config, extract the toy's median
   amplitude-vs-depth profile and fit λ_apparent over the same window used on data
   (z ∈ [8, 22] mm — the λ_att fit window, runbook §5). Confirm λ_apparent ≈ input λ.
   If script 25 does not already emit an amplitude-vs-depth profile, add it (follow
   how 17/19 build theirs: per-strip amplitude vs arrival depth z = v·t).
2. Attachment OFF: same config with `--lam=1e9`. Fit λ_apparent again. Scan diffusion
   `--dt=` from nominal to ~3× nominal, and the hit threshold if exposed (THR_HIT=100
   equivalent inside the toy — check the constants at the top of the script).
3. Deliverable number: **the maximum fake λ that diffusion+threshold alone can
   produce**. If even 3× diffusion gives λ_apparent ≳ 60–100 mm while data shows
   15 mm, the degeneracy is dead — one sentence + one panel in the paper.
4. Also record what killing attachment does to T_sat and the extent slope — the toy
   should show the recorded column growing back toward 30 mm without attachment,
   which is the positive statement ("attachment is what truncates the column").

**Acceptance:** step 1 closure (λ_apparent ≈ λ_input ± ~15 %) must pass before steps
2–3 mean anything.

**Gotcha:** the toy's `--lam=10` closure value vs the measured 15.4 mm — the toy's λ
is an effective per-cluster survival scale, not necessarily identical to the fitted
amplitude λ; that is WHY step 1 calibrates the mapping before step 2 uses it.

---

## 40b — Angle-binned decay (attachment must depend on z only)

**Referee attack:** "Amplitude loss might scale with track path length or slant, not
pure drift depth."

**Physics:** attachment survival is exp(−z/λ) — a function of drift DEPTH only. A
path-length artifact (dE/dx sampling, threshold-vs-slant) scales with 1/cosθ and would
make the fitted λ drift with track angle.

**Method:**
1. Take the amplitude-vs-depth machinery from `17_gap_attachment_test.py` /
   `19_amplitude_attachment_plot.py` (same run `sat_det3`, veto50 cache, depth scale
   z = v_geom·t with v from `geometry_vdrift_scan.csv`).
2. Split the track sample into 3–4 bins of |tanθ_ref| (e.g. 0.06–0.15, 0.15–0.3,
   0.3–0.55 — stay inside the standard ridge window 0.06–0.55; head-on tracks have no
   depth ladder, exclude them).
3. Fit λ per angle bin, per plane, over the standard z ∈ [8, 22] mm window. Use the
   MEDIAN amplitude per depth bin (as 19 does) to be robust to Landau tails.
4. Normalize each bin's profile at the shallow end before overlaying (per-bin
   normalization removes trivial dE/dx-per-strip differences; only the SLOPE matters).
5. Deliverable: λ vs |tanθ| plot with a flat-line fit. Flat within errors = attachment
   confirmed depth-only. A trend = quantify and discuss (some 1/cosθ leakage via the
   strip-level threshold is conceivable — if seen, check whether its size can be
   reproduced by the toy from 40a with attachment ON and angle-binned the same way).

**Acceptance:** the all-angle combined λ must reproduce the published 15.4 mm
(±1–2 mm) before trusting the per-bin splits.

**Gotcha:** per-strip amplitude depends on the track segment length per strip
(∝ pitch·√(1+tan²θ)/…) — that's an overall per-bin normalization, removed by step 4.
Do not fit λ across bins jointly without it.

---

## 40c — det2/det3 overlay: same v, different λ (the water/O₂ decomposition figure)

**Referee attack:** "Your out-of-sample check on det2 FAILED for attachment
(λ 40 vs 17 mm) — the gas model doesn't replicate."

**Response to build:** that difference is the point — velocity is a bulk-gas property
(shared gas line → same v everywhere), attachment is dominated by O₂ whose budget is
per-detector (volume, leak history). One figure makes the argument visually.

**Method:**
1. det3 curve: amplitude-vs-depth at 1000 V from the sat run (17/19 machinery,
   existing outputs `drift_velocity/mx17_3/gap_attachment_test.csv` may already hold
   the profile — check before recomputing).
2. det2 curve: same machinery on `o22_long_det2` (1000 V drift; caches exist —
   `29_det2_validation.py` already measured λ≈40 mm; check for an existing per-depth
   CSV under `.../mx17_2/...` before recomputing).
3. Overlay panel A: the two amplitude-vs-depth profiles (normalized at shallow z),
   with fitted λ = 15.4 vs ≈40 mm labeled.
4. Overlay panel B: the velocity agreement — det3 v_geom(1000 V) = 34.3 ± 0.3 and
   det2 v (geometry and/or unshared time-fit ≈ 35.0/35.3) on the Magboltz v(E) curves
   for Ar/iso + {0, 0.5, 1, 1.5} % H₂O (tables in `garfield_sim/results/` — the water
   grid JSONs; see `15_drift_velocity_vs_magboltz.py` for how they load).
5. Caption logic to encode: same v (water, shared) + different λ (O₂, per-detector) +
   Magboltz: water alone has η=0 (cannot attach) while ~0.2 % O₂ gives λ≈15 mm and
   ~0.1 % gives λ≈40 mm (pull the exact η values from
   `garfield_sim/results/attachment_*.json`; 18's code shows the conversion).
6. Optional strengthener if quick: per-plane λ (x vs y) for det3 as an internal
   systematic band on the 15.4 mm.

**Acceptance:** reproduced λ values match the published ones (15.4 / ≈40 mm) before
styling the figure.

**Gotcha:** det2 ran resist 525 V (different gain) — normalize profiles, never compare
absolute amplitudes. det2's depth scale must use det2's OWN v, not det3's.

---

## Outputs (all three)

- Scripts: `40a_toy_diffusion_bound.py`, `40b_angle_binned_attachment.py`,
  `40c_det2_det3_attachment_overlay.py` (or one `40_gap_skeptic_tests.py --part=a|b|c`).
- Figures + CSVs under `.../drift_velocity/mx17_3/skeptic_tests/` (40c also writes the
  overlay to a run-neutral place, e.g. `~/x17/cosmic_bench/Analysis/_combined_det3_weekend/`).
- Appended results section in this file with the three headline sentences:
  (a) "diffusion+threshold alone can fake at most λ = X mm ≫ 15 mm observed";
  (b) "fitted λ is angle-independent (slope consistent with 0)";
  (c) "det2 and det3 share v(E) but differ 2.5× in λ — water sets v, O₂ sets λ".
