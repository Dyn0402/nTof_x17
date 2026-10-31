# det4's drift gap, geometrically (2026-11-01)

The question: the SPS gas fit profiles det4's gap at 27.7 ± 0.1 mm (stat; quoted at 27.9, one profile step = 0.9 mm),
while the bench input is 30.48 ± 0.44 mm (`rerun/calib/physics_inputs.json`), and S4 could not re-measure it on pass 1.
This measures the gap with no drift velocity and no pulse model: the strip extent L of an inclined track's column is
gap × |tan θ|.

## Verdict

**det4's gap is 28.6 ± 0.5 mm, not 30.5.** The bench Y view gives it with no x reference, on an estimator validated
on det3 and det2. The SPS rotated runs agree: their extent rises to a 28.1–28.9 mm plateau as the thresholds go down to
the ZS floor, in both gases. The SPS gas fit's gap (27.7–27.9 ± 0.9) is consistent with it, so the gas fit's water
stands. The bench input's 30.48 is the X view's stack estimator (`gap_extent.py`), and only on det4 does it disagree
with the Y view: by 1.9 mm (3σ). det4's X view crosses its dead amplification stripes.

| measurement | gap (mm) | note |
|---|---|---|
| **bench Y view, det4 pooled** (g_det4 + 6-23 long / longer, 217 events) | **28.9–29.0 ± 0.4** raw, **28.6 ± 0.5** after the validated +0.3 ± 0.2 offset | stable over three thresholds; g_det4 alone 28.5–28.8 ± 0.6, 6-23 long 29.5–29.8 ± 0.9 |
| validation: bench Y view, det3 (sat_det3) | 28.35–28.45 ± 0.27 | gap_extent (X) 28.0–28.2; S4 re-check 28.19 ± 0.19 |
| validation: bench Y view, det2 (o22) | 31.4–31.8 ± 0.7 | input 31.12 ± 0.42; S4 31.36–31.55 |
| validation: bench X view, det3 | 27.63–27.73 ± 0.55 (slope), intercept −0.09 mm | gap_extent 28.0–28.2 |
| SPS rot25, earliest-start plateau, thresholds at the ZS floor (90/30 ADC) | 28.1–28.9 | run57 d700 (CO₂) 28.2–28.5, run59 (CO₂) 28.1–28.2, run62 d700 (CF₄) 28.1, run63 d675 (CF₄) 28.9 / 27.9; a lower bound (see below) |
| SPS rot25, same, pass thresholds (150/60 ADC) | 25.2–26.9 | rises with field: 25.2–25.8 at 208–217 V/cm, 25.9–26.9 at 234–243 V/cm |
| SPS gas fit (RAW run71 χ² vs gap, timing + Magboltz) | 27.7 ± 0.1 stat, ± 0.9 (shaper) | `rerun/sps_gasfit/README.md` |
| bench input (X view, `gap_extent.py`, pre-pass reference) | 30.48 ± 0.44 (halves 30.96 / 31.13; satmask 30.59 / 30.57) | X view crosses det4's stripes |
| S4 on the pass reference (X view) | halves 33.8 / 30.0 | see "the S4 split" |

## Method

**Per event, in the projected view** (`gap_geom.py`, `bench_gap.py`; one estimator for SPS and bench):
1. The strips with a peak ≥ PK_MIN.
2. A Theil–Sen line of peak time against strip position, through the strips ≥ PK_LINE. Strips more than 4 samples off the
   line are pile-up or noise and count as zero charge.
3. A fit to the strips' window-integral charge: an attenuated box blurred by the footprint,
   q(y) = A·exp(k(y − a))·[Φ((y − a)/σ) − Φ((y − b)/σ)].
   - σ is the flat-mount footprint of the same gas (SPS run55 / run63 flat) or the bench head-on events.
   - The half-height edges a, b of a symmetric blur sit at the column ends whatever σ is.
   - The free k absorbs attachment and gain variation along the column.
4. L = |b − a|. Both edges must lie at least 1.6 mm inside the readout window in position.

**SPS:** the gap is L / tan(mount), with the mount at 25.52 ± 0.12° (config 25.64°, run57's alignment cos φ 25.4°).
- Cache: `sps_extent.py` runs the pass's own builder (`sps_build_cache.build_cache`: final oscillator and hot rules,
  code 48244778) at ±32 strips, so the 13–14 mm projected column fits.
- Rows: 3000 events per row, on 13 SPS rows on condor.

**Bench:** M3's tan per event; the gap is the median of L / |tan| and, as a check, the slope of L against |tan|.
- The Y view uses tracks inclined in y–z at one x (|tan_x| < 0.06; `bench_ycache.py`, M3 through the pass alignment,
  pass-1 decode).
- det4's dead stripes are x intervals, so such a column is live or dead as a whole.
- det4's X view cannot be used per event: there the estimator reads 15–18 mm (the column is cut at every stripe it
  crosses), against 27.7 on det3.

**Validation.** The same estimator reads det3 Y +0.25 and det2 Y +0.1 to +0.5 mm above their stack values; det3 X reads
−0.4. So +0.3 ± 0.2 is taken as its offset in Y.

## Why the SPS extent is a lower bound

- **The far end is faint:**
  - the fit's attachment along the column is k = −0.016 /mm in CF₄ and −0.033 /mm in CO₂ (in the gas fit's O₂ ordering);
  - a slow column spreads each strip's charge over more samples;
  - zero suppression removes the samples below ~30 ADC.

  So the cathode end goes below threshold first.
- **The tell is the field:**
  - at the pass thresholds the plateau rises with field (25.2 → 26.9 mm from 208 to 243 V/cm);
  - at the ZS floor the 234–243 V/cm rows of both gases meet at 28.1–28.9.
- **The window:**
  - late-starting events lose the far end to the 64-sample window (L falls with the start time: `by_t0` in the JSONs);
  - the quoted plateau is the earliest-start bins only;
  - the 32-sample rows (run61) are window-cut throughout: 12–14 mm, not used.
- **With RAW**, there is no rotated RAW row, so the floor can't be lowered further.

## The bench X view and the S4 split

**X view vs Y view.**
- The X-view stack (`gap_extent.py`) agrees with the Y view on det3 (28.0–28.2 against 28.4) and det2 (31.1 against 31.6).
- It disagrees on det4 only: 30.5–31.1 against 28.6, about 3σ.
- What det4's X view has and the others' don't is the dead amplification stripes, which an X-inclined column crosses.
  That `gap_extent`'s per-strip exposure normalisation doesn't fully remove them is the working explanation. It is not
  demonstrated.

**The S4 ± tan split (33.8 / 30.0 on the pass reference).**
- The pre-pass reference's halves agree (30.96 / 31.13), so the split comes from the reference, not the chamber.
- `headon_cache.ref_only` uses only the alignment's z and x sign (not its rotation or offsets); with the pass alignment
  (z_x 713, θ 90.05°) the mesh edge sits at z = −2.9 mm (−2.0 pre-pass).
- A ±tan split is what a tilt δ ≈ 0.015 in tan between the reference and the drift direction does. Which reference
  term produces it is not identified here.
- It doesn't change the verdict: the Y view uses no mesh position at all.

## What follows

- **The SPS gas fit's water stands.** At the nearest profiled gap to 28.6 (28.8 mm) the water is 0.10–0.19 % lower and the
  O₂ about 30 ppm lower:
  - CF₄ 1.57–1.70 % instead of 1.68–1.85 %;
  - CO₂ 1.75–1.94 % instead of 1.94–2.12 %.

  That is inside the gap term the gas fit already carries (±0.16 %). The ordering of the O₂ is unchanged.
- **det4's bench input gap (`physics_inputs.json` 30.48 ± 0.44) should be 28.6 ± 0.5.** It's not changed here.
  Everything that uses it would move by −6 %:
  - det4's timing v (S4 re-check 32.70 → ~30.7 µm/ns);
  - the det4 field labels (600 V / 28.6 mm = 210 V/cm, not 197);
  - the §6 det4 golden-row water (0.42 %, which needs a slower gas at a shorter gap, i.e. more water).

  The angles do not move: their scale is M3-calibrated per row.
- **The registry already labels SPS fields with 28.8 mm** (700 V → 243.2 V/cm). That is consistent with the geometric
  gap to 0.2 mm.

## Limitations

- **The bench Y sample is small** (217 events pooled), because M3 rarely gives a Y-inclined, X-flat track.
- **The validation offset is from two chambers.**
- **The SPS value is a lower bound.** Its agreement with the bench at the ZS floor is the evidence, not a measurement.
- **The geometric gap is an average over the illuminated area.** A dished cathode (det3's 25.7–29.2 mm map) would make it
  position-dependent; on det4's bench Y events the x and y halves agree to 1.1σ (28.6 ± 0.5 / 29.4 ± 0.6).

## Files

- **Scripts:**
  - `sps_extent.py`: wide-window strip charges from the pass's SPS builder;
  - `gap_geom.py`: the SPS estimator;
  - `bench_gap.py` and `bench_scan.py`: the bench X / Y estimator and its threshold scan on S4's cache;
  - `bench_ycache.py` and `bench_yscan.py`: Y-inclined bench caches and their scan;
  - `condor/`.
- **Results** (`results/`):
  - `sps_gap_v3_{a,b,c,d}.json`: pass thresholds, per row, with `by_t0`;
  - `sps_gap_lowth.json`: thresholds 90/30;
  - `bench_scan.json`: S4 cache, det3 / det4, X / Y;
  - `bench_yscan.json`: Y caches, det2 / det3 / det4.
- **EOS** (`/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/det4_gap/`):
  - `sps/*.npz`;
  - `bench/Analysis/<row>/gap/ycache.pkl`.
- **Condor:** `~/x17rerun/det4_gap/` on lxplus.
