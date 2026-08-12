# Efficiency slide — refreshed numbers, figures and replacement markup

**Written 2026-08-10.** Answers the three TODOs in the HTML comment above the
efficiency slide (slide 12) and the `Efficiency` entry of [`NOTES.md`](NOTES.md).

`index.html` is **not** edited here — the markup in §5/§6 is ready to paste.

---

## 0. The answer in four lines

* **All five numbers are refreshed.** Nothing had to be left at its 2026-07-14
  value. Cost: **~35 s of CPU**, all five chambers, because every input cache was
  already on disk.
* **det3 and det2 barely move** (92.9 → 93.5 %, 91.3 → 91.9 %). **det6, det7 and
  det4 move by 14–21 points** (57.8 → 75.4, 43.1 → 56.9, 20.7 → 41.9 %).
* So the old fleet bars are **wrong, not merely stale**, and the old caption is
  wrong in its physics too: det4 does *not* "see ~70 % of muons" — it sees
  **95.8 %**.
* The breakdown figure is regenerated with **no hardcoded percentages**; the
  "88.8 %" was a literal in the generator's annotation string, and both the
  conference generator and the engineer-package one now derive it.

---

## 1. Old vs new, with provenance

Efficiency = fraction of M3 reference muons crossing the active area whose
reconstructed detector point lies **within 5 mm** of the reference track.
Denominator and category definitions are identical in both columns —
`mx_june_wft/02_efficiency.py` is one accounting used for both reconstruction
chains, which is what makes the comparison legitimate.

| Chamber | Run key (sub-run) | rays | **OLD** 2026-07-14 | **NEW** 2026-08-10 | Δ | same accounting, old *hit-time* chain |
|---|---|---:|---:|---:|---:|---:|
| **det3 (A)** | `sat_det3` (6-27 saturday, resist 490 V / drift 1000 V) | 7,055 | 92.9 % | **93.5 %** | +0.6 | 93.1 % |
| **det2 (B)** | `o22_long_det2` (6-22 `longer_run`, 495 V / 1000 V) | 3,669 | 91.3 % | **91.9 %** | +0.6 | 92.1 % |
| **det6 (C)** | `g_det6_long` (6-26 `long_run`, 495 V / 700 V) | 9,626 | 57.8 % | **75.4 %** | **+17.6** | 71.2 % |
| **det7 (D)** | `g_det7_long` (6-26 `long_run`, 495 V / 700 V) | 9,429 | 43.1 % | **56.9 %** | **+13.8** | 52.7 % |
| **det4 (E)** | `g_det4` (6-24 `long_run`, 495 V / 600 V) | 12,259 | 20.7 % | **41.9 %** | **+21.2** | 40.7 % |

**Provenance of every NEW value — one command, one file each.**

```bash
cd /home/dylan/PycharmProjects/nTof_x17
for K in sat_det3 o22_long_det2 g_det6_long g_det7_long g_det4; do
    .venv/bin/python mx_june_wft/02_efficiency.py $K --max-dropped -1
done
```

Output file (the number quoted is the `within_R` field):

```
~/x17/cosmic_bench/Analysis/<run>/<sub_run>/<mx17_N>/wft/efficiency/efficiency_breakdown.json
```

| Chamber | output file |
|---|---|
| det3 | `mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/efficiency/efficiency_breakdown.json` |
| det2 | `mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2/wft/efficiency/efficiency_breakdown.json` |
| det6 | `mx17_det6_det7_overnight_6-26-26/long_run/mx17_6/wft/efficiency/efficiency_breakdown.json` |
| det7 | `mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/wft/efficiency/efficiency_breakdown.json` |
| det4 | `mx17_det4_day_6-24-26/long_run/mx17_4/wft/efficiency/efficiency_breakdown.json` |

The `_hits` variant of the same filename holds the last column.
`../.venv/bin/python mpgd26/make_efficiency_breakdown.py --print` prints the
whole table straight out of those JSONs.

**Provenance of the OLD values:** the 2026-07-14 changelog at the top of
`mx_june_cosmic_qa/JUNE_RESULTS_SUMMARY.md` and its §1 fleet table, written by
`09_efficiency_breakdown.py` on the hit-time chain under the χ² < 1.0 & NClus = 4
M3 recipe. That recipe is unchanged and still current (`qa_config.py`
`M3_CHI2_CUT = 1.0`, `M3_MIN_NCLUS = 4`, passed explicitly) — **the recipe is not
what moved.** See §3.

### Supporting numbers, all from the same JSONs

| | det3 (A) | det2 (B) | det6 (C) | det7 (D) | det4 (E) |
|---|---:|---:|---:|---:|---:|
| within 5 mm — **the headline** | **93.5** | **91.9** | **75.4** | **56.9** | **41.9** |
| within 10 mm | 94.5 | 93.1 | 76.3 | 60.9 | 43.6 |
| reconstructed at all | 97.3 | 96.3 | 78.8 | 66.3 | 49.3 |
| produced any signal (`has_any`) | 100.0 | 100.0 | 100.0 | 100.0 | **95.8** |
| detected, point > 5 mm off track | 3.7 | 4.4 | 3.4 | 9.4 | 7.4 |
| sparked during the crossing | 2.5 | 3.3 | 17.9 | 26.9 | 8.2 |
| fired, no valid X+Y point | 0.3 | 0.4 | 3.3 | 6.8 | **38.3** |
| silent (genuine blindness) | 0.00 | 0.00 | 0.00 | 0.00 | 4.2 |
| spark fraction, all firing events | 8.2 | 9.7 | 22.3 | 37.4 | 9.8 |
| core σ\|r\| [mm] | 0.46 | 0.44 | 0.43 | 0.64 | 0.67 |

(Percentages of active-area crossings except the last two rows. The five
categories sum to 100 by construction.)

---

## 2. What is *not* refreshed — say so on the slide, not in the notes

| Quantity on or near this slide | Status |
|---|---|
| The five within-5-mm efficiencies | **refreshed**, §1 |
| det3's loss budget (all five categories) | **refreshed**, §4 |
| The 10 mm recovery figure | **refreshed and measured** — 94.5 % on det3, cross-checked two independent ways (§4) |
| Spark fractions in the caption | **refreshed** — det6 22.3 %, det7 37.4 % of firing events (the old slide's "23 % / 33 %" were % of *crossings* on the old chain; both conventions moved, so quote the convention) |
| det4's "sees ~70 % of muons" | **refreshed and now wrong by 26 points** — `has_any` = 95.8 %. Must change. |
| **HV-scan peak efficiencies** (det6 76.2 % @ 480 V, det7 63.1 % @ 440 V, det2 89.9 %, det3 90.8 %) | ⛔ **could not re-derive — still 2026-07-14 hit-chain values.** Not used in the markup below, and must not be mixed with §1. Cost to refresh in §7. |
| `angular_resolution.png` / the 1.66° σ_θ on the resolution slide | ⛔ not this task, but flagging it: superseded on the same basis change (det3 σ_θ is now **1.08 / 1.11°**, not 1.66°). Separate refresh. |
| det3 on `g_det3_wknd` (the run the old 92.9 % was measured on) | ⛔ **could not re-derive; the headline run is changed instead.** Reasons in §3.3. |

---

## 3. What changed, and why

### 3.1 Two real changes, in this order

1. **The June cosmics were fully reprocessed on the matched-filter waveform
   analyzer** (`REPROCESSING_2026-07-24.md`, 139 sub-runs / 825 files, 0 failures,
   ~+40 % hits, biggest gain on the low-gain chambers). This is why det4/det6/det7
   move most: they were losing low-amplitude hits the old analyzer never admitted.
   The first attempt at scoring it *regressed* position badly
   (`RERUN_RESULTS_20260725_011307.md`: det3 93.4 → 84.1 %); the cause was residual
   coherent noise surviving CNS, which both wrecked the cluster anchor and faked
   the > 50-strip discharge tag. The fix (`DET3_RECO_FIX_2026-07-25.md`) is a
   per-plane **relative** significance floor, keep strips with
   `significance ≥ 0.10 × plane max`, and it is in every cache used here — verified,
   not assumed: all five `cache/event_results_veto50.meta.json` read
   `{"sigrel": 0.1, "veto": 50}`.
2. **Position now comes from the waveform-first forward fit, not from hit times**
   (`RECONSTRUCTION_BASIS.md`, decided 2026-07-28). A per-strip hit time on these
   resistive-strip detectors aggregates delayed copies of the neighbours' charge;
   an efficiency defined by *where the point landed* is a position measurement and
   so belongs to the waveform basis. Detection (`has_any`) deliberately stays
   hits-defined — whether the chamber fired is a property of the analyzer's
   trigger, not of the fit.

The last column of §1 separates the two: the reprocessing + floor does almost all
of the work, and the change of basis adds **+0.4 to +4.2 points** on top (det6
71.2 → 75.4, det7 52.7 → 56.9). Both effects push the same way.

### 3.2 Reasons to believe this and not the old numbers

* `02_efficiency.py` **reproduces the old chain's published 93.4 % / 0.48 mm on
  det3 as 93.13 % / 0.448 mm** when pointed at the hits — that agreement is what
  validates the accounting, and it is the reason the last column of §1 is a fair
  comparison rather than two different definitions.
* Every number in §1 **bit-reproduced** when I re-ran all five today against the
  JSONs stored on 2026-07-29…31, so nothing has drifted since the campaign closed.
* det6's efficiency is **insensitive to the calibration choice**: its rejected
  RC-ladder generation (`wft/lp_attempt_20260731/`) gives 74.86 % against the
  adopted legacy kernel's 75.41 %. det6 is the one chamber deliberately left on
  the legacy kernel — for its *angle* resolution, not its efficiency.
* det4's efficiency is likewise **calibration-independent** across four bundles
  fitted from completely different constraints: 41.65 / 41.69 / 41.90 / 41.93 /
  41.96 % (`BEAM_CONSTRAINED_CALIB_2026-08-05.md` §"Reconstruction A/B/C/D", which
  states it outright: *"Position is at parity everywhere — within-5-mm and core σ
  do not care which calibration is used"*). Note the local mirror still carries the
  07-31 `calib_bundle_lp` generation, not the 08-05 `calib_bundle_beamv34` that was
  promoted on the other machine; at parity, this does not affect the quoted 41.9 %.
* The M3 reference recipe is **unchanged** — χ² < 1.0 & NClus = 4, and
  `02_efficiency.py` imports `M3_CHI2_CUT`/`M3_MIN_NCLUS` from `qa_config` and
  passes both explicitly to `M3RefTracking(...)` (line ~106), as CLAUDE.md
  requires. So none of the movement in §1 is a reference-cut artefact.

### 3.3 The headline det3 run changes — and why that is the honest option

The old 92.9 % was measured on **`g_det3_wknd`** (6-27/28 weekend, 22.4 k rays).
That run cannot be refreshed cheaply and must not be quoted as-is:

* Its hit-chain cache is from the **broken** 2026-07-25 rerun — it has **no
  `.meta.json`**, i.e. it predates the significance floor, and its on-disk
  `efficiency_breakdown.txt` reads **82.3 %**. Anyone regenerating a figure from
  that file today gets 82.3 % on the bars.
* Its waveform reconstruction is a **partial** one: `events.meta.json` has 12,175
  events against ~22.4 k rays, because the parquet was produced for the drift-gap
  consistency campaign, not for an efficiency. I ran `02_efficiency.py g_det3_wknd`
  to check and it returns 50.6 % with 44.6 % `hit_no_reco` — that is missing
  coverage, not a measurement. **I deleted that output** rather than leave a
  50.6 % file on disk to trap the next reader.

So the refreshed det3 headline is **`sat_det3`** (6-27 saturday, 7,055 rays) at
**93.5 %** — the run that `ANALYSIS_STATE_2026-07-31.md` and `fleet_state.py`
already treat as the golden det3 dataset, on the same chamber, same slot, same
gas, same operating point, one day apart. Smaller sample, current basis. Slide
title *"93 % on the best chamber"* survives unchanged.

---

## 4. The regenerated figures

### New generator — `mpgd26/make_efficiency_breakdown.py`

```bash
cd mpgd26 && ../.venv/bin/python make_efficiency_breakdown.py
```

Writes both files below directly into `slides/assets/img/`, reading **only** the
`efficiency_breakdown.json` reductions. No bulk data, no PDF conversion step. Also
`--print` for the §1 table and `--only breakdown|tail`.

| file | what it is |
|---|---|
| `assets/img/efficiency_breakdown.png` | det3's loss budget — 93.5 / 3.7 / 2.5 / 0.3 / 0.00 %, 7,055 muons, plain-language row labels |
| `assets/img/efficiency_residual_tail.png` | **new** — the \|r\| distribution on log counts to 30 mm with the 5 mm match marked, beside efficiency-vs-match-radius |

### The stale "88.8 %" is fixed at the source, on both paths

The old figure came from `engineer_package/make_efficiency_breakdown.py`, whose
annotation string contained the efficiency **as a literal**:

```python
f'The two biggest "losses" off the 88.8% are NOT the chamber\n'   # ← the bug
```

The bars were parsed from disk, the annotation was not, so when the M3 recipe
changed on 2026-07-14 the two disagreed by four points and the figure shipped
that way. Both generators now derive it (`{cats["reco_near"][1]:.1f}%` /
`{d["within_R"]:.1f}%`), and the new one carries a comment saying why no number
in that box may ever be a literal again. The engineer-package script also gained a
⛔ header recording that its `BREAKDOWN_TXT` input is the broken-rerun file (§3.3)
and must be re-derived before it is run.

The unverifiable claim went too. The old box asserted *"at a 10 mm match the
efficiency recovers to ~95 %"* with no source. It is now **measured**: 94.5 %,
by two independent routes that agree exactly —

```bash
.venv/bin/python mx_june_wft/02_efficiency.py sat_det3 --max-dropped -1 --r 10
#   within 10.0mm  94.53 %
```

and the `eff_vs_R["10.0"]` field of the R = 5 JSON, which is the one the figure
reads. `eff_vs_R` and the `r_hist_*` arrays are a small reduction I added to
`02_efficiency.py` for exactly this purpose: same `rlist`, same denominator as
`within_R`, so a figure built from the JSON **cannot** disagree with the
breakdown. `eff_vs_R[str(R)] == within_R` by construction. The existing outputs
were unaffected — all five `within_R` values re-ran identical.

### Recommendation on the residual pairing (TODO item 2)

**Generate a tail-focused figure — do not reuse or crop `spatial_residuals.png`.**
Three reasons:

1. **Basis mismatch.** `spatial_residuals.png` is `engineer_package/figures/10-…`,
   a 2026-07-14 hit-chain figure. Pairing it with a waveform-first breakdown puts
   two incompatible reconstructions side by side on one slide, which is exactly the
   provenance problem this pass exists to remove.
2. **Wrong range.** Its log panels run to ±400 mm, so their visible tail is the
   pathological one. The slice being explained is 5–15 mm, which is one bin wide
   there.
3. **The recovery curve is the actual argument.** "Detected but > 5 mm off track"
   is only benign if those muons are *near*-misses. The right panel shows the
   efficiency climbing 93.5 → 94.5 % from 5 to 10 mm and then flattening well below
   the 97.3 % reconstructed-at-all line — so the 3.7 % is mostly genuine
   near-misses plus a thin real tail. That is a one-glance answer to the obvious
   question from the floor, and no crop of the old image gives it.

The wft chain does have its own residual figure
(`…/mx17_3/wft/alignment/residuals.png`, 6 panels, current basis) if a
conventional X/Y residual plot is wanted instead — but it is an alignment
diagnostic, over-dense for a 15-minute talk.

---

## 5. Replacement markup — main slide (det3 alone)

Replaces the whole `<!-- 12: B6 efficiency -->` section, and the TODO comment
above it. Same class vocabulary as the rest of the deck.

```html
    <!-- 12: B6 efficiency.  Refreshed 2026-08-10 on the waveform-first chain --
         see slides/HANDOFF_efficiency.md for old-vs-new and provenance.  The
         five-chamber fleet comparison moved to the backup slide "The fleet, and
         what limits each chamber".  Both figures come from ONE command,
         mpgd26/make_efficiency_breakdown.py, which reads only the JSON written by
         mx_june_wft/02_efficiency.py -- no number on this slide is hand-typed
         into a figure. -->
    <section class="slide">
      <div class="kicker">Characterization</div>
      <div class="title">Efficiency: 93% on the best chamber</div>
      <div class="cols cols-2">
        <div class="figure">
          <div class="imgwrap"><img src="assets/img/efficiency_breakdown.png" alt="det3 loss budget: 93.5% reconstructed within 5 mm, 3.7% detected but off track, 2.5% spark coincidence, 0.3% no valid point, 0.0% silent"></div>
          <div class="fig-label">Every crossing muon accounted for &mdash; the residue off 93.5% is a spark coincidence and a near-miss tail, never a failure to detect</div>
        </div>
        <div class="figure">
          <div class="imgwrap"><img src="assets/img/efficiency_residual_tail.png" alt="Left: log-scale distribution of detector-to-reference distance with the 5 mm match marked. Right: efficiency rising from 65% at 1 mm to 94.5% at 10 mm and flattening below the 97.3% reconstructed-at-all line."></div>
          <div class="fig-label">The &ldquo;off-track&rdquo; slice is near-misses: opening the match 5&nbsp;&rarr;&nbsp;10&nbsp;mm recovers 93.5&nbsp;&rarr;&nbsp;94.5%, then it flattens</div>
        </div>
      </div>
      <div class="caption">det3, 7,055 reference muons (6-27 run, resist 490&nbsp;V / drift 1000&nbsp;V). Efficiency = a reconstructed point within 5&nbsp;mm of the M3 reference track, &chi;&sup2;&lt;1.0 &amp; NClus=4 reference recipe; position from the waveform-first fit, detection from the hits. Genuine blindness is <b>0.00%</b> &mdash; the chamber fired on all 7,055. The other four chambers are in backup.</div>
    </section>
```

**If a headline stat tile is wanted instead of the second figure** (a tighter
option for a 15-minute slot — drop `efficiency_residual_tail.png` to backup and
keep the breakdown full-width):

```html
      <div class="figure-solo"><img src="assets/img/efficiency_breakdown.png" alt="det3 loss budget"></div>
      <div class="stat-row">
        <div class="stat"><div class="num">93.5%</div><div class="lbl">within 5&nbsp;mm of the reference track</div></div>
        <div class="stat"><div class="num">100%</div><div class="lbl">of crossings produce a signal &mdash; zero genuine blindness</div></div>
        <div class="stat"><div class="num">0.46&nbsp;mm</div><div class="lbl">core position residual (reference-limited, see next slide)</div></div>
      </div>
```

---

## 6. New backup slide — the fleet

Place with the other characterization backups, after the `Backup` divider.
Bar widths equal the values, as in the current markup.

```html
    <!-- Backup — the five-chamber fleet comparison, moved off the main
         efficiency slide 2026-08-10.  ALL FIVE NUMBERS REFRESHED on the
         waveform-first chain; the previous 92.9/91.3/57.8/43.1/20.7 set was
         2026-07-14 and is superseded by 14-21 points on the last three
         chambers.  Provenance: slides/HANDOFF_efficiency.md §1. -->
    <section class="slide">
      <div class="kicker">Backup &middot; Characterization</div>
      <div class="title-sm">The fleet, and what limits each chamber</div>
      <div class="cols cols-2">
        <div class="figure">
          <div class="bar-chart">
            <div class="bar-row"><div class="bar-name">det3</div><div class="bar-track"><div class="bar-fill" style="width:93.5%"></div></div><div class="bar-val">93.5%</div></div>
            <div class="bar-row"><div class="bar-name">det2</div><div class="bar-track"><div class="bar-fill" style="width:91.9%"></div></div><div class="bar-val">91.9%</div></div>
            <div class="bar-row"><div class="bar-name">det6</div><div class="bar-track"><div class="bar-fill" style="width:75.4%"></div></div><div class="bar-val">75.4%</div></div>
            <div class="bar-row"><div class="bar-name">det7</div><div class="bar-track"><div class="bar-fill" style="width:56.9%"></div></div><div class="bar-val">56.9%</div></div>
            <div class="bar-row"><div class="bar-name">det4</div><div class="bar-track"><div class="bar-fill" style="width:41.9%"></div></div><div class="bar-val">41.9%</div></div>
          </div>
          <div class="bar-note">Reconstructed within 5&nbsp;mm of the M3 reference track (&chi;&sup2;&lt;1.0 &amp; NClus=4). One high-statistics sub-run per chamber, 3.7k&ndash;12.3k reference muons, waveform-first reconstruction, 2026-08-10.</div>
        </div>
        <div class="figure" style="justify-content:center;">
          <table class="spec-table">
            <tr><td class="k">det3 / det2</td><td>Healthy. <b>100%</b> of crossings fire; the whole residue is a 2.5&ndash;3.3% spark coincidence and a 3.7&ndash;4.4% near-miss tail.</td></tr>
            <tr><td class="k">det6 / det7</td><td><b>Spark-limited</b> &mdash; 22.3% / 37.4% of firing events are &gt;50-strip discharges. Both ran at resist 495&nbsp;V, <i>above</i> their measured optima (~480 / ~440&nbsp;V): this is a chosen operating point, not a ceiling.</td></tr>
            <tr><td class="k">det4</td><td><b>Gain-limited</b>, and it is not blind: it produces a signal on <b>95.8%</b> of crossings, but on 38.3% too few strips fire to form a valid X+Y point. Independently confirmed at the SPS: 62% of its area does not amplify.</td></tr>
            <tr><td class="k">Position core &sigma;</td><td>0.43&ndash;0.46&nbsp;mm on det2/det3/det6, 0.64/0.67&nbsp;mm on det7/det4 &mdash; the good chambers are at the reference telescope's own floor.</td></tr>
          </table>
        </div>
      </div>
      <div class="caption">Superseded 2026-07-14 values were 92.9 / 91.3 / 57.8 / 43.1 / 20.7%. The last three chambers gain 14&ndash;21 points from the matched-filter reprocessing of the raw waveforms &mdash; low-amplitude hits the previous analyzer never admitted &mdash; plus a further 0.4&ndash;4 points from reconstructing position from the waveforms instead of the strip hit times.</div>
    </section>
```

**Do not add the HV-scan peaks to this slide** (det6 76.2 % @ 480 V etc.). They
are 2026-07-14 hit-chain numbers and would sit next to refreshed ones. The
wording above says the operating point was above the optimum without quoting an
unrefreshed number.

---

## 7. Cost of the remaining work

| Job | Cost | Worth it? |
|---|---|---|
| Everything in §1 and §4 | **~35 s CPU**, done | — |
| Refresh **`g_det3_wknd`** so det3 can be quoted on the 22.4 k-ray run: rebuild the hits caches (`03_alignment_and_tpc.py g_det3_wknd --refit` and `--no-veto`, ~10 min each) then a full waveform reconstruction (~13–19 min on 8 jobs) | **~45–60 min** | **No.** `sat_det3` is the same chamber, slot, gas and operating point one day later, and is already the campaign's golden det3 run. This would buy 3× statistics on a number whose error is ~0.3 %. |
| Refresh the **HV-scan peaks** on the current basis: 6–8 resist-HV sub-runs × 4 chambers, each needing its own reconstruction pass, plus per-point alignment | **~10–15 h wall**, condor-shaped | Not for this talk — the peaks are not on the slide. Do it if an HV-optimisation slide is ever wanted. |
| Refresh **`angular_resolution.png`** and the resolution slide's σ_θ onto the waveform basis (det3 1.66° → 1.08/1.11°) | reductions already exist (`03_angles.py` outputs, `fleet_state.py`); ~1 h of figure work | **Yes, separately** — the resolution slide currently quotes a superseded number, and its own TODO already flags a second problem (the M3 reference floor is not deconvolved). |
| Refresh the **efficiency maps** (`08`/`12`) if a 2-D map is ever put on a slide | already exist on the current basis: `…/mx17_3/wft/maps/efficiency_r_10_mm_waveform_first.png` | free |

---

## 8. Files touched by this pass

| File | Change |
|---|---|
| `mpgd26/make_efficiency_breakdown.py` | **new** — the deck's two efficiency figures, derived entirely from the analysis JSONs |
| `mpgd26/slides/assets/img/efficiency_breakdown.png` | regenerated, self-consistent, waveform-first `sat_det3` |
| `mpgd26/slides/assets/img/efficiency_residual_tail.png` | **new** |
| `mpgd26/slides/NOTES.md` | provenance table row replaced; the `Efficiency` open item closed with the numbers |
| `mpgd26/slides/HANDOFF_efficiency.md` | **new** — this file |
| `mx_june_wft/02_efficiency.py` | added `eff_vs_R` + `r_hist_edges`/`r_hist_counts` to the JSON so figures can be built from the reduction alone. Additive; all five `within_R` values re-ran identical. |
| `mx_june_cosmic_qa/engineer_package/make_efficiency_breakdown.py` | the hardcoded `88.8%` and the unsourced `~95%` removed from the annotation; ⛔ header added recording that its input file is the broken-rerun one |
| `…/mx17_det3_p2_det1_overnight_6-27-26/…/mx17_3/wft/efficiency/` | **deleted** — a partial-coverage 50.6 % breakdown I generated while checking §3.3; removed so it cannot be mistaken for a measurement |

`mpgd26/slides/index.html` was **not** touched.
