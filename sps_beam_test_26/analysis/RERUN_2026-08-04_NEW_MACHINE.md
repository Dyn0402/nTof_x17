# run_71 rerun on the new machine — what was lost, what was recovered, and the tilt is 0.9°, not 0.4°

The campaign desktop is gone; this documents rebuilding the run_71 RAW chain
from EOS + the repo on a fresh machine, 2026-08-04 (evening). Everything below
is reproducible from what is now committed — that was the point.

## 1. Recovered: the RAW decoder patch (it existed nowhere)

The flush-on-eventID decoder fix (`RAW_RUN71_STATUS.md` §3b) had been left
*uncommitted in the working tree of the campaign machine*: not on GitHub, not
on mx17-daq (checked — its clone is clean at `70c7c1d`). Reimplemented from
the §3b description and validated against every number in that section:

| target (doc) | reimplementation |
|---|---|
| group 023: 17,696 entries | 17,696 |
| eventId steps +1 only | +1 ×17,695, nothing else |
| duplicated (channel,sample) 0/3000 | 0/3000 |
| max blocks/entry 512 | 512 |
| acceptance 0.765, flat | 0.7659, spread 0.051 |
| ZS regression 122,350 entries, identical | identical, all branches |

Now committed AND pushed: `mm_strip_reconstruction` `913af62`, together with
the `decode_stats` tree + `sample_acceptance` histogram it writes into every
output (`decode_loss_report.py` reads them back unchanged). Closes item 3 of
`RAW_RUN71_STATUS.md` §4.

## 2. Restaged: EOS is sufficient, banco is not reachable

* FEU3 raw fdf (groups 002–007, 023–035), FEU1 raw fdf, FEU1 combined_hits,
  pedestal set, `dream_daq.log`, `hv_monitor.csv` — all pulled from
  `root://eospublic.cern.ch//eos/experiment/ntof/data/x17/p2_sps_july/`.
* FEU1 was **re-decoded from raw with the patched decoder** — banco's
  `combined_hits` for run_71 are pre-fix (merged events) *and* carry ZS
  analyzer flags on RAW data; `pair_dataset.py` already prefers the own
  decode, which now exists.
* banco itself (128.141.41.210) times out, direct and via lxplus — likely
  powered down post-campaign.

**Casualties of the machine loss, found so far** (each was a file that lived
only on the data disk):

1. `robust_waveforms.py` / `kernel_refit_clean.py` from
   `reanalysis_2026-08-04/` — **recreated, in the repo this time** (see §3).
2. `stripes_g_det4.npz` — rebuilt (bands only) from the committed
   `stripes_g_det4.json`; the c/med profiles need `04_stripe_metrics.py` +
   the June bench hits from EOS if ever wanted again.
3. `urw_mapping/mapping_urwell.csv` — **recovered later the same evening**:
   the campaign laptop regenerated it from its analysis code
   (`record_mapping_alignment.py`) and pushed it to the repo
   (`analysis/urw_mapping/`, commit 793e541, merged here as 37248a7). The
   independent det4-correlation done on this machine while it was missing
   agrees with it wherever the beam gave signal (front ch 64–127 and back
   ch 256–319: +1.00 mm/ch onto the same det4 coordinate — front-x/back-x
   in the csv). With it, `pair_dataset.py` and `flat_align_eff.py` run:

   | plateau | clean | \|res\| [mm] | fired | within 5 mm | in-band |
   |---|---:|---:|---:|---:|---:|
   | raw700 | 2191 | 0.64 | 68.1 % | 41.5 % | 57.6 % |
   | raw450 | 2125 | 0.67 | 69.0 % | 39.9 % | 53.9 % |
   | raw275 | 2193 | 0.73 | 60.1 % | 29.5 % | 42.1 % |

   Alignment roll +89.50°, det(A) +0.9972 — a proper rotation, and the
   efficiency ladder falls with drift voltage as it must.

## 3. The clean chain, recreated — what reproduces and what does not

`robust_waveforms.py` (clean gates → per-sample median / trim20 library, in
absolute window time AND peak-aligned, per-event normalised by the central
peak) and `kernel_refit_clean.py` (cascade refit + charge budget). Selection:
leading strip per view, 400–3000 ADC, ch 510/372 never central, |pre-window
mean| < 15 ADC, per-strip pre-level subtracted. Two lessons re-learned on the
way, both now encoded in the scripts' comments:

* **The central strip is the leading strip, not the rounded centroid** — the
  centroid rounds onto the second strip when charge shares, and the "±1"
  median then contains the true maximum (first attempt gave ±1 peak = 0.44).
* **In absolute window time the central peak is washed out** (median-trace
  peak ≈ 121 ADC from 400–3000 ADC events): at low drift field each event
  peaks wherever a large cluster lands on the ladder. Peak-aligned,
  per-event-normalised traces are the only meaningful basis for the kernel
  and the budget.

Against `RAW_RUN71_REANALYSIS_2026-08-04.md` (whose scripts are gone):

**Reproduces:**

| observable | doc | this rerun |
|---|---|---|
| event-wise ±1 peak shift (Y) | +54–61 ns | **+60 ns**, all three plateaus |
| charge budget, area ±1 | 0.71–0.77 | 0.74–0.75 |
| area ±2 | 0.40–0.48 | 0.39–0.40 |
| area ±3 | 0.15–0.18 | 0.15 |
| central share of 7-strip integral | ~27 % | 28 % |
| peak ±2 | 0.06–0.08 | 0.058–0.063 |
| drift-invariance verdict | passes (τ ±0.6 %, c1 ±2 %, c2 ±6 %) | **passes: τ ±3.7 %, c1 ±1.8 %, c2 ±3.5 % — and now across all THREE fields incl. 700 V** |
| central response contained, undershoot after | yes | yes (returns <5 % by 2.1–2.4 µs, Y) |
| 700 V end-of-ladder lobe | −0.30 of peak ~3.1 µs | present, last-4 −28 % |
| 275 V most negative-sagged, no lobe | yes | yes |
| X view tail-contaminated, quote Y | yes | yes (+19 % X tails) |

**Does not reproduce exactly** (recipe details lost with the scripts):

* cascade τ_s: 850 ± 30 ns here vs 1308/1316 ns; c1 0.52–0.54 vs 0.63; c2
  0.43–0.46 vs 0.62–0.66; ±1 peak 0.31–0.32 vs 0.16–0.19; last-4 level −11 %
  (450 V) / −23 % (275 V) vs −3.8/−6.1 %. These all sit downstream of exact
  fit-window, alignment and normalisation choices. The doc itself already
  ruled that the cascade parameters are **not physical charge fractions and
  must only be quoted through a 2-D RC-sheet model fitted to the library**;
  that conclusion is unchanged, and the invariance verdict — the thing the
  run was taken for — is basis-independent and confirmed. The absolute
  numbers to carry forward are the *library* (npz, in
  `staging/run_71/reanalysis_clean_cmmasked/`), not the cascade betas.

Also checked and rejected while chasing the tails: common-mode signal bias.
A signal-masked CM (`extract_det4_only.py --cm masked`, strips within ±10 of
either lead excluded from the block median) changes the library negligibly;
`--cm none` is unusable in beam (the CM wanders within the 3.84 µs window,
so no pre-level can absorb it — consistent with the ZS-study finding).

## 4. The tilt, redone clean (§6 open item) — the old magnitude was a v_drift artefact

`tilt_clean.py`, centroid walk on the clean selection, v from the data
(end-lobe v(233 V/cm) = 14 µm/ns, v ∝ E):

| view | plateau | tan θ | θ |
|---|---|---:|---:|
| X | raw700 | −0.0160 | 0.91° |
| X | raw450 | −0.0134 | 0.77° |
| X | raw275 | −0.0166 | 0.95° |
| Y | raw700 | +0.0050 | 0.29° |
| Y | raw450 | +0.0054 | 0.31° |
| Y | raw275 | +0.0030 | 0.17° |

* **tan θ is drift-field-invariant across a 2.5× velocity range** — the walk
  is geometric, not an electronics or drift artefact. This is the internal
  check the contaminated measurement could never do.
* **The historical "0.2–0.4°" was the same slope divided by the DRY-gas
  v = 34 µm/ns** (tilt_m70V.py's default): −0.22 µm/ns / 34 → 0.37°,
  exactly the old range. With the measured wet-gas velocity the same walk is
  **θ_X ≈ 0.9°** (≈ 2 mm across the 130 mm plate — mechanically plausible).
  The sign (negative in X) is unchanged from the three historical
  measurements.
* Y is not perfectly clean either: **θ_Y ≈ +0.2–0.3°**, consistent sign
  across plateaus. Small enough to keep quoting Y for the kernel; large
  enough to state.
* The ±1 arrival-time antisymmetry estimator is unusable at 60 ns sampling
  (medians quantise to whole samples); the centroid walk is the estimator.

**Consequence for the record:** every past tilt magnitude quoted for det4
scales with the v_drift assumed at the time. The invariant to carry is
**tan θ_X = −0.015 ± 0.002** (this measurement); any angle quoted from it
must state its v.

## 5. State of the chain on this machine

```
staging/run_71/  fdf (FEU1+FEU3) · dec_/hits_ (patched decoder, both FEUs)
                 wf_run71_raw_det4only{,_cmmasked,_nocm}.npz
                 reanalysis_clean_cmmasked/robust_library_run71_raw.npz
run_63/operating_03/  group 004 + pedestals (ZS decoder-regression pair)
```

`pair_dataset.py` runs and pairs — `mapping_urwell.csv` was recovered the
same evening (§2.3), and the paired chain results are in §2.3's table.
Everything else in the read-order list of `README.md` stands.
