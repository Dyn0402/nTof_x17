> # ⛔ RETIRED — do not build on this
>
> **Superseded by `../FINDINGS_2026-07-29_pre_ship_tests.md`.** Archived 2026-07-30.
>
> The work order for the pre-ship tests. It was run on 2026-07-29.
>
> **Read `../FINDINGS_2026-07-29_pre_ship_tests.md` for the results**, and note that two of its conclusions were themselves later retracted the same evening (there is no ADC wrap-around, and `satuflag` is good on the liquids) — see `../FINDINGS_2026-07-29_signed_decoding.md` and `../FINDINGS_2026-07-30_saturation_walls_plastics.md`.

---

# Pre-ship tests: what to run before the final UserInput goes to n_TOF

> **RUN 2026-07-29. Results in `FINDINGS_2026-07-29_pre_ship_tests.md`.**
> Summary: T1 confirms the headline on 2.5x the sample (96.3 % / 0.5 % over 252
> bunches). T5 says **drop the fast/slow boundary** -- `afast` fills but implies
> a tail fraction of 0.04-0.12 against a raw-measured 0.21 and drifts 2x with
> amplitude. T4 is **not closed**: the per-hit raw classification it asks for
> could not be made trustworthy, because PSA `tof` and the raw sample index do
> not align per hit. T6 holds on LIQA/C/D but **not LIQB**. T8 found that
> `satuflag` is never set on any wall. The text below is the original work
> order and is left unchanged.


**2026-07-29.** The candidate final UserInput is `v12_liqpileup`. Its wall and
plastic configuration is bit-identical to v11 (verified during v12 grading), so
every wall/plastic conclusion in `STATUS.md` carries to it unchanged.

This file is a work order for a follow-up session: each test is self-contained,
says what to run, what data it needs, and **what result forces which decision**.
Run them in the order listed; T4 and T5 are the ones the ship decision actually
waits on.

## Current verdict, per subsystem

- **Walls (SiPM): CONFIRMED.** Templates + AREA/AMP cuts win on every metric
  (chi2 0.85–1.06, wall-only matcher 98.9 %, quality guards flat). Remaining
  weakness is *sample size*, not evidence direction — see T1/T2.
- **Plastics (PSS): CONFIRMED.** Shape fitting is the one big win
  (95.2 → 96.4 %), and the floor argument is strong: v8/v10/v11 reconstruct
  the plastics very differently and land on identical efficiency, with the
  amp>2000 population identical to <1 %. The residual 2.5 % is analysis-side,
  not PSA-side. T1 firms up the headline number; nothing else is open.
- **Liquids: NOT SETTLED.** The *negative* results are solid (templates can't
  help — photon-statistics floor; raw waveforms won't yield more — measured
  0.67x). What is thin is the *positive* content of v12 itself:
  1. **(L1)** Are the +14–21 % extra hits from `STEP SIZE` 1/3 real pulses,
     or split/fake fragments? Only chi2-neutrality and the pileup flag speak
     to this so far — nobody has looked at the new hits against raw waveforms.
  2. **(L2)** Does the 5000/30 fast/slow boundary produce anything usable?
     `afast` fills but `aslow` stays 0 (slow component lies outside the pulse
     boundary), so the PSD motivation in `liq_study/FINDINGS_liquids.md` §5
     did **not** materialize. Keep-or-drop has not been decided on evidence.

## Data each test needs

| test | needs | where / how |
|---|---|---|
| T1, T2, T3 | processed partials (v4, v8/v11, v12, official) | `xrdcp` from `/eos/experiment/ntof/data/x17/reproc/<variant>/completed/224572/` (needs `ssh -K` / valid EOS auth) |
| T4 | raw stream1 chunks + v11/v12 partials | regenerate chunks with the loop in `REVIEW.md` §3 (~3 GB); raw ages off EOS — check `eos_stream1_inventory_2026-07-28.txt` first |
| T5 | v12 partials only | as above |
| T6 | `liq_study/liq_pulses.npz` (committed) | already in repo — check it carries LIQB/LIQC; if not, needs the T4 chunks |
| T7, T8 | same inputs as T4 / T5 | optional |

Traps that will bite a fresh session: `REVIEW.md` §6, especially *never merge a
run*, *caches keyed by run number only* (use the sandboxing in
`validate_reprocessing.py` / `dream_regression.py`), and *build the
DREAM↔bunch join before pointing the reader at candidate partials*.

---

## T4 — are the +14–21 % extra liquid hits real? **[BLOCKING]**

The `STEP SIZE` 2/4 → 1/3 change is the whole liquid yield claim. Finer
derivative windows resolve pileup but can also double-fire on noise or split
one pulse into two. chi2 stayed neutral-to-better and the pileup flag rose
+50 %, both consistent with real splits — but that is circumstantial.

**Method.** Regenerate the raw chunks. For matching (tree, bunch, time
window): diff v12 hits against v11 hits, take the v12-only set ("new hits").
Overlay a sample of ~300–500 new hits (across LIQA–D, weighted by yield gain)
on the raw waveform and classify each:

- (a) a distinct local rise ≥ ~5σ baseline within a few ns of the hit time —
  a real pulse the coarser step merged or missed;
- (b) a resolved shoulder on an existing pulse — a real split;
- (c) nothing there beyond an existing pulse's smooth tail or baseline — fake.

`liq_study/deconv_vs_psa.py` already contains the raw-block reader and
PSA-hit matching; reuse its I/O rather than rewriting it (and remember: every
correlate/interp alignment in this repo is suspect until checked —
`REVIEW.md` §5).

**Cheap proxies first** (no raw data needed, do these before downloading
anything): (i) amplitude spectrum of new hits vs pre-existing hits — a fake
population is pinned at low amplitude; (ii) time-of-flight profile of new
hits — real hits follow the physics rate profile, fakes are flat; (iii) what
fraction of new hits sit inside 150 ns of a pre-existing hit (splits) vs in
the open (recoveries). If the proxies look clean AND raw is unavailable, they
can carry the decision with a note; if raw is available, do the overlay.

**Decision.** ≥ ~85 % in classes (a)+(b) → **ship v12 as-is**. A large (c)
fraction → drop the `STEP SIZE` change (revert LIQ to v11's 2/4, keep the
boundary question T5 separate) and re-grade once. Ambiguous → ship v11's
liquid step with v12's boundary; the yield gain is not worth a fake-hit
contamination in a handoff we can't easily patch later.

## T5 — what does the 5000/30 boundary actually deliver? **[BLOCKING, small]**

**Method.** Run `liq_study/check_psd_output.py` on v12 partials. Measure:
`afast` fill fraction, `aslow` fill fraction (expected 0), and — the real
question — whether `afast` carries information: for *isolated late-time*
pulses, does `(area − afast)/area` reproduce the tight 0.21 tail band seen in
the raw-waveform PSD (`liq_study/liq_psd.png`, pulses > 3000 ADC)?

**Decision.**
- Band reproduced on the isolated subset → **keep** the boundary. It costs
  nothing and gives partial PSD on the isolated minority; document loudly in
  `ntof_handoff/` that `aslow` is NOT filled and `afast` is only meaningful
  for isolated pulses.
- `afast` is degenerate (≈ area for everything, no discrimination) → **drop
  the boundary from the shipped UserInput**. A filled-but-meaningless PSD
  field in the official campaign output is worse than an empty one — people
  will use it.

## T1 — matcher efficiency on a bigger sample [confirmatory]

The 96.4 % headline rests on 100 bunches of one DREAM sub-run (`REVIEW.md`
§4.1 calls this the thinnest sample under the biggest claim).

**Method.** `dream_regression.py` on more bunches of run_79, and/or on
stat090_0001. Build the DREAM↔bunch join for the whole run first. Grade v4
and v12 on the same bunches.

**Decision.** v12 ≥ ~96 % at ≤ 1 % false, with the gap to v4 preserved →
headline confirmed, quote the larger sample in the handoff report. Gap
collapses toward v4's 95.2 → the v8 win was a fluctuation of the small
sample; **stop the ship** and re-examine before sending anything.

## T2 — fit-chi2 on more partials [confirmatory, cheap]

chi2 comparisons used only partial 0016 (~20 bunches). Rerun
`compare_fits.py` on partials 0001+0002 (bunches 1–397): official vs v4 vs
v11/v12. **Need:** the *ordering* preserved (templates+fitting better). The
2x margins make a reversal unlikely; if the ordering flips on the larger
sample, the wall/plastic template claim needs re-grading before ship.

## T3 — sideband robustness in quality_metrics [confirmatory, cheap]

The accidental subtraction changes T2 from 38.8 → 6.46 ns, so the off-time
sideband is doing a lot of work. **Method:** move and resize the sideband
(±50 % width, two different offsets) and recompute T1/T2/A1; also recompute
on a late-time low-rate bunch subset where accidentals are small. **Need:**
widths stable to ~10–20 %. Instability would not change the UserInput (v11
beat v8 on chi2 and hit count independently of these guards) but would mean
the quoted timing numbers in `report/` need re-deriving before the handoff.

## T6 — photon-statistics floor on LIQB/LIQC [report robustness]

The sqrt(A) scaling was measured on LIQA/LIQD only. Run
`liq_study/is_it_photon_statistics.py` over LIQB/LIQC (from the committed
npz if it carries them, else re-extract from T4's chunks). **Need:**
`resid/sqrt(A)` flat to ~15 %. Confirms the "stop optimizing liquid
templates" claim covers the whole family; a violation on B/C would reopen
the template question for those two only.

## T7 — joint-refit deconvolution [OPTIONAL, non-blocking]

"PSA beats deconvolution 1.5x" used a simple greedy subtractor. If anyone
wants to close `REVIEW.md` §7's open item: add a joint least-squares
amplitude re-solve over all found pulses per iteration (matching pursuit
with refit) on the same 3 bunches. Only if it finds > ~1.2x the PSA's hits
with physical amplitudes does the raw-waveform question reopen. Expected
outcome: it does not. Do not let this block the ship.

## T8 — saturation census [documentation]

Both liquids break the sqrt(A) scaling at the ~31 000 ADC rail. Count the
fraction of liquid hits at rail in v12 output and state it in
`ntof_handoff/` so downstream users flag rather than fit them. One number,
one sentence.

---

## Decision summary

| outcome needed | test | if it fails |
|---|---|---|
| new liquid hits are ≥85 % real | T4 | revert LIQ STEP SIZE to 2/4, re-grade |
| `afast` carries the 0.21 band | T5 | drop the 30 ns boundary from the shipped UserInput |
| 96.4 % holds on a larger sample | T1 | stop the ship, re-examine v8 family |
| chi2 ordering holds | T2 | re-grade wall/plastic templates |
| guard metrics sideband-stable | T3 | re-derive report numbers (UserInput unchanged) |
| sqrt(A) floor holds on B/C | T6 | reopen templates for B/C only |

**Ship rule:** T4 and T5 decide the final UserInput content; T1 is the
go/no-go on the headline claim; T2/T3/T6/T8 gate the *report*, not the
UserInput; T7 is optional. When T1/T4/T5 are green, proceed with
`STATUS.md` "Next" step 1 (send `ntof_handoff/`).
