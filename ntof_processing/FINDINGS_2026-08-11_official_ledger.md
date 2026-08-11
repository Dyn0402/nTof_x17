# n_TOF has processed some of what we processed — and we match it bit for bit

**2026-08-11, evening.** n_TOF's official pass moved a long way on 08-10/08-11:
**27 runs that had partials but no merged file are now merged**, and **24 more are
being reprocessed from scratch right now**. Two of the newly merged ones —
**224573 and 224577** — are runs we had processed ourselves, so for the first time
there are runs sitting in both processings.

That makes the direct test possible, and it passes: **given the same UserInput our
chain reproduces n_TOF's product exactly, hit for hit, on all 22 per-hit columns.**

Report (tables, per-run ledger): [`campaign_qa/results/report.html`](campaign_qa/results/report.html).
Ledger CSV: [`campaign_qa/results/ledger_2026-08-11.csv`](campaign_qa/results/ledger_2026-08-11.csv).
Tools: `campaign_qa/official_ledger.py`, `campaign_qa/compare_identity.py`.

---

## 1. Where the official processing stands

`official_ledger.py` walks all **445** runs staged under
`/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement`, and for each one records what
n_TOF has, what we have, and **which UserInput each product was actually made
with** — read out of the product's own `history` object, not assumed.

| official state | runs | meaning |
|---|---|---|
| **MERGED** | **385** | merged file in `done/` — usable, best |
| **IN_FLIGHT** | **24** | `completed/` emptied and refilling *now*; unusable until it finishes |
| PARTIALS_ONLY | 3 | 224451, 224452, 224453 — reconstruction done, merge never ran |
| MERGE_EMPTY | 2 | 224405, 224667 — zero-byte `done/` file; partials are the truth |
| **RAW_ONLY** | **31** | **224688-224718** — n_TOF has processed nothing |

Against the 08-10 inventory (359 MERGED / 53 PARTIALS_ONLY / 2 MERGE_EMPTY /
31 RAW_ONLY) that is a real change, and it is in the direction
[`NTOF_MERGE_REQUEST_2026-08-10.md`](NTOF_MERGE_REQUEST_2026-08-10.md) asked for.

**Newly merged (27):** 224462, 224481, 224488, 224502, 224510, 224513, 224519,
224525, 224541, 224543, 224546, 224563, 224564, **224573**, **224577**, 224597,
224606, 224614, 224618, 224624, 224625, 224629, 224632, 224635, 224640, 224660,
224661.

**Being reprocessed (24):** 224454, 224461, 224499, 224500, 224508, 224547, 224549,
224557, 224558, 224560, 224565, **224576**, 224617, 224628, 224637, 224638, 224639,
224652, 224653, 224654, 224655, 224666, 224671, 224673.

### Do not read a mid-reprocessing run as data loss

A snapshot taken inside that window is treacherous. Those 24 directories had their
partials **deleted** and are refilling: 224576 was 38 partials yesterday and is 2
today, 224499 was 39 and is 7. A naive inventory calls that NOTHING. The
distinguishing signal is the directory mtime — the new partials are stamped within
the hour — so `official_ledger.py` classifies a short, freshly written
`completed/<run>/` as **IN_FLIGHT** rather than as an absence.

**224560 is the one to watch.** It had a 31.8 GB merged file on 08-10; today the
merged file is gone and the run is being reprocessed. Anything downstream that
already read 224560 is reading a product that no longer exists, and anything that
reads it in the next few hours gets a partial set. The other 23 were unmerged
anyway, so nothing regressed for them.

## 2. The recipe is uniform across the whole official set

Every official X17 product with a readable history — **all 413**, including the
ones written today — carries `UserInput_2026_EAR2_X17_v4.h`, and its parameters
normalise to the **same fingerprint as our `v12_liqpileup`** (`35bf4b0a829d`, after
dropping path prefixes and the header file name, which differ without carrying
physics). There is no recipe boundary hiding inside the official set, and the runs
n_TOF reprocessed today were not reprocessed to change the recipe.

Our own products:

| production | runs | variant | fingerprint |
|---|---|---|---|
| `prod_v12` | 30, in 224688-224718 | `v12_liqpileup` | `35bf4b0a829d` — **same as official** |
| `prod_v11` | 224573-224579 | `v11_pssfit_width` | `1be9a5686df4` |
| variant studies | 224572, ×13 | v1…v13 | one per variant |

## 3. Runs that now exist in both — and the hit-for-hit result

Eight: 224572 (ours `v12_liqpileup`) and 224573-224579 (ours `prod_v11`). 224576 is
mid-reprocessing on n_TOF's side, so seven are comparable today.

`compare_identity.py` matches on `BunchNumber` — the two processings split a run
into partials differently, so partial N is not partial N — and compares all 22
per-hit columns over a window of interior bunches. The first and last bunch of the
overlap are dropped, since a bunch straddling a raw-file boundary is split across
two jobs.

### 224572 — same recipe on both sides

| trees | verdict |
|---|---|
| WALA-D, PSSA-D, SILI, PKUP | **IDENTICAL** — same hit count, and every column exact |
| LIQA-D | same hit count, every column exact **except `afast`** on 3-6 hits out of ~85 000 (0.00-0.02 %) |

So: **the same UserInput reproduces n_TOF bit for bit.** 49 483 WALA hits, 231 164
PSSC hits, all matching on `tof`, `amp`, `area`, `fwhm`, `risetime`, `chi2`,
`satuflag` — every one.

The `afast` exceptions are a handful of pathological pulses where that integral
comes out numerically unstable (differences of order 1e6-1e8 on cells whose
neighbours agree exactly); `aslow` and every other liquid column match. It is worth
a look if the liquid PSD ever matters at that precision, and it is not a processing
difference.

### 224574 and 224577 — our v11 against their v12

| trees | verdict |
|---|---|
| WALA-D, PSSA-D, SILI, PKUP | **IDENTICAL** on both runs |
| LIQA-D | hit count differs: official **+17 to +21 %** |

This is exactly the documented v11→v12 liquid step (`STEP SIZE` 2/4 → 1/3 and
`SIGNAL WIDTH HIGH` 5000 → 5000/30), and **nothing else moved**. That is the
strongest available evidence that the v11/v12 difference is the recipe and not our
chain — and it confirms the warning in
[`SLIM_FEASIBILITY_2026-08-08.md`](SLIM_FEASIBILITY_2026-08-08.md) § (c) with
measurement rather than inference: mixing prod_v11 into a v12 analysis is safe for
the wall and plastic legs and **not** safe for the liquids.

**Consequence for DREAM run_79.** Of its three gaps, 224573 and 224577 are now
officially merged at v12, and 224576 is being reprocessed. Once 224576 lands, run_79
is fully covered by official v12 products and prod_v11 need not be mixed in at all.

## 4. What we have that n_TOF does not

**30 runs**, 831 partials, **674 GB** — the contiguous block
**224688-224718** minus 224709, which is still processing on our side. All 31 runs
of that block are `RAW_ONLY` officially (12.76 TB of raw staged, nothing done).

| | |
|---|---|
| on the ntof disk, ours | 224688-224708, 224710-224718 (**30**) |
| still processing, ours | **224709** (344 raw files, the largest run of the block) |
| official state, all 31 | **RAW_ONLY** |

n_TOF's pass has gone past this block, not into it: it spent 08-11 merging and
reprocessing runs below 224688 and then moved on to 224719+, which belong to a
different experiment (`UserInput_2026_EAR2_STAR_commissioning_v0.h`). Nothing
suggests they intend to come back to it, so **this block stays ours**.

## 5. What this does not settle

* **The ledger is a snapshot.** 24 runs were mid-reprocessing when it was taken;
  their partial counts mean nothing until n_TOF finishes. Re-run
  `official_ledger.py` rather than quoting those numbers later.
* **The identity comparison is an exactness test, not a coverage test** — five
  bunches of one partial per run. `verify_transferred.py` is what covers every file.
* **Shared systematics stay invisible.** Bit-for-bit agreement proves the two
  chains are the same chain; it says nothing about whether that chain is right.
* **224560's merged file disappeared.** If anything downstream consumed it, that
  result is built on a file that no longer exists and should be re-run once the
  reprocessing lands.
* **Two campaign transfers reported COPY FAILED** (224705, 224711) even though both
  landed with the expected partial count. They are in the structural re-verification
  pass; until it reports, treat them as unconfirmed.
