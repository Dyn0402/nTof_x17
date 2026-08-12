# Pulling n_TOF *waveforms* in the DREAM trigger windows — size estimate

**Verdict: the output is cheap (10–80 GB for the whole campaign, vs 12 GB of
hits today). The input is not — the samples exist only in raw stream1, so
producing it means re-reading ~28 TB, some of it from CTA tape.**

> **Update, 2026-08-12 (same day): built, and two numbers here were superseded.**
> The tool is [`waveform_pull/`](waveform_pull/README.md). It is **block-driven**
> — every ZS block overlapping the window, not just the ones with a PSA hit —
> which is strictly more complete and, measured, within 30 % of the hit-driven
> size (§3 below counts hits; §3a counts blocks).
>
> The **input is 34.9 TB, not 28 TB**, and only **9.5 TB of it needs a tape
> recall**: of the campaign's 83 n_TOF runs, 56 are still on the EOS disk
> staging area and 27 have expired (measured `lxplus/raw_inventory.sh`,
> 2026-08-12). Every one of the 83 is intact on CTA. **The disk copies expire
> ~2 weeks after the run, so this number is worse every day** and the on-disk
> runs should be pulled first.
>
> The adopted window is **±5 µs** with the +100 µs control, which is the "never
> come back" choice: ~4 GB per DREAM sub-run, **~390 GB for the campaign**, i.e.
> ~1 % of the raw it is cut from. The recall is what costs; the width is nearly
> free against it.

Estimated 2026-08-12. Sizes measured on `run_79/stat090_0000 × 224572`
(106,127 DREAM triggers, the ±1 µs slim in `slim_qa_dev/ref_v3/`), block
lengths and compression measured on the local raw chunk
`/media/dylan/data/x17/ntof_raw_224572/head_20.bin`. Campaign scaling is the
170 OK segments of `SLIM_CAMPAIGN_2026-08-12.md` = **13.23 M triggers**.

---

## 1. Where the samples are, and what one costs

Processed n_TOF output (ours or official) carries PSA results only — no
samples. Waveforms exist **only in the raw stream1 banks**
(`ntof_raw.parse_acqc`, format note in the DAQ repo):

| | |
|---|---|
| sampling | 1 GS/s, 1 sample = 1 ns; 20 ms window = 2e7 samples/channel/pulse |
| sample type | **signed int16**, 2 B/sample |
| structure | zero-suppressed blocks `(start, n, int16[n])`; payload begins `PRE_SAMPLES = 259` before `start` |
| always-kept head | 30 µs (50 µs PKUP) from t=0 — the gamma flash, irrelevant here |

Measured ZS block length **after** the flash block (head_20.bin, 3 events,
221 k blocks):

| family | blocks/event | mean samples | median | p90 | p99 |
|---|---|---|---|---|---|
| WAL | 54 900 | 928 | 834 | 1166 | 2250 |
| PSS | 15 400 | 1110 | 800 | 1574 | 4835 |
| LIQ | 3 130 | 926 | 773 | 1222 | 3459 |

So **a whole ZS block is ~1.9 kB** and a block is already ~1 µs wide — much
wider than a pulse. Cutting a narrower snippet out of it is worth real bytes.

Compression of real sample bytes (zlib-6, i.e. what a ROOT file would do):
**WAL 2.16×, PSS 2.48×, LIQ 2.50×**. Delta-encoding first does not help
(2.2×). Use **2.2×** as the zipped factor. (SILI is noisier, 1.4× — not in
scope.)

## 2. How many snippets a trigger needs

Hits inside the window, per DREAM trigger, from the slim itself
(signal only; the +100 µs control adds ~5 % at ±25 ns):

| window | hits/trigger | WAL | PSS | LIQ | snippets/trigger after merging overlaps |
|---|---|---|---|---|---|
| **±25 ns** | 3.44 | 2.11 | 1.25 | 0.08 | 3.23 |
| ±150 ns | 7.32 | 2.40 | 4.69 | 0.23 | 3.78 |
| ±1 µs (what the slim stores) | 16.45 | 3.63 | 11.81 | 1.00 | 5.7–7.3 |

Merging matters only at ±1 µs, where several hits on one channel fall in one
block. At ±25 ns essentially every hit is its own snippet.

## 3. The answer

Bytes = snippets/trigger × snippet samples × 2 B. Uncompressed / ×2.2 zipped:

### ±25 ns (all hits in window, signal only)

| snippet around each hit | per segment | whole campaign | zipped |
|---|---|---|---|
| **`[-50, +150]` ns** (shape only) | 138 MB | **17.2 GB** | 7.8 GB |
| `[-100, +300]` ns | 276 MB | 34.4 GB | 15.6 GB |
| `[-259, +700]` ns (≈ the whole ZS block) | 660 MB | 82.2 GB | 37.4 GB |
| `[-259, +1400]` ns | 1.14 GB | 142 GB | 65 GB |

### Minimal debug set — only the matched partner (best hit per trigger, 0.99/trigger)

| snippet | per segment | campaign |
|---|---|---|
| `[-50, +150]` ns | 42 MB | **5.2 GB** |
| `[-100, +300]` ns | 84 MB | 10.5 GB |
| `[-259, +700]` ns | 201 MB | 25.1 GB |

### If the window were kept at the slim's ±1 µs

443 MB – 2.3 GB per segment, i.e. **52–283 GB** campaign (24–128 GB zipped).
The PSS tail is what costs this; it is 11.8 of the 16.5 hits/trigger.

**Reference points.** The hits-only slim is 94 MB for this segment and 12 GB
for the campaign; DREAM `decoded_root` is ~12 GB per sub-run. So a ±25 ns
waveform pull at a 200 ns snippet is **1.5× the hits slim** and still ~1 % of
the DREAM data it accompanies. Even whole-block at ±25 ns (82 GB) is a
laptop-disk-sized product.

### Caveats on these numbers

- run_79 × 224572 is a **dedicated**-intensity pair and its slim file (94 MB)
  is ~1.3× the campaign mean (12 GB / 170 = 70 MB). Scaling by trigger count
  therefore likely **over**-estimates by ~25 %; call the band ±30 %.
- The 170 segments are 75 % of the beam. Full recovery adds ~30 %.
- Counts are hits *found by the PSA*. A waveform pull that wants to see what
  the PSA missed must be driven off the prediction, not off the hit list, and
  then it is one snippet per (trigger × channel of interest) — 12 channels ×
  0.99 triggers × 200 samples × 2 B ≈ 500 MB/segment, 62 GB campaign, for a
  complete unbiased 12-channel picture at ±25 ns.

## 3a. The same size, counted as BLOCKS (what was actually built)

The tool keeps whole ZS blocks overlapping the window regardless of hits, so the
size follows the block *rate*, not the hit rate. Measured block density per
detector per µs in ten bins of time-since-flash (`head_60.bin`, 3 events),
convolved with the real distribution of trigger times from the reference slim
(105,115 non-flash triggers; the beam sits at 1–80 ms, median ~18 ms):

| half-width | blocks/trigger | kB/trigger | per segment | campaign, signal only |
|---|---|---|---|---|
| ±25 ns | 1.50 | 2.8 | 302 MB | 38 GB |
| ±250 ns | 2.18 | 4.1 | 438 MB | 55 GB |
| ±1 µs | 4.46 | 8.4 | 892 MB | 111 GB |
| ±2 µs | 7.50 | 14.1 | 1.5 GB | 187 GB |
| **±5 µs** | **16.61** | **31.2** | **3.3 GB** | **413 GB** |

The +100 µs control roughly doubles all of these. **±5 µs with control is
therefore ~825 GB of samples, ~390 GB on disk after ZLIB(1)**, and that is the
adopted configuration. Cross-checked two ways that agree to 30 %: the block-rate
convolution above, and the hit-driven merge of §3; and directly against a real
pull — 7.5 MB of samples per bunch measured on bunches 398/399 of 224572 × 1012
bunches × 170 segments' worth of bunches lands in the same place.

Independent confirmation from the built product: a ±5 µs pull holds **1531
samples for every slim hit in the window**. That surplus is the point — it is
the part no hit-driven pull would contain.

## 4. The real cost is reading the raw

| | |
|---|---|
| raw stream1 | **0.197 TB per beam-hour** (mean of 39 measured runs, `NTOF_REPROCESSING_REQUEST_2026-08-08.md`) |
| DREAM-overlapped beam in the campaign | 8 668 min = 144.5 h |
| **raw to read** | **34.9 TB** — measured directly off CTA for all 83 runs, superseding the 28 TB this row estimated |
| … still on EOS disk | 25.4 TB, 56 runs — readable now, no recall |
| … expired, needs a recall | **9.5 TB, 27 runs** — all present and complete on tape |
| decode speed | ~500 MB/s per core (measured, `iter_banks`+`parse_acqc`) — **I/O-bound, not CPU-bound** |
| observed xrdcp | 225 MB/s (30 GB in 133 s, the slim campaign's own measurement) |

At the slim campaign's parallelism (~40 condor jobs) that is a few hours of
wall clock, but it is 28 TB of EOS reads, and **not all of it is on disk** —
several July runs already needed a CTA recall for stream1
(`recover_224526/`, and the `stream1: no -- recall from tape` rows in the
reprocessing request). Availability must be checked per run before promising a
date.

Two consequences for how to build it:

- **Do it selectively, not as a campaign.** One n_TOF run's raw is ~0.5 TB; a
  single segment's worth of waveforms is a ~100–600 MB product off it. For
  debugging, pull the runs you are actually arguing about.
- **The slim already tells the puller exactly what to fetch.** `hits` has
  `eventId`, `det`, `detn`, `tof`; `events` has the bunch. That is a complete
  (bunch, channel, sample-range) fetch list, so the raw pass is a filter with
  no physics in it — no clock fit, no PSA, no re-validation. It should be a
  separate tool reading the existing slim, not a new mode of `slim_pipeline`.

## 5. What is not estimated here

- Whether `PRE_SAMPLES = 259` and the `tof = start + j − 259` convention hold
  for every family (measured on LIQA only, 135/135 pulses).
- Read rates from tape-recalled files.
- Any ROOT/awkward overhead for variable-length arrays — the numbers above are
  payload bytes; expect a few % on top.
