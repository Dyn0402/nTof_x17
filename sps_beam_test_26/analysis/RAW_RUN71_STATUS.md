# run_71 — the RAW run: FEU packet loss, and why the run is probably still good

2026-08-03 05:22–05:52, RAW, flat, resist 769.8 V, drift 700 → 450 → 275 V.

**Diagnosis, second pass (this supersedes the first).** The problem is **FEU
packet loss under RAW bandwidth**, ~24 % of sample-groups. The decoder is not
mis-parsing anything; it is behaving correctly on a lossy stream. Because the
loss is **uniform across the sample window and independent of signal
amplitude**, the physics the run was taken for is very likely still
recoverable — which is not true of the ZS censoring it was meant to replace.

---

## 1. Corrections to the first pass

Three things in the earlier version of this note were wrong:

1. **"The decoder mis-assigns `sampleID`; the ZS-tuned expression aliases."**
   Wrong. The sample header decodes correctly and increments by exactly +1
   every 8 blocks, cleanly, everywhere it is present.
2. **"All the data is present, nothing was dropped (ratio 1.027)."** Wrong,
   and wrong for a specific reason: I normalised against the decoder's 13,194
   `nt` entries, but those entries are not physics events — some contain two.
   Against the true ~17,500 events the ratio is ~0.76, not 1.03.
3. **"Do not retake; retaking risks burning beam on the same wall."** Wrong,
   and it was the operationally important error. Retaking at a lower data rate
   *would* fix this.

## 2. What the stream actually looks like

The RAW frame layout, confirmed against the decoder and the file size:
per sample, per Dream chip, a block of `4 header + 64 channel + 6 trailer`
words; 8 Dreams per sample-group; 64 sample-groups per event; 75,776 B/event,
matching the 1.0 GB per 13,194-entry file.

Measured on group 023:

- **Block structure is perfect.** Every block is exactly
  `dream*64 + channels 0..63`; the decoder reports zero bad reads over the
  whole file.
- **Dream order is cleanly periodic** 0,1,…,7 — sample-major.
- **The sample header increments +1 per group**, correctly.
- **But whole 5-group units are missing.** Header steps are +1 (35,729×) with
  +6 (1,599×) and +11 (225×) — i.e. 5 and 10 consecutive sample-groups absent.
  Inferred loss: **21–24 % of sample-groups**.
- **Events run short and then merge.** Runs between header resets are 39–59,
  essentially never 64; resets are −63 only 460 times out of ~1,400, the rest
  −59, −58, −54, −53. One `nt` entry (event 400543) contained 104 groups: the
  tail of one event (samples 5–59) followed by a whole one (0–63).

### Why the merging happens

`DreamDecoder.cpp` flushes an `nt` entry on the FEU's **end-of-event marker**
(`is_final_trailer(data) && get_EoE(data) == 1`, ~line 322). When the packet
carrying that marker is one of the dropped ones, the decoder never flushes and
the next event accumulates into the same entry. The merging is a *symptom* of
the loss, not an independent bug.

This also reconciles the event counting: 13,194 entries with eventId gaps of
2 (2,715×), 3 (643×) and 4 (118×) sum to ≈ 17,550, against an eventId span of
17,696. So **almost no events were lost outright** — the FEU shipped something
for nearly every trigger, just not all of it.

## 3. The result that matters: the loss is flat

Per-sample-index acceptance, over 2,002 re-split events:

```
 0-15  0.80 0.80 0.80 0.80 0.80 0.78 0.78 0.78 0.78 0.78 0.73 0.73 0.73 0.73 0.73 0.76
16-31  0.76 0.76 0.76 0.76 0.74 0.74 0.74 0.74 0.74 0.78 0.78 0.78 0.78 0.78 0.74 0.74
32-47  0.74 0.74 0.74 0.80 0.80 0.80 0.80 0.80 0.76 0.76 0.76 0.76 0.76 0.78 0.78 0.78
48-63  0.78 0.78 0.77 0.77 0.77 0.77 0.77 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75
```

| | |
|---|---|
| mean acceptance | **0.765** |
| first 32 samples | 0.764 |
| last 32 samples | **0.766** |
| range | 0.73 – 0.80 |

**Flat.** No preference for late samples, so the dispersed tail — the entire
reason for taking RAW — is *not* preferentially cut. And unlike zero
suppression, the loss does not depend on pulse amplitude, so it does not
sculpt the shape. It is a straight 24 % statistics tax with a known, uniform,
correctable acceptance.

That is qualitatively different from the ZS problem it replaced, where the
central strip's window closed ~400 ns after its peak and the truncation was
amplitude-dependent.

## 3b. RESOLVED — the data is fully recovered

Proven at the word level, then fixed in the decoder.

### The fdf is already unambiguous

`analysis/fdf_scan.py` walks the raw 16-bit words (**big-endian** — `read16`
does `ntohs`; reading little-endian gives convincing nonsense). On group 023,
80 MB scanned, 66,423 FEU frames:

| | |
|---|---|
| FEU header length | **8 words, every single frame** |
| eventID | 400539 … 401896, steps of **exactly +1, no gaps** |
| sampleID | 0 … 63 |
| repeated sampleID within an event | **0** |
| sampleID not strictly increasing within an event | **0** |
| mean frames per event | 48.9 → acceptance **0.764** |

So every surviving frame carries its eventID, frames are contiguous by event,
and sample indices inside an event are unique and ordered. **Event boundaries
are recoverable with certainty** — nothing about the raw file is ambiguous.
Only the ~24 % of sample-groups the FEU never shipped are gone, and they are
gone for good.

### The fix: flush on eventID change

`DreamDecoder.cpp` flushed only on the FEU end-of-event marker. When that
packet is dropped, the next event accumulates into the same entry. Patched to
**also flush whenever a new FEU header announces a different eventID**, stamping
the outgoing entry with the accumulated event's header, plus a final flush at
EOF for the last event. Validated on group 023:

| | before | after |
|---|---:|---:|
| entries | 13,194 | **17,696** |
| eventId steps | +1 (9,682), +2 (2,715), +3 (643) | **+1 only (17,695)** |
| max blocks per entry | 2,432 | **512** (the physical maximum) |
| entries with duplicated (channel, sample) | 852 / 3,000 | **0 / 3,000** |

**+34 % events recovered**, no merging, no duplicates.

**ZS regression: byte-identical.** Re-decoding run_63 `operating_03` group 004
gives the same 122,350 entries, identical eventIds, and identical
channel/sample/amplitude for the first 4,000 entries. The change is additive —
on ZS data the EoE fires normally and the new branch sees an empty buffer.

*The patch is in the working tree of `mm_strip_reconstruction`, not committed.*

### Should the fdfs be rewritten?

**No.** It was a reasonable thought, but the scan above shows the raw files
already carry everything needed: eventID on every frame, strictly increasing
sampleIDs, no ambiguity. Rewriting would mean fabricating end-of-event markers
into archival data to work around a decoder that has now been fixed — all risk,
no information gained. The end-of-event marker is in fact the *less* reliable
signal of the two: a loose word-level scan finds ~25 trailer words per event,
so EoE is not even a clean one-per-event tag, whereas eventID is exact.

The natural archival product is the **re-decoded ROOT**, not a patched fdf.

## 4. What to do

Decoder fix: **done** (§3b). Remaining:

1. **Re-decode all staged groups** with the patched decoder and run the chain:
   `decode_dataset.py run71_raw` → `pair_dataset.py run71_raw` →
   `flat_align_eff.py run71_raw` → `kernel_fit_m70V.py`.
2. **Normalise the mean waveform by the per-sample-index acceptance.**
   `kernel_fit_m70V.py` already computes that array (`ACC`) — it becomes a
   division rather than new machinery. Without it every `beta` comes out ~24 %
   low.
3. **Commit the decoder patch** (currently uncommitted in the working tree) and
   re-decode any other RAW data that exists, since every RAW run ever taken
   with this decoder has the same merging.

Expected cost of the loss: ~24 % of statistics; drift points fall from ~1,430
to ~1,090 usable events each. Acceptable.

**No retake is possible** (no beam for three years), so the flat ~24 % loss is
permanent — but it is uniform and amplitude-independent, which is the property
that matters. For any future RAW running: the ceiling is 512 ch × 64 samples ×
2 B × ~306 ev/s ≈ 20 MB/s, above what the FEU link sustains; prescale the
trigger ~30 % rather than cutting samples to 32, which would clip the tail and
trade a correctable loss for an uncorrectable one.

## 5. What is verified good about the run

Unchanged from the first pass, and all still true:

- RAW confirmed: `zero_suppress=False`, `pedestal_subtraction=False`,
  `common_noise_subtraction=False`, 64 × 60 ns, latency 32.
- Mount flat and unchanged — H4 TAX open 01:00:50 → 06:03:08, no access.
- Resist 769.8 V throughout; drift 700.4 V / 19.6 min, 450.4 V / 5.0 min,
  275.0 V / 5.0 min, matching `run71_two_points.log` exactly.
- Pedestals are the true raw-baseline set (means 344–2947 ADC); post-CNS noise
  10.4 ADC against 297 ADC raw, so `--cns 1 --zs-baseline 0` is right.
- All 512 channels present in every event; channel mapping intact (det4's dead
  stripes reproduce at 144–160 and 180–212 mm).
- Sub-run named `cfg_gain4.5_peaktime50` is P2's label; the Dream register
  `0x081f 0xd023` gives code 2 = **180 ns**, same shaping as run_56/63.
- banco's pipeline ran newest-group-first, so `combined_hits` exist for groups
  023–035 — exactly the 450 V and 275 V block.

## 6. Staged locally

`/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/`: groups 023–035
FEU3 (14 GB), 13 `combined_hits`, pedestal set, `hv_run71.csv`,
`dec_test_023.root` / `hits_test_023.root` / `decode_full.log` as the
diagnostic decode. `datasets.py` carries `run71_raw` with the three drift
plateaus and `raw=True`.
