# The liquid leg, redone: corrected saturation cut, full-pair statistics, and
# what is recoverable from a clipped pulse

**2026-07-30.** Redoes §3 of `archive/FINDINGS_2026-07-29_dream_crosscheck.md`, which
rested on (a) the retracted `amp > 31 000` "wrap" cut and (b) only 300 bunches of
`stat090_0000`. Superseded by nothing yet; the numbers below are the ones to
quote.

Headline: **the corrected cut does not move the result** — on the full sub-run it
reproduces the 07-29 table cell by cell, with same-arm excesses of 2.7-5.9x at
−5…−25 ns, replicated on both sub-runs — and **a physics-time clipped liquid pulse keeps its arrival time to
better than a nanosecond**, so saturated hits are recoverable as time-only hits
even though their amplitude is not.

---

## 1. What was wrong with the cut, and what it costs

`liq_coincidence.py` dropped every hit with `amp > 31 000` and ignored
`satuflag`. Both came from the unsigned-decode error: with samples read as
`<u2`, the ~31 200 baseline looked like the rail. Signed, the ceiling is
**~63 800** — the old cut sat at half of the real dynamic range.

Whole-run census (`liq_study/amp_ceiling_census.py`, all 16 partials,
3018 bunches):

| | LIQA | LIQB | LIQC | LIQD |
|---|---|---|---|---|
| hits | 50 955 430 | 56 453 914 | 14 846 655 | 34 891 436 |
| cut by the old `amp > 31 000` | 30 784 | 9 970 | 5 192 | 10 347 |
| …of which legitimate (31-63.8 k) | **22 940 (75 %)** | 6 332 (64 %) | 3 952 (76 %) | 8 639 (84 %) |
| genuinely over ceiling | 7 844 | 3 638 | 1 240 | 1 708 |
| …unflagged by `satuflag` | 698 (8.9 %) | 449 (12.3 %) | 189 (15.2 %) | 213 (12.5 %) |
| `satuflag` set | 12 030 | 7 395 | 5 000 | 5 678 |
| …flagged with `amp` back in range | 4 884 | 4 206 | 3 949 | 4 183 |

The cut is now `satuflag | amp > 63 800`. Both halves are needed: the flag alone
misses 9-15 % of over-ceiling hits, the amplitude alone misses ~4 000 flagged
hits per tree whose extrapolated `amp` lands back inside the range.

**Why §3's conclusions were safe anyway.** At physics times the hits the old cut
wrongly discarded are 601 (LIQA), 26 (LIQB), 0 (LIQC), 197 (LIQD) out of
0.9 M / 0.86 M / 0.13 M / 0.56 M in one partial — **~0.07 %**. That cannot move a
9-16 % coincidence fraction or a 5-7x excess. Verified directly: on a 50-bunch
slice with the corrected cut the same-arm diagonal is A/LIQA 0.134 vs 0.014
control at −5 ns, B/LIQB 0.143 vs 0.026 at −5, C/LIQC 0.014 vs 0.006 at −25,
D/LIQD 0.092 vs 0.022 at −5 — the old structure, unchanged.

The cut matters for anything **amplitude-differential**. A 31 000 rail truncates
the liquid energy scale at half range, which would silently corrupt the Phase-5
merged record (LIQ amplitudes in mV and MeVee). That is the reason to fix it, not
the coincidence numbers.

## 2. Full-pair statistics

`mm_activity_crosscheck.py` on the whole of `stat090_0000` (1012 bunches,
105 115 non-flash DREAM events — 3.3x the 300-bunch sample §2 used):

```
  matched any arm 95.7 %, exactly one arm 95.2 %, >1 arm 0.5 %

  MM chamber activity (>=2 strips in both planes)
  DREAM events matched to      chA     chB     chC     chD        n
    arm A only                81.4%   58.0%   26.8%   65.1%   21 640
    arm B only                39.0%   79.4%   32.9%   68.4%   27 131
    arm C only                37.3%   60.7%   76.3%   68.1%   26 496
    arm D only                34.7%   59.2%   28.9%   85.2%   24 816
    no arm (unmatched)        49.9%   64.1%   38.9%   72.4%    4 504

  matched  : cluster in any chamber  96.7 %  (n = 100 611)
  unmatched: cluster in any chamber  96.2 %  (n =   4 504)
```

Both §2 conclusions hold at full statistics: the diagonal is enhanced in every
row (76-85 % on-arm), and the matcher **does not select against MM content** —
96.7 % vs 96.2 % for the events it misses, so the residual inefficiency is
scintillator-leg, not fake DREAM triggers. (The 300-bunch sample gave 96.4 vs
96.5; the ordering flips but the difference is well inside the point either way.)

### The liquid coincidence, corrected cut, full sub-run

Same 100 083 exclusively-matched events. Cell = coincident same-bunch LIQ hits
per event within ±100 ns of the residual peak / the same in a +100 µs shifted
control, @ the peak position:

```
  matched arm     LIQA               LIQB               LIQC               LIQD
    A (n=21640)  0.165/0.028@  -5   0.032/0.028@+1445  0.006/0.004@ -535  0.013/0.009@-1985
    B (n=27131)  0.041/0.040@+335   0.151/0.032@  -5   0.006/0.006@  +15  0.027/0.024@+1835
    C (n=26496)  0.043/0.031@  -5   0.039/0.032@ -385  0.018/0.005@ -25   0.024/0.023@  -15
    D (n=24816)  0.044/0.031@ -15   0.036/0.029@ -675  0.006/0.004@ +615  0.094/0.019@ -15

  saturated hits dropped by the corrected cut, over the 1012 bunches:
    LIQA 4 315 of 17 179 922   (4 081 flagged + 234 over-ceiling-but-unflagged)
    LIQB 2 645 of 19 033 979   (2 485 + 160)
    LIQC 1 740 of  5 000 653   (1 684 +  56)
    LIQD 1 991 of 11 755 978   (1 911 +  80)
```

Every same-arm cell is on the diagonal at **−5 to −25 ns** with a
**3.6-5.9x excess** over the accidental floor (A/LIQA 5.9x, B/LIQB 4.7x,
D/LIQD 4.9x, C/LIQC 3.6x); sub-run 0001 below gives 2.7-5.9x on the same cells. Every off-arm cell sits at the floor with an unstable
peak position (+335 to −1985 ns). Same-arm liquid light accompanies
**16.5 % / 15.1 % / 1.8 % / 9.4 %** of matched events on LIQA/B/C/D.

**This supersedes §3 of the 07-29 findings and agrees with it cell by cell**
(0.159/0.025@−5 → 0.165/0.028@−5 on A/LIQA; 0.150/0.031@−5 → 0.151/0.032@−5 on
B/LIQB; 0.020/0.003@−25 → 0.018/0.005@−25 on C/LIQC; 0.090/0.019@−15 →
0.094/0.019@−15 on D/LIQD), which is the direct demonstration that neither the
wrong cut nor the 300-bunch sample was distorting anything. The one number to
restate: the excess is **2.7-5.9x** across both sub-runs, not the "5-7x" quoted
from 300 bunches —
the control floor is better measured now, and LIQC's ratio in particular drops
from 6.7x to 3.6x while its signal cell is unchanged.

The v12 LIQ time base is wall-aligned to −5…−25 ns, as before.

### `stat090_0001`: the independent replication

Disjoint bunch range (1165-2213), 103 198 exclusively-matched events, a
different hour of DREAM data:

```
  matched arm     LIQA               LIQB               LIQC               LIQD
    A (n=22045)  0.164/0.028@  -5   0.033/0.025@ +585  0.005/0.005@ -445  0.022/0.016@+1325
    B (n=28242)  0.045/0.036@+395   0.146/0.030@  -5   0.007/0.005@-1745  0.027/0.020@ -15
    C (n=27539)  0.042/0.033@-1185  0.035/0.033@ -605  0.016/0.006@ -15   0.026/0.023@ -15
    D (n=25372)  0.035/0.026@-1605  0.036/0.033@ -15   0.005/0.007@ +185  0.092/0.021@ -15
```

Diagonal against sub-run 0000, signal cell and peak position:

| | 0000 | 0001 |
|---|---|---|
| A / LIQA | 0.165 @ −5 ns | 0.164 @ −5 ns |
| B / LIQB | 0.151 @ −5 ns | 0.146 @ −5 ns |
| C / LIQC | 0.018 @ −25 ns | 0.016 @ −15 ns |
| D / LIQD | 0.094 @ −15 ns | 0.092 @ −15 ns |

Two independent hours agreeing to ≤0.005 per event in every diagonal cell, with
the same −5…−25 ns offsets. **Quote the excess as 2.7-5.9x**: pooling both
sub-runs, A/LIQA is 5.9x, B/LIQB 4.7-4.9x, D/LIQD 4.4-4.9x and C/LIQC 2.7-3.6x —
LIQC is the weak one, as it was in every earlier version of this measurement
(3.4x fewer hits overall).

## 3. The saturated pulses: time is recoverable, amplitude is not

### 3.1 How many

Per subrun (`stat090_0000`, 1012 bunches): LIQA drops **4 315** hits of
17 179 922 — 4 081 flagged plus 234 over-ceiling-but-unflagged. On a 50-bunch
slice, 45 of LIQA's 214 saturated hits (21 %) are at physics time and the rest
are in the γ-flash; LIQB/LIQC had none at physics time and LIQD 6. So the
physics-time saturated population is of order **0.005 % of hits**, and it is
almost entirely LIQA — consistent with the raw census, where LIQA is the only
liquid that clips at physics times with any regularity.

### 3.2 The distortion is confined to the top of the pulse

From the trees, with an **amplitude-matched** clean control (`amp` > 20 000 —
this matters, an all-amplitude control is dominated by near-threshold pulses with
`fwhm` ~2 ns and fakes a difference):

| | saturated `fwhm` | clean `fwhm` | saturated `fwtm` | clean `fwtm` |
|---|---|---|---|---|
| LIQA | 8.3 | 6.1 | **15.9** | **15.3** |
| LIQB | 10.4 | 6.2 | 13.0 | 14.2 |
| LIQC | 11.2 | 6.2 | 17.0 | 10.8 |
| LIQD | 11.7 | 6.4 | 21.5 | 15.8 |

Width at **half** height grows 35-85 %; width at **tenth** height is
essentially unchanged on LIQA (15.9 vs 15.3 ns). The clip flattens the peak and
leaves the rest of the pulse alone — which predicts that a constant-fraction
arrival time taken low on the rising edge should be untouched.

### 3.3 It is, at physics times — measured against the raw traces

`liq_study/clipped_timing_check.py`. `tof` cannot be compared to a fraction of
the pulse's own amplitude (for a clipped pulse that is the unknown), so both
populations are referenced to a **fixed absolute depth** on the rising edge:
`dt = tof − t_cross(base − 5 000)`, using the 259-pre-sample time base. Control
is unclipped pulses of depth > 40 000. Seven raw chunks, all four liquids:

```
                     detector-chunks   pulses   per-chunk median dt
  clipped, PHYSICS          7            24     3.5 - 3.8 ns   (median 3.7)   LIQA only
  unclipped control        ~20          ~60     3.5 - 3.7 ns                  LIQA, LIQD
  clipped, FLASH           28            80     2.5 - 129 ns   (median 62.4)  all four
```

**A physics-time clipped pulse is timed exactly as well as an unclipped one** —
the shift is +0.1 to +0.2 ns on LIQA in 5 of 7 chunks, and the per-pulse spread
is as tight as the control's. The 114-129 ns tail that shows up when the
populations are pooled is **entirely flash-region**, where the pulse sits on a
recovering baseline with neighbours inside its own window and the PSA is timing a
merged pulse. Nothing is usable in the flash anyway.

Caveat on statistics: **24 physics-time clipped pulses**, all LIQA, spread over
7 detector-chunks. The result is tight (every chunk's median inside 3.5-3.8 ns,
against 3.5-3.7 for the controls) and consistent, but it is tens of pulses, not
thousands — and it says nothing directly about LIQB/LIQC/LIQD, which simply do
not clip at physics times often enough to test.

### 3.4 So what to do with them

- **Keep cutting them from anything amplitude-based.** `amp` on a flagged hit
  runs to 7.6 × 10⁶ (LIQA) and is not a measurement in either direction; `area`
  is no help either, being exactly proportional to `amp` in these trees
  (`area/amp` is a per-tree constant — 7.55 on LIQA).
- **They may be kept as TIME hits at physics times**, with `amp` treated as a
  lower bound of 63 800. Gate on `tof > 1 ms`; do not do this in the flash.
- **Recovering the amplitude needs the raw trace.** The clip is 2-5 samples at
  physics time and `fwtm` is intact, so a template refit on the unclipped part
  would work in principle. It is not worth building for ~0.005 % of hits unless
  a specific analysis needs the top of the liquid energy scale.

## 4. Tools

| script | what |
|---|---|
| `liq_study/amp_ceiling_census.py` | whole-run amp/`satuflag` census, correct bands |
| `liq_study/clipped_timing_check.py` | raw-vs-PSA timing of clipped pulses, split physics/flash |
| `ntof_dream_merge/liq_coincidence.py` | corrected cut (`satuflag \| amp > 63 800`) |
| `ntof_dream_merge/liq_saturated_study.py` | population, timing and width of the cut hits |
