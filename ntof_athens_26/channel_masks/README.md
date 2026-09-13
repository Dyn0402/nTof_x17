# channel_masks — is the hot/dead classification hardware, or is it occupancy?

Follows [`HANDOFF_CHANNEL_MASKS.md`](../chi2_bimodality/HANDOFF_CHANNEL_MASKS.md)
§5, whose first two steps are: run the classifier over the whole campaign, and
**check stability first** — "if the classification moves run to run it is
measuring occupancy, not hardware, and step 1 produces a mask that varies for no
physical reason".

```bash
X17_ROOT=D:/x17 X17_BEAM_JULY=D:/x17/beam_july \
  ../../.venv/Scripts/python.exe mask_stability.py                  # ~4 min
  ../../.venv/Scripts/python.exe mask_stability.py --per-subrun run_79
```

Writes to `<out>/channel_masks/`.

---

## 1 · The answer, short

**It is hardware.** Three independent reasons:

1. **A blind validation passes.** The classifier recovers chamber A's x-view
   connector-8 fault — which `CLAUDE.md` records from the DAQ side and which
   nothing in this module is told about — as **43 of channels 448–511 dead in
   run_79 and 0 of 64 in every post-access run**. The block sits at 0.29× the
   rest of the plane in run_79 and 0.83–0.93× afterwards.

   > On the raw count only **22** of the 64 have literally zero hits, against
   > the "41 … recorded no hits at all" in `CLAUDE.md`. The two are counting
   > different things: this pools 11 sub-runs and 1.1 M events, so a
   > disconnected channel picking up a handful of noise hits is not zero. The
   > comparable number is the 43 the occupancy threshold calls dead.
2. **The hot *set* is stable run to run.** Jaccard against run_145, post-access:
   D·x 0.52–0.81, D·y 0.70–0.89, A·x 0.57–0.86, A·y 0.44–0.69. The same physical
   channels, six weeks apart.
3. **It moves when the hardware moves, and only then.** The one large change in
   the five staged runs sits exactly at the 27 July access.

So the handoff's step-2 worry is retired and step 3 (patch the bench bundles) is
not blocked on it.

## 2 · D is half noise in every run, not just run_145

`hits_in_hot` — the fraction of a plane's hits carried by channels classified
hot, which is what actually reaches a fit:

| arm·plane | run_79 | run_86 | run_116 | run_145 | run_162 |
|---|---:|---:|---:|---:|---:|
| A·x | 2.6 % | 5.1 % | 3.5 % | 6.0 % | 6.3 % |
| A·y | 13.8 % | 10.4 % | 15.5 % | **21.7 %** | 15.6 % |
| B·x | **32.7 %** | 7.0 % | 4.9 % | 0 % | 0 % |
| B·y | 12.0 % | 1.3 % | 0 % | 0.6 % | 0 % |
| C·x | **15.8 %** | 0.8 % | 1.7 % | 0 % | 1.6 % |
| C·y | **15.0 %** | 0.5 % | 0.6 % | 0 % | 0 % |
| D·x | 50.4 % | 56.3 % | 45.2 % | **55.6 %** | 52.2 % |
| D·y | 32.9 % | 43.6 % | 41.9 % | **50.1 %** | 53.0 % |

The run_145 column reproduces `noisy_channels_summary_run_145.csv` exactly on
all eight planes. **D's headline generalises**: 45–56 % (x) and 33–53 % (y) in
every run.

## 3 · ⚠ "Chamber C is clean" is true after the access and false for run_79

**This is new and it contradicts a load-bearing sentence of the handoff.** In
run_79 — the first long production run, and the run the handoff's own §3
contamination bound never covered — C carries **15.8 % (x) and 15.0 % (y)** of
its hits in hot channels, and B·x carries **32.7 %**. Post-access all three fall
to ~0.

Checked per sub-run, because pooling 11 sub-runs shrinks the Poisson error and a
threshold test can cross on statistics alone. It does not: B·x is 28–38 % and
C·x 15–16 % in **every one of run_79's 11 sub-runs separately**.

Two consequences:

- The handoff's §4 argument — "C has **zero** hot channels and still sits a
  factor 1.9 above A on short tracks, so masking is not the explanation for
  A-vs-C" — is safe, because it is a `chi2_bimodality` result drawn from the
  whole campaign where C is overwhelmingly post-access. But the *sentence* "C is
  clean" must not be carried to run_79.
- Run_79 is already flagged in `CLAUDE.md` for the dead A-x connector. It now
  has a second, unrelated condition, on two other chambers.

## 4 · What is blocked, and on what exactly

Step 1 proper — the whole campaign, per run condition — needs `combined_hits`
that is **not staged on this machine**. Of the 289 campaign sub-runs with a
`combined_hits_root` the local tree has **21, in 5 runs**; the other `combined_hits_root` directories exist and are
empty, which is why a campaign pass returns quietly rather than failing.
`missing_runs.csv` names them, so the staging job is one rsync rather than a
rediscovery.

One worry the handoff raises is already moot: it asks for the classification to
be redone per run condition because the 23 July `RdClk_Div` change doubled the
noise floor. **Every staged campaign run is run_79 or later, so all of them are
on the noisy side of that boundary** — the split does not cut this sample. The
other half of that worry stands: `HOT_FACTOR = 5.0` and `DEAD_THRESH = 0.2` were
tuned on run_145 alone, and nothing here re-tunes them.

## 5 · What this does not settle

- **Steps 3–5 are untouched**: patching the bench bundles, re-reconstructing
  stratified, and re-running `chi2_bimodality` all need the waveform path and a
  condor pass. Nothing here changes a fit.
- **The `noisy` (shape) class is not computed.** This module skips
  `noise.flag_noise` and the clustering, which is what makes it ~6 s per sub-run
  instead of ~5 min; it reproduces `dead`/`hot` exactly and produces no `noisy`.
  The handoff itself calls `noisy` the weaker evidence.
- **Five runs is not thirty-six.** The stability conclusion rests on run_79, 86,
  116, 145 and 162. They span six weeks and both access conditions, which is why
  it is worth stating at all, but it is not the campaign.
- **The prediction in the handoff's §5 step 5 is still open** — that masking
  should *remove* low-χ² tracks from D and A·y. That needs the re-reconstruction.
