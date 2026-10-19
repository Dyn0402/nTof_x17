# The campaign dataset — what it is, how to load it, what will bite you

Built 2026-09-09 from the full 2026 n_TOF campaign: **36 runs, 293 sub-runs,
25.6 M DREAM triggers.** Two tables, both local, both parquet.

| | path | size |
|---|---|---|
| **n_TOF hits** | `<out>/slim/ntof_hits_<run>_<subrun>.parquet` (293 files) | 9.6 GB |
| **tracks** | `<out>/stage3_campaign/tracks_campaign.parquet` | ~10 GB |
| **reco, FULL pass** | `<out>/reco_fullpass/<run>/<subrun>/mx17_<arm>/` | 28 GB |

> **SUPERSEDED IN PART, 2026-09-10 — read
> [`HANDOFF_FULLPASS_2026-09-10.md`](HANDOFF_FULLPASS_2026-09-10.md).**
> The track table above was built from the **allowlist** pass, which kept only
> **12.8 %** of the triggers that reconstruct into a two-track event. The whole
> campaign has since been reconstructed **blind** into `<out>/reco_fullpass`
> (47.8 M events, 9.5x). The track table has **not** been rebuilt on it,
> because doing so needs the angle-scale decision first: run_86 and run_145
> disagree on `k` by **24-29 % on C and D**.
>
> Two directory names that will catch you: **`<out>/fullpass` is the ALLOWLIST
> pass** despite its name, and `<out>/reco_fullpass` is the real full pass.
> `campaign_tracks` takes `--fullpass <path>` — pass it explicitly.

`<out>` is `/media/dylan/data/x17/sept26_prelim` (`$X17_SEPT26_OUT`).

```python
import pandas as pd
from sept26_prelim_analysis.slim_export import read_export

tr = pd.read_parquet('.../stage3_campaign/tracks_campaign.parquet')
hits = read_export('run_145', ['stat090_0000'])      # n_TOF, one sub-run
```

---

## The three things that will bite you

### 1. Never join on `event_id` alone

`event_id` is unique **within a sub-run**, not across them. The identity is
`(run, subrun, tag, event_id, arm, track_id)`. `source_imaging` builds its key
as `subrun + ':' + event_id` for this reason.

### 2. The 27 July access is a condition boundary — there is a column for it

Found from this campaign census (`OVERNIGHT_2026-09-08.md`). Median clean
strips per trigger steps down across it: **arm A ~10 → 0, arm C ~2 → 0**, and
**run_79/run_81 carry about twice every other run's INTER and INTRA fraction.**
INTER is the signal topology.

```python
tr.groupby('condition').size()      # pre_access_27jul / post_access_27jul
```

It is not the beam and not the trigger (n_TOF arm coincidence is identical
either side), and not the hot mask (run_79/81 have *fewer* masked channels).
**Quote the two sides separately or exclude run_79/81; do not pool a rate
across it.** This is a third boundary alongside the two in `../CLAUDE.md`
(23 July noise floor; A-x connector 8 through run_79).

### 3. `angle_calibrated` is a filter in disguise

`source_imaging._track_table` keeps only `gated & angle_calibrated`. A run
whose `k_arm` never certified contributes **nothing** to the pair analysis,
silently. Check before you conclude anything about a run:

```python
tr.groupby('run').angle_calibrated.mean()
```

Chamber **B never certifies** — its in-situ angle scale is not physical
(`PLAN.md` §2.3), so B tracks carry raw angles only, campaign-wide. Raw
tangents are always present, so a later calibration needs no re-reconstruction:

```python
tanx = tr.tan_raw_x / tr.k_arm
```

---

## Provenance and what is NOT in here

* Geometry comes from the **waveform fit**, never from `combined_hits` times
  (`../RECONSTRUCTION_BASIS.md`). The hit times are in the stage-1 census only,
  for candidate finding and QA.
* Stage 2 reconstructed **the stage-1 allowlist, ~5 % of triggers**, plus a
  prescaled control drawn from `SINGLE`/`BUSY`/`NONE`. `select_reason` says why
  each event was fitted. The control is what measures what the filter missed —
  it is not padding.
* Every arm is seeded from its **run_145 bundle with `v_drift` pinned at
  42.6 µm/ns**; per-run gas variation is absorbed by `k_arm` downstream.
  **The falsifier: if `k` varies strongly run to run, one bundle is wrong.**
  **IT HAS NOW FIRED, 2026-09-10:** run_86 and run_145, both measured on full
  passes and both on the same side of the 27 July access, disagree by
  **-28.5 % on C and -24.1 % on D** (A agrees to -2.9 %). See
  `HANDOFF_FULLPASS_2026-09-10.md` §5. Check
  `<out>/kcal/k_arm_<run>.json` across runs before trusting a pooled angle.
* **`hot_seed_strata` is run_145 only.** The production `no_hotstrip` cut is
  therefore inert on other runs, and D's angle scale is uncorrected for hot
  strips there (~0.55 % on k). `dropped_events` returns `{}` rather than
  guessing, so this degrades visibly, not silently.
* The n_TOF slim keeps the **full ±1000 ns `dt_ns`** and all `is_control` hits,
  deliberately: the accept window stays re-tunable offline without another
  campaign pass, and `is_control` is the accidental normalisation.

## The tight-coincidence cut, and its trap

`tight_coincidence.py` requires each arm within ±30 ns of its own trigger and
the two arms within 20 ns of each other. **It enriches a single-particle
background:** an opposing-chamber pair above ~170° is one particle crossing the
target and punching through both chambers, and it is perfectly time-coincident
*because it is one particle*. Use `tight_pair` (which excludes them), not
`tight`, for anything about X17; `back_to_back` isolates them, and they are the
cleanest back-to-back calibration line available.
