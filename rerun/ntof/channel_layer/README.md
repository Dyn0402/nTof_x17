# n_TOF channel layer (P2 of `rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md`)

One dead/hot mask per (chamber A–D, mask era) for the n_TOF pass, measured on n_TOF's own hits and built from the
pass-1 bench lists under the F12 rule. Results: [`report.html`](report.html) (generated). Masks: `masks/ntof_<arm>_<era>.json`.

## What is here

| file | what |
|---|---|
| `cabling.py` → `cabling_check.json` | n_TOF vs bench channel→strip maps through `common.Mx17StripMap` for all 77 registry run_configs |
| `rules.py` | the rules on one view: flank dead / silent (`ws_dead_charge/flank_census.py`), shorts (`seedfit/channel_census.py`), SPS hot seed probe, near oscillators, coupled channels (`model_audit/sps_build_cache.py`, code 48244778 constants) |
| `census.py` | `sample`: first and last acquisition of every kept sub-run (1,062 acquisitions, 543 sub-runs). `probe`: reads 7 hit branches of each acquisition's `combined_hits_root` over XRootD (nothing copied), writes per-acquisition accumulators and decisions to `$WORK_DATA/ntof_pass1/channel_layer/acq/` |
| `make_masks.py` | sums the census per era, applies the mask rule, writes `masks/`, `acq_table.csv` |
| `make_report.py` | `report.html` + `figures/` |

Run (desktop; XRootD needs a throwaway venv, e.g. `python3 -m venv /tmp/xrdvenv_p2 && /tmp/xrdvenv_p2/bin/pip install --no-cache-dir xrootd fsspec-xrootd uproot awkward scipy pandas`):

```bash
/tmp/xrdvenv_p2/bin/python -I rerun/ntof/channel_layer/census.py sample
/tmp/xrdvenv_p2/bin/python -I rerun/ntof/channel_layer/census.py probe --workers 3            # ~11 min
/tmp/xrdvenv_p2/bin/python -I rerun/ntof/channel_layer/census.py probe --sample $WORK_DATA/ntof_pass1/channel_layer/controls.csv
.venv/bin/python rerun/ntof/channel_layer/make_masks.py                                         # ~5 min
.venv/bin/python rerun/ntof/channel_layer/make_report.py
```

`census.py sample` also needs `$WORK_DATA/ntof_pass1/channel_layer/hits_listing.json`, an EOS listing of each sub-run's
hits directories. It was made by a one-off XRootD `dirlist` over the registry's `eos_dir` values. `controls.csv` holds 12
out-of-scope acquisitions (runs 8, 10, 14, 17, 19, 161) used only to check the known dropouts.

Hits come from `combined_hits_root`. `combined_hits_root_rr1_cm0` exists for only 15 sub-runs (the Q4 test), and on run_145 the
two are field-identical (plan §6). Hits are used for QA only, never for geometry.

## The rule (per chamber, view and era)

Censored list `dead` = the union of:
1. **chamber_side**: F12 class `detector` (`rerun/model_audit/ws_dead_charge/f12_table.json`: a fault seen through ≥ 2
   different electronics), carried by position. A position F12 flagged dead or silent but not short is released if this
   census sees it fire normally beside bright neighbours (flank f ≥ 0.6 on ≥ 20 flank events, not silent). That is the
   same release `fe_unify2/build_masks.py` applies to census over-calls.
2. **bench_carried**: the bench census dead + shorts (`rerun/channel_layer/<det>_<key>.json`) ∪ the pass-1 golden
   bundle's `dead`. Two kinds of channel are dropped:
   - F12 `electronics` channels: they follow the bench FEU channel, not the strip. Example: det3's 307/398/400 on bench
     FEU 7.
   - Bench-dead channels that the n_TOF flank test judges healthy.

   Bench shorts are always carried, because a short fires and the flank test cannot disprove it.
3. **n_TOF's own faults**:
   - flank dead (f < 0.25) and silent (< 5 % of the ±32-strip occupancy, only where the expected count is ≥ 20), on the era
     sums;
   - a connector that is ≥ 32/64 silent on the era sum is masked whole;
   - shorts (amplitude cosine > 0.7 with a strip ≥ 2 pitches away) that replicate in ≥ 50 % of the era's per-acquisition
     votes (73 in E2, 46 in E1);
   - SPS seed-hot channels (> 3 % of a 1000-event block, > 10× the live ±4 neighbours) and near oscillators that are hot in
     ≥ 25 % of the era's acquisitions with ≥ 1000 events. They are censored, as at SPS.

Wildcard list `hot` (bundle `hot`: never seeded, noise ×10 in the fit; `HANDOFF_D_NOISY_CHANNELS.md` §3.2):
- **beam_occupancy_hot**: occupancy on the era's beam acquisitions > 8 % and > 8× the view median. These are the old
  `candidate_filter` constants.
- SPS **coupled** channels with persistent pairs.

Weak channels are listed and not masked, as on the bench.

## Mask eras (proposed registry column `mask_era`)

| era | acquisitions | definition |
|---|---|---|
| `E1_pre_access` | run_79, run_80, run_81 (255 in the registry, 46 sampled) | registry `access_epoch == pre_access_27jul` |
| `E2_post_access` | run_83 … run_162 (3,682 in the registry, 1,016 sampled) | `post_access_27jul` |

Within each era no other in-scope connector dropout was found, so no third era is needed. Proposed diff to
`rerun/ntof/make_ntof_registry.py`, after line 145:

```python
            row['mask_era'] = 'E1_pre_access' if row['access_epoch'] == 'pre_access_27jul' else 'E2_post_access'
```

Also add `'mask_era'` to the value-count print tuple at line 183, and a docstring line:
`mask_era  E1_pre_access / E2_post_access: n_TOF channel layer (rerun/ntof/channel_layer/masks/ntof_<arm>_<era>.json)`.

## Masks (channels; x / y)

| arm (bench) | E1 dead | E1 hot | E2 dead | E2 hot | bench pass-1 bundle dead |
|---|---|---|---|---|---|
| A (det3) | 67 / 32 | 0 / 0 | 1 / 35 | 0 / 0 | 3 / 1 |
| B (det2) | 10 / 34 | 3 / 0 | 8 / 24 | 0 / 0 | 6 / 11 |
| C (det6) | 41 / 15 | 0 / 0 | 38 / 21 | 0 / 0 | 40 / 21 |
| D (det7) | 157 / 86 | 19 / 10 | 154 / 61 | 20 / 21 | 137 / 13 |

Per-component counts and every channel's evidence are in the JSONs. `summary.json` has the counts.

## Findings

- **Cabling**: the bench and n_TOF channel→strip maps are identical for all four chambers, both views, all 512 channels,
  in all 77 n_TOF run_configs. Positions equal the bench `BPOS` frame exactly, so no remap is needed (N9 closed).
- **A-x connector 8 is reproduced.** It reads 0.0002 occupancy in every E1 acquisition (run_79, and run_81 up to the access),
  and is live from run_83. E1 masks all 64 channels.
- **The 9 August A-y connector-8 dropout is outside the scope.** run_161 (an A HV scan, not in the registry) has 0 hits on
  all 64 channels. run_160 (last acquisition before 16:21) and run_162 are live.
- **Early July** (controls, pre-production):
  - B-y connector 8 is silent on 4 Jul 18:03 and 8 Jul 09:09–18:17, but live on 4 Jul 17:49 and 5 Jul. The episode was
    intermittent, not continuous as the pedestal snapshots suggest.
  - A-y connector 8 is not silent in hits on 4–5 July (pedestal: 3–8 July). It reads ~0.3× the view median on 8–10 July.
- **The bench lists carry badly to n_TOF.** As F12 predicted:
  - A's bench mask is all FEU-7 electronics and is dropped.
  - C-x 210–219 (F12 chamber-side, silent) fire 0.65–0.97 beside bright neighbours at n_TOF and are released.
  - D-x's bench short groups (period 5) do replicate at n_TOF.
- **A-y post-access**: 23 shorts in a period-3 group on connector 8 (448–470) plus hot 448/454. None of this is there
  before the access or on the bench. It is the same connector that dropped out on 9 August, probably an n_TOF cable or
  connector fault after the access.
- **D's old "hot cells" are beam-induced.** D-x 92–112 and 451–464, and D-y 42–63, read 10–21× the view median on beam
  but 1–2× on beam-off cosmics. The SPS seed rule does not see contiguous blocks, so they go to the wildcard list (E2: D-x 16 plus 4 coupled,
  D-y 21), not to `dead`.
- **B-x 258/259 are noisy channels**, ×65–100 the median on cosmics. They are seed-hot and censored.
- **Old run_145 mask (102 channels).** Of its 102 channels, the new E2 lists hold:
  - A-y 16 of 18;
  - D-x 28 of 31 (13 dead, 15 hot);
  - D-y 40 of 51 (19 dead, 21 hot).

  Not taken: A-x 15, A-y 51/63, B-x 460, D-x 368–371 (5× the median, under the 8× rule), and D-y 46–59 / 387–414 (in
  part).

## How ntof_bundle.py should consume them (proposed; not applied)

`ntof_bundle.build` currently keeps the golden bundle's bench `dead` list by channel number. Add `--mask-era`. With it,
the bundle's channel layer is replaced by the n_TOF mask; without it, the build refuses to run, so a bench list can no
longer reach n_TOF silently:

```python
MASKS = os.path.join(HERE, 'channel_layer', 'masks')
MASK_ERAS = ('E1_pre_access', 'E2_post_access')

def build(..., mask_era=None):
    ...
    if mask_era not in MASK_ERAS:
        raise SystemExit(f'--mask-era must be one of {MASK_ERAS} (registry column mask_era)')
    mp = os.path.join(MASKS, f'ntof_{arm}_{mask_era}.json')
    m = json.load(open(mp))
    rec['channel_layer'] = dict(file=os.path.relpath(mp, REPO), mask_era=mask_era,
                                bench_dead_replaced={p: list(cal.dead.get(p, [])) for p in ('x', 'y')},
                                n_dead={p: len(m['planes'][p]['dead']) for p in ('x', 'y')},
                                n_hot={p: len(m['planes'][p]['hot']) for p in ('x', 'y')})
    cal.dead = {p: [int(c) for c in m['planes'][p]['dead']] for p in ('x', 'y')}
    cal.hot = {p: [int(c) for c in m['planes'][p]['hot']] for p in ('x', 'y')}
    cal.conditions['mask_era'] = mask_era
```

The argument goes into `main()` as `ap.add_argument('--mask-era', required=True, choices=MASK_ERAS)`. The docstring's
"carried … the dead list" line becomes "replaced: dead/hot by the n_TOF channel layer of the row's mask era".

The pass then builds 8 bundles (4 arms × 2 eras). `ntof_reco.py` picks the bundle directory from the registry row's
`mask_era`. A sub-run never spans eras: the boundary is the 27 July access between run_81 and run_83. `ntof_reco.py`
should still assert that all of a sub-run's rows share one `mask_era`, and `check_conditions` will catch a mismatch
through `conditions['mask_era']`. No `wft/` change is needed: `dead` is already censored and dead-seed-bridged, and `hot`
is already seed-excluded and down-weighted (`wft.model.HOT_NOISE_INFLATION`).

The SPS per-event coupled-channel censoring (`censor_x/censor_y`) has no n_TOF counterpart. A static list (in `hot`) is the
approximation used here. Only D-x shows persistent coupling (oscillators 477, 487).

## What this does not rule out

- A fault confined to acquisitions that were not sampled. Two per sub-run were sampled, and a per-acquisition SPS probe in
  the driver would catch it.
- Hot channels that are hot in fewer than 25 % of the acquisitions. They are listed with their frequency.
- E1 rests on 46 acquisitions taken during A's ×3.3 common-mode episode. Its short and hot lists are less certain than E2's.
- The beam-occupancy wildcard is not split by time of flight (flash vs late). A ToF-resolved look needs the n_TOF join.
