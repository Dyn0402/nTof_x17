# HANDOFF: build M3 tracking v2 and reprocess all June rays on lxplus

*Written 7-05 after the M3 tracking audit + v2 campaign. Everything below has
been executed at least once end-to-end; steps marked VERIFIED were run
exactly as written. Companion context: `HANDOFF_recofar_and_m3_tracking.md`
(Thread B is now closed) and the memory note `m3-tracking-audit`.*

## What v2 is

Repo: `github.com/Dyn0402/cosmic_bench_m3_tracking`, branch `main`
(local checkout: `~/CLionProjects/cosmic_bench_m3_tracking`). Key commits,
newest first:

- `reprocess script: handle multiple datrun sets per subrun`
- `Makefile: link boost from the active root-config's stack`
- `Add lxplus reprocessing script for June-runs M3 tracking (v2)`
- `Add config_ref_june2026.json` — June-refreshed per-plane offsets
- `Tracking v2: layer-drop rescue, NClus branches, charge-weighted centroid`
- `Fix latent bugs in ray fitting and QA paths`
- PR #1 (already merged before this work): ZS stale-channel fix

Physics changes vs what produced the June `m3_tracking_root` files:
1. **Layer-drop rescue resurrected** → ~+25% events with a ray.
2. **ZS fix** (PR #1) → ZS runs (ArIso 6-16, zs tests) become usable: on
   ArIso file 001 good-chi2 rays went from 153 to 68,044.
3. **NClusX/NClusY branches** in the rays tree — REQUIRED downstream:
   35% of rays have a 2-point coordinate (only ~38% land within 5 mm of the
   DUT vs ~85% for full fits) and most have denormal-tiny, NOT exactly zero,
   chi2, so no chi2 cut removes them.
4. **Charge-weighted centroid** (MGv2), **June-2026 offsets**
   (`config_ref_june2026.json` — always use this config, not `_2022`).

**Consumer selection ("the recipe"): `NClusX>=3 & NClusY>=3 & Chi2X+Chi2Y<5`**
(chi2 here = unweighted sum of squared residuals in mm²; use <2 for premium
purity). Validated on the Saturday det3 run vs the DUT:
reco_near 65.2→73.5%, reco_far 11.0→5.0%, within-5mm 85.6→93.6%, −10% rays.

## Build — local (VERIFIED)

```bash
cd ~/CLionProjects/cosmic_bench_m3_tracking
# ROOT 6.36.06 at ~/Software/root_6_36_06 already on PATH
make cleanall && make tracking DataReader -j8
```
Notes: `cpp11.patch` is already applied in-tree (do not re-apply). The
Makefile now emits header dependencies (`-MMD`); after pulling changes, a
plain `make` is safe, but when in doubt `make cleanall` — a stale-object
`sizeof` mismatch here manifests as `free(): invalid pointer` at exit.
Config JSONs contain `//` comments; modern boost's JSON parser rejects them —
any script feeding configs must strip comments (the shipped scripts do).

## Build — lxplus (VERIFIED)

```bash
ssh lxplus
git clone https://github.com/Dyn0402/cosmic_bench_m3_tracking.git m3_tracking_v2
cd m3_tracking_v2
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
make cleanall && make tracking DataReader -j4
```
Gives ROOT 6.34.02 / gcc 13.1 / boost 1.87. The Makefile links boost from the
LCG view (commit `Makefile: link boost...`); without that fix the system
boost 1.75 gets picked at link time → undefined `boost::filesystem` refs.
The clone at `~dneff/m3_tracking_v2` is already built and working.

## Reprocess all June M3 tracking on lxplus

Data: `/eos/experiment/ntof/data/x17/cosmic_bench/june_tests/<run>/<subrun>/raw_daq_data/`
— M3 raw files are the `*_01.fdf` (FEU 1). Output goes to
`<subrun>/m3_tracking_root_v2/` next to the originals. **Never modify or
delete the original `m3_tracking_root/`** — every June analysis to date
points there, and the old/new comparison is part of the validation.

```bash
cd ~/m3_tracking_v2
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
./scripts/reprocess_m3_lxplus.sh                    # everything
./scripts/reprocess_m3_lxplus.sh 'det3_saturday'    # or a filtered slice
```

Script behavior (VERIFIED on `mx17_det3_test_6-22-26/short_run`, including a
subrun holding an aborted extra datrun set):
- Iterates every `<run>/<subrun>` with M3 fdfs, every datrun set within it,
  runs read→ped→analyse→rays per data file with `config_ref_june2026.json`.
- Pedestals: prefers the pedthr matching the datrun timestamp, falls back to
  any pedthr in the subrun; if none, the subrun is reported in the final
  "Skipped" list — rerun just those with `PEDOVERRIDE` (edit the variable at
  the top; point it at a pedestal basename, e.g. one from
  `june_tests/pedestals/`, everything before `000_01.fdf`).
- **Restartable**: existing non-empty outputs are skipped.
- Working files go to `$TMPDIR`; logs `track_NNN.log` are kept there for
  failures.

Parallelism: one file takes ~3–5 min (0.5–1 GB fdf), and there are roughly
100–150 M3 files across june_tests → a single session is an overnight job;
3–4 sessions with **disjoint** filters (e.g. `'det1_det2'`, `'det2_det3'`,
`'det3_'`, `'det4|det6'`) split it to a few hours. Do not run two workers on
the same subrun (output-file race). Use tmux on lxplus; plain ssh sessions
die. Condor is overkill unless you want it.

## Validation (do this before switching any analysis to v2)

Per subrun, against the original rays where they exist:
1. `evn` sets must be IDENTICAL (same events read, same gaps) — VERIFIED on
   short_run: 20,452 events, same set. If evn sets differ, something is
   wrong with file/pedestal matching — stop and compare configs.
2. `NClusX`/`NClusY` branches present; events-with-ray up by ~15–30% on
   healthy runs (short_run: 13,186 → 16,584).
3. ZS runs (`mx17_det3_ArIso_Test_6-16-26`, zs tests): expect a dramatic
   good-ray increase; the original files there are junk (pre-ZS-fix).
4. For one anchor run (Saturday det3), rerun the DUT comparison and confirm
   the recipe reproduces reco_near ≈ 73.5%, reco_far ≈ 5.0%
   (`/tmp .../scratchpad/sat_final_eval.py` has the reference implementation;
   local v2 rays already at
   `~/x17/cosmic_bench/_m3check/sat_long/m3_tracking_root_v2/`).

## Consumer-side switch (nTof_x17, after reprocessing)

- Point `m3_tracking_dir` at `m3_tracking_root_v2/` (qa_config paths).
- `cosmic_bench_analysis/M3RefTracking.py` must be extended to load
  `NClusX`/`NClusY` (add to the default `variables` list AND to every
  per-track masking list in `cut_on_chi2` / `cut_on_det_size` /
  `get_single_track_events` — the class masks a hardcoded var list, and any
  jagged branch not in those lists silently desynchronizes) and to apply the
  recipe `NClusX>=3 & NClusY>=3` alongside the chi2 cut. Old rays files
  without NClus: fall back to `Chi2>0` (imperfect proxy — catches only ~8
  of the 35% two-point rays; prefer reprocessed files).
- The `chi2_cut=20` used everywhere in June QA becomes `5` (or 2) with the
  NClus requirement. Re-deriving the June PDF efficiency tables on v2 rays
  will move reco_near UP by ~6–8 points fleet-wide and shrink reco_far to
  its physical core.
- Detector alignments (alignment.json) were fit against old rays; the June
  offsets are gauge-fixed so DUT-frame shifts are <0.15 mm and get absorbed
  on the next alignment fit — rerun `03_alignment_and_tpc.py` per run before
  quoting sub-mm numbers.

## Gotchas inherited from this campaign

- M3 `evn` is the FEU hardware trigger counter: match detector events by
  VALUE, never by tree index. Gaps are real (spark-correlated DAQ drops).
- The rays tree is NOT evn-sorted (multithreaded analyse step) — sort or
  match by value.
- `evttime` is in 8 ns ticks (125 MHz); divide by 125e6 for seconds.
- DAQ-PC deployment (rays_daplxa) still runs the OLD binaries — coordinate
  separately if live production should move to v2 (needs `root_6_30_02` +
  `devtoolset-9` there; untested with v2, the Makefile boost fix should hold
  since it links from the active root-config's stack).
