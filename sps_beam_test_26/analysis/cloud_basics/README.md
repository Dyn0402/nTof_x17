# cloud_basics — detector-model basics: footprint, diffusion, resistive spread, gas

The narrative and every number live in **`FINDINGS.md`** (§1–27, dated, with retractions marked).
The report is generated: `~/x17/cosmic_bench/cloud_basics/attachment/report.html`, published as
<https://dylan-neff.web.cern.ch/notes/mx17-beam-attachment.html>.

## Where things stand (2026-10-10)

- **OPEN PROBLEM (2026-10-12, §29): the X width excess.** On every bench dataset X widens with depth about as fast as
  dry gas (D_T 400–520 µm/√cm), while v needs 0.6–1 % water (Magboltz 270–325). It does not depend on gain, it covers
  the whole chamber, and no Ar/iso/water mix fits det3's v(E) and D_T(E). The beam X shows it too. Unexplained;
  follow-up list in HANDOFF.md ("X width excess"). Meanwhile Dd comes from the X widths, not from v.

- **Late drift charge is lost in the beam gas, in both views** (§16). It is attachment: the loss rate is
  the same per unit time at three fields, there is no undershoot, X = Y, and it is flat in rate, spill,
  position and gain.
- **Gas compositions as model curves** (§25). One physics model with no free loss rate.
  - Beam run_71: 1.51 % water + 0.078 % air (163 ppm O2).
  - run_63 the night before: 230 → 147 ppm O2 over hours.
  - CO2 period: ~290 ppm.
  - Bench det3, six fields: 0.95 % water, no air (≲ 10 ppm O2).
  - Water and O2 are not in room-air proportion.
- **SPS constants on the bench** (§27, 2026-10-11; notes/mx17-sps-to-bench): the beam O2 does not spoil
  them. Injection test of the SPS measurement code with the run_71 loss on/off: footprint, D_rc, c2/c1
  move ≤ 7 %; only the decompose b1/τ split moves (−27 %). det4 D_rc ≈ 2.2e-4 ±10 % (beam, loss- and
  diffusion-corrected). On the bench D_rc and σ0 are degenerate with the assumed drift diffusion, and
  det4 bench sits 1.5–2.7× its beam D_rc: open.
- **Open questions closed:** the faint-tercile excess was a missing-sample selection artefact (§19); the
  0.7 µs ripple is a trigger-locked pickup (§20).
- **X model** (§17, §23): refits, resistive-strip snapping and footprint tails all fail to close X;
  det4 X vs its amplification stripes is next.
- **Traps learned the hard way** (each cost a reversal; see §10, §13, §14, §16):
  - never align stacks on the pulse (threshold or peak);
  - keep the FEU's dropped RAW samples NaN, never zero;
  - zero suppression distorts additively;
  - Magboltz η needs smoothing in E;
  - a bench single-field plateau cannot separate undershoot, field gradient and air.

## Pipeline (run from this directory with `../../../.venv/bin/python`)

### Data
| script | input → output | what |
|---|---|---|
| `../extract_det4_only.py run71_raw --cm masked --keep 12` | run_71 decoded_root → `wf_run71_raw_det4only_cmmasked_keep12.npz` (~10 min, ~8 GB RAM) | RAW beam waveforms, masked common mode, ±12 strips |
| `headon_stack.py <npz> --tag masked_k12` | → `results/headon_masked_k12.json` | trigger-placed head-on stacks, NaN-aware, widths ±0…±12, bootstrap bands |
| `zs_headon.py`, `zs_timestack.py` | run_63/run_56 ZS caches → `results/zs_*.json` | ZS stacks, both views, two selections |
| `ladder_profile.py` | run_63 rotated → `results/ladder_profile.json` | template-free charge vs depth (Y ladder) |
| `bench_stack.py det2=<pkl>:<bundle> …` | bench big caches (EOS `plane_ratio/inputs`) → `results/bench_stack.json` | all-strip trigger-placed bench stacks |
| `bench_driftscan.py <det3 bundle>` | det3 6-27 drift scan (EOS `june_tests/Run/mx17_det3_saturday_scan_6-27-26`) → `results/bench_driftscan.json` | six fields, masked CM |
| `bench_ladder.py`, `bench_trig.py` | bench big caches | bench ladder; single-field head-on fit (§15 note: weak) |

### Discriminators and checks
`headon_split.py` (rate/spill/position/charge), `gain_vs_loss.py` (charging), `split_toy.py`
(selection toy), `faint_test.py` (§19), `zs_emulate.py` (ZS distortion on RAW), `ripple_test.py` (§20),
`spike_test.py` / `att_compare.py` (superseded §13–14, kept for the record).

### Gas model
| script | what |
|---|---|
| `magboltz_drift.py <tag> [E=<field> c=<ncoll>]` | Garfield/Magboltz v, D_L, D_T, η; tags `<beam\|co2\|bench>_w<water>_a<air>` |
| `make_hs_jobs.py`, `condor_mb_hs.{sh,sub}` | high-statistics grid on lxplus condor (`~/cloud_basics_condor/att`), outputs → `results/air_hs/` |
| `gasmodel.py` | `GasGrid(gas, 'air_hs')`: interpolation in (water, air, E); η/D smoothed in E |
| `predict.py` | composition → arriving current → shaped signal (`current_field`: field profile, diffusion, survival) |
| `beam_comp_fit.py --source air_hs` | beam composition from run_71 (3 fields × X/Y) + run_63 ladder v |
| `driftscan_fit.py --source air_hs --emin 30` | bench det3 composition from six fields |
| `bench_fit.py --source air_hs` | bench chambers at their single field |
| `air_fit.py` | first-pass (coarse grid) v + r fit, superseded by `beam_comp_fit.py` |

### Figures and report
```
make_figures.py          # F1-F7, fits.json  (observable, discriminators)
make_comp_figures.py     # F9-F13, compositions.json  (compositions as model curves; run_63/CO2 per block)
make_report.py --inline /tmp/mx17-beam-attachment.html   # report.html (+ self-contained copy for the notes site)
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py /tmp/mx17-beam-attachment.html \
    --tags "X17, beam test, micromegas, gas, attachment" --force --deploy
make_attachment_deck.py  # the same study as a slide note (observable explained first, beam/bench consistency);
                         # writes <OUT>/mx17-attachment-slides.html, every chart redrawn from results/*.json
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
    ~/x17/cosmic_bench/cloud_basics/attachment/mx17-attachment-slides.html --slug mx17-attachment-slides --force --deploy
```
Slide cross-references in the deck are written `§§<slide id>§§` and resolved at build time, so adding a
slide renumbers them.

### The detector model as a slide note (2026-10-16)
`make_model_deck.py`: the physical forward model step by step (drift, footprint, RC spread, template, gas, parameters),
then against the r06 copy kernel and the held-out bench; writes
`~/x17/cosmic_bench/cloud_basics/model_deck/mx17-detector-model-slides.html`. Publish with
`add-note.py … --slug mx17-detector-model-slides --force --deploy`. Reads the production bundles from the data drive
(embedded fallback copy of their hypers otherwise).

### SPS → bench transfer (§27)
| script | what |
|---|---|
| `attach_transfer.py --nev 3000 --seeds 4 --workers 14` | injection toy → `results/attach_transfer.json` (~10 min on 14 cores) |
| `rc_drift.py` | RC fit with drift diffusion written in (beam + bench, D_T arms) → `results/rc_drift.json` (~40 min, single core) |
| `width_tan_split.py` | bench width growth split by reference \|tan\| → `results/width_tan_split.json` (inconclusive) |
| `make_transfer_deck.py` | slide note `~/x17/cosmic_bench/cloud_basics/transfer/mx17-sps-to-bench.html`; publish with `add-note.py … --slug mx17-sps-to-bench --force --deploy` |

### X width excess (§29; report `~/x17/cosmic_bench/cloud_basics/xwidth/report.html`)
| script | what |
|---|---|
| `headon_cache.py <key> [--subrun S --base DIR]` | every head-on reference event of a run → `xwidth/caches/headon_<key>.pkl`; `--subrun` = ref-only (M3 through the key's alignment), for the det3 scan sub-runs (EOS `cosmic_bench/june_tests/Run/`) |
| `x_width.py [--keys …]` | X-only head-on fit per dataset (σ0, Dd, T free), terciles in position → `results/x_width.json` |
| `x_width_map.py` | model-free early/late X width and fits in position quintiles (det3 pooled, det4) → `results/x_width_map.json` |
| `x_width_beam.py` | template-free X fit, beam run_71 + bench calibration → `results/x_width_beam.json` |
| `make_iso_jobs.py [--ext]`, `iso_fit.py` | Magboltz Ar/iso × water grid (condor 4440949/4440956 → `results/iso/`); joint v + D_T fit → `results/iso_fit.json` |
| `make_xwidth_report.py` | figures + report.html |

### X / kernel benches (lxplus condor)
`make_rc_arms.py`, `run_bench_rc.sh` + `condor_bench_rc.sub` + `auto_bench_rc.sh` (RC refits, §17);
`make_lor_arms.py`, `run_bench_lor.sh` + `condor_bench_lor.sub` (footprint tails, §23);
`compare_bench.py` → `results/compare_bench_{rc,lor}.json`. Payloads are tarballs of `wft/` +
`../plane_ratio/` built from this branch; check the md5 on lxplus before submitting.
