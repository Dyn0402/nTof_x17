# Micromegas Gain Simulation (Garfield++)

Simulates gas gain vs. mesh voltage for a Micromegas detector using
Garfield++ / AvalancheMicroscopic. The goal is to characterise detector
performance at two operating sites for the n_TOF X17 experiment (EAR2):

- **CEA Saclay** (160 m altitude, ~745.8 Torr) — cosmic muon tests
- **CERN** (450 m altitude, ~720.8 Torr) — beam tests

Lower pressure at CERN → higher E/N at same voltage → higher gain. The
~3.4% pressure difference produces an expected ~10–25% gain difference,
which requires 400–2000 events per voltage step to resolve statistically
(Polya/exponential distributions with σ ≈ μ).

**Detector geometry:** 150 µm amplification gap (mesh to resistive anode).

---

## File Overview

```
garfield_sim/
├── setup_garfield.sh         # THE environment: ROOT + pinned Garfield++, any host
├── probe_penning.py          # Re-probe Garfield's built-in Penning table after a pin change
├── mm_config.py              # Central config: gases, pressures, geometry, voltages
├── mm_generate_gas.py        # Step 1 — generate Magboltz gas tables (parallel)
├── mm_gain_scan.py           # Step 2a — local sequential gain scan
├── mm_gain_scan_parallel.py  # Step 2b — local parallel gain scan
├── mm_plot.py                # Step 3 — plot gain vs voltage / field
├── mm_condor_submit.py       # HTCondor — submit jobs to lxplus
├── mm_condor_worker.py       # HTCondor — single-job worker (runs on worker node)
├── mm_condor_job.sh          # HTCondor — bash wrapper (sources setup_garfield.sh)
├── mm_condor_collect.py      # HTCondor — merge fragment results into JSON
├── gas_tables/               # Cached .gas files (Magboltz output, one per gas×pressure)
└── results/                  # JSON result files + summary.csv + plots
```

---

## Workflow

### Local (small N, development)

```bash
# 1. Generate gas tables (slow — Magboltz; skip if .gas files already exist)
python3 mm_generate_gas.py                  # auto-detect cores
python3 mm_generate_gas.py --workers 4     # explicit
python3 mm_generate_gas.py --force         # regenerate existing files
python3 mm_generate_gas.py --ncoll 5       # faster, less accurate (default: 10)

# 2. Run gain scan
python3 mm_gain_scan_parallel.py            # parallel (recommended)
python3 mm_gain_scan.py                     # sequential (simpler, for debugging)

# 3. Plot results
python3 mm_plot.py
```

### HTCondor (large N, lxplus)

```bash
# On lxplus — submit jobs
python3 mm_condor_submit.py                 # submit all enabled gases in RUN_CONFIG
python3 mm_condor_submit.py --dry-run       # preview JDL without submitting
python3 mm_condor_submit.py --gas Ar_CF4   # filter by gas label substring

# Monitor
condor_q -name $(myschedd show | grep -o '[^ ]*\.cern\.ch')

# After jobs complete — merge fragment JSONs into result files
python3 mm_condor_collect.py
python3 mm_condor_collect.py --no-append    # discard existing results, start fresh
```

Gas tables must be generated **before** submitting to HTCondor (gas files are
transferred to worker nodes as HTCondor input files).

### Gas tables on HTCondor (preferred on lxplus)

`mm_generate_gas.py` runs every table in a local process pool — fine on a
workstation, wrong on lxplus (interactive nodes kill long CPU hogs, and one
hung Magboltz stalls the pool). Use one job per table instead:

```bash
# edit gasgen_points.txt: one "GAS_LABEL PRESSURE_LABEL" pair per line
condor_submit mm_gasgen.sub          # finished .gas lands straight in gas_tables/
```

`mm_gasgen_one.py` takes the composition and field grid from `mm_config.py`
(transferred with the job), so the config stays the single source of truth.

### Picking a voltage window before you burn CPU

Avalanche cost explodes with gain — a 200-event batch is minutes at G ~ 10³ and
hours at G ~ 10⁵. `mm_alpha_predict.py` reads α and η straight out of a new
`.gas` file and predicts the gain curve, calibrating the Penning shortfall
`K = ln G_sim / ((α−η)·d)` on gases that were already simulated:

```bash
python3 mm_alpha_predict.py --calibrate Ar_iC4H10_95_5 Ar_iC4H10_90_10 \
                            --predict Ar_CO2_iC4H10_93_5_2 --pressure CERN_450m
```

K runs ~1.5 (Ar/CO₂ 70/30) to ~2.5 (Ar/iC₄H₁₀ 98/2) and tracks the argon
fraction, so treat the prediction as ±30 V on the window and confirm with a
cheap prescan:

```bash
python3 mm_condor_submit.py --gas Ar_CO2_iC4H10_93_5_2_rP040 \
        --pressure CERN --batches 1 --events-per-batch 20 \
        --voltages 380:560:20 --jobs-dir $PWD/jobs_prescan
```

`--jobs-dir` keeps low-statistics prescan fragments out of the production
merge. Then set `TERNARY_VOLTAGES` in `mm_condor_submit.py` and submit for
real. **Always pass `--gas`** — an unfiltered submit would refill every missing
fragment of every gas in `RUN_CONFIG` (thousands of jobs).

---

## Gas Mixtures

All gases are defined in `mm_config.py` (`GAS_MIXTURES`) and are picked up
automatically by `mm_generate_gas.py` and the local scan scripts.

| Label | Composition | Penning |
|---|---|---|
| `He_C2H6_96p5_3p5` | He/C₂H₆ 96.5/3.5% | manual, rP=0.40 (He metastable 19.8 eV ≫ C₂H₆ IP 11.5 eV; not in Garfield++ built-in table) |
| `Ar_iC4H10_95_5` | Ar/iC₄H₁₀ 95/5% | auto (Sahin et al. JINST 5 2010) |
| `Ne_iC4H10_95_5` | Ne/iC₄H₁₀ 95/5% | manual, rP=0.50 (central estimate; run 0.40/0.50/0.60 to bracket) |
| `Ar_CO2_70_30` | Ar/CO₂ 70/30% | auto (no Penning: Ar* 11.55 eV < CO₂ IP 13.78 eV) |
| `Ar_CF4_90_10` | Ar/CF₄ 90/10% | auto |
| `Ar_CF4_iC4H10_88_10_2` | Ar/CF₄/iC₄H₁₀ 88/10/2% | auto (Ar*→iC₄H₁₀ dominant channel) |
| `Ne_CF4_90_10` | Ne/CF₄ 90/10% | manual, rP=0.40 (Ne metastable 16.6 eV; not in Garfield++ built-in table) |
| `Ar_CF4_CO2_45_40_15` | Ar/CF₄/CO₂ 45/40/15% | auto |
| `CF4_100` | Pure CF₄ | auto (single component) |
| `Ar_CO2_iC4H10_93_5_2` | Ar/CO₂/iC₄H₁₀ 93/5/2% | **manual, rP = 0.40** (central; run as `_rP030/_rP040/_rP050`) — auto does **nothing** for this ternary, see below |

**⚠ Ar/CO₂/iC₄H₁₀ 93/5/2 (the n_TOF operating gas) — Penning must be manual.**
Probed against LCG_108 Garfield++ on 2026-07-31 and **re-confirmed
unchanged** against the pinned master `927e5c21` on 2026-08-06
(`probe_penning.py`; every rP below reproduces exactly):
`EnablePenningTransfer()` prints *"Penning transfer probability for
Ar/CO2/iC4H10 is not implemented"* and returns **false**, i.e. `mode: "auto"`
would simulate this mixture with **zero** Penning transfer while the
`Ar_iC4H10_95_5` reference runs at rP = 0.40 — a large, silent bias on any
gain comparison. Garfield's built-in binaries at 720.8 Torr for reference:

| Ar/CO₂ | 1% | 3% | 5% | 7% | 10% | 15% | 20% | 30% |
|---|---|---|---|---|---|---|---|---|
| rP | 0.171 | 0.266 | 0.330 | 0.376 | 0.424 | 0.476 | 0.509 | 0.547 |

Ar/iC₄H₁₀ is a flat 0.400 at every fraction (Garfield only has the 10%
measurement of Sahin et al.). At 7% total quencher the ternary sits at
rP ≈ 0.40; it is simulated at 0.30 / 0.40 / 0.50 so the Penning systematic
propagates into the HV map instead of hiding inside it.

**Penning notes:**
- `mode: "auto"` calls `EnablePenningTransfer()` — uses Garfield++'s built-in
  parameterisation if the mixture is known, otherwise no Penning.
- `mode: "manual"` calls `EnablePenningTransfer(rP, 0., noble_gas)` and must be
  applied **after** `LoadGasFile()`.
- Penning is applied in simulation only — gas table generation (Magboltz) is
  Penning-agnostic.

---

## Pressures

Computed from the barometric formula (scale height 8500 m):

| Site | Altitude | Pressure |
|---|---|---|
| Saclay | 160 m | ~745.8 Torr |
| CERN | 450 m | ~720.8 Torr |

---

## Result JSON Schema

Every result file (`results/<gas_label>_<pressure_label>.json`):

```json
{
  "gas":             "He_C2H6_96p5_3p5",
  "pressure_label":  "Saclay_160m",
  "pressure_torr":   745.8,
  "gap_cm":          0.015,
  "temp_k":          293.15,
  "n_events":        200,
  "penning":         {"mode": "manual", "rP": 0.40, "gas": "he"},
  "voltages":        [400.0, 405.0, ...],
  "fields":          [26667.0, ...],
  "gain_mean":       [...],
  "gain_median":     [...],
  "gain_std":        [...],
  "gain_rms_rel":    [...],
  "gain_raw":        [[123, 456, ...], ...],
  "survival":        [...],
  "n_attached":      [...],
  "runtime_s":       [...],
  "total_runtime_s": 1234.5,
  "partial":         false
}
```

`gain_raw[i]` contains one integer per surviving electron at voltage `voltages[i]`.
Total events = `len(gain_raw[i]) + n_attached[i]`.

---

## Computing Environments

**One entry point on every host:**

```bash
source setup_garfield.sh
```

It exports ROOT + Garfield++ and prints what it resolved. It is the only file
in this directory that names a Garfield or LCG path — everything else (the
wrappers, the `.sub` files, `mm_condor_submit.py`) goes through it.

**Pinned Garfield++: master `927e5c21` (2026-08-06)**, built from source on all
three hosts. We do *not* use the Garfield in the LCG views: LCG_108 ships
`6fb94b35` (2025-07-07, 664 commits behind) and LCG_109 ships `78fe1bd3`
(2026-02-02, 281 behind), and the MX17 response chain needs master-only
features (`Examples/ResistiveMicromegas`, `AvalancheMicroscopic::GetIons`, the
neBEM OpenMP race fix, interface-crossing checks, the FFT convolution fix, the
arbitrary-PSD noise generators). Rationale in `setup_garfield.sh`; the plan is
`MX17_Geant/design/RESPONSE_SIM_PLAN.md`.

| Host | ROOT | compiler | Garfield install |
|---|---|---|---|
| laptop `dylan-Yoga` (Ubuntu 24.04) | 6.36.06 (source) | gcc 13.3 | `~/garfield/install` |
| desktop `dylan-MS-7C84` (Ubuntu 22.04) | 6.30.02 (`~/Software/root_6_30`) | gcc 11.4 | `~/Software/garfield/install` |
| lxplus (el9) | 6.38.00 (LCG_109 view) | gcc 14.3 | `/afs/cern.ch/user/d/dneff/work/garfield_install/lcg109-927e5c21` |

All three pass the upstream test suite (`ctest`: 22/22).

`ctypes.c_double` / `ctypes.c_int` are still needed for Garfield++ output args
(`ROOT.Double` and `ROOT.Long` were removed in ROOT 6.22).

### HTCondor worker nodes
Jobs do **not** read the AFS install — `setup_garfield.sh` unpacks a shipped
`garfield-927e5c21.tar.gz` (6.7 MB) from the scratch directory, so nothing
depends on the worker holding an AFS token. The `.sub` files and
`mm_condor_submit.py` already list it in `transfer_input_files`.

### Moving the pin
Rebuild on each host, re-tar the lxplus install, then edit
`MX17_GARFIELD_PIN` in `setup_garfield.sh`, `GARFIELD_PIN` in
`mm_condor_submit.py`, and the tarball name in the `.sub` files. Re-run
`probe_penning.py` afterwards and reconcile it against the tables below — a
silent change in Garfield's built-in Penning table would bias every gain
comparison.

**EOS paths** (accessible from both lxplus and worker nodes):
```
/afs/cern.ch/user/d/dneff/work/git/nTof_x17/garfield_sim/gas_tables/
/afs/cern.ch/user/d/dneff/work/git/nTof_x17/garfield_sim/jobs/      # per-job fragments
/afs/cern.ch/user/d/dneff/work/git/nTof_x17/garfield_sim/results/
/afs/cern.ch/user/d/dneff/work/git/nTof_x17/garfield_sim/logs/
```

---

## Technical Notes

- **ROOT/Garfield++ is not thread-safe.** Always use `multiprocessing` with
  `mp.get_context("spawn")` — never fork. Import ROOT/Garfield only inside
  worker functions, never at module level.
- **Seed electron** is injected at `z = gap_cm` (the mesh) and drifts toward
  `z = 0` (the anode): `aval.AvalancheElectron(0., 0., gap_cm, 0., 0.)`.
- **Append mode:** re-running either the local scan or `mm_condor_collect.py`
  concatenates new `gain_raw` events onto existing results without discarding
  prior data.
- **HTCondor schedd:** submit uses `myschedd` to resolve a custom schedd name;
  falls back to the default schedd if unavailable.
