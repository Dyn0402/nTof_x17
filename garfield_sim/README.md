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
├── mm_config.py              # Central config: gases, pressures, geometry, voltages
├── mm_generate_gas.py        # Step 1 — generate Magboltz gas tables (parallel)
├── mm_gain_scan.py           # Step 2a — local sequential gain scan
├── mm_gain_scan_parallel.py  # Step 2b — local parallel gain scan
├── mm_plot.py                # Step 3 — plot gain vs voltage / field
├── mm_condor_submit.py       # HTCondor — submit jobs to lxplus
├── mm_condor_worker.py       # HTCondor — single-job worker (runs on worker node)
├── mm_condor_job.sh          # HTCondor — bash wrapper (sources LCG environment)
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

### Local (`dylan-Yoga`, Ubuntu 24.04)
- ROOT 6.36.06 built from source
- Garfield++ built from source
- Python 3.12
- `ctypes.c_double` / `ctypes.c_int` for Garfield++ output args (`ROOT.Double`
  and `ROOT.Long` were removed in ROOT 6.22)

### lxplus / HTCondor worker nodes (el9 / RHEL9)
Source the LCG 108 view before running anything:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/setup.sh
```
This provides ROOT 6.32+ and Garfield++ 2025.x.

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
