# Claude Code Briefing — Micromegas Gain Simulation (HTCondor Extension)

This document gives full context for continuing development of a Garfield++
Micromegas gain simulation. The existing code lives in the repo. Your task is
to add HTCondor/lxplus distributed job support. Read this fully before touching
anything.

---

## 1. What This Project Is

Dylan Neff (`dneff`) is a physicist at CERN working on the n_TOF X17 experiment
(EAR2), searching for the hypothetical X17 boson using Micromegas detectors.

This simulation computes **gas gain vs mesh voltage** for a Micromegas detector
using Garfield++ / AvalancheMicroscopic. The motivation is to characterise
detector performance at two sites:
- **CEA Saclay** (160 m altitude, ~745.8 Torr) — where cosmic muon tests happen
- **CERN** (450 m altitude, ~720.8 Torr) — where beam tests happen

Lower pressure at CERN → higher E/N at same voltage → higher gain. The pressure
difference is ~3.4%, producing an expected ~10–25% gain difference. This effect
is real but requires ~400–2000 events per voltage step to resolve statistically
(gain distributions are Polya/exponential with σ ≈ μ, so σ(mean) ∝ 1/√N).

**Detector geometry:**
- Amplification gap: 150 µm (mesh to resistive anode)
- Active area: 40×40 cm, 512×512 strips (irrelevant for gain simulation)
- Resistive anode treated as absorbing boundary (electrons stop at z=0)

**Gas mixtures:**
- `He_C2H6_96p5_3p5`: He/C₂H₆ 96.5/3.5% — Penning manual, rP=0.40 for He
  (He metastables at 19.8 eV >> IP(C₂H₆) ≈ 11.5 eV; rP not in Garfield++
  built-in table, 0.40 is a conservative estimate to validate against data)
- `Ar_iC4H10_95_5`: Ar/iC₄H₁₀ 95/5% — Penning auto (built-in table,
  Sahin et al. JINST 5 2010)

---

## 2. Existing File Structure

```
garfield_sim/
├── mm_config.py                 # Central config — gases, pressures, geometry
├── mm_generate_gas.py           # Parallel Magboltz gas table generation
├── mm_gain_scan.py              # Sequential gain scan (reference)
├── mm_gain_scan_parallel.py     # Parallel gain scan (flat multiprocessing pool)
├── mm_plot.py                   # Plotting (pure matplotlib, no ROOT needed)
├── gas_tables/                  # Cached .gas files (Magboltz output)
│   ├── He_C2H6_96p5_3p5_Saclay_160m.gas
│   ├── He_C2H6_96p5_3p5_CERN_450m.gas
│   ├── Ar_iC4H10_95_5_Saclay_160m.gas
│   └── Ar_iC4H10_95_5_CERN_450m.gas
└── results/                     # JSON results + summary.csv
    ├── He_C2H6_96p5_3p5_Saclay_160m.json
    └── ...
```

**DO NOT** modify `mm_config.py`, `mm_generate_gas.py`, `mm_gain_scan.py`, or
`mm_plot.py`. Add new files only.

---

## 3. Key Design Decisions Already Made

### Python / ROOT compatibility
- ROOT 6.36.06 on local machine (Ubuntu 24). `ROOT.Double` and `ROOT.Long`
  were removed in ROOT 6.22. Use `ctypes.c_double()` / `ctypes.c_int()` for
  output arguments to Garfield++ methods. Already fixed in all existing files.
- `ElectronTownsend` signature: `(ex, ey, ez, bx, by, bz, alpha)` — 7 args.
- `AvalancheElectron` signature: `(x, y, z, t, e_kinetic)` — 5 args.
- `GetAvalancheSize(ne_out, ni_out)` with ctypes ints.

### Multiprocessing
- ROOT/Garfield++ is NOT thread-safe. Use `multiprocessing` with
  `mp.get_context("spawn")` — never fork. Import ROOT/Garfield ONLY inside
  worker functions, never at module level in the main process.
- Nested pools are forbidden (daemon processes cannot have children). The local
  parallel script uses a single flat pool where each task = one (combo, voltage).

### Append mode
- Results are stored with full `gain_raw` lists (one int per surviving electron).
- `load_existing_raw()` in `mm_gain_scan_parallel.py` reads prior JSON and
  returns `{voltage: {"gain_raw": [...], "n_attached": int}}`.
- New events are concatenated onto prior `gain_raw` before recomputing stats.
- `n_events` in JSON = events added in most recent run (not total).
- Total events recoverable as `len(gain_raw[i]) + n_attached[i]`.

### Result JSON schema
Every result file (`results/<gas>_<pressure>.json`) has this structure:
```json
{
  "gas": "He_C2H6_96p5_3p5",
  "pressure_label": "Saclay_160m",
  "pressure_torr": 745.8,
  "gap_cm": 0.015,
  "temp_k": 293.15,
  "n_events": 200,
  "penning": {"mode": "manual", "rP": 0.40, "gas": "he"},
  "voltages": [400.0, 405.0, ...],
  "fields":   [26667.0, ...],
  "gain_mean":   [...],
  "gain_median": [...],
  "gain_std":    [...],
  "gain_rms_rel": [...],
  "gain_raw":    [[123, 456, ...], ...],
  "survival":    [...],
  "n_attached":  [...],
  "runtime_s":   [...],
  "total_runtime_s": 1234.5,
  "partial": false
}
```
`mm_plot.py` reads this schema directly. Any HTCondor extension must produce
or merge into this exact format.

---

## 4. Computing Environment

### Local machine
- Hostname: `dylan-Yoga`, Ubuntu 24.04
- ROOT 6.36.06 built from source
- Garfield++ built from source against that ROOT
- Python 3.12
- Project path: `~/PycharmProjects/nTof_x17/garfield_sim/`

### lxplus
- Username: `dneff`
- AFS: `/afs/cern.ch/user/d/dneff/`
- EOS: `/eos/user/d/dneff/`
- Scheduler: uses `myschedd` (Dylan has set up a custom schedd previously;
  use `condor_submit -sched $(myschedd)` or check `myschedd` output first)
- Shell: bash
- Has used HTCondor before with `queue run from` syntax for X17 analysis jobs

### Environment on lxplus worker nodes
Use the **LCG 108 view** — it includes ROOT and Garfield++ pre-built:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/setup.sh
```
This provides ROOT 6.32+ and Garfield++ 2025.x. Note: this ROOT version also
does not have `ROOT.Double`/`ROOT.Long` — ctypes approach is correct here too.

### EOS paths for HTCondor jobs
Gas files (read-only inputs) live at:
```
/eos/user/d/dneff/garfield_sim/gas_tables/<gas>_<pressure>.gas
```
Per-job output fragments go to:
```
/eos/user/d/dneff/garfield_sim/jobs/<gas>_<pressure>_<voltage>V_b<batch>.json
```
Final merged results go to:
```
/eos/user/d/dneff/garfield_sim/results/<gas>_<pressure>.json
```
HTCondor logs go to:
```
/eos/user/d/dneff/garfield_sim/logs/
```

**Important**: EOS is accessible from lxplus interactive nodes and from worker
nodes via the standard EOS FUSE mount at `/eos/`. No special setup needed on
workers — just use the path directly.

---

## 5. The Task: HTCondor Distributed Gain Scan

### What to build

Three new files (add to the repo, don't touch existing files):

#### `mm_condor_worker.py`
Single-job worker script. Runs on an lxplus worker node. Takes CLI args:
- `--gas-file PATH` — full EOS path to .gas file
- `--penning-mode auto|manual`
- `--penning-rP FLOAT` — (only if manual)
- `--penning-gas STR` — noble gas name (only if manual, e.g. "he")
- `--voltage FLOAT` — mesh voltage in V
- `--events INT` — number of avalanches to simulate
- `--gap-cm FLOAT` — amplification gap in cm (default 0.015)
- `--output PATH` — full EOS path to write JSON output fragment
- `--batch-id STR` — unique identifier for this batch (used in output filename)

Loads gas from EOS, runs avalanches, writes a **fragment JSON**:
```json
{
  "gas_label": "He_C2H6_96p5_3p5",
  "pressure_label": "Saclay_160m",
  "pressure_torr": 745.8,
  "voltage": 475.0,
  "field": 31666.7,
  "batch_id": "003",
  "n_events_attempted": 200,
  "gain_raw": [1234, 567, ...],
  "n_attached": 12,
  "runtime_s": 183.4
}
```

#### `mm_condor_submit.py`
Submission script. Run interactively on lxplus. Does:
1. Reads gas/pressure/voltage config (hardcoded for this specific run — see §6)
2. For each (gas, pressure, voltage): submits `batches_per_point` HTCondor jobs,
   each running `events_per_batch` avalanches
3. Writes a single `.jdl` file and calls `condor_submit`
4. Prints a summary of submitted jobs and a `condor_q` command to monitor

Key considerations:
- Use `getenv = True` in JDL so the LCG environment is inherited — BUT actually
  it's cleaner to source the LCG view inside the job executable wrapper
- Each job needs a small bash wrapper that sources the LCG view then calls
  `mm_condor_worker.py` — call it `mm_condor_job.sh`
- Job output goes to per-job fragment files (not merged — collector does that)
- Use `request_cpus = 1`, `request_memory = 2GB`, `request_disk = 1GB`
- `+MaxRuntime = 3600` (1 hour wall time per job — conservative)
- Use `myschedd` for submission

#### `mm_condor_collect.py`
Collector script. Run interactively on lxplus (or locally if EOS is mounted)
after jobs complete. Does:
1. Scans the `jobs/` directory for all fragment JSONs
2. Groups them by (gas_label, pressure_label, voltage)
3. For each group, concatenates all `gain_raw` lists and sums `n_attached`
4. Loads any existing merged result (from `results/`) and appends to it
   (same append logic as `load_existing_raw` in `mm_gain_scan_parallel.py`)
5. Recomputes statistics and writes final result JSONs in the standard schema
6. Prints a summary table: gas × pressure × voltage → total events collected

Also supports `--no-append` to discard existing results and start fresh.

#### `mm_condor_job.sh`
Bash wrapper sourced by HTCondor. Template:
```bash
#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/setup.sh
cd /path/to/repo   # or use transferred files
python3 mm_condor_worker.py "$@"
```

---

## 6. Specific Run Parameters

This is the concrete run Dylan wants submitted:

### He/C₂H₆ 96.5/3.5% — both Saclay and CERN
- Voltages: 450, 460, 470, 480, 490, 500, 510, 520, 530 V (step 10 V, 9 pts)
- Target: 2000 events per (pressure, voltage) point
- Strategy: 10 batches × 200 events = 2000 total
- Gas files:
  - `He_C2H6_96p5_3p5_Saclay_160m.gas`
  - `He_C2H6_96p5_3p5_CERN_450m.gas`
- Penning: manual, rP=0.40, gas="he"

### Ar/iC₄H₁₀ 95/5% — both Saclay and CERN
- Voltages: 400, 410, 420, 430, 440, 450, 460, 470, 480, 490 V (step 10 V, 10 pts)
- Target: 2000 events per (pressure, voltage) point
- Strategy: 10 batches × 200 events = 2000 total
- Gas files:
  - `Ar_iC4H10_95_5_Saclay_160m.gas`
  - `Ar_iC4H10_95_5_CERN_450m.gas`
- Penning: auto

### Total jobs
- He: 2 pressures × 9 voltages × 10 batches = 180 jobs
- Ar: 2 pressures × 10 voltages × 10 batches = 200 jobs
- **Total: 380 jobs**

These should all run in parallel on HTCondor. Each job runs ~2–15 min depending
on gas and voltage (higher voltage = higher gain = more electrons to track = slower).

### Adding more events later
Running `mm_condor_submit.py` again and then `mm_condor_collect.py` will append
to existing results (because `mm_condor_collect.py` uses the same append logic).
Use `--no-append` on the collector to start fresh.

---

## 7. Pressure Values

```python
import math
def altitude_to_torr(h_m):
    p_pa = 101325.0 * math.exp(-h_m / 8500.0)
    return p_pa * 0.00750062

PRESSURES = {
    "Saclay_160m": altitude_to_torr(160),  # ≈ 745.8 Torr
    "CERN_450m":   altitude_to_torr(450),  # ≈ 720.8 Torr
}
```

---

## 8. Important Gotchas

1. **ctypes, not ROOT.Double/ROOT.Long** — these were removed in ROOT 6.22.
   All output args use `ctypes.c_int(0)` / `ctypes.c_double(0.)` and read
   back via `.value`.

2. **ElectronTownsend takes 7 args**: `(ex, ey, ez, bx, by, bz, alpha)`.
   Field is along z, so call as `(0., 0., e_field, 0., 0., 0., alpha)`.

3. **Seed electron placement**: inject at `z = gap_cm` (the mesh), drifts
   toward `z = 0` (the anode). Call:
   `aval.AvalancheElectron(0., 0., gap_cm, 0., 0.)` — last arg is KE in eV.

4. **ComponentConstant geometry**: field in +z, area set with:
   ```python
   cmp.SetArea(-1., -1., 0., 1., 1., gap_cm)
   cmp.SetElectricField(0., 0., e_field)
   ```

5. **Penning must be applied AFTER LoadGasFile** — it scales the Townsend
   coefficient based on collision frequencies stored in the .gas file.

6. **ROOT gErrorIgnoreLevel**: set `ROOT.gErrorIgnoreLevel = ROOT.kWarning`
   to suppress the verbose Magboltz/Garfield info messages in job logs.

7. **EOS latency**: EOS reads/writes from worker nodes are reliable but slower
   than local disk. Writing fragment JSONs (small, ~50KB each) is fine.
   Don't write ROOT files from workers — JSON only.

8. **Job environment**: worker nodes run el9 (RHEL9). The LCG_108 view is
   the correct one for this OS. Don't use el7 views.

9. **myschedd**: Dylan uses a custom HTCondor schedd. The submit script should
   call `myschedd` to get the schedd name and pass it to `condor_submit -name`.
   If `myschedd` is not available, fall back to default (omit `-name` flag).

10. **Fragment file naming**: use a format that's unambiguous and sortable:
    `<gas>_<pressure>_<voltage>V_b<batch_id:03d>.json`
    e.g. `He_C2H6_96p5_3p5_Saclay_160m_475V_b003.json`

---

## 9. Files NOT to Modify

- `mm_config.py`
- `mm_generate_gas.py`
- `mm_gain_scan.py`
- `mm_gain_scan_parallel.py`
- `mm_plot.py`

The collector output must be compatible with `mm_plot.py` (reads the standard
JSON schema from `results/`).

---

## 10. Starting Point for Claude Code

Open the repo directory. Read the existing files first, particularly:
- `mm_config.py` — for gas/pressure definitions and constants
- `mm_gain_scan_parallel.py` — for the `load_existing_raw`, `_recompute_stats`,
  and `_assemble_and_save` functions which the collector should reuse or
  replicate faithfully

Then implement the three new files in this order:
1. `mm_condor_worker.py` (the actual physics, simplest)
2. `mm_condor_job.sh` (bash wrapper)
3. `mm_condor_submit.py` (submission logic)
4. `mm_condor_collect.py` (result merging)

Test `mm_condor_worker.py` locally first with a small `--events 5` run to
verify it produces valid fragment JSON before submitting to HTCondor.
