# DAQ fork study — banco (CERN P2/SPS) vs mx17-daq (nTOF bench)

**Written:** 2026-07-25, ~01:00–01:30 CEST, overnight, by Claude (Opus 5) at Dylan's request.
**Audience:** the model picking this up in the morning.
**Question asked:** meticulous comparison of the two DAQs — specifically `dream_daq_control.py`
and `daq_control.py` — and *"why does the banco machine have so many environment variables
being set?"*

Everything below marked **[verified]** was executed and observed during this session.
Everything marked **[inferred]** is my reading of the code/history and has *not* been
confirmed by running it. Please keep that distinction — several of the interesting claims
are inferred.

---

## 0. TL;DR

1. **The two DAQs are the same codebase, forked.** Shared history up to
   `82d44da3` (2026-07-04). Since then: banco +57 commits, mx17-daq +124. 318 commits shared.
2. **The env-var question has two independent answers**, and they get conflated:
   - **(a) Application-level `DAQ_*` vars** exist because banco has **one** run-config file
     (`run_config_beam.py`) that must express *every* run type, so run selection is encoded as
     environment variables. The bench has **45 separate `run_configs/*.py` files** instead, so
     it needs almost none.
   - **(b) Shell-level vars** (`ROOTSYS`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `XILINXD_LICENSE_FILE`, …)
     exist because banco is a **shared multi-user lab machine** with ROOT, EPICS, Xilinx, the ISEG
     SDK and FEU firmware tooling all sourced from one `.bashrc`. mx17-daq is a dedicated
     single-purpose box whose `.bashrc` sources ROOT and nothing else.
3. **The env-var design has already caused two real failures**, both found tonight — the
   GUI could not start any scan, and the GUI Start Run button was broken outright (§4.3).
4. **Both forks independently solved the same "RunCtrl won't exit" problem, differently**,
   and neither knows about the other's fix (§5.2). This is the single most valuable
   cross-pollination candidate.
5. `Client.py`, `common_functions.py`, `DAQController.py`, `get_config_py.py`,
   `run_config_base.py` are **byte-identical** across both machines. The divergence is entirely
   in `dream_daq_control.py`, `daq_control.py`, `hv_control.py`, and the config layer.

---

## 1. Machine and process inventory [verified]

| | **banco** | **mx17-daq** |
|---|---|---|
| ssh alias | `banco_cern` (128.141.21.144) | `daq` (128.141.177.17) |
| hostname | `dedippcq196.extra.cea.fr` | `mx17-daq` |
| user | `banco` (shared) | `mx17` (dedicated) |
| repo path | `~/DAQ_Control_Dream_Beam` | `~/PycharmProjects/nTof_x17_DAQ` |
| GitHub | `akallitss/DAQ_Control_Dream_Beam` | `Dyn0402/nTof_x17_DAQ` |
| commits | 375 | 442 |
| top-level `*.py` | 29 | 42 |
| run-config generators | 3 | 46 (`run_configs/`) |
| tmux sessions | 9 | 14 |
| `.bashrc` | 173 lines, 8 exports | 137 lines, 2 exports |

**banco tmux:** `backup_watcher, beam_watcher, daq_control, dream_daq, flask_server, hv_control,
mem_guardian, pedestal_watcher, processor_watcher`

**mx17-daq tmux:** the same minus `mem_guardian`, plus `claude_daq, gas_watcher,
he3_pressure_watcher, space_watcher, stream1_watcher, system_stats_watcher`

> **mx17-daq had a live run in progress** during this study
> (`daq_control.py run_config_overnight_hv_stats.json`, ~2h42m elapsed at 01:05).
> **I touched nothing on that machine — read-only throughout.** Verify it survived the night
> before drawing conclusions from its data.

Shared home directory on banco contains `benjamin/ camille/ fabien/ francesco/ gregoire/
maxence/ yann/ ak271430/ dylan/` — this is the key context for §4.

---

## 2. Fork ancestry [verified]

Both repos begin with identical commits:

```
3801df3 2025-11-30 Initial commit
f57bfa9 2025-12-09 Added initial copy of EIC DAQ and started ntof version. Nothing working yet
39f3750 2026-01-15 Added flask_app and started a bit
```

Computed by intersecting full commit-hash lists from both machines:

```
banco commits: 375   daq commits: 442   SHARED: 318
fork point   : 82d44da3  2026-07-04  "Post-sub-run pause: manual button + optional
                                      configured per-sub-run pause"
banco-only   : 57
daq-only     : 124
```

So they ran as one codebase until **2026-07-04** and have diverged for three weeks. Both are
descendants of `Cosmic_Bench_DAQ_Control` (still visible in file docstrings:
*"Created as Cosmic_Bench_DAQ_Control/..."*).

**Implication:** a merge is genuinely feasible — this is not a rewrite-vs-rewrite situation.
The shared 318 commits mean `git` can do most of the work. Nobody appears to have tried.

---

## 3. What is byte-identical [verified]

```
IDENTICAL : Client.py            (93 lines)
IDENTICAL : common_functions.py  (144)
IDENTICAL : DAQController.py     (99)
IDENTICAL : get_config_py.py     (29)
IDENTICAL : run_config_base.py   (45)
DIFFERS   : iterate_run_num.py   (3 changed lines)
```

The client/server transport layer, the shared helpers and the config base class have **not**
diverged at all. Whatever merge strategy gets chosen, this whole layer is free.

---

## 4. THE ENVIRONMENT VARIABLE QUESTION

This is the part Dylan actually asked about. There are two distinct phenomena.

### 4.1 Application-level `DAQ_*` vars — a symptom of one-file-vs-many-files [verified]

banco's `run_config_beam.py` reads **22 distinct `DAQ_*` variables**:

| Variable | Default | Purpose |
|---|---|---|
| `DAQ_SITE` | `local` | `local` (simulate) vs `sps` (real hardware) |
| `DAQ_TRIGGER` | `external` | `external` (beam) vs `self` (Fe55) |
| `DAQ_RUN_PLAN` | `drift_then_mesh` | **added by me tonight**, see §7 |
| `DAQ_RUN_NAME` | — | override the run name |
| `DAQ_POWER_OFF` | `1` | power HV off at end of run |
| `DAQ_BEAM_DRIFT_SCAN` | `0` | enable drift scan mode |
| `DAQ_DRIFT_START_V` / `_STEP_V` / `_POINTS` / `_SUBRUN_MIN` | `450`/`50`/`10`/`10` | drift scan shape |
| `DAQ_BEAM_HV_SCAN` | `0` | enable mesh scan mode |
| `DAQ_MESH_NOMINAL` / `_POINTS` / `_STEP_V` / `_SUBRUN_MIN` | `2`/`6`/`10`/`20` | mesh scan shape |
| `DAQ_LATENCY_SCAN` | `0` | enable latency scan mode |
| `DAQ_P2IN_CHECK` / `_MIN` / `_DRIFT` / `_MESH` | `0`/`12`/`600`/`400` | P2_IN alive-check mode |

The bench, by contrast, reads env vars in `run_configs/*.py` — but they are almost all
**within-experiment fine-tuning**, not experiment selection:

```
run_configs/run_config_overnight_hv_stats.py:  SUBRUN_MIN, N_CYCLES, DREAM_ZS, DET_D_RES_OFFSET
run_configs/run_config_cosmics_hv_bounce.py:   SUBRUN_MIN, N_CYCLES, DET_D_OFFSET
run_configs/run_config_clock_window_test.py:   CLK_BLOCKS, CLK_RUN_NAME
run_configs/run_config_pace.py:                PACED_PERIOD_MS
...
```

**The structural difference, stated plainly:**

```
banco :  1 config file  x  22 env vars   -> "which experiment" is an ENV VAR
bench : 46 config files x  ~2 env vars   -> "which experiment" is a FILE
```

Raw counts of `os.environ` reads outside `.venv` are nearly identical (**banco 45, bench 47**) —
so it is *not* that banco uses more environment. It uses them for a **different job**: banco
encodes *run-type selection*, the bench encodes *parameter tweaks within an already-chosen run type*.

`run_configs/README.md` on the bench documents the choice explicitly:

> Experiment-specific **run-config generators**. Each `run_config_*.py` here builds a JSON into
> `config/json_run_configs/` (that JSON is what actually drives a run).
> *Moved out of the repo root on 2026-07-24 for tidiness.*

### 4.2 Why that design was chosen [inferred]

I did not find a design document. My reading:

- banco's `run_config_beam.py` began as a *single beam config* and grew mode-by-mode as the
  test beam demanded new scans (latency → mesh → drift → P2_IN check). Each new mode was
  cheapest to add as another `if ENV_FLAG:` branch rather than a new file.
- The bench had many more distinct experiments (46) over a longer period, and hit the
  readability wall earlier — hence the split into `run_configs/`.
- banco is operated by **several people** (Alexandra, Dylan, others). A one-file config with
  env switches is arguably easier to hand over verbally (*"run it with `DAQ_BEAM_DRIFT_SCAN=1`"*)
  than a directory of 46 files. That is a real, if fragile, benefit.

### 4.3 Why it is actively harmful — two real failures, both found tonight [verified]

**Failure 1 — the GUI cannot set environment variables, so the scans were unreachable.**

Flask's Start Run regenerates the config by running `python run_config_beam.py` as a subprocess
(`flask_app/app.py:378`). A subprocess inherits Flask's environment; the browser cannot inject
anything. So every env-gated mode was **impossible to start from the GUI**. The workaround was
`~/overnight_scans.sh` — a shell script whose *entire purpose* is to set env vars around two
`daq_control.py` invocations:

```bash
DAQ_SITE=sps DAQ_TRIGGER=external DAQ_BEAM_DRIFT_SCAN=1 \
  DAQ_RUN_NAME=drift_scan_final DAQ_POWER_OFF=0  ... python - ...
```

That script failed twice on the night of 07-24 (23:43 and 00:16), produced no data, and left no
usable log. See §8.

**Failure 2 — an env var silently broke the Start Run button entirely.**

`export DAQ_SITE=sps` lives in banco's `~/.bashrc:164`. Flask is started from a tmux login
shell, so it inherits `DAQ_SITE=sps` → `SIMULATE=False` → `run_config_beam.py` prints

```
WARNING: /local/home/banco/DAQ_Control_Dream_Beam/hv_creds.txt not found — using default admin/admin HV credentials.
```

…**to stdout**. But `get_config_py.py` runs `run_config_beam.py` and parses its **stdout as JSON**
to show the run name in Start Run's confirmation dialog. So:

```python
json.loads('WARNING: ...hv_creds.txt not found...\n{"run_name": "..."}')
# ValueError: Expecting value: line 1 column 1 (char 0)
```

`/get_config_py` returned 500 and the button aborted before doing anything. This is guarded by
`if not SIMULATE`, so it fires **only on the real site** — invisible in any local test.
I fixed it (warning → `sys.stderr`, commit `9218fb4`) and verified the route now returns
`{"run_name":"drift_mesh_scan_1","success":true}`.

**The pattern to take away:** environment-as-configuration is invisible to the GUI, invisible in
logs, and inherited from `.bashrc` in ways nobody remembers. Both failures are *directly* caused
by it. The bench's file-per-experiment approach has neither failure mode, because the artifact
that drives a run is a **JSON file on disk you can read**.

### 4.4 Shell-level environment — the other half of the question [verified]

banco `~/.bashrc`:

```bash
120: export XILINXD_LICENSE_FILE=2100@irfupcg128
126: export ROOTSYS=/local/home/banco/P2/root
127: source /local/home/banco/P2/root/bin/thisroot.sh      # sets PATH, LD_LIBRARY_PATH, PYTHONPATH
158: export ISEG_SDK_PATH=".../icsPythonForLinux/platform/linux/64"
159: export PYTHONPATH=$PYTHONPATH:$ISEG_SDK_PATH
160: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISEG_SDK_PATH
163: export PATH="$PATH:/local/home/banco/Feu/Firmware/.../Linux/bin"
164: export DAQ_SITE=sps
170: export P2_BASE=/local/home/banco/P2_data/TB_July2026_H4
171: export SPS_DATA_ROOT=$P2_BASE/runs
172: export SPS_ANALYSIS_ROOT=$P2_BASE/analysis
173: export SPS_COSMIC_BENCH_ROOT=$P2_BASE/runs
```

mx17-daq `~/.bashrc`:

```bash
125: source /home/mx17/Software/root/bin/thisroot.sh
131: export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
```

**Why banco has so much:** it is a **shared CEA/Saclay lab machine** hosting FPGA development
(Xilinx), the P2 ROOT build, the ISEG HV SDK, EPICS, and FEU firmware tooling — for at least
nine different users. mx17-daq is a **dedicated nTOF DAQ box** that does one job.

**The concrete damage:** ROOT's `thisroot.sh` and the ISEG SDK both prepend to `PYTHONPATH` and
`LD_LIBRARY_PATH`, which then leak into the repo's `.venv` Python and can shadow venv packages
or load mismatched shared libraries. Somebody already discovered this the hard way —
`overnight_scans.sh:6` carries the scar tissue:

```bash
ENV="env -u LD_LIBRARY_PATH -u PYTHONPATH"
```

Every Python invocation in that script is wrapped in it. **This is undocumented anywhere else in
the repo**, and any new script written on banco that forgets the wrapper is a latent bug.
[inferred] I did not reproduce a concrete failure from the leaked paths — only observed the
defensive wrapper and confirmed the vars are set. **Worth reproducing in the morning**: run
`.venv/bin/python -c "import sys; print(sys.path)"` with and without the wrapper and diff.

---

## 5. `dream_daq_control.py` — the deep comparison

banco **1019** lines, bench **958**, **477** differing lines. Both forks added substantial,
*non-overlapping* functionality.

### 5.1 Function inventory [verified]

```
shared (17): _to_bit, _to_zs_typ, clear_terminal, copy_files_on_the_fly, file_num_still_running,
             get_pedestals, get_tmux_pane, listen_for_stop, main, make_config_from_template,
             move_data_files, prepare_terminal, repl, runctrl_batch_watchdog, set_active_feus,
             set_dream_roles, update_config_value

banco only (4): pedestal_run_watchdog, run_timeout_watchdog,
                set_feu_mem_file_refs, clear_feu_mem_file_refs

bench only (1): _skip  (nested helper in copy_files_on_the_fly)
```

### 5.2 ⭐ The same bug, two different fixes — highest-value finding

Both machines run a RunCtrl build that **fails to exit in batch mode** and drops to its
interactive `***` prompt, which would hang a sub-run forever. Each fork fixed it independently:

**banco's fix — watchdog threads that kill the process:**

```python
# pedestal_run_watchdog: poll for per-FEU _ped.prg/_thr.prg; when all present,
# grace 10 s, then SIGINT (cleanup trap restores terminal), then kill.
# Falls back to go_timeout so a failed run cannot hang the DAQ.

# run_timeout_watchdog: for data runs, SIGINT once run_time*60 +
# max_run_time_addition has elapsed.
```

**the bench's fix — make RunCtrl exit on its own, then clean up the junk:**

```python
# A pedestal run enables BOTH PedThrRun and DataRun: the data phase exists only so
# batch RunCtrl self-exits, and its _datrun_ FDFs are empty junk to be discarded
discard_data_run = bool(effective_do_pedestal_threshold_run) and bool(effective_do_data_run)
# ...copier is given skip_substrings=['_datrun_'], and the files are deleted afterwards
# because 0-byte FDFs can deadlock the processor.
```

**Assessment [inferred]:** banco's is the better fix. It addresses the failure directly
(kill the hung process) rather than working around it by generating garbage data that then has to
be filtered from the copier *and* deleted from two directories *and* guarded against deadlocking
the processor. The bench's approach has three places to get wrong; banco's has one. **But**
banco's watchdog is the newer, less-exercised code, and `pedestal_run_watchdog`'s completion test
(`all FEUs have both `_ped.prg` and `_thr.prg`) would hang until `go_timeout` if a FEU is
legitimately absent from `included_feus` — worth reviewing before porting.

Recommend porting banco's two watchdogs to the bench and retiring `discard_data_run`.
**Do not do this blind** — the bench is mid-run and its pedestal path is load-bearing.

### 5.3 What the bench has that banco lacks — FEU/firmware tuning [verified]

The bench threads **14 extra hardware knobs** from run config → `.cfg`, which banco does not
have at all:

```
sample_period, inter_packet_delay, multipack_thr, multipack_enb, sparse_rd,
rdclk_div, wrclk_div, rd_del, adc_dat_rdy_del, trig_veto_len,
ovr_wrn_hwm, ovr_wrn_lwm, loc_throt, daq_run_events
```

Plus `SAMPLE_PERIOD_CLOCK_DIVS`, mapping sample period → SCA clock dividers. These carry
genuinely deep commentary — measured rates, firmware register addresses, safety caveats:

> *"Measured 2026-07-23: read clock 6.0→4.0 (16.7→25 MHz) buys a clean 1.5x rate (7231→10847 Hz,
> 0 drops, tracers 100%, baseline unchanged)… CAVEAT: 25 MHz is a mild overclock of the ASIC's
> rated 20 MHz RCk."*

This is the "bells and whistles" Dylan mentioned. It is nTOF-10G-specific work (`docs/CLOCK_RATE_SCAN_2026-07-23.md`)
and probably **should not** be pushed to banco wholesale — but `daq_run_events` exists on both
with *different semantics*, which is a latent trap:

- bench: *"Per-FEU event cap. RunCtrl stops at whichever comes FIRST: this many events/FEU or Sys DaqRun Time"*
- banco: *"caps a run by event count; 0 = infinite. Run length is governed by Sys DaqRun Time"*

[inferred] These may describe the same hardware behaviour in different words, or may reflect a
real divergence. **Verify before merging anything in this area.**

### 5.4 What banco has that the bench lacks [verified]

- **`sim/` package** (`fake_dream_daq.py`, `fake_caen.py`) — full offline simulation, entered via
  `effective_info.get('simulate')`, replaying sample FDFs instead of driving RunCtrl.
  The bench has **no `sim/` directory at all**. This is why `DAQ_SITE=local` is meaningful on
  banco and meaningless on the bench. Strong candidate to port — it makes DAQ logic testable
  without hardware.
- **`set_feu_mem_file_refs` / `clear_feu_mem_file_refs`** — rewrites the run `.cfg` to point at the
  copied `.prg` pedestal files, and warns loudly when pedestals are missing while ZS is on:
  ```
  'No pedestals found in %s and PedThrRun is off — FEUs will zero-suppress with unprogrammed thresholds'
  ```
  That warning is exactly the failure mode that bit the P2 beam test. Port it.
- **Explicit trigger-source forcing** — banco writes `Sys DaqRun Trig = 'Slf'|'Ext'` from the run
  config, *"so a template/mode mismatch can't silently take data on the wrong trigger."*
  The bench inherits whatever the template ships. **This is a genuine robustness gap on the bench.**
- **TCM multiplicity window** passthrough (`trg_mult_more_than` / `trg_mult_less_than`).
- Ordering fix: banco copies the `.cfg` to the raw dir **after** writing pedestal refs, *"so the
  archived copy is the cfg RunCtrl actually used."* The bench copies before. The bench's archived
  cfg is therefore **not** what ran. Small, real, easy to port.

---

## 6. `daq_control.py` and `hv_control.py`

### 6.1 `daq_control.py` — banco 343 lines, bench 551, 210 differing [verified]

```
shared (7): _remove_flag, _sleep_unless_stop, check_weiner_lv_status,
            file_num_still_running, found_file_num, main, run_daq_controller
banco only: (none)
bench only: _snapshot_n1081b, _make_scan_control, _apply_n1081b_with_retry
```

**banco has added nothing here.** The entire 208-line difference is the bench's **N1081B
discriminator integration** — in-process trigger-board control that replaced a standalone
`n1081b_scan_watcher.py`. Notable design choices worth stealing conceptually:

- **Fail-closed pre-flight:** *"REFUSING to start"* if scan control can't be built and the config
  didn't explicitly opt out with `n1081b_scan="off"`.
- **Verified-by-readback apply with hold-and-retry:** on failure it sets the PAUSE flag and holds
  the run rather than taking data with an unknown trigger config.
- **Default OFF:** a config must opt in before anything touches the trigger boards.

This is nTOF-specific hardware and does not belong on banco. The *pattern* — verify hardware
config by read-back, fail closed, hold rather than take bad data — absolutely does.

### 6.2 `hv_control.py` — banco 207 lines, bench 306, 245 differing [verified]

```
shared: main, monitor_hvs, power_off_hvs, set_hvs
banco only: get_hv_controller
bench only: HVRampError, _notify_alerter, _set_and_wait_for_ramp  (+ hv_alerts.HVAlerter)
```

The bench has a materially more defensive HV path:

- `HVRampError` + `DEFAULT_RAMP_TIMEOUT_S` — *"Bound the ramp wait so a dead crate costs one
  sub-run rather than hanging the run."*
- `begin_ramp` / `end_ramp` bracketing so the alerter suppresses the deviation/over-current alerts
  a ramp inevitably trips, watches for a **stalled** ramp instead, and is told explicitly when a
  ramp fails *"so the skipped sub-run is not silent."*
- On `HVRampError`, `daq_control` skips the sub-run and reports over the server socket.

**banco has none of this.** [inferred] On banco a dead or stuck CAEN crate during a ramp
plausibly hangs the sub-run indefinitely. Given banco runs unattended overnight scans, this is
the **highest-risk gap I found in the banco direction**. Recommend porting the bench's ramp
timeout + `HVRampError` path to banco with priority.

Also note banco is running with **no `hv_creds.txt`**, falling back to `admin`/`admin`
[verified — the crate accepts it; HV ramped and monitored correctly during my pedestal test].

---

## 7. State I left banco in — read this before touching anything

**Four commits on `main`, none pushed** (`Dyn0402` has read-only access to
`akallitss/DAQ_Control_Dream_Beam`; push rejected for `main` *and* for a new branch):

```
9218fb4  Send the hv_creds warning to stderr — it was breaking GUI Start Run
c5e222c  RUN_PLAN knob: make the drift+mesh scans startable from the GUI
11b9810  banco sps site: point reconstruction_build at mm_strip_reconstruction
bb08dbb  Decode watchdog + loud Git Reset confirmation
47e9f14  <- origin/main
```

⚠️ **The GUI's "Git Reset" button runs `git reset --hard origin && git pull` and will destroy
all four.** A confirm dialog now pops first (`bb08dbb`) but it warns about *uncommitted* changes,
not unpushed commits. Backups: `~/precommit_backup_260725_0040/` (originals + `.patch` files +
a git bundle), and `stash@{0}` holds the pre-rebase `run_config_beam.py`.

**`RUN_PLAN = 'drift_then_mesh'` is now the default**, so GUI Start Run produces
`drift_mesh_scan_1`: 10 drift points (450→900 V) then 1 nominal + 12 mesh points (−5 V steps),
10 min each = **23 sub-runs, 230 min, HV off at end**. Verified by generating the schedule; **not
yet actually run.**

Two open judgement calls Dylan has not answered:
1. **P2_IN steps down to 570/370 during the mesh half** (`BEAM_SCAN_DETS` includes it on
   `origin/main`). P2_IN was only just reinstated after repair — he may want it held fixed.
2. **Pedestals do not cover FEU 3.** The set I took at 00:29 predates the P2_IN reinstatement and
   covers FEUs 1/4/5 only; ZS is on. Retaking pedestals is ~2 min and now includes FEU 3.

---

## 8. Loose end: the failed overnight scan [verified]

`~/overnight_scans.sh` ran twice (23:43, 00:16), died in the first sub-run both times, produced
no data, and the mesh half never started. Evidence:

- `overnight_scans.log` stops dead at `DRIFT scan: run` both times — no `DRIFT scan: finished`,
  so the **bash script itself** was SIGTERM'd.
- `dream_daq` pane: `FileNotFoundError: .../dream_run/drift_scan_final/drift_450/`
- `dream_run/` is now **empty, mtime 00:23**.
- Ruled out: memory (`mem_guardian` logged no kill since 07-19, 59 GB free), disk (288 GB free),
  the GUI Stop button (no `STOP_RUN` in `daq_events.log` since 07-24 14:23), and the
  `git fetch` failure (non-fatal — `git show` used the cached ref).
- **Prime suspect [inferred]:** the GUI Disk Space tab (`flask_app/space_manager.py`, subject of
  banco's three most recent commits) deleting the run directory out from under the DAQ. Nothing
  logs deletions from that tab, so this is unproven.

**Diagnosability bug worth fixing regardless:** `daq_control.py`'s stdout is block-buffered into
`>>"$LOG"`, so SIGTERM discards everything buffered. Add `PYTHONUNBUFFERED=1` or `python -u`.

---

## 9. Recommended actions, ranked

**Port bench → banco:**
1. **HV ramp timeout + `HVRampError` + alerter bracketing** (§6.2). Highest risk gap; banco runs
   unattended.
2. `PYTHONUNBUFFERED=1` on all logged Python invocations (§8).

**Port banco → bench:**
3. **`pedestal_run_watchdog` / `run_timeout_watchdog`**, retiring `discard_data_run` (§5.2).
4. **Explicit `Sys DaqRun Trig` forcing** — prevents silently taking data on the wrong trigger (§5.4).
5. **Missing-pedestals-with-ZS-on warning** + `set_feu_mem_file_refs` (§5.4).
6. `.cfg`-copied-after-pedestal-refs ordering fix (§5.4).
7. The `sim/` package, to make DAQ logic testable without hardware (§5.4).

**Structural, banco:**
8. **Get write access to `akallitss/DAQ_Control_Dream_Beam`, or fork it under `Dyn0402`.** Four
   commits currently exist on one disk with a button that deletes them.
9. **Split `run_config_beam.py` into `run_configs/`** following the bench's model, retiring the
   `DAQ_*` mode-selection vars. `RUN_PLAN` (added tonight) is a stopgap that keeps the monolith.
10. Move `export DAQ_SITE=sps` out of `.bashrc` into an explicit launcher, so config no longer
    depends on which shell started Flask (§4.3).
11. Document `env -u LD_LIBRARY_PATH -u PYTHONPATH` in the repo README, or fix it properly by
    having scripts call `.venv/bin/python` with a sanitized environment (§4.4).

**Verify first (do not merge blind):**
12. `daq_run_events` semantics differ between the forks (§5.3).
13. `pedestal_run_watchdog` behaviour when a FEU in `included_feus` never produces `.prg` (§5.2).
14. Whether leaked `PYTHONPATH`/`LD_LIBRARY_PATH` actually breaks the venv on banco (§4.4).

---

## 10. Reproducing this study

```bash
# fetch the core files from both machines
for f in dream_daq_control.py daq_control.py common_functions.py Client.py \
         DAQController.py hv_control.py iterate_run_num.py get_config_py.py run_config_base.py; do
  scp banco_cern:DAQ_Control_Dream_Beam/$f            banco/$f
  scp daq:PycharmProjects/nTof_x17_DAQ/$f             daq/$f
done
diff -u daq/dream_daq_control.py banco/dream_daq_control.py

# fork point: intersect full commit-hash lists
ssh banco_cern 'cd ~/DAQ_Control_Dream_Beam && git log --format="%H %ad %s" --date=short' > banco_log.txt
ssh daq 'cd ~/PycharmProjects/nTof_x17_DAQ && git log --format="%H %ad %s" --date=short' > daq_log.txt

# env inventory (exclude .venv or you get 100+ lines of vendored noise)
ssh banco_cern 'cd ~/DAQ_Control_Dream_Beam && grep -rn "os.environ" --include="*.py" . \
                | grep -vE "\.venv/|flask_app/local/"'

# reproduce a flask subprocess exactly: same argv, cwd and environment
# (read /proc/<flask-pid>/environ and /proc/<flask-pid>/cwd — this is how the
#  Start Run stdout-pollution bug in §4.3 was isolated)
```

Working files from this session: `scratchpad/daqstudy/{banco,daq}/` and `*_log.txt`
(session scratchpad — will not survive indefinitely; re-fetch with the above if gone).

---

## 11a. FOLLOW-UP: full audit of banco's shell variables (2026-07-25, ~02:00)

Dylan asked to find every usage of the `.bashrc` shell variables and judge whether they can be
phased out. Done. **One of them is a live, confirmed bug**; most of the rest are dead weight.

### 11a.1 Verdict table

| Variable | Consumers found | Verdict |
|---|---|---|
| `PATH += Feu/.../Linux/bin` | `dream_daq_control.py:142` runs **`['RunCtrl', '-c', …]` by bare name** | ⚠️ **ESSENTIAL — keep** (or make the path explicit in code) |
| `DAQ_SITE=sps` | `run_config_beam.py:31` + 21 sibling `DAQ_*` vars | ⚠️ **Load-bearing but misplaced** — caused tonight's Start Run bug |
| `LD_LIBRARY_PATH += P2/root/lib` | *nothing in the repo* — but see §11a.2 | 🔴 **ACTIVELY HARMFUL — remove** |
| `PYTHONPATH += P2/root/lib` | *nothing* (0 `import ROOT` in the repo) | 🟡 Latent risk — remove |
| `PYTHONPATH/LD_LIBRARY_PATH += ISEG_SDK_PATH` | *nothing* (0 iseg/ics imports; `ISEG_SDK_PATH` referenced nowhere) | 🟢 **DEAD — remove** |
| `ROOTSYS` + `source thisroot.sh` | *nothing in the repo* | 🟡 Remove from `.bashrc`; source on demand |
| `XILINXD_LICENSE_FILE` | *nothing*; the Vivado `settings64.sh` line is **already commented out** | 🟢 **DEAD — remove** |
| `P2_BASE` / `SPS_DATA_ROOT` / `SPS_ANALYSIS_ROOT` / `SPS_COSMIC_BENCH_ROOT` | `P2_basket_analysis/sps_beam_analysis/sps_config.py` (+6 analysis scripts) | 🟡 **Real consumer** — but has an auto-detect fallback; see §11a.4 |

### 11a.2 🔴 The live bug: `LD_LIBRARY_PATH` loads the WRONG ROOT into the decoder [verified]

**Two ROOT installations exist on banco:**

```
/local/home/banco/P2/root          -> 6.26/10     (on LD_LIBRARY_PATH via .bashrc:127)
/local/home/banco/opt/root_v6.32.02 -> 6.32.02    (what the decoder was BUILT against)
```

The decoder binary carries a **`RUNPATH`**, not an `RPATH`:

```
$ readelf -d .../mm_strip_reconstruction/cmake-build-release/decoder/decode
 0x…1d (RUNPATH)  Library runpath: [/local/home/banco/opt/root_v6.32.02/lib]
```

**`LD_LIBRARY_PATH` takes precedence over `RUNPATH`** (it does not over `RPATH`). Measured:

```
##### ldd WITH .bashrc LD_LIBRARY_PATH (interactive login shell) #####
    libCore.so => /local/home/banco/P2/root/lib/libCore.so        <-- ROOT 6.26
    libRIO.so  => /local/home/banco/P2/root/lib/libRIO.so
    libTree.so => /local/home/banco/P2/root/lib/libTree.so

##### ldd WITHOUT LD_LIBRARY_PATH #####
    libCore.so => /local/home/banco/opt/root_v6.32.02/lib/libCore.so   <-- ROOT 6.32, correct
    libRIO.so  => /local/home/banco/opt/root_v6.32.02/lib/libRIO.so
    libTree.so => /local/home/banco/opt/root_v6.32.02/lib/libTree.so
```

So **a decoder built against ROOT 6.32.02 silently loads ROOT 6.26 shared libraries** whenever it
is launched from a shell that sourced `.bashrc`. That is a cross-minor-version C++ ABI mismatch.

This explains the otherwise-mysterious `ENV="env -u LD_LIBRARY_PATH -u PYTHONPATH"` in
`overnight_scans.sh:6` — somebody hit this and papered over it without recording why.

> **⚠️ HYPOTHESIS worth testing first thing [inferred, NOT verified]:** the decoder hangs that
> motivated banco's `processor_watcher` upgrade —
> *"The DreamDecoder can infinite-loop on certain FDFs (100% CPU, input position and output ROOT
> both frozen — seen on the banco P2 setup 2026-07-23/24)"* — may be **this ABI mismatch**, not a
> decoder bug. A 6.26 `libRIO`/`libTree` driving 6.32-compiled TTree code is exactly the kind of
> thing that wedges on some inputs and not others. **Test:** re-run one of the preserved `.hang`
> FDFs with and without `LD_LIBRARY_PATH`. If it only hangs with, the fix is one `.bashrc` line,
> not a watchdog. The `.hang` files were deliberately preserved as reproducers — use them.

**Current exposure [verified]:** all live DAQ processes are presently **clean** —
`processor_watcher`, `pedestal_watcher`, `dream_daq_control` and Flask all have
`LD_LIBRARY_PATH` unset in `/proc/<pid>/environ`. The poisoning path is a **human running
something by hand** from an interactive login shell — precisely how `overnight_scans.sh` was run.
So this is a loaded gun, not a currently-firing one.

### 11a.3 Why `DAQ_SITE` leaks but `LD_LIBRARY_PATH` doesn't [partly unresolved]

Measured on the pane shells:

```
bash pid=81175  parent=tmux: server    LD_LIBRARY_PATH_set=0   DAQ_SITE_set=1
bash pid=81306  parent=tmux: server    LD_LIBRARY_PATH_set=0   DAQ_SITE_set=1
bash pid=349732 parent=xfce4-terminal  LD_LIBRARY_PATH_set=0   DAQ_SITE_set=0
```

`.bashrc` has the standard Ubuntu non-interactive guard at line 6 (`case $- in … *) return;;`), so
it applies to interactive shells only. tmux panes carry `DAQ_SITE=sps` (inherited via the tmux
server's global environment, from whatever shell first started the server) but **not**
`LD_LIBRARY_PATH`. I could not fully explain that asymmetry in the time available — the exact
propagation path is **unresolved** and worth five minutes in the morning, because it determines
whether a `tmux send-keys`-launched run (i.e. **every GUI-started run**) is exposed.

What is certain: `DAQ_SITE=sps` reaching Flask via tmux is what made `SIMULATE=False`, which
emitted the stdout warning that broke Start Run (§4.3).

### 11a.4 `PYTHONPATH` — latent, not live [verified]

With `.bashrc` sourced, two foreign directories are injected **ahead of the venv's site-packages**:

```
sys.path:  /tmp
           /local/home/banco/P2/root/lib                                  <-- injected
           /local/home/banco/benjamin/Programs/icsPythonForLinux/…/64     <-- injected
           …
           /local/home/banco/DAQ_Control_Dream_Beam/.venv/lib/python3.12/site-packages
```

I checked for actual shadowing rather than assuming it:

```
ROOT lib exposes : 181 importable top-level names
ISEG SDK exposes : 2   ('icsClientPython_3', 'python-3.5.7')
venv site-pkgs   : 56
*** COLLISIONS: 0 ***
```

**No name currently collides**, and `numpy`/`flask`/`requests` all still resolve to the venv.
So this is a latent hazard, not a live failure — I want to be precise about that, since §4.4 of
this document was written before I had measured it. Two things still make it worth removing:
183 foreign names sit ahead of site-packages awaiting a future collision, and the search path
depends on **another user's home directory** (`benjamin/`, last modified 2020) — if that is
deleted or its permissions change, the DAQ's Python environment changes underneath it.

Note also that ROOT is on `PYTHONPATH` but **not importable** (`ImportError`, plus a
`SyntaxWarning` printed from `cppyy/__init__.py`) — it is built for a different interpreter than
the venv's 3.12. Dead weight that also emits warnings, which is the exact failure class that
broke Start Run.

### 11a.5 `P2_BASE` / `SPS_*` — real consumer, but replaceable [verified]

The only consumer is `~/P2_basket_analysis/sps_beam_analysis/sps_config.py`, which resolves roots
in three tiers:

```python
#   1. explicit env var  (SPS_DATA_ROOT / SPS_ANALYSIS_ROOT / SPS_COSMIC_BENCH_ROOT)
#   2. banco auto-detect (the DAQ machine: /local/home/banco exists)
#   3. laptop default
_ON_BANCO = os.path.isdir('/local/home/banco')
```

The banco auto-detect default is `/local/home/banco/P2_data` — one level **above** the campaign
directory the env vars point at (`…/P2_data/TB_July2026_H4/runs`). So the env vars are currently
load-bearing: drop them without editing `sps_config.py` and the analysis looks in the wrong place.

They are also duplicated state — `docs/project_notes/p2-sps-beam-daq-setup.md:82` already warns
that a new beamtest requires changing the campaign dir in **two** places, `.bashrc` *and*
`run_config_beam.py SITES['sps']`. That is the same one-source-of-truth problem as the `DAQ_*`
vars, in a different costume.

### 11a.6 Proposed phase-out

**Safe to delete outright — zero consumers found:**

```bash
export XILINXD_LICENSE_FILE=2100@irfupcg128        # Vivado settings64.sh is already commented out
export ISEG_SDK_PATH="…/benjamin/…/linux/64"       # nothing imports iseg/ics anywhere
export PYTHONPATH=$PYTHONPATH:$ISEG_SDK_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISEG_SDK_PATH
```

**Remove from `.bashrc`, source on demand** (nothing in the DAQ repo imports ROOT, and `RunCtrl`
does not link it — verified with `ldd`):

```bash
export ROOTSYS=/local/home/banco/P2/root
source /local/home/banco/P2/root/bin/thisroot.sh
```
Replace with an alias, so ROOT is opt-in per shell:
```bash
alias setup_root='source /local/home/banco/P2/root/bin/thisroot.sh'
```
⚠️ **Check with Alexandra first** — she may run interactive ROOT/`sps_beam_analysis` sessions and
expect ROOT on the default path. This is the only change in this list with a human cost.

**Keep, but move out of `.bashrc` into an explicit launcher** so config no longer depends on which
shell started a process:

```bash
export DAQ_SITE=sps        # -> bash_scripts/daq_env.sh, sourced by start_tmux.sh / start_flask.sh
export P2_BASE=…           # -> or better, kill entirely by setting the campaign in sps_config.py
export SPS_DATA_ROOT=…
```

**Keep as-is (essential):**

```bash
export PATH="$PATH:/local/home/banco/Feu/Firmware/Implementation/Projects/Software/Linux/bin"
```
`dream_daq_control.py:142` invokes `RunCtrl` by bare name, so this must be on `PATH` for any
process that starts a run. Making it explicit in code (an absolute path, or a `runctrl_bin`
config key) would be strictly better than depending on `.bashrc` — the DAQ currently cannot run
at all from a shell that skipped `.bashrc`.

**Order of operations (least → most risk):**
1. Delete the four dead exports (Xilinx, ISEG ×3). No consumers — zero risk.
2. Test the ABI hypothesis on a preserved `.hang` FDF (§11a.2). **Do this before anything else** —
   it may retire a whole class of decoder bugs.
3. Remove `ROOTSYS`/`thisroot.sh` from `.bashrc`, add the `setup_root` alias, after checking with
   Alexandra.
4. Move `DAQ_SITE` into an explicit launcher script.
5. Make the `RunCtrl` path explicit in code.
6. Longer term: fold `P2_BASE`/`SPS_*` into `sps_config.py`, and split `run_config_beam.py` into
   `run_configs/` (§9.9), which removes the `DAQ_*` family entirely.

Nothing in this section has been applied — `.bashrc` on banco is **unchanged**. Note that editing
it affects only *new* shells; the running tmux processes keep their current environment until
restarted.

---

## 11. Caveats on this document

- I did **not** run anything on mx17-daq beyond read-only inspection; a run was in progress.
- On banco I **did** take a pedestal run (00:29, successful) and made the four commits in §7,
  all at Dylan's request, earlier in the session.
- The comparison covers `dream_daq_control.py`, `daq_control.py`, `hv_control.py` and the config
  layer, as asked. **Not** covered: `flask_app/` (banco's is ~1300 lines shorter), the watcher
  processes (`processor_watcher`, `qa_watcher`, `backup_watcher`, and the bench's five extra),
  `n1081b/` internals, `beam_monitor/`, `space_manager.py`. Several of those are where the bench's
  remaining 124-commit lead lives.
- Every "banco lacks X" statement is about `main` on banco as of 2026-07-25 01:30 **including my
  four unpushed commits**. If someone resets that branch, re-derive before trusting.
