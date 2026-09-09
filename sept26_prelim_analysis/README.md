# sept26_prelim_analysis

The first end-to-end pass over the 2026 n_TOF EAR2 campaign: from DREAM
triggers to an **opening-angle distribution for e⁺e⁻ pair candidates**,
preliminary, for the 30 September collaboration meeting.

> ## → Start at [`PLAN.md`](PLAN.md), then [`STATUS.md`](STATUS.md)
>
> `PLAN.md` is the plan of record — the chain stage by stage, what it will not
> establish, and the D1–D15 register of what is deferred to October.
> `STATUS.md` is the live state and **the resume point if a session drops**;
> its "Resume here" section is written for the next session to execute in order.

Board (progress, log, deferred, open questions):
<https://dylan-neff.web.cern.ch/x17/analysis.html>
Note (the plan, published): <https://dylan-neff.web.cern.ch/notes/x17-prelim-plan.html>

---

## Layout

| | |
|---|---|
| `PLAN.md` | the plan of record. Read fully before writing analysis code here. |
| `STATUS.md` | live status, blockers, decisions, benchmarks, the log, and the ordered next steps |
| `make_note.py` | builds `report.html` — the published progress page. The prose lives in the script; there is no template to keep in sync. |
| `report.html` | generated, committed so the page can be republished without a rebuild |

The analysis modules are added stage by stage; `PLAN.md` §3 says what each
stage contributes, and `rerun_chain.sh` is the executable answer to "what runs
in what order".

### The expected-IPC modules

These four are the exception to everything else in the package: they take **no
run and no data**, because they are nuclear physics rather than measurement.
They run in seconds and publish `/x17/ipc-continuum/`.

| | |
|---|---|
| `ipc_born.py` | the Born (one-photon-exchange) internal-pair continuum, one closed-form curve per multipole. Self-validating: `validate()` checks it against Wilkinson's published E0 energy-sharing law and against the same E0 distribution derived as a contact operator, and `main()` exits non-zero if either fails. |
| `ipc_channels.py` | the reaction side — which multipoles the >1 ms (<2 eV) window actually makes, what the `2.1e-3` IPC/capture in the rate table really is, and the first-look aluminium comparison. Every input is a named constant with its source in the comment. |
| `make_ipc_figures.py` | four figures into `sept26_prelim/ipc/figures/` |
| `make_ipc_report.py` | `report.html` + `index.html` into `sept26_prelim/ipc/` |

**`pair_physics.py` is superseded by these for the IPC half** — its four-ansatz
band spans a factor of 38 in the fraction above 109°, none of which is
irreducible at Z = 2. Its X17 half is exact two-body kinematics and stands.
The wiring of the two-channel model into `opening_angle.py` is **not done**.

## Building the note

```bash
.venv/bin/python sept26_prelim_analysis/make_note.py
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
    sept26_prelim_analysis/report.html --slug x17-prelim-plan \
    --force --deploy
```

The `--slug` never changes: republishing to the same slug is how the page is
updated. `--deploy` needs a Kerberos ticket (`kinit dneff@CERN.CH`).

## Conventions this package holds itself to

Inherited from `../CLAUDE.md` and the packages before it, restated because they
are the ones this analysis is most likely to break:

- **Geometry comes from waveforms, never from hit times.** `../RECONSTRUCTION_BASIS.md`
  is binding. Hits are for candidate finding and QA only — which is exactly
  what stage 1 uses them for, and nothing more.
- **Anything calibrated is per chamber *and* per run condition.** A bundle used
  outside its conditions is a silent error, so the bundle name travels in every
  output row.
- **Never collapse an ambiguity at write time.** A two-track event's X↔Y pairing
  is genuinely degenerate; both hypotheses get written with their scores.
- **Every analysis ships an HTML report**, generated rather than hand-written,
  with relative figure links so the DAQ page and the notes site both serve it.
- **Figures are built for a projected slide** — 16:9, base font ≥ 18 pt at final
  size, one message per figure, a `Preliminary` badge on anything touching the
  reconstruction, and the numbers exported as CSV beside the PNG.
- **An empty table cell is honest; a guessed one is not.** `STATUS.md`'s
  benchmark table renders "not measured yet" rather than omitting the row.

## Which machine

| | |
|---|---|
| **Ubuntu laptop** | where this analysis runs. Repo `~/PycharmProjects/nTof_x17`, python `.venv/bin/python`, data `/media/dylan/data/x17/`. Has working `kinit` / `ssh lxplus`. |
| **lxplus + condor** | where the reconstruction runs, because the waveforms are on EOS and the link home is the bottleneck. `ntof_tracking/condor/`. |
| **Windows box** | the plan was written here, and that is all it is good for. No Kerberos, no lxplus, and `/d/x17/beam_july/runs` holds only run_55. |
