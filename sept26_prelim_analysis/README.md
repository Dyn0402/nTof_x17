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
| `report_style.py` | the one stylesheet every report in the package wears — `HEAD` is fonts + CSS + the small progressive-enhancement script, and goes straight into a report's `<head>`. Element-first, so a generator that emits plain `<main>`, `<table>`, `<p class="lede">` is already styled. |
| `publish_x17.sh` | which output directory is served at which `/x17/<slug>/`, and the rsync that puts it there. The registry is the list of record; `rerun_chain.sh` calls it for the run_145 pages. |

The analysis modules are added stage by stage; `PLAN.md` §3 says what each
stage contributes, and `rerun_chain.sh` is the executable answer to "what runs
in what order".

### The expected-IPC modules

These five are the exception to everything else in the package: they take **no
run and no data**, because they are nuclear physics rather than measurement.
They publish `/x17/ipc-continuum/`, and everything except the aluminium module
runs in seconds.

| | |
|---|---|
| `ipc_born.py` | the Born (one-photon-exchange) internal-pair continuum, one closed-form curve per multipole. `grid_spectrum()` returns the whole dN/dθ by quadrature — the spectrum, not a fraction beyond a threshold — and `frac_above()` reads any threshold off it. Self-validating: `validate()` checks it against Wilkinson's published E0 energy-sharing law, against the same E0 distribution derived as a contact operator, and the quadrature spectrum against the sampled one in total variation, and `main()` exits non-zero if any of it fails. |
| `ipc_channels.py` | the reaction side — which multipoles the >1 ms (<2 eV) window actually makes, what the `2.1e-3` IPC/capture in the rate table really is, and `energy_invariance()`, which shows the prediction is the same curve from 1 ms to 1 s. Every input is a named constant with its source in the comment. |
| `ipc_aluminium.py` | the capsule, line by line, from the ²⁷Al(n,γ) capture scheme staged in `data/nuclear/`. Assigns E1/M1 from the parity of the level each primary feeds, sums the per-line Born curves into one spectrum, models escape and scattering in the wall, and does the capture bookkeeping three ways. `missing()` is the machine-readable version of `IPC_MISSING.md`. Takes ~4 min, almost all of it in the wall smearing. |
| `ipc_diagrams.py` | the inline-SVG mechanism drawings the pages open with. Theme-aware, no data to ship, and the ²⁷Al level scheme is drawn from `ipc_aluminium.line_list()` so it cannot drift from the tables beside it. `python -m …ipc_diagrams > preview.html` renders them all. |
| `make_ipc_figures.py` | nine figures into `sept26_prelim/ipc/figures/` |
| `make_ipc_report.py` | `report.html` + `index.html` into `sept26_prelim/ipc/` |

### The GANIL / NFS background study

A separate question with the same machinery: what would these two backgrounds
look like at a **1–40 MeV** neutron beam rather than a thermal one? Publishes
`/x17/ganil-background/`.

| | |
|---|---|
| `endf.py` | just enough ENDF-6 to read MF = 3 (a cross section) off an evaluation, plus the discrete inelastic level energies, which are minus each MT's QI. No dependency, and it raises rather than guessing on anything past MF = 3. Data staged in `data/nuclear/*.mf3.endf`. |
| `ganil_background.py` | the kinematics (`E_x = S_n + 0.749 E_n`, so the X17 opening angle slides from 104° to 39°), the gas rates from ENDF, and the capsule's inelastic γ inventory. `quiet_band()` is the recommendation and it is computed from the level scheme, not chosen. |
| `make_ganil_figures.py` | five figures into `sept26_prelim/ganil/figures/` |
| `make_ganil_report.py` | `report.html` + `index.html` into `sept26_prelim/ganil/` |

**The headline:** below 2.29 MeV the capsule makes ~3 wide-angle pairs per gas
pair, against 10⁴–10⁶ at n_TOF, because its two strongest inelastic lines are
under the pair threshold and its first useful level has not opened. The gas
also converts ~280× more of its neutrons into radiative capture than at
thermal. Both evaluations stop at 20 MeV, which caps the study rather than the
facility.

**`pair_physics.py` is superseded by these for the IPC half** — its four-ansatz
band spans a factor of 38 in the fraction above 109°, none of which is
irreducible at Z = 2. Its X17 half is exact two-body kinematics and stands.
The wiring of the two-channel model into `opening_angle.py` is **not done**.

`IPC_MISSING.md` is the standing list of what these modules cannot answer,
ordered by how much each gap could move the result. The largest is not nuclear:
the ³He rate table appears to compute radiative captures without self-shielding
in a cell that is optically thick, which would be worth ~10² on every expected
yield the experiment quotes.

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
- **Figures are ordinary document figures** — `figstyle.py` sets a normal shape
  (~3:2 for a single panel), 10.5 pt type, thin spines and a faint grid. Shape
  follows the data, not a slide frame. One message per figure, a `Preliminary`
  badge on anything touching the reconstruction, and the numbers exported as
  CSV beside the PNG.
- **An empty table cell is honest; a guessed one is not.** `STATUS.md`'s
  benchmark table renders "not measured yet" rather than omitting the row.

## Which machine

| | |
|---|---|
| **Ubuntu laptop** | where this analysis runs. Repo `~/PycharmProjects/nTof_x17`, python `.venv/bin/python`, data `/media/dylan/data/x17/`. Has working `kinit` / `ssh lxplus`. |
| **lxplus + condor** | where the reconstruction runs, because the waveforms are on EOS and the link home is the bottleneck. `ntof_tracking/condor/`. |
| **Windows box** | the plan was written here. No Kerberos and no lxplus, but the data disk is NTFS, so the products and every `report.html` read directly off it. |

### Running it somewhere else

Nothing in this package spells a data path of its own -- not the Python, and
since 2026-09-11 not the shell either. `paths.py` resolves every root, and the
chain scripts ask it rather than repeating a literal:

```bash
OUT=$($PY -m sept26_prelim_analysis.paths --path out) || exit 1
```

So one variable moves the whole tree, on either OS:

```bash
export X17_ROOT=/mnt/d/x17          # WSL, say, or a second disk
export X17_SEPT26_OUT=/somewhere/else   # just this analysis's output
python -m sept26_prelim_analysis.paths  # what resolves, and what exists
```

A root that is missing or unwritable is one line on stderr and a non-zero exit,
at the top of the run, not an empty glob forty seconds in.
