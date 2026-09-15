# HANDOFF — where the thermal neutrons go, step by step (for the Linux box)

**Written 2026-09-14 on the Windows box, for a session on the Linux box that has
`~/CLionProjects/MX17_Full_Geant` and its outputs.** Nothing here has been run.
The Windows side has no Geant4 outputs; every Geant4 number below is quoted
second-hand from `ntof_run_report/make_report.py` §6 and must be re-read from its
source before it is trusted.

Destination: the Athens talk, slide 40 onward (`slides/ntof_athens_talk.pptx`,
1 October 2026). It replaces the interim `figures/thermal_pair_sources` from
`make_thermal_sim_figures.py`. That figure put three different questions on one
page, and the audience (the n_TOF collaboration, mostly not MPGD people) could
not follow it.

---

## 0 · What is wanted, in one picture

A **funnel**. Each step is its own slide-sized figure: a **diagram** of the step
on the left, and on the right **one split** giving fractions *and* counts per 30
days. The segment that the next figure opens up is highlighted.

```
neutrons reaching the capsule (thermal window)
 ├─ F1  ³He(n,p) → p + t                      ← the majority; invisible to us
 └─ F1  (n,γ) capture                           (plus an "other" bucket)
      ├─ F2  γ only, nothing charged reaches a detector
      └─ F2  ≥ 1 charged particle in a detector
           ├─ F3  from e⁺e⁻ pair production in/near the capsule
           │    └─ F4  … by source reaction (²⁷Al, ¹²C, ¹H, ³He, other)
           └─ F3  charged particle made somewhere else
F5  (separate) what fires a trigger leg: a capsule pair vs other, other broken down
```

Two deliverables from the Linux box. The first is required, the second is
optional:

1. **The numbers.** A reduction script in `MX17_Full_Geant`
   (suggested: `scripts/thermal_accounting.py`). It writes
   `nTof_x17/ntof_athens_26/data/thermal_accounting/accounting.json` plus one CSV
   per figure (§4 has the schema). This file is the contract.
2. **The figures**, *if convenient*. Otherwise the Windows side builds them from
   the JSON by extending `make_thermal_sim_figures.py`, so they keep the deck's
   house style (`mpgd26/plotstyle.py`). §5 is the diagram spec either way.

---

## 1 · Before writing anything: find out what the existing products already hold

Earlier campaigns already computed pieces of this, so check them first:

| product | why it matters |
|---|---|
| `analysis/trigger_provenance/` | already attributes trigger legs to the **capture nucleus** (96 % Al), so the ancestry from leg to capture exists somewhere. Find out how it is done. |
| `scripts/plot_timedist_bysource_thermal.py`, `analysis/thermal_2cm/timedist_2cm.npz` | the by-source rates: SiPM singles, plastic singles, arm coincidence |
| `al_pair_background/VERDICT.md`, `analysis/al_pair/` | 5.95×10⁶ Al pairs/day produced, 6.5×10⁵ in MM acceptance. Is that **internal or external** pair creation? It matters for F4. |
| `docs/report/thermal_note.pdf` | the self-shielding (×50–100). It is also where the neutron-fate numbers for F1 probably already exist. |
| `docs/al_gamma_yield_check/RESULT.md` | 4 121 Al capture γ per pulse |
| `CAMPAIGN_STATUS.md`, `HANDOFF_THERMAL_TRIGGER.md` | which campaign and geometry is current |
| `src/EventAction.cc`, `src/SteppingAction.cc`, `include/SimConfig.hh` | what truth is actually stored per event |

**The deciding question:** does the stored truth give, for every charged track
that deposits energy in a sensitive volume, **(a)** its creator process,
**(b)** its creation volume and position, and **(c)** the capture it descends
from (target nucleus and capture volume)?

- If all three are there, this is a reduction job only.
- If not, add them to `SteppingAction`/`EventAction` (§3) and rerun a thermal
  campaign on condor. 10⁹ EAR2-flux neutrons, the same size as the trigger
  campaign, is enough for every node except ³He(n,γ) (§2.4).

---

## 2 · The definitions — fix these first and write them into the JSON

A funnel is only as good as its buckets. **Every split must sum to 100 % of its
parent, with an explicit "other"**, and every choice below must be recorded.

### 2.0 Denominator and window

- **Denominator: neutrons that enter the capsule assembly** (gas + Al shell +
  carbon-fibre shell). Neutrons that miss the capsule are not in the funnel, but
  count them and report the fraction. Their captures on the structure feed the
  trigger, so F5 includes them.
- **Window: E_n < 2 eV**, which is t > 1 ms after the flash (the flash veto),
  matching the data (2 eV → 0.44 meV, median 31 meV). If the sim is binned by
  time instead, cut at t > 1 ms and say so.
- **Counts per 30 days**, normalised **two ways**:
  - `per30d_ratetable`: using slide 39's flux, the neutrons per pulse per decade
    from `mpgd26/data/x17_rate_3He.txt` and 1.929×10⁴ pulses/day. This is what
    slide 40's `thermal_branching` panel (c) now uses, so the numbers agree
    across slides.
  - `per30d_sim`: the sim's own normalisation (10⁹ EAR2 neutrons ≙ N pulses).

  Headline with the rate-table one, and keep both in the JSON.

### 2.1 F1 — what the neutron does

Buckets, per neutron entering the capsule in the window:

| bucket | note |
|---|---|
| `he3_np` | ³He(n,p)t. The 573 keV proton and 191 keV triton stop in the 500 atm gas and never leave the capsule, so this is **invisible to us**. Expect ≈ 99.97 % (analytic, `ipc_aluminium.bookkeeping`). |
| `ncapture_wall` | (n,γ) in the capsule shell, split ²⁷Al / ¹²C / ¹H (binder, only if modelled) |
| `ncapture_he3` | ³He(n,γ)⁴He. ~1×10⁻⁸ per neutron, which is ~10 events in 10⁹. **Do not quote the MC count**; use the analytic value (§2.4). |
| `ncapture_elsewhere` | a neutron that scatters out of the capsule and is captured on the structure, a chamber, a scintillator or the hall |
| `no_absorption` | leaves the world |

On the slide F1 shows two segments, (n,p) against all (n,γ), with "other" as a
sliver. The JSON keeps every bucket.

### 2.2 F2 — does the capture produce anything a detector sees?

Parent: every (n,γ) capture in F1, **all capture buckets together** (wall, ³He,
elsewhere). Report the split per capture bucket as well.

- `gamma_only`: no charged particle deposits energy in any sensitive volume.
- `charged_in_detector`: ≥ 1 charged particle deposits energy in a sensitive
  volume.

"Sensitive" is reported at **two tiers** (keep both; the slide uses tier A):

- **tier A**: any Micromegas drift volume, ≥ 1 keV total by charged tracks
  (a track, not a Compton flicker). Check the threshold against what `wft` could
  reconstruct and record it.
- **tier B**: a scintillator (SiPM wall or plastic) above 0.5 MIP.

### 2.3 F3 — where the charged particle came from

Parent: `charged_in_detector`. Attribute each event by its **leading
depositor**: the charged track, *with its descendants*, that deposits the most
energy in the tier-A volume. Then bucket that track by where and how it was
created:

| bucket | definition |
|---|---|
| `pair_near_capsule` | an e⁺ or e⁻ created by **pair production** (G4 process `conv`, or the internal-pair add-on of §2.4) with vertex **in/near the capsule** |
| `compton_near_capsule` | `compt`/`phot` electron with vertex in/near the capsule |
| `pair_elsewhere` | `conv` elsewhere, split by volume class: chamber frame/window, drift gas, scintillator, lead, air, other |
| `compton_elsewhere` | the same volume classes |
| `neutron_induced` | recoil protons, (n,p)/(n,α) products outside the capsule |
| `other` | |

**"In/near the capsule"** means inside the capsule assembly's logical volumes:
gas, Al shell, CF shell, and the holder/mount if it is in the geometry. As a
sensitivity check, also try **r < 30 mm from the capsule centre** and report the
difference. The slide shows `pair_near_capsule` against everything else; the
rest are kept for F5.

### 2.4 F4 — which reaction made the capsule pair

Parent: `pair_near_capsule`. Split by the reaction that produced the parent γ,
crossed with the conversion mechanism:

| | external conversion (Geant4 `conv`) | internal pair creation |
|---|---|---|
| ²⁷Al(n,γ) | from the sim | **not in Geant4**, analytic (see below) |
| ¹²C(n,γ) | from the sim | analytic |
| ¹H(n,γ) (binder) | from the sim, if modelled | analytic |
| ³He(n,γ) | ≈ 0 (see below) | analytic |
| other (γ from captures elsewhere converting in the capsule) | from the sim | — |

**Internal pair creation is not in Geant4.** `G4PhotonEvaporation` emits the
cascade as γ, with optional conversion *electrons*, but no internal pairs.
Confirm this for the Geant4 version and physics list in use, and record it. So
the sim's capsule pairs are external conversions only. The internal pairs have to
be added analytically **per capture, as a clearly separate, labelled segment**:

- per wall capture: **2.10×10⁻³** pairs (`sept26_prelim_analysis/ipc_aluminium.py`,
  EGAF/PGAA line list, Al and C weighted by captures; secondaries taken as M1).
  The all-E1 version is 2.69×10⁻³; carry that as the bracket.
- per ³He radiative capture: **4.66×10⁻³** pairs (`ipc_born` M1 + E0).
  **Not 2.1×10⁻³**: that number is Viviani et al.'s MeV ratio; see
  `nTof_x17/CLAUDE.md`.
- ³He radiative captures per neutron: **1.03×10⁻⁸**, self-shielded analytic, at
  31 meV.

Report what `al_pair_background/VERDICT.md` counted, internal or external. If it
already added internal pairs, reuse its method and say so.

### 2.5 F5 — what fires the trigger (a separate breakdown)

Use the same trigger emulation as `trigger_provenance` (wall ∧ plastic in one
arm, 0.5 MIP, 20 ns). Record its exact settings.

- Parent: **trigger legs** (sim: 205 per pulse, 96 % Al). Report per pulse and
  per 30 days.
- Attribute each leg by the **leading depositor** summed over that arm's wall
  bar and plastic bar, using the §2.3 buckets. If the leading ancestry carries
  under 70 % of the leg's energy, the leg goes in `mixed`.
- Slide split: `pair_near_capsule` against `other`, with `other` broken down into
  - Compton from capsule-capture γ, by nucleus
  - conversion or Compton in the structure, by volume class
  - γ from captures outside the capsule
  - neutron-induced
  - mixed / pile-up
- Report the same breakdown again for **pair-tags** (both arms fire in one
  event; sim 0.8–1.8 per pulse). This connects to the data slide: 29 % [17, 40]
  of our two-arm pairs are prompt (`sept26_prelim_analysis/HANDOFF_ACCIDENTAL_TIMING.md` §0).
  For those, split the sim pair-tags into **one ancestry (a genuine pair or one
  particle crossing both arms)** and **two independent ancestries (accidental)**.

---

## 3 · If truth has to be added

The minimum per event, as written from `SteppingAction`/`EventAction`:

- **per capture**: target Z/A (from the `nCapture` step's secondaries or its
  target nucleus), volume, position, time, and a capture index.
- **per charged track with E_kin > 50 keV at creation**: track ID, parent ID,
  PDG, creator process, creation volume and position, E_kin, and the **capture
  index inherited down the ancestry** (propagate it through a
  `G4VUserTrackInformation`).
- **per sensitive volume**: energy deposited, keyed by the *root* ancestry
  (capture index plus the first charged track). Without this key, "leading
  depositor" cannot be computed after the fact.

Keep the file small: only events with a capture or a deposit. Tag the campaign
with the `MX17_Full_Geant` git hash, geometry version, physics list and the
number of primaries.

---

## 4 · The JSON contract

```json
{
  "schema": "athens/thermal_accounting/1",
  "provenance": {"git": "...", "campaign": "...", "n_primaries": 1e9,
                 "physics_list": "...", "geant4": "11.x",
                 "geometry": "...", "window": "E_n < 2 eV",
                 "near_capsule": "logical volumes [...]",
                 "tierA_threshold_keV": 1.0, "trigger": {"...": "..."},
                 "ipc_added_analytically": true},
  "normalisation": {"neutrons_entering_capsule": ..., "pulses_per_day_ratetable": 19290,
                    "per30d_ratetable_factor": ..., "per30d_sim_factor": ...},
  "nodes": [
    {"id": "F1.he3_np", "parent": "root", "label": "³He(n,p) → p + t",
     "mc_count": ..., "weight": ..., "frac_of_parent": ..., "frac_err": ...,
     "per_neutron": ..., "per30d_ratetable": ..., "per30d_sim": ...,
     "source": "geant4 | analytic", "note": "..."}
  ]
}
```

- `frac_err` is a binomial error on the MC count.
- An analytic node has `mc_count: null` and a `note` naming the module.
- One CSV per figure (`F1.csv` … `F5.csv`) with the same columns.

---

## 5 · The diagrams, one per figure

Keep every figure to the same layout: the diagram on the left ~45 %, the split
on the right (one horizontal 100 % bar, 2–4 segments, each carrying its
**percentage and count per 30 days**), a one-line headline, no footnote. Draw the
capsule and arms to scale from `sept26_prelim_analysis/geometry.py`, reusing
`make_overhead_figure.py`'s transverse section so it matches the fans slide. The
capsule is not at the frame origin (−9 mm in x, +30 mm in y); in a cartoon the
nominal position is fine.

| figure | headline (draft) | diagram |
|---|---|---|
| F1 | "Almost every neutron makes a proton we can't see" | capsule section; a neutron enters. In the gas: a short p and t stub that stays inside. In the shell: a small capture star emitting a γ. The (n,p) segment is dominant and greyed as "invisible"; the (n,γ) segment is highlighted. |
| F2 | "Most capture γ fly straight through" | capsule plus the four chambers. Several γ lines (dashed) cross a chamber with no hits; one produces an electron track in a drift gap. |
| F3 | "The charged particles we see: made at the capsule, or elsewhere" | a pair vertex (e⁺ and e⁻ tracks) at the shell, against a Compton electron starting in a chamber frame |
| F4 | "Which nucleus made the pair" | the capsule shell with the Al and C layers labelled, a ³He nucleus in the gas; icons for internal pair creation (pair straight from the nucleus) and external conversion (γ, then a pair a little way out) |
| F5 | "What fires the trigger" | one arm: a chamber, the SiPM wall and a plastic bar, with the particle that made the leg coloured by its bucket |

Colours: ³He = `plotstyle.ACCENT`, the wall = slate `#5b6b7d`, "invisible" = pale
`#e3e7ec`. The highlight that carries into the next figure = `plotstyle.COPPER`.

---

## 6 · Numbers to reproduce, or to explain if they move

| quantity | value | source |
|---|---|---|
| ³He absorbs what enters (31 meV) | 99.97 % | analytic, `ipc_aluminium.bookkeeping` |
| wall captures per neutron | 1.4×10⁻³ (single pass) – 1.7×10⁻² (rate table, with wall scattering) | analytic / `results_3He` |
| ³He(n,γ) per neutron | 1.03×10⁻⁸ | analytic |
| σ(n,γ)/σ(n,p) at 25 meV | 1.0×10⁻⁸ | ENDF, and the sim reproduces it |
| Al capture γ per pulse | 4 121 | Geant4, quoted |
| SiPM singles / plastic singles / trigger legs per pulse | 2 063 (80 % Al) / 942 (57 % Al) / 205 (96 % Al) | Geant4, quoted |
| pair-tags per pulse | 0.8–1.8 | Geant4, quoted |
| data trigger legs per pulse | 429, i.e. 2.1× the sim | `mx_july_beam_qa/31_sim_data_compare.py` |

**The 2.1× data/sim excess on trigger legs is unexplained.** It is on the
scintillator response, not the transport (the 500 ns sideband gives ~5 %
accidentals). Fractions are therefore safer than absolute rates on F5, so say
that on the slide.

## 7 · What this will not settle

- Whether the prompt pairs in the **data** are aluminium pairs. The sim can say
  what they are *expected* to be; nothing in the setup measures total pair
  energy.
- The hydrogen content of the carbon-fibre binder, if the geometry does not model
  it. Report it as a named gap, not a zero.
