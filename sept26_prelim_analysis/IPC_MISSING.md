# The IPC prediction — what is settled, what is not, and what to do next

Companion to `sept26_prelim/ipc/report.html` and to the published page at
<https://dylan-neff.web.cern.ch/x17/ipc-continuum/>.  The report is generated
and always current; this file is the standing list of what it *cannot* answer,
so a reader does not have to reverse-engineer the gaps out of the caveats.

Last revised 2026-09-09.

---

## Settled, and not worth reopening

| | |
|---|---|
| **The QED half of internal pair creation** | One closed-form curve per multipole, from one-photon exchange. At Z = 2, αZ = 0.015, so Born is essentially exact. Validated three ways in `ipc_born.validate()` — against Wilkinson's published E0 energy-sharing law, against the same E0 distribution derived as a contact operator, and (new) the quadrature spectrum against the sampled one over the whole curve in total variation. `main()` exits non-zero if any of it fails. |
| **Which channels are open below 2 eV** | Two, both s-wave: 1⁺(³S₁)→0⁺ is M1 and is the measured 55 μb radiative capture; 0⁺(¹S₀)→0⁺ is E0 and emits no real photon at all. The p-wave 1⁻ resonance that dominates Viviani et al.'s Table V is down by 10⁻⁵ in our window. |
| **That the prediction does not move with arrival time** | Four separate reasons, all evaluated in `ipc_channels.energy_invariance()`; two of them cancel *exactly* because they are ratios of 1/v channels. Total variation between the spectrum at 1 ms and at 1 s is 2×10⁻⁹. One template covers the whole window. |
| **Which ²⁷Al lines make the wide-angle pairs** | The 2.3–4.3 MeV E1 primaries, not the 7.7 MeV ones. The two hard primaries feed positive-parity levels (3⁺ ground state, 2⁺ at 30.6 keV) so they are M1. |

## Open, ordered by how much it could move the answer

### 1. The ³He self-shielding in `results_3He` — ×140, and it is not nuclear

The 500 atm cell over 4 cm has an optical depth of ~150 to ³He(n,p) at
thermal, so it absorbs essentially every neutron that enters it.  The rate
table's `He3-captures` column matches a **thin-target** formula
(`N × n·σ_nγ`), which for an opaque absorber overestimates the radiative
capture count by the optical depth.  If that reading is right:

* every expected IPC and X17 yield in `calculation_tables/results_3He` — and
  in the INTC proposal that quotes them — is high by ~2 orders of magnitude;
* the capsule-to-gas ratio on the report page is correspondingly worse.

**What settles it:** one question to whoever produced the table.  Nothing in
this analysis can tell whether the code applies self-shielding elsewhere and
this column simply reports something else.  Until then the report gives the
comparison three ways rather than picking one.

### 2. Neutron transport in the capsule wall — ×1 to ×6

The table reports ~1.9 elastic scatters per neutron in the capsule (`GC-nel`),
which is consistent with the carbon fibre's own elastic optical depth of 0.46.
A thermal neutron therefore random-walks in the wall before it captures, and
the analytic single-pass capture probability in `ipc_aluminium.bookkeeping()`
is a **floor**.  The table's own `GC-captures` is 6× that floor.

**What settles it:** a Geant4 or MCNP run of the real capsule geometry.  The
same run gives §3 and §6 below for free.

### 3. Hydrogen in the carbon-fibre binder — ×1 to ×3

The geometry header gives a carbon areal density and nothing else.  Epoxy is
~5 wt % hydrogen, and ¹H captures at 0.333 b against carbon's 3.5 mb, so even
a modest binder fraction can outweigh all the carbon in the capture budget.
The 2223 keV line is soft but comfortably above the pair threshold, and soft
pairs are wide.

**What settles it:** the capsule's actual material spec.

### 4. Multipolarity of the secondary cascade — ×1.4

Primaries are assigned from the parity of the level they feed, which is what
EGAF gives.  Secondaries are not assigned, and are carried as an all-M1 to
all-E1 bracket: 2.1×10⁻⁴ against 3.8×10⁻⁴ wide-angle pairs per capture.
Closing it needs parities at both ends plus mixing ratios.

**What settles it:** reading the transition records EGAF already has, rather
than only the level records.  Worth doing only after §1 and §2, which are
larger.

### 5. Coulomb corrections to the Born form at Z = 13 — ~10 %

αZ = 0.095 for aluminium against 0.015 for helium.  The Born pair spectrum is
still a good description but is no longer exact, and the correction grows
towards wide angles and asymmetric energy sharing — both of which matter here.
The ³He numbers do not need it.

**What settles it:** a Dirac–Coulomb IPC code, or the published Z-dependence
tables for pair conversion coefficients.

### 6. Acceptance for pairs born in the capsule wall — unknown, possibly decisive

Everything in the report is **production**.  A pair born in the wall starts
2 cm off the gas centre, so the pointing, vertex and DCA cuts treat it
differently from a gas pair — possibly much better.  This is the most likely
place for the aluminium background to shrink, and it is not modelled anywhere.

**What settles it:** the Geant4 chain, with the `data/nuclear/` line list as
the primary generator's input.

### 7. External conversion in the wall — unknown, and it pollutes the control region

A 7.7 MeV capture photon converting *in* the aluminium makes a real pair too,
and there are ~500 photons per internal pair.  Those pairs are born collimated
(θ ~ m/E) so they should not reach 109°, but they will dominate the
small-angle region — which is exactly the intra-chamber sample the analysis
plans to use as the IPC control region and to fit the E0 fraction in.
Multiple scattering in the wall moves some of them.  Not modelled at all.

**What settles it:** the same Geant4 run, with conversion enabled.

### 8. What would make all of this easy, and is not in this setup

The wide-angle aluminium pairs carry 2–4 MeV between the two tracks; the ³He
pairs carry 20.6 MeV.  **Nothing here measures total pair energy** — no magnet,
no calorimetry.  The n_TOF proposal's own detector carries a 50 mT coil for
precisely this.  Without it the discriminants left are the vertex and the
opening-angle shape, and the shape is worth about a factor of two.

---

## Not open, but easy to misread

* **The 2 eV → 5.9 keV window.**  The invariance argument uses 1/v for both
  ²⁷Al and ³He, which holds below the first ²⁷Al resonance at 5903 eV — 34 μs
  of flight.  Anything reaching back to tens of microseconds would need the
  resonance region.  The flash veto is at 1 ms, so nothing does.
* **`IPC/capture = 2.1e-3`.**  Correct where it came from (Viviani et al. at
  E_n = 0.17–2 MeV) and wrong here, by a factor of two in normalisation and by
  a whole shape.  It is not a "conservative" number; it is a number from a
  different regime.
* **The E0 estimate.**  A single-level Breit–Wigner on a state 0.37 MeV below
  threshold, using a matrix element from a paper about why *ab initio* theory
  cannot reproduce that matrix element.  "Tens of percent of the pair yield",
  not a prediction.  The report's E0-fraction band (6–52 %) is the honest span.
