# NFS / GANIL — what the background study says, and what it needs

Companion to `sept26_prelim/ganil/report.html`. The report is generated and
always current; this file is the standing list of what it assumes and what
would have to happen next.

Written 2026-09-09. Scope: **the background only.** The page quotes no signal
rate, deliberately — see "What is deliberately absent" below.

---

## The result in five lines

| | |
|---|---|
| **The excitation is a variable** | `E_x = 20.578 + 0.749·E_n` → 21.3 MeV at 1 MeV, 50.6 MeV at 40 MeV. |
| **So the X17 angle moves** | `cos θ_min = 1 − 2m²/E_x²` → 104° at 1 MeV, 39° at 40 MeV. 109° is a property of 20.58 MeV, not of the boson. |
| **The gas gets ~280× better** | Radiative captures per neutron *entering the cell* go from 1.0×10⁻⁸ at thermal to 2.9×10⁻⁶ at 2 MeV. Not because (n,γ) rises — it barely moves — but because the 5333 b (n,p) channel that consumes every thermal neutron and makes nothing collapses to under a barn. |
| **The (n,p) two-prong load drops ~10⁴×** | 9.7×10⁷ proton–triton pairs per radiative capture at thermal; 8.9×10³ at 2 MeV. |
| **And there is a quiet window below 2.29 MeV** | The capsule's two strongest inelastic lines (843.8 and 1014.5 keV in ²⁷Al) are *below* the 1.022 MeV pair threshold, and its first level that is not — 2.211 MeV — needs a 2.29 MeV neutron. In that band the wall makes ~3 wide-angle pairs per gas pair, against 10⁴–10⁶ at n_TOF. |

**The recommendation is therefore: run below 2.29 MeV.** Four to five orders of
magnitude of capsule background removed by choosing the beam energy, with no
change to the apparatus, and the X17 angle stays at 98–104°, close enough to
n_TOF's 109° that the same acceptance applies.

## The new handle, which n_TOF cannot have

At n_TOF every neutron gives the same excitation, so the expected spectrum is
one template and time-of-flight is free for other uses. At NFS the opposite is
true and it is *better*: the signal angle is a **known function of a measured
quantity**. The gas continuum follows it (same excitation). **The capsule
background does not follow it at all** — a 2.211 MeV inelastic photon from
²⁷Al is 2.211 MeV whatever the neutron did. A background that does not track
θ_min(E_n) is rejected by that correlation however large it is.

That argument is worth more than the loss of the fixed 109°, and it is the
reason a white beam is usable rather than only the quasi-monoenergetic
⁷Li(p,n) mode.

---

## What is assumed, worst first

### 1. The capsule cascade is one photon per level — ×0.5 to ×1

Each discrete inelastic level (ENDF MF = 3, MT = 51…90) is de-excited by a
single photon of the level energy straight to the ground state. **Exact for
¹²C**, whose 4.44 MeV level has nowhere else to go. For ²⁷Al the upper levels
mostly feed the 844 and 1014 keV states instead, so the real photons are
*softer* than assumed — fewer pairs, but wider ones, so the error does not all
go one way.

**Fix:** ENDF/B-VIII.0 carries the real photon production in MF = 6 with
ZAP = 0. `endf.py` reads MF = 3 only. This is the single biggest improvement
available and it is a day of parsing.

### 2. Above 20 MeV neither library carries what is needed — hard stop

Two independent walls, from opposite directions, and they land at the same
energy:

* **³He(n,γ)** simply ends at 20 MeV in ENDF/B-VIII.0 *and* in TENDL-2021.
  Nothing evaluated exists above it.
* **The Al and C discrete inelastic levels are zeroed above 20 MeV**, with the
  strength moved into MT = 91, the continuum, whose photons this module cannot
  see.

So the study runs 1–20 MeV. The figures show an extrapolation above that,
labelled, and nothing is concluded from it. If running above 20 MeV is ever the
plan, **that cross section has to be measured or calculated first.**

### 3. Discrete-level completeness falls above 8 MeV — ×1 to ×2

Below 8 MeV the named ²⁷Al levels carry **100 %** of the evaluated inelastic
cross section, so the line sum is the whole of it. By 10–20 MeV they hold only
~49 %. The capsule numbers are therefore complete exactly where the
recommendation lives and an **under**-estimate by up to a factor of two where
it does not. This makes the case for the low-energy window stronger, not
weaker.

### 4. The multipole of the gas capture above 1 MeV — ×1.5

Taken as E1, because direct/semi-direct capture is E1-dominated in light
nuclei. Above the 20.58 MeV threshold the ⁴He compound states are broad and
overlapping and no partial-wave decomposition exists at these energies. M1 is
carried as the alternative and moves the wide-angle yield by about 1.5.

### 5. Neutron transport, as at n_TOF — ×1 to ×6

Single-pass optical depths, no scattering in the wall, no beam profile, no
self-shielding beyond the analytic sphere. The same Geant4 run fixes both
pages.

### 6. Charged-particle and conversion backgrounds — unmodelled

At MeV energies the cell also makes recoil protons, (n,p) and (n,d) charged
products, and far more high-energy photons that can convert externally in
material. None of that is a *pair from a transition*, so none of it is on the
page — and all of it is a trigger load that a rate projection would have to
carry.

---

## What is deliberately absent

**No signal rate.** The anomaly is reported for the 20.21 and 21.01 MeV states
of ⁴He. At 1–40 MeV neutrons the compound sits above both, in a region with no
resonance to enhance anything, and the X17-to-photon ratio there is a model
statement rather than a measurement. The `X17/IPC = 2.5×10⁻²` used in one
figure is the rate table's assumption, drawn only to put the signal on the same
axis as the background. **The background is calculable; the signal is not, and
quoting one anyway would be the easiest way to get this wrong.**

**No flux, and therefore no events per day.** Every quantity is per neutron
entering the cell, which is the part that is ours. Converting to a rate needs
the NFS flux against energy at the intended flight path, and that is the one
input that has to come from GANIL.

---

## Next steps, in order

1. **Get the NFS flux** against energy at the flight path being considered,
   especially in the 1–2.3 MeV band. Everything else here is ready to be
   multiplied by it.
2. **Read MF = 6** for the inelastic photon production. Replaces assumption 1
   and fixes assumption 3 at the same time.
3. **Geant4 the capsule at 1–3 MeV.** Acceptance for wall-born pairs is the
   largest unknown on both this page and the n_TOF one, and it is the same
   simulation.
4. **Decide whether the quiet window is compatible with the beam.** The
   ⁷Li(p,n) quasi-monoenergetic mode sits in it naturally; a white beam works
   too, because the band is a cut on a measured quantity rather than a property
   of the beam. Which one is cheaper in integrated flux is a facility question.
5. **Only then** revisit the signal side — and if the X17 rate at
   E_x = 21–51 MeV matters, that is a theory question for the same people who
   would supply the thermal ³He matrix elements (see `IPC_MISSING.md`).
