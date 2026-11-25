# Water in the chamber gas: where it comes from, and how a future chamber could keep it out

Written 2026-11-26. The chambers are gone, so nothing here can be tested on them. This note is for the design of
future chambers (LNL, upgrades). It also records what the existing data say about the source of the water.

## 1. What we measured

Every setup shows water in the gas at the few-tenths-of-a-percent level. Water slows the drift velocity, so it is
the main reason v differs between chambers and drifts with time. The numbers come from the drift velocity, inverted
on the Magboltz grid for the right mixture (memory `gas-water-synthesis-2026-10-08`, `rerun/paper1_gas/README.md`).

| setup | flow | water | water inflow x·Q |
|---|---|---|---|
| SPS H4, chamber E | ~2 L/h | ~1.5 % | ~30 mL/h |
| n_TOF, chamber A (first in the line) | 7.8 L/h | 0.15–0.3 % | 12–23 mL/h |
| n_TOF, chambers C and D (downstream) | 7.8 L/h | 0.35–0.7 % | 27–54 mL/h |

The water fraction times the flow, x·Q, is roughly the same in every setup, about 20–50 mL of vapour per hour. One
mL of vapour is 0.8 mg of water, so this is 16–40 mg of water per hour. A constant inflow means the water does not
come with the bottle gas. It comes from a steady source on the way: permeation through plastic, or outgassing from
the chamber materials.

At n_TOF the chambers are in series, A → B → C → D (`mx_july_beam_qa/DRIFT_WINDOW_HANDOFF.md`). Chamber A already
carries 12–23 mL/h. Chambers B, C and D together add roughly another 15–30 mL/h, so each of them adds roughly
5–10 mL/h (4–8 mg/h). The drift velocity also follows the outdoor water vapour (r = −0.84 on chamber C,
`rerun/ntof/vt/README.md`), so the source responds to the humidity of the hall within about a day.

## 2. The Mylar entrance window as the source

Each chamber is closed by a 60 µm Mylar (PET) window with a 400 × 400 mm free span (`docs/detector_spec/spec.py`).
PET lets water vapour through. Until now nobody estimated how much water enters this way. Here is the estimate.

**The datasheet rate.** Mylar datasheets quote the water vapour transmission rate at 38 °C and 90 % relative
humidity (ASTM F1249). Mylar 850H gives 17 g/m²/day at 30 µm, and Mylar A gives 4 g/m²/day at 25 µm. The rate
scales as one over the thickness, so a 60 µm film passes roughly 2–9 g/m²/day at those conditions.

**Scaling to the hall.** The rate is driven by the difference in water vapour pressure across the film. The gas side
is nearly dry. At 38 °C and 90 % the outside vapour pressure is 60 hPa. In the hall it was roughly 14–21 hPa, if the
hall air holds about as much water as the outdoor air did over the campaign. The rate is therefore about 0.25–0.35
of the datasheet value, roughly 0.5–3 g/m²/day. We ignore the temperature dependence of the permeability itself,
which adds perhaps a factor of 2 to the uncertainty.

**One window.** Over 0.16 m² this gives 0.08–0.5 g/day, which is **3–20 mg/h per chamber**.

| window | water through it | compared with the 4–8 mg/h each chamber adds |
|---|---|---|
| bare 60 µm Mylar | 3–20 mg/h | the same size, so it can be the whole source |
| aluminised, coating intact | 0.1–0.6 mg/h | negligible |

An aluminium coating lowers the transmission of PET by a factor of about 35–40 (Mylar M813: 2.0 against 0.05
g/100 in²/day; plain against metallised PET: 31 against 0.8 g/m²/day). However, a thin coating cracks where the film
is creased or stretched. The window bulges by roughly 8 mm under the gas overpressure, so an intact coating cannot be
assumed either.

**So the answer depends on one fact we do not have.** `spec.py` assumes 100 nm of aluminium on the gas side, but this
is from memory and the CAD carries no coatings. The same file notes that the run budget planned *new aluminised
windows* for all four chambers, which suggests the windows in use may not have been aluminised. The question is
already on Stephan's list (`q_window`). If the windows were bare, the window is the most likely single source of the
water. If they were aluminised and intact, it is not, and the tubing or the chamber walls are.

**What the chain pattern says.** A source of the same size in every chamber makes the water rise step by step down
the line. If most of the water enters at the chambers themselves, the swing of v with humidity should then scale as
1 : 2 : 3 : 4 for A, B, C, D. The measured swings are A 1, C 3.0 and D 2.9 (`rerun/ntof/vt/README.md`). Chamber C
agrees with a per-chamber source. Chamber D does not, but D has the weakest v measurement (±3.3 % per sub-run, no
absolute anchor). Therefore the data are consistent with a per-chamber source. However, the data cannot tell the
window apart from the tubing between chambers or from the chamber walls, since all three enter once per chamber.
This is a stated limitation. We have no record of the tubing material or length.

## 3. Ways a future chamber could keep the water out

**A desiccant inside the chamber (looked at, and not recommended).** The chamber would need about 0.6 g of water
absorbed per day. A 3A molecular sieve holds 10–20 % of its weight before water breaks through, so one month needs
100–200 g of sieve, about 150–300 cm³ of beads. There are five further problems:

1. Zeolite beads shed fine dust, and dust on the micromesh causes sparks.
2. A sieve is an aluminosilicate. At n_TOF, aluminium captures neutrons and gives ²⁸Al β decays, which are already
   half of the Geant4 drift-gas hits. The sieve also adds material in front of or beside the drift volume.
3. Only the 3A pore size is safe. The 4A, 5A and 13X sieves also adsorb CO₂, so they would change a CO₂ mixture.
4. A sieve holds less water when warm. It would release water as the hall warms, so v would follow the temperature.
   For calibration, water that is stable matters more than water that is low.
5. A saturated sieve lets water through with no warning, and a sieve must be baked at 250–300 °C before use.

For these reasons a desiccant inside the gas volume is not feasible for these chambers.

**What would work, roughly in order of how much it would help:**

1. **An aluminised window, or a double window.** If the window is the source (Sec. 2), aluminising it lowers the
   water by a factor of ~35. A second foil with a slow flush of dry gas between the two foils works even when the
   coating cracks. The window should be stretched so that it does not crease.
2. **Metal tubing (stainless steel or copper)** for the lines, especially between chambers. Every plastic permeates
   water, PFA included.
3. **Parallel gas lines instead of a series chain.** Each chamber then gets dry gas, and the chambers no longer pass
   their water on to the next one.
4. **A higher flow.** Since x·Q is constant, doubling the flow halves the water. It is the simplest change.
5. **A 3A sieve or purifier cartridge in the line, just before the inlet.** It dries the gas with no dust or material
   near the detector. However, it removes only water that enters upstream of it, so it does not help against the
   window or the chamber walls.
6. **A hygrometer at each chamber's outlet.** Today the water is inferred from v. A direct measurement would make v a
   cross-check, and it would tell the window, the tubing and the walls apart.

## Sources

- Mylar 850H datasheet: <https://polymerfilms.com/wp-content/uploads/2023/09/Mylar850H-Datasheets.pdf>
- Mylar M813 datasheet (plain and metallised): <https://polymerfilms.com/wp-content/uploads/2023/07/MylarM813-Datasheet.pdf>
- Mylar electrical insulation brochure (Mylar A, 25 µm): <https://mylar.com/wp-content/uploads/2026/07/Mylar-Specialty-Films-Electrical-Insulation-Brochure.pdf>
- Metallised film comparison table: <https://en.wikipedia.org/wiki/Metallised_film>
- Toray VMPET 4102 (12 µm metallised PET, 0.88 g/m²/day): <https://www.toray.com.my/my/products/films/product_file_my/file/Typical_properties_of_Metalion_VMPET%20type_4102.pdf>
