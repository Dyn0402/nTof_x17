"""The MX17 Micromegas module as we understand it: the data behind the detector-spec page.

Every number here is taken from the Geant4 module description, which is the single source of truth for the
stack, or from the notes and measurements behind it:

  MX17_Geant/shared/MX17ModuleGeometry.hh        AsBuiltSpec(): every thickness and transverse face
  MX17_Geant/design/GEOMETRY_FROM_CAD.md          what the mechanical STEP file says (2026-06-05 export)
  MX17_Geant/design/NEEDED_INPUTS.md              the register of what the model assumes
  MX17_Geant/design/RESPONSE_SIM_PLAN.md          ESL, coverlay adhesive, sheet resistance (response side)
  MX17_Geant/src/DetectorConstruction.cc          material densities and compositions
  nTof_x17/common/mx17_active_area.py             the measured efficient area (Y-edge passivation)
  nTof_x17/sept26_prelim_analysis/STATUS.md       chamber B field cage, the gas chain, per-chamber state

If one of those changes, change it here and rebuild (build_page.py). Status of each value:

  ok    measured, or confirmed by someone who built or ran the detector
  cad   read from the mechanical CAD or the production gerbers (a drawing, not a measurement)
  asm   assumed: a placeholder or a guess the model runs on
  open  unknown; the model does not use it, or uses a scan
"""

UPDATED = "2026-10-16"
TITLE = "MX17 Micromegas: detector specification for review"
SLUG = "mx17-detector-spec"
FEEDBACK_ENDPOINT = ("https://script.google.com/macros/s/AKfycby6urhEQy27p2JtRg2o4W60exJZgpk1ALF-YGcUrHJQE2rBKHbfvmI3SJwz"
                     "o_kET3Gt/exec")   # shared with facilities/budget.html; this page sends page="detector"
CONTACT = "dylan.neff@cern.ch"

STATUS = {
    "ok":   ("Confirmed", "measured, or confirmed by someone who built or ran it"),
    "cad":  ("CAD / gerber", "read from the mechanical drawing or the production gerbers"),
    "asm":  ("Assumed", "a placeholder the simulation runs on"),
    "open": ("Unknown", "not known; please tell us"),
}

GROUPS = [
    ("entrance", "Entrance window and drift"),
    ("amp", "Mesh and amplification"),
    ("readout", "Resistive layer and readout board"),
    ("back", "Backing"),
    ("lateral", "Frames, field cage, front-end"),
]

# --------------------------------------------------------------------------------------------- the elements
# side: thickness in mm drawn in the side view (None: not a layer of the stack); x0/x1 transverse extent in mm in
# the active-area frame (the readout board and everything behind it sit offset +15 mm).
P = lambda k, v, st, src="": dict(k=k, v=v, st=st, src=src)

ELEMENTS = [
    dict(id="window", group="entrance", name="Gas window (front foil)", color="mylar",
         short="Mylar 60 µm, aluminised",
         params=[P("Material", "Mylar (PET)", "cad", "CAD part name"),
                 P("Thickness", "60 µm", "cad", "STEP solid; the sims used 40 µm before 2026-08-06"),
                 P("Aluminium coating", "0.1 µm (100 nm), one side, facing the gas", "asm", "Dylan's hardware knowledge; thickness never measured"),
                 P("Single or double aluminised?", "unknown", "open"),
                 P("Face", "440 × 440 mm", "cad"),
                 P("Free span (flange aperture)", "400.2 mm", "cad"),
                 P("Bulge under overpressure", "8 mm sag (Hencky, ~3 mbar assumed)", "asm", "NEEDED_INPUTS §2; p^⅓ scaling, 1–10 mbar → 6–12 mm")],
         notes="The window is the gas seal, not the drift cathode: the cathode is a separate copper-clad kapton foil "
               "5 mm further in. The CAD carries no coatings, so its silence about the aluminium is not evidence "
               "of absence. The run budget plans new aluminised-mylar windows for all four chambers.",
         questions=["q_window", "q_bulge"]),
    dict(id="flange", group="entrance", name="Window flange and gas gap", color="gas",
         short="5 mm of gas, Al ring",
         params=[P("Flange", "aluminium ring, 440 mm outer, 400.2 mm aperture, 5.000 mm thick", "cad"),
                 P("Gap window → cathode", "5.0 mm of chamber gas", "cad"),
                 P("Is the gap flushed by the chamber gas flow?", "assumed yes (modelled as chamber gas)", "asm")],
         notes="Field-free in the model: charge liberated here does not drift to the mesh.",
         questions=["q_flange"]),
    dict(id="cathode", group="entrance", name="Drift cathode", color="kapton",
         short="Kapton 50 µm + Cu",
         params=[P("Foil", "Kapton (polyimide), 50 µm", "cad"),
                 P("Copper cladding", "9 µm, facing the drift gas", "asm", "not in the CAD; side assumed (NEEDED_INPUTS §5)"),
                 P("Face", "406 × 406 mm", "cad"),
                 P("HV", "−HV through the drift channel; 600–800 V at n_TOF (700 V = 233 V/cm in production)", "ok", "HV monitor")],
         questions=["q_cathode"]),
    dict(id="drift", group="entrance", name="Drift gap", color="drift",
         short="30.000 mm",
         params=[P("Gap, cathode → mesh", "30.000 mm (the Al frame is the spacer)", "cad", "GEOMETRY_FROM_CAD; gas in the model 29.991 mm, the Cu takes 9 µm"),
                 P("Transverse (inside the field cage)", "406.76 × 406.76 mm", "cad"),
                 P("Gas at n_TOF", "Ar/iC₄H₁₀ 90/10, flowed in series A → B → C → D → exhaust", "ok"),
                 P("Gas on the June cosmic bench", "Ar/iC₄H₁₀ 95/5", "ok"),
                 P("Water in the gas", "implied by the measured drift velocity, and rising along the chain A → D", "ok", "inferred from v against Magboltz, not a hygrometer"),
                 P("Drift field", "233 V/cm at 700 V (production)", "ok")],
         notes="The CAD puts the readout board top 0.1 mm inside the frame's z range; the model keeps "
               "kapton → mesh at exactly 30.000 mm (NEEDED_INPUTS §8).",
         questions=["q_drift"]),
    dict(id="mesh", group="amp", name="Micromesh", color="mesh",
         short="woven SS, spec unknown",
         params=[P("Type", "woven stainless steel", "asm"),
                 P("Wire diameter", "19 µm (placeholder copied from the P2 detector)", "asm", "NEEDED_INPUTS §1"),
                 P("Opening", "48 µm → pitch 67 µm (≈ 380 lines/inch)", "asm"),
                 P("Thickness", "38 µm = 2 × wire, i.e. not calendered", "asm"),
                 P("Calendered?", "unknown", "open", "Alexandra did not know either (2026-10-16)"),
                 P("Geant4 model", "38 µm slab of steel at fill factor 0.223 (areal mass of the weave)", "asm"),
                 P("Garfield model", "woven 19 µm wires; electron transparency 0.990 at 333 V/cm (null-collision tracking)", "asm"),
                 P("Optical transparency", "(48/67)² ≈ 0.51 for the placeholder weave", "asm"),
                 P("Face", "410 × 410 mm", "cad"),
                 P("Potential", "ground; the resistive strips carry +HV", "ok")],
         notes="The mesh spec moves three things: its areal mass (backscatter and conversions next to the gap), "
               "the electron transparency, and where the effective electrode plane sits. In the woven field map the "
               "equivalent plane is ~8 µm inside the weave, so the effective gap is 158 µm. A calendered mesh moves "
               "that plane down a few µm and raises the gap field by ~1–2 %.",
         questions=["q_mesh", "q_calender"]),
    dict(id="ampgap", group="amp", name="Amplification gap", color="amp",
         short="150 µm",
         params=[P("Gap, resistive strips → mesh underside", "150 µm", "ok", "Dylan, 2026-08-06; the 128 µm bulk standard is not ours"),
                 P("Amplification HV", "490 V on the bench; 520–560 V at n_TOF", "ok", "HV logs"),
                 P("Field", "490 V / 150 µm = 32.7 kV/cm nominal; 31.0 kV/cm mid-gap in the woven-mesh map", "asm")],
         questions=["q_ampgap"]),
    dict(id="pillars", group="amp", name="Bulk pillars", color="pillar",
         short="Ø 0.6 mm, 4.68 mm grid",
         params=[P("Diameter", "0.6 mm", "cad", "3498A_bulk.gbr"),
                 P("Grid", "85 × 85 on a 4.68 mm pitch, spanning ±196.56 mm", "cad", "exact in the gerber"),
                 P("Height", "the amplification gap, 150 µm", "ok"),
                 P("Material", "photoimageable coverlay (modelled as kapton)", "asm"),
                 P("Dead area", "1.3 % of the gap", "cad")],
         questions=["q_ampgap"]),
    dict(id="esl", group="readout", name="Resistive strips (ESL)", color="esl",
         short="550/800 µm, 10 µm thick?",
         params=[P("Strip width", "550 µm", "open", "Dylan, 2026-08-06, from memory; not in the gerber set. Please confirm"),
                 P("Gap / pitch", "250 µm / 800 µm", "open", "as above; note the readout pitch is 780 µm, not 800"),
                 P("Orientation", "along y (\"vertical\")", "asm"),
                 P("Number", "515 strips over the 412 mm deposit", "asm", "412 mm boundary from 3498A_top-resist.gbr; 515 = 412 / 0.8"),
                 P("Thickness (print depth)", "10 µm", "open", "Dylan: \"something like 10 µm\"; was a 100 µm guess before 2026-08-07"),
                 P("Paste", "screen-printed ESL resistive paste; grade unknown", "open"),
                 P("Paste in Geant4", "C/H/O organic, 1.4 g/cm³ (no carbon loading)", "asm"),
                 P("Sheet resistance", "unknown; 1.4–2.6 MΩ/□ inferred from the charge spread (2 MΩ/□ used)", "open", "RESPONSE_SIM_PLAN T14"),
                 P("Grounding / HV feed", "copper bus strips at both y-ends of the active area, nothing in between", "ok", "Dylan, 2026-08-07")],
         notes="The strips are the first solid the avalanche sees and they set how charge spreads to the neighbouring "
               "readout strips, which is the effect that biases hit times (RECONSTRUCTION_BASIS.md). Their "
               "thickness sets the depth of the gas grooves between them; the sheet resistance and the insulator "
               "under them (next layer) together set the spreading time.",
         questions=["q_esl_geom", "q_esl_depth", "q_esl_r", "q_passivation"]),
    dict(id="coverlay", group="readout", name="Kapton under the resistive strips", color="kapton",
         short="Kapton 50 µm + glue",
         params=[P("Kapton (coverlay)", "50 µm", "asm", "matched to the MX17_Full_Geant stack, never measured"),
                 P("Adhesive", "25 µm acrylic as supplied → 18.8 µm over the pads after it flows into the gaps", "asm", "response model only; Geant4 has no glue layer"),
                 P("Effective insulator, ESL → pads", "70.5 µm (kapton + glue in series)", "asm"),
                 P("Face", "470 × 470 mm (board)", "cad")],
         notes="This thickness and the ESL sheet resistance trade off exactly in the charge-spreading time, so the "
               "data cannot separate them: a real number here pins the sheet resistance.",
         questions=["q_kapton", "q_stackup"]),
    dict(id="pcb_l3", group="readout", name="PCB L3: guard ring", color="cu",
         short="Cu, outside active only",
         params=[P("Copper", "26 µm", "asm", "½ oz (18 µm) is the common fab default"),
                 P("Coverage", "0 inside the 399.36 mm window; 0.34 outside", "cad", "gerbers, analyze_cu_coverage.py")],
         notes="In the model the guard ring sits between the kapton and the pads, so 264 µm of FR4 separates the "
               "ESL from the pads in Geant4. That ordering is an artefact of how the model fills the board, and "
               "it disagrees with the response model, which laminates the kapton straight onto the pads.",
         questions=["q_stackup"]),
    dict(id="pcb_l4", group="readout", name="PCB L4: pads", color="cu",
         short="0.68 mm pads, 0.78 mm pitch",
         params=[P("Pattern", "512 × 512 square pads, 0.68 mm, on a 0.78 mm pitch", "cad", "every one of the 262 144 flashes checked"),
                 P("Active area (metallised)", "399.36 × 399.36 mm (512 × 0.78)", "cad"),
                 P("Copper", "26 µm", "asm"),
                 P("Coverage", "0.760 inside the window", "cad")],
         questions=["q_readout", "q_stackup"]),
    dict(id="pcb_l5", group="readout", name="PCB L5: Y strips", color="cu",
         short="dot + bus along y",
         params=[P("Pattern", "Ø 0.50 mm dot per cell + 0.098 mm bus along y", "cad"),
                 P("Channels", "512 at 0.78 mm", "cad"),
                 P("Copper", "26 µm", "asm"),
                 P("Coverage", "0.458 inside the window", "cad")],
         questions=["q_readout"]),
    dict(id="pcb_l6", group="readout", name="PCB L6: X strips", color="cu",
         short="dot + bus along x",
         params=[P("Pattern", "Ø 0.50 mm dot per cell + 0.098 mm bus along x", "cad"),
                 P("Channels", "512 at 0.78 mm", "cad"),
                 P("Copper", "26 µm", "asm"),
                 P("Coverage", "0.457 inside the window", "cad")],
         questions=["q_readout"]),
    dict(id="pcb_l7", group="readout", name="PCB L7: fan-out", color="cu",
         short="vias and routing",
         params=[P("Copper", "26 µm", "asm"),
                 P("Coverage", "0.53 inside the window, 0.06 outside", "cad"),
                 P("L8", "outline only, no copper", "cad")]),
    dict(id="pcb_fr4", group="readout", name="PCB dielectric (FR4)", color="fr4",
         short="5 × 264 µm (residual)",
         params=[P("Board body, mesh front → back", "1.70 mm", "cad", "one solid in the CAD"),
                 P("Dielectric per layer", "264.4 µm FR4 × 5", "asm", "the residual: 1.70 mm minus every other layer, split evenly"),
                 P("Material", "FR4, 1.85 g/cm³", "asm"),
                 P("Board face", "470 × 470 mm, offset (+15, +15) mm from the active-area axis", "cad", "STEP + gerbers")],
         notes="Because the FR4 is the residual, any other layer that changes silently moves it: when the ESL went "
               "100 → 10 µm the FR4 grew 246 → 264 µm. The alternative, a fixed dielectric stack and a 1.61 mm board, "
               "fits what we know equally well.",
         questions=["q_stackup"]),
    dict(id="rohacell", group="back", name="Rohacell", color="rohacell",
         short="5 mm",
         params=[P("Thickness", "5.0 mm", "cad"),
                 P("Grade / density", "Rohacell 51, 0.052 g/cm³", "asm", "grade not known"),
                 P("Face", "470 × 470 mm, offset (+15, +15)", "cad")],
         questions=["q_back"]),
    dict(id="backfoil", group="back", name="Aluminised mylar behind the Rohacell", color="mylar",
         short="25 µm + Al?",
         params=[P("Mylar", "25 µm", "asm", "not in the CAD; replaced a 50 µm bare-Al guess"),
                 P("Aluminium", "0.1 µm, facing downstream", "asm"),
                 P("Face", "470 × 470 mm", "asm")],
         questions=["q_back"]),
    dict(id="plate", group="back", name="Aluminium support plate", color="al",
         short="8 mm, 402 mm aperture",
         params=[P("Thickness", "8.0 mm", "cad"),
                 P("Aperture", "402 × 402 mm through-cut, concentric with the active area (1.3 mm clearance)", "cad", "Dylan confirmed: no Al behind the active area"),
                 P("Face", "470 × 470 mm, offset (+15, +15)", "cad"),
                 P("Holes", "~100 bolt holes, r ≤ 2.5 mm, at the periphery", "cad")]),
    dict(id="frame", group="lateral", name="Aluminium gas frame", color="al",
         short="30 mm spacer",
         params=[P("Size", "440 mm outer, 410.0 mm aperture, 15 mm wall, 30.000 mm tall", "cad", "aperture also in DFS3498A_gasframe.gbr"),
                 P("Material", "aluminium (from the part name)", "cad")]),
    dict(id="fieldcage", group="lateral", name="Field cage", color="fr4",
         short="4 PCBs, 1.62 mm",
         params=[P("Boards", "4 × 408.4 × 1.62 mm PCBs, 31.6 mm tall, lining the 410 mm aperture", "cad"),
                 P("Divider", "3 × 1.30 GΩ (700 V / 0.180 µA)", "ok", "HV monitor"),
                 P("Ring strips: number, pitch, width", "unknown", "open"),
                 P("Chamber B", "no ring chain at all: no uniform drift field", "ok", "STATUS.md; B is a hit detector")],
         questions=["q_fieldcage"]),
    dict(id="m1", group="lateral", name="Front-end M1 cards", color="fe",
         short="4 cards, on the board edge",
         params=[P("Cards", "4, two per connector edge (+x, +y), lying flat on the drift side of the board", "cad"),
                 P("Outline", "41 × 160 mm (gerber); 41 × 180 × 6.6 mm envelope (CAD)", "cad"),
                 P("Position", "radial 219.4–260.4 mm, centred at ±100 mm along the edge", "cad"),
                 P("Laminate", "1.6 mm, 6 Cu layers (coverage 0.04–0.14)", "asm"),
                 P("Connection", "pogo pins onto the board's connector pads; Mec8 outputs on top (not modelled)", "cad")]),
    dict(id="active", group="lateral", name="Active area", color="active",
         short="399.4 × 359.9 mm efficient",
         params=[P("Metallised (pads)", "399.36 × 399.36 mm", "cad"),
                 P("Efficient", "399.4 mm in X × 359.9 ± 1.8 mm in Y", "ok", "June bench vs the M3 telescope, and n_TOF run_79, agree to 1–2 mm"),
                 P("Dead bands", "~19 mm at each end of Y, centred", "ok"),
                 P("Cause", "probably the covered bus termination of the resistive strips", "open")],
         questions=["q_passivation"]),
]

# side view stack: (element id, thickness mm, x0, x1, label). x in the active-area frame. Order = beam order.
OFF = 15.0
B0, B1 = -235 + OFF, 235 + OFF   # the 470 mm plates (board, rohacell, plate), centred +15
STACK = [
    ("window", 0.060, -220, 220, "Mylar window"),
    ("window", 0.0001, -220, 220, "Al coating"),
    ("flange", 5.000, -200.1, 200.1, "Gas gap (window flange)"),
    ("cathode", 0.050, -203, 203, "Kapton cathode"),
    ("cathode", 0.009, -203, 203, "Cu cladding"),
    ("drift", 29.991, -203.38, 203.38, "Drift gas"),
    ("mesh", 0.038, -205, 205, "Micromesh"),
    ("ampgap", 0.150, -205, 205, "Amplification gap"),
    ("esl", 0.010, -206, 206, "Resistive strips (ESL)"),
    ("coverlay", 0.050, B0, B1, "Kapton"),
    ("pcb_l3", 0.026, B0, B1, "Cu L3 guard ring"),
    ("pcb_fr4", 0.2644, B0, B1, "FR4"),
    ("pcb_l4", 0.026, B0, B1, "Cu L4 pads"),
    ("pcb_fr4", 0.2644, B0, B1, "FR4"),
    ("pcb_l5", 0.026, B0, B1, "Cu L5 Y strips"),
    ("pcb_fr4", 0.2644, B0, B1, "FR4"),
    ("pcb_l6", 0.026, B0, B1, "Cu L6 X strips"),
    ("pcb_fr4", 0.2644, B0, B1, "FR4"),
    ("pcb_l7", 0.026, B0, B1, "Cu L7 fan-out"),
    ("pcb_fr4", 0.2644, B0, B1, "FR4"),
    ("rohacell", 5.000, B0, B1, "Rohacell"),
    ("backfoil", 0.025, B0, B1, "Back mylar"),
    ("backfoil", 0.0001, B0, B1, "Al coating"),
    ("plate", 8.000, B0, B1, "Al support plate"),
]

# --------------------------------------------------------------------------------------------- the questions
# pri: 1 = moves a published number, 2 = moves a systematic, 3 = completeness
Q = lambda id, el, pri, q, now, why: dict(id=id, el=el, pri=pri, q=q, now=now, why=why)

QUESTIONS = [
    Q("q_esl_geom", "esl", 1,
      "What are the width and the pitch of the resistive strips on our boards? Do they run along y, and how many are there?",
      "550 µm strips with 250 µm gaps (800 µm pitch), along y, 515 over 412 mm, all from memory. The strip artwork is not in the gerbers we have.",
      "The strip pattern sets how charge spreads between readout strips. The 800 µm resistive pitch against the 780 µm readout "
      "pitch beats slowly across the board: was that mismatch intended?"),
    Q("q_esl_depth", "esl", 1,
      "How thick are the resistive strips after printing and curing, and what paste was used (grade and supplier)?",
      "10 µm (\"something like 10 µm\"); an earlier model used 100 µm. The paste is modelled as plain C/H/O at 1.4 g/cm³.",
      "It sets the depth of the gas grooves between strips, the ESL material, and, with the sheet resistance, the spreading time."),
    Q("q_esl_r", "esl", 1,
      "What is the sheet resistance of the strips (MΩ/□), or the resistance along one strip end to end? Is there a measurement for these boards?",
      "Unknown. The charge-spread data give 1.4–2.6 MΩ/□ if the insulator under the strips is 70.5 µm; we use 2 MΩ/□.",
      "With the insulator thickness, this is the one input the charge-sharing model cannot get from the data."),
    Q("q_kapton", "coverlay", 1,
      "What insulator is between the resistive strips and the pads: kapton thickness, and how it is bonded (adhesive type and thickness)?",
      "50 µm kapton on 25 µm acrylic adhesive, which flows into the pad gaps and leaves 18.8 µm over the pads (70.5 µm in total).",
      "This and the sheet resistance trade off exactly in the spreading time, so the data cannot separate them."),
    Q("q_mesh", "mesh", 1,
      "What mesh is on these detectors: material, wire diameter, opening (or lines per inch), thickness, supplier?",
      "A placeholder copied from P2: stainless steel, 19 µm wire, 48 µm opening (67 µm pitch, ~380 lpi), 38 µm thick.",
      "It sets the electron transparency, the effective gap (where the mesh electrode plane is) and the mesh's material."),
    Q("q_calender", "mesh", 2,
      "Is the mesh calendered? If so, what is its thickness after calendering?",
      "Assumed not calendered (38 µm = 2 × wire). Nobody we asked knows.",
      "A calendered mesh is thinner and flatter: the effective electrode plane moves a few µm and the gap field rises ~1–2 %."),
    Q("q_window", "window", 2,
      "The front window: is it 60 µm mylar? Aluminised on one side or both, and how thick is the aluminium (nm)? Which side faces the gas?",
      "60 µm (from the CAD), 100 nm Al on the gas side only. Thickness and sides never measured.",
      "Material in front of the drift for electrons and positrons; it also decides whether the window is an electrode."),
    Q("q_cathode", "cathode", 2,
      "The drift cathode: 50 µm kapton with copper on one side? How thick is the copper, and does it face the drift gas?",
      "50 µm kapton (CAD) + 9 µm copper facing the gas (assumed).",
      "Material budget, and where exactly the drift field starts."),
    Q("q_ampgap", "ampgap", 2,
      "The amplification gap: 150 µm? Which photoresist (and how many laminations), and are the pillars Ø 0.6 mm on a 4.68 mm grid?",
      "150 µm (confirmed by Dylan); pillars from the bulk gerber, Ø 0.6 mm on 4.68 mm.",
      "The gap sets the gain and the field map."),
    Q("q_stackup", "pcb_fr4", 2,
      "Is there a stack-up drawing for the readout board? We need the order of the layers, the copper thickness and each dielectric thickness.",
      "Five 26 µm copper layers and 5 × 264 µm FR4, the FR4 being whatever is left of the CAD's 1.70 mm. The model puts the L3 guard ring "
      "between the kapton and the pads.",
      "Backscatter off the board, and the pad-to-ESL distance the signal model depends on."),
    Q("q_readout", "pcb_l4", 3,
      "Can you confirm the readout scheme: 512 × 512 pads (0.68 mm, 0.78 mm pitch) on L4, each pad connected by a via to an X strip (L6) or a Y strip (L5)? "
      "Which pads go to which view?",
      "Read from the gerbers: pads on L4, dots and buses along y on L5 and along x on L6. The pad-to-view assignment is not in the model.",
      "It decides how the two views share the charge, and the X/Y asymmetry we see."),
    Q("q_passivation", "active", 2,
      "We measure a ~19 mm dead band at each end of Y on every chamber (efficient area 399.4 × 359.9 mm). Is that the covered copper bus where the resistive strips are terminated?",
      "Measured on the bench and at n_TOF; the cause is our guess.",
      "Acceptance. Knowing the cause tells us whether the band is the same on every board."),
    Q("q_fieldcage", "fieldcage", 2,
      "The field cage: how many ring strips per board, at what pitch and width, and what is the resistor chain? Is chamber B's field cage missing or damaged?",
      "Four 1.62 mm PCBs; we measure 3 × 1.30 GΩ. Chamber B draws no drift current and has no uniform field.",
      "Field uniformity near the edges, and what B's data can be used for."),
    Q("q_drift", "drift", 3,
      "Is the drift gap exactly 30.0 mm at the centre once the window is under pressure, i.e. is the cathode taut and flat?",
      "30.000 mm (the frame height).",
      "Every depth and angle scales with the gap."),
    Q("q_flange", "flange", 3,
      "Is the 5 mm space between the window and the cathode filled by the chamber gas flow, or is it closed?",
      "Assumed to be chamber gas.",
      "Material and gas exchange; water pickup."),
    Q("q_bulge", "window", 3,
      "What overpressure did the chambers run at, or how much does the window bulge?",
      "~3 mbar → 8 mm sag (calculated).",
      "Gas in front of the drift; minor."),
    Q("q_back", "rohacell", 3,
      "Behind the board: which Rohacell grade, and what is the aluminised mylar on its back (thickness, aluminium)?",
      "Rohacell 51, 5 mm; 25 µm mylar + 100 nm Al.",
      "Material behind the gas (backscatter for the Sr-90 runs); small."),
]

GLOSSARY = [
    ("Calendering", "A finishing step for woven mesh: the mesh is pressed between heavy polished rollers, which "
     "flattens the wires where they cross (the knuckles). The mesh comes out thinner than twice the wire diameter "
     "(a 2 × 19 = 38 µm weave might end up ~25–30 µm) with flat, smooth faces. It is done so the mesh lies flat and "
     "laminates evenly in the bulk process. For us it changes the mesh thickness and material, slightly changes the "
     "opening, and moves the effective electrode plane, so the effective amplification gap and field change."),
    ("ESL", "The screen-printed resistive paste the resistive strips are made of (named after the paste maker, "
     "Electro-Science Laboratories)."),
    ("Bulk Micromegas", "The mesh is laminated onto the readout board between layers of photoresist, which is then "
     "developed into the pillars that hold the mesh at a fixed height."),
    ("Coverlay", "A kapton film with adhesive, laminated onto a PCB as an insulating cover."),
    ("Effective gap", "The distance from the anode to the plane where a flat electrode would give the same field as "
     "the real woven mesh. It is a little larger than the pillar height."),
]

CHAMBERS = [
    ("A", "det3", "good: 92.9 % on the bench", "X connector 8 (channels 448–511 of FEU 3) dead through run_79"),
    ("B", "det2", "91.3 % on the bench", "no field-cage ring chain: no uniform drift field, used as a hit detector only"),
    ("C", "det6", "57.8 %, spark-limited", "10 dead channels"),
    ("D", "det7", "43.1 %, the worst", "~130 of 512 X channels dead at n_TOF"),
]
