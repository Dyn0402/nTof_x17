# Readout board 3498-A stack-up

The readout board (CERN MPT workshop, offer 3498-A, 2025-03-27) was laid out by Bertrand Mehl. Stephan forwarded the
stack-up spreadsheet on 2026-11-24.

- `stack-up_3498-A_original.xlsx`: the spreadsheet as Stephan sent it.
- `MX17_readout_3498A_stackup_annotated.xlsx`: our reading of it. It adds the Gerber layer (from `TF.FileFunction`), the
  file, the thickness and copper as numbers, and notes. Rows we are unsure of are marked "?". Built by
  `make_annotated.py` (`PYLIB=<dir with openpyxl> python3 make_annotated.py <original.xlsx> <out.xlsx>`).
- `email_bertrand_2026-11-24.txt`: the short email sent to Bertrand (cc Stephan, Pierfrancesco) with the annotated
  sheet and the Gerber zip (`MX17_Geant/design/gerbers/readout_pcb`). Dylan edited it before sending.
- `followup_questions_draft.txt`: the long first draft. It holds the questions held back for a follow-up email.

What the spreadsheet says, top (mesh side) down: Ni/Au; L3 top copper 9+17 µm; 25 µm UBE polyimide + 25 µm EMC prepreg;
L4 pads 17+17 µm; 100 µm prepreg; L5 Y strips 17 µm; 100 µm EMC prepreg; L6 X strips 17 µm; 100 µm prepreg;
L7 17+17 µm ("GND"); 25 µm EMC prepreg + 25 µm UBE polyimide; L8 no copper. Sum 528 µm (offer: 530 µm).

How this disagrees with our models (not changed yet, waiting for Bertrand's answer):
- Board thickness: the CAD and Geant4 use a 1.70 mm board with 5 × 264 µm FR4 as a residual. The stack-up gives 0.53 mm.
- Copper: the layers add to about 128 µm. The offer says about 68 µm. Geant4 uses 5 × 26 µm.
- Insulator above the pads: the response model uses 50 µm Kapton + 25 µm glue (d_eff 70.5 µm, `GLUE_FLOW_MODEL`
  "conserved"). The stack-up gives 25 µm polyimide + 25 µm prepreg.
- L7 is called GND, but the Gerber has only 0.65 mm via lands under the pads.
