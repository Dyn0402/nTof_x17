import sys, os
sys.path.insert(0, os.environ['PYLIB'])
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from copy import copy
src, out = sys.argv[1], sys.argv[2]
wb = openpyxl.load_workbook(src)
orig = wb.active
orig.title = "Stephan original"
ws = wb.copy_worksheet(orig)
ws.title = "Annotated (Dylan)"
wb.move_sheet(ws, offset=-1)
wb.active = 0
ws.column_dimensions['A'].width = 62
# header row 2
hdr = ["Stack-up entry (from Stephan's sheet)", "Gerber layer (file attribute)", "Gerber file", "Thickness (µm)",
       "Copper (µm)", "What it carries / our reading", "Not sure?"]
widths = [62, 14, 22, 12, 11, 70, 10]
bold = Font(bold=True); wrap = Alignment(wrap_text=True, vertical="top")
for i, (h, w) in enumerate(zip(hdr, widths)):
    c = ws.cell(row=2, column=i + 1, value=h); c.font = bold; c.alignment = wrap
    ws.column_dimensions[chr(65 + i)].width = w
unsure = PatternFill("solid", fgColor="FFF2CC")
rows = {  # row: (gerber layer, file, thickness, copper, note, unsure)
 3: ("", "", None, None, "Ni/Au finish on the top copper. Thicknesses of Ni and Au?", True),
 4: ("L3", "DFS3498A_top-cu", 26, 26, "Top copper: connector pads. 9 µm foil + 17 µm plating?", False),
 5: ("", "", 25, None, "Polyimide (UBE Upilex?)", True),
 6: ("", "", 25, None, "Prepreg. Is 25 + 25 = 50 µm the final thickness over the pads after pressing?", True),
 7: ("L4", "DFS3498A_L2-pads", 34, 34, "Pads, 0.68 mm squares on a 0.78 mm pitch. 17 µm + 17 µm via plating?", False),
 8: ("", "", 100, None, "Prepreg. Same material as the green 'EMC prepreg' rows, or a different one?", True),
 9: ("L5", "DFS3498A_L3-TrackY", 17, 17, "Y strips", False),
 10: ("", "", 100, None, "EMC prepreg (product, glass style, resin content?)", True),
 11: ("L6", "DFS3498A_L4-TrackX", 17, 17, "X strips", False),
 12: ("", "", 100, None, "Prepreg", True),
 13: ("L7", "DFS3498A_L5", 34, 34, "Called GND here, but the Gerber has only a 0.65 mm land under each pad in the active area. "
       "We think it is only the landing for the pad vias (drilled L4 to L7). Is that right, or is it grounded anywhere?", True),
 14: ("", "", 25, None, "EMC prepreg", False),
 15: ("", "", 25, None, "Polyimide", False),
 16: ("L8", "DFS3498A_Bot", 0, 0, "No copper (the Gerber holds only the board outline)", False),
}
for r, (lay, f, t, cu, note, uns) in rows.items():
    for col, v in zip("BCDEF", (lay, f, t, cu, note)):
        c = ws[f"{col}{r}"]; c.value = v; c.alignment = wrap
    if uns:
        ws[f"G{r}"] = "?"; ws[f"G{r}"].alignment = Alignment(horizontal="center", vertical="top")
        for col in "BCDEFG": ws[f"{col}{r}"].fill = unsure
for r in range(3, 17): ws[f"A{r}"].alignment = Alignment(vertical="top")
r = 18
ws[f"A{r}"] = "Total"; ws[f"A{r}"].font = bold
ws[f"D{r}"] = "=SUM(D3:D16)"; ws[f"E{r}"] = "=SUM(E3:E16)"
ws[f"D{r}"].font = bold; ws[f"E{r}"].font = bold
ws[f"F{r}"] = ("Offer 3498-A: 530 µm total, of which about 68 µm copper. Our sum gives 528 µm, but about 128 µm copper. "
               "Our mechanical CAD has the board as 1.70 mm. What is the as-built thickness?")
ws[f"F{r}"].alignment = wrap; ws[f"G{r}"] = "?"
for col in "BCDEFG": ws[f"{col}{r}"].fill = unsure
r = 20
ws[f"A{r}"] = "Vias (from the drill files)"; ws[f"A{r}"].font = bold
vias = [("Ø0.25 mm, L3 to L4", "DFS3498A_pla1-2"),
        ("Ø0.25 / Ø0.50 mm, L4 to L7: one per pad, with a land on each layer it crosses", "DFS3498A_pla2-5")]
for i, (d, f) in enumerate(vias):
    ws[f"A{r+1+i}"] = d; ws[f"C{r+1+i}"] = f; ws[f"A{r+1+i}"].alignment = wrap
ws[f"F{r+1}"] = "Filled or open?"; ws[f"G{r+1}"] = "?"
for col in "FG": ws[f"{col}{r+1}"].fill = unsure
r = 24
ws[f"A{r}"] = "Not part of the circuit stack"; ws[f"A{r}"].font = bold
ws[f"A{r+1}"] = "L1 3498A_bulk: bulk / pillar artwork.  L2 3498A_top-resist: outline only, 412 x 412 mm."
ws[f"A{r+3}"] = "Rows shaded light orange and marked '?' are the ones we are not sure about. Column A and its colors are Stephan's original."
ws[f"A{r+3}"].font = Font(italic=True)
ws.freeze_panes = "B3"
wb.save(out)
