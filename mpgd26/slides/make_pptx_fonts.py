"""Cut static instances out of the Noto variable fonts, for to_pptx.py.

Takes the directory holding the upstream variable fonts and writes the faces
into <dir>/static.  Fetch the sources first:

  https://github.com/google/fonts/raw/main/ofl/notosansdisplay/NotoSansDisplay%5Bwdth,wght%5D.ttf
  https://github.com/google/fonts/raw/main/ofl/notosansdisplay/NotoSansDisplay-Italic%5Bwdth,wght%5D.ttf
  https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf

saved as NotoSansDisplay-VF.ttf, NotoSansDisplay-Italic-VF.ttf and
NotoSans-VF.ttf.  Noto Sans and Noto Sans Mono ship real statics and can be
taken as-is from notofonts.github.io/fonts/<Family>/hinted/ttf/.


The deck asks for font-weight 400 / 600 / 650 / 700 / 800.  PowerPoint can only
address two faces per family name (regular and bold), so anything heavier than
regular but lighter than bold needs a family name of its own -- exactly the way
Windows ships "Segoe UI Semibold" beside "Segoe UI".  Cutting the instances
rather than letting the browser synthesise them means the measuring browser and
PowerPoint use the same outlines and therefore the same advance widths.
"""
import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

DL = Path(sys.argv[1])
OUT = DL / "static"
OUT.mkdir(exist_ok=True)

# (source vf, weight, output family name, subfamily, output filename)
JOBS = [
    ("NotoSansDisplay-VF.ttf", 400, "Noto Sans Display", "Regular",
     "NotoSansDisplay-Regular.ttf"),
    ("NotoSansDisplay-VF.ttf", 700, "Noto Sans Display", "Bold",
     "NotoSansDisplay-Bold.ttf"),
    ("NotoSansDisplay-VF.ttf", 600, "Noto Sans Display SemiBold", "Regular",
     "NotoSansDisplaySemiBold-Regular.ttf"),
    ("NotoSansDisplay-VF.ttf", 800, "Noto Sans Display ExtraBold", "Regular",
     "NotoSansDisplayExtraBold-Regular.ttf"),
    ("NotoSansDisplay-Italic-VF.ttf", 400, "Noto Sans Display", "Italic",
     "NotoSansDisplay-Italic.ttf"),
    ("NotoSansDisplay-Italic-VF.ttf", 700, "Noto Sans Display", "Bold Italic",
     "NotoSansDisplay-BoldItalic.ttf"),
    ("NotoSans-VF.ttf", 600, "Noto Sans SemiBold", "Regular",
     "NotoSansSemiBold-Regular.ttf"),
]

# name-table ids: 1 family, 2 subfamily, 3 unique, 4 full, 6 postscript,
# 16 typographic family, 17 typographic subfamily.
MAC, WIN = (1, 0, 0), (3, 1, 0x409)


def setname(font, nid, value):
    for plat in (MAC, WIN):
        font["name"].setName(value, nid, *plat)


for src, wght, family, sub, fname in JOBS:
    f = TTFont(DL / src)
    axes = {"wght": wght}
    if "wdth" in {a.axisTag for a in f["fvar"].axes}:
        axes["wdth"] = 100
    inst = instancer.instantiateVariableFont(f, axes, inplace=True,
                                             updateFontNames=False)
    ps = family.replace(" ", "") + "-" + sub.replace(" ", "")
    setname(inst, 1, family)
    setname(inst, 2, sub)
    setname(inst, 3, "%s %s; converted for MPGD2026 deck" % (family, sub))
    setname(inst, 4, "%s %s" % (family, sub))
    setname(inst, 6, ps)
    # The typographic names (16/17) override the Win32 ones when they are
    # present, which is how a "SemiBold" face gets silently folded back into
    # its parent family.  Set them to match rather than trusting a removal.
    setname(inst, 16, family)
    setname(inst, 17, sub)
    # OS/2 usWeightClass and the bold/italic bits drive how Windows and
    # PowerPoint pair the faces up inside a family.
    inst["OS/2"].usWeightClass = 700 if "Bold" in sub else 400
    bold = sub.startswith("Bold")
    ital = "Italic" in sub
    # Bit 6 (regular) must be clear whenever bit 0 (italic) or bit 5 (bold) is
    # set, or Windows refuses to pair the face into its family.
    inst["OS/2"].fsSelection = ((inst["OS/2"].fsSelection & ~0x61)
                                | (0x20 if bold else 0) | (0x01 if ital else 0)
                                | (0x40 if not bold and not ital else 0))
    inst["head"].macStyle = (1 if bold else 0) | (2 if ital else 0)
    inst.save(OUT / fname)
    print("cut %-42s %s %s (wght %d)" % (fname, family, sub, wght))

print("\nstatic faces in", OUT)
