# What a good poster looks like: 25 flash-talk posters from SQM 2026, QM 2025 and QM 2023

Studied 2026-11-18 to redesign the INFN-IN2P3 poster (v2 → v3). At all three conferences the committee picks a
handful of posters for a flash talk in the plenary (QM 2025: 11 of about 540). The flash talks come from each Indico
timetable (`/export/timetable/<id>.json`), and the posters are the attachments on the matching poster contribution.

| Conference | Indico | Flash-talk posters read |
|---|---|---|
| SQM 2026, UCLA | indico.global/event/13943 | Shen (STAR dark photons), Likmeta (ALICE flow), Kahangirwe (CMF-HRG EoS), Schmidt (LHCb UPC), Lim (ALICE ρ⁰) |
| QM 2025, Frankfurt | indico.cern.ch/event/1334113 | T. Hwang (ALICE EECs, award), Garetti (ALICE charmonium, award), Jin (STAR light nuclei), Omana Kuttan (HEIDi), De Lescluze (QNMs), Avramescu (glasma), Authier (LHCb SMOG2), J. Hwang (sPHENIX INTT), Das (ATLAS γ+Pb), Gholami (NJL), Lee (CMS DPS) |
| QM 2023, Houston | indico.cern.ch/event/1139644 | Shih (sPHENIX INTT), Riedel (ALICE femto), Stefaniak (HADES), Hirvonen (deep learning flow), Blankenship (PHENIX), Shangase (LHCb), Ji (STAR hypertriton), Chen (ALEPH ridge), Romero (ATLAS dijets) |

The SQM PDFs are saved in `$WORK_DATA/claude-jobs/078243ef/posters/dl/` (scratch). indico.cern.ch blocks scripts
(an Anubis bot check), so the QM posters were only viewed in the browser.

## The honest caveat

These posters were picked for the physics and the presenter, not the design. Most are conventional: A0 portrait,
a logo header, and two or three columns of boxed sections. Nobody uses the "#betterposter" style with one giant
sentence in the middle. So the lessons below are what these posters **share**, plus what the few that stand out do
differently. That is a safe target for a committee that awards the best poster.

## What they share

1. **A short, plain story in reading order.** The sections are motivation → detector or method → results →
   summary. They are numbered or laid out as columns, and you read each column from top to bottom. No poster
   makes you hunt for the next box.
2. **Results get the most space.** Typically 40–60 % of the area. The motivation and the detector are short.
3. **Few figures, large figures.** Six to ten plots at most, each wide enough that the axis labels can be read
   from about 1.5 m. Figures are cropped to the panel that carries the point.
4. **Every section ends with its conclusion in words.** It is a bold line or a coloured box under the plot.
   Riedel has a boxed sentence per section. Garetti's sub-headings are claims ("larger QGP effects expected for
   central collisions"). The INTT poster says "Clear correlation → real signal!". Romero's figure titles are
   findings ("Larger dijets are more balanced").
5. **A Summary or Conclusions box at the end, nearly always** (all but one or two). It has three to five bullets and usually an
   outlook.
6. **Big numbers under detector plots.** The two INTT posters, the closest analogue to ours, print
   "99.33 ± 0.04 %" or "Signal 99.7 %" in large type under each beam-test plot.
7. **Short bullets.** One or two lines each, with key words in a highlight colour that is used consistently.
   Equations appear only where the method depends on them.
8. **A contact, a QR code (STAR), and small references at the very bottom.**

## What the ones that stand out do

- **Avramescu (QM25)** puts the main result in a full-width band at the top, under the title, before any
  motivation. The take-home message is read first.
- **Omana Kuttan (QM25)** uses a narrative spine: Why? → Applications → The model → Performance → What's next,
  joined by a curve with icons. It is memorable, and it works for a method or tool poster.
- **Riedel (QM23)** uses four big plots, each with a boxed one-sentence conclusion and little else.
- **Shih (QM23 INTT)** tells a detector poster as a commissioning story: design → beam test → installation → first
  collisions, with each cross-check worded as a conclusion.

## What was wrong with mock-up v2

- **No main message.** The title was a description, and there was no take-home line and no summary box.
- **Seven panels in rows of different widths.** The reading order zig-zagged.
- **Paper figures pasted at panel size.** Their 7–9 pt text printed at about 3 mm: unreadable at poster distance
  (panel 2(b), the small plots in panel 5, the n_TOF histogram).
- **Too much hardware.** The renders took about 35 % of the area and the performance results about 15 %.
- **Six equal-weight number tiles**, and grey captions under everything.

## What v3 does about it

- A full-width hero band (Avramescu): the box render beside the take-home sentence in large type.
- Two columns, read top to bottom. On the left, why → chamber → reconstruction. On the right, performance → n_TOF →
  summary.
- Every heading states its point ("1° per chamber, flat over ±20°"; "In the n_TOF beam, the tracks find the ³He
  target"), and the result sits in a box or tile under the figure.
- The two key plots are redrawn at their printed size with 21–28 pt text (`make_poster_figs.py`, SVG). The
  exploded chamber's layer labels are HTML type at poster size, not baked-in pixels.
- A dark Summary and next-step box closes the right column, with the ATOMKI acceptance bar, the QR codes and the
  contact.
- Body text is 8.6 mm (about 24 pt); headings are 11.6 mm.
