# Paper II split: Dylan's decisions (recorded 2026-11-12/13)

Source: the decision page https://claude.ai/artifact/GPYyBp8UmXDK5wgt3z31zu (`index.html` here), collection `decisions`,
read back 2026-11-13. No notes were attached to any choice, and D12 (free text) was left empty. Every choice was the
recommended option except **D08**.

**This replaces the 11-09 decision** to keep the review as a separate third paper with its own framing. That paper is now
IIb (internally still "Paper III"; see D10).

## The decisions

| # | Question | Decision |
|---|---|---|
| D01 | One paper or two? | **Two.** IIa = reconstruction (`papers/paper2/`), IIb = the bias law and the survey (`papers/tpc_mixing_review/`). |
| D02 | Where does H2c go? | **IIa, as one table row** (H0, H2c, W). IIb cites it and shows the variant and SPS scans. |
| D03 | "What a hit readout must record" (old Paper II §6.1) | **IIb owns it**; IIa keeps one sentence. |
| D04 | Who owns the compression numbers (20–30 %, C = 7.0 / 11.7)? | **IIb owns them**; IIa shows one motivating figure and cites IIb. |
| D05 | When does IIb go out? | **A few weeks after I and IIa**, so its open items don't set the date. Paper I depends only on IIa. |
| D06 | IIb framing | **A measurement with a literature survey**, not a review across TPC families. |
| D07 | IIb venue | **JINST.** IIa stays at NIM A. |
| D08 | Does the first-author reasoning move to IIb? | **First author on both; no change needed.** (Not the recommendation, which was to move it to IIb.) |
| D09 | Paper I design lesson | **Reword:** the resistive layer adds to the bias in one view and costs little once waveforms are fitted. |
| D10 | Naming | **Keep I / II / III internally**; choose public titles at submission. |
| D11 | What the next session does | **All of it:** update the strategy and outline docs, and make the Paper I and Paper II text edits. |

## Work list that follows (for the next session)

Paper I is shared with Stephan on Overleaf. Before touching `paper1/` or `paper1_v2/`: `papers/overleaf.sh status <paper>`,
then `pull`; stop and report on a conflict or a hold. Push small, list the pushed files.

1. **`papers/JOURNAL_STRATEGY.md`.** Three papers: I (NIM A or JINST, as already planned), IIa (NIM A, `elsarticle`), IIb
   (JINST). IIb follows I + IIa by a few weeks. The "general claim" text (lines ~18–32) moves to IIb, but per D08 the
   reputation paragraph stays as "first author on both" (no either/or). Drop the EPJ C option for IIb. Fix the "delayed
   copies" wording (line ~18): the bias is the aggregate-time compression, not delayed copies (CLAUDE.md, `P0_XBIAS_2026-11-05.md`).
2. **`papers/paper2/OUTLINE_PLAN_2026-11-05.md`.** Re-scope to IIa: forward model, calibration, fleet; H0/H2c/W as one
   table; one motivating compression figure citing IIb; §6.1 (readout requirements) cut to one sentence pointing at IIb.
   Move the readout-requirements and compression material into IIb's plan (`papers/tpc_mixing_review/README.md` or a new outline).
3. **`papers/paper2/main.tex`** title, abstract, §3 intent to match IIa.
4. **`papers/tpc_mixing_review/`**: framing as a measurement with a survey (D06), JINST class (D07); it now owns the
   compression numbers, C/N² + b, remedies, readout requirements and the survey. The draft corrections listed in
   `HANDOFF.md` ("Paper III" section) are still unapplied; they are IIb work, not blocking I/IIa.
5. **Paper I text** (D09), in **both** `paper1/` and `paper1_v2/` (v2 is the reorganised copy on its own Overleaf project;
   check with Dylan which one is live if unsure, don't assume):
   - design lesson: `paper1/sections/02_design.tex:15-18`, `paper1_v2/sections/03_design.tex` — retire "delayed copies
     29 % at 47 ns", reword per D09.
   - `paper1/sections/03_setups.tex:73-77`: split the citation (Paper I cites IIa for the reconstruction, IIb for the bias law).
   - `paper1/main.tex:51` stale `\requote`; `paper1_v2/drafting.sty` has a match too, check it.
6. Update `papers/README.md` if it lists the papers, and the hub to-do "Think about splitting up? Ask Gianpiero" (Dylan's
   to tick, not ours; mention it).
