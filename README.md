# MPGD2026 proceedings (JINST)

Dylan Neff, "Searching for X17 at n_TOF with large-area Micromegas detectors in µTPC mode",
oral talk, MPGD2026, Prague, 3 Sept 2026. **Deadline: 17 January 2027**, submitted via jinst.sissa.it
(Submit → "submit a proceedings contribution" → MPGD2026).

- Limit: **5 pages** for an oral talk, not counting authors/affiliations, acknowledgements or references.
  The current draft's body runs over pages 2–6.
- `main.tex` follows the official JINST template; `jinstpub.sty` is SISSA's own copy (same as the Overleaf template).
- `figures/` are copies of the talk's assets (`mpgd26/slides/assets/img/`). Their provenance is in
  `mpgd26/slides/NOTES.md`.
- `\askDylan{…}` (cyan) and `% TODO Dylan` mark open items. Remove all of them before submitting.
- `draft_preview.pdf` (gitignored, local only) was compiled locally with Times (`mathptmx`) in place of `newtx`, because this machine's TeX Live 2023
  cannot load the current newtx. Overleaf compiles the real thing, so expect small differences in line breaks.

Overleaf project (Dylan's copy of the JINST template): https://www.overleaf.com/project/6ad72207b122b98dea142d04
Link once, from a machine with the Overleaf git token:

    papers/overleaf.sh link mpgd26_proceedings https://git.overleaf.com/6ad72207b122b98dea142d04

That replaces the template's files at the project's head with this folder; they stay in the Overleaf history.
Then set `main.tex` as the main document.
