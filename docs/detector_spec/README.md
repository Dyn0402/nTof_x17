# MX17 detector specification page (for review before publication)

**Live:** <https://dylan-neff.web.cern.ch/notes/mx17-detector-spec.html> (unlisted). Written 2026-10-16 so that
Stephan (and anyone who built the chambers) can check every number in the module stack and answer the open questions
in place.

- `spec.py` — **the data**: every part, each value with a status (`ok` confirmed, `cad` CAD/gerber, `asm` assumed,
  `open` unknown) and its source, the side-view stack, the open questions, the glossary (calendering), the four chambers.
  The numbers come from `MX17_Geant/shared/MX17ModuleGeometry.hh` (`AsBuiltSpec`) and `MX17_Geant/design/*.md`;
  when an answer comes in, update both the Geant4 spec and this file.
- `template.html` — the page: side view (log-scaled thickness, true-scale bar), top view with zoom insets, the
  amplification region to scale, questions, all parameters, chambers, comments. Data is injected at `/*__DATA__*/null`.
- `build_page.py` — builds `out/mx17-detector-spec.html`; `--publish` adds it to the site's notes and deploys.

## Comments

The page uses the same Google Apps Script as the run-budget page
(`x17_facility_search/budget/feedback_apps_script.gs`), sending `page: "detector"`; those comments go to a
"Detector" tab of the budget feedback sheet. The page only POSTs once a GET with `?page=detector` answers
`page: "detector"`. Until the script is redeployed with that version it falls back to `mailto:` links, so nothing
can land in the budget's tab.

**To switch the comment box on (Dylan, once):** open the v2 feedback sheet → Extensions → Apps Script, replace the
code with the current `feedback_apps_script.gs`, Save, then Deploy → Manage deployments → edit → Version: **New
version** → Deploy. The URL stays the same, so neither page needs a rebuild. Check:
`curl -sL "<endpoint>?page=detector" | head -c 100` should show `"page":"detector"`.

Read answers back with the Drive connector (`read_file_content` on the sheet, tab "Detector"); mark one handled by
writing anything in its `status` column.

Local test with a mock backend: `DETECTOR_FEEDBACK_ENDPOINT=http://127.0.0.1:8766/mock python3 build_page.py`.
