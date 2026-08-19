# Editing the deck in the browser that renders it

A trial implementation of click-and-type editing for `slides/index.html`. It is
**deliberately separate from the deck**: `index.html` contains none of this, the
normal workflow is unaffected, and nothing here runs unless you start the server.

```bash
cd mpgd26/slides
../../.venv/bin/python edit/edit_server.py            # edit index.html for real
../../.venv/bin/python edit/edit_server.py --scratch  # edit a throwaway copy
```

* `http://localhost:8017/` — the deck exactly as it is on disk, no edit machinery.
* `http://localhost:8017/?edit` — every text element is live. Click a title, a
  bullet, a caption, a table cell, a stat number, and type.

| key | |
|---|---|
| **Ctrl+S** | write the changed fields back to `index.html` |
| **Ctrl+D** | show the diff a save *would* make, before making it |
| **Ctrl+B / Ctrl+I** | bold / italic — emits `<b>`/`<i>`, the deck's own idiom |
| **Enter** | `<br>` |
| **Esc** | leave the field (or close the diff) |
| **← →** | page the deck, as always — but only when you are not in a field |

The badge bottom-left counts unsaved fields. 578 fields are editable.

## Why it does not just save the DOM

Reserializing the page would convert all 936 HTML entities to literal
characters, normalise attribute quoting and reflow whitespace — one 2000-line
diff on the first save. Instead, `deck_source.py` parses the file, records the
**byte span** of each editable element, and serves the page with a `data-eid`
on each. A save splices the new inner HTML into that span. Every other byte of
the file is untouched, so a one-word change is a one-line diff and your
interleaved rationale comments survive verbatim.

The test suite asserts this directly: re-saving all 578 fields with their own
unchanged content reproduces `index.html` **byte for byte**.

## What it will and will not write

* **Editable** = an element carrying text whose children are all inline
  (`b`, `i`, `sub`, `sup`, `code`, `span`, `br`). Titles, kickers, bullets,
  captions, figure labels, table cells, stat numbers and labels.
* **Not editable** = layout scaffolding, images, bar-chart geometry, and
  anything outside a `<section class="slide">`. That last exclusion is not
  cosmetic: the slide counter lives in `.deck` and the deck's own script
  rewrites it at load, so offering it for editing wrote the live slide number
  (`1 / 15` → `2 / 82`) into the source. Found by the browser test.
* Markup the browser invents — `<span style="font-weight:700">`, bare `<span>`
  residue, `<font>`, pasted rich text — is normalised away in the page and
  again on the server. Inline `style` survives **only** if that exact value was
  already in that field (the legend swatches and the copper PRELIMINARY spans
  need it; nothing new can be introduced).
* Characters the file only ever writes as entities (`&nbsp;`, `&minus;`, the
  greek letters) are restored on save. Characters the file writes both ways
  (`—` appears 145 times literally and 155 times as `&mdash;`) are left exactly
  as typed — normalising them would rewrite text you did not touch.

## Safety

* A GET never writes. Only Ctrl+S writes.
* Every save copies the file to `edit/.backups/index.<timestamp>.html` first.
* The page carries the hash of the file it was served from; if `index.html`
  changed on disk meanwhile, the save is refused with a 409 rather than
  clobbering the other change.
* The write is atomic (temp file + rename).
* Downstream is unaffected: after a save, `make_pdf.sh` still splits 82 slides,
  `tools/mirror_slides_to_site.py` still sees its 80 asset references, and the
  slides still print.

## Tests

```bash
../../.venv/bin/python edit/test_edit.py      # 50 checks, no browser
../../.venv/bin/python edit/test_browser.py   # 24 checks, drives real Chrome
```

`test_edit.py` covers the parse, the byte-exact identity property, id
injection, the sanitiser, and a full HTTP save round-trip against a scratch
copy. `test_browser.py` drives headless Chrome over the DevTools protocol
(stdlib WebSocket client, no playwright) and is the one that found the real
bugs: the space bar paging the deck out from under the caret, `Ctrl+B`
producing a styled span, the sanitiser eating the legend swatch colours, and
the slide counter being written back into the source.

Both leave `slides/index.html` untouched — they work on `edit/scratch.html`.

## Status

Trial, not the default workflow. Nothing in the deck depends on it. If it earns
its place, the only thing worth adding is a `--watch` reload so an edit made in
PyCharm shows up in the browser too.
