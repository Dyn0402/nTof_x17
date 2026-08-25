# Editing the deck in the browser that renders it

A trial implementation of click-and-type editing for `slides/index.html`. It is
**deliberately separate from the deck**: nothing here runs unless you start the
server, and the normal workflow (`make_pdf.sh`, the mirror, hand edits in
PyCharm) is unaffected. The one thing it needed *in* the deck is the
`--fs-scale` block in the stylesheet, described below.

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
| **Alt+↑ / Alt+↓ / Alt+0** | bigger / smaller / back to original, at the scope the size panel is set to |
| **Alt+Shift+↑ / ↓ / 0** | force the *block* scope for one keystroke |
| **Ctrl+Shift+. / ,** | same as Alt+↑ / Alt+↓, if that is in your fingers |
| **Enter** | `<br>` |
| **Esc** | leave the field (or close the diff) |
| **← →** | page the deck, as always — but only when you are not in a field |

**The size panel** sits above the badge, bottom-left:

```
SIZE  [text] [block] [slide]   ul.bullets   [ − ] 0.95× [ + ] [reset]
```

`text` sizes the field the caret is in, `block` the list / column / table
around it, `slide` everything on screen. It shows what it is pointing at and
its current size, hovering it outlines that element, and the `−` `+` `reset`
buttons do the same thing as the keys without moving the caret. Use it rather
than the keys for the block scope: **Alt+Shift is the keyboard-layout chord on
most Linux desktops** and may be swallowed before the page ever sees it.

The badge counts unsaved edits and unsaved resizes separately. Every text
element in every slide is editable; every element *containing* text, up to and
including the slide itself, is resizable.

## Why it does not just save the DOM

Reserializing the page would convert every HTML entity to a literal character,
normalise attribute quoting and reflow whitespace — one 2000-line diff on the
first save. Instead, `deck_source.py` parses the file and records **byte
spans**: the inner HTML of each editable element, and the start tag of each
resizable one. A save splices only those. Every other byte is untouched, so a
one-word change is a one-line diff and your interleaved rationale comments
survive verbatim.

The test suite asserts this directly: re-saving every field with its own
unchanged content, and clearing every size that was never set, reproduces
`index.html` **byte for byte**.

## Text size

Resizing writes exactly one thing: `style="--fs-scale:0.92"` on the element's
own start tag. Nothing else, ever — the value is a bounded number and the
server refuses anything that is not.

That works because the deck's stylesheet was changed once, so that **every
`font-size` that reads the type scale multiplies by it**:

```css
.title{ font-size: calc(var(--fs-title) * var(--fs-scale, 1)); }
.spec-table{ font-size: calc(var(--fs-body)*.93 * var(--fs-scale, 1)); }
[style*="--fs-scale"]{ font-size: calc(1em * var(--fs-scale, 1)); }   /* first rule in the sheet */
```

**The factor has to be in the declaration, not in the `--fs-*` definition.**
This shipped the other way round first and silently did almost nothing: a
`var()` inside a custom property is substituted where that property is
*declared* — at `:root` — so `--fs-scale` set on a slide never reached it. Only
elements that inherit their size moved (through the fallback rule below), plus
the title slide, which declares its sizes directly. That was enough to make the
first round of tests pass. Section 9 of `test_edit.py` now sets a scale on one
element of all ~100 kinds in the deck and demands that it and everything under
it actually moves.

Written this way, `--fs-scale` on a title, a bullet list or a whole `<section>`
rescales the text under it and stays responsive — the `clamp()` keeps doing its
work. The last rule catches elements with no font-size of their own (a `<li>`
inside `.bullets`, a table cell): `1em` is their inherited size, so it
multiplies that. It is the first rule in the sheet on purpose — every class rule
out-ranks it — and it matches only an explicit style attribute, so a
container's scale is never applied twice.

**Nested scales replace, they do not multiply.** Setting 0.9 on a slide and 1.1
on a title in it gives that title 1.1, not 0.99.

With no element carrying the property the change is inert, and that is
measured, not assumed: with every `--fs-scale` stripped from the markup, all
2003 elements of the deck compute the same `font-size` and `line-height` with
the mechanism as without it, at 1600×900.

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
  again on the server. Two exceptions, both because the deck needs them:
  inline `style` survives if that exact value was already in the field (legend
  swatches, the copper PRELIMINARY spans), and an attribute-free `<span>`
  survives in a field that already had one — `.fig-head span`, `.bar-name span`
  and `.bar-val span` are the deck's own sub-labels and are indistinguishable
  from contenteditable residue by shape alone.
* Characters the file only ever writes as entities (`&nbsp;`, `&minus;`, the
  greek letters) are restored on save. Characters the file writes both ways
  (`—` appears literally as often as `&mdash;`) are left exactly as typed —
  normalising them would rewrite text you did not touch.

## Safety

* A GET never writes. Only Ctrl+S writes.
* Every save copies the file to `edit/.backups/index.<timestamp>.html` first.
* The page carries the hash of the file it was served from; if `index.html`
  changed on disk meanwhile, the save is refused with a 409 rather than
  clobbering the other change.
* The write is atomic (temp file + rename).
* The two kinds of span can never overlap — an editable's children are all
  inline, a resizable is never inline — and `apply()` raises rather than write
  a half-formed tag if they ever did.
* Downstream is checked, not assumed: `test_edit.py` re-splits the saved file
  the way `make_pdf.sh` does, prints a resized slide on its own through the
  same single-slide path, and measures that every text size on it moved by the
  factor that was written.

## Tests

```bash
../../.venv/bin/python edit/test_edit.py      # 97 checks, chrome for the last three sections
../../.venv/bin/python edit/test_browser.py   # 51 checks, drives real Chrome
```

`test_edit.py` covers the parse, the byte-exact identity property, id
injection, the sanitiser, `--fs-scale` on a tag, a full HTTP save round-trip
against a scratch copy, the print pipeline, and the deck-wide scaling sweep. `test_browser.py` drives
headless Chrome over the DevTools protocol (stdlib WebSocket client, no
playwright) and is the one that finds the real bugs: the space bar paging the
deck out from under the caret, `Ctrl+B` producing a styled span, the sanitiser
eating the legend swatch colours, the slide counter being written back into the
source, and the deck's own `<span>` sub-labels being stripped on any edit to
the field around them.

Both leave `slides/index.html` untouched — they work on `edit/scratch.html`.

## Status

Trial, not the default workflow. Nothing in the deck depends on it except the
`--fs-scale` block in the stylesheet, which is inert on its own. If it earns
its place, the thing worth adding is a `--watch` reload so an edit made in
PyCharm shows up in the browser too.
