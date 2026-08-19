"""Byte-exact source model of slides/index.html.

The point of this module: let a browser edit the rendered deck and write the
result back into the *source file* without reserializing it. A DOM round-trip
would turn all 936 HTML entities into literal characters, normalise attribute
quoting and reflow whitespace — one enormous diff on the first save. Instead
every editable element is tagged, at serve time, with a `data-eid` that maps to
a byte span in the file; a save splices the new inner HTML into that span and
leaves every other byte alone.

Editable = an element that carries text and whose element children are all
inline (b/i/sub/sup/code/span/br/...). That picks up titles, kickers, captions,
bullets, table cells and stat labels, and skips the layout scaffolding, images
and bar-chart geometry.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
        'link', 'meta', 'param', 'source', 'track', 'wbr'}

# Children an editable element may contain and still be edited as one unit.
INLINE = {'b', 'i', 'em', 'strong', 'sub', 'sup', 'code', 'span',
          'a', 'small', 'u', 'mark', 'br'}

NEVER = {'script', 'style', 'title', 'head', 'html', 'body'}

# Chrome's contenteditable is told to emit these (styleWithCSS off); anything
# else that arrives is unwrapped rather than trusted.
ALLOWED_TAGS = {'b', 'i', 'em', 'strong', 'sub', 'sup', 'code', 'span',
                'small', 'u', 'mark', 'br', 'a'}
ALLOWED_ATTRS = {'class', 'href', 'title'}
# `style` is a special case: the deck uses inline style for the legend swatches
# (<i style="background:#a02c52">) and the copper PRELIMINARY spans, so it
# cannot simply be stripped — but it is also exactly how a browser smuggles
# junk formatting back in. A style attribute survives only if that same value
# was already in the field before the edit.
TAG_CANON = {'strong': 'b', 'em': 'i'}


@dataclass
class Element:
    tag: str
    attrs: dict
    tag_start: int           # '<' of the start tag
    content_start: int       # first byte after the start tag
    content_end: int = -1    # '<' of the end tag
    child_tags: set = field(default_factory=set)
    has_text: bool = False
    depth: int = 0

    @property
    def editable(self) -> bool:
        return (self.content_end > self.content_start
                and self.tag not in VOID and self.tag not in NEVER
                and self.tag not in INLINE          # edited inside its parent
                and self.has_text
                and all(t in INLINE for t in self.child_tags))


class _Parser(HTMLParser):
    def __init__(self, src: str):
        super().__init__(convert_charrefs=False)
        self.src = src
        self.line_starts = [0]
        for i, ch in enumerate(src):
            if ch == '\n':
                self.line_starts.append(i + 1)
        self.stack: list[Element] = []
        self.elements: list[Element] = []

    def _off(self) -> int:
        line, col = self.getpos()
        return self.line_starts[line - 1] + col

    def _note_child(self, tag: str):
        if self.stack:
            self.stack[-1].child_tags.add(tag)

    def handle_starttag(self, tag, attrs):
        self._note_child(tag)
        if tag in VOID:
            return
        start = self._off()
        el = Element(tag, dict(attrs), start, start + len(self.get_starttag_text()),
                     depth=len(self.stack))
        self.stack.append(el)

    def handle_startendtag(self, tag, attrs):
        self._note_child(tag)

    def handle_endtag(self, tag):
        end = self._off()
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i].tag == tag:
                # Anything still open above this is unclosed markup; drop it.
                for el in self.stack[i + 1:]:
                    el.content_end = end
                    self.elements.append(el)
                el = self.stack[i]
                el.content_end = end
                self.elements.append(el)
                del self.stack[i:]
                return
        # stray close tag — ignore

    def handle_data(self, data):
        if data.strip():
            for el in self.stack:
                el.has_text = True

    def handle_entityref(self, name):
        for el in self.stack:
            el.has_text = True

    def handle_charref(self, name):
        for el in self.stack:
            el.has_text = True


class DeckSource:
    """Parsed view of one deck file. Immutable; rebuild after every write."""

    def __init__(self, src: str):
        self.src = src
        p = _Parser(src)
        p.feed(src)
        p.close()
        deck_lo, deck_hi = self._deck_range(src, p.elements)
        self.elements = [e for e in p.elements
                         if deck_lo <= e.tag_start < deck_hi]
        # Only text inside a <section class="slide"> is deck content. The
        # counter and the progress bar also live in .deck and carry text, but
        # the deck's own script rewrites them at runtime — offering them for
        # editing writes the live slide number back into the source file.
        self.slides = [e for e in self.elements if e.tag == 'section'
                       and 'slide' in e.attrs.get('class', '').split()]
        self.editables = [e for e in self.elements
                          if e.editable and self._in_slide(e)]
        self.editables.sort(key=lambda e: e.content_start)
        self.by_eid = {i: e for i, e in enumerate(self.editables)}
        self.entity_map = entity_preferences(src)

    def _in_slide(self, e) -> bool:
        return any(s.content_start <= e.tag_start < s.content_end
                   for s in self.slides)

    @staticmethod
    def _deck_range(src, elements):
        """Only elements inside <div class="deck"> are offered for editing."""
        for e in elements:
            if e.tag == 'div' and 'deck' in e.attrs.get('class', '').split():
                return e.content_start, e.content_end
        return 0, len(src)

    def content(self, eid: int) -> str:
        e = self.by_eid[eid]
        return self.src[e.content_start:e.content_end]

    def inject_ids(self) -> str:
        """Return the source with data-eid="N" added to every editable tag."""
        out = self.src
        for eid in sorted(self.by_eid, reverse=True):
            e = self.by_eid[eid]
            cut = e.tag_start + 1 + len(e.tag)
            out = out[:cut] + f' data-eid="{eid}"' + out[cut:]
        return out

    def apply(self, edits: dict[int, str]) -> str:
        """Splice new inner HTML into the spans of `edits` (eid -> html)."""
        out = self.src
        for eid in sorted(edits, reverse=True):
            e = self.by_eid[eid]
            styles = frozenset(_STYLE_RE.findall(
                self.src[e.content_start:e.content_end]))
            new = restore_entities(
                sanitize(edits[eid], styles), self.entity_map)
            out = out[:e.content_start] + new + out[e.content_end:]
        return out


# --------------------------------------------------------------------------
# entity handling
# --------------------------------------------------------------------------

_STYLE_RE = re.compile(r'style="([^"]*)"')
_ENT_RE = re.compile(r'&([a-zA-Z][a-zA-Z0-9]*|#\d+|#[xX][0-9a-fA-F]+);')
# These already arrive escaped from innerHTML; re-escaping would double them.
_ENT_SKIP = {'<', '>', '&', '"'}


def entity_preferences(src: str) -> dict[str, str]:
    """Characters this file writes ONLY as a named entity.

    Derived from the file, and deliberately strict: the deck is mixed for most
    characters (145 literal em dashes against 155 `&mdash;`), so normalising
    them would rewrite text you did not touch inside a field you did. Restoring
    only the never-literal ones — `&nbsp;`, `&minus;`, the greek letters —
    keeps a save byte-exact everywhere the caret never went, while still
    protecting the characters that are invisible or confusable in the source.
    """
    ent_count: dict[str, dict[str, int]] = {}
    for m in _ENT_RE.finditer(src):
        text = m.group(0)
        ch = html.unescape(text)
        if len(ch) != 1 or ch in _ENT_SKIP:
            continue
        ent_count.setdefault(ch, {}).setdefault(text, 0)
        ent_count[ch][text] += 1

    prefs = {}
    for ch, forms in ent_count.items():
        best = max(forms, key=forms.get)
        if src.count(ch) == 0:         # never written literally in this file
            prefs[ch] = best
    prefs['\u00a0'] = '&nbsp;'         # always: invisible in a diff otherwise
    return prefs


def restore_entities(text: str, prefs: dict[str, str]) -> str:
    out = []
    for ch in text:
        out.append(prefs.get(ch, ch))
    return ''.join(out)


# --------------------------------------------------------------------------
# sanitiser — backstop for whatever contenteditable actually produces
# --------------------------------------------------------------------------

class _Sanitizer(HTMLParser):
    def __init__(self, keep_styles=frozenset()):
        super().__init__(convert_charrefs=False)
        self.out = []
        self.open: list[str | None] = []
        self.keep_styles = keep_styles

    def handle_starttag(self, tag, attrs):
        tag = TAG_CANON.get(tag, tag)
        if tag == 'br':
            self.out.append('<br>')
            return
        if tag in ('div', 'p'):
            # Chrome wraps a hard return in a block; the deck's idiom is <br>.
            if self.out:
                self.out.append('<br>')
            self.open.append(None)
            return
        if tag not in ALLOWED_TAGS:
            self.open.append(None)      # unwrap: keep the text, drop the tag
            return
        keep = {k: v for k, v in attrs if k in ALLOWED_ATTRS and v}
        style = dict(attrs).get('style')
        if style and style in self.keep_styles:
            keep['style'] = style
        if tag == 'span' and not keep:
            self.open.append(None)      # bare <span> = contenteditable residue
            return
        s = ''.join(f' {k}="{html.escape(v, quote=True)}"' for k, v in keep.items())
        self.out.append(f'<{tag}{s}>')
        self.open.append(tag)

    def handle_startendtag(self, tag, attrs):
        if TAG_CANON.get(tag, tag) == 'br':
            self.out.append('<br>')

    def handle_endtag(self, tag):
        tag = TAG_CANON.get(tag, tag)
        if tag == 'br':
            return
        while self.open:
            t = self.open.pop()
            if t is None:
                if tag in ('div', 'p'):
                    return
                continue
            self.out.append(f'</{t}>')
            if t == tag:
                return

    def handle_data(self, data):
        self.out.append(html.escape(data, quote=False))

    def handle_entityref(self, name):
        self.out.append(f'&{name};')

    def handle_charref(self, name):
        self.out.append(f'&#{name};')

    def result(self) -> str:
        for t in reversed(self.open):
            if t:
                self.out.append(f'</{t}>')
        self.open.clear()
        return ''.join(self.out)


def sanitize(fragment: str, keep_styles=frozenset()) -> str:
    p = _Sanitizer(keep_styles)
    p.feed(fragment)
    p.close()
    return p.result()
