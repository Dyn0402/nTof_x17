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

There is a second, wider set: RESIZABLE elements. Text size is not content, it
is one custom property on the start tag -- style="--fs-scale:.92" -- and it
inherits, so it can be set on something that is not editable at all: a bullet
list, a column, a whole <section class="slide">. Those get a `data-fsid` and
their START TAG is the byte span that a save rewrites. The two kinds of span
never overlap (an editable's children are all inline; a resizable is never
inline), which is what lets one save carry both.
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
    def resizable(self) -> bool:
        """Can carry --fs-scale. Wider than `editable`: containers count."""
        return (self.content_end > self.content_start
                and self.tag not in VOID and self.tag not in NEVER
                and self.tag not in INLINE   # would sit inside a content span
                and self.has_text)

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
        # A slide is resizable as a whole (that is the "this one is overfull"
        # fix), so the containment test here includes the slide itself.
        self.resizables = [e for e in self.elements
                           if e.resizable and (e in self.slides or self._in_slide(e))]
        self.resizables.sort(key=lambda e: (e.tag_start, e.depth))
        self.by_fsid = {i: e for i, e in enumerate(self.resizables)}
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

    def start_tag(self, fsid: int) -> str:
        e = self.by_fsid[fsid]
        return self.src[e.tag_start:e.content_start]

    def scale(self, fsid: int):
        """The --fs-scale currently written on this element, or None."""
        m = _SCALE_RE.search(self.start_tag(fsid))
        return m.group(1).strip() if m else None

    def inject_ids(self) -> str:
        """Add data-eid to every editable tag and data-fsid to every resizable."""
        add: dict[int, list[str]] = {}
        for eid, e in self.by_eid.items():
            add.setdefault(id(e), []).append(f'data-eid="{eid}"')
        for fsid, e in self.by_fsid.items():
            add.setdefault(id(e), []).append(f'data-fsid="{fsid}"')
        # At the END of the start tag, not after the tag name: `<section
        # class="slide ...` is a literal that make_pdf.sh and several checks
        # match on, and it must survive being served.
        cuts = {}
        for e in list(self.by_eid.values()) + list(self.by_fsid.values()):
            cuts[id(e)] = self.src.rindex('>', e.tag_start, e.content_start)
        out = self.src
        for key in sorted(cuts, key=lambda k: cuts[k], reverse=True):
            cut = cuts[key]
            out = out[:cut] + ' ' + ' '.join(add[key]) + out[cut:]
        return out

    def apply(self, edits: dict[int, str] = None,
              scales: dict[int, object] = None) -> str:
        """Rewrite the spans named by `edits` (eid -> inner html) and `scales`
        (fsid -> factor, or None to clear). Every other byte is left alone."""
        reps = []                       # (start, end, replacement)
        for eid, frag in (edits or {}).items():
            e = self.by_eid[int(eid)]
            was = self.src[e.content_start:e.content_end]
            keep = frozenset(_STYLE_RE.findall(was))
            # `.fig-head span`, `.bar-name span` and `.bar-val span` are the
            # deck's own sub-labels, written as an attribute-free <span>, which
            # is otherwise exactly what contenteditable leaves behind. Keep them
            # in the fields that already had one; the page marks the deck's own
            # so the browser does not strip them either.
            reps.append((e.content_start, e.content_end,
                         restore_entities(
                             sanitize(frag, keep, bool(BARE_SPAN_RE.search(was))),
                             self.entity_map)))
        for fsid, k in (scales or {}).items():
            e = self.by_fsid[int(fsid)]
            reps.append((e.tag_start, e.content_start,
                         set_scale(self.src[e.tag_start:e.content_start], k)))

        reps.sort(reverse=True)
        prev = len(self.src)
        for start, end, _ in reps:       # a half-written tag is unrecoverable
            if end > prev:
                raise ValueError(f'overlapping edit spans at {start}..{end}')
            prev = start
        out = self.src
        for start, end, new in reps:
            out = out[:start] + new + out[end:]
        return out


# --------------------------------------------------------------------------
# --fs-scale on a start tag
# --------------------------------------------------------------------------

_SCALE_RE = re.compile(r'--fs-scale\s*:\s*([^;"\']*)')
_SCALE_DECL_RE = re.compile(r'\s*;?\s*--fs-scale\s*:[^;"]*;?')
_STYLE_ATTR_RE = re.compile(r'(\s+)style\s*=\s*"([^"]*)"')
SCALE_MIN, SCALE_MAX = 0.3, 3.0


def _clean_scale(k) -> str:
    """A --fs-scale value is one bounded number. Nothing else is accepted --
    this string is written straight into a style attribute."""
    v = float(k)
    if not (SCALE_MIN <= v <= SCALE_MAX):
        raise ValueError(f'--fs-scale {v} outside [{SCALE_MIN}, {SCALE_MAX}]')
    return f'{round(v, 4):g}'


def set_scale(tag_text: str, k) -> str:
    """Return `tag_text` with --fs-scale set to k, or removed when k is None.

    Removing restores the tag byte for byte, including dropping a style
    attribute that we were the only reason for -- so resetting a size to 1x
    leaves no trace in the file.
    """
    m = _STYLE_ATTR_RE.search(tag_text)
    if k is None and not (m and _SCALE_RE.search(m.group(2))):
        return tag_text                       # nothing of ours to take out
    # Whatever else the style attribute says is left as its author wrote it:
    # ours is appended as a suffix and removed as a suffix, so setting a size
    # and resetting it gives the original bytes back.
    rest = _SCALE_DECL_RE.sub('', m.group(2)).strip() if m else ''
    if k is None:
        repl = f'{m.group(1)}style="{rest}"' if rest else ''
        return tag_text[:m.start()] + repl + tag_text[m.end():]

    val = f'--fs-scale:{_clean_scale(k)}'
    style = f'{rest}; {val}' if rest else val
    if m:
        return tag_text[:m.start()] + f'{m.group(1)}style="{style}"' + tag_text[m.end():]
    close = tag_text.rstrip()
    end = len(tag_text) - (len(tag_text) - len(close))
    return tag_text[:end - 1] + f' style="{style}"' + tag_text[end - 1:]


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
    def __init__(self, keep_styles=frozenset(), allow_bare_span=False):
        super().__init__(convert_charrefs=False)
        self.out = []
        self.open: list[str | None] = []
        self.keep_styles = keep_styles
        self.allow_bare_span = allow_bare_span

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
        if tag == 'span' and not keep and not self.allow_bare_span:
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


BARE_SPAN_RE = re.compile(r'<span\s*>')


def sanitize(fragment: str, keep_styles=frozenset(), allow_bare_span=False) -> str:
    p = _Sanitizer(keep_styles, allow_bare_span)
    p.feed(fragment)
    p.close()
    return p.result()
