#!/usr/bin/env python3
"""note_outline.py -- keep a hub note's headings in step with a paper's outline (sections, subsections, figures, tables).

    .venv/bin/python papers/note_outline.py paper1                 # dry run: print the new outline and what moves
    .venv/bin/python papers/note_outline.py paper1 --write         # rewrite the note (then commit the notes repo)
    .venv/bin/python papers/note_outline.py paper1 --note <path.md>

The note (default: ~/.claude-hub/notes/dylan/archive/Paper I Feedback.md for paper1) gets one heading per item of the
paper, in paper order, numbered as LaTeX numbers them:

    ## 5 Charge balance and efficiency              (\\section; appendix sections are lettered, \\section* unnumbered)
    ### 5.3 Efficiency maps of the five chambers     (\\subsection)
    #### Fig. 9 · Detection-efficiency maps · `effmap`      (figure, keyed by its \\label without "fig:")
    #### Table 2 · Chamber A efficiency … · `single`        (table, keyed by its \\label without "tab:")

Everything written under a heading (to the next heading of any level) belongs to that heading and moves with it: when
a figure or section moves in the paper, re-run this and its notes follow it, renumbered. Headings are matched by key:
figures/tables by the `key` in backticks (stable when a figure moves or its caption changes), sections by their title
(a renamed section can't be matched: its notes go to a final "Unplaced" section, never deleted, and the run says so).
Text before the first ## heading (the title and the guide) is kept as it is. Two standing headings come first:
"Dylan’s Notes" (general notes, not tied to a section) and "Title" (the paper title, from \\title in main.tex).

Numbering follows the source: figures and tables count up in \\input order (article class, not reset per section),
\\appendix switches sections to A, B, … . A figure placed by LaTeX on a later page keeps its number, so this matches
the PDF; check with --aux <main.aux> to compare against a build.
"""
import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTES = Path(os.path.expanduser('~/.claude-hub/notes/dylan/archive'))
DEFAULT_NOTE = {'paper1': NOTES / 'Paper I Feedback.md', 'paper2': NOTES / 'Paper II Feedback.md'}
UNPLACED = 'Unplaced (headings no longer in the paper)'


# ------------------------------------------------------------------------------------------------ LaTeX side
def braced(s, i):
    """The {...} group starting at s[i] == '{': (content, index after the closing brace)."""
    assert s[i] == '{', s[i:i + 20]
    d, k = 0, i
    while True:
        c = s[k]
        if c == '\\':
            k += 2
            continue
        d += (c == '{') - (c == '}')
        k += 1
        if d == 0:
            return s[i + 1:k - 1], k


def detex(s):
    s = re.sub(r'%.*', '', s)
    s = s.replace('~', ' ').replace('``', '"').replace("''", '"').replace('\\,', ' ').replace('\\_', '_')
    s = s.replace('\\textdegree{}', '°').replace('\\textmu ', 'µ').replace('\\textmu{}', 'µ').replace('--', '–')
    for g, u in (('delta', 'δ'), ('theta', 'θ'), ('sigma', 'σ'), ('tau', 'τ'), ('varepsilon', 'ε'), ('epsilon', 'ε'),
                 ('chi', 'χ'), ('mu', 'µ'), ('times', '×'), ('pm', '±'), ('approx', '≈'), ('le', '≤'), ('ge', '≥')):
        s = re.sub(r'\\' + g + r'(?![a-zA-Z])\s*', u, s)
    s = re.sub(r'\\(?:emph|textit|textbf|texttt|mathrm|text)\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\(?:cref|Cref|ref)\{[^}]*\}', '[ref]', s)
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)
    s = s.replace('{', '').replace('}', '').replace('$', '')
    return re.sub(r'\s+', ' ', s).strip()


def short_caption(cap, n=64):
    """First clause of a caption, for a heading (the key carries the identity, this only helps reading)."""
    t = detex(cap)
    t = re.split(r'(?<=[a-z0-9\)])[.:;](?:\s|$)', t, maxsplit=1)[0]
    return t if len(t) <= n else t[:n].rsplit(' ', 1)[0] + ' …'


TOKEN = re.compile(r'\\((?:sub)*section\*?(?=\{)|appendix\b|begin\{(figure|table)\*?\}|lossfig(?=\{)|input(?=\{))')


def outline(paper):
    """[(level, number, title, kind, key)] in document order; kind in section/subsection/figure/table."""
    root = HERE / paper
    items, st = [], dict(sec=0, sub=0, fig=0, tab=0, app=False)

    def walk(text):
        text = re.sub(r'(?<!\\)%.*', '', text)
        text = re.sub(r'^\\(?:re)?newcommand.*$', '', text, flags=re.M)   # e.g. A1's \lossfig definition
        for m in TOKEN.finditer(text):
            tok = m.group(1)
            if tok == 'input':
                name, _ = braced(text, m.end())
                walk((root / (name if name.endswith('.tex') else name + '.tex')).read_text())
            elif tok == 'appendix':
                st['app'], st['sec'] = True, 0
            elif tok.startswith('section'):
                title, _ = braced(text, m.end())
                if tok.endswith('*'):
                    items.append((2, '', detex(title), 'section', None))
                    continue
                st['sec'] += 1; st['sub'] = 0
                num = chr(64 + st['sec']) if st['app'] else str(st['sec'])
                items.append((2, num, detex(title), 'section', None))
            elif tok.startswith('subsection'):
                title, _ = braced(text, m.end())
                st['sub'] += 1
                num = (chr(64 + st['sec']) if st['app'] else str(st['sec'])) + f'.{st["sub"]}'
                items.append((3, num, detex(title), 'subsection', None))
            elif tok == 'lossfig':                         # \lossfig{<id>}{<caption>} (A1): label fig:<id>
                fid, k = braced(text, m.end())
                cap, _ = braced(text, k)
                st['fig'] += 1
                items.append((4, str(st['fig']), short_caption(cap), 'figure', fid))
            elif m.group(2):                               # figure / table environment
                kind = m.group(2)
                end = text.find('\\end{' + kind, m.end())
                body = text[m.end():end]
                lab = re.search(r'\\label\{(?:fig|tab):([^}]*)\}', body)
                img = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}#]+)\}', body)
                c = body.find('\\caption{')
                cap = braced(body, c + 8)[0] if c >= 0 else ''
                cap = re.sub(r'\\askDylan\{.*?\}|\\askStephan\{.*?\}', '', cap)
                # figures: the file id (as in figures.toml, figure_log.toml and the deck); tables: the label
                key = (img.group(1) if (kind == 'figure' and img) else lab.group(1) if lab else None) or ( f'{kind}{st["fig" if kind == "figure" else "tab"] + 1}')
                st['fig' if kind == 'figure' else 'tab'] += 1
                items.append((4, str(st['fig' if kind == 'figure' else 'tab']), short_caption(cap),
                              kind, key))
    main = (root / 'main.tex').read_text()
    # standing headings at the top of the note, not parts of the body: general notes, then the title
    items.append((2, '', 'Dylan’s Notes', 'section', None))
    if '\\title{' in main:
        items.append((2, '', 'Title', 'section', None))
    if '\\begin{abstract}' in main:
        items.append((2, '', 'Abstract', 'section', None))
    walk(main)
    return items


def heading(it):
    level, num, title, kind, key = it
    if kind == 'figure':
        return f'{"#" * level} Fig. {num} · {title} · `{key}`'
    if kind == 'table':
        return f'{"#" * level} Table {num} · {title} · `{key}`'
    return f'{"#" * level} {num + " " if num else ""}{title}'


# ------------------------------------------------------------------------------------------------ note side
HEAD = re.compile(r'^(#{2,4})\s+(.*?)\s*$')


def norm_title(t):
    t = t.replace('\\', '').replace('`', '')
    t = re.sub(r'^(?:Appendix\s+)?(?:[A-Z]|\d+)(?:\.\d+)*\s+', '', t)    # leading number / letter
    return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()


def note_key(text):
    t = text.replace('\\_', '_')
    m = re.search(r'`([^`]+)`\s*$', t)
    if m and re.match(r'(Fig\.|Table)\s', t):
        return ('obj', m.group(1))
    return ('sec', norm_title(t))


def item_key(it):
    _, _, title, kind, key = it
    return ('obj', key) if kind in ('figure', 'table') else ('sec', norm_title(title))


def parse_note(path):
    """(preamble lines, [(heading text, key, body lines)])."""
    lines = path.read_text().split('\n') if path.exists() else []
    pre, blocks, cur = [], [], None
    for ln in lines:
        m = HEAD.match(ln)
        if m:
            cur = [m.group(2), note_key(m.group(2)), []]
            blocks.append(cur)
        elif cur is None:
            pre.append(ln)
        else:
            cur[2].append(ln)
    return pre, blocks


def is_empty(body):
    return all(not l.strip() or l.strip() == '&nbsp;' for l in body)


def trim(body):
    b = list(body)
    while b and (not b[-1].strip() or b[-1].strip() == '&nbsp;'):
        b.pop()
    while b and not b[0].strip():
        b.pop(0)
    return b


GUIDE = """How this note works (generated by `papers/note_outline.py`; edit freely):

- Headings follow the paper: sections, subsections, figures and tables, numbered as in the PDF. When the paper is
  reorganised, the headings are regenerated and everything under a heading moves with it.
- `- [ ]` items are for Claude to do. When one is done, Claude checks it off (`- [x]`) and adds a 💬 line under it
  with the details. Uncheck an item to reopen it (add a line saying what's still wrong).
- Plain text under a section or subsection heading is a draft: write it however it comes (stream of consciousness,
  fragments, numbers, "say X before Y"). Ask Claude to translate a section: it becomes paper text in the .tex, goes to
  Overleaf, and a 💬 line under your draft says where it went.
"""


def build(items, pre, blocks):
    by_key = {}
    for h, k, body in blocks:
        if is_empty(body):
            continue
        if k in by_key:                                  # same heading twice: keep both bodies
            by_key[k] += [''] + trim(body)
        else:
            by_key[k] = trim(body)
    used, out = set(), []
    pre = trim(pre) if pre else []
    if not pre:
        pre = ['# Paper I Feedback']
    if not any('note_outline.py' in l for l in pre):
        pre = pre + [''] + GUIDE.rstrip('\n').split('\n')
    out += pre
    for it in items:
        k = item_key(it)
        out += ['', heading(it)]
        if k in by_key:
            out += [''] + by_key[k]
            used.add(k)
    orphans = [(h, k, trim(b)) for h, k, b in blocks if not is_empty(b) and k not in used and k[1] != norm_title(UNPLACED)]
    if orphans:
        out += ['', f'## {UNPLACED}']
        seen = set()
        for h, k, b in orphans:
            if k in seen:
                continue
            seen.add(k)
            out += ['', f'#### {h}', ''] + b
    return '\n'.join(out).rstrip('\n') + '\n', orphans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paper')
    ap.add_argument('--note')
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--aux', help='main.aux of a build: check figure/table numbers against it')
    a = ap.parse_args()
    items = outline(a.paper)
    if a.aux:
        aux = Path(a.aux).read_text()
        bad = 0
        for lvl, num, title, kind, key in items:
            if kind in ('figure', 'table'):
                lab = ('fig:' if kind == 'figure' else 'tab:') + key
                m = re.search(r'\\newlabel\{' + re.escape(lab) + r'\}\{\{([^}]*)\}', aux)
                if not m and kind == 'figure':
                    continue                                   # keyed by file id; label differs
                if m and m.group(1) != num:
                    print(f'aux mismatch: {lab} source {num} pdf {m.group(1)}'); bad += 1
        print('aux check:', 'ok' if not bad else f'{bad} mismatches')
    note = Path(a.note) if a.note else DEFAULT_NOTE[a.paper]
    pre, blocks = parse_note(note)
    text, orphans = build(items, pre, blocks)
    nfig = sum(1 for i in items if i[3] == 'figure')
    print(f'{note}: {len([i for i in items if i[3] == "section"])} sections, '
          f'{len([i for i in items if i[3] == "subsection"])} subsections, {nfig} figures, '
          f'{len([i for i in items if i[3] == "table"])} tables; {len(blocks)} existing headings')
    for h, k, b in orphans:
        print(f'  UNPLACED (no longer in the paper): {h!r} ({len(b)} lines kept)')
    if a.write:
        old = note.read_text() if note.exists() else ''
        if old == text:
            print('unchanged')
        else:
            tmp = note.with_suffix('.md.tmp')
            tmp.write_text(text)
            os.replace(tmp, note)
            print('written')
    else:
        print('--- dry run (use --write) ---')
        print(text)


if __name__ == '__main__':
    sys.exit(main())
