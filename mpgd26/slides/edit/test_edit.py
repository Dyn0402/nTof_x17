#!/usr/bin/env python3
"""Tests for the in-browser deck editor. No browser needed for 1-6.

    ../../.venv/bin/python edit/test_edit.py

The one that matters is IDENTITY: re-saving every editable field with its own
unchanged content must reproduce index.html byte for byte. If that holds, a
real save can only touch the fields you actually typed in.
"""
from __future__ import annotations

import difflib
import html
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_source import (DeckSource, sanitize, restore_entities,
                         entity_preferences, set_scale, INLINE)

HERE = Path(__file__).resolve().parent
DECK = HERE.parent / 'index.html'

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not cond:
        FAILED.append(name)


def head(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1. parse
head('1. parse')
src = DECK.read_text(encoding='utf-8')
doc = DeckSource(src)
check('file parses', len(doc.elements) > 0, f'{len(doc.elements)} elements in the deck')
check('editable fields found', 400 < len(doc.editables) < 2000,
      f'{len(doc.editables)} fields')

kinds = {}
for e in doc.editables:
    key = f"{e.tag}.{e.attrs.get('class', '').split(' ')[0] or '-'}"
    kinds[key] = kinds.get(key, 0) + 1
print('     ' + ', '.join(f'{k}:{v}' for k, v in
                          sorted(kinds.items(), key=lambda x: -x[1])[:14]))

# spans must be well formed and non-overlapping (nesting is excluded by the
# all-children-inline rule, so a flat sort is enough)
ok = True
prev_end = -1
for e in doc.editables:
    if not (0 <= e.content_start < e.content_end <= len(src)) or e.content_start < prev_end:
        ok = False
        break
    prev_end = e.content_end
check('spans ordered and disjoint', ok)

check('slide sections all found', len(doc.slides) == src.count('<section class="slide'),
      f'{len(doc.slides)} slides')
check('resizables are a superset of the fields',
      set(id(e) for e in doc.editables) <= set(id(e) for e in doc.resizables),
      f'{len(doc.resizables)} resizable elements')
check('every slide is resizable as a whole',
      all(any(r is sl for r in doc.resizables) for sl in doc.slides))
check('no inline element is resizable',
      not [e for e in doc.resizables if e.tag in INLINE],
      'their start tags would sit inside a content span')
ov = [(a, b) for a in doc.editables for b in doc.resizables
      if a.content_start < b.content_start and b.tag_start < a.content_end]
check('no resizable start tag sits inside a field', not ov,
      'the two kinds of span must never overlap')
runtime = [e for e in doc.editables
           if e.attrs.get('id') in ('counter', 'progress')
           or 'counter' in e.attrs.get('class', '')
           or 'nav-btn' in e.attrs.get('class', '')]
check('runtime chrome is NOT editable', not runtime,
      'the deck script rewrites those at load; a save would persist the live '
      'slide number')
check('every field lives inside a slide section',
      all(any(s.content_start <= e.tag_start < s.content_end for s in doc.slides)
          for e in doc.editables))

# Where the factor has to live: a var() inside a custom property is substituted
# where the property is DECLARED, so multiplying the --fs-* definitions does
# nothing. Every font-size that reads the type scale has to carry it instead --
# and no definition should, or the two would compound.
css = src[src.index('<style>'):src.index('</style>')]
decls = re.findall(r'font-size:\s*((?:[^;{}]|\([^()]*(?:\([^()]*\))?[^()]*\))*)', css)
missing = [d for d in decls if 'var(--fs-' in d and 'var(--fs-scale' not in d]
check('every font-size built on the type scale carries the factor', not missing,
      '; '.join(d.strip()[:44] for d in missing[:3]))
defs = [m for m in re.findall(r'--fs-(?!scale)[a-z-]+:[^;]+;', css)
        if 'var(--fs-scale' in m]
check('no --fs-* definition carries it (it would compound)', not defs,
      '; '.join(d.strip()[:44] for d in defs[:3]))

titles = [doc.content(i) for i, e in enumerate(doc.editables)
          if 'title' in e.attrs.get('class', '')]
check('slide titles are fields', len(titles) >= 70, f'{len(titles)} title fields')

# ------------------------------------------------------------ 2. identity
head('2. identity round-trip (the safety property)')
rebuilt = doc.apply({i: doc.content(i) for i in doc.by_eid})
check('re-saving every field unchanged is byte-identical', rebuilt == src)
was = {i: doc.scale(i) for i in doc.by_fsid}         # what the deck already has
check('re-writing every size it already has is byte-identical',
      doc.apply(scales=was) == src, f'{sum(v is not None for v in was.values())} set')
check('clearing a size that was never set is byte-identical',
      doc.apply(scales={i: None for i in doc.by_fsid if was[i] is None}) == src)
check('text and sizes together are byte-identical',
      doc.apply({i: doc.content(i) for i in doc.by_eid}, was) == src)
scaled = doc.apply(scales={i: 0.9 for i in doc.by_fsid})
sdoc = DeckSource(scaled)
check('a deck with every element scaled still parses',
      len(sdoc.resizables) == len(doc.resizables)
      and scaled.count('--fs-scale:0.9') == len(doc.resizables))
check('putting every size back where it was restores the original bytes',
      sdoc.apply(scales=was) == src)
if rebuilt != src:
    d = list(difflib.unified_diff(src.splitlines(), rebuilt.splitlines(),
                                  'disk', 'rebuilt', lineterm='', n=0))
    print('\n'.join(d[:40]))
    print(f'     ({len(d)} diff lines)')

# ---------------------------------------------------- 3. id injection
head('3. data-eid injection')
inj = doc.inject_ids()
check('every field got an id', inj.count('data-eid="') == len(doc.editables))
check('slides all survive', inj.count('<section class="slide') == src.count('<section class="slide'))
check('every resizable got an id', inj.count('data-fsid="') == len(doc.resizables))
check('nothing else changed',
      len(inj) - len(src) == sum(len(f' data-eid="{i}"') for i in doc.by_eid)
                           + sum(len(f' data-fsid="{i}"') for i in doc.by_fsid))
check('the slide literal survives injection',
      inj.count('<section class="slide') == src.count('<section class="slide'),
      'make_pdf.sh splits on it')
inj_doc = DeckSource(inj)
check('injected page re-parses to the same field count',
      len(inj_doc.editables) == len(doc.editables),
      f'{len(inj_doc.editables)} vs {len(doc.editables)}')
check('injected page re-parses to the same resizable count',
      len(inj_doc.resizables) == len(doc.resizables))

# -------------------------------------------------- 4. sanitiser / entities
head('4. sanitiser and entity restoration')
prefs = doc.entity_map
check('minus restored (never literal in this file)', prefs.get('−') == '&minus;')
check('epsilon restored (never literal in this file)', prefs.get('ε') == '&epsilon;')
check('em dash left alone (the file writes 145 of them literally)', '—' not in prefs)
check('middot left alone (5 literal in the file)', '·' not in prefs)
check('nbsp always restored', prefs.get(' ') == '&nbsp;')
check('⁸ stays literal (source writes it raw)', '⁸' not in prefs)
check('< is never re-escaped', '<' not in prefs and '&' not in prefs)

cases = [
    # what Chrome hands back                        what should land in the file
    ('plain text', 'plain text'),
    ('a <b>bold</b> word', 'a <b>bold</b> word'),
    ('<strong>x</strong> <em>y</em>', '<b>x</b> <i>y</i>'),
    ('A · B — C', 'A · B — C'),
    ('at \u221240 and 3 \u03b5', 'at &minus;40 and 3 &epsilon;'),
    ('10 mm', '10&nbsp;mm'),
    ('<span style="font-weight:700">s</span>', 's'),   # bare span = residue
    ('<span class="chip">keep</span>', '<span class="chip">keep</span>'),
    ('<div>line2</div>', 'line2'),
    ('line1<div>line2</div>', 'line1<br>line2'),
    ('drop <script>alert(1)</script>', 'drop alert(1)'),
    ('<b>unclosed', '<b>unclosed</b>'),
    ('5 &lt; 6 &amp; 7', '5 &lt; 6 &amp; 7'),
    ('⁸Be stays raw', '⁸Be stays raw'),
    ('<img src=x onerror=1>', ''),
    ('keep<br>break', 'keep<br>break'),
]
for raw, want in cases:
    got = restore_entities(sanitize(raw), prefs)
    check(f'sanitise {raw[:34]!r}', got == want, '' if got == want else f'got {got!r}')

# -------------------------------------------------- 4b. --fs-scale on a tag
head('4b. --fs-scale on a start tag')
scases = [
    ('<div class="title">', 1.15, '<div class="title" style="--fs-scale:1.15">'),
    ('<section class="slide">', .92, '<section class="slide" style="--fs-scale:0.92">'),
    ('<div class="figure" style="flex:1;min-height:0">', .8,
     '<div class="figure" style="flex:1;min-height:0; --fs-scale:0.8">'),
]
for tag, k, want in scases:
    got = set_scale(tag, k)
    check(f'set_scale {tag[:32]!r}', got == want, '' if got == want else f'got {got!r}')
    check('   and back to the original bytes', set_scale(got, None) == tag)
check('an existing scale is replaced, not doubled',
      set_scale(set_scale('<div class="t">', 1.2), .8)
      == '<div class="t" style="--fs-scale:0.8">')
check('a tag with no style of its own is left with none',
      set_scale(set_scale('<div class="t">', 1.2), None) == '<div class="t">')
for bad in (0.1, 9, 'big', float('nan')):
    try:
        set_scale('<div>', bad)
        check(f'refuses {bad!r}', False)
    except (ValueError, TypeError):
        check(f'refuses {bad!r}', True)
check('nothing but a number reaches the style attribute',
      set_scale('<div>', 1.2) == '<div style="--fs-scale:1.2">')

# -------------------------------------------------- 5. a realistic edit
head('5. a realistic edit produces a one-line diff')
tgt = next(i for i, e in enumerate(doc.editables)
           if 'title' in e.attrs.get('class', '').split()
           and 'MPGD' not in doc.content(i))
before = doc.content(tgt)
after = doc.apply({tgt: 'A rewritten slide title — with <b>emphasis</b>, at \u221240'})
dl = [l for l in difflib.unified_diff(src.splitlines(), after.splitlines(),
                                      lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
check('exactly one line changed', len(dl) == 2, f'{len(dl)//2} line(s); was {before[:40]!r}')
check('entity restored in the written line', any('&minus;' in l for l in dl if l[0] == '+'))
check('rest of the file untouched',
      after[:doc.editables[tgt].content_start] == src[:doc.editables[tgt].content_start])
check('edited file still parses to the same field count',
      len(DeckSource(after).editables) == len(doc.editables))

# -------------------------------------------------- 6. server, over HTTP
head('6. server round-trip (real HTTP, scratch copy)')
scratch = HERE / 'scratch.html'
proc = subprocess.Popen(
    [sys.executable, str(HERE / 'edit_server.py'), '--scratch', '--port', '8099'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_up(url, tries=40):
    for _ in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=1).read()
        except Exception:
            time.sleep(0.15)
    return None


base = 'http://127.0.0.1:8099'
raw = wait_up(base + '/status')
check('server starts', raw is not None)
if raw:
    status = json.loads(raw)
    plain = urllib.request.urlopen(base + '/').read().decode()
    edit = urllib.request.urlopen(base + '/?edit').read().decode()
    check('plain GET is the untouched deck', 'data-eid' not in plain and plain == scratch.read_text())
    check('?edit injects the editor', 'edit-mode-js' in edit and 'data-eid="0"' in edit)
    check('an asset is served', len(urllib.request.urlopen(
        base + '/assets/img/microtpc.png').read()) > 1000)

    # edit two fields the way the browser would, and preview first
    d2 = DeckSource(scratch.read_text())
    e1 = next(i for i, e in enumerate(d2.editables) if 'title' in e.attrs.get('class', '').split())
    e2 = next(i for i, e in enumerate(d2.editables) if e.tag == 'li')
    z1 = next(i for i, e in enumerate(d2.resizables)
              if 'slide' in e.attrs.get('class', '').split())
    payload = json.dumps({'hash': status['hash'], 'edits': {
        str(e1): 'Edited over HTTP · <b>bold</b>',
        str(e2): 'a bullet with 10 mm and a — dash'},
        'sizes': {str(z1): '0.92'}}).encode()

    def post(path, body=payload):
        req = urllib.request.Request(base + path, data=body,
                                     headers={'Content-Type': 'application/json'})
        try:
            r = urllib.request.urlopen(req)
            return r.status, json.loads(r.read())
        except urllib.error.HTTPError as ex:
            return ex.code, json.loads(ex.read())

    code, out = post('/preview')
    check('preview returns a diff', code == 200 and '+' in out.get('diff', ''))
    check('preview did not write', scratch.read_text() == plain)
    ndiff = len([l for l in out['diff'].splitlines()
                 if l[:1] in '+-' and l[:3] not in ('+++', '---')])
    check('preview diff is minimal', ndiff == 6, f'{ndiff} changed lines')
    check('the size shows up in the preview', '--fs-scale:0.92' in out['diff'])

    code, out = post('/save')
    check('save applies', code == 200 and out.get('applied') == 3)
    saved = scratch.read_text()
    check('the size landed on a slide tag', '--fs-scale:0.92"' in saved)
    check('saved text is in the file', 'Edited over HTTP · <b>bold</b>' in saved)
    check('nbsp written as entity', '10&nbsp;mm' in saved)
    check('saved file still parses', len(DeckSource(saved).editables) == len(doc.editables))
    check('file changed by exactly the three lines',
          len([1 for a, b in zip(plain.splitlines(), saved.splitlines()) if a != b]) == 3)

    s2 = DeckSource(saved)
    z2 = next(i for i in s2.by_fsid if s2.scale(i) == '0.92')
    check('the scale reads back off the tag', s2.scale(z2) == '0.92')
    ln = saved[:s2.by_fsid[z2].tag_start].count(chr(10))
    check('a reset takes the whole style attribute back out',
          s2.apply(scales={z2: None}).splitlines()[ln] == plain.splitlines()[ln])

    code, out = post('/save')          # same (now stale) hash
    check('a stale page is refused', code == 409, out.get('error', '')[:40])

    check('backup written', any((HERE / '.backups').glob('scratch.*.html')))
    check('real index.html untouched', DECK.read_text() == src)

proc.terminate()
proc.wait(timeout=5)

# -------------------------------------------------- 7. browser render
head('7. headless chrome renders the editor')
chrome = subprocess.run(
    ['google-chrome', '--headless', '--disable-gpu', '--no-sandbox', '--dump-dom',
     '--virtual-time-budget=4000', f'file://{HERE / "chrome_probe.html"}'],
    capture_output=True, text=True, timeout=120) if (HERE / 'chrome_probe.html').exists() else None
if chrome is None:
    (HERE / 'chrome_probe.html').write_text(inj + '\n<!-- probe -->')
    chrome = subprocess.run(
        ['google-chrome', '--headless', '--disable-gpu', '--no-sandbox', '--dump-dom',
         '--virtual-time-budget=4000', f'file://{HERE / "chrome_probe.html"}'],
        capture_output=True, text=True, timeout=120)
dom = chrome.stdout
check('chrome parses the injected deck', '<section class="slide' in dom)
check('all slides survive the browser parse',
      dom.count('<section class="slide') == src.count('<section class="slide'),
      f"{dom.count('<section class=' + chr(34) + 'slide')} of {src.count('<section class=' + chr(34) + 'slide')}")
check('ids survive the browser parse',
      dom.count('data-eid=') == len(doc.editables),
      f'{dom.count("data-eid=")} of {len(doc.editables)}')
check('resize ids survive the browser parse',
      dom.count('data-fsid=') == len(doc.resizables),
      f'{dom.count("data-fsid=")} of {len(doc.resizables)}')
(HERE / 'chrome_probe.html').unlink(missing_ok=True)

# ------------------------------------------- 8. downstream: make_pdf.sh
head('8. the print pipeline still gets what it expects')
saved_deck = scratch.read_text(encoding='utf-8')      # has a slide-level size
sections = re.findall(r'<section class="slide.*?</section>', saved_deck, flags=re.S)
sdoc = DeckSource(saved_deck)
check('make_pdf.sh still splits every slide',
      len(sections) == len(sdoc.slides), f'{len(sections)} of {len(sdoc.slides)}')
check("make_pdf.sh's class regex still reads the slide's classes",
      all(re.match(r'<section class="([^"]*)"', sec) for sec in sections),
      'a size is appended after class=, never in front of it')

scaled = next(sec for sec in sections if '--fs-scale' in sec.split('>')[0])
page = saved_deck[:saved_deck.index('<div class="stage">')]


def measured(section_html):
    """Render one slide the way make_pdf.sh does and read its text sizes."""
    doc = (page + '<div class="stage"><div class="deck">\n' + section_html
           + '\n</div></div>\n<script>window.addEventListener("load",()=>{'
             'const o=[];document.querySelectorAll(".slide *").forEach(e=>'
             'o.push(parseFloat(getComputedStyle(e).fontSize)));'
             'const p=document.createElement("pre");p.id="m";'
             'p.textContent=JSON.stringify(o);document.body.appendChild(p);});</script>')
    probe = HERE / 'print_probe.html'
    probe.write_text(doc)
    out = subprocess.run(
        ['google-chrome', '--headless', '--disable-gpu', '--no-sandbox',
         '--window-size=1600,900', '--dump-dom', '--virtual-time-budget=4000',
         f'file://{probe}'], capture_output=True, text=True, timeout=120).stdout
    probe.unlink(missing_ok=True)
    m = re.search(r'<pre id="m">(.*?)</pre>', out, re.S)
    return json.loads(m.group(1)) if m else []


with_size = measured(scaled)
without = measured(re.sub(r'\s*style="--fs-scale:[^"]*"', '', scaled, count=1))
check('the slide renders on its own', len(with_size) > 5 and len(with_size) == len(without),
      f'{len(with_size)} elements')
ratios = [a / b for a, b in zip(with_size, without) if b > 0]
check('every text size on it scaled by 0.92',
      ratios and all(abs(r - 0.92) < 0.01 for r in ratios),
      f'{min(ratios):.3f}..{max(ratios):.3f}' if ratios else 'no text')

# ------------------------------------------- 9. every kind of text scales
# The check that was missing. --fs-scale first shipped multiplied into the
# --fs-* VARIABLE definitions, which does nothing: a var() inside a custom
# property is substituted where the property is declared (:root), not where it
# is used, so a scale set on a slide never reached it. Only elements that
# inherit their size moved, through the 1em fallback rule, and the title slide
# -- which declares its sizes directly -- so the first tests passed. This walks
# the whole deck instead: set a scale on one element of every kind and demand
# that it, and everything under it, actually moves.
head('9. --fs-scale moves every kind of text in the deck')
probe = HERE / 'scale_probe.html'
probe.write_text(src + """
<script>
window.addEventListener('load', () => {
  const sizes = el => Array.from(el.querySelectorAll('*')).concat([el])
      .map(n => parseFloat(getComputedStyle(n).fontSize));
  const out = [], seen = {};
  document.querySelectorAll('.slide').forEach(sl => {
    sl.querySelectorAll('*').forEach(el => {
      const cls = (el.getAttribute('class') || '').split(' ')[0]
                  || el.tagName.toLowerCase();
      if (seen[cls] > 1 || !el.textContent.trim()) return;
      seen[cls] = (seen[cls] || 0) + 1;
      const was = el.style.getPropertyValue('--fs-scale');
      const before = sizes(el);
      el.style.setProperty('--fs-scale', '0.5');
      const after = sizes(el);
      if (was) el.style.setProperty('--fs-scale', was);
      else el.style.removeProperty('--fs-scale');
      out.push([cls, before.length,
                before.filter((b, i) => Math.abs(after[i] - b) > 0.05).length]);
    });
  });
  const p = document.createElement('pre'); p.id = 'scale';
  p.textContent = JSON.stringify(out);
  document.documentElement.appendChild(p);
});
</script>""")
chrome = subprocess.run(
    ['google-chrome', '--headless', '--disable-gpu', '--no-sandbox',
     '--window-size=1600,900', '--dump-dom', '--virtual-time-budget=6000',
     f'file://{probe}'], capture_output=True, text=True, timeout=180).stdout
probe.unlink(missing_ok=True)
m = re.search(r'<pre id="scale">(.*?)</pre>', chrome, re.S)
rows = json.loads(html.unescape(m.group(1))) if m else []
check('the sweep ran', len(rows) > 30, f'{len(rows)} element kinds sampled')
dead = sorted({cls for cls, n, moved in rows if moved == 0})
partial = sorted({cls for cls, n, moved in rows if 0 < moved < n})
check('nothing ignores a scale set on it', not dead, ', '.join(dead[:8]))
check('nothing is left behind when its container is scaled', not partial,
      ', '.join(partial[:8]))

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
