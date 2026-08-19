#!/usr/bin/env python3
"""Tests for the in-browser deck editor. No browser needed for 1-6.

    ../../.venv/bin/python edit/test_edit.py

The one that matters is IDENTITY: re-saving every editable field with its own
unchanged content must reproduce index.html byte for byte. If that holds, a
real save can only touch the fields you actually typed in.
"""
from __future__ import annotations

import difflib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_source import DeckSource, sanitize, restore_entities, entity_preferences

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

check('82 slide sections found', len(doc.slides) == src.count('<section class="slide'))
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

titles = [doc.content(i) for i, e in enumerate(doc.editables)
          if 'title' in e.attrs.get('class', '')]
check('slide titles are fields', len(titles) >= 70, f'{len(titles)} title fields')

# ------------------------------------------------------------ 2. identity
head('2. identity round-trip (the safety property)')
rebuilt = doc.apply({i: doc.content(i) for i in doc.by_eid})
check('re-saving every field unchanged is byte-identical', rebuilt == src)
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
check('nothing else changed',
      len(inj) - len(src) == sum(len(f' data-eid="{i}"') for i in doc.by_eid))
inj_doc = DeckSource(inj)
check('injected page re-parses to the same field count',
      len(inj_doc.editables) == len(doc.editables),
      f'{len(inj_doc.editables)} vs {len(doc.editables)}')

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
    payload = json.dumps({'hash': status['hash'], 'edits': {
        str(e1): 'Edited over HTTP · <b>bold</b>',
        str(e2): 'a bullet with 10 mm and a — dash'}}).encode()

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
    check('preview diff is minimal', ndiff == 4, f'{ndiff} changed lines')

    code, out = post('/save')
    check('save applies', code == 200 and out.get('applied') == 2)
    saved = scratch.read_text()
    check('saved text is in the file', 'Edited over HTTP · <b>bold</b>' in saved)
    check('nbsp written as entity', '10&nbsp;mm' in saved)
    check('saved file still parses', len(DeckSource(saved).editables) == len(doc.editables))
    check('file changed by exactly the two lines',
          len([1 for a, b in zip(plain.splitlines(), saved.splitlines()) if a != b]) == 2)

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
(HERE / 'chrome_probe.html').unlink(missing_ok=True)

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
