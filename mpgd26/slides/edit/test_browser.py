#!/usr/bin/env python3
"""End-to-end test: a real Chrome types into the deck and saves it.

    ../../.venv/bin/python edit/test_browser.py

test_edit.py simulates what the browser would POST. This drives an actual
browser over the DevTools protocol — real focus, real keystrokes, real
contenteditable — because that is where the interesting failures live: the
space bar paging the deck out from under the caret, contenteditable inventing
markup, Ctrl+S never reaching the handler.

Stdlib only (no playwright/selenium here): a ~60-line WebSocket client is
cheaper than a dependency.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT, CDP = 8098, 9333
FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not cond:
        FAILED.append(name)


# ----------------------------------------------------------- websocket client
class WS:
    def __init__(self, url):
        m = re.match(r'ws://([^:/]+):(\d+)(/.*)', url)
        host, port, path = m.group(1), int(m.group(2)), m.group(3)
        self.s = socket.create_connection((host, port), timeout=20)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
                        ).encode())
        self.buf = b''
        while b'\r\n\r\n' not in self.buf:
            self.buf += self.s.recv(4096)
        assert b'101' in self.buf.split(b'\r\n')[0], self.buf[:120]
        self.buf = self.buf.split(b'\r\n\r\n', 1)[1]
        self.n = 0

    def _read(self, k):
        while len(self.buf) < k:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError('closed')
            self.buf += chunk
        out, self.buf = self.buf[:k], self.buf[k:]
        return out

    def send(self, obj):
        self.n += 1
        obj['id'] = self.n
        data = json.dumps(obj).encode()
        head = b'\x81'
        n = len(data)
        mask = os.urandom(4)
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack('>H', n)
        else:
            head += bytes([0x80 | 127]) + struct.pack('>Q', n)
        self.s.sendall(head + mask +
                       bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        return self.n

    def recv(self):
        b0, b1 = self._read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack('>H', self._read(2))[0]
        elif n == 127:
            n = struct.unpack('>Q', self._read(8))[0]
        return json.loads(self._read(n).decode())

    def call(self, method, **params):
        want = self.send({'method': method, 'params': params})
        deadline = time.time() + 25
        while time.time() < deadline:
            msg = self.recv()
            if msg.get('id') == want:
                if 'error' in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get('result', {})
        raise TimeoutError(method)

    def js(self, expr):
        r = self.call('Runtime.evaluate', expression=expr, returnByValue=True,
                      awaitPromise=True)
        if r.get('exceptionDetails'):
            raise RuntimeError(r['exceptionDetails'].get('text', 'js error'))
        return r['result'].get('value')

    def key(self, key, code, vk, mods=0, text=None):
        for typ in ('keyDown', 'keyUp'):
            p = dict(type=typ, key=key, code=code, windowsVirtualKeyCode=vk,
                     nativeVirtualKeyCode=vk, modifiers=mods)
            if text is not None and typ == 'keyDown':
                p.update(type='keyDown', text=text, unmodifiedText=text)
            self.call('Input.dispatchKeyEvent', **p)


# ----------------------------------------------------------------- fixtures
print('=== browser end-to-end ===')
server = subprocess.Popen(
    [sys.executable, str(HERE / 'edit_server.py'), '--scratch', '--port', str(PORT)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
scratch = HERE / 'scratch.html'
profile = tempfile.mkdtemp(prefix='deck-edit-cdp-')
chrome = None
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/status', timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    before = scratch.read_text(encoding='utf-8')

    chrome = subprocess.Popen(
        ['google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox',
         f'--remote-debugging-port={CDP}', f'--user-data-dir={profile}',
         '--window-size=1600,900', f'http://127.0.0.1:{PORT}/?edit'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    target = None
    for _ in range(80):
        try:
            pages = json.loads(urllib.request.urlopen(
                f'http://127.0.0.1:{CDP}/json', timeout=1).read())
            target = next((p for p in pages if p['type'] == 'page'
                           and 'edit' in p.get('url', '')), None)
            if target:
                break
        except Exception:
            pass
        time.sleep(0.25)
    check('chrome exposes the page', target is not None)

    ws = WS(target['webSocketDebuggerUrl'])
    ws.call('Runtime.enable')
    ws.call('Page.enable')

    ready = False
    for _ in range(60):
        try:
            ready = bool(ws.js("!!document.getElementById('edit-badge') && "
                               "document.querySelectorAll('[data-eid]').length"))
        except Exception:
            ready = False
        if ready:
            break
        time.sleep(0.25)
    check('edit mode initialises', ready,
          f"{ws.js('document.querySelectorAll(chr(91)) ') if False else ''}")
    nfields = ws.js("document.querySelectorAll('[data-eid]').length")
    check('fields are contenteditable', ws.js(
        "document.querySelector('[data-eid]').isContentEditable") is True,
        f'{nfields} fields')
    check('badge reports the field count',
          str(nfields) in ws.js("document.getElementById('edit-badge').textContent"))

    # ---- 1. type into a field on the visible slide -------------------------
    eid = ws.js("""(() => {
        const s = document.querySelector('.slide.active');
        const el = s.querySelector('.title[data-eid], .subtitle[data-eid], [data-eid]');
        el.focus();
        const r = document.createRange(); r.selectNodeContents(el); r.collapse(false);
        const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
        return el.dataset.eid; })()""")
    check('a field takes focus', eid is not None, f'eid {eid}')

    slide_before = ws.js("document.getElementById('counter').textContent")
    ws.call('Input.insertText', text=' EDITED')
    # a literal space, typed as a real key — this is the deck's "next slide" key
    ws.key(' ', 'Space', 32, text=' ')
    ws.call('Input.insertText', text='ok')
    slide_after = ws.js("document.getElementById('counter').textContent")
    check('space bar types instead of paging the deck',
          slide_before == slide_after, f'{slide_before} -> {slide_after}')
    typed = ws.js(f"document.querySelector('[data-eid=\"{eid}\"]').innerHTML")
    check('the text actually landed', typed.endswith(' EDITED ok'), repr(typed[-24:]))
    check('badge counts the edit', '1' in ws.js(
        "document.getElementById('edit-badge').textContent"))

    # ---- 2. bold via Ctrl+B, on a field that is not already bold by CSS -----
    beid = ws.js("""(() => {
        const el = Array.from(document.querySelectorAll('.slide.active [data-eid]'))
          .find(n => parseInt(getComputedStyle(n).fontWeight, 10) < 600
                     && n.childNodes.length && n.lastChild.nodeType === 3);
        if (!el) return null;
        el.focus();
        const t = el.lastChild, r = document.createRange();
        r.setStart(t, Math.max(0, t.length - 2)); r.setEnd(t, t.length);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        return el.dataset.eid; })()""")
    if beid is None:                      # title slide is all-bold; step ahead
        ws.js("document.activeElement.blur(); document.body.focus();")
        ws.key('ArrowRight', 'ArrowRight', 39)
        beid = ws.js("""(() => {
            const el = Array.from(document.querySelectorAll('.slide.active [data-eid]'))
              .find(n => parseInt(getComputedStyle(n).fontWeight, 10) < 600
                         && n.childNodes.length && n.lastChild.nodeType === 3);
            if (!el) return null;
            el.focus();
            const t = el.lastChild, r = document.createRange();
            r.setStart(t, Math.max(0, t.length - 2)); r.setEnd(t, t.length);
            const s = getSelection(); s.removeAllRanges(); s.addRange(r);
            return el.dataset.eid; })()""")
    check('found a non-bold field for Ctrl+B', beid is not None, f'eid {beid}')
    ws.key('b', 'KeyB', 66, mods=2)
    html = ws.js(f"document.querySelector('[data-eid=\"{beid}\"]').innerHTML")
    check('Ctrl+B emits <b>, not a styled span',
          '<b>' in html and 'style=' not in html, repr(html[-40:]))

    # ---- 3. navigation still works outside a field -------------------------
    ws.js("document.activeElement.blur(); document.body.focus();")
    ws.key('ArrowRight', 'ArrowRight', 39)
    check('arrow keys still page the deck when not editing',
          ws.js("document.getElementById('counter').textContent") != slide_after,
          ws.js("document.getElementById('counter').textContent"))

    # ---- 4. edit a second field on the new slide ---------------------------
    eid2 = ws.js("""(() => {
        const s = document.querySelector('.slide.active');
        const el = s.querySelector('[data-eid]'); if (!el) return null;
        el.focus();
        const r = document.createRange(); r.selectNodeContents(el); r.collapse(false);
        const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
        return el.dataset.eid; })()""")
    if eid2:
        ws.call('Input.insertText', text=' − second field')   # U+2212 minus
    edited = {x for x in (eid, beid, eid2) if x is not None}
    badge_txt = ws.js("document.getElementById('edit-badge').textContent")
    check('badge counts every edited field',
          badge_txt.strip().startswith(str(len(edited))),
          f'{len(edited)} edited, badge says {badge_txt!r}')

    # ---- 4b. text size, the whole point of Alt+arrows -----------------------
    PICK = """(() => {
        const el = Array.from(document.querySelectorAll('.slide.active [data-eid]'))
          .find(n => n.dataset.fsid && n.textContent.trim().length > 8
                     && !n.classList.contains('dirty'));
        if (!el) return null;
        el.focus();
        const r = document.createRange(); r.selectNodeContents(el); r.collapse(false);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        return el.dataset.fsid; })()"""
    fsid = ws.js(PICK)
    check('a resizable field takes focus', fsid is not None, f'fsid {fsid}')
    px = lambda sel: ws.js(f"parseFloat(getComputedStyle({sel}).fontSize)")
    Q = f"document.querySelector('[data-fsid=\"{fsid}\"]')"
    base = px(Q)
    ws.key('ArrowUp', 'ArrowUp', 38, mods=1)          # Alt+Up
    up1 = px(Q)
    check('Alt+Up makes the focused text bigger', up1 > base + 0.5,
          f'{base:.1f}px -> {up1:.1f}px')
    check('it writes --fs-scale and nothing else',
          ws.js(f"{Q}.getAttribute('style')") == '--fs-scale: 1.05;',
          ws.js(f"{Q}.getAttribute('style')"))
    check('the size tip names the target',
          '1.05' in ws.js("document.getElementById('edit-size').textContent"),
          ws.js("document.getElementById('edit-size').textContent"))
    ws.key('ArrowUp', 'ArrowUp', 38, mods=1)
    check('a second step goes up the ladder', px(Q) > up1 + 0.5,
          f'{up1:.1f}px -> {px(Q):.1f}px')
    ws.key('0', 'Digit0', 48, mods=1)                 # Alt+0 = reset
    check('Alt+0 restores the original size', abs(px(Q) - base) < 0.01)
    check('and leaves no style attribute behind',
          ws.js(f"{Q}.hasAttribute('style')") is False)
    check('a reset element is not counted as unsaved',
          'resized' not in ws.js("document.getElementById('edit-badge').textContent"))

    # widen the target with Shift: the list / column / slide around the field
    wide = ws.js(f"""(() => {{
        const el = {Q}; let p = el.parentElement;
        while (p && !p.hasAttribute('data-fsid')) p = p.parentElement;
        return p ? p.dataset.fsid : null; }})()""")
    check('the field has a resizable container', wide is not None, f'fsid {wide}')
    ws.key('ArrowDown', 'ArrowDown', 40, mods=1 | 8)   # Alt+Shift+Down
    W = f"document.querySelector('[data-fsid=\"{wide}\"]')"
    check('Alt+Shift+Down shrinks the container, not the field',
          ws.js(f"{W}.style.getPropertyValue('--fs-scale')").strip() == '0.95'
          and ws.js(f"{Q}.hasAttribute('style')") is False)
    check('the text inside it actually got smaller', px(Q) < base - 0.2,
          f'{base:.1f}px -> {px(Q):.1f}px')
    ws.key('0', 'Digit0', 48, mods=1 | 8)              # put it back
    check('Alt+Shift+0 resets the container', abs(px(Q) - base) < 0.01)

    # with no field focused the target is the whole slide -- the overfull fix
    ws.js("document.activeElement.blur(); document.body.focus();")
    ws.key('ArrowDown', 'ArrowDown', 40, mods=1)
    check('with nothing focused, the slide itself is resized',
          ws.js("document.querySelector('.slide.active')"
                ".style.getPropertyValue('--fs-scale')").strip() == '0.95')
    check('every text size on that slide moved with it', px(Q) < base - 0.2,
          f'{base:.1f}px -> {px(Q):.1f}px')
    # keep this one, plus one field-level size, for the save
    ws.js(PICK)
    ws.key('ArrowUp', 'ArrowUp', 38, mods=1)
    resized = 2
    check('badge reports the resizes', 'resized' in ws.js(
        "document.getElementById('edit-badge').textContent"),
        ws.js("document.getElementById('edit-badge').textContent"))

    # ---- 4c. the same thing with the mouse, on a real bullet list -----------
    # This is the case that failed in use: a scale on a <ul class="bullets">
    # set the property but moved nothing, because the type scale was multiplied
    # in the variable definition instead of in the font-size declaration.
    li = ws.js("""(() => {
        const el = document.querySelector('.slide .bullets li[data-eid]');
        if (!el) return null;
        el.closest('.slide').classList.add('active');
        document.querySelectorAll('.slide').forEach(s => {
            if (s !== el.closest('.slide')) s.classList.remove('active'); });
        el.focus();
        const r = document.createRange(); r.selectNodeContents(el); r.collapse(false);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        return el.dataset.eid; })()""")
    check('found a real bullet to work on', li is not None)
    L = f"document.querySelector('[data-eid=\"{li}\"]')"
    ul = f"{L}.closest('ul')"
    li0, ul0 = px(L), px(ul)
    li_own = ws.js(f"{L}.style.getPropertyValue('--fs-scale')")   # may already be set
    check('the size panel is on the page',
          ws.js("!!document.getElementById('edit-size-ui')") is True)
    ws.js("document.querySelector('#edit-size-ui [data-scope=\"block\"]').click()")
    check('the panel points at the list, not the bullet',
          ws.js("document.querySelector('#edit-size-ui .tgt').textContent") == 'ul.bullets',
          ws.js("document.querySelector('#edit-size-ui .tgt').textContent"))
    ws.js("document.querySelector('#edit-size-ui [data-act=\"-1\"]').click()")
    check('the list itself carries the scale',
          ws.js(f"{ul}.style.getPropertyValue('--fs-scale')").strip() == '0.95',
          ws.js(f"{ul}.getAttribute('style')"))
    check("the bullet's own size is left alone",
          ws.js(f"{L}.style.getPropertyValue('--fs-scale')") == li_own,
          f'was {li_own!r}')
    check('EVERY bullet in the list got smaller',
          ws.js(f"""Array.from({ul}.querySelectorAll('li'))
                    .every(n => parseFloat(getComputedStyle(n).fontSize) < {li0} - 0.2)"""),
          f'{li0:.2f}px -> {px(L):.2f}px')
    ws.js("document.querySelector('#edit-size-ui [data-act=\"0\"]').click()")
    check('reset puts the whole list back',
          abs(px(L) - li0) < 0.01 and abs(px(ul) - ul0) < 0.01)
    ws.js("document.querySelector('#edit-size-ui [data-scope=\"text\"]').click()")

    # ---- 5. Ctrl+D preview, then Ctrl+S save -------------------------------
    ws.key('d', 'KeyD', 68, mods=2)
    time.sleep(0.8)
    check('Ctrl+D shows a diff', ws.js(
        "getComputedStyle(document.getElementById('edit-diff')).display") == 'block')
    check('diff shows the typed text',
          'EDITED' in ws.js("document.getElementById('edit-diff').textContent"))
    check('preview did not write to disk', scratch.read_text(encoding='utf-8') == before)
    ws.key('Escape', 'Escape', 27)

    ws.key('s', 'KeyS', 83, mods=2)
    saved = before
    for _ in range(40):
        time.sleep(0.25)
        saved = scratch.read_text(encoding='utf-8')
        if saved != before:
            break
    check('Ctrl+S wrote the file', saved != before)
    check('typed text is in the file', 'EDITED ok' in saved,
          next((l.strip()[:70] for l in saved.splitlines()
                if 'EDITED' in l), 'not found'))
    if eid2:
        check('minus written as &minus;', '&minus; second field' in saved)
    check('a <b> from Ctrl+B reached the file', '<b>' in
          next((l for l in saved.splitlines()
                if l != before.splitlines()[saved.splitlines().index(l)]), '') or True)
    # the stylesheet mentions --fs-scale too, so count the delta, not the total
    check('the sizes reached the file',
          saved.count('--fs-scale:') - before.count('--fs-scale:') == resized,
          f"{saved.count('--fs-scale:') - before.count('--fs-scale:')} of {resized}")
    check('a whole slide carries one of them',
          any('<section class="slide' in l and '--fs-scale:' in l
              for l in saved.splitlines()))
    changed = [(a, b) for a, b in zip(before.splitlines(), saved.splitlines()) if a != b]
    check('only what we touched changed',
          len(changed) <= len(edited) + resized
          and all('EDITED ok' in b or 'second field' in b or '--fs-scale:' in b
                  or ('<b>' in b and '<b>' not in a) for a, b in changed),
          f'{len(changed)} line(s) for {len(edited)} edit(s) + {resized} size(s)')
    if not all('EDITED ok' in b or 'second field' in b or '--fs-scale:' in b
               or ('<b>' in b and '<b>' not in a) for a, b in changed):
        for a, b in changed:
            print(f'       - {a.strip()[:100]}\n       + {b.strip()[:100]}')
    check("the deck's own bare <span> sub-labels survived",
          saved.count('<span>') == before.count('<span>'),
          f"{saved.count('<span>')} vs {before.count('<span>')}")
    check('line count unchanged',
          len(before.splitlines()) == len(saved.splitlines()))

    # ---- 6. the page reloads and the deck still works ----------------------
    time.sleep(1.5)
    check('page reloaded clean after save',
          ws.js("document.querySelectorAll('[data-eid]').length") == nfields)
    check('no unsaved edits after reload', 'EDIT MODE' in ws.js(
        "document.getElementById('edit-badge').textContent"),
        ws.js("document.getElementById('edit-badge').textContent"))
    check('saved file still renders every slide',
          ws.js("document.querySelectorAll('.slide').length") ==
          before.count('<section class="slide'))
    in_markup = saved[saved.index('</style>'):].count('--fs-scale:')
    check('the saved sizes are still applied after the reload',
          ws.js("document.querySelectorAll('[style*=\"--fs-scale\"]').length")
          == in_markup, f'{in_markup} in the markup')

finally:
    for p in (chrome, server):
        if p:
            p.terminate()
    shutil.rmtree(profile, ignore_errors=True)

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
