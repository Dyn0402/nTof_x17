#!/usr/bin/env python3
"""WYSIWYG text editing for slides/index.html, in the browser that renders it.

    ../../.venv/bin/python edit/edit_server.py            # edits index.html
    ../../.venv/bin/python edit/edit_server.py --scratch  # edits a throwaway copy

Open http://localhost:8017/ — that is the deck exactly as it is on disk, with
no edit machinery in it. http://localhost:8017/?edit turns every text element
into a live text field: click a title, a bullet, a caption or a table cell and
type. Ctrl+S writes back. Ctrl+D shows the diff a save would make, first.

The file on disk is never touched by a GET, only by an explicit save, and every
save leaves a timestamped copy in slides/edit/.backups/ first. Writes are
byte-exact everywhere except the spans you actually edited (see deck_source).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import http.server
import json
import mimetypes
import os
import shutil
import socketserver
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_source import DeckSource   # noqa: E402

HERE = Path(__file__).resolve().parent
SLIDES = HERE.parent
BACKUPS = HERE / '.backups'

EDIT_CSS = """
<style id="edit-mode-css">
  [data-eid]{ transition: background .12s, box-shadow .12s; }
  [data-eid]:hover{ box-shadow: inset 0 0 0 1px rgba(235,129,27,.45); border-radius:2px; }
  [data-eid]:focus{ outline:none; background:rgba(235,129,27,.07);
                    box-shadow: inset 0 0 0 1.5px var(--accent, #eb811b); border-radius:2px; }
  [data-eid].dirty{ background:rgba(235,129,27,.10); }
  #edit-badge{
    position:fixed; left:14px; bottom:12px; z-index:9999;
    font:600 12px/1.5 var(--sans, sans-serif); letter-spacing:.04em;
    background:#23373b; color:#fff; padding:7px 12px; border-radius:20px;
    box-shadow:0 2px 10px rgba(0,0,0,.25); user-select:none; cursor:default;
  }
  #edit-badge.dirty{ background:var(--accent, #eb811b); }
  #edit-badge.saved{ background:#2f7d55; }
  #edit-badge b{ font-weight:800; }
  #edit-diff{
    position:fixed; inset:6vh 6vw; z-index:10000; background:#12211f; color:#dfe7e6;
    font:12px/1.45 var(--mono, monospace); white-space:pre; overflow:auto;
    padding:18px 20px; border-radius:8px; box-shadow:0 10px 60px rgba(0,0,0,.5);
    display:none;
  }
  #edit-diff .add{ color:#7ee0a5; } #edit-diff .del{ color:#ff8f7a; }
  #edit-diff .hdr{ color:#7fb6c4; }
  @media print{ #edit-badge,#edit-diff{ display:none !important; } }
</style>
"""

EDIT_JS = r"""
<script id="edit-mode-js">
(function(){
  const HASH = "__HASH__";
  const orig = new Map();          // eid -> innerHTML as served
  const badge = document.createElement('div');
  badge.id = 'edit-badge';
  document.body.appendChild(badge);
  const diffbox = document.createElement('div');
  diffbox.id = 'edit-diff';
  document.body.appendChild(diffbox);

  document.execCommand('styleWithCSS', false, false);   // emit <b>, not <span style>

  const nodes = Array.from(document.querySelectorAll('[data-eid]'));
  const styles = new Map();        // eid -> the inline styles the deck itself uses
  nodes.forEach(el => {
    orig.set(el.dataset.eid, el.innerHTML);
    styles.set(el.dataset.eid, new Set(
      Array.from(el.querySelectorAll('[style]')).map(n => n.getAttribute('style'))));
    el.setAttribute('contenteditable', 'true');
    el.spellcheck = true;
  });

  // contenteditable answers Ctrl+B with <span style="font-weight:700"> and
  // leaves bare <span>s behind after a delete. The server strips those anyway;
  // doing it here too keeps the page, the diff preview and the file agreeing.
  function unwrap(n){ while (n.firstChild) n.parentNode.insertBefore(n.firstChild, n);
                      n.remove(); }
  function retag(n, tag){
    const t = document.createElement(tag);
    while (n.firstChild) t.appendChild(n.firstChild);
    n.replaceWith(t);
  }
  function normalize(el){
    const keep = styles.get(el.dataset.eid) || new Set();
    el.querySelectorAll('font').forEach(unwrap);
    Array.from(el.querySelectorAll('span[style]')).forEach(s => {
      const st = s.getAttribute('style') || '';
      if (keep.has(st)) return;
      if (/font-weight:\s*(bold|[6-9]00)/.test(st))      retag(s, 'b');
      else if (/font-style:\s*italic/.test(st))          retag(s, 'i');
      else                                               unwrap(s);
    });
    Array.from(el.querySelectorAll('span:not([class]):not([style])')).forEach(unwrap);
    el.normalize();
  }
  function normalizeAll(){ dirtyList().forEach(normalize); }

  function dirtyList(){
    return nodes.filter(el => el.innerHTML !== orig.get(el.dataset.eid));
  }
  function paint(){
    const d = dirtyList();
    nodes.forEach(el => el.classList.toggle(
      'dirty', el.innerHTML !== orig.get(el.dataset.eid)));
    badge.classList.toggle('dirty', d.length > 0);
    badge.classList.remove('saved');
    badge.innerHTML = d.length
      ? `<b>${d.length}</b> unsaved &middot; Ctrl+S save &middot; Ctrl+D diff`
      : `EDIT MODE &middot; ${nodes.length} fields &middot; Ctrl+S save`;
  }
  paint();

  document.addEventListener('input', e => {
    if (e.target.closest('[data-eid]')) paint();
  });
  document.addEventListener('blur', e => {
    const f = e.target.closest && e.target.closest('[data-eid]');
    if (f){ normalize(f); paint(); }      // caret-safe moment to tidy markup
  }, true);

  // The deck's own key handler lives on document and would page the slides
  // while you type (space = next slide). Shield it during editing.
  document.addEventListener('keydown', e => {
    const inField = e.target.closest && e.target.closest('[data-eid]');
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's'){
      e.preventDefault(); e.stopPropagation(); save(); return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd'){
      e.preventDefault(); e.stopPropagation(); preview(); return;
    }
    if (diffbox.style.display === 'block' && e.key === 'Escape'){
      diffbox.style.display = 'none'; e.stopPropagation(); return;
    }
    if (!inField) return;
    if (e.key === 'Escape'){ e.target.blur(); e.stopPropagation(); return; }
    if ((e.ctrlKey || e.metaKey) && 'bi'.includes(e.key.toLowerCase())){
      e.preventDefault(); e.stopPropagation();
      document.execCommand('styleWithCSS', false, false);
      document.execCommand(e.key.toLowerCase() === 'b' ? 'bold' : 'italic');
      normalize(inField); paint();
      return;
    }
    if (e.key === 'Enter'){
      e.preventDefault();
      document.execCommand('insertLineBreak');   // <br>, the deck's idiom
    }
    e.stopPropagation();          // never let a keystroke reach the navigator
  }, true);

  document.addEventListener('paste', e => {
    if (!e.target.closest('[data-eid]')) return;
    e.preventDefault();
    const t = (e.clipboardData || window.clipboardData).getData('text/plain');
    document.execCommand('insertText', false, t);
  }, true);

  // Clicking to place the caret must not also advance the slide.
  document.addEventListener('click', e => {
    if (e.target.closest('[data-eid]')) e.stopPropagation();
  }, true);

  window.addEventListener('beforeunload', e => {
    if (dirtyList().length){ e.preventDefault(); e.returnValue = ''; }
  });

  function payload(){
    normalizeAll();          // send exactly what the page is showing
    const edits = {};
    dirtyList().forEach(el => { edits[el.dataset.eid] = el.innerHTML; });
    return JSON.stringify({hash: HASH, edits: edits});
  }

  async function post(path){
    const r = await fetch(path, {method:'POST',
      headers:{'Content-Type':'application/json'}, body: payload()});
    return {ok: r.ok, status: r.status, data: await r.json()};
  }

  async function save(){
    const d = dirtyList();
    if (!d.length){ flash('nothing to save'); return; }
    const r = await post('/save');
    if (!r.ok){
      alert('Save refused (' + r.status + '): ' + (r.data.error || '') +
            '\nYour edits are still in the page — copy anything you need.');
      return;
    }
    badge.classList.remove('dirty'); badge.classList.add('saved');
    badge.innerHTML = `saved <b>${r.data.applied}</b> field(s) &rarr; reloading`;
    nodes.forEach(el => orig.set(el.dataset.eid, el.innerHTML));  // disarm unload
    setTimeout(() => location.reload(), 450);
  }

  async function preview(){
    const r = await post('/preview');
    const lines = (r.data.diff || '(no changes)').split('\n');
    diffbox.innerHTML = lines.map(l => {
      const cls = l.startsWith('+') ? 'add' : l.startsWith('-') ? 'del'
                : l.startsWith('@') ? 'hdr' : '';
      const esc = l.replace(/&/g,'&amp;').replace(/</g,'&lt;');
      return cls ? `<span class="${cls}">${esc}</span>` : esc;
    }).join('\n') + '\n\n— Esc to close, Ctrl+S to save —';
    diffbox.style.display = 'block';
  }

  function flash(msg){
    const was = badge.innerHTML; badge.innerHTML = msg;
    setTimeout(() => paint(), 900);
  }
})();
</script>
"""


class Deck:
    """The file under edit, plus its parse. Reloaded whenever it changes."""

    def __init__(self, path: Path):
        self.path = path
        self.load()

    def load(self):
        self.src = self.path.read_text(encoding='utf-8')
        self.hash = hashlib.sha1(self.src.encode()).hexdigest()[:16]
        self.doc = DeckSource(self.src)

    def reload_if_changed(self):
        cur = hashlib.sha1(self.path.read_bytes()).hexdigest()[:16]
        if cur != self.hash:
            self.load()

    def served_html(self) -> str:
        html = self.doc.inject_ids()
        js = EDIT_JS.replace('__HASH__', self.hash)
        return html + EDIT_CSS + js

    def build(self, edits: dict[str, str]) -> str:
        return self.doc.apply({int(k): v for k, v in edits.items()})

    def save(self, edits: dict[str, str]) -> int:
        new = self.build(edits)
        BACKUPS.mkdir(exist_ok=True)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        shutil.copy2(self.path, BACKUPS / f'{self.path.stem}.{stamp}.html')
        tmp = self.path.with_suffix('.html.tmp')
        tmp.write_text(new, encoding='utf-8')
        os.replace(tmp, self.path)
        self.load()
        return len(edits)


class Handler(http.server.BaseHTTPRequestHandler):
    deck: Deck = None
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        if '/save' in fmt % args or '/preview' in fmt % args:
            sys.stderr.write("  %s\n" % (fmt % args))

    # ---------------- GET ----------------
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            self.deck.reload_if_changed()
            edit = 'edit' in parse_qs(u.query, keep_blank_values=True)
            body = self.deck.served_html() if edit else self.deck.src
            self._send(body.encode('utf-8'), 'text/html; charset=utf-8')
            return
        if u.path == '/status':
            self._json(200, {'file': str(self.deck.path), 'hash': self.deck.hash,
                             'fields': len(self.deck.doc.editables)})
            return
        # static assets, restricted to the slides directory
        target = (SLIDES / u.path.lstrip('/')).resolve()
        if not str(target).startswith(str(SLIDES)) or not target.is_file():
            self._json(404, {'error': 'not found'})
            return
        ctype = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        self._send(target.read_bytes(), ctype)

    # ---------------- POST ----------------
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get('Content-Length', 0))
        try:
            req = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError as e:
            self._json(400, {'error': f'bad json: {e}'})
            return
        edits = req.get('edits') or {}

        if req.get('hash') != self.deck.hash:
            self._json(409, {'error': 'index.html changed on disk since this '
                                      'page was loaded — reload and redo the edit'})
            return
        try:
            if u.path == '/preview':
                new = self.deck.build(edits)
                diff = difflib.unified_diff(
                    self.deck.src.splitlines(), new.splitlines(),
                    'index.html (disk)', 'index.html (after save)',
                    lineterm='', n=1)
                self._json(200, {'diff': '\n'.join(diff)})
                return
            if u.path == '/save':
                applied = self.deck.save(edits)
                print(f"  saved {applied} field(s) -> {self.deck.path}")
                self._json(200, {'applied': applied, 'hash': self.deck.hash})
                return
        except Exception as e:                       # never half-write the deck
            self._json(500, {'error': f'{type(e).__name__}: {e}'})
            return
        self._json(404, {'error': 'no such endpoint'})

    # ---------------- plumbing ----------------
    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', default=str(SLIDES / 'index.html'))
    ap.add_argument('--port', type=int, default=8017)
    ap.add_argument('--scratch', action='store_true',
                    help='copy the deck to edit/scratch.html and edit that '
                         'instead — for trying the editor out')
    a = ap.parse_args()

    path = Path(a.file).resolve()
    if a.scratch:
        scratch = HERE / 'scratch.html'
        shutil.copy2(path, scratch)
        path = scratch

    Handler.deck = Deck(path)
    print(f"editing : {path}")
    print(f"fields  : {len(Handler.deck.doc.editables)} editable text elements")
    print(f"deck    : http://localhost:{a.port}/          (read-only, as on disk)")
    print(f"editor  : http://localhost:{a.port}/?edit     (click text, Ctrl+S)")
    with Server(('127.0.0.1', a.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nbye')


if __name__ == '__main__':
    main()
