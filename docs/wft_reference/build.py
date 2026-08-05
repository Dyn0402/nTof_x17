#!/usr/bin/env python3
"""
Assemble the WFT reference document.

Concatenates `sections/*.html` in filename order, inlines every `{{FIG:name}}`
token as a base64 PNG from the figure directory, numbers the figures, builds
the table of contents from the headings, and appends the page script.

    ../../.venv/bin/python build.py [--figs DIR] [--out page.html]

The result is a single self-contained fragment suitable for the Artifact tool
(no <html>/<head>/<body> wrapper — those are added at publish time).
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIGS = ('/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17/'
                '6d4eafa1-3125-425d-94fe-b5fb7b7ea0b0/scratchpad/figs')

# A downloaded file has no HTTP headers, so it needs its own declarations. Without
# the charset, Chrome guesses per platform — desktop tends to land on UTF-8 while
# Android does not, which turns every Greek letter into mojibake. Without the
# viewport, Android renders at a 980 px virtual viewport and the mobile layout
# never engages.
STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>The waveform-first reconstruction — a complete reference</title>
</head>
<body>
"""

STANDALONE_TAIL = """
</body>
</html>
"""

SCRIPT = """
<script>
(function () {
  var main = document.querySelector('main');
  var toc  = document.getElementById('toc');
  var nodes = main.querySelectorAll('section.part, h2, h3');
  var entries = [];

  nodes.forEach(function (n) {
    var id, label, cls;
    if (n.classList && n.classList.contains('part')) {
      id = n.id;
      label = n.querySelector('.partno').textContent + ' — ' +
              n.querySelector('h2').textContent;
      cls = 'lvl-part';
    } else {
      if (n.closest('section.part')) return;
      id = n.id;
      label = n.textContent;
      cls = n.tagName === 'H2' ? 'lvl-2' : 'lvl-3';
    }
    if (!id) return;
    var li = document.createElement('li');
    var a = document.createElement('a');
    a.href = '#' + id;
    a.textContent = label;
    a.className = cls;
    li.appendChild(a);
    toc.appendChild(li);
    entries.push({ el: n, a: a, label: label });
  });

  /* scroll spy + progress bar */
  var bar = document.getElementById('progress');
  var now = document.getElementById('nowreading');
  var active = null;
  var ticking = false;

  function update() {
    ticking = false;
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';

    var best = null;
    for (var i = 0; i < entries.length; i++) {
      var top = entries[i].el.getBoundingClientRect().top;
      if (top < 140) best = entries[i]; else break;
    }
    if (best !== active) {
      if (active) active.a.classList.remove('active');
      active = best;
      if (active) {
        active.a.classList.add('active');
        if (now) now.textContent = active.label;
        var box = document.getElementById('tocwrap');
        if (window.innerWidth > 1000) {
          var r = active.a.getBoundingClientRect();
          var b = box.getBoundingClientRect();
          if (r.top < b.top + 40 || r.bottom > b.bottom - 40) {
            box.scrollTop += (r.top - b.top) - box.clientHeight * 0.4;
          }
        }
      }
    }
  }
  function onScroll() {
    if (!ticking) { ticking = true; requestAnimationFrame(update); }
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  update();

  /* mobile drawer */
  var btn = document.getElementById('tocbtn');
  var wrap = document.getElementById('tocwrap');
  var scrim = document.getElementById('scrim');
  function setOpen(v) {
    wrap.classList.toggle('open', v);
    scrim.classList.toggle('on', v);
    btn.setAttribute('aria-expanded', v ? 'true' : 'false');
  }
  btn.addEventListener('click', function () {
    setOpen(!wrap.classList.contains('open'));
  });
  scrim.addEventListener('click', function () { setOpen(false); });
  toc.addEventListener('click', function (e) {
    if (e.target.tagName === 'A' && window.innerWidth <= 1000) setOpen(false);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') setOpen(false);
  });
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', default=DEFAULT_FIGS)
    ap.add_argument('--out', default=os.path.join(HERE, 'page.html'))
    ap.add_argument('--parts', default=os.path.join(HERE, 'sections'))
    ap.add_argument('--standalone', default=None,
                    help='path for the downloadable copy (default: '
                         'wft_reference_standalone.html beside --out)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.parts, '*.html')))
    if not files:
        sys.exit(f'no sections in {args.parts}')
    body = '\n'.join(open(f).read() for f in files)

    # ---- inline figures ----------------------------------------------------
    used, missing = [], []

    def sub(m):
        name = m.group(1)
        path = os.path.join(args.figs, name + '.png')
        if not os.path.exists(path):
            missing.append(name)
            return (f'<div class="box warn"><span class="lbl">missing figure'
                    f'</span>{html.escape(name)}.png was not generated</div>')
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        used.append((name, len(b64)))
        return (f'<div class="figscroll"><img alt="{html.escape(name)}" '
                f'loading="lazy" src="data:image/png;base64,{b64}"></div>')

    body = re.sub(r'\{\{FIG:([A-Za-z0-9_]+)\}\}', sub, body)

    # ---- close the shell and append the script -----------------------------
    body += """
</main>

<div class="footer">
  <p>Generated from the live <code>sat_det3</code> calibration bundle,
  calibration cache and reconstruction. Figure sources:
  <code>docs/wft_reference/figsrc/</code>. Document source:
  <code>docs/wft_reference/sections/</code>, assembled by
  <code>build.py</code>.</p>
</div>

</div>
""" + SCRIPT

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(body)

    # the standalone copy — for downloading and opening as a local file, where
    # there are no HTTP headers to supply the charset
    alone = args.standalone or os.path.join(
        os.path.dirname(args.out), 'wft_reference_standalone.html')
    # the body-level <title> exists for the Artifact publisher, which wraps the
    # fragment in its own head; here it would be a second, invalid one
    body_alone = re.sub(r'^\s*<title>.*?</title>\s*', '', body,
                        count=1, flags=re.S)
    with open(alone, 'w', encoding='utf-8') as f:
        f.write(STANDALONE_HEAD + body_alone + STANDALONE_TAIL)

    total = os.path.getsize(args.out)
    print(f'{len(files)} sections, {len(used)} figures inlined')
    if missing:
        print('MISSING:', ', '.join(sorted(set(missing))))
    print(f'wrote {args.out}  {total/1e6:.2f} MB '
          f'({"OK" if total < 16e6 else "TOO BIG"} against the 16 MB limit)')
    print(f'wrote {alone}  {os.path.getsize(alone)/1e6:.2f} MB  '
          f'(standalone: doctype + charset + viewport)')

    # a rough check that no figure token slipped through
    left = re.findall(r'\{\{[^}]+\}\}', body)
    if left:
        print('UNRESOLVED TOKENS:', set(left))


if __name__ == '__main__':
    main()
