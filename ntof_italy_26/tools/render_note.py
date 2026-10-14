#!/usr/bin/env python3
"""
render_note.py -- render one slide (or one figure of a slide) of a published
slidedoc note (dylan-cern-site notes/*.html) to a PNG for the PowerPoint deck.

    python render_note.py NOTE.html SECTION_ID OUT.png [--svg ARIA_LABEL [--nth N]] [--scale 2]
                          [--bg ffffff] [--crop X,Y,W,H]

Without --svg the whole 1920x1080 section is rendered (--crop keeps a box of it,
in slide CSS px); with --svg only the SVG whose aria-label matches (the --nth
one) is rendered. --bg replaces the slide's background colour. Uses headless
Chrome (or Edge), so the hover tips are simply absent.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

BROWSERS = [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']


def balanced(s: str, start: int, tag: str) -> str:
    """Return the element starting at s[start] ('<tag ...'), with nested same tags."""
    depth, i = 0, start
    pat = re.compile(rf'<{tag}\b|</{tag}>')
    for m in pat.finditer(s, start):
        depth += 1 if m.group(0) != f'</{tag}>' else -1
        if depth == 0:
            return s[start:m.end()]
    raise ValueError(f'unbalanced <{tag}>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('note', type=Path)
    ap.add_argument('section')
    ap.add_argument('out', type=Path)
    ap.add_argument('--svg', default=None, help='aria-label of the one SVG to render')
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--pad', type=int, default=24)
    ap.add_argument('--nth', type=int, default=0, help='which of several SVGs with the same label')
    ap.add_argument('--bg', default=None, help='background colour override, hex without #')
    ap.add_argument('--crop', default=None, help='X,Y,W,H box of the section, CSS px')
    a = ap.parse_args()

    s = a.note.read_text(encoding='utf8')
    head = re.search(r'<head>(.*?)</head>', s, re.S).group(1)
    head = re.sub(r'<script.*?</script>', '', head, flags=re.S)
    i = s.index(f'<section id="{a.section}"')
    sec = balanced(s, i, 'section')
    style = re.search(r'<section[^>]*style="([^"]*)"', sec).group(1)
    if a.bg:
        new = re.sub(r'background:[^;]+', f'background:#{a.bg}', style)
        sec, style = sec.replace(style, new, 1), new
    if a.svg:
        j = [m.start() for m in re.finditer(re.escape(f'aria-label="{a.svg}"'), sec)][a.nth]
        j = sec.rindex('<svg', 0, j)
        el = balanced(sec, j, 'svg')
        w, h = (int(float(v)) for v in re.match(r'<svg width="([\d.]+)" height="([\d.]+)"', el).groups())
        bg = re.search(r'background:([^;]+)', style).group(1)
        font = re.search(r'font-family:([^;]+)', style).group(1)
        W, H = w + 2 * a.pad, h + 2 * a.pad
        body = (f'<div style="background:{bg};font-family:{font};width:{W}px;height:{H}px;padding:{a.pad}px;'
                f'box-sizing:border-box">{el}</div>')
    else:
        W, H = 1920, 1080
        body = f'<div style="position:relative;width:1920px;height:1080px;overflow:hidden">{sec}</div>'
    page = (f'<!doctype html><html><head>{head}<style>body{{margin:0;background:transparent}}'
            f'section{{width:1920px;height:1080px;box-sizing:border-box}} section aside{{display:none}}</style>'
            f'</head><body>{body}</body></html>')
    with tempfile.TemporaryDirectory() as td:
        html = Path(td) / 'page.html'
        html.write_text(page, encoding='utf8')
        exe = next(b for b in BROWSERS if Path(b).exists())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([exe, '--headless=new', '--disable-gpu', '--hide-scrollbars',
                        f'--force-device-scale-factor={a.scale}', f'--window-size={W},{H}',
                        '--virtual-time-budget=4000', f'--screenshot={a.out.resolve()}',
                        html.resolve().as_uri()], check=True, capture_output=True, timeout=120)
    if a.crop:
        from PIL import Image
        x, y, w, h = (int(v) for v in a.crop.split(','))
        k = a.scale
        Image.open(a.out).crop((int(x * k), int(y * k), int((x + w) * k), int((y + h) * k))).save(a.out)
    print('wrote', a.out, f'({W}x{H} css px, x{a.scale})')


if __name__ == '__main__':
    main()
