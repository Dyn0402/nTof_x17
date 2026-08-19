"""Self-contained mirror of mpgd26/slides/index.html for the CERN notes site.

Usage (refresh the published copy after editing the deck):
    ../../.venv/bin/python tools/mirror_slides_to_site.py /tmp/mpgd26-slides.html
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        /tmp/mpgd26-slides.html --slug mpgd26-talk-slides --force --deploy
(the note keeps its original date; --force replaces in place; the output is
wrapped as a complete <!doctype html> document, which add-note.py requires)

Every assets/ reference becomes a data URI. Raster figures are downscaled to
<= 1600 px wide and re-encoded WebP (the deck never displays wider); animated
GIFs and anything WebP refuses pass through untouched.
"""
import base64, io, os, re, sys
from PIL import Image

SRC = '/home/dylan/PycharmProjects/nTof_x17/mpgd26/slides/index.html'
OUT = sys.argv[1]
os.chdir(os.path.dirname(SRC))
html = open(SRC).read()

stats = {'in': 0, 'out': 0, 'n': 0}

def data_uri(path):
    raw = open(path, 'rb').read()
    stats['in'] += len(raw); stats['n'] += 1
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'gif':  # animated build-up — leave alone
        out, mime = raw, 'image/gif'
    else:
        im = Image.open(io.BytesIO(raw))
        if im.width > 1600:
            im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, 'WEBP', quality=82, method=4)
        out, mime = buf.getvalue(), 'image/webp'
        if len(out) >= len(raw):  # tiny PNGs can win — keep the smaller
            out, mime = raw, f'image/{"jpeg" if ext in ("jpg","jpeg") else ext}'
    stats['out'] += len(out)
    return f'data:{mime};base64,' + base64.b64encode(out).decode()

def sub(m):
    path = m.group(2)
    if not os.path.isfile(path):
        print('MISSING:', path, file=sys.stderr); return m.group(0)
    return m.group(1) + data_uri(path) + m.group(3)

html = re.sub(r'(src=")(assets/[^"]+)(")', sub, html)
left = [p for p in re.findall(r'assets/[A-Za-z0-9_/.-]+', html)]
open(OUT, 'w').write(html)
print(f"{stats['n']} images, {stats['in']/1e6:.1f} MB -> {stats['out']/1e6:.1f} MB; "
      f"html {os.path.getsize(OUT)/1e6:.1f} MB; unresolved refs: {len(left)}")
for p in left[:10]: print('  left:', p)

# Wrap as a complete document (the deck itself has no doctype/head/body; the
# site's add-note.py treats doctype-less files as front-matter fragments).
c = open(OUT).read()
i = c.index('</style>') + len('</style>')
head, body = c[:i], c[i:]
head = head.replace('<meta charset="utf-8">',
    '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">', 1)
open(OUT, 'w').write('<!doctype html>\n<html lang="en">\n<head>\n' + head
                     + '\n</head>\n<body>' + body + '\n</body>\n</html>\n')
