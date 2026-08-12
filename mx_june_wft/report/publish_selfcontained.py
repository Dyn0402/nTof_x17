#!/usr/bin/env python3
"""
publish_selfcontained.py — turn the fleet report into one self-contained HTML
file for the CERN site's notes system (which copies a single file and serves
it offline: no external images allowed).

Every figures/*.png reference is inlined as a data: URI, downscaled to
≤1250 px width; large rasters go to JPEG on a white background (the site is
offline-first PWA — total size matters). The report itself is unchanged.

    ../../.venv/bin/python mx_june_wft/report/publish_selfcontained.py \
        [--report .../fleet_report/report.html] [--out NOTE.html]
"""
import argparse
import base64
import io
import os
import re

from PIL import Image

DEFAULT = '/home/dylan/x17/cosmic_bench/Analysis/fleet_report/report.html'


def data_uri(path, max_w=1100, jpeg_over=70_000):
    im = Image.open(path)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    if buf.tell() > jpeg_over:
        rgb = Image.new('RGB', im.size, 'white')
        rgb.paste(im, mask=im.split()[3] if im.mode == 'RGBA' else None)
        buf = io.BytesIO()
        rgb.save(buf, format='JPEG', quality=76, optimize=True)
        mime = 'image/jpeg'
    else:
        mime = 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default=DEFAULT)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    base = os.path.dirname(args.report)
    out = args.out or os.path.join(base, 'note_selfcontained.html')

    doc = open(args.report).read()
    uris, missing = {}, []

    def repl_img(m):
        rel = m.group(1)
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            missing.append(rel)
            return m.group(0)
        if rel not in uris:
            uris[rel] = data_uri(p)
        return f'src="{uris[rel]}"'

    doc = re.sub(r'src="(figures/[^"]+)"', repl_img, doc)
    # the <a href="figures/..."> full-size links point nowhere in a
    # self-contained file — let the anchor open the embedded image instead
    doc = re.sub(r'<a href="figures/[^"]+" target="_blank" rel="noopener">'
                 r'(<img [^>]*>)</a>', r'\1', doc)
    doc = doc.replace(
        '</title>',
        '</title>\n<meta name="description" content="Full characterization '
        'of the five June cosmic-bench micro-TPC chambers on the frozen '
        'waveform-first reconstruction: fleet efficiency and resolution, '
        '2 mm-criterion maps, HV scans, per-detector QA, campaign '
        'logistics.">')

    with open(out, 'w') as f:
        f.write(doc)
    print(f'wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, '
          f'{len(uris)} images inlined, {len(missing)} missing)')
    if missing:
        print('MISSING:', missing)


if __name__ == '__main__':
    main()
