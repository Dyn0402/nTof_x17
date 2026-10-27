#!/usr/bin/env python3
"""render_check.py <note.html> <out_prefix> [n_slides] -- headless Chrome screenshot of a slide note, cut into one PNG
per slide (frames are 1400 px wide at 16:9 plus the Details bar), for checking every slide by eye before publishing."""
import subprocess
import sys

from PIL import Image

html, pre = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 14
H = 90 + n * 860
subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--hide-scrollbars', '--virtual-time-budget=5000',
                f'--window-size=1432,{H}', f'--screenshot={pre}_full.png', f'file://{html}'], capture_output=True)
im = Image.open(f'{pre}_full.png').convert('RGB')
# find frame tops: rows where the dark page background switches to the slide colour at x = 40
W, Hh = im.size
px = im.load()
tops, inside = [], False
for y in range(Hh):
    r, g, b = px[40, y]
    light = r > 200 or (r < 40 and g < 45 and b < 60 and False)
    is_slide = abs(r - 0xf5) < 6 and abs(g - 0xf3) < 6 or (abs(r - 0x15) < 4 and abs(g - 0x1a) < 4 and abs(b - 0x24) < 4)
    if is_slide and not inside:
        tops.append(y); inside = True
    elif not is_slide and inside:
        inside = False
for k, t in enumerate(tops):
    im.crop((0, t, W, min(t + 790, Hh))).save(f'{pre}_s{k + 1:02d}.png')
print(len(tops), 'slides', tops[:3])
