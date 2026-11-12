#!/usr/bin/env python3
"""build_lab.py -- the event display lab (interactive HTML playground for the Paper I displays).

    .venv/bin/python papers/paper1_v2/display_lab/extract_events.py p1     # events.json from the local pass-1 copies
    .venv/bin/python papers/paper1_v2/display_lab/build_lab.py [events.json] [out.html]

Embeds events.json into template.html.  Publish: dylan-cern-site/scripts/add-note.py out.html --slug event-display-lab.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'events.json')
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'event-display-lab.html')
t = open(os.path.join(HERE, 'template.html')).read()
assert '/*__DATA__*/null' in t
open(dst, 'w').write(t.replace('/*__DATA__*/null', open(src).read().replace('</', '<\\/')))
print('wrote', dst, os.path.getsize(dst) // 1024, 'kB')
