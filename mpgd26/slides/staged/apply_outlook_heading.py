#!/usr/bin/env python3
"""Stage the Summary slide's outlook figure into the deck.  NOT YET APPLIED.

    cd mpgd26/slides && ../../.venv/bin/python staged/apply_outlook_heading.py

Two things, and they belong together -- the deck should never carry the heading
without the figure it heads, or the figure without the version of it we agreed:

  1. copy the current render into assets/img/x17_outlook.png.  The deck asset is
     DELIBERATELY STALE right now: the figure went through four rounds of edits
     on 2026-08-24 while the slide was left alone on Dylan's instruction
     ("Don't edit the slides for now, just point me to the image"), so the
     copy in assets/ is the first version, not the current one.
  2. put a heading above the figure -- Dylan: "put in HTML text above this
     diagram in large text something like 'Next Steps for Analysis:' to make
     clear this hasn't been done but will be done".

WHY THE HEADING IS MARKUP AND NOT PART OF THE FIGURE.  The figure is shared with
report.html and with anything else that wants it; the claim "this has not been
done yet" is a claim about where the talk stands on the day, and it belongs to
the slide.  Keeping it in HTML also means it is live text -- editable in the
browser editor, and it scales with the deck's own type scale rather than being
baked into a PNG at whatever size the canvas happened to be.

It is styled from the deck's own tokens rather than a new class: --fs-title-sm
is the slide-subtitle size, and taking it at 0.86 puts the heading clearly above
body text and clearly below the slide title, which is the hierarchy it wants.
--accent-ink is the deck's orange darkened for text on white, i.e. the one
colour in the palette reserved for "look here".

Idempotent: run it twice and the second run reports nothing to do.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.dirname(HERE)
MPGD26 = os.path.dirname(SLIDES)

SRC = os.path.join(MPGD26, 'figures', 'x17_outlook_light.png')
DST = os.path.join(SLIDES, 'assets', 'img', 'x17_outlook.png')
INDEX = os.path.join(SLIDES, 'index.html')

ANCHOR = '''        <div class="figure" style="flex:1;min-height:0;margin-top:.3em">
          <div class="imgwrap bare"><img src="assets/img/x17_outlook.png"'''

HEADING = '''        <div class="outlook-head" style="font-family:var(--display);\
font-size:calc(calc(var(--fs-title-sm)*.86) * var(--fs-scale, 1));\
font-weight:700;color:var(--accent-ink);letter-spacing:-.005em;\
margin:.55em 0 -.05em 0">Next steps for the analysis</div>
'''


def main():
    changed = []

    if not os.path.exists(SRC):
        sys.exit(f'missing {SRC} -- run:  '
                 f'../.venv/bin/python make_x17.py --layout outlook')
    same = (os.path.exists(DST)
            and open(SRC, 'rb').read() == open(DST, 'rb').read())
    if same:
        print('  asset already current')
    else:
        shutil.copyfile(SRC, DST)
        changed.append('assets/img/x17_outlook.png')
        print(f'  wrote {DST}')

    html = open(INDEX, encoding='utf-8').read()
    if 'class="outlook-head"' in html:
        print('  heading already in place')
    elif ANCHOR not in html:
        sys.exit('could not find the outlook figure block in index.html -- '
                 'the Summary slide has been edited; place the heading by hand')
    else:
        open(INDEX, 'w', encoding='utf-8').write(
            html.replace(ANCHOR, HEADING + ANCHOR, 1))
        changed.append('index.html')
        print('  added the "Next steps for the analysis" heading')

    if changed:
        print('\nchanged: ' + ', '.join(changed))
        print('now re-publish and rebuild the PDF:\n'
              '  cd .. && ../.venv/bin/python tools/mirror_slides_to_site.py '
              '/tmp/mpgd26-slides.html\n'
              '  python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py '
              '/tmp/mpgd26-slides.html --slug mpgd26-talk-slides --force '
              '--deploy\n'
              '  slides/make_pdf.sh')
    else:
        print('\nnothing to do')


if __name__ == '__main__':
    main()
