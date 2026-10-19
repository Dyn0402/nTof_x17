#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_photos.py -- slide-sized copies of the photographs of the real station.

Everything else in this package draws its own pictures; these are the two that
cannot be drawn, so they get the one thing a drawing does not need: a home.  The
full-resolution originals live in ``photos/`` under the names the camera gave
them, and this script makes the copies the deck actually loads.  Downloads
folders get emptied.

Each entry is (original, slide name, what it shows).  The description is not
decoration: it is what goes in the slide's ``alt`` text and what tells the next
person which photograph is which without opening them.

    ../.venv/bin/python make_photos.py            # rebuild the slide copies
    ../.venv/bin/python make_photos.py --list     # just say what is here

Crops are deliberately NOT done here.  Both frames are portrait phone shots with
a third of the frame spare, and the right crop depends on the slide they end up
on -- which is still open (see the placeholder note in slides/index.html).  When
that is settled, add the box to the table rather than cropping the original.
"""
from __future__ import annotations

import argparse
import os

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'photos')
DST = os.path.join(HERE, 'slides', 'assets', 'img')

LONG_EDGE = 1500          # a slide is 1920 wide and these sit in half of it
QUALITY = 86

PHOTOS = [
    ('PXL_20260810_072347028.jpg', 'photo_station_topdown.jpg',
     'looking down into the assembled station: the four chambers in their '
     'pinwheel around the target, arms lettered A-D on the frame'),
    ('PXL_20260810_071943424.jpg', 'photo_arm_outside.jpg',
     'one arm from outside: CFRP liquid vessel, the plastics\' PMTs on their '
     'light guides above it, trigger-wall front-ends behind'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--list', action='store_true',
                    help='print the table and do nothing else')
    a = ap.parse_args()

    for src, dst, what in PHOTOS:
        print(f'{dst:32s} {what}')
        if a.list:
            continue
        p_in = os.path.join(SRC, src)
        if not os.path.isfile(p_in):
            raise SystemExit(f'missing original: {p_in}\n'
                             f'The originals are the point of photos/ -- if it '
                             f'is empty, recover them before regenerating.')
        # EXIF first: a phone stores portrait shots as landscape plus a rotation
        # tag, and thumbnail() would otherwise size the wrong edge.
        im = ImageOps.exif_transpose(Image.open(p_in))
        im.thumbnail((LONG_EDGE, LONG_EDGE), Image.LANCZOS)
        p_out = os.path.join(DST, dst)
        im.save(p_out, quality=QUALITY, optimize=True)
        print(f'  -> {p_out}  {im.size[0]}x{im.size[1]}  '
              f'{os.path.getsize(p_out) // 1024} kB')


if __name__ == '__main__':
    main()
