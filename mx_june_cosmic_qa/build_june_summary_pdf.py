#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_june_summary_pdf.py

Assemble the June cosmic-bench summary slide deck (MX17_June_summary.pdf) from the
curated PNGs under _june_summary/pdf_src/: a title page, then for each numbered
subdir a section-divider page followed by one page per PNG (its relative path as a
header). Reproduces the original generic folder->slides layout so the deck can be
regenerated after the curated figures change (e.g. the fleet overview page).

Usage: build_june_summary_pdf.py [--src DIR] [--out PDF] [--title STR]
"""
import os
import sys
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

SRC = '/home/dylan/x17/cosmic_bench/Analysis/_june_summary/pdf_src'
OUT = '/home/dylan/x17/cosmic_bench/Analysis/_june_summary/MX17_June_summary.pdf'
TITLE = 'Cosmic-bench QA'
FIGSIZE = (10, 7.5)   # landscape 4:3 slide


def _arg(flag, default):
    return next((a.split('=', 1)[1] for a in sys.argv if a.startswith(flag)), default)


def text_page(pdf, title, sub, tsize=22, ssize=11):
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.5, 0.55, title, ha='center', va='center', fontsize=tsize, fontweight='bold')
    if sub:
        fig.text(0.5, 0.42, sub, ha='center', va='center', fontsize=ssize)
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, path, header):
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.5, 0.97, header, ha='center', va='top', fontsize=8)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
    ax.axis('off')
    ax.imshow(mpimg.imread(path))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    src = _arg('--src=', SRC)
    out = _arg('--out=', OUT)
    title = _arg('--title=', TITLE)
    n = 0
    with PdfPages(out) as pdf:
        text_page(pdf, title, os.path.basename(src.rstrip('/')))
        n += 1
        for sub in sorted(os.listdir(src)):
            d = os.path.join(src, sub)
            if not os.path.isdir(d):
                continue
            pngs = sorted(glob.glob(os.path.join(d, '*.png')))
            if not pngs:
                continue
            text_page(pdf, sub, f'{len(pngs)} plots', tsize=18)
            n += 1
            for p in pngs:
                image_page(pdf, p, f'{sub}/{os.path.basename(p)}')
                n += 1
    print(f'Wrote {n} pages -> {out}')


if __name__ == '__main__':
    main()
