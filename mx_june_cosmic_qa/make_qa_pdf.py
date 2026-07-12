#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_qa_pdf.py

Bundle all QA plots for one detector + subrun into a single PDF (one figure per
page, grouped by QA stage with section dividers).

Usage:
    python make_qa_pdf.py <key>            # gather from that key's Analysis/ output dir
    python make_qa_pdf.py --dir <path>     # gather from an explicit directory

Writes <out_dir>/QA_<run>_<subrun>_<det>.pdf  (or QA_summary.pdf for --dir).
"""
import os, sys, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from qa_config import config_from_argv, setup_paths, get_config
setup_paths()

# Preferred stage order; anything else is appended alphabetically after these.
STAGE_ORDER = ['raw_detector_qa', 'm3_reference_qa', 'deep_qa',
               'alignment_tpc', 'alignment_tpc_veto50', 'efficiency',
               'correlation_outliers', 'cluster_quality', 'final_alignment']


def _stage_key(stage):
    return (STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER), stage)


def _text_page(pdf, title, subtitle='', fontsize=24):
    fig = plt.figure(figsize=(11, 8.5)); fig.text(0.5, 0.6, title, ha='center', va='center',
                                                  fontsize=fontsize, weight='bold', wrap=True)
    if subtitle:
        fig.text(0.5, 0.45, subtitle, ha='center', va='center', fontsize=13, wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _image_page(pdf, path, caption):
    try:
        img = plt.imread(path)
    except Exception as e:
        print(f'  skip {path}: {e}'); return
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.9]); ax.imshow(img); ax.axis('off')
    fig.text(0.5, 0.975, caption, ha='center', va='top', fontsize=9)
    pdf.savefig(fig, dpi=150); plt.close(fig)


def main():
    if '--dir' in sys.argv:
        out_base = os.path.abspath(sys.argv[sys.argv.index('--dir') + 1])
        label = os.path.basename(out_base.rstrip('/')); pdf_name = 'QA_summary.pdf'
    else:
        cfg = config_from_argv()
        out_base = cfg.OUT_BASE
        label = f'{cfg.RUN} / {cfg.SUB_RUN} / {cfg.DET_NAME}'
        pdf_name = f'QA_{cfg.RUN}_{cfg.SUB_RUN}_{cfg.DET_NAME}.pdf'

    if not os.path.isdir(out_base):
        sys.exit(f'No output dir: {out_base}\n(run the QA scripts for this key first)')

    # collect pngs grouped by immediate stage subdir
    pngs = [p for p in glob.glob(os.path.join(out_base, '**', '*.png'), recursive=True)
            if os.sep + 'cache' + os.sep not in p]
    if not pngs:
        sys.exit(f'No PNGs found under {out_base}')
    by_stage = {}
    for p in pngs:
        rel = os.path.relpath(p, out_base)
        stage = rel.split(os.sep)[0] if os.sep in rel else '(top level)'
        by_stage.setdefault(stage, []).append(p)

    pdf_path = os.path.join(out_base, pdf_name)
    n = 0
    with PdfPages(pdf_path) as pdf:
        _text_page(pdf, 'Cosmic-bench QA', label)
        for stage in sorted(by_stage, key=_stage_key):
            files = sorted(by_stage[stage])
            _text_page(pdf, stage, f'{len(files)} plots', fontsize=20)
            for f in files:
                _image_page(pdf, f, os.path.relpath(f, out_base)); n += 1
    print(f'Wrote {pdf_path}\n  {n} plots across {len(by_stage)} stages.')


if __name__ == '__main__':
    main()
