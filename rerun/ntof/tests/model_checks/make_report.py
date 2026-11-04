#!/usr/bin/env python3
"""make_report.py -- report.html for the three n_TOF model checks (template residuals, saturation clip, Dp).

    make_report.py [--data /media/ucla/work/ntof_pass1/model_checks] [--out <data>/report.html]

Reads only the JSON outputs of template_resid.py, template_toy.py, sat_census.py, sat_compare.py, dp_stack.py and
mb_9010.py under --data, draws the figures into <out dir>/figures/ and writes the report with relative links, so the
same file reads from disk or from the DAQ page's Analysis tab.  A copy goes next to this script
(rerun/ntof/tests/model_checks/report.html, figures/ alongside).  Verdict text is computed from the numbers.
"""
import argparse
import glob
import html
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/media/ucla/work/ntof_pass1/model_checks'
ARMS = ('A', 'C', 'D')
BENCH = dict(A='det3', C='det6', D='det7')


def J(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fig_template(data, figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = []
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, arm in zip(axs, ARMS):
        cos = f'{data}/template/' + ('ho_r149_D.json' if arm == 'D' else f'r149s1_{arm}.json')
        for lab, f, st in ((f'bench {BENCH[arm]}', f'{data}/template/bench_{arm}.json', 'k-'),
                           ('n_TOF cosmics', cos, 'C0-'),
                           ('n_TOF beam', f'{data}/template/r145s0_{arm}.json', 'C1-')):
            d = J(f)
            if not d or 'prod' not in d['planes']['x']:
                continue
            s = d['planes']['x']['prod']
            k = np.array(d['k_off']); m = np.array(s['res_med'])[2]; e = np.array(s['res_err'])[2]
            ax.errorbar(k, m, e, fmt=st, ms=3, lw=1, capsize=0, label=f'{lab} (n={s["n"]})')
        ax.axhline(0, color='0.7', lw=0.6)
        ax.set_title(f'{arm}: centre strip, X', fontsize=10)
        ax.set_xlabel('samples from the peak')
        ax.legend(fontsize=7, frameon=False)
    axs[0].set_ylabel('(data - model) / model peak')
    fig.tight_layout()
    p = 'figures/template_centre.png'
    fig.savefig(os.path.join(figs, os.path.basename(p)), dpi=130)
    plt.close(fig)
    out.append(p)
    return out


def table(rows, head):
    h = '<table><tr>' + ''.join(f'<th>{html.escape(x)}</th>' for x in head) + '</tr>'
    for r in rows:
        h += '<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>'
    return h + '</table>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, 'report.html')
    figs = os.path.join(os.path.dirname(out), 'figures')
    os.makedirs(figs, exist_ok=True)
    fl = fig_template(a.data, figs)
    import sys
    sys.path.insert(0, HERE)
    import summarise
    S = summarise.collect(a.data)
    summarise.conclude(S)
    fl += summarise.figures(S, figs)
    sec = summarise.html_sections(S)
    css = ('body{font-family:sans-serif;max-width:1100px;margin:2em auto;line-height:1.4;color:#222}'
           'table{border-collapse:collapse;margin:.6em 0;font-size:.9em}td,th{border:1px solid #ccc;padding:3px 7px;'
           'text-align:right}th{background:#f3f3f3}.v{background:#f6f8fb;border-left:4px solid #369;padding:.6em 1em}'
           'img{max-width:100%}figcaption{font-size:.85em;color:#555}')
    doc = [f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>n_TOF pass 1: model checks</title>'
           f'<style>{css}</style></head><body><h1>n_TOF pass 1: the three model checks before the bundles freeze</h1>',
           sec]
    doc.append('</body></html>')
    open(out, 'w').write('\n'.join(doc))
    # copy next to the code
    dst = os.path.join(HERE, 'report.html')
    if os.path.abspath(dst) != os.path.abspath(out):
        shutil.copy(out, dst)
        os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
        for f in glob.glob(os.path.join(figs, '*.png')):
            shutil.copy(f, os.path.join(HERE, 'figures', os.path.basename(f)))
    print('wrote', out)


if __name__ == '__main__':
    main()
