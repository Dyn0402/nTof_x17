#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_note.py -- build the standalone HTML walkthrough from steps.json and
figures/*.png.  Every number in the prose comes out of steps.json, so re-running
make_figures.py and then this script keeps the text and the figures in step.

    ../../.venv/bin/python make_note.py            # -> forward_fit_det3.html
"""
from __future__ import annotations

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'steps.json')))
OUT = os.path.join(HERE, 'forward_fit_det3.html')


def img(name, alt):
    p = os.path.join(HERE, 'figures', name + '.png')
    b = base64.b64encode(open(p, 'rb').read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{alt}">'
            '</figure>')


def cap(t):
    return f'<figcaption>{t}</figcaption>'


ev, raw, tr = S['event'], S['raw'], S['track']
ker, col, nn = S['kernel'], S['column'], S['nnls']
dec, res, sc = S['decompose'], S['residual'], S['scan']
rat, ens = S['ratio'], S['ensemble']
ky, kx = ker['y'], ker['x']
h = ens['held']
fr = ens['full_run']
import math
th_raw = math.degrees(math.atan(ev['tan_raw']))
th_cor = math.degrees(math.atan(ev['tan_corr']))
th_ref = math.degrees(math.atan(ev['tan_ref']))

