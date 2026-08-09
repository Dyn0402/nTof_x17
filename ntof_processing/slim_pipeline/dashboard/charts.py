#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
charts.py -- inline SVG chart primitives for the clock-QA dashboard.

Self-contained by requirement: the dashboard is published as a single HTML file
that must render with no network, so there is no plotting library, no CDN and
no embedded raster. Every chart is SVG built here as text.

Conventions that are not negotiable, and why:
  * one y axis, ever. Two scales on one frame is the most common way to imply a
    relationship that is not there.
  * categorical colour is assigned by ENTITY (arm A is always blue), never by
    position in a filtered list, so hiding a series never repaints the others.
  * marks are thin, the grid is recessive, and every mark carries a
    `data-tip` for the shared hover layer in the page.
  * palettes are the validated ones -- see PALETTE below. Do not add a fifth
    categorical hue: there are four arms, and a fifth slot would have to be
    generated rather than chosen, which is where CVD-unsafe pairs come from.
"""
from __future__ import annotations

import html
import math

# Validated with dataviz/scripts/validate_palette.js:
#   light  surface #fcfcfb -- all six checks pass, worst adjacent dE 15.0 (deutan)
#   dark   surface #1a1a19 -- all six checks pass
# The four hues map onto the four arms A/B/C/D, which is the only categorical
# dimension this dashboard has.
PALETTE = ['#2563eb', '#ea580c', '#0891b2', '#7c3aed']
PALETTE_DARK = ['#3b82f6', '#e8690b', '#0d9fbd', '#a855f7']
ARM_COLOUR = dict(zip('ABCD', PALETTE))

STATUS = dict(PASS='var(--good)', WARN='var(--warn)', FAIL='var(--bad)',
              NA='var(--muted)')


def esc(s):
    return html.escape(str(s), quote=True)


def _nice(lo, hi, n=5):
    """Round axis bounds outward to human numbers."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0.0, 1.0, [0.0, 1.0]
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.1, 1e-9)
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / max(n, 1)))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    lo2 = math.floor(lo / step) * step
    hi2 = math.ceil(hi / step) * step
    ticks, t = [], lo2
    while t <= hi2 + step * 1e-6:
        ticks.append(round(t, 12))
        t += step
    return lo2, hi2, ticks


def _fmt(v):
    a = abs(v)
    if v == 0:
        return '0'
    if a >= 1e4 or a < 1e-3:
        return f'{v:.1e}'
    if a >= 100:
        return f'{v:.0f}'
    if a >= 1:
        return f'{v:.2f}'.rstrip('0').rstrip('.')
    return f'{v:.4f}'.rstrip('0').rstrip('.')


class Frame:
    """A plot frame with linear axes, a recessive grid and a title."""

    # Left pad fits a rotated axis label AND a full-width tick like "1.4e-04";
    # at 56 they overlapped.
    def __init__(self, w=760, h=260, pad=(52, 16, 34, 74), title='',
                 ylabel='', xlabel=''):
        self.w, self.h = w, h
        self.t, self.r, self.b, self.l = pad      # noqa: E741
        self.title, self.ylabel, self.xlabel = title, ylabel, xlabel
        self.body = []
        self.x0, self.x1 = 0.0, 1.0
        self.y0, self.y1 = 0.0, 1.0
        self.xticks, self.yticks = [], []
        self.logy = False

    # -- scales ------------------------------------------------------------
    def xlim(self, lo, hi, ticks=None):
        self.x0, self.x1, tk = _nice(lo, hi)
        self.xticks = ticks if ticks is not None else tk
        return self

    def ylim(self, lo, hi, ticks=None, log=False):
        self.logy = log
        if log:
            lo = max(lo, 1e-12)
            hi = max(hi, lo * 10)
            self.y0, self.y1 = math.log10(lo), math.log10(hi)
            e0, e1 = math.floor(self.y0), math.ceil(self.y1)
            self.y0, self.y1 = e0, e1
            self.yticks = [10 ** e for e in range(int(e0), int(e1) + 1)]
        else:
            self.y0, self.y1, tk = _nice(lo, hi)
            self.yticks = ticks if ticks is not None else tk
        return self

    def px(self, x):
        if self.x1 == self.x0:
            return self.l
        return self.l + (x - self.x0) / (self.x1 - self.x0) * self._pw()

    def py(self, y):
        if self.logy:
            y = math.log10(max(y, 1e-12))
        if self.y1 == self.y0:
            return self.h - self.b
        return (self.h - self.b
                - (y - self.y0) / (self.y1 - self.y0) * self._ph())

    def _pw(self):
        return self.w - self.l - self.r

    def _ph(self):
        return self.h - self.t - self.b

    # -- marks -------------------------------------------------------------
    def band(self, lo, hi, label=''):
        """A horizontal reference band -- the expected range."""
        y1, y0 = self.py(hi), self.py(lo)
        self.body.append(
            f'<rect class="band" x="{self.l:.1f}" y="{y1:.1f}" '
            f'width="{self._pw():.1f}" height="{abs(y0-y1):.1f}"><title>'
            f'{esc(label)}</title></rect>')
        return self

    def hline(self, y, cls='ref', label=''):
        yy = self.py(y)
        self.body.append(
            f'<line class="{cls}" x1="{self.l:.1f}" y1="{yy:.1f}" '
            f'x2="{self.w-self.r:.1f}" y2="{yy:.1f}"><title>{esc(label)}'
            f'</title></line>')
        return self

    def points(self, xs, ys, colours=None, tips=None, r=3.2):
        for i, (x, y) in enumerate(zip(xs, ys)):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            c = (colours[i] if isinstance(colours, list) else colours) \
                or 'var(--series1)'
            tip = esc(tips[i]) if tips else ''
            self.body.append(
                f'<circle class="pt" cx="{self.px(x):.1f}" '
                f'cy="{self.py(y):.1f}" r="{r}" fill="{c}" '
                f'data-tip="{tip}"><title>{tip}</title></circle>')
        return self

    def line(self, xs, ys, colour='var(--series1)', width=2, dash=None,
             tip=''):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)
               if math.isfinite(x) and math.isfinite(y)]
        if len(pts) < 2:
            return self
        d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.body.append(
            f'<path class="ln" d="{d}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}"{da}><title>{esc(tip)}</title></path>')
        return self

    def step_hist(self, lo, bin_w, counts, colour='var(--series1)', width=2,
                  tip=''):
        xs, ys = [], []
        for i, c in enumerate(counts):
            xs += [lo + i * bin_w, lo + (i + 1) * bin_w]
            ys += [c, c]
        return self.line(xs, ys, colour, width, tip=tip)

    def bars(self, xs, ys, colour='var(--series1)', w=None, tips=None):
        bw = w or (self._pw() / max(len(xs), 1) * 0.7)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if not math.isfinite(y):
                continue
            px, py = self.px(x), self.py(y)
            y0 = self.py(max(self.y0, 0) if not self.logy else self.y0)
            tip = esc(tips[i]) if tips else ''
            self.body.append(
                f'<rect class="bar" x="{px-bw/2:.1f}" y="{min(py,y0):.1f}" '
                f'width="{bw:.1f}" height="{abs(y0-py):.1f}" rx="2" '
                f'fill="{colour}" data-tip="{tip}"><title>{tip}</title></rect>')
        return self

    # -- render ------------------------------------------------------------
    def svg(self):
        g = []
        for t in self.yticks:
            y = self.py(t)
            if not math.isfinite(y):
                continue
            g.append(f'<line class="grid" x1="{self.l}" y1="{y:.1f}" '
                     f'x2="{self.w-self.r}" y2="{y:.1f}"/>')
            g.append(f'<text class="tick" x="{self.l-8}" y="{y+4:.1f}" '
                     f'text-anchor="end">{_fmt(t)}</text>')
        for t in self.xticks:
            x = self.px(t)
            if not math.isfinite(x) or x < self.l - 1 or x > self.w - self.r + 1:
                continue
            g.append(f'<text class="tick" x="{x:.1f}" y="{self.h-self.b+18}" '
                     f'text-anchor="middle">{_fmt(t)}</text>')
        g.append(f'<line class="axis" x1="{self.l}" y1="{self.h-self.b}" '
                 f'x2="{self.w-self.r}" y2="{self.h-self.b}"/>')
        head = (f'<text class="ctitle" x="{self.l}" y="16">'
                f'{esc(self.title)}</text>' if self.title else '')
        yl = (f'<text class="axlabel" transform="translate(14,'
              f'{self.t+self._ph()/2}) rotate(-90)" text-anchor="middle">'
              f'{esc(self.ylabel)}</text>' if self.ylabel else '')
        xl = (f'<text class="axlabel" x="{self.l+self._pw()/2}" '
              f'y="{self.h-2}" text-anchor="middle">{esc(self.xlabel)}</text>'
              if self.xlabel else '')
        return (f'<svg viewBox="0 0 {self.w} {self.h}" '
                f'preserveAspectRatio="xMidYMid meet" role="img">'
                f'{head}{yl}{xl}{"".join(g)}{"".join(self.body)}</svg>')


def legend(items):
    """items = [(label, colour)] -- always present for >= 2 series."""
    sp = ''.join(
        f'<span class="lg"><i style="background:{c}"></i>{esc(l)}</span>'
        for l, c in items)
    return f'<div class="legend">{sp}</div>'
