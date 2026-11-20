#!/usr/bin/env python3
"""Check that a prose rewrite of a .tex file kept its content: numbers, references, citations, labels,
drafting boxes, comments and float environments must be the same multiset before and after.

    python3 papers/check_rewrite.py <before.tex> <after.tex>      # or two directories of sections

Prints what was lost or gained; exit status 1 if anything differs. See WRITING_STYLE.md.
"""
import re, sys, pathlib
from collections import Counter

def strip_comments(s):
    return re.sub(r'(?<!\\)%.*', '', s)

def tokens(s):
    c = Counter()
    for line in s.splitlines():
        if re.match(r'\s*%', line):
            c['COMMENT ' + line.strip()] += 1
    body = strip_comments(s)
    for m in re.finditer(r'\\(?:[cC]ref|crefp|ref|cite|label|eqref)\{[^}]*\}(?:\{[^}]*\})?', body):
        k = m.group(0); k = k[0] + k[1].lower() + k[2:]          # \Cref == \cref
        c[k] += 1
    for name in ('askDylan', 'askStephan', 'analysis', 'requote', 'sectionstate', 'intent', 'tbw',
                 'includegraphics', 'figorbox', 'begin', 'end', 'input'):
        c['\\' + name] += len(re.findall(r'\\' + name + r'\b', body))
    words = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
             'eight': '8', 'nine': '9', 'ten': '10'}
    for m in re.finditer(r'(?<![A-Za-z_:\-{])\d+(?:\.\d+)?', body):
        c['NUM ' + m.group(0)] += 1
    for w, d in words.items():
        n = len(re.findall(r'\b' + w + r'\b', body, flags=re.I))
        if n: c['WORDNUM ' + d] += n
    return c

def compare(a, b):
    ca, cb = tokens(a.read_text()), tokens(b.read_text())
    lost, gained = ca - cb, cb - ca
    soft = lambda k: k.startswith('WORDNUM')
    hard = [k for k in list(lost) + list(gained) if not soft(k)]
    if lost or gained:
        print(f'== {b.name}')
        for k, n in sorted(lost.items()):   print(f'  lost   x{n}: {k[:110]}')
        for k, n in sorted(gained.items()): print(f'  gained x{n}: {k[:110]}')
    return bool(hard)

a, b = map(pathlib.Path, sys.argv[1:3])
pairs = [(f, b / f.name) for f in sorted(a.glob('*.tex'))] if a.is_dir() else [(a, b)]
bad = [compare(x, y) for x, y in pairs if y.exists()]
print('OK: content tokens unchanged' if not any(bad) else 'DIFFERENCES above (WORDNUM lines are only a hint)')
sys.exit(1 if any(bad) else 0)
