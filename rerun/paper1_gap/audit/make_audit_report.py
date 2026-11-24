#!/usr/bin/env python3
"""make_audit_report.py -- report.html of the 2026-11-24 audit from AUDIT_2026-11-24.md (the single source) and figures/.

The markdown subset used by the audit note is converted here (no markdown package in the venv): #/##/### headings,
paragraphs, - lists (one nesting level), | tables, ![caption](figures/x.png) figures, **bold**, `code`, [text](link).
Figures are linked relatively (figures/...), so the report works from disk, from the DAQ page and from EOS.
"""
import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'AUDIT_2026-11-24.md')
CSS = """
body{font:16px/1.55 Georgia,serif;max-width:60em;margin:2em auto;padding:0 1em;color:#1b1b1b;background:#fff}
h1{font-size:1.6em} h2{font-size:1.25em;margin-top:2em;border-bottom:1px solid #ccc} h3{font-size:1.05em;margin-top:1.4em}
table{border-collapse:collapse;margin:1em 0;font-size:.9em} td,th{border:1px solid #bbb;padding:.25em .6em;text-align:left}
th{background:#f0f0f0} code{font-size:.88em;background:#f4f4f4;padding:0 .2em} figure{margin:1.4em 0}
figure img{max-width:100%;border:1px solid #ddd} figcaption{font-size:.9em;color:#333;margin-top:.3em}
li{margin:.25em 0} .verdict{background:#f6f8fb;border-left:4px solid #4a6fa5;padding:.6em 1em}
"""


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
    return s


def convert(md):
    out, i, L = [], 0, md.split('\n')
    while i < len(L):
        l = L[i]
        if not l.strip():
            i += 1
        elif l.startswith('#'):
            n = len(l) - len(l.lstrip('#'))
            out.append(f'<h{n}>{inline(l[n:].strip())}</h{n}>'); i += 1
        elif l.startswith('!['):
            m = re.match(r'!\[(.*)\]\((.*)\)\s*$', l)
            out.append(f'<figure><img src="{m.group(2)}" alt=""><figcaption>{inline(m.group(1))}</figcaption></figure>'); i += 1
        elif l.startswith('|'):
            rows = []
            while i < len(L) and L[i].startswith('|'):
                rows.append([c.strip() for c in L[i].strip().strip('|').split('|')]); i += 1
            body = [r for r in rows[1:] if not all(re.fullmatch(r':?-+:?', c) for c in r)]
            out.append('<table><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in rows[0]) + '</tr>'
                       + ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in body) + '</table>')
        elif re.match(r'\s*(-|\d+\.) ', l):
            stack = []          # (indent, tag)
            while i < len(L) and re.match(r'\s*(-|\d+\.) ', L[i]):
                ind = len(L[i]) - len(L[i].lstrip())
                tag = 'ol' if re.match(r'\s*\d+\. ', L[i]) else 'ul'
                while stack and stack[-1][0] > ind:
                    out.append(f'</{stack.pop()[1]}>')
                if not stack or stack[-1][0] < ind:
                    stack.append((ind, tag)); out.append(f'<{tag}>')
                text = re.sub(r'^\s*(-|\d+\.) ', '', L[i]); i += 1
                while i < len(L) and L[i].startswith(' ' * (ind + 2)) and not re.match(r'\s*(-|\d+\.) ', L[i]):
                    text += ' ' + L[i].strip(); i += 1
                out.append(f'<li>{inline(text)}</li>')
            while stack:
                out.append(f'</{stack.pop()[1]}>')
        else:
            p = []
            while i < len(L) and L[i].strip() and not re.match(r'(#|\||!\[|\s*(-|\d+\.) )', L[i]):
                p.append(L[i].strip()); i += 1
            out.append(f'<p>{inline(" ".join(p))}</p>')
    return '\n'.join(out)


def main():
    md = open(SRC).read()
    title = md.split('\n', 1)[0].lstrip('# ').strip()
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{CSS}</style></head><body>\n'
            f'{convert(md)}\n</body></html>\n')
    open(os.path.join(HERE, 'report.html'), 'w').write(page)
    missing = [m for m in re.findall(r'\]\((figures/[^)]+)\)', md) if not os.path.exists(os.path.join(HERE, m))]
    print('wrote report.html;', 'missing figures:', missing or 'none')


if __name__ == '__main__':
    main()
