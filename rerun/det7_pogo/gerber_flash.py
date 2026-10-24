"""Minimal Gerber flash reader with net (TO.N) attributes. Only D03 flashes and
D01 draws (as segments) are kept; enough for connector pad fields."""
import re
from dataclasses import dataclass

@dataclass
class Flash:
    x: float; y: float; ap: int; net: str | None; polarity: str

def parse(path, keep_draws=False):
    aps = {}
    flashes, draws = [], []
    fmt_dec = 6
    cur_ap = None; net = None; x = y = 0.0; pol = 'D'
    num = re.compile(r'([XYIJD])([-+]?\d+)')
    with open(path, errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if line.startswith('G04 #@! TO.N,'):
                net = line[len('G04 #@! TO.N,'):].rstrip('*').split(',')[0]; continue
            if line.startswith('G04 #@! TD'):
                net = None; continue
            if line.startswith('G04'):
                continue
            m = re.match(r'%FSLAX(\d)(\d)Y', line)
            if m: fmt_dec = int(m.group(2)); continue
            m = re.match(r'%ADD(\d+)([A-Za-z_0-9]+),?([^*]*)\*%', line)
            if m:
                aps[int(m.group(1))] = (m.group(2), [float(v) for v in m.group(3).split('X') if v]); continue
            if line.startswith('%LP'):
                pol = line[3]; continue
            if line.startswith('%'):
                continue
            m = re.match(r'^(?:G\d\d)?D(\d+)\*$', line)
            if m and int(m.group(1)) >= 10:
                cur_ap = int(m.group(1)); continue
            toks = dict(num.findall(line))
            if not toks: continue
            nx = int(toks['X']) / 10**fmt_dec if 'X' in toks else x
            ny = int(toks['Y']) / 10**fmt_dec if 'Y' in toks else y
            d = int(toks.get('D', 1)) if 'D' in toks else 1
            if d == 3:
                flashes.append(Flash(nx, ny, cur_ap, net, pol))
            elif d == 1 and keep_draws:
                draws.append((x, y, nx, ny, cur_ap, net, pol))
            x, y = nx, ny
    return aps, flashes, draws
