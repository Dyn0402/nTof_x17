"""Contact model: the JZ pogo-pin array displaced (shift dx, dy and rotation theta about
the pad-field centre) relative to the 10 x 30 pad field. Each pin is a disk of radius rp;
it contacts every pad whose copper lies within rp of the pin axis. Pads: BOXR 0.6 x 0.8 mm,
corner radius 0.25, rotated 45 deg, on a 1 mm grid (M1 bottom and readout board alike).

Graph: pins + M1 pads + readout pads. Mode 'readout': pins sit on their own M1 pad, displaced
only relative to the readout board. Mode 'carrier': the JZ carrier itself is displaced and the
M1 card is not, so the same offset applies on both faces (the logbook says the X1 JZ connector
on det7 was glued only on 06-30, after this run).

Roles at each grid position come from the M1 trace (m1_pogo_to_mec8.json): signal channel,
GND (wired to the Mec8 ground pins) or NC (no trace). The readout board is assumed to carry
the strip of the same nominal position on signal pads, ground on GND and nothing on NC."""
import json, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import conventions as Cv

NC, NR = 10, 30
X0, Y0 = 228.5, 85.339995
XC, YC = X0 + 4.5, Y0 + 14.5
HALF = np.array([0.3 - 0.25, 0.4 - 0.25]); RAD = 0.25
c45 = np.cos(np.pi / 4)

POS = Cv.feu_positions((0, 0, 0), (0, 1, 2, 3))          # FEU channel (card 1) -> (col, row)
CH_AT = {v: k for k, v in POS.items()}
ROLE = {}
for o in Cv.POGO:
    ROLE[(o['col'], o['row'])] = 'S' if len(o['mec']) == 1 else ('G' if len(o['mec']) > 1 else 'N')
GRID = [(c, r) for r in range(NR) for c in range(NC)]
GIDX = {g: i for i, g in enumerate(GRID)}
GXY = np.array([(X0 + c, Y0 + r) for c, r in GRID])


def pad_dist(px, py, cx, cy):
    """distance from points (px,py) to the pad copper centred at (cx,cy) (<0 inside)."""
    dx, dy = px - cx, py - cy
    u = c45 * dx + c45 * dy          # rotate by -45 deg into the pad frame
    v = -c45 * dx + c45 * dy
    qx = np.maximum(np.abs(u) - HALF[0], 0); qy = np.maximum(np.abs(v) - HALF[1], 0)
    return np.hypot(qx, qy) - RAD


NB = [(a, b) for a in range(-2, 3) for b in range(-2, 3)]


def contacts(dx, dy, th_deg, rp):
    """list of (pin index, pad index) in contact."""
    t = np.deg2rad(th_deg)
    ct, st = np.cos(t), np.sin(t)
    x = GXY[:, 0] - XC; y = GXY[:, 1] - YC
    px = XC + ct * x - st * y + dx; py = YC + st * x + ct * y + dy
    ci = np.rint(px - X0).astype(int); ri = np.rint(py - Y0).astype(int)
    pins, pads = [], []
    for a, b in NB:
        cc, rr = ci + a, ri + b
        ok = (cc >= 0) & (cc < NC) & (rr >= 0) & (rr < NR)
        d = pad_dist(px, py, X0 + cc, Y0 + rr)
        hit = ok & (d < rp)
        pins.append(np.flatnonzero(hit)); pads.append((rr * NC + cc)[hit])
    return np.concatenate(pins), np.concatenate(pads)


def nets(dx, dy, th, rp, mode='readout'):
    """Return per-channel net label array over FEU 0..255, plus flags grounded / no-strip / strips per channel."""
    n = len(GRID)
    # nodes: 0..n-1 pins, n..2n-1 M1 pads, 2n..3n-1 readout pads, 3n GND
    pi, pj = contacts(dx, dy, th, rp)
    rows, cols = [pi], [2 * n + pj]
    if mode == 'carrier':
        rows.append(pi); cols.append(n + pj)
    else:
        rows.append(np.arange(n)); cols.append(n + np.arange(n))
    G = 3 * n
    for i, gp in enumerate(GRID):
        if ROLE[gp] == 'G':
            rows += [np.array([n + i, 2 * n + i])]; cols += [np.array([G, G])]
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(G + 1, G + 1))
    _, lab = connected_components(A, directed=False)
    gl = lab[G]
    out = {}
    strips_of = {}
    for i, gp in enumerate(GRID):
        if ROLE[gp] == 'S':
            strips_of.setdefault(lab[2 * n + i], []).append(CH_AT[gp])   # readout pad i carries the strip nominally read by channel CH_AT[g]
    for gp, ch in CH_AT.items():
        L = lab[n + GIDX[gp]]
        out[ch] = dict(net=int(L), grounded=bool(L == gl), strips=sorted(strips_of.get(L, [])))
    return out


def neighbour_pairs():
    P = []
    for (c, r), ch in CH_AT.items():
        for dc, dr in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
            o = CH_AT.get((c + dc, r + dr))
            if o is not None:
                P.append((ch, o, 'h' if dr == 0 else ('v' if dc == 0 else 'd')))
    return P


PAIRS = neighbour_pairs()


def score(pred, Cmat, T=0.9):
    tp = fp = fn = tn = 0
    for a, b, k in PAIRS:
        m = Cmat[a, b] > T
        p = (pred[a]['net'] == pred[b]['net']) and not pred[a]['grounded']
        tp += m and p; fp += p and not m; fn += m and not p; tn += (not m) and (not p)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, f1=f1)


# ---------- fast path for scans ----------
_N = len(GRID)
_SIG = np.array([GIDX[g] for g in sorted(CH_AT, key=lambda g: CH_AT[g])])     # grid index of FEU ch 0..255
assert len(_SIG) == 256 and all(CH_AT[GRID[_SIG[k]]] == k for k in range(256))
_GND = np.array([i for i, g in enumerate(GRID) if ROLE[g] == 'G'])
_PA = np.array([p[0] for p in PAIRS]); _PB = np.array([p[1] for p in PAIRS])
_PK = np.array([p[2] for p in PAIRS])


def fast_labels(dx, dy, th, rp, mode='readout'):
    n = _N; G = 3 * n
    pi, pj = contacts(dx, dy, th, rp)
    r = [pi]; c = [2 * n + pj]
    if mode == 'carrier':
        r.append(pi); c.append(n + pj)
    else:
        r.append(np.arange(n)); c.append(n + np.arange(n))
    r += [n + _GND, 2 * n + _GND]; c += [np.full(len(_GND), G)] * 2
    r = np.concatenate(r); c = np.concatenate(c)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(G + 1, G + 1)).tocsr()
    _, lab = connected_components(A, directed=False)
    chl = lab[n + _SIG]
    grounded = chl == lab[G]
    sig_ro = lab[2 * n + _SIG]                  # readout pad of each nominal strip
    has_strip = np.isin(chl, sig_ro)
    return chl, grounded, has_strip


def fast_score(chl, grounded, M):
    """M: boolean measured label for each PAIRS entry."""
    p = (chl[_PA] == chl[_PB]) & ~grounded[_PA]
    tp = int((p & M).sum()); fp = int((p & ~M).sum()); fn = int((~p & M).sum())
    return tp, fp, fn
