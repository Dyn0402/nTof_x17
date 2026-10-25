"""M1 card pogo-field layout: DREAM channel -> (card, col, row) on the 10 x 30 pad grid (1 mm pitch).

Every MX17 readout plane is two M1 cards (DREAM connectors 1-4 = ch 0-255, 5-8 = ch 256-511), each with one
pogo field. The pad -> Mec8 pin wiring comes from the M1 gerbers (rerun/det7_pogo/m1_trace.py, copied here as
m1_pogo_to_mec8.json); the Mec8 pin -> DREAM channel convention was chosen by the det7 X data out of 48
(rerun/det7_pogo/conventions.py: Mec8 k -> FEU connector k+1, local = 2 * signal-pin index + row). The same
convention is assumed for the second card and for every chamber: the M1 card is one design.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NC, NR = 10, 30


def _sig_index(pin):
    return pin - 1 if pin < 17 else pin - 2          # Mec8 grounds at pins 0, 17, 34


def card_layout():
    """local channel 0..255 -> (col, row); plus the GND and NC pad positions."""
    pogo = json.load(open(os.path.join(HERE, 'm1_pogo_to_mec8.json')))
    pos, gnd, nc = {}, [], []
    for o in pogo:
        if len(o['mec']) == 1:
            m = o['mec'][0]
            pos[64 * m['conn'] + 2 * _sig_index(m['pin']) + m['row']] = (o['col'], o['row'])
        elif len(o['mec']) > 1:
            gnd.append((o['col'], o['row']))
        else:
            nc.append((o['col'], o['row']))
    assert len(pos) == 256
    return pos, gnd, nc


def plane_positions():
    """(512, 3) array: card, col, row for every channel of a plane."""
    pos, _, _ = card_layout()
    out = np.zeros((512, 3), int)
    for ch in range(512):
        c, r = pos[ch % 256]
        out[ch] = (ch // 256, c, r)
    return out


def pogo_neighbours():
    """(512, 512) int8: 0 not adjacent, 1 horizontal (radial), 2 vertical (along the edge), 3 diagonal.
    Only pads on the same card are neighbours."""
    P = plane_positions()
    A = np.zeros((512, 512), np.int8)
    same = P[:, None, 0] == P[None, :, 0]
    dc = np.abs(P[:, None, 1] - P[None, :, 1]); dr = np.abs(P[:, None, 2] - P[None, :, 2])
    A[same & (dc == 1) & (dr == 0)] = 1
    A[same & (dc == 0) & (dr == 1)] = 2
    A[same & (dc == 1) & (dr == 1)] = 3
    return A
