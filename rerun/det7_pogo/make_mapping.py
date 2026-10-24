"""Measured per-channel strip mapping for det7 X (FEU 6), June bench g_det7_long.
A net = channels joined by pogo-grid neighbour links with amplitude correlation > 0.9
(links between non-neighbours are not used, so physical charge sharing cannot merge nets).
Every channel in a net is assumed to read the strips nominally belonging to all channels of
the net. Links 0.7-0.9 are listed as 'possible'. Writes det7_x_strip_map_measured.json."""
import json, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import model as M

C = np.load('corr_feu6.npz')['C']
cl = M.Cv.CL['x']
v = C[M._PA, M._PB]
def comps(T):
    m = v > T
    A = coo_matrix((np.ones(m.sum()), (M._PA[m], M._PB[m])), shape=(256, 256))
    return connected_components(A, directed=False)[1]
lab9, lab7 = comps(0.9), comps(0.7)
chan = {}
for ch in range(512):
    d = dict(feu_channel=ch, connector=ch // 64 + 1, local=ch % 64,
             noisy=ch in cl['noisy'], dead=ch in cl['dead'], weak=ch in cl['weak'])
    if ch < 256:
        c, r = M.POS[ch]
        d['pogo'] = dict(col=c, row=r)
        net = sorted(int(k) for k in np.flatnonzero(lab9 == lab9[ch]))
        pos = sorted(int(k) for k in np.flatnonzero(lab7 == lab7[ch]))
        d['net'] = net
        d['strips'] = net              # strips named by the channel that nominally reads them (linear map)
        d['possible_net_0p7'] = pos if pos != net else None
    else:
        d['net'] = [ch]; d['strips'] = [ch]; d['possible_net_0p7'] = None
    d['n_strips'] = len(d['strips'])
    chan[ch] = d
nets = sorted({tuple(d['net']) for d in chan.values() if len(d['net']) > 1}, key=len, reverse=True)
out = dict(run_key='g_det7_long', detector='det7 (mx17_7, D)', view='x', feu=6,
           basis='amplitude correlation > 0.9 between pogo-grid neighbours (rerun/det7_pogo/corr_feu6.npz); '
                 'pogo grid from the M1 gerbers (m1_trace.py); strips named by the FEU channel of the linear map (mx17_m1_map.csv)',
           forward_model_note='a net of n shorted channels: each channel reads (1/n) x the sum of the net\'s strip signals '
                              '(n low-impedance inputs tied together share the charge); treat the net as ONE virtual channel '
                              'with a comb of strips and a free gain, and drop the n-1 duplicates from the likelihood',
           nets=[list(n) for n in nets], n_shorted_channels=sum(len(n) for n in nets),
           channels=[chan[k] for k in range(512)])
json.dump(out, open('det7_x_strip_map_measured.json', 'w'), indent=1)
print(len(nets), 'nets,', out['n_shorted_channels'], 'channels'); print(nets)
