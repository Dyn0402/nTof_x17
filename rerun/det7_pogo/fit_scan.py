"""Grid scan of the rigid pin-array displacement (dx, dy, theta, rp) for the faulty M1 card
(det7 X, FEU 6 channels 0-255). Score: neighbour pairs on the pogo grid (h/v/diagonal) whose
measured amplitude correlation > T against pairs predicted to share a net. Writes scan_<mode>.npz."""
import sys, numpy as np, multiprocessing as mp
import model as M

T = 0.9
C = np.load('corr_feu6.npz')['C']
MEAS = C[M._PA, M._PB] > T
DX = np.round(np.arange(-0.7, 0.7001, 0.025), 4)
TH = np.round(np.arange(-3, 3.001, 0.1), 3)
RP = [0.06, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3]


def work(args):
    th, rp, mode = args
    out = np.zeros((len(DX), len(DX), 3), np.int32)
    for i, dx in enumerate(DX):
        for j, dy in enumerate(DX):
            l = M.fast_labels(dx, dy, th, rp, mode)
            out[i, j] = M.fast_score(l[0], l[1], MEAS)
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'readout'
    jobs = [(th, rp, mode) for th in TH for rp in RP]
    with mp.Pool(8) as p:
        res = p.map(work, jobs, chunksize=4)
    R = np.array(res).reshape(len(TH), len(RP), len(DX), len(DX), 3)
    np.savez_compressed(f'scan_{mode}.npz', R=R, DX=DX, TH=TH, RP=RP, T=T, nmeas=int(MEAS.sum()))
    tp, fp, fn = R[..., 0], R[..., 1], R[..., 2]
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    for k in np.argsort(f1.ravel())[::-1][:10]:
        a, b, i, j = np.unravel_index(k, f1.shape)
        print(f'th {TH[a]:+.1f} rp {RP[b]} dx {DX[i]:+.3f} dy {DX[j]:+.3f}  F1 {f1[a,b,i,j]:.3f} tp {tp[a,b,i,j]} fp {fp[a,b,i,j]} fn {fn[a,b,i,j]}')
