#!/usr/bin/env python3
"""
g4_mix.py -- what the scintillators see after the γ-flash, from Geant4: the particle
mix and the deposit (light) spectrum against time since the flash.

    python -m ntof_scint_recovery.g4_mix <neutron_bg_job*.root> [...] --out g4mix_<n>.npz
    python -m ntof_scint_recovery.g4_mix --merge <g4mix_*.npz> --out g4mix.npz

Input: the MX17_Full_Geant ``neutrons_fullrange`` campaign (5e8 EAR2 neutrons,
1 meV-100 MeV, sampled from the Phase-3 EAR2 flux, n_TOF apparatus with walls,
plastics and liquids; per-step HitTree).  Each event is one beam neutron; the run
is 2.263e7 n/pulse over the full range (MX17_Full_Geant HANDOFF_MEV_ANALYSIS).

Time since the flash at the detector = TOF(E_n, L) + hit time - L/c, L = 19.5 m
(the hit time runs from the gun 20 cm upstream of the capsule).

Per (event, arm, counter) the steps are clustered in time (a gap > ``GAP_NS``
starts a new cluster), as one digitised pulse would integrate them.  Light is
the step deposit with Birks quenching from the step's kinetic energy:
e±/γ unquenched; p, d, t, α through dE/dx(E) (PSTAR water x 0.97, velocity-scaled,
z²); heavier ions, and deposits booked on neutron steps (recoils HP does not create),
at 1 % (C recoils make almost no light).  kB = 0.126 mm/MeV.

A cluster's class is the particle that made most of its light:
0 e± (γ-induced), 1 proton (neutron recoil), 2 d/t/α/ions, 3 other.

Histograms (counts per file; the merge sums them):
``H[fam, arm, t, light, cls]``  every cluster, fam 0 wall / 1 plastic / 2 liquid;
``HE[fam, arm, t, edep, cls]``  the same in deposited (unquenched) energy;
``PW[arm, t, light, cls]``       plastic clusters with a wall cluster of the same arm
                                 (>= ``WALL_MIN`` MeV light) within +-``COINC_NS``;
``n_events``.

Not in the simulation: the in-beam γ background of the spallation target (delayed
2.2 MeV γ from the moderator), the experimental hall (room-return neutrons, captures in
concrete and the dump), and the γ-flash itself.  The beam profile is the pre-2026-10-14
one (r50 2.5 mm, ~5x too narrow).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

L_M = 19.5
C_M_PER_NS = 0.299792458
MN = 939.565         # MeV
GAP_NS = 30.0
COINC_NS = 30.0
WALL_MIN = 0.05
KB = 0.126           # mm/MeV
RHO = 1.03           # g/cm3
T_EDGES_US = np.concatenate([np.logspace(-2, 4, 121), [3e4, 3.2e4]])
L_EDGES = np.logspace(-3, 2, 101)
FAMS = ('wall', 'plastic', 'liquid')
CLS = ('e/γ', 'proton', 'd/t/α/ion', 'other')
N_PER_PULSE = 2.263e7     # fullrange: n/pulse from the flux file (7e12 ppp nominal)

#: PSTAR water electronic stopping power, MeV cm2/g
_PE = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000])
_PS = np.array([176.9, 412.0, 499.6, 610.0, 798.0, 817.0, 808.0, 720.0, 418.0, 260.8, 162.4, 79.11,
                45.67, 26.07, 12.45, 7.289, 2.211])


def s_proton(e_mev):
    """dE/dx of a proton in plastic, MeV/mm."""
    e = np.clip(e_mev, _PE[0], _PE[-1])
    return np.exp(np.interp(np.log(e), np.log(_PE), np.log(_PS))) * 0.97 * RHO / 10.0


def quench(particle: np.ndarray, ke: np.ndarray, edep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Light (MeVee) and class per step."""
    light = edep.copy()
    cls = np.full(len(edep), 3, np.int8)
    em = np.isin(particle, ('e-', 'e+', 'gamma'))
    cls[em] = 0
    p = particle == 'proton'
    light[p] = edep[p] / (1 + KB * s_proton(ke[p]))
    cls[p] = 1
    for name, m, z2 in (('deuteron', 2, 1), ('triton', 3, 1), ('alpha', 4, 4), ('He3', 3, 4)):
        k = particle == name
        light[k] = edep[k] / (1 + KB * z2 * s_proton(ke[k] / m))
        cls[k] = 2
    # heavy recoils, and the deposit HP books on the neutron step itself (recoil nuclei it does not create)
    ion = ~em & ~p & ~np.isin(particle, ('deuteron', 'triton', 'alpha', 'He3', 'pi+', 'pi-', 'mu+', 'mu-'))
    light[ion] = 0.01 * edep[ion]
    cls[ion] = 2
    return light, cls


def tof_ns(e_ev: np.ndarray) -> np.ndarray:
    T = e_ev * 1e-6
    gam = 1 + T / MN
    beta = np.sqrt(1 - 1 / gam ** 2)
    return L_M / (beta * C_M_PER_NS)


def counter(det: np.ndarray) -> np.ndarray:
    """0 wall, 1 plastic L, 2 plastic R, 3 liquid 1, 4 liquid 2, -1 other."""
    c = np.full(len(det), -1, np.int8)
    for i, pre in enumerate(('PlasticScint', 'BackScintL', 'BackScintR', 'LiqScint_1', 'LiqScint_2')):
        c[np.char.startswith(det, pre)] = i
    return c


FAM_OF = np.array([0, 1, 1, 2, 2])


def reduce_file(path: str, H: dict) -> None:
    import uproot
    f = uproot.open(path)
    eE = f['EventTree'].arrays(['eventID', 'neutron_E_eV'], library='np')
    nev = len(eE['eventID'])
    Emap = np.zeros(int(eE['eventID'].max()) + 1)
    Emap[eE['eventID']] = eE['neutron_E_eV']
    H['n_events'] += nev
    cols = ['eventID', 'armID', 'detType', 'particle', 'edep', 'ke', 'time']
    for h in f['HitTree'].iterate(cols, library='np', step_size='400 MB'):
        det = np.asarray(h['detType']).astype(str)
        cnt = counter(det)
        part = np.asarray(h['particle']).astype(str)
        keep = (cnt >= 0) & (h['edep'] > 0)
        if not keep.any():
            continue
        ev = h['eventID'][keep].astype(np.int64)
        arm = h['armID'][keep].astype(np.int64)
        cn = cnt[keep].astype(np.int64)
        edep = h['edep'][keep] * 1e-6                  # eV -> MeV
        light, cls = quench(part[keep], h['ke'][keep], edep)
        t = tof_ns(Emap[ev]) + h['time'][keep] - L_M / C_M_PER_NS
        ok = np.isfinite(t) & (t < 3.3e7)              # radioactive decays at 1e8+ ns
        ev, arm, cn, edep, light, cls, t = ev[ok], arm[ok], cn[ok], edep[ok], light[ok], cls[ok], t[ok]
        # clusters per (event, arm, counter) in time
        key = (ev * 4 + arm) * 5 + cn
        o = np.lexsort((t, key))
        key, t, edep, light, cls, arm, cn = key[o], t[o], edep[o], light[o], cls[o], arm[o], cn[o]
        new = np.r_[True, (np.diff(key) != 0) | (np.diff(t) > GAP_NS)]
        cid = np.cumsum(new) - 1
        nc = cid[-1] + 1
        cl_t = t[new]
        cl_key = key[new]
        cl_arm = arm[new]
        cl_fam = FAM_OF[cn[new]]
        cl_L = np.bincount(cid, light, nc)
        cl_E = np.bincount(cid, edep, nc)
        Lc = np.zeros((nc, 4))
        np.add.at(Lc, (cid, cls), light)
        cl_cls = np.argmax(Lc, axis=1)
        it = np.digitize(cl_t / 1e3, T_EDGES_US) - 1
        il = np.clip(np.digitize(cl_L, L_EDGES) - 1, 0, len(L_EDGES) - 2)
        ie = np.clip(np.digitize(cl_E, L_EDGES) - 1, 0, len(L_EDGES) - 2)
        g = (it >= 0) & (it < len(T_EDGES_US) - 1) & (cl_L >= L_EDGES[0])
        np.add.at(H['H'], (cl_fam[g], cl_arm[g], it[g], il[g], cl_cls[g]), 1)
        g2 = (it >= 0) & (it < len(T_EDGES_US) - 1) & (cl_E >= L_EDGES[0])
        np.add.at(H['HE'], (cl_fam[g2], cl_arm[g2], it[g2], ie[g2], cl_cls[g2]), 1)
        # plastic behind a wall hit: same event and arm, wall light >= WALL_MIN within +-COINC_NS
        evarm = cl_key // 5
        w = (cl_fam == 0) & (cl_L >= WALL_MIN)
        wk = evarm[w].astype(np.float64) * 1e11 + cl_t[w]
        ws = np.sort(wk)
        p = np.nonzero((cl_fam == 1) & g)[0]
        if len(p) and len(ws):
            pk = evarm[p].astype(np.float64) * 1e11 + cl_t[p]
            a = np.searchsorted(ws, pk - COINC_NS)
            b = np.searchsorted(ws, pk + COINC_NS, side='right')
            m = b > a
            q = p[m]
            np.add.at(H['PW'], (cl_arm[q], it[q], il[q], cl_cls[q]), 1)


def empty() -> dict:
    nT, nL = len(T_EDGES_US) - 1, len(L_EDGES) - 1
    return dict(H=np.zeros((3, 4, nT, nL, 4), np.float64), HE=np.zeros((3, 4, nT, nL, 4), np.float64),
                PW=np.zeros((4, nT, nL, 4), np.float64), n_events=0)


def save(H: dict, out: Path, n_files: int) -> None:
    np.savez_compressed(out, **H, n_files=n_files, t_edges_us=T_EDGES_US, l_edges=L_EDGES,
                        fams=np.array(FAMS), cls=np.array(CLS), n_per_pulse=N_PER_PULSE,
                        gap_ns=GAP_NS, coinc_ns=COINC_NS, wall_min=WALL_MIN, kb=KB)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args(argv)
    H = empty()
    n = 0
    for p in a.inputs:
        if a.merge:
            z = np.load(p)
            for k in ('H', 'HE', 'PW'):
                H[k] += z[k]
            H['n_events'] += int(z['n_events'])
            n += int(z['n_files'])
        else:
            try:
                reduce_file(p, H)
                n += 1
            except Exception as e:                      # a bad file must not kill the job
                print(f'skip {p}: {e!r}', file=sys.stderr)
    save(H, a.out, n)
    print(f'{n} files, {H["n_events"]:.3g} events -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
