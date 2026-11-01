"""Shared loading and the telescope -> det4 map for the small-region SPS analysis (sps_region/README.md)."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'sps_audit'))
from analyze import (seeded_fit, gauss_sigma, Z_BACK, Z_DET4, FID_MM, zone_of, pointing, ZONES, BINARY)  # noqa: E402,F401

#: slim telescope tables (tel_run{53,63,66,71}.parquet), footprint/ and timing/ copies of the EOS products
DATA = os.environ.get('SPS_REGION_DATA') or os.path.join(os.environ.get('WORK_DATA', os.path.expanduser('~/x17_work')), 'sps_region')
#: det4 FEU-3 channels that oscillate in every row (sps_audit README section 6) and their in-phase neighbours
HOT_Y_MM = {'ch 372-376 (oscillator + coupled)': (154.6, 158.7), 'ch 510 (oscillator)': (249.9, 250.8)}
#: instrumented window, both views (connectors 3-6)
WIN = (99.84, 299.52)


def load_tel(tag):
    d = pd.read_parquet(os.path.join(DATA, f'tel_{tag}.parquet'))
    tel = np.all([(d[f'{p}_n'] == 1).values for p in ('fx', 'fy', 'bx', 'by')], axis=0)
    return d[tel].reset_index(drop=True)


def telescope_at(d, z=Z_DET4):
    tx = (d['fx_p'] + (d['bx_p'] - d['fx_p']) * z / Z_BACK).values
    ty = (d['fy_p'] + (d['by_p'] - d['fy_p']) * z / Z_BACK).values
    return tx, ty


def align(d, est='c_pos'):
    """Seeded robust affine telescope ~ det4 (the sps_audit method) and its inverse: det4 (X, Y) predicted from the
    telescope for EVERY track, so tracks crossing a dead stripe land where they really crossed det4."""
    ex, ey = d[f'd4_x_{est}'].values, d[f'd4_y_{est}'].values
    ok = np.isfinite(ex) & np.isfinite(ey)
    tx, ty = telescope_at(d)
    M = np.column_stack([ex[ok], ey[ok], np.ones(ok.sum())])
    cx, rx = seeded_fit(M, tx[ok], ey[ok]); cy, ry = seeded_fit(M, ty[ok], ex[ok])
    RX = np.full(len(d), np.nan); RY = np.full(len(d), np.nan); RX[ok] = rx; RY[ok] = ry
    good = ok.copy(); good[ok] = (np.abs(rx) < FID_MM) & (np.abs(ry) < FID_MM)
    T = np.column_stack([tx, ty, np.ones(len(d))])
    bX, *_ = np.linalg.lstsq(T[good], ex[good], rcond=None); bY, *_ = np.linalg.lstsq(T[good], ey[good], rcond=None)
    return dict(px=T @ bX, py=T @ bY, rx=RX, ry=RY, cx=cx, cy=cy)
