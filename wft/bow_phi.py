"""Per-mounting spline angle map, the ``bow['phi']`` block of a calibration bundle (bench re-finalization, step 2).

What it is.  A tensor cubic B-spline potential Phi(U, W) whose gradient is the drift tilt of the two views,
    A_x = dPhi/dU + cx,    A_y = kappa dPhi/dW + cy,
and two scale maps S_x(U, W), S_y(U, W) (both on a coarser spline grid).  All of them are in tan units, so the block is
independent of the drift field of the row it is applied to.  The conversion (wft.reco.tan_bow / bow_pairs) is
    tan_p = (1e3 w - w0_p) / (kw_p v (1 + S_p(U, W))) - A_p(U, W),
with (U, W) = ((x - 200)/100, (y - 200)/100) at the reconstructed mid-gap position, as for the quadratic bow block.
When the block is present it REPLACES that block's tilt (``a_<p>`` / ``A_<p>``) and scale (``s_<p>``, and the bundle's
``v_map`` scale); the mesh height ``h`` stays (positions use it).

Domain.  The spline was fitted on the interior only (U in [-1.75, 1.75], W in [-1.55, 1.55] for chamber A, i.e. >= 30 mm
from the active edge).  A track outside is CLAMPED to the domain edge: the map is evaluated at the nearest domain point,
so the tilt and the scale are held at their edge values.  A non-finite (U, W) is not evaluated: the caller falls back
to the plain bow block for that event.

Layout of the block (JSON lists only; tensors are flattened with the U index running slowest, index = i * K + j):
    version          1
    units            'tan'
    degree           3
    K, KS            number of basis functions per axis for Phi and for the scale maps (KS = 0: constant scale)
    U_domain, W_domain   [lo, hi]
    knots_U, knots_W         full knot vectors of Phi (length K + 4)
    knots_S_U, knots_S_W     full knot vectors of the scale maps (length KS + 4), or None when KS = 0
    kappa            tan-unit weight of dPhi/dW in A_y (1.0 for maps converted from a2 with kappa = v_y/v_x)
    phi              K*K coefficients
    cx, cy           constant tilt of each view [tan]
    sx, sy           KS*KS coefficients of S_x, S_y (S, not 1 + S); a single coefficient when KS = 0
    provenance       free text / dict (mounting, rows, fit script), not used
    outside          optional: 'clamp' (default: a track outside the domain is evaluated at the nearest domain point) or
                     'block' (interior only: outside the domain the view keeps the plain bow block, with the constants of
                     ``fallback``)
    domain_x, domain_y   optional, per view: {'U': [lo, hi], 'W': [lo, hi]}, the region where that view uses the map when
                     outside = 'block' (default U_domain, W_domain).  Chamber D's X view: U >= 0 (x >= 200 mm).
    fallback         required with outside = 'block': {'w0': {'x', 'y'}, 'kw': {'x', 'y'}} [um/ns, as in the bundle], the
                     per-row constants the plain bow block was calibrated with (pass 1).  The bundle's own w0, kw are the
                     ones measured WITH the map, so they cannot be used for the plain block.  v_drift is common.
The constants cx, cy and the mean of S are absorbed by the per-row w0 and kw, which are measured after the map is applied.
"""
import numpy as np
from scipy.interpolate import BSpline

PHI_VERSION = 1
_cache = {}


def _spl(t, K, deg, nu):
    key = (tuple(np.round(t, 12)), K, deg, nu)
    f = _cache.get(key)
    if f is None:
        f = BSpline(np.asarray(t, float), np.eye(K), deg)
        if nu:
            f = f.derivative(nu)
        if len(_cache) > 64:
            _cache.clear()
        _cache[key] = f
    return f


def _basis(x, t, K, deg, lo, hi, nu=0):
    """(n, K) basis (or its nu-th derivative) at x clamped to [lo, hi]."""
    return np.atleast_2d(_spl(t, K, deg, nu)(np.clip(x, lo, hi))).reshape(len(np.atleast_1d(x)), K)


def validate(phi: dict, where: str = '') -> None:
    """Raise ValueError for a malformed block (called when a bundle is loaded or saved)."""
    tag = f'bow[phi]{" in " + where if where else ""}'
    if int(phi.get('version', 0)) != PHI_VERSION:
        raise ValueError(f'{tag}: version {phi.get("version")!r}, this code reads {PHI_VERSION}')
    if phi.get('units', 'tan') != 'tan':
        raise ValueError(f'{tag}: units must be tan')
    deg, K, KS = int(phi.get('degree', 3)), int(phi['K']), int(phi.get('KS', 0))
    for k in ('U_domain', 'W_domain'):
        lo, hi = phi[k]
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(f'{tag}: bad {k}')
    for nm, kk, n in (('knots_U', K, K + deg + 1), ('knots_W', K, K + deg + 1)):
        if len(phi[nm]) != n:
            raise ValueError(f'{tag}: {nm} has {len(phi[nm])} entries, expected {n}')
    if len(phi['phi']) != K * K:
        raise ValueError(f'{tag}: phi has {len(phi["phi"])} coefficients, expected {K * K}')
    if KS:
        for nm in ('knots_S_U', 'knots_S_W'):
            if not phi.get(nm) or len(phi[nm]) != KS + deg + 1:
                raise ValueError(f'{tag}: {nm} must have {KS + deg + 1} entries')
        for nm in ('sx', 'sy'):
            if len(phi[nm]) != KS * KS:
                raise ValueError(f'{tag}: {nm} has {len(phi[nm])} coefficients, expected {KS * KS}')
    else:
        for nm in ('sx', 'sy'):
            if len(phi[nm]) != 1:
                raise ValueError(f'{tag}: {nm} must hold one coefficient when KS = 0')
    for nm in ('phi', 'sx', 'sy'):
        if not np.all(np.isfinite(phi[nm])):
            raise ValueError(f'{tag}: non-finite {nm}')
    out = phi.get('outside', 'clamp')
    if out not in ('clamp', 'block'):
        raise ValueError(f"{tag}: outside must be 'clamp' or 'block', not {out!r}")
    for v in 'xy':
        d = phi.get(f'domain_{v}')
        if d is not None:
            for k in ('U', 'W'):
                lo, hi = d[k]
                if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                    raise ValueError(f'{tag}: bad domain_{v}[{k}]')
    if out == 'block':
        fb = phi.get('fallback') or {}
        for k in ('w0', 'kw'):
            for v in 'xy':
                if not np.isfinite(float((fb.get(k) or {}).get(v, np.nan))):
                    raise ValueError(f"{tag}: outside = 'block' needs fallback[{k}][{v}]")


def evaluate(phi: dict, U, W) -> tuple:
    """(A_x, A_y, S_x, S_y) at mid-gap positions (U, W) (arrays or scalars; returns arrays of the broadcast length).
    Positions are clamped to the domain; NaN stays NaN."""
    U = np.atleast_1d(np.asarray(U, float)); W = np.atleast_1d(np.asarray(W, float))
    U, W = np.broadcast_arrays(U, W)
    deg, K, KS = int(phi.get('degree', 3)), int(phi['K']), int(phi.get('KS', 0))
    (ulo, uhi), (wlo, whi) = phi['U_domain'], phi['W_domain']
    tU, tW = phi['knots_U'], phi['knots_W']
    C = np.asarray(phi['phi'], float).reshape(K, K)
    Bu, Bw = _basis(U, tU, K, deg, ulo, uhi), _basis(W, tW, K, deg, wlo, whi)
    dBu, dBw = _basis(U, tU, K, deg, ulo, uhi, 1), _basis(W, tW, K, deg, wlo, whi, 1)
    ax = np.einsum('ni,ij,nj->n', dBu, C, Bw) + phi['cx']
    ay = float(phi.get('kappa', 1.0)) * np.einsum('ni,ij,nj->n', Bu, C, dBw) + phi['cy']
    if KS == 0:
        sx = np.full(len(U), float(phi['sx'][0])); sy = np.full(len(U), float(phi['sy'][0]))
    else:
        Su = _basis(U, phi['knots_S_U'], KS, deg, ulo, uhi); Sw = _basis(W, phi['knots_S_W'], KS, deg, wlo, whi)
        sx = np.einsum('ni,ij,nj->n', Su, np.asarray(phi['sx'], float).reshape(KS, KS), Sw)
        sy = np.einsum('ni,ij,nj->n', Su, np.asarray(phi['sy'], float).reshape(KS, KS), Sw)
    return ax, ay, sx, sy


def interior_only(phi: dict) -> bool:
    """True when a track outside the domain keeps the plain bow block (``outside`` = 'block')."""
    return phi.get('outside', 'clamp') == 'block'


def in_domain(phi: dict, U, W, view: str):
    """Boolean array: (U, W) lies inside the region where ``view`` ('x' or 'y') uses the map (closed interval; NaN is
    outside).  Per-view ``domain_<view>`` if the block has it, else U_domain and W_domain."""
    d = phi.get(f'domain_{view}') or {}
    (ulo, uhi), (wlo, whi) = d.get('U', phi['U_domain']), d.get('W', phi['W_domain'])
    U = np.asarray(U, float)
    W = np.asarray(W, float)
    with np.errstate(invalid='ignore'):
        return (U >= ulo) & (U <= uhi) & (W >= wlo) & (W <= whi)


def from_a2_map(M: dict, provenance=None) -> dict:
    """Convert an a2_final map (rerun/paper1_gap/audit/results/a2_final_map_*.json, um/ns) to a ``bow['phi']`` block in tan.

    a2 models w = a(U, W) + tan s(U, W) with s = v (1 + S).  With v_p = M['v_used'][p] (the robust ladder slope of the
    training set) the tan-unit maps are A_p = a_p / v_p and S_p = s_p / v_p - 1, so  A_p * v_p == a2 a_p  and
    (1 + S_p) * v_p == a2 s_p  to rounding (B-splines sum to 1, so the -1 is subtracted from every S coefficient).
    Phi is divided by v_x, and kappa becomes kappa_a2 v_x / v_y (= 1 to 1e-4 for the prototype, since a2 fixes
    kappa = v_y / v_x from the same pilot fit)."""
    vx, vy = float(M['v_used']['x']), float(M['v_used']['y'])
    KS = int(M['KS'])
    blk = dict(version=PHI_VERSION, units='tan', degree=int(M.get('degree', 3)), K=int(M['K']), KS=KS,
               U_domain=list(M['U_domain']), W_domain=list(M['W_domain']),
               knots_U=list(M['knots_U']), knots_W=list(M['knots_W']),
               knots_S_U=(list(M['knots_S_U']) if KS else None), knots_S_W=(list(M['knots_S_W']) if KS else None),
               kappa=float(M['kappa']) * vx / vy,
               phi=(np.asarray(M['phi'], float) / vx).tolist(), cx=float(M['cx']) / vx, cy=float(M['cy']) / vy,
               sx=(np.asarray(M['sx'], float) / vx - 1.0).tolist(), sy=(np.asarray(M['sy'], float) / vy - 1.0).tolist(),
               provenance=provenance if provenance is not None else dict(source='a2_final map', v_used=M['v_used']))
    validate(blk)
    return blk
