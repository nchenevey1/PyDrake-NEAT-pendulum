"""
Rigorous region-of-attraction certificate for a discrete-time closed loop x+ = F(x).

Coordinates. V(x) = |z|^2 with z = W (x - x*), W an arbitrary float matrix (V = e^T W^T W e).
x(z) = x* + W^{-1} z, and W^{-1} is enclosed entry-wise by an interval matrix computed with
exact rational arithmetic (`fractions`). In z-coordinates the next state is y(z) = W (F(x(z)) - x*)
and the one-step difference is, *exactly*, for any float matrix A_hat,

    dV(z) = |y(z)|^2 - |z|^2 = z^T N z + 2 (A_hat z)^T r(z) + |r(z)|^2,
    N = A_hat^T A_hat - I,     r(z) = y(z) - A_hat z.

A_hat is chosen as the linearisation in z-coordinates, and W so that N is (nearly) diagonal;
this makes the quadratic part tight. On a box Z = c + [-h, h] the remainder is enclosed by
the mean-value form  r(Z) in r(c) + (W J_F(X) W^{-1} - A_hat) [-h, h],  with X an interval
hull of x(Z) and J_F(X) the interval Jacobian (interval.run_prog).

Claim (all three parts are checked):
  (a) dV(z) < 0 for every z with eps^2 < |z|^2 <= rho          [branch and bound over boxes]
  (b) on the ball |z| <= eps, F is a contraction in |.| that maps the ball into itself
      (|y(0)| + kappa eps <= eps, kappa = sup ||D y||_2 < 1)     [interval Jacobian + exact PD test]
  (c) {|z|^2 <= rho} lies inside the upright set (|th1 - pi| < tol, |th2| < tol) and the track
      (|x| <= limit_x)                                              [interval bound of W^{-1}]
Then every trajectory starting in Omega = {|z|^2 <= rho} stays in Omega (so it is upright and on
the track at every step) and converges to the unique fixed point inside the eps-ball.

The branch and bound lowers rho whenever it meets a box it cannot verify (down to the box's
minimum |z|^2), and lowers it to just below |c|^2 when the interval bound proves dV(c) >= 0 at a
box center (a genuine counterexample for that level). The final rho is the largest level for
which every box meeting the sublevel set was verified.
"""

from dataclasses import dataclass, field, asdict
from fractions import Fraction
import math
import time

import numba as nb
import numpy as np
import scipy.linalg as sl

from . import interval as I
from .model import ControllerData, jacobian_fd, phi_gamma, trace_closed_loop

dn, up, iadd, isub, imul, isqr = I.dn, I.up, I.iadd, I.isub, I.imul, I.isqr


# ============================================================================ exact helpers

def exact_inverse_enclosure(W):
    """Entry-wise interval enclosure [lo, hi] of W^{-1}, W a float matrix (exact rationals)."""
    n = W.shape[0]
    M = [[Fraction(float(W[i, j])) for j in range(n)] + [Fraction(int(i == j)) for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[piv][col] == 0:
            raise ValueError("singular W")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    inv = [[M[i][n + j] for j in range(n)] for i in range(n)]
    lo = np.array([[_fdown(v) for v in row] for row in inv])
    hi = np.array([[_fup(v) for v in row] for row in inv])
    return lo, hi


def _fdown(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) <= q else math.nextafter(f, -math.inf)


def _fup(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) >= q else math.nextafter(f, math.inf)


def exact_is_pd(S) -> bool:
    """Exact test that the symmetric rational matrix S (list of lists of Fractions) is PD."""
    n = len(S)
    A = [[Fraction(S[i][j]) for j in range(n)] for i in range(n)]
    for k in range(n):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= f * A[k][j]
    return True


def exact_spectral_norm_bound(M, s: float) -> bool:
    """Exact proof that ||M||_2 < s for a float matrix M: s^2 I - M^T M is PD."""
    n = M.shape[1]
    F = [[Fraction(float(v)) for v in row] for row in M]
    S = [[(Fraction(s) ** 2 if i == j else 0) - sum(F[k][i] * F[k][j] for k in range(len(F)))
          for j in range(n)] for i in range(n)]
    return exact_is_pd(S)


# ============================================================================ numba kernels

@nb.njit(cache=True)
def _imatvec_affine(xs, Mlo, Mhi, zlo, zhi, Xlo, Xhi):
    """X = xs + M Z (interval M, interval Z)."""
    n = xs.shape[0]
    for i in range(n):
        s0 = xs[i]; s1 = xs[i]
        for j in range(Mlo.shape[1]):
            p0, p1 = imul(Mlo[i, j], Mhi[i, j], zlo[j], zhi[j])
            s0, s1 = iadd(s0, s1, p0, p1)
        Xlo[i] = s0; Xhi[i] = s1


@nb.njit(cache=True)
def _imatmul(Alo, Ahi, Blo, Bhi, Clo, Chi):
    n, m = Alo.shape
    p = Blo.shape[1]
    for i in range(n):
        for k in range(p):
            s0 = 0.0; s1 = 0.0
            for j in range(m):
                q0, q1 = imul(Alo[i, j], Ahi[i, j], Blo[j, k], Bhi[j, k])
                s0, s1 = iadd(s0, s1, q0, q1)
            Clo[i, k] = s0; Chi[i, k] = s1


@nb.njit(cache=True)
def eval_box(D, zlo, zhi, V, VC, G, res):
    """Bounds on dV over the z-box [zlo, zhi] (see module docstring).

    Program outputs: u (cart force), m = M(q)^{-1} e_0, p = accelerations at u = 0, so a = m u + p.
    With float m0, gu, Jp (linearisation at z = 0) and Jz = m0 gu + Jp:
        s(z) = a(x(z)) - Jz z = m0 su(z) + (m(z) - m0) u(z) + sp(z),
        su = u - gu z,   sp = p - Jp z   (enclosed by mean-value forms),
        dV = z^T N z + 2 (A_hat z)^T E z + 2 (bm z) su + 2 (B z)^T [(m - m0) u + sp] + |E z + WG s|^2
    with B = WG^T A_hat, bm = m0^T B. Keeping the control channel u scalar preserves the strong
    cancellation in W Gamma m0 (the input direction in z-coordinates).

    res[0] = upper bound of dV over the box, res[1] = lower bound of dV at the box center,
    res[2] = lower bound of |c|^2, res[3] = dimension to split."""
    (ops, cst, out, xs, Mlo, Mhi, Ah, Nlo, Nhi, Elo, Ehi, WGlo, WGhi, Blo, Bhi, WPMlo, WPMhi,
     m0, gu, Jp, bmlo, bmhi) = D
    n = 6
    c = np.empty(n); h = np.empty(n)
    for j in range(n):
        c[j] = 0.5 * (zlo[j] + zhi[j])
        h[j] = max(up(zhi[j] - c[j]), up(c[j] - zlo[j]))
    Xlo = np.empty(n); Xhi = np.empty(n)
    _imatvec_affine(xs, Mlo, Mhi, zlo, zhi, Xlo, Xhi)
    Clo = np.empty(n); Chi = np.empty(n)
    _imatvec_affine(xs, Mlo, Mhi, c, c, Clo, Chi)
    I.run_prog_mv(ops, cst, Xlo, Xhi, Clo, Chi, h, V, VC, G, Mlo, Mhi)
    ou = out[0]
    # su over the box
    su0 = VC[ou, 0]; su1 = VC[ou, 1]
    for j in range(n):
        p0, p1 = imul(gu[j], gu[j], c[j], c[j])
        su0, su1 = isub(su0, su1, p0, p1)
    for j in range(n):
        g0, g1 = isub(G[ou, j, 0], G[ou, j, 1], gu[j], gu[j])
        p0, p1 = imul(g0, g1, -h[j], h[j])
        su0, su1 = iadd(su0, su1, p0, p1)
    U0 = V[ou, 0]; U1 = V[ou, 1]
    # t_k = (m_k - m0_k) u + sp_k ;  s_k = m0_k su + t_k
    T0 = np.empty(3); T1 = np.empty(3); S0 = np.empty(3); S1 = np.empty(3)
    ac0 = np.empty(3); ac1 = np.empty(3)
    for k in range(3):
        om = out[1 + k]; op = out[4 + k]
        d0, d1 = isub(V[om, 0], V[om, 1], m0[k], m0[k])
        t0, t1 = imul(d0, d1, U0, U1)
        q0 = VC[op, 0]; q1 = VC[op, 1]
        for j in range(n):
            p0, p1 = imul(Jp[k, j], Jp[k, j], c[j], c[j])
            q0, q1 = isub(q0, q1, p0, p1)
        for j in range(n):
            g0, g1 = isub(G[op, j, 0], G[op, j, 1], Jp[k, j], Jp[k, j])
            p0, p1 = imul(g0, g1, -h[j], h[j])
            q0, q1 = iadd(q0, q1, p0, p1)
        t0, t1 = iadd(t0, t1, q0, q1)
        T0[k] = t0; T1[k] = t1
        p0, p1 = imul(m0[k], m0[k], su0, su1)
        S0[k], S1[k] = iadd(p0, p1, t0, t1)
        # a(c) = m(c) u(c) + p(c)
        p0, p1 = imul(VC[om, 0], VC[om, 1], VC[ou, 0], VC[ou, 1])
        ac0[k], ac1[k] = iadd(p0, p1, VC[op, 0], VC[op, 1])
    acc0 = 0.0; acc1 = 0.0
    # 2 (bm z) su
    v0 = 0.0; v1 = 0.0
    for j in range(n):
        p0, p1 = imul(bmlo[j], bmhi[j], zlo[j], zhi[j])
        v0, v1 = iadd(v0, v1, p0, p1)
    p0, p1 = imul(v0, v1, su0, su1)
    acc0, acc1 = iadd(acc0, acc1, 2.0 * p0, 2.0 * p1)       # *2 is exact
    # 2 (B z)^T t
    for k in range(3):
        v0 = 0.0; v1 = 0.0
        for j in range(n):
            p0, p1 = imul(Blo[k, j], Bhi[k, j], zlo[j], zhi[j])
            v0, v1 = iadd(v0, v1, p0, p1)
        p0, p1 = imul(v0, v1, T0[k], T1[k])
        acc0, acc1 = iadd(acc0, acc1, 2.0 * p0, 2.0 * p1)
    # 2 (A_hat z)^T E z + |E z + WG s|^2
    for i in range(n):
        a0 = 0.0; a1 = 0.0; e0 = 0.0; e1 = 0.0
        for j in range(n):
            p0, p1 = imul(Ah[i, j], Ah[i, j], zlo[j], zhi[j])
            a0, a1 = iadd(a0, a1, p0, p1)
            p0, p1 = imul(Elo[i, j], Ehi[i, j], zlo[j], zhi[j])
            e0, e1 = iadd(e0, e1, p0, p1)
        p0, p1 = imul(a0, a1, e0, e1)
        acc0, acc1 = iadd(acc0, acc1, 2.0 * p0, 2.0 * p1)
        r0 = e0; r1 = e1
        for k in range(3):
            p0, p1 = imul(WGlo[i, k], WGhi[i, k], S0[k], S1[k])
            r0, r1 = iadd(r0, r1, p0, p1)
        q0, q1 = isqr(r0, r1)
        acc0, acc1 = iadd(acc0, acc1, q0, q1)
    # z^T N z
    for i in range(n):
        for j in range(n):
            if i == j:
                q0, q1 = isqr(zlo[i], zhi[i])
            else:
                q0, q1 = imul(zlo[i], zhi[i], zlo[j], zhi[j])
            p0, p1 = imul(Nlo[i, j], Nhi[i, j], q0, q1)
            acc0, acc1 = iadd(acc0, acc1, p0, p1)
    # dV at the center: y(c) = W Phi W^{-1} c + WG a(x(c))
    dv0 = 0.0; dv1 = 0.0
    for i in range(n):
        y0 = 0.0; y1 = 0.0
        for j in range(n):
            p0, p1 = imul(WPMlo[i, j], WPMhi[i, j], c[j], c[j])
            y0, y1 = iadd(y0, y1, p0, p1)
        for k in range(3):
            p0, p1 = imul(WGlo[i, k], WGhi[i, k], ac0[k], ac1[k])
            y0, y1 = iadd(y0, y1, p0, p1)
        q0, q1 = isqr(y0, y1)
        dv0, dv1 = iadd(dv0, dv1, q0, q1)
    c20 = 0.0; c21 = 0.0
    for i in range(n):
        q0, q1 = isqr(c[i], c[i])
        c20, c21 = iadd(c20, c21, q0, q1)
    dv0, dv1 = isub(dv0, dv1, c20, c21)
    # split hint: largest first-order contribution of h_j to the bound
    vb = 0.0
    for i in range(n):
        vb += abs(bmlo[i]) * (abs(c[i]) + h[i])
    best = -1.0; bj = 0
    for j in range(n):
        gw = max(abs(G[ou, j, 0] - gu[j]), abs(G[ou, j, 1] - gu[j]))
        sc = h[j] * (2.0 * vb * gw + abs(Nlo[j, j]) * (abs(c[j]) + h[j]))
        if sc > best:
            best = sc; bj = j
    res[0] = acc1 if acc1 == acc1 else np.inf
    res[1] = dv0 if dv0 == dv0 else -np.inf
    res[2] = c20
    res[3] = bj
    return 0


@nb.njit(cache=True)
def _norm2_bounds(zlo, zhi):
    mn = 0.0; mx = 0.0
    for j in range(zlo.shape[0]):
        lo = zlo[j]; hi = zhi[j]
        if lo > 0.0:
            a = lo
        elif hi < 0.0:
            a = -hi
        else:
            a = 0.0
        b = max(abs(lo), abs(hi))
        mn = mn + a * a
        mx = mx + b * b
    # 12 roundings to nearest, each relative <= 2^-53: widen by a relative 1e-15
    return mn * (1.0 - 1e-15), mx * (1.0 + 1e-15) + 1e-300


@nb.njit(cache=True)
def bnb_one(D, blo, bhi, rho0, eps2, minw, max_boxes, stats, worst):
    """Branch and bound on one initial box. Returns the certified level (<= rho0), or -1 if the
    box budget ran out. stats: [evaluated, verified, outside, in_ball, failed_small, counterex].
    worst: center of the box that set the final rho (diagnostic)."""
    ops = D[0]
    n = 6
    nv = ops.shape[0]
    V = np.zeros((nv, 2)); VC = np.zeros((nv, 2)); G = np.zeros((nv, n, 2))
    res = np.zeros(4)
    cap = 4096
    SL = np.empty((cap, n)); SH = np.empty((cap, n))
    SL[0] = blo; SH[0] = bhi; sp = 1
    rho = rho0
    zlo = np.empty(n); zhi = np.empty(n)
    while sp > 0:
        sp -= 1
        zlo[:] = SL[sp]; zhi[:] = SH[sp]
        mn2, mx2 = _norm2_bounds(zlo, zhi)
        if mn2 > rho:
            stats[2] += 1
            continue
        if mx2 <= eps2:
            stats[3] += 1
            continue
        if stats[0] >= max_boxes:
            return -1.0
        stats[0] += 1
        eval_box(D, zlo, zhi, V, VC, G, res)
        if res[0] < 0.0:
            stats[1] += 1
            continue
        if res[1] >= 0.0:
            # proven dV(c) >= 0 at the center: the level must stay below |c|^2
            stats[5] += 1
            r2 = res[2] * (1.0 - 1e-15)
            if r2 < rho:
                rho = r2
                for j in range(n):
                    worst[j] = 0.5 * (zlo[j] + zhi[j])
        wmax = 0.0
        for j in range(n):
            wmax = max(wmax, zhi[j] - zlo[j])
        if wmax < minw:
            stats[4] += 1
            if mn2 < rho:
                rho = mn2
                for j in range(n):
                    worst[j] = 0.5 * (zlo[j] + zhi[j])
            continue
        j = int(res[3])
        if zhi[j] - zlo[j] < 0.25 * wmax:     # keep boxes from getting too thin
            for k in range(n):
                if zhi[k] - zlo[k] == wmax:
                    j = k
                    break
        m = 0.5 * (zlo[j] + zhi[j])
        if sp + 2 > cap:
            return -1.0
        SL[sp] = zlo; SH[sp] = zhi; SH[sp, j] = m; sp += 1
        SL[sp] = zlo; SH[sp] = zhi; SL[sp, j] = m; sp += 1
    return rho


@nb.njit(cache=True, parallel=True)
def bnb_many(D, BL, BH, rho0, eps2, minw, max_boxes):
    K = BL.shape[0]
    rhos = np.empty(K)
    stats = np.zeros((K, 6))
    worst = np.zeros((K, 6))
    for k in nb.prange(K):
        st = np.zeros(6)
        wv = np.zeros(6)
        rhos[k] = bnb_one(D, BL[k], BH[k], rho0, eps2, minw, max_boxes, st, wv)
        stats[k] = st
        worst[k] = wv
    return rhos, stats, worst


@nb.njit(cache=True)
def interval_N(Ah, Nlo, Nhi):
    n = Ah.shape[0]
    for i in range(n):
        for j in range(n):
            s0 = 0.0; s1 = 0.0
            for k in range(n):
                p0, p1 = imul(Ah[k, i], Ah[k, i], Ah[k, j], Ah[k, j])
                s0, s1 = iadd(s0, s1, p0, p1)
            if i == j:
                s0, s1 = isub(s0, s1, 1.0, 1.0)
            Nlo[i, j] = s0; Nhi[i, j] = s1


# ============================================================================ setup / driver

@dataclass
class Setup:
    """Everything that defines a certificate candidate (all float arrays are exact data)."""
    xs: np.ndarray        # (6,) expansion point (float approximation of the fixed point)
    W: np.ndarray         # (6, 6) V(x) = |W (x - xs)|^2
    Ah: np.ndarray        # (6, 6) linear part in z-coordinates
    g: object = None      # physics intervals (robustness); None = nominal
    masses: object = None
    force_dist: object = None

    def to_dict(self):
        return {"xs": self.xs.tolist(), "W": self.W.tolist(), "Ah": self.Ah.tolist(),
                "g": self.g, "masses": self.masses, "force_dist": self.force_dist}

    @staticmethod
    def from_dict(d):
        tup = lambda v: None if v is None else (tuple(v) if not isinstance(v[0], (list, tuple))
                                                else tuple(tuple(x) for x in v))
        masses = d.get("masses")
        if masses is not None:
            masses = tuple(tuple(m) if isinstance(m, (list, tuple)) else m for m in masses)
        return Setup(np.array(d["xs"]), np.array(d["W"]), np.array(d["Ah"]), tup(d.get("g")),
                     masses, tup(d.get("force_dist")))


def jacobian_ad(cd: ControllerData, xs):
    """Jacobian of F at xs from the interval program (midpoints of a thin-box evaluation)."""
    p = trace_closed_loop(cd)
    ops, cst, out = p.arrays()
    V = np.zeros((len(ops), 2)); G = np.zeros((len(ops), 6, 2))
    E = np.eye(6)
    xs = np.asarray(xs, dtype=np.float64)
    I.run_prog(ops, cst, xs, xs, 6, V, G, E, E)
    mid = lambda k: 0.5 * (V[k, 0] + V[k, 1])
    gmid = lambda k: 0.5 * (G[k, :, 0] + G[k, :, 1])
    u = mid(out[0])
    Ja = np.array([mid(out[1 + k]) * gmid(out[0]) + u * gmid(out[1 + k]) + gmid(out[4 + k])
                   for k in range(3)])
    Phi = phi_gamma(cd.dt)
    Gam = np.vstack([cd.dt ** 2 * np.eye(3), cd.dt * np.eye(3)])
    return Phi + Gam @ Ja


def lyapunov_setup(cd: ControllerData, xs, Q, beta: float = 1.0, A=None) -> Setup:
    """V from the shifted discrete Lyapunov equation (A/beta)^T P (A/beta) - P = -Q, i.e.
    A^T P A - P = -(1 - beta^2) P - beta^2 Q: for spectral radius(A) < beta < 1 this guarantees a
    linear decrease rate of at least 1 - beta^2 in every direction (beta = 1: plain Lyapunov).
    Returned in coordinates where N = A_hat^T A_hat - I is diagonal (z = U^T L^T e, P = L L^T)."""
    A = jacobian_ad(cd, xs) if A is None else A
    P = sl.solve_discrete_lyapunov((A / beta).T, Q)
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P)
    At = L.T @ A @ np.linalg.inv(L.T)
    lam, U = np.linalg.eigh(At.T @ At)
    W = U.T @ L.T
    Ah = U.T @ At @ U
    return Setup(np.asarray(xs, dtype=np.float64), W, Ah)


@dataclass
class Prepared:
    setup: Setup
    ops: np.ndarray
    cst: np.ndarray
    out: np.ndarray
    Mlo: np.ndarray
    Mhi: np.ndarray
    data: tuple            # kernel data tuple (see eval_box)
    y0: tuple              # enclosure of y(0) = W (F(x*) - x*)


def prepare(cd: ControllerData, st: Setup) -> Prepared:
    p = trace_closed_loop(cd, g=st.g, masses=st.masses, force_dist=st.force_dist)
    ops, cst, out = p.arrays()
    xs = np.ascontiguousarray(st.xs, dtype=np.float64)
    W = np.ascontiguousarray(st.W, dtype=np.float64)
    Ah = np.ascontiguousarray(st.Ah, dtype=np.float64)
    Mlo, Mhi = exact_inverse_enclosure(W)
    dt = cd.dt
    dt2lo, dt2hi = isqr(dt, dt)
    # Phi, Gamma as interval matrices: F(x) = Phi x + Gamma a(x)
    Phlo = np.eye(6); Phi_hi = np.eye(6)
    for k in range(3):
        Phlo[k, k + 3] = Phi_hi[k, k + 3] = dt
    Gmlo = np.zeros((6, 3)); Gmhi = np.zeros((6, 3))
    for k in range(3):
        Gmlo[k, k], Gmhi[k, k] = dt2lo, dt2hi
        Gmlo[k + 3, k] = Gmhi[k + 3, k] = dt
    WG0 = np.zeros((6, 3)); WG1 = np.zeros((6, 3))
    _imatmul(W, W, Gmlo, Gmhi, WG0, WG1)
    WP0 = np.zeros((6, 6)); WP1 = np.zeros((6, 6))
    _imatmul(W, W, Phlo, Phi_hi, WP0, WP1)
    WPM0 = np.zeros((6, 6)); WPM1 = np.zeros((6, 6))
    _imatmul(WP0, WP1, Mlo, Mhi, WPM0, WPM1)
    # linearisation at z = 0 (any float values are valid; midpoints are tight)
    V = np.zeros((len(ops), 2)); G = np.zeros((len(ops), 6, 2))
    I.run_prog(ops, cst, xs, xs, 6, V, G, Mlo, Mhi)
    gmid = lambda k: np.ascontiguousarray(0.5 * (G[k, :, 0] + G[k, :, 1]))
    m0 = np.array([0.5 * (V[out[1 + k], 0] + V[out[1 + k], 1]) for k in range(3)])
    gu = gmid(out[0])
    Jp = np.ascontiguousarray(np.array([gmid(out[4 + k]) for k in range(3)]))
    # a(x*) = m u + p  ->  y(0) = WG a(x*)
    a0lo = np.zeros((3, 1)); a0hi = np.zeros((3, 1))
    for k in range(3):
        q0, q1 = imul(V[out[1 + k], 0], V[out[1 + k], 1], V[out[0], 0], V[out[0], 1])
        a0lo[k, 0], a0hi[k, 0] = iadd(q0, q1, V[out[4 + k], 0], V[out[4 + k], 1])
    y0lo = np.zeros((6, 1)); y0hi = np.zeros((6, 1))
    _imatmul(WG0, WG1, a0lo, a0hi, y0lo, y0hi)
    # Jz = m0 gu + Jp ; A_true = W Phi W^{-1} + WG Jz ; E = A_true - A_hat
    Jzlo = np.zeros((3, 6)); Jzhi = np.zeros((3, 6))
    for k in range(3):
        for j in range(6):
            q0, q1 = imul(m0[k], m0[k], gu[j], gu[j])
            Jzlo[k, j], Jzhi[k, j] = iadd(q0, q1, Jp[k, j], Jp[k, j])
    T0 = np.zeros((6, 6)); T1 = np.zeros((6, 6))
    _imatmul(WG0, WG1, Jzlo, Jzhi, T0, T1)
    Elo = np.zeros((6, 6)); Ehi = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            s0, s1 = iadd(WPM0[i, j], WPM1[i, j], T0[i, j], T1[i, j])
            Elo[i, j], Ehi[i, j] = isub(s0, s1, Ah[i, j], Ah[i, j])
    # B = WG^T A_hat ; bm = m0^T B
    Blo = np.zeros((3, 6)); Bhi = np.zeros((3, 6))
    _imatmul(np.ascontiguousarray(WG0.T), np.ascontiguousarray(WG1.T), Ah, Ah, Blo, Bhi)
    bmlo = np.zeros((1, 6)); bmhi = np.zeros((1, 6))
    m0r = m0.reshape(1, 3).copy()
    _imatmul(m0r, m0r, Blo, Bhi, bmlo, bmhi)
    Nlo = np.zeros((6, 6)); Nhi = np.zeros((6, 6))
    interval_N(Ah, Nlo, Nhi)
    data = (ops, cst, out, xs, Mlo, Mhi, Ah, Nlo, Nhi, Elo, Ehi, WG0, WG1, Blo, Bhi, WPM0, WPM1,
            m0, gu, Jp, bmlo[0].copy(), bmhi[0].copy())
    return Prepared(st, ops, cst, out, Mlo, Mhi, data, (y0lo[:, 0], y0hi[:, 0]))


def rho_safe(pp: Prepared, task_limits) -> float:
    """Largest rho (rounded down) with {|z|^2 <= rho} inside |x| <= lx, |th1 - pi| < tol,
    |th2| < tol. max_{|z|^2<=rho} |e_i| = sqrt(rho) * ||row_i(W^{-1})||_2."""
    lx, tol = task_limits
    xs = pp.setup.xs
    rn = np.sqrt(np.sum(np.maximum(np.abs(pp.Mlo), np.abs(pp.Mhi)) ** 2, axis=1)) * (1 + 1e-14)
    pi_err = abs(Fraction(math.pi) - Fraction(xs[1]))  # |x*_1 - pi| (pi irrational: bound below)
    pi_err = float(pi_err) + 1e-15
    room = np.array([min(lx - xs[0], lx + xs[0]) * (1 - 1e-14),
                     (tol - pi_err - abs(xs[1] - math.pi)) * (1 - 1e-14),
                     (tol - abs(xs[2])) * (1 - 1e-14)])
    if np.any(room <= 0):
        return 0.0
    r = (room / rn[:3]) ** 2 * (1 - 1e-12)
    return float(np.min(r))


def ball_check(pp: Prepared, eps: float):
    """Part (b): returns (ok, kappa, |y(0)|).

    On the box [-eps, eps]^6 (which contains the eps-ball) D y = A_hat + E + WG D s with
    D s = m0 (D u - gu) + (m - m0) D u + u D m + (D p - Jp), so
    ||D y||_2 <= ||A_hat||_2 + ||E||_F + |WG m0| |D u - gu| + ||WG||_F ||rest||_F = kappa
    (exact PD test for ||A_hat||_2). y(0) = WG a(x*) since x(0) = x* exactly."""
    (ops, cst, out, xs, Mlo, Mhi, Ah, Nlo, Nhi, Elo, Ehi, WG0, WG1, Blo, Bhi, _, _,
     m0, gu, Jp, _, _) = pp.data
    V = np.zeros((len(ops), 2)); VC = np.zeros((len(ops), 2)); G = np.zeros((len(ops), 6, 2))
    zl = np.full(6, -eps); zh = np.full(6, eps)
    Xlo = np.zeros(6); Xhi = np.zeros(6)
    _imatvec_affine(xs, Mlo, Mhi, zl, zh, Xlo, Xhi)
    I.run_prog_mv(ops, cst, Xlo, Xhi, xs, xs, np.full(6, eps), V, VC, G, Mlo, Mhi)
    amax = lambda lo, hi: np.maximum(np.abs(lo), np.abs(hi))
    fro = lambda A: float(np.sqrt(np.sum(A ** 2))) * (1 + 1e-12)
    ou = out[0]
    Du = amax(G[ou, :, 0] - gu, G[ou, :, 1] - gu) * (1 + 1e-14) + 1e-300
    U = max(abs(V[ou, 0]), abs(V[ou, 1]))
    rest = np.zeros((3, 6))
    for k in range(3):
        om, op = out[1 + k], out[4 + k]
        dm = max(abs(V[om, 0] - m0[k]), abs(V[om, 1] - m0[k]))
        rest[k] = (dm * amax(G[ou, :, 0], G[ou, :, 1]) + U * amax(G[om, :, 0], G[om, :, 1])
                   + amax(G[op, :, 0] - Jp[k], G[op, :, 1] - Jp[k])) * (1 + 1e-14) + 1e-300
    if not (np.all(np.isfinite(Du)) and np.all(np.isfinite(rest))):
        return False, np.inf, np.inf
    WGm = np.zeros((6, 1)); WGm1 = np.zeros((6, 1))
    _imatmul(WG0, WG1, m0.reshape(3, 1).copy(), m0.reshape(3, 1).copy(), WGm, WGm1)
    s = float(np.linalg.norm(Ah, 2)) * (1 + 1e-12) + 1e-300
    if not exact_spectral_norm_bound(Ah, s):
        return False, np.inf, np.inf
    kappa = (s + fro(amax(Elo, Ehi)) + fro(amax(WGm, WGm1)) * fro(Du)
             + fro(amax(WG0, WG1)) * fro(rest)) * (1 + 1e-12)
    y0 = fro(amax(*pp.y0))
    ok = kappa < 1.0 and y0 + kappa * eps <= eps * (1 - 1e-12)
    return bool(ok), kappa, y0


def largest_ball(pp: Prepared, eps_max: float, iters: int = 40):
    lo, hi = 0.0, eps_max
    if ball_check(pp, hi)[0]:
        return hi
    for _ in range(iters):
        m = math.sqrt(lo * hi) if lo > 0 else hi * 1e-6
        if ball_check(pp, m)[0]:
            lo = m
        else:
            hi = m
        if lo > 0 and hi / lo < 1.05:
            break
    return lo


def initial_boxes(R: float, n: int):
    edges = np.linspace(-R, R, n + 1)
    idx = np.stack(np.meshgrid(*[np.arange(n)] * 6, indexing="ij"), -1).reshape(-1, 6)
    BL = edges[idx]; BH = edges[idx + 1]
    return np.ascontiguousarray(BL), np.ascontiguousarray(BH)


@dataclass
class Result:
    rho: float
    rho0: float
    rho_safe: float
    eps: float
    kappa: float
    complete: bool
    stats: dict
    worst_z: list
    seconds: float
    shard: tuple = (0, 1)

    def to_dict(self):
        return asdict(self)


def certify(cd: ControllerData, st: Setup, task_limits=(2.8, 0.2), rho0=None, grid: int = 4,
            minw_rel: float = 1e-3, max_boxes: int = 20_000_000, shard=(0, 1), eps=None,
            verbose: bool = True) -> Result:
    """Run (a), (b), (c). With shard = (i, n) only every n-th initial box (offset i) is
    processed; the certificate is then the min of the shard levels (see combine())."""
    t0 = time.time()
    pp = prepare(cd, st)
    rs = rho_safe(pp, task_limits)
    rho0 = rs if rho0 is None else min(rho0, rs)
    R = math.sqrt(rho0) * (1 + 1e-12)
    if eps is None:
        eps = largest_ball(pp, 0.5 * R)
    ok, kappa, y0 = ball_check(pp, eps)
    if not ok or eps <= 0:
        raise RuntimeError(f"ball check failed (eps={eps}, kappa={kappa}, y0={y0})")
    BL, BH = initial_boxes(R, grid)
    sel = np.arange(shard[0], len(BL), shard[1])
    BL, BH = BL[sel], BH[sel]
    minw = minw_rel * R
    rhos, stats, worst = bnb_many(pp.data, BL, BH, rho0, eps * eps * (1 - 1e-12), minw, max_boxes)
    complete = bool(np.all(rhos >= 0))
    k = int(np.argmin(rhos))
    rho = float(rhos[k]) if complete else -1.0
    rho = min(rho, rho0)
    if rho < eps * eps:
        rho = -1.0 if not complete else rho
    tot = stats.sum(0)
    names = ["evaluated", "verified", "outside", "in_ball", "failed_small", "counterexample"]
    res = Result(rho, rho0, rs, eps, kappa, complete, {nm: int(v) for nm, v in zip(names, tot)},
                 worst[k].tolist(), time.time() - t0, tuple(shard))
    if verbose:
        print(f"[certify] {cd.name}: rho={rho:.6g} (rho0 {rho0:.6g}, safe {rs:.6g}) eps={eps:.3g} "
              f"kappa={kappa:.6f} boxes={res.stats} {res.seconds:.1f}s")
    return res


def combine(results):
    """Certified level from shard results (every shard must be complete)."""
    if not all(r["complete"] for r in results):
        return -1.0
    return min(r["rho"] for r in results)


def ellipsoid_volume(W, rho):
    """Volume of {x : |W (x - xs)|^2 <= rho} in state units (m, rad, rad, m/s, rad/s, rad/s)."""
    return math.pi ** 3 / 6 * rho ** 3 / abs(np.linalg.det(W))
