"""
Rigorous interval arithmetic with first-order (slope) enclosures, in numba.

The closed-loop map is traced once into a straight-line program (`Prog`) of elementary
operations. `run_prog` evaluates that program on an input box X and returns, for every
output component, an interval enclosure of its range over X and an interval enclosure of
its gradient (slope) over X, so that for all x, c in X:  f(x) - f(c) in G(X) (x - c).

Rounding: numba cannot change the FPU rounding mode, so every result of +, -, *, / (correctly
rounded to nearest by IEEE 754, error <= ulp/2) is widened outward by at least one ulp:
dn(x) = (x - |x| 2^-52) - 1e-300. For normal x = m 2^e (1 <= m < 2), x - |x| 2^-52 lies in
[x - 2 ulp, x - ulp], and rounding to nearest keeps it <= x - ulp; the subnormal term covers
underflow. (Equivalent to, and tested against, np.nextafter, which is ~10x slower in numba.) Results of
libm sin / cos / tanh (accurate to a few ulps) are widened by a relative 4e-15 (~18 ulps) plus
1e-300 absolute. Constants are float64 values and are used exactly (the "exact dynamics" are
the real-number map with the float64 constants the simulator uses, e.g. dt = float(0.01)).

clip(v, L, H) is handled as a Lipschitz function with slope [0, 1] when the input interval
straddles a breakpoint, which keeps the mean-value enclosure valid for the saturated LQR and
the policy's velocity clipping.
"""

import math

import numba as nb
import numpy as np

OP_INPUT, OP_CONST, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG, OP_TANH, OP_SIN, OP_COS, OP_SQR, OP_CLIP = range(12)
ND = 6  # number of derivative directions (state dimension)

EPS = 2.0 ** -52     # relative outward step: |x| * 2^-52 >= ulp(x) for normal x
TINY = 1e-300        # covers underflow (any value >= 2^-1074 works; a normal number avoids
                     # the very slow subnormal arithmetic path)
TRANS_REL = 4e-15
TRANS_ABS = 1e-300


# ============================================================================ tracer

class Prog:
    """Straight-line program recorded by operator overloading on `Var`."""

    def __init__(self):
        self.ops = []          # (op, a, b, c)
        self.cst = []          # (lo, hi)
        self._cidx = {}
        self.outputs = []

    def input(self, i):
        self.ops.append((OP_INPUT, i, 0, 0))
        return Var(self, len(self.ops) - 1)

    def const(self, lo, hi=None):
        hi = lo if hi is None else hi
        key = (float(lo), float(hi))
        if key not in self._cidx:
            self.cst.append(key)
            self._cidx[key] = len(self.cst) - 1
        self.ops.append((OP_CONST, self._cidx[key], 0, 0))
        return Var(self, len(self.ops) - 1)

    def emit(self, op, a=0, b=0, c=0):
        self.ops.append((op, a, b, c))
        return Var(self, len(self.ops) - 1)

    def arrays(self):
        ops = np.array(self.ops, dtype=np.int64).reshape(-1, 4)
        cst = np.array(self.cst, dtype=np.float64).reshape(-1, 2)
        out = np.array([o.i for o in self.outputs], dtype=np.int64)
        return ops, cst, out


class Var:
    __slots__ = ("p", "i")

    def __init__(self, p, i):
        self.p = p; self.i = i

    def _v(self, o):
        if isinstance(o, Var):
            return o
        return self.p.const(float(o))

    def __add__(self, o): return self.p.emit(OP_ADD, self.i, self._v(o).i)
    def __radd__(self, o): return self.p.emit(OP_ADD, self._v(o).i, self.i)
    def __sub__(self, o): return self.p.emit(OP_SUB, self.i, self._v(o).i)
    def __rsub__(self, o): return self.p.emit(OP_SUB, self._v(o).i, self.i)
    def __mul__(self, o): return self.p.emit(OP_MUL, self.i, self._v(o).i)
    def __rmul__(self, o): return self.p.emit(OP_MUL, self._v(o).i, self.i)
    def __truediv__(self, o): return self.p.emit(OP_DIV, self.i, self._v(o).i)
    def __rtruediv__(self, o): return self.p.emit(OP_DIV, self._v(o).i, self.i)
    def __neg__(self): return self.p.emit(OP_NEG, self.i)


def tanh(v): return v.p.emit(OP_TANH, v.i)
def sin(v): return v.p.emit(OP_SIN, v.i)
def cos(v): return v.p.emit(OP_COS, v.i)
def sqr(v): return v.p.emit(OP_SQR, v.i)


def clip(v, lo, hi):
    return v.p.emit(OP_CLIP, v.i, v.p.const(lo).i, v.p.const(hi).i)


# ============================================================================ scalar intervals

@nb.njit(cache=True, inline="always")
def dn(x):
    # >= 1 ulp below x (see module docstring); nextafter is exact but ~10x slower
    return (x - abs(x) * EPS) - TINY


@nb.njit(cache=True, inline="always")
def up(x):
    return (x + abs(x) * EPS) + TINY


@nb.njit(cache=True, inline="always")
def tdn(x):
    return x - (abs(x) * TRANS_REL + TRANS_ABS)


@nb.njit(cache=True, inline="always")
def tup(x):
    return x + (abs(x) * TRANS_REL + TRANS_ABS)


@nb.njit(cache=True, inline="always")
def iadd(alo, ahi, blo, bhi):
    if alo == 0.0 and ahi == 0.0:
        return blo, bhi            # exact
    if blo == 0.0 and bhi == 0.0:
        return alo, ahi
    return dn(alo + blo), up(ahi + bhi)


@nb.njit(cache=True, inline="always")
def isub(alo, ahi, blo, bhi):
    return dn(alo - bhi), up(ahi - blo)


@nb.njit(cache=True, inline="always")
def imul(alo, ahi, blo, bhi):
    if (alo == 0.0 and ahi == 0.0) or (blo == 0.0 and bhi == 0.0):
        return 0.0, 0.0            # exact
    p1 = alo * blo; p2 = alo * bhi; p3 = ahi * blo; p4 = ahi * bhi
    lo = min(min(p1, p2), min(p3, p4))
    hi = max(max(p1, p2), max(p3, p4))
    if lo != lo or hi != hi:          # nan (inf * 0): give up on this enclosure
        return -np.inf, np.inf
    return dn(lo), up(hi)


@nb.njit(cache=True, inline="always")
def idiv(alo, ahi, blo, bhi):
    if blo <= 0.0 <= bhi:
        return -np.inf, np.inf
    q1 = alo / blo; q2 = alo / bhi; q3 = ahi / blo; q4 = ahi / bhi
    lo = min(min(q1, q2), min(q3, q4))
    hi = max(max(q1, q2), max(q3, q4))
    if lo != lo or hi != hi:
        return -np.inf, np.inf
    return dn(lo), up(hi)


@nb.njit(cache=True, inline="always")
def isqr(lo, hi):
    if lo >= 0.0:
        return dn(lo * lo), up(hi * hi)
    if hi <= 0.0:
        return dn(hi * hi), up(lo * lo)
    m = max(-lo, hi)
    return 0.0, up(m * m)


@nb.njit(cache=True, inline="always")
def itanh(lo, hi):
    return max(-1.0, tdn(math.tanh(lo))), min(1.0, tup(math.tanh(hi)))


@nb.njit(cache=True)
def _contains_point(lo, hi, phase):
    """Does [lo, hi] (generously widened) contain phase + 2 pi k for some integer k?"""
    tp = 2.0 * math.pi
    d = 1e-9 * (1.0 + abs(lo) + abs(hi))
    k = math.ceil((lo - d - phase) / tp)
    return phase + tp * k <= hi + d


@nb.njit(cache=True)
def isin(lo, hi):
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo >= 6.0:
        return -1.0, 1.0
    a = math.sin(lo); b = math.sin(hi)
    ylo = tdn(min(a, b)); yhi = tup(max(a, b))
    if _contains_point(lo, hi, 0.5 * math.pi):
        yhi = 1.0
    if _contains_point(lo, hi, -0.5 * math.pi):
        ylo = -1.0
    return max(-1.0, ylo), min(1.0, yhi)


@nb.njit(cache=True)
def icos(lo, hi):
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo >= 6.0:
        return -1.0, 1.0
    a = math.cos(lo); b = math.cos(hi)
    ylo = tdn(min(a, b)); yhi = tup(max(a, b))
    if _contains_point(lo, hi, 0.0):
        yhi = 1.0
    if _contains_point(lo, hi, math.pi):
        ylo = -1.0
    return max(-1.0, ylo), min(1.0, yhi)


# ============================================================================ program interpreter

@nb.njit(cache=True)
def _op(k, ops, cst, xlo, xhi, nd, V, G, Slo, Shi):
    """One operation of run_prog (natural interval extension + forward slopes)."""
    op = ops[k, 0]; a = ops[k, 1]; b = ops[k, 2]; c = ops[k, 3]
    if op == OP_INPUT:
        V[k, 0] = xlo[a]; V[k, 1] = xhi[a]
        for j in range(nd):
            G[k, j, 0] = Slo[a, j]; G[k, j, 1] = Shi[a, j]
    elif op == OP_CONST:
        V[k, 0] = cst[a, 0]; V[k, 1] = cst[a, 1]
        for j in range(nd):
            G[k, j, 0] = 0.0; G[k, j, 1] = 0.0
    elif op == OP_ADD:
        V[k, 0], V[k, 1] = iadd(V[a, 0], V[a, 1], V[b, 0], V[b, 1])
        for j in range(nd):
            G[k, j, 0], G[k, j, 1] = iadd(G[a, j, 0], G[a, j, 1], G[b, j, 0], G[b, j, 1])
    elif op == OP_SUB:
        V[k, 0], V[k, 1] = isub(V[a, 0], V[a, 1], V[b, 0], V[b, 1])
        for j in range(nd):
            G[k, j, 0], G[k, j, 1] = isub(G[a, j, 0], G[a, j, 1], G[b, j, 0], G[b, j, 1])
    elif op == OP_NEG:
        V[k, 0] = -V[a, 1]; V[k, 1] = -V[a, 0]
        for j in range(nd):
            G[k, j, 0] = -G[a, j, 1]; G[k, j, 1] = -G[a, j, 0]
    elif op == OP_MUL:
        alo = V[a, 0]; ahi = V[a, 1]; blo = V[b, 0]; bhi = V[b, 1]
        V[k, 0], V[k, 1] = imul(alo, ahi, blo, bhi)
        for j in range(nd):
            if G[a, j, 0] == 0.0 and G[a, j, 1] == 0.0:
                G[k, j, 0], G[k, j, 1] = imul(alo, ahi, G[b, j, 0], G[b, j, 1])
            elif G[b, j, 0] == 0.0 and G[b, j, 1] == 0.0:
                G[k, j, 0], G[k, j, 1] = imul(blo, bhi, G[a, j, 0], G[a, j, 1])
            else:
                p0, p1 = imul(alo, ahi, G[b, j, 0], G[b, j, 1])
                q0, q1 = imul(blo, bhi, G[a, j, 0], G[a, j, 1])
                G[k, j, 0], G[k, j, 1] = iadd(p0, p1, q0, q1)
    elif op == OP_DIV:
        alo = V[a, 0]; ahi = V[a, 1]; blo = V[b, 0]; bhi = V[b, 1]
        V[k, 0], V[k, 1] = idiv(alo, ahi, blo, bhi)
        if nd > 0:
            b2lo, b2hi = isqr(blo, bhi)
            for j in range(nd):
                # (ga * b - a * gb) / b^2
                p0, p1 = imul(G[a, j, 0], G[a, j, 1], blo, bhi)
                q0, q1 = imul(alo, ahi, G[b, j, 0], G[b, j, 1])
                n0, n1 = isub(p0, p1, q0, q1)
                G[k, j, 0], G[k, j, 1] = idiv(n0, n1, b2lo, b2hi)
    elif op == OP_SQR:
        V[k, 0], V[k, 1] = isqr(V[a, 0], V[a, 1])
        if nd > 0:
            d0, d1 = imul(2.0, 2.0, V[a, 0], V[a, 1])
            for j in range(nd):
                G[k, j, 0], G[k, j, 1] = imul(d0, d1, G[a, j, 0], G[a, j, 1])
    elif op == OP_TANH:
        V[k, 0], V[k, 1] = itanh(V[a, 0], V[a, 1])
        if nd > 0:
            t0, t1 = isqr(V[k, 0], V[k, 1])
            d0, d1 = isub(1.0, 1.0, t0, t1)
            d0 = max(d0, 0.0)
            for j in range(nd):
                G[k, j, 0], G[k, j, 1] = imul(d0, d1, G[a, j, 0], G[a, j, 1])
    elif op == OP_SIN:
        V[k, 0], V[k, 1] = isin(V[a, 0], V[a, 1])
        if nd > 0:
            d0, d1 = icos(V[a, 0], V[a, 1])
            for j in range(nd):
                G[k, j, 0], G[k, j, 1] = imul(d0, d1, G[a, j, 0], G[a, j, 1])
    elif op == OP_COS:
        V[k, 0], V[k, 1] = icos(V[a, 0], V[a, 1])
        if nd > 0:
            s0, s1 = isin(V[a, 0], V[a, 1])
            d0 = -s1; d1 = -s0
            for j in range(nd):
                G[k, j, 0], G[k, j, 1] = imul(d0, d1, G[a, j, 0], G[a, j, 1])
    elif op == OP_CLIP:
        L = cst[ops[b, 1], 0]; H = cst[ops[c, 1], 0]
        lo = V[a, 0]; hi = V[a, 1]
        V[k, 0] = min(max(lo, L), H); V[k, 1] = min(max(hi, L), H)
        if lo >= L and hi <= H:
            s0 = 1.0; s1 = 1.0
        elif hi < L or lo > H:
            s0 = 0.0; s1 = 0.0
        else:
            s0 = 0.0; s1 = 1.0
        for j in range(nd):
            G[k, j, 0], G[k, j, 1] = imul(s0, s1, G[a, j, 0], G[a, j, 1])
    return 0


@nb.njit(cache=True)
def run_prog(ops, cst, xlo, xhi, nd, V, G, Slo, Shi):
    """Evaluate the program on the box [xlo, xhi].

    V[k] = (lo, hi) range enclosure of op k; G[k, j] = (lo, hi) enclosure of d op_k / d t_j over
    the box (only if nd > 0), where the inputs are x = x0 + S t with the interval seed matrix
    S = [Slo, Shi] (n_in x nd). S = I gives the plain Jacobian; S = W^{-1} gives derivatives in
    the Lyapunov coordinates directly (tighter than multiplying afterwards)."""
    for k in range(ops.shape[0]):
        _op(k, ops, cst, xlo, xhi, nd, V, G, Slo, Shi)
    return 0


@nb.njit(cache=True)
def run_prog_mv(ops, cst, xlo, xhi, clo, chi, h, V, VC, G, Slo, Shi):
    """run_prog over the box with every intermediate range tightened by its mean-value form.

    The box is x = x_c + S t, t in [-h, h], with the center x_c in [clo, chi]. The center values
    VC are evaluated in lockstep; then for each op, V_k <- V_k  intersect  (VC_k + G_k [-h, h]),
    which is valid because G_k encloses the slope of op k over the box. This removes most of the
    dependency loss of the natural extension (e.g. cancellations inside the network's pre-activations)."""
    nd = h.shape[0]
    for k in range(ops.shape[0]):
        _op(k, ops, cst, clo, chi, 0, VC, G, Slo, Shi)
        _op(k, ops, cst, xlo, xhi, nd, V, G, Slo, Shi)
        m0 = VC[k, 0]; m1 = VC[k, 1]
        for j in range(nd):
            p0, p1 = imul(G[k, j, 0], G[k, j, 1], -h[j], h[j])
            m0, m1 = iadd(m0, m1, p0, p1)
        if m0 > V[k, 0]:
            V[k, 0] = m0
        if m1 < V[k, 1]:
            V[k, 1] = m1
    return 0


@nb.njit(cache=True)
def run_prog_float(ops, cst, x, V):
    """Plain float evaluation (midpoint constants); reference for tests."""
    n = ops.shape[0]
    for k in range(n):
        op = ops[k, 0]; a = ops[k, 1]; b = ops[k, 2]; c = ops[k, 3]
        if op == OP_INPUT:
            V[k] = x[a]
        elif op == OP_CONST:
            V[k] = 0.5 * (cst[a, 0] + cst[a, 1])
        elif op == OP_ADD:
            V[k] = V[a] + V[b]
        elif op == OP_SUB:
            V[k] = V[a] - V[b]
        elif op == OP_NEG:
            V[k] = -V[a]
        elif op == OP_MUL:
            V[k] = V[a] * V[b]
        elif op == OP_DIV:
            V[k] = V[a] / V[b]
        elif op == OP_SQR:
            V[k] = V[a] * V[a]
        elif op == OP_TANH:
            V[k] = math.tanh(V[a])
        elif op == OP_SIN:
            V[k] = math.sin(V[a])
        elif op == OP_COS:
            V[k] = math.cos(V[a])
        elif op == OP_CLIP:
            V[k] = min(max(V[a], cst[ops[b, 1], 0]), cst[ops[c, 1], 0])
    return 0
