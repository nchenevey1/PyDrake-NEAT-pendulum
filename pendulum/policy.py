"""
Unified policy representation.

Every method (NEAT, MLP/RNN + ES, hybrids) produces a `GraphPolicy`: a small
neural graph evaluated with *sequential in-place* semantics, which is exactly
what the legacy `CartNN.forward` did:

  buffer = [inputs (n_in) | non-input nodes in evaluation order]
  every control step:
      buffer[:n_in] = observation
      for each non-input node j (in order):
          buffer[j] = act(bias_j + sum_c w_c * src_value_c)

A connection is *delayed* when it reads the value its source had at the end of
the previous control step (recurrence). For legacy NEAT graphs a connection is
delayed iff its source is not evaluated before its destination, which is the
same thing the in-place buffer produced. Feed-forward MLPs have no delayed
connections; Elman RNNs mark their hidden->hidden connections delayed.

Action = value of the output node (tanh => [-1, 1]), scaled by max_force.
`action_mode=1` reproduces the legacy double tanh (action = tanh(tanh(z)),
which caps the force at tanh(1) = 76% of max_force).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

ACT_TANH = 0
ACT_IDENTITY = 1
ACT_RELU = 2
ACT_NAMES = {"tanh": ACT_TANH, "identity": ACT_IDENTITY, "relu": ACT_RELU}


@dataclass
class GraphPolicy:
    n_in: int
    bias: np.ndarray            # (H,) biases of non-input nodes, in evaluation order
    act: np.ndarray             # (H,) activation codes
    src: np.ndarray             # (C,) source buffer index
    dst: np.ndarray             # (C,) destination buffer index (>= n_in)
    w: np.ndarray               # (C,) weights
    delayed: np.ndarray         # (C,) bool
    out_pos: int                # buffer index of output node
    action_mode: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.bias = np.asarray(self.bias, dtype=np.float64).reshape(-1)
        self.act = np.asarray(self.act, dtype=np.int32).reshape(-1)
        self.src = np.asarray(self.src, dtype=np.int32).reshape(-1)
        self.dst = np.asarray(self.dst, dtype=np.int32).reshape(-1)
        self.w = np.asarray(self.w, dtype=np.float64).reshape(-1)
        self.delayed = np.asarray(self.delayed, dtype=bool).reshape(-1)
        # Kernel requires connections grouped by destination
        order = np.argsort(self.dst, kind="stable")
        self.src, self.dst, self.w, self.delayed = (
            self.src[order], self.dst[order], self.w[order], self.delayed[order])

    # ------------------------------------------------------------------ size
    @property
    def n_nodes(self) -> int:
        """Non-input nodes (hidden + output)."""
        return len(self.bias)

    @property
    def n_hidden(self) -> int:
        return self.n_nodes - 1

    @property
    def n_connections(self) -> int:
        return int(np.count_nonzero(self.w))

    @property
    def n_params(self) -> int:
        """Deployable parameter count: nonzero weights + biases of non-input nodes."""
        return self.n_connections + self.n_nodes

    @property
    def is_recurrent(self) -> bool:
        return bool(np.any(self.delayed & (self.w != 0)))

    # ------------------------------------------------------------ parameters
    def get_flat(self) -> np.ndarray:
        return np.concatenate([self.w, self.bias])

    def with_flat(self, theta: np.ndarray) -> "GraphPolicy":
        theta = np.asarray(theta, dtype=np.float64)
        c = len(self.w)
        return GraphPolicy(self.n_in, theta[c:].copy(), self.act.copy(), self.src.copy(),
                           self.dst.copy(), theta[:c].copy(), self.delayed.copy(),
                           self.out_pos, self.action_mode, dict(self.meta))

    @property
    def n_flat(self) -> int:
        return len(self.w) + len(self.bias)

    def copy(self) -> "GraphPolicy":
        return self.with_flat(self.get_flat())

    # ---------------------------------------------------------- compression
    def quantized(self, bits: int = 8) -> "GraphPolicy":
        """Symmetric per-tensor uniform quantization of weights and biases."""
        def q(x):
            m = np.max(np.abs(x)) if x.size else 0.0
            if m == 0:
                return x.copy()
            levels = 2 ** (bits - 1) - 1
            scale = m / levels
            return np.round(x / scale) * scale
        p = self.with_flat(np.concatenate([q(self.w), q(self.bias)]))
        p.meta["quantized_bits"] = bits
        return p

    def without_connections(self, idx: Sequence[int]) -> "GraphPolicy":
        theta = self.get_flat()
        theta[np.asarray(idx, dtype=int)] = 0.0
        return self.with_flat(theta).compacted()

    def compacted(self) -> "GraphPolicy":
        """Drop zero-weight connections and nodes that cannot influence the output."""
        keep_c = self.w != 0
        src, dst, w, dl = self.src[keep_c], self.dst[keep_c], self.w[keep_c], self.delayed[keep_c]
        # Reverse reachability from output
        needed = {self.out_pos}
        changed = True
        while changed:
            changed = False
            for s, d in zip(src, dst):
                if d in needed and s >= self.n_in and s not in needed:
                    needed.add(int(s)); changed = True
        old_nodes = [j for j in range(self.n_in, self.n_in + self.n_nodes) if j in needed]
        remap = {j: j for j in range(self.n_in)}
        for k, j in enumerate(old_nodes):
            remap[j] = self.n_in + k
        keep = np.array([d in needed for d in dst], dtype=bool)
        idx = np.array([j - self.n_in for j in old_nodes], dtype=int)
        return GraphPolicy(
            self.n_in, self.bias[idx], self.act[idx],
            [remap[int(s)] for s in src[keep]], [remap[int(d)] for d in dst[keep]],
            w[keep], dl[keep], remap[self.out_pos], self.action_mode, dict(self.meta))

    # --------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {
            "n_in": self.n_in, "bias": self.bias.tolist(), "act": self.act.tolist(),
            "src": self.src.tolist(), "dst": self.dst.tolist(), "w": self.w.tolist(),
            "delayed": self.delayed.astype(int).tolist(), "out_pos": self.out_pos,
            "action_mode": self.action_mode, "meta": self.meta,
            "n_params": self.n_params,
        }

    @staticmethod
    def from_dict(d: dict) -> "GraphPolicy":
        return GraphPolicy(d["n_in"], d["bias"], d["act"], d["src"], d["dst"], d["w"],
                           d["delayed"], d["out_pos"], d.get("action_mode", 0), d.get("meta", {}))

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path) -> "GraphPolicy":
        with open(path) as f:
            return GraphPolicy.from_dict(json.load(f))

    def describe(self) -> str:
        rec = " recurrent" if self.is_recurrent else ""
        return f"{self.n_hidden}H/{self.n_connections}C ({self.n_params} params{rec})"

    # ---------------------------------------------------- python reference
    def make_stepper(self):
        """Pure-python reference forward (used by tests / visualization)."""
        size = self.n_in + self.n_nodes
        buf = np.zeros(size)

        def step(obs):
            prev = buf.copy()
            buf[:self.n_in] = obs
            for k in range(self.n_nodes):
                j = self.n_in + k
                s = self.bias[k]
                m = self.dst == j
                for sv, wv, dv in zip(self.src[m], self.w[m], self.delayed[m]):
                    s += wv * (prev[sv] if dv else buf[sv])
                a = self.act[k]
                buf[j] = np.tanh(s) if a == ACT_TANH else (max(s, 0.0) if a == ACT_RELU else s)
            out = buf[self.out_pos]
            return np.tanh(out) if self.action_mode == 1 else out
        return step


# =============================================================================
# Fixed-topology builders
# =============================================================================

def build_mlp(n_in: int, hidden: Sequence[int], rng: Optional[np.random.Generator] = None,
              init_std: float = 1.0, act: str = "tanh") -> GraphPolicy:
    """Dense feed-forward MLP n_in -> hidden... -> 1 (tanh output)."""
    rng = rng or np.random.default_rng(0)
    layers = [list(range(n_in))]
    nxt = n_in
    for h in list(hidden) + [1]:
        layers.append(list(range(nxt, nxt + h)))
        nxt += h
    src, dst, w = [], [], []
    for a, b in zip(layers[:-1], layers[1:]):
        std = init_std / np.sqrt(len(a))
        for j in b:
            for i in a:
                src.append(i); dst.append(j); w.append(rng.normal(0, std))
    n_nodes = nxt - n_in
    acts = [ACT_NAMES[act]] * (n_nodes - 1) + [ACT_TANH]
    return GraphPolicy(n_in, np.zeros(n_nodes), acts, src, dst, w, np.zeros(len(w), bool),
                       nxt - 1, 0, {"arch": f"mlp{list(hidden)}"})


def build_rnn(n_in: int, hidden: int, rng: Optional[np.random.Generator] = None,
              init_std: float = 1.0) -> GraphPolicy:
    """Elman RNN: h_t = tanh(W x_t + U h_{t-1} + b), a_t = tanh(v h_t + c)."""
    rng = rng or np.random.default_rng(0)
    hid = list(range(n_in, n_in + hidden))
    out = n_in + hidden
    src, dst, w, dl = [], [], [], []
    for j in hid:
        for i in range(n_in):
            src.append(i); dst.append(j); w.append(rng.normal(0, init_std / np.sqrt(n_in))); dl.append(False)
        for i in hid:
            src.append(i); dst.append(j); w.append(rng.normal(0, 0.5 * init_std / np.sqrt(hidden))); dl.append(True)
    for i in hid:
        src.append(i); dst.append(out); w.append(rng.normal(0, init_std / np.sqrt(hidden))); dl.append(False)
    return GraphPolicy(n_in, np.zeros(hidden + 1), [ACT_TANH] * (hidden + 1), src, dst, w, dl,
                       out, 0, {"arch": f"rnn[{hidden}]"})


def build_arch(spec: str, n_in: int, rng: Optional[np.random.Generator] = None) -> GraphPolicy:
    """'linear' | 'mlp:4' | 'mlp:8,4' | 'rnn:4'"""
    spec = spec.strip().lower()
    if spec == "linear":
        return build_mlp(n_in, [], rng)
    kind, _, sizes = spec.partition(":")
    sizes = [int(s) for s in sizes.split(",") if s]
    if kind == "mlp":
        return build_mlp(n_in, sizes, rng)
    if kind == "rnn":
        assert len(sizes) == 1
        return build_rnn(n_in, sizes[0], rng)
    raise ValueError(f"Unknown architecture spec: {spec}")


# =============================================================================
# Packing for the numba kernel
# =============================================================================

@dataclass
class PackedPolicies:
    n_in: int
    node_ptr: np.ndarray     # (P+1,) offsets into node arrays
    bias: np.ndarray
    act: np.ndarray
    cptr: np.ndarray         # (N+1,) per node: offset into connection arrays
    src: np.ndarray
    w: np.ndarray
    delayed: np.ndarray
    out_pos: np.ndarray      # (P,)
    action_mode: np.ndarray  # (P,)

    def __len__(self):
        return len(self.out_pos)


def pack(policies: List[GraphPolicy]) -> PackedPolicies:
    n_in = policies[0].n_in
    node_ptr = np.zeros(len(policies) + 1, dtype=np.int64)
    biases, acts, cptrs, srcs, ws, dls = [], [], [], [], [], []
    c_off = 0
    for i, p in enumerate(policies):
        assert p.n_in == n_in
        node_ptr[i + 1] = node_ptr[i] + p.n_nodes
        biases.append(p.bias); acts.append(p.act)
        # per-node connection ranges (connections already sorted by dst)
        counts = np.bincount(p.dst - n_in, minlength=p.n_nodes)
        cptrs.append(c_off + np.concatenate([[0], np.cumsum(counts)[:-1]]))
        c_off += len(p.w)
        srcs.append(p.src); ws.append(p.w); dls.append(p.delayed)
    cptr = np.concatenate(cptrs + [np.array([c_off])]).astype(np.int64)
    return PackedPolicies(
        n_in, node_ptr, np.concatenate(biases), np.concatenate(acts).astype(np.int32), cptr,
        np.concatenate(srcs).astype(np.int32), np.concatenate(ws),
        np.concatenate(dls).astype(np.uint8),
        np.array([p.out_pos for p in policies], dtype=np.int64),
        np.array([p.action_mode for p in policies], dtype=np.int64))
