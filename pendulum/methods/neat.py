"""
NEAT (topology + weights), ported from legacy/train.py + legacy/neural_net.py.

Same algorithm and defaults as the legacy trainer (dynamic speciation threshold
towards a target species count, fitness sharing, per-species elitism,
tournament selection from the top 75 %, crossover + mutation, species
stagnation culling, sigma "shock" on stagnation). Changes:

* Innovation numbers live in a per-run tracker that is checkpointed. The legacy
  class-level counter restarted from 0 on resume, so new genes could reuse
  existing innovation numbers and crossover then mixed unrelated connections.
  The tracker also gives the same structural mutation the same innovation /
  node id (standard NEAT), which makes crossover and speciation meaningful.
* Optional `init: random` gives every initial genome its own weights (legacy:
  the whole first generation was one cloned genome).
* Optional `delete_conn_rate` removes connections (pressure toward small nets).
* Genomes compile to `GraphPolicy` with the same in-place recurrent semantics
  as `CartNN.forward`; nodes that cannot reach the output are pruned and the
  size penalty counts only deployable parameters.
* Output uses a single tanh (`legacy_double_tanh: true` restores tanh(tanh(z))).
"""

import random
from typing import Dict, List, Optional

import numba as nb
import numpy as np

from ..policy import ACT_TANH, BIAS_CLIP, WEIGHT_CLIP, GraphPolicy
from .base import Method

NEAT_DEFAULTS = {
    "pop_size": 1200,
    "species_target": 100,
    "species_step": 0.05,
    "species_threshold_init": 3.0,
    "species_threshold_min": 0.5,
    "species_stagnation": 25,
    "weight_rate": 0.6,
    "bias_rate": 0.2,
    "struct_rate": 0.25,          # add-connection prob; add-node prob = struct_rate / 2
    "delete_conn_rate": 0.0,
    "sigma": 0.5,
    "sigma_stagnation": 10,
    "record_margin": 1.0,         # score improvement that resets stagnation
    "weight_clip": WEIGHT_CLIP,
    "bias_clip": BIAS_CLIP,
    "reset_prob": 0.1,            # prob. a mutated weight is re-drawn instead of perturbed
    "allow_recurrent": True,
    "init": "random",             # random | shared (legacy)
    "legacy_double_tanh": False,
    "tournament_size": 3,
    "tournament_frac": 0.75,
    "compat_c1": 1.0, "compat_c2": 1.0, "compat_c3": 0.4,
}


class ConnGene:
    __slots__ = ("innov", "src", "dst", "w", "enabled")

    def __init__(self, innov, src, dst, w, enabled=True):
        self.innov, self.src, self.dst, self.w, self.enabled = innov, src, dst, w, enabled

    def copy(self):
        return ConnGene(self.innov, self.src, self.dst, self.w, self.enabled)


class Genome:
    """nodes: {id: [kind, bias]} with kind in {'input','hidden','output'}; conns: {innov: ConnGene}"""
    __slots__ = ("n_in", "nodes", "conns", "fitness", "score", "_policy", "_arrays")

    def __init__(self, n_in: int):
        self.n_in = n_in
        self.nodes: Dict[int, list] = {}
        self.conns: Dict[int, ConnGene] = {}
        self.fitness = 0.0
        self.score = 0.0
        self._policy: Optional[GraphPolicy] = None
        self._arrays = None

    def arrays(self):
        """(sorted innovation numbers, weights) for the numba speciation kernel."""
        if self._arrays is None:
            keys = np.array(sorted(self.conns), dtype=np.int64)
            self._arrays = (keys, np.array([self.conns[k].w for k in keys], dtype=np.float64))
        return self._arrays

    @property
    def out_id(self) -> int:
        return self.n_in

    def copy(self) -> "Genome":
        g = Genome(self.n_in)
        g.nodes = {k: list(v) for k, v in self.nodes.items()}
        g.conns = {k: c.copy() for k, c in self.conns.items()}
        return g

    def compile(self, double_tanh: bool = False) -> GraphPolicy:
        if self._policy is None:
            self._policy = compile_genome(self, double_tanh)
        return self._policy


class InnovationTracker:
    def __init__(self, n_in: int):
        self.next_innov = 0
        self.next_node = n_in + 1
        self.conn_map: Dict[tuple, int] = {}
        self.split_map: Dict[int, int] = {}

    def conn(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.conn_map:
            self.next_innov += 1
            self.conn_map[key] = self.next_innov
        return self.conn_map[key]

    def split(self, innov: int, genome: Genome) -> int:
        nid = self.split_map.get(innov)
        if nid is None:
            nid = self.next_node
            self.next_node += 1
            self.split_map[innov] = nid
        if nid in genome.nodes:   # same connection split twice in one lineage
            nid = self.next_node
            self.next_node += 1
        return nid


def compile_genome(g: Genome, double_tanh: bool = False) -> GraphPolicy:
    """Topological order exactly as legacy CartNN._get_topological_sort (Kahn, FIFO,
    nodes in id order, leftover cycle nodes appended in id order)."""
    n_in = g.n_in
    node_ids = sorted(g.nodes)
    conns = sorted((c for c in g.conns.values() if c.enabled), key=lambda c: c.innov)
    adj = {n: [] for n in node_ids}
    indeg = {n: 0 for n in node_ids}
    for c in conns:
        adj[c.src].append(c.dst)
        indeg[c.dst] += 1
    queue = [n for n in node_ids if indeg[n] == 0]
    order, qi = [], 0
    while qi < len(queue):
        n = queue[qi]; qi += 1
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(order) < len(node_ids):
        seen = set(order)
        order.extend(n for n in node_ids if n not in seen)
    non_input = [n for n in order if g.nodes[n][0] != "input"]
    pos = {i: i for i in range(n_in)}
    for k, n in enumerate(non_input):
        pos[n] = n_in + k
    src = [pos[c.src] for c in conns]
    dst = [pos[c.dst] for c in conns]
    w = [c.w for c in conns]
    delayed = [s >= d for s, d in zip(src, dst)]
    bias = [g.nodes[n][1] for n in non_input]
    pol = GraphPolicy(n_in, bias, [ACT_TANH] * len(non_input), src, dst, w, delayed,
                      pos[g.out_id], 1 if double_tanh else 0, {"arch": "neat"})
    return pol.compacted()


def compatibility_distance(g1: Genome, g2: Genome, c1: float, c2: float, c3: float) -> float:
    k1, k2 = g1.conns, g2.conns
    if not k1 and not k2:
        return 0.0
    max1 = max(k1) if k1 else 0
    max2 = max(k2) if k2 else 0
    cutoff = min(max1, max2)
    matching = 0
    wdiff = 0.0
    disjoint = 0
    excess = 0
    for k, c in k1.items():
        o = k2.get(k)
        if o is not None:
            matching += 1
            wdiff += abs(c.w - o.w)
        elif k > cutoff:
            excess += 1
        else:
            disjoint += 1
    for k in k2:
        if k not in k1:
            if k > cutoff:
                excess += 1
            else:
                disjoint += 1
    n = max(len(k1), len(k2), 1)
    if n < 20:
        n = 1
    avg_w = wdiff / matching if matching else 0.0
    return c1 * excess / n + c2 * disjoint / n + c3 * avg_w


@nb.njit(cache=True)
def _distance(k1, w1, k2, w2, c1, c2, c3):
    n1 = k1.shape[0]; n2 = k2.shape[0]
    if n1 == 0 and n2 == 0:
        return 0.0
    i = 0; j = 0; matching = 0; disjoint = 0; wd = 0.0
    while i < n1 and j < n2:
        if k1[i] == k2[j]:
            matching += 1; wd += abs(w1[i] - w2[j]); i += 1; j += 1
        elif k1[i] < k2[j]:
            disjoint += 1; i += 1
        else:
            disjoint += 1; j += 1
    excess = (n1 - i) + (n2 - j)
    n = max(n1, n2, 1)
    if n < 20:
        n = 1
    avg = wd / matching if matching > 0 else 0.0
    return c1 * excess / n + c2 * disjoint / n + c3 * avg


@nb.njit(cache=True)
def _assign_species(gk, gw, goff, rk, rw, roff, thr, c1, c2, c3):
    """First-fit assignment of genomes to species representatives (legacy order).
    Genomes that fit no species found a new one and become its representative.
    Returns assign[g] (species index; indices >= n_reps are new species) and the
    genome index of each new species' representative."""
    G = goff.shape[0] - 1
    R = roff.shape[0] - 1
    assign = np.empty(G, dtype=np.int64)
    new_reps = np.empty(G, dtype=np.int64)
    n_new = 0
    for g in range(G):
        a = gk[goff[g]:goff[g + 1]]; aw = gw[goff[g]:goff[g + 1]]
        found = -1
        for r in range(R):
            if _distance(a, aw, rk[roff[r]:roff[r + 1]], rw[roff[r]:roff[r + 1]], c1, c2, c3) < thr:
                found = r
                break
        if found < 0:
            for q in range(n_new):
                h = new_reps[q]
                if _distance(a, aw, gk[goff[h]:goff[h + 1]], gw[goff[h]:goff[h + 1]], c1, c2, c3) < thr:
                    found = R + q
                    break
        if found < 0:
            new_reps[n_new] = g
            found = R + n_new
            n_new += 1
        assign[g] = found
    return assign, new_reps[:n_new]


def _pack_arrays(genomes):
    arrs = [g.arrays() for g in genomes]
    off = np.zeros(len(arrs) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(a[0]) for a in arrs])
    if arrs:
        return (np.concatenate([a[0] for a in arrs]) if off[-1] else np.zeros(0, np.int64),
                np.concatenate([a[1] for a in arrs]) if off[-1] else np.zeros(0),
                off)
    return np.zeros(0, np.int64), np.zeros(0), off


class Species:
    def __init__(self, rep: Genome, sid: int):
        self.id = sid
        self.rep = rep
        self.members: List[Genome] = [rep]
        self.best_fitness = -float("inf")
        self.stagnation = 0
        self.age = 0

    def reset(self):
        if self.members:
            self.rep = self.members[0]
        self.members = []


class NEAT(Method):
    name = "neat"

    def __init__(self, cfg: dict, n_in: int, rng: np.random.Generator):
        super().__init__({**NEAT_DEFAULTS, **(cfg or {})}, n_in, rng)
        c = self.cfg
        self.py_rng = random.Random(int(rng.integers(2**31)))
        self.tracker = InnovationTracker(n_in)
        self.species: List[Species] = []
        self.species_counter = 0
        self.threshold = c["species_threshold_init"]
        self.sigma = c["sigma"]
        self.best_ever = -9999.0
        self.stagnation = 0
        self.generation = 0
        self._champions: List[GraphPolicy] = []
        proto = self._new_genome()
        self.population = [proto.copy() if c["init"] == "shared" else self._new_genome()
                           for _ in range(c["pop_size"])]

    # ------------------------------------------------------------ genome ops
    def _new_genome(self) -> Genome:
        g = Genome(self.n_in)
        for i in range(self.n_in):
            g.nodes[i] = ["input", 0.0]
        g.nodes[g.out_id] = ["output", float(self.rng.normal(0, 1.0))]
        for i in range(self.n_in):
            innov = self.tracker.conn(i, g.out_id)
            g.conns[innov] = ConnGene(innov, i, g.out_id, float(self.rng.normal(0, 1.0)))
        return g

    def _mutate(self, g: Genome) -> Genome:
        c, r, pr = self.cfg, self.rng, self.py_rng
        for cg in g.conns.values():
            if pr.random() < c["weight_rate"]:
                if pr.random() < c["reset_prob"]:
                    cg.w = float(r.normal(0, 1.0))
                else:
                    cg.w += float(r.normal(0, self.sigma))
                cg.w = min(max(cg.w, -c["weight_clip"]), c["weight_clip"])
        for nd in g.nodes.values():
            if nd[0] == "input":
                continue
            if pr.random() < c["bias_rate"]:
                if pr.random() < c["reset_prob"]:
                    nd[1] = float(r.normal(0, 1.0))
                else:
                    nd[1] += float(r.normal(0, self.sigma))
                nd[1] = min(max(nd[1], -c["bias_clip"]), c["bias_clip"])
        if pr.random() < c["struct_rate"]:
            self._add_connection(g)
        if pr.random() < c["struct_rate"] * 0.5:
            self._add_node(g)
        if c["delete_conn_rate"] > 0 and pr.random() < c["delete_conn_rate"]:
            self._delete_connection(g)
        g._policy = None
        g._arrays = None
        return g

    def _add_connection(self, g: Genome):
        existing = {(cg.src, cg.dst) for cg in g.conns.values()}
        ids = list(g.nodes)
        cands = []
        for i in ids:
            for j in ids:
                if i == j or g.nodes[j][0] == "input" or (i, j) in existing:
                    continue
                cands.append((i, j))
        if not self.cfg["allow_recurrent"]:
            cands = [(i, j) for i, j in cands if not self._creates_cycle(g, i, j)]
        if not cands:
            return
        i, j = cands[self.py_rng.randrange(len(cands))]
        innov = self.tracker.conn(i, j)
        g.conns[innov] = ConnGene(innov, i, j, float(self.rng.normal(0, 1.0)))

    @staticmethod
    def _creates_cycle(g: Genome, i: int, j: int) -> bool:
        # adding i->j creates a cycle iff i is reachable from j
        adj = {}
        for cg in g.conns.values():
            if cg.enabled:
                adj.setdefault(cg.src, []).append(cg.dst)
        stack, seen = [j], set()
        while stack:
            n = stack.pop()
            if n == i:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, []))
        return False

    def _add_node(self, g: Genome):
        active = [cg for cg in g.conns.values() if cg.enabled]
        if not active:
            return
        cg = active[self.py_rng.randrange(len(active))]
        cg.enabled = False
        nid = self.tracker.split(cg.innov, g)
        g.nodes[nid] = ["hidden", 0.0]
        i1 = self.tracker.conn(cg.src, nid)
        i2 = self.tracker.conn(nid, cg.dst)
        g.conns[i1] = ConnGene(i1, cg.src, nid, 1.0)
        g.conns[i2] = ConnGene(i2, nid, cg.dst, cg.w)

    def _delete_connection(self, g: Genome):
        active = [k for k, cg in g.conns.items() if cg.enabled]
        if len(active) > 1:
            del g.conns[active[self.py_rng.randrange(len(active))]]

    def _crossover(self, p1: Genome, p2: Genome) -> Genome:
        best, other = (p1, p2) if p1.fitness > p2.fitness else (p2, p1)
        child = Genome(self.n_in)
        child.nodes = {k: list(v) for k, v in best.nodes.items()}
        for k, cg in best.conns.items():
            o = other.conns.get(k)
            src = cg if (o is None or self.py_rng.random() < 0.5) else o
            child.conns[k] = src.copy()
        return child

    # -------------------------------------------------------------- ask/tell
    def ask(self) -> List[GraphPolicy]:
        dt = self.cfg["legacy_double_tanh"]
        return [g.compile(dt) for g in self.population]

    def tell(self, fitness, scores, stats) -> None:
        c = self.cfg
        for g, f, s in zip(self.population, fitness, scores):
            g.fitness = max(0.001, float(f))
            g.score = float(s)
        order = np.argsort(-np.asarray(fitness), kind="stable")
        self._champions = [self.population[i].compile(c["legacy_double_tanh"]) for i in order[:3]]
        self._update_sigma(float(np.max(scores)))
        self._speciate()
        self._reproduce()
        self.generation += 1

    def _update_sigma(self, best_score: float):
        c = self.cfg
        if best_score > self.best_ever + c["record_margin"]:
            self.best_ever = best_score
            self.stagnation = 0
            self.sigma = c["sigma"]
        else:
            self.stagnation += 1
        lim = c["sigma_stagnation"]
        if self.stagnation > lim:
            stuck = self.stagnation - lim - 1
            if stuck % 10 == 0:
                self.sigma = c["sigma"] * min(1 + self.stagnation / 10, 5)
            else:
                self.sigma = max(c["sigma"], self.sigma * 0.9)

    def on_physics_change(self) -> None:
        pass

    def _speciate(self):
        c = self.cfg
        for s in self.species:
            s.reset()
        gk, gw, goff = _pack_arrays(self.population)
        rk, rw, roff = _pack_arrays([s.rep for s in self.species])
        assign, new_reps = _assign_species(gk, gw, goff, rk, rw, roff, float(self.threshold),
                                           float(c["compat_c1"]), float(c["compat_c2"]), float(c["compat_c3"]))
        for gi in new_reps:
            self.species_counter += 1
            sp = Species(self.population[gi], self.species_counter)
            sp.members = []
            self.species.append(sp)
        for g, a in zip(self.population, assign):
            self.species[a].members.append(g)
        self.species = [s for s in self.species if s.members]
        if len(self.species) < c["species_target"]:
            self.threshold = max(c["species_threshold_min"], self.threshold - c["species_step"])
        elif len(self.species) > c["species_target"]:
            self.threshold += c["species_step"]

    def _reproduce(self):
        c = self.cfg
        for s in self.species:
            s.members.sort(key=lambda g: g.fitness, reverse=True)
        best_sid = max(self.species, key=lambda s: s.members[0].fitness).id
        surviving, adj_sums = [], {}
        for s in self.species:
            cur = s.members[0].fitness
            if cur > s.best_fitness:
                s.best_fitness = cur
                s.stagnation = 0
            else:
                s.stagnation += 1
            s.age += 1
            if s.stagnation < c["species_stagnation"] or s.id == best_sid or len(self.species) == 1:
                n = len(s.members)
                adj_sums[s.id] = sum(g.fitness / n for g in s.members)
                surviving.append(s)
        total = sum(adj_sums.values())
        target = c["pop_size"]
        counts, rema = {}, {}
        for s in surviving:
            share = adj_sums[s.id] / total * target
            counts[s.id] = max(1, int(share))
            rema[s.id] = share - counts[s.id]
        diff = target - sum(counts.values())
        if diff > 0:
            srt = sorted(rema.items(), key=lambda x: x[1], reverse=True)
            for i in range(diff):
                counts[srt[i % len(srt)][0]] += 1
        elif diff < 0:
            srt = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for i in range(-diff):
                sid = srt[i % len(srt)][0]
                if counts[sid] > 1:
                    counts[sid] -= 1
        nxt = []
        for s in surviving:
            n = counts[s.id]
            elite = s.members[0].copy()
            elite._policy = s.members[0]._policy
            elite._arrays = s.members[0]._arrays
            nxt.append(elite)
            for _ in range(n - 1):
                p1, p2 = self._tournament(s), self._tournament(s)
                nxt.append(self._mutate(self._crossover(p1, p2)))
        self.population = nxt
        self.species = surviving

    def _tournament(self, s: Species) -> Genome:
        pool = s.members[:max(1, int(len(s.members) * self.cfg["tournament_frac"]))]
        k = min(self.cfg["tournament_size"], len(pool))
        return max(self.py_rng.sample(pool, k), key=lambda g: g.fitness)

    def champions(self) -> List[GraphPolicy]:
        return list(self._champions)

    def best_genome(self) -> Genome:
        return max(self.population, key=lambda g: g.fitness)

    def info(self) -> Dict:
        pols = [g._policy for g in self.population if g._policy is not None]
        return {
            "species": len(self.species),
            "threshold": round(self.threshold, 3),
            "sigma": round(self.sigma, 3),
            "avg_params": round(float(np.mean([p.n_params for p in pols])), 2) if pols else None,
            "avg_hidden": round(float(np.mean([p.n_hidden for p in pols])), 2) if pols else None,
        }


def from_legacy_genome(lg) -> Genome:
    """Convert a `legacy/neural_net.Genome` (e.g. from an old checkpoint) to a Genome.
    Compile with `double_tanh=True` to reproduce the legacy CartNN output exactly."""
    g = Genome(lg.input_size)
    for nid, n in lg.nodes.items():
        g.nodes[int(nid)] = [n.type, float(n.bias)]
    for k, c in lg.connections.items():
        g.conns[int(k)] = ConnGene(int(k), int(c.in_node), int(c.out_node), float(c.weight), bool(c.enabled))
    return g
