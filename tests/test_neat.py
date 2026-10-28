import sys
from pathlib import Path

import numpy as np
import pytest

from pendulum.methods.neat import (NEAT, _assign_species, _pack_arrays, compatibility_distance,
                                   from_legacy_genome)


def test_numba_speciation_matches_python():
    rng = np.random.default_rng(0)
    m = NEAT({"pop_size": 300}, 9, rng)
    for g in m.population:
        for _ in range(4):
            m._mutate(g)
    pop, thr = m.population, 1.0
    reps = pop[:5]
    gk, gw, goff = _pack_arrays(pop)
    rk, rw, roff = _pack_arrays(reps)
    assign, new = _assign_species(gk, gw, goff, rk, rw, roff, thr, 1.0, 1.0, 0.4)
    # python first-fit
    reps_py = list(reps)
    for g, a in zip(pop, assign):
        for r_i, r in enumerate(reps_py):
            if compatibility_distance(g, r, 1.0, 1.0, 0.4) < thr:
                assert a == r_i
                break
        else:
            assert a == len(reps_py)
            reps_py.append(g)


def test_neat_runs_and_grows():
    rng = np.random.default_rng(1)
    m = NEAT({"pop_size": 100, "species_target": 5}, 9, rng)
    for _ in range(5):
        pols = m.ask()
        f = rng.random(len(pols))
        m.tell(f, f, None)
    assert len(m.population) == 100
    assert max(len(g.nodes) for g in m.population) > 10


def test_compiled_genome_matches_legacy_cartnn():
    torch = pytest.importorskip("torch")
    legacy = Path(__file__).resolve().parent.parent / "legacy"
    sys.path.insert(0, str(legacy))
    try:
        import neural_net as ln
        from config import Config
    finally:
        sys.path.remove(str(legacy))
    np.random.seed(0)
    import random
    random.seed(0)
    from pendulum.policy import GraphPolicy
    checked = 0
    for trial in range(30):
        g = ln.Genome(9, 1)
        for _ in range(12):  # grow structure incl. recurrent connections
            g = ln.mutate_neat(g, Config, sigma=0.5)
            ln._mutate_add_connection(g)
            if random.random() < 0.5:
                ln._mutate_add_node(g)
        net = ln.CartNN(g)
        pol = from_legacy_genome(g).compile(double_tanh=True)
        step = pol.make_stepper()
        rng = np.random.default_rng(trial)
        with torch.inference_mode():
            for t in range(20):
                x = rng.uniform(-1, 1, 9).astype(np.float32)
                ref = net(torch.as_tensor(x)).item()
                ours = step(x.astype(np.float64))
                assert abs(ref - ours) < 1e-5, (trial, t)
        checked += pol.is_recurrent
    assert checked > 0, "should include recurrent genomes"


def test_cmaes_restart_keeps_progress():
    """A restart must continue from the current mean, never from a random re-init."""
    from pendulum.methods.es import CMAES
    rng = np.random.default_rng(0)
    m = CMAES({"arch": "linear", "popsize": 8, "sigma0": 0.2}, 9, rng)
    target = np.full(m.template.n_flat, 3.0)
    for _ in range(200):
        pols = m.ask()
        f = np.array([-np.sum((p.get_flat() - target) ** 2) for p in pols])
        m.tell(f, f, None)
    before = np.array(m.es.mean)
    m.es.opts["tolupsigma"] = 1e-30  # force a stop
    pols = m.ask(); f = np.zeros(len(pols)); m.tell(f, f, None)
    assert m.restarts >= 1
    assert np.linalg.norm(np.array(m.es.mean) - before) < 1.0
    assert np.linalg.norm(before - target) < 1.0


def test_cmaes_flat_fitness_does_not_restart():
    from pendulum.methods.es import CMAES
    m = CMAES({"arch": "mlp:2", "popsize": 8}, 9, np.random.default_rng(1))
    for _ in range(100):
        pols = m.ask(); f = np.zeros(len(pols)); m.tell(f, f, None)
    assert m.restarts == 0
