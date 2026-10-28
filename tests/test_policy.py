import numpy as np

from pendulum.policy import GraphPolicy, build_arch, pack
from pendulum.rollout import simulate
from pendulum.task import TARGET_PHYSICS, PhysicsParams


def random_graph(rng, n_in=9, n_nodes=6, n_conn=25):
    src = rng.integers(0, n_in + n_nodes, n_conn)
    dst = rng.integers(n_in, n_in + n_nodes, n_conn)
    return GraphPolicy(n_in, rng.normal(0, 1, n_nodes), [0] * n_nodes, src, dst,
                       rng.normal(0, 1.5, n_conn), src >= dst, n_in + n_nodes - 1, int(rng.integers(2)))


def replay(policy, traj, task_obs):
    step = policy.make_stepper()
    return np.array([step(task_obs(s)) for s in traj[:-1, :6]])


from pendulum.task import observe as obs_of  # noqa: E402


def test_kernel_forward_matches_python_reference():
    rng = np.random.default_rng(1)
    pols = [random_graph(rng) for _ in range(10)] + [build_arch("mlp:4", 9, rng), build_arch("rnn:3", 9, rng),
                                                     build_arch("mlp:3,2", 9, rng), build_arch("linear", 9, rng)]
    for p in pols:
        _, traj = simulate(p, np.full(6, 0.03), PhysicsParams(2.0, 1.0, 300.0))
        np.testing.assert_allclose(traj[1:, 6], np.clip(replay(p, traj, obs_of), -1, 1), atol=1e-12)


def test_compaction_and_serialization_preserve_behaviour(tmp_path):
    rng = np.random.default_rng(2)
    for _ in range(10):
        p = random_graph(rng)
        p.w[rng.integers(0, len(p.w), 5)] = 0.0
        c = p.compacted()
        assert c.n_params <= p.n_params
        c.save(tmp_path / "p.json")
        q = GraphPolicy.load(tmp_path / "p.json")
        for a, b in [(p, c), (p, q)]:
            s1, t1 = simulate(a, np.full(6, 0.02), TARGET_PHYSICS)
            s2, t2 = simulate(b, np.full(6, 0.02), TARGET_PHYSICS)
            np.testing.assert_allclose(t1, t2, atol=1e-12)


def test_param_counts():
    assert build_arch("linear", 9).n_params == 10
    assert build_arch("mlp:4", 9).n_params == 9 * 4 + 4 + 4 + 1
    assert build_arch("rnn:4", 9).n_params == 9 * 4 + 16 + 4 + 4 + 1
    q = build_arch("mlp:4", 9, np.random.default_rng(0)).quantized(4)
    assert len(np.unique(np.abs(q.w))) <= 8


def test_pack_offsets():
    rng = np.random.default_rng(3)
    pols = [random_graph(rng, n_nodes=int(rng.integers(1, 6)), n_conn=int(rng.integers(1, 20))) for _ in range(20)]
    pk = pack(pols)
    assert pk.cptr[-1] == sum(len(p.w) for p in pols)
    assert pk.node_ptr[-1] == sum(p.n_nodes for p in pols)
