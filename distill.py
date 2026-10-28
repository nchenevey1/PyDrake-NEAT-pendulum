"""Distil the classic expert (trajopt + TVLQR + LQR) into small reactive MLPs with DAgger.

Answers: can a small *reactive* network (9-D observation -> force, no clock) represent a
controller that solves the task? If yes, the neuroevolution failures are a search /
credit-assignment problem, not a capacity problem.

  python distill.py --hidden 4 8 16 32 --iters 12
Writes results/distill/<arch>.json (GraphPolicy) and results/distill/summary.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pendulum.classic import ClassicController, design
from pendulum.evaluation import evaluate_policies, test_ics, val_ics
from pendulum.policy import build_mlp
from pendulum.rollout import simulate
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS, observe, sample_initial_states


def to_policy(net: torch.nn.Sequential, hidden):
    """Copy torch MLP weights into a GraphPolicy (same tanh MLP semantics)."""
    pol = build_mlp(9, hidden)
    w, b = [], []
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    for L in lin:  # GraphPolicy MLP connection order: per destination node, all sources
        w.append(L.weight.detach().double().numpy().reshape(-1))
        b.append(L.bias.detach().double().numpy())
    return pol.with_flat(np.concatenate(w + b))


def expert_labels(expert: ClassicController, traj: np.ndarray):
    """Expert action at every state the student visited (time-indexed expert)."""
    expert.reset()
    obs, acts = [], []
    for k in range(len(traj) - 1):
        o = observe(traj[k, :6])
        obs.append(o)
        acts.append(expert(o))   # advances the expert's clock exactly like the episode
    return np.array(obs), np.array(acts)


def make_net(hidden):
    layers, d = [], 9
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
        d = h
    layers += [torch.nn.Linear(d, 1), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers).double()


def train(net, X, Y, steps=4000, batch=1024, lr=3e-3):
    """Fixed number of Adam steps per DAgger iteration (cost independent of dataset size)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    X = torch.as_tensor(X); Y = torch.as_tensor(Y)[:, None]
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,))
        loss = torch.mean((net(X[idx]) - Y[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return float(torch.mean((net(X) - Y) ** 2))


def dagger(expert, hidden, iters, n_ics, seed, max_data=400_000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net(hidden)
    X = np.zeros((0, 9)); Y = np.zeros(0)
    history = []
    for it in range(iters):
        ics = sample_initial_states(n_ics, rng, DEFAULT_TASK.ic_noise)
        for ic in ics:
            if it == 0:   # iteration 0: expert's own trajectories (behaviour cloning)
                traj = expert_rollout(expert, ic)
            else:         # DAgger: student's trajectories, expert labels
                _, traj = simulate(to_policy(net, hidden), ic, TARGET_PHYSICS)
            o, a = expert_labels(expert, traj)
            X = np.vstack([X, o]); Y = np.concatenate([Y, a])
        if len(X) > max_data:
            keep = rng.choice(len(X), max_data, replace=False)
            X, Y = X[keep], Y[keep]
        loss = train(net, X, Y)
        res = evaluate_policies([to_policy(net, hidden)], val_ics(64))[0]
        history.append({"iter": it, "data": len(X), "mse": loss, "val_success": res["success_rate"],
                        "val_streak_s": res["mean_streak_s"]})
        print(f"  {hidden} it {it}: data {len(X)} mse {loss:.2e} val success {res['success_rate']:.2f} "
              f"streak {res['mean_streak_s']:.1f}s", flush=True)
    return net, history


def expert_rollout(expert, ic, act_noise=0.0, rng=None, return_labels=False):
    """Expert rollout; with act_noise > 0 the *executed* action is perturbed (DART) while the
    label stays the expert's clean action, so the data covers the expert's recovery tube."""
    from pendulum.rollout import physics_step
    expert.reset()
    s = np.array(ic, dtype=np.float64)
    traj, obs, labels = [s.copy()], [], []
    for _ in range(DEFAULT_TASK.max_steps):
        o = observe(s)
        a = expert(o)
        obs.append(o); labels.append(a)
        a_exec = float(np.clip(a + (act_noise * rng.standard_normal() if act_noise else 0.0), -1, 1))
        physics_step(s, a_exec * TARGET_PHYSICS.max_force, TARGET_PHYSICS.g, TARGET_PHYSICS.drag,
                     DEFAULT_TASK.dt)
        traj.append(s.copy())
        k = len(traj) - 1
        up = abs((s[1] - np.pi + np.pi) % (2 * np.pi) - np.pi) < DEFAULT_TASK.upright_tol and \
            abs((s[2] + np.pi) % (2 * np.pi) - np.pi) < DEFAULT_TASK.upright_tol
        if return_labels and (not np.all(np.isfinite(s)) or abs(s[0]) > DEFAULT_TASK.limit_x
                              or (k > DEFAULT_TASK.swing_up_steps and not up)):
            break   # the episode would have terminated here; later states are irrelevant
    if return_labels:
        o, l = np.array(obs), np.array(labels)
        ok = np.all(np.isfinite(o), axis=1) & np.isfinite(l)
        return o[ok], l[ok]
    return np.array(traj)


def dart(expert, hidden, rounds, n_ics, seed, act_noise=0.15):
    """DART (Laskey et al. 2017): behaviour cloning on noise-injected expert rollouts."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net(hidden)
    X = np.zeros((0, 9)); Y = np.zeros(0)
    history = []
    for it in range(rounds):
        for ic in sample_initial_states(n_ics, rng, DEFAULT_TASK.ic_noise):
            o, a = expert_rollout(expert, ic, act_noise, rng, return_labels=True)
            X = np.vstack([X, o]); Y = np.concatenate([Y, a])
        loss = train(net, X, Y)
        res = evaluate_policies([to_policy(net, hidden)], val_ics(64))[0]
        history.append({"iter": it, "data": len(X), "mse": loss, "val_success": res["success_rate"],
                        "val_streak_s": res["mean_streak_s"]})
        print(f"  {hidden} it {it}: data {len(X)} mse {loss:.2e} val success {res['success_rate']:.2f} "
              f"streak {res['mean_streak_s']:.1f}s", flush=True)
    return net, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", nargs="+", default=["4", "8", "16", "32"], help="e.g. 8 or 16,8")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--ics", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/distill")
    ap.add_argument("--mode", choices=["dagger", "dart"], default="dagger")
    ap.add_argument("--noise", type=float, default=0.15, help="DART action-noise std")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cpath = Path("results/classic/controller.npz")
    expert = ClassicController.load(cpath) if cpath.exists() else design(verbose=False)
    summary = {}
    for hs in args.hidden:
        hidden = [int(h) for h in hs.split(",")]
        if args.mode == "dart":
            net, hist = dart(expert, hidden, args.iters, args.ics, args.seed, args.noise)
        else:
            net, hist = dagger(expert, hidden, args.iters, args.ics, args.seed)
        pol = to_policy(net, hidden)
        pol.meta["arch"] = f"distilled mlp{hidden}"
        name = f"{args.mode}_mlp" + "-".join(map(str, hidden))
        pol.save(out / f"{name}.json")
        test = evaluate_policies([pol], test_ics(256))[0]
        q8 = evaluate_policies([pol.quantized(8)], test_ics(256))[0]
        summary[name] = {"params": pol.n_params, "test_success": test["success_rate"],
                         "test_streak_s": test["mean_streak_s"], "int8_success": q8["success_rate"],
                         "history": hist}
        print(f"{name}: {pol.n_params} params, TEST success {test['success_rate']:.3f}, "
              f"int8 {q8['success_rate']:.3f}", flush=True)
        json.dump(summary, open(out / f"summary_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
