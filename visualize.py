"""Render a policy on the target physics.

  python visualize.py runs/neat_s0/best_val.json --out swingup.gif      # matplotlib animation (headless)
  python visualize.py runs/neat_s0/best_val.json --drake               # cross-check in Drake, live meshcat-free plot
"""

import argparse

import numpy as np

from pendulum.policy import GraphPolicy
from pendulum.rollout import simulate, STAT_NAMES
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS, TEST_SEED, fixed_initial_states


def animate(traj, out, dt, fps=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    stride = max(1, int(round(1 / (fps * dt))))
    frames = traj[::stride]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal")
    ax.axhline(0, color="0.7", lw=1); ax.axvline(-2.8, color="r", lw=0.5); ax.axvline(2.8, color="r", lw=0.5)
    cart, = ax.plot([], [], "s", ms=18, color="#993333")
    poles, = ax.plot([], [], "-o", lw=2, color="k", ms=6)
    txt = ax.text(-3.3, 2.0, "", fontsize=9)

    def upd(i):
        x, t1, t2 = frames[i, :3]
        p1 = (x - np.sin(t1), -np.cos(t1))
        p2 = (p1[0] - np.sin(t1 + t2), p1[1] - np.cos(t1 + t2))
        cart.set_data([x], [0])
        poles.set_data([x, p1[0], p2[0]], [0, p1[1], p2[1]])
        txt.set_text(f"t={i * stride * dt:5.2f}s  u={frames[i, 6]:+.2f}")
        return cart, poles, txt
    anim = FuncAnimation(fig, upd, frames=len(frames), blit=True)
    writer = FFMpegWriter(fps=fps) if out.endswith(".mp4") else PillowWriter(fps=fps)
    anim.save(out, writer=writer)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--ic", type=int, default=0, help="index into the test IC set (-1 = nominal rest)")
    ap.add_argument("--out", default="episode.gif")
    ap.add_argument("--drake", action="store_true", help="also replay the actions in Drake and compare")
    args = ap.parse_args()
    pol = GraphPolicy.load(args.policy)
    ic = np.zeros(6) if args.ic < 0 else fixed_initial_states(args.ic + 1, TEST_SEED, DEFAULT_TASK.ic_noise)[args.ic]
    stats, traj = simulate(pol, ic, TARGET_PHYSICS)
    print(pol.describe(), dict(zip(STAT_NAMES, np.round(stats, 3))))
    if args.drake:
        from pendulum.drake_env import DrakeEnv
        env = DrakeEnv(TARGET_PHYSICS)
        env.reset(ic)
        step = pol.make_stepper()
        from pendulum.task import observe as obs_of
        s = ic.copy()
        err = 0.0
        for k in range(1, len(traj)):
            a = float(np.clip(step(obs_of(s)), -1, 1))
            s = env.step(a)
            err = max(err, np.abs(s - traj[k, :6]).max())
        print(f"Drake closed-loop replay: max state deviation from fast sim = {err:.2e}")
    animate(traj, args.out, DEFAULT_TASK.dt)


if __name__ == "__main__":
    main()
