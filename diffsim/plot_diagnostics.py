"""Plots for results/diffsim/diag/*.json -> results/diffsim/*.png (python diffsim/plot_diagnostics.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "diffsim" / "diag"
OUT = ROOT / "results" / "diffsim"

# categorical slots in fixed order (reference palette, light mode); random inits are muted gray
C = {"open loop": "#2a78d6", "solved": "#eb6834", "stalled mlp:4": "#1baf7a", "stalled mlp:8": "#6250d6"}
GRAY = "#a3a29c"
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "legend.frameon": False, "figure.dpi": 130})


def key(name):
    if name.startswith("solved"):
        return "solved"
    if name.startswith("stalled mlp:4"):
        return "stalled mlp:4"
    if name.startswith("stalled mlp:8"):
        return "stalled mlp:8"
    return None


def label(name):
    return {"solved": "solved mlp:4 (100 % test)", "stalled mlp:4": "stalled mlp:4 (curriculum 0.30 s)",
            "stalled mlp:8": "stalled mlp:8 (curriculum 0.55 s)"}.get(key(name), name)


def fig_chaos_gradnorm():
    ch = json.load(open(D / "chaos.json")); gn = json.load(open(D / "gradnorm.json"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    t = np.array(ch["t"]); tol = np.array(ch["t_open_loop"])
    a = ax[0]
    for n, r in ch["closed_loop"].items():
        k = key(n)
        a.semilogy(t, r["sigma_max_median"], color=C[k] if k else GRAY, lw=2 if k else 1,
                   label=label(n) if k else None, zorder=3 if k else 1)
    a.semilogy(tol, ch["open_loop"]["sigma_max_median"], color=C["open loop"], label="open loop (reference forces)")
    a.plot([], [], color=GRAY, lw=1, label="random inits (9)")
    a.set_title(r"State sensitivity $\sigma_{max}(\partial s_t/\partial s_0)$, median of 32 starts", fontsize=9, loc="left")
    a.set_xlabel("time [s]")
    a.legend(fontsize=7, loc="upper left")

    h = np.array(gn["h_s"])
    for i, (obj, ttl) in enumerate([("run", r"$|\nabla_\theta J|$, running cost, median over 64 starts"),
                                    ("snr", "Start agreement |mean g|²/mean |g|² (running)")]):
        a = ax[i + 1]
        for n, r in gn["policies"].items():
            k = key(n)
            y = r["run"]["per_start_median"] if obj == "run" else r["run"]["snr"]
            (a.semilogy if obj == "run" else a.plot)(h, y, color=C[k] if k else GRAY, lw=2 if k else 1,
                                                     zorder=3 if k else 1)
        a.set_title(ttl, fontsize=9, loc="left"); a.set_xlabel("horizon H [s]")
    lam = ch["open_loop"]["rate_per_s"]
    ax[1].semilogy(h, 0.004 * np.exp(lam * (h - 0.5)), color=INK2, lw=1, ls="--")
    ax[1].text(1.6, 0.0005 * np.exp(lam * 1.1), f"open-loop rate e^{{{lam:.1f} t}}", fontsize=7, color=INK2)
    ax[2].axhline(1 / 64, color=INK2, lw=1, ls="--"); ax[2].text(0.1, 0.05, "1/K: unrelated directions", fontsize=7, color=INK2)
    ax[2].set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(OUT / "chaos_gradnorm.png"); plt.close(fig)


def fig_fo_zo():
    pols = {}
    for f in ("solved", "stalled", "random"):
        d = json.load(open(D / f"cosine_{f}.json")); h = np.array(d["h_s"]); pols.update(d["policies"])
    tm = json.load(open(D / "timing.json"))
    ratio = np.mean([r["ratio"] for r in tm["archs"].values()])     # fwd+bwd / fwd wall time
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for j, (kind, sig) in enumerate([("term", 0.01), ("term", 0.1), ("run", 0.01), ("run", 0.1)]):
        for n, o in pols.items():
            k = key(n)
            rows = [o[f"{kind}|{x:.2f}|{sig}"] for x in h]
            sty = dict(color=C[k] if k else GRAY, lw=2 if k else 1, zorder=3 if k else 1, marker="o", ms=3 if k else 2)
            ax[0, j].plot(h, [r["cos_FO_ZO"] for r in rows], **sty)
            # equal-compute variance ratio: FO sample costs `ratio` forward rollouts, antithetic ZO costs 2
            ax[1, j].semilogy(h, [r["relvar_FO"] * ratio / (r["relvar_ZO"] * 2) for r in rows], **sty)
        ax[0, j].set_title(f"{'terminal' if kind == 'term' else 'running'} cost, σ = {sig}", fontsize=9, loc="left")
        ax[0, j].set_ylim(-1.05, 1.05); ax[0, j].axhline(0, color=INK2, lw=0.6)
        ax[1, j].axhline(1, color=INK2, lw=1, ls="--")
        ax[1, j].set_xlabel("horizon H [s]")
    ax[0, 0].set_ylabel("cos(FO, ZO) of the mean gradients")
    ax[1, 0].set_ylabel("FO / ZO variance at equal compute\n(< 1: first order better)")
    for k in ("solved", "stalled mlp:4", "stalled mlp:8"):
        ax[0, 3].plot([], [], color=C[k], label=label(k + " x"))
    ax[0, 3].plot([], [], color=GRAY, lw=1, label="random init mlp:2/4/8")
    ax[0, 3].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"First-order (BPTT at θ+σε) vs zeroth-order (antithetic randomized smoothing) gradients; "
                 f"32 starts × 64 perturbations; backward costed at {ratio:.1f}× forward", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "fo_vs_zo.png"); plt.close(fig)


if __name__ == "__main__":
    fig_chaos_gradnorm(); fig_fo_zo()
    print("wrote", OUT / "chaos_gradnorm.png", OUT / "fo_vs_zo.png")
