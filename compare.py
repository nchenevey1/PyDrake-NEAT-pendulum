"""Aggregate runs into a comparison table + learning-curve plots.

  python compare.py                 # scans runs/, writes results/comparison.md and results/*.png
  python compare.py --runs-dir runs --out results
Works on finished runs (report.json) and on runs still in progress (latest log records).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def load_run(d: Path):
    cfg = yaml.safe_load(open(d / "config.yaml"))
    gens, evals = [], []
    for line in open(d / "log.jsonl"):
        r = json.loads(line)
        (gens if r["type"] == "GEN" else evals).append(r)
    rep = json.load(open(d / "report.json")) if (d / "report.json").exists() else None
    return {"dir": d, "cfg": cfg, "gens": gens, "evals": evals, "report": rep}


def arch_of(cfg):
    mc = cfg.get("method_cfg", {})
    if cfg["method"] in ("neat",):
        return "evolved"
    if cfg["method"] == "neat_cma":
        return "evolved->fixed"
    return mc.get("arch", "mlp:4")


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def row(run):
    g = run["gens"][-1] if run["gens"] else {}
    rep = run["report"] or {}
    r = rep.get("report", {})
    ev = run["evals"][-1]["best_val"] if run["evals"] else {}
    first_target = next((x["gen"] for x in run["gens"] if x["phys"]["difficulty"] >= 1.0), None)
    ms = rep.get("milestones") or {}
    def ms_steps(k):
        m = ms.get(k)
        return f"{m['env_steps'] / 1e9:.2f}" if m else "-"
    # milestones for in-progress runs
    if not ms and run["evals"]:
        for e in run["evals"]:
            for thr in (0.5, 0.9):
                k = f"val_success>={thr}"
                if e["best_val"]["success_rate"] >= thr and k not in ms:
                    ms[k] = {"env_steps": e["env_steps"]}
    pr = r.get("pruned", {})
    return {
        "run": run["dir"].name,
        "method": run["cfg"]["method"],
        "arch": arch_of(run["cfg"]),
        "curr": run["cfg"]["curriculum"]["mode"],
        "fitness": run["cfg"]["train"]["fitness"],
        "done": "yes" if rep else "running",
        "gens": g.get("gen"),
        "steps_B": g.get("env_steps", 0) / 1e9,
        "wall_h": g.get("wall", 0) / 3600,
        "difficulty": g.get("phys", {}).get("difficulty"),
        "target_gen": first_target,
        "val": ev.get("success_rate"),
        "params": r.get("params", ev.get("params")),
        "test20": r.get("test_20s", {}).get("success_rate"),
        "test60": r.get("test_60s", {}).get("success_rate"),
        "wide": r.get("test_wide_ic_0.2", {}).get("success_rate"),
        "int8": r.get("int8_20s", {}).get("success_rate"),
        "int4": r.get("int4_20s", {}).get("success_rate"),
        "pruned_params": pr.get("params"),
        "pruned_test20": pr.get("test_20s", {}).get("success_rate"),
        "steps_to_val50_B": ms_steps("val_success>=0.5") if ms else "-",
        "steps_to_val90_B": ms_steps("val_success>=0.9") if ms else "-",
    }


COLS = [("run", "run"), ("method", "method"), ("arch", "arch"), ("curr", "curriculum"), ("fitness", "fitness"),
        ("done", "done"), ("gens", "gens"), ("steps_B", "env steps (B)"), ("wall_h", "wall (h)"),
        ("difficulty", "difficulty"), ("target_gen", "gen @ target physics"),
        ("val", "val success"), ("params", "params"), ("test20", "TEST success 20s"),
        ("test60", "TEST 60s"), ("wide", "TEST wide IC"), ("int8", "int8"), ("int4", "int4"),
        ("pruned_params", "pruned params"), ("pruned_test20", "pruned TEST"),
        ("steps_to_val50_B", "steps to val>=0.5 (B)"), ("steps_to_val90_B", "steps to val>=0.9 (B)")]


def plots(runs, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for r in runs:
        groups[re.sub(r"_s\d+$", "", r["dir"].name)].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("tab10")
    for i, (name, rs) in enumerate(sorted(groups.items())):
        c = cmap(i % 10)
        for j, r in enumerate(rs):
            lab = name if j == 0 else None
            if r["evals"]:
                x = [e["env_steps"] / 1e9 for e in r["evals"]]
                y = [e["best_val"]["success_rate"] for e in r["evals"]]
                axes[0].plot(x, y, color=c, label=lab, lw=1.2)
                w = np.interp([e["gen"] for e in r["evals"]], [g["gen"] for g in r["gens"]],
                              [g["wall"] / 3600 for g in r["gens"]])
                axes[1].plot(w, y, color=c, label=lab, lw=1.2)
            if r["gens"]:
                axes[2].plot([g["env_steps"] / 1e9 for g in r["gens"]],
                             [g["phys"]["difficulty"] for g in r["gens"]], color=c, label=lab, lw=1.2)
    axes[0].set_xlabel("training env steps (billions)"); axes[0].set_ylabel("best validation success (target physics)")
    axes[1].set_xlabel("wall-clock (hours)"); axes[1].set_ylabel("best validation success")
    axes[2].set_xlabel("training env steps (billions)"); axes[2].set_ylabel("curriculum difficulty")
    for a in axes:
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(d) for d in sorted(Path(args.runs_dir).iterdir()) if (d / "log.jsonl").exists()]
    rows = [row(r) for r in runs]
    lines = ["| " + " | ".join(h for _, h in COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for k, _ in COLS) + " |")
    table = "\n".join(lines)
    (out / "comparison.md").write_text(table + "\n")
    json.dump(rows, open(out / "comparison.json", "w"), indent=1, default=str)
    print(table)
    try:
        plots(runs, out)
    except Exception as e:  # plotting is optional
        print(f"[plots skipped: {e}]")


if __name__ == "__main__":
    main()
