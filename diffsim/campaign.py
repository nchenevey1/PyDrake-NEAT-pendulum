"""Experiment campaign for the diffsim study (run by .github/workflows/diffsim.yml).

  python diffsim/campaign.py tune   METHOD BUDGET_B OUT      4 configs of METHOD in parallel (mlp:4, tuning seed 100)
  python diffsim/campaign.py select TUNE_DIR CHOSEN.json     best config per method, on VALIDATION only
  python diffsim/campaign.py main   METHOD ARCH BUDGET_B "SEEDS" CHOSEN.json OUT    seeds in parallel
  python diffsim/campaign.py report MAIN_DIR OUT_DIR [--drake]                     tables + Drake check

Every method gets the same tuning budget: 4 configurations x the same env-step budget, one tuning
seed (100, never used in the main runs), selection by validation success, then the deepest
curriculum level reached, then the mean validation streak. Each run is pinned to one CPU core
(taskset), so wall-clock times are per core on the same runner type.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = str(ROOT / "diffsim" / "train.py")
TUNE_ARCH = "mlp:4"
TUNE_SEED = 100

GRIDS = {
    "cmaes": [{"popsize": 64, "sigma0": 0.5}, {"popsize": 32, "sigma0": 0.5},
              {"popsize": 128, "sigma0": 0.5}, {"popsize": 64, "sigma0": 0.25}],
    "bptt_curr": [{"lr": lr, "L": L} for lr in (1e-2, 3e-3) for L in (50, 100)],
    "zo_curr": [{"lr": lr, "sigma": s} for lr in (1e-2, 3e-3) for s in (0.03, 0.1)],
    "bptt_full": [{"lr": lr, "temp0": t} for lr in (1e-2, 3e-3) for t in (0.3, 0.05)],
    "bptt_trunc": [{"lr": lr, "W": w} for lr in (1e-2, 3e-3) for w in (25, 50)],
    "bptt_dense": [{"lr": lr} for lr in (3e-2, 1e-2, 3e-3, 1e-3)],
    "hybrid": [{"lr": lr, "switch": sw} for lr in (1e-2, 3e-3) for sw in (0.5, 0.75)],
}
CURRICULUM = {"cmaes", "bptt_curr", "zo_curr", "hybrid"}


def env1():
    """One thread per run (the run is also pinned to one core)."""
    e = dict(os.environ)
    e.update({"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
              "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    return e


def run_parallel(cmds, logs):
    ncpu = len(os.sched_getaffinity(0))
    procs = []
    for i, (c, lg) in enumerate(zip(cmds, logs)):
        f = open(lg, "w")
        procs.append((subprocess.Popen(["taskset", "-c", str(i % ncpu)] + c, stdout=f, stderr=subprocess.STDOUT,
                                       env=env1(), cwd=ROOT), f))
    rc = 0
    for p, f in procs:
        rc |= p.wait(); f.close()
    return rc


def tune(method, budget, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cmds, logs = [], []
    for i, hp in enumerate(GRIDS[method]):
        cmds.append([sys.executable, TRAIN, method, TUNE_ARCH, str(TUNE_SEED), str(budget), "--hp", json.dumps(hp),
                     "--out", str(out), "--tag", f"cfg{i}", "--no-test"])
        logs.append(out / f"{method}_cfg{i}.stdout")
    return run_parallel(cmds, logs)


def selection_key(r):
    lvl = -r["level_min"] if r["method"] in CURRICULUM and r.get("level_min") is not None else 0
    return (r["best_val"], lvl, r["best_val_streak"])


def select(tune_dir, chosen):
    rs = [json.load(open(p)) for p in Path(tune_dir).rglob("*.json") if not p.name.endswith(".policy.json")
          and not p.name.endswith(".final.json")]
    rs.sort(key=lambda r: r["name"])          # ties keep the lowest config index (the method's default first)
    best = {}
    for r in rs:
        m = r["method"]
        if m not in best or selection_key(r) > selection_key(best[m]):
            best[m] = r
    out = {m: {"hp": r["hp"], "tune_key": list(selection_key(r)), "name": r["name"]} for m, r in best.items()}
    out["_all"] = [{"name": r["name"], "hp": r["hp"], "key": list(selection_key(r))} for r in rs]
    json.dump(out, open(chosen, "w"), indent=1)
    print(json.dumps({m: v for m, v in out.items() if m != "_all"}, indent=1))


def main_runs(method, arch, budget, seeds, chosen, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    hp = json.load(open(chosen))[method]["hp"]
    cmds, logs = [], []
    for s in seeds.split():
        cmds.append([sys.executable, TRAIN, method, arch, s, str(budget), "--hp", json.dumps(hp), "--out", str(out)])
        logs.append(out / f"{method}_{arch.replace(':', '')}_s{s}.stdout")
    return run_parallel(cmds, logs)


def report(main_dir, out_dir, drake=False):
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "diffsim"))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rs = [json.load(open(p)) for p in Path(main_dir).rglob("*.json") if not p.name.endswith(".policy.json")
          and not p.name.endswith(".final.json")]
    rs.sort(key=lambda r: (r["method"], r["arch"], r["seed"]))
    if drake:
        from pendulum.policy import GraphPolicy
        from train import drake_check
        for r in rs:
            if r.get("test_20s", 0) > 0:
                pol = next(Path(main_dir).rglob(f"{r['name']}.policy.json"))
                ok, n = drake_check(GraphPolicy.load(pol))
                r["drake"] = f"{ok}/{n}"
    lines = ["| method | arch | seed | test 20 s | test 60 s | Drake | closest start [s] | best val | "
             "units at best [B] | used [B] | wall [min] | success per B units | ratio |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        f = lambda v, d=3: "–" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
        lines.append(f"| {r['method']} | {r['arch']} | {r['seed']} | {f(r.get('test_20s'))} | {f(r.get('test_60s'))} | "
                     f"{r.get('drake', '–')} | {f(r.get('level_min_s'), 2)} | {f(r['best_val'], 2)} | "
                     f"{r['units_at_best'] / 1e9:.2f} | {r['used'] / 1e9:.2f} | {r['wall_s'] / 60:.1f} | "
                     f"{f(r.get('success_per_Bstep'), 2)} | {f(r.get('ratio'), 1)} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    json.dump(rs, open(out_dir / "report.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "tune":
        sys.exit(tune(a[0], float(a[1]), a[2]))
    if cmd == "select":
        select(a[0], a[1])
    if cmd == "main":
        sys.exit(main_runs(a[0], a[1], float(a[2]), a[3], a[4], a[5]))
    if cmd == "report":
        report(a[0], a[1], drake="--drake" in a)
