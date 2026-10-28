"""Train one configuration.

  python run.py configs/neat.yaml --seed 0
  python run.py configs/cmaes_mlp4.yaml --seed 1 --set budget.generations=3000 --set threads=2
  python run.py configs/neat.yaml --seed 0 --resume          # continue / extend a run
"""

import argparse
import json
from pathlib import Path

from pendulum.config import deep_merge, load_config, parse_override


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. budget.generations=500")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    overrides = {}
    for s in args.set:
        overrides = deep_merge(overrides, parse_override(s))
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, overrides)
    run_dir = Path(args.runs_dir) / f"{cfg['name']}_s{cfg['seed']}"

    from pendulum.trainer import Trainer  # import after arg parsing (numba compile)
    summary = Trainer(cfg, run_dir, resume=args.resume).run()
    if summary:
        rep = summary.get("report", {})
        print(json.dumps({k: summary[k] for k in ("name", "generations", "env_steps", "wall_s",
                                                  "final_difficulty")}, indent=None))
        if rep:
            t = rep["test_20s"]
            print(f"TEST (target physics, {t['n']} ICs): success {t['success_rate']:.3f}, "
                  f"params {rep['params']}, 60s success {rep['test_60s']['success_rate']:.3f}")


if __name__ == "__main__":
    main()
