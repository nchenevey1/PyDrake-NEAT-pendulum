"""Evaluate a saved policy on the true objective (target physics, held-out random ICs).

  python evaluate.py runs/neat_s0/best_val.json
  python evaluate.py runs/neat_s0/best_val.json --n 1000 --no-prune
  python evaluate.py runs/x_s0/best_val.json --drake 16 --out runs/x_s0/verify.json   # + Drake check
  python evaluate.py models/compact_curriculum_02_001/best_model.pth --legacy   # old checkpoints
"""

import argparse
import json
import sys
from pathlib import Path

from pendulum.evaluation import drake_check, full_report
from pendulum.policy import GraphPolicy
from pendulum.task import LEGACY_OBS_SCALE, task_for_policy


def load_legacy(path):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
    import torch
    from pendulum.methods.neat import from_legacy_genome
    data = torch.load(path, map_location="cpu", weights_only=False)
    genome = data["genome"] if "genome" in data else data["population"][0].genome
    pol = from_legacy_genome(genome).compile(double_tanh=True)
    pol.meta["obs_scale"] = dict(LEGACY_OBS_SCALE)
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="load a legacy .pth (NEAT genome)")
    ap.add_argument("--save-pruned", default=None)
    ap.add_argument("--drake", type=int, default=0, metavar="N",
                    help="also run N test starts closed-loop in Drake (20 s and 60 s)")
    ap.add_argument("--out", default=None, help="write the report as JSON to this file")
    args = ap.parse_args()
    pol = load_legacy(args.policy) if args.legacy else GraphPolicy.load(args.policy)
    task = task_for_policy(pol)          # the observation scales stored with the policy
    rep = full_report(pol, n_test=args.n, task=task, prune=not args.no_prune)
    rep["obs_scale"] = pol.meta["obs_scale"]
    pr = rep.pop("_pruned_policy", None)
    if pr is not None and args.save_pruned:
        pr.save(args.save_pruned)
    if args.drake:
        rep["drake_20s"] = drake_check(pol, args.drake, task)
        rep["drake_60s"] = drake_check(pol, args.drake, task, duration=60.0)
    print(json.dumps(rep, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
