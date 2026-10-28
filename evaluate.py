"""Evaluate a saved policy on the true objective (target physics, held-out random ICs).

  python evaluate.py runs/neat_s0/best_val.json
  python evaluate.py runs/neat_s0/best_val.json --n 1000 --no-prune
  python evaluate.py models/compact_curriculum_02_001/best_model.pth --legacy   # old checkpoints
"""

import argparse
import json
import sys
from pathlib import Path

from pendulum.evaluation import full_report
from pendulum.policy import GraphPolicy


def load_legacy(path):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
    import torch
    from pendulum.methods.neat import from_legacy_genome
    data = torch.load(path, map_location="cpu", weights_only=False)
    genome = data["genome"] if "genome" in data else data["population"][0].genome
    return from_legacy_genome(genome).compile(double_tanh=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="load a legacy .pth (NEAT genome)")
    ap.add_argument("--save-pruned", default=None)
    args = ap.parse_args()
    pol = load_legacy(args.policy) if args.legacy else GraphPolicy.load(args.policy)
    rep = full_report(pol, n_test=args.n, prune=not args.no_prune)
    pr = rep.pop("_pruned_policy", None)
    if pr is not None and args.save_pruned:
        pr.save(args.save_pruned)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
