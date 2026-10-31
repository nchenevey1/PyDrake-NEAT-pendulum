"""Turn a sweep plan (sweeps/<plan>.yaml) into a GitHub Actions job matrix.

  python ci/sweep_plan.py sweep-smoke-1        # tag name: plan "smoke"
  python ci/sweep_plan.py reliability-mlp4     # plan name (workflow_dispatch)

Plan file:
  configs: [configs/cmaes_mlp4_revcurr.yaml]   # one job per (config, seed)
  seeds: [0, 1, 2, 3, 4]
  set: ["budget.env_steps=4000000000"]         # overrides for every run (run.py --set)
  drake_starts: 16                             # closed-loop Drake check of each selected policy

Prints `plan=...` and `matrix=...` lines for $GITHUB_OUTPUT.
"""
import json
import re
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def plan_name(ref: str) -> str:
    """'sweep-<plan>-<n>' -> '<plan>'; anything else is taken as the plan name itself."""
    m = re.fullmatch(r"sweep-(.+?)(-\d+)?", ref)
    return m.group(1) if m else ref


def matrix(plan: str) -> dict:
    path = ROOT / "sweeps" / f"{plan}.yaml"
    if not path.exists():
        raise SystemExit(f"no sweep plan {path.relative_to(ROOT)}")
    p = yaml.safe_load(open(path))
    sets = " ".join(f"--set {shlex.quote(str(s))}" for s in p.get("set", []))
    include = []
    for c in p["configs"]:
        if not (ROOT / c).exists():
            raise SystemExit(f"plan {plan}: missing config {c}")
        name = (yaml.safe_load(open(ROOT / c)) or {}).get("name") or Path(c).stem
        for s in p["seeds"]:
            include.append({"config": c, "name": name, "seed": int(s), "set": sets,
                            "drake": int(p.get("drake_starts", 16))})
    if not include:
        raise SystemExit(f"plan {plan}: empty matrix")
    if len(include) > 256:
        raise SystemExit(f"plan {plan}: {len(include)} jobs exceed the 256-job matrix limit")
    return {"include": include}


if __name__ == "__main__":
    name = plan_name(sys.argv[1])
    print(f"plan={name}")
    print(f"matrix={json.dumps(matrix(name))}")
