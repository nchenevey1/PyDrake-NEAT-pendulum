"""Run many (config, seed) jobs with a fixed number of parallel workers.

Finished jobs (report.json present) are skipped; interrupted jobs resume from
their checkpoint, so the sweep can simply be re-launched after a restart.

  python sweep.py --jobs 4 --seeds 0 1 configs/*.yaml
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    queue = []
    for seed in args.seeds:            # all configs for seed 0 first, then seed 1, ...
        for c in args.configs:
            name = yaml.safe_load(open(c)).get("name") or Path(c).stem
            d = Path(args.runs_dir) / f"{name}_s{seed}"
            if (d / "report.json").exists():
                continue
            cmd = [sys.executable, "run.py", c, "--seed", str(seed), "--runs-dir", args.runs_dir]
            if (d / "checkpoint.pkl").exists():
                cmd.append("--resume")
            for s in args.set:
                cmd += ["--set", s]
            queue.append((d, cmd))
    print(f"[sweep] {len(queue)} jobs, {args.jobs} parallel", flush=True)
    running = []
    while queue or running:
        while queue and len(running) < args.jobs:
            d, cmd = queue.pop(0)
            d.mkdir(parents=True, exist_ok=True)
            log = open(d / "stdout.log", "a")
            running.append((d, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[sweep] start {d.name}", flush=True)
        for item in list(running):
            d, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print(f"[sweep] {'done' if p.returncode == 0 else f'FAILED rc={p.returncode}'} {d.name}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
