"""Summarize finished runs (report.json + verify.json per run directory) as a Markdown table
and a JSON file. Test numbers are reported only; selection used validation.

  python ci/summarize.py runs [--json summary.json]
"""
import argparse
import json
from pathlib import Path


def _rate(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def row(run_dir: Path) -> dict:
    rep = json.load(open(run_dir / "report.json"))
    ver = json.load(open(run_dir / "verify.json")) if (run_dir / "verify.json").exists() else {}
    cur = rep.get("curriculum", {})
    r = rep.get("report", {})
    return {
        "run": run_dir.name, "seed": rep.get("seed"), "generations": rep.get("generations"),
        "env_steps": rep.get("env_steps"), "wall_s": rep.get("wall_s"),
        "best_start_t": cur.get("best_start_t"), "final_start_t": cur.get("final_start_t"),
        "final_step": cur.get("final_step"), "backoffs": cur.get("backoffs"),
        "params": r.get("params"),
        "val": _rate(rep, "val", "success_rate"),
        "test_20s": _rate(r, "test_20s", "success_rate"),
        "test_60s": _rate(r, "test_60s", "success_rate"),
        "int8": _rate(r, "int8_20s", "success_rate"),
        "weight_noise_0.01_median": _rate(r, "weight_noise_0.01", "median"),
        "drake_20s": _rate(ver, "drake_20s", "successes"), "drake_60s": _rate(ver, "drake_60s", "successes"),
        "drake_n": _rate(ver, "drake_20s", "n"),
        "pruned_params": _rate(r, "pruned", "params"),
    }


def fmt(v, pct=False):
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.1f} %"
    return f"{v:g}" if isinstance(v, float) else str(v)


def table(rows) -> str:
    head = ("| run | closest start | final start (step) | back-offs | env steps | val | test 20 s | "
            "test 60 s | Drake 20 s / 60 s | int8 | weight noise 0.01 (median) |")
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        dr = "-" if r["drake_n"] is None else f"{r['drake_20s']}/{r['drake_n']} / {r['drake_60s']}/{r['drake_n']}"
        out.append(
            f"| {r['run']} | {fmt(r['best_start_t'])} s | {fmt(r['final_start_t'])} s ({fmt(r['final_step'])}) | "
            f"{fmt(r['backoffs'])} | {r['env_steps'] / 1e9:.2f}B | {fmt(r['val'], 1)} | {fmt(r['test_20s'], 1)} | "
            f"{fmt(r['test_60s'], 1)} | {dr} | {fmt(r['int8'], 1)} | {fmt(r['weight_noise_0.01_median'], 1)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = [row(d) for d in sorted(Path(args.runs_dir).glob("*/")) if (d / "report.json").exists()]
    n_ok = sum(1 for r in rows if (r["test_20s"] or 0) >= 0.9)
    print(table(rows))
    print(f"\n{n_ok}/{len(rows)} runs reach >= 90 % test success (20 s).")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
