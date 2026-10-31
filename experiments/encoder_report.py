"""Tables and the main figure of the encoder study (docs/ENCODER_SENSING.md).

  python experiments/encoder_report.py [--stage2 DIR]

Reads results/encoder/stage1*.jsonl, results/encoder/classic.json and (optionally) the stage-2
report JSONs in DIR (default results/encoder/stage2). Writes results/encoder/summary.md and
results/encoder/fig_success_vs_bits_{stage1,stage2}.png.
"""
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("results/encoder")
CPRS = [256, 1024, 4096, 16384]
# family -> (color, label); three categorical slots (validated all-pairs), full state in grey
FAMILY = {"fd": ("#2a78d6", "finite difference"), "hist": ("#eb6834", "position history"),
          "pos": ("#1baf7a", "RNN on positions"), "full": ("#8a8984", "full state (upper bound)")}
MARK = {"mlp:2": "o", "mlp:4": "s", "mlp:8": "D", "rnn:2": "v", "rnn:4": "^", "rnn:8": "P"}


def family(mode):
    return "hist" if mode.startswith(("hist", "raw")) else mode


def load_stage1():
    rows = []
    for p in sorted(glob.glob(str(R / "stage1*.jsonl"))):
        rows += [json.loads(l) for l in open(p)]
    best = {}
    for r in rows:   # best seed per cell
        key = (r["mode"], r["arch"], r["cpr"], r["x_res"], r["setting"])
        if key not in best or r["test_20s"] > best[key]["test_20s"]:
            best[key] = r
    return list(best.values())


def stage1_tables(rows):
    out = []
    for setting in ("ideal", "real"):
        sub = [r for r in rows if r["setting"] == setting and r["x_res"] in (0.0, 1e-4)]
        if not sub:
            continue
        title = "no delay, float force" if setting == "ideal" else "one-step delay, 8-bit force"
        out.append(f"\n**Balance only, {title}** (test success at 20 s / 60 s, 256 starts; "
                   f"**bold** = ≥ 90 % at 20 s; † = ≥ 90 % at 20 s but held loosely, hold error > 0.05 rad)\n")
        out.append("| Controller | Params | " + " | ".join(f"CPR {c}" for c in CPRS) + " | Coarsest CPR ≥ 90 % |")
        out.append("|---|---|" + "---|" * (len(CPRS) + 1))
        cells = sorted({(r["mode"], r["arch"]) for r in sub}, key=lambda c: (["full", "fd", "hist3", "hist4", "raw2", "pos"].index(c[0]), c[1]))
        for mode, arch in cells:
            rr = {r["cpr"]: r for r in sub if r["mode"] == mode and r["arch"] == arch}
            params = next(iter(rr.values()))["params"]
            cols, ok = [], []
            for c in CPRS:
                r = rr.get(0 if mode == "full" else c)
                if r is None:
                    cols.append("–"); continue
                s = f"{r['test_20s'] * 100:.0f} / {r['test_60s'] * 100:.0f}"
                if r["test_20s"] >= 0.9:
                    loose = r.get("hold_err_rad") is not None and r["hold_err_rad"] > 0.05
                    s = f"**{s}**" + ("†" if loose else "")
                    ok.append(c)
                cols.append(s)
            coarse = "any" if mode == "full" and ok else (str(min(ok)) if ok else "none")
            out.append(f"| {mode} {arch} | {params} | " + " | ".join(cols) + f" | {coarse} |")
    return "\n".join(out)


def classic_table():
    p = R / "classic.json"
    if not p.exists():
        return ""
    rows = json.load(open(p))
    out = ["\n**Classic reference through the encoders** (swing-up test 20 s on 256 starts; balance on 64; "
           "60 s on 64; α chosen on 32 validation starts)\n",
           "| Controller | Setting | Stored numbers | " + " | ".join(f"CPR {c}" for c in CPRS) + " |",
           "|---|---|---|" + "---|" * len(CPRS)]
    by = defaultdict(dict)
    for r in rows:
        by[(r["controller"], r["setting"])][r["cpr"]] = r
    for (ctl, st), rr in by.items():
        n = next(iter(rr.values()))["stored_numbers"]
        cols = [f"{rr[c]['test_20s'] * 100:.0f} / {rr[c]['test_60s_64'] * 100:.0f} (bal {rr[c]['balance_20s'] * 100:.0f})"
                if c in rr else "–" for c in CPRS]
        out.append(f"| {ctl} | {st} | {n} | " + " | ".join(cols) + " |")
    out.append("\nCells: swing-up 20 s / 60 s (balance only), in %.")
    return "\n".join(out)


def load_stage2(d):
    rows = []
    for p in sorted(glob.glob(str(Path(d) / "**" / "*.json"), recursive=True)):
        r = json.load(open(p))
        if "tag" in r:
            rows.append(r)
    return rows


def stage2_table(rows):
    if not rows:
        return ""
    out = ["\n**Stage 2: swing-up + balance (real task)**, per seed\n",
           "| Run | Params | Memory | Closest start (s) | Steps (B) | Test 20 s | 60 s | int8 | Delay switched | Drake |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["tag"]):
        d = r.get("drake")
        dr = f"{d['drake']}/{d['n']}" if isinstance(d, dict) else "–"
        out.append(f"| {r['tag']} | {r['params']} | {r['memory_words']} | {r['closest_start_s']} | {r['steps_B']} | "
                   f"{r['test_20s'] * 100:.1f} | {r['test_60s'] * 100:.1f} | {r['int8_20s'] * 100:.1f} | "
                   f"{r['delay_switched_20s'] * 100:.1f} | {dr} |")
    return "\n".join(out)


def figure(points, path, title):
    """points: dicts with family, arch, cpr (0 = full), params, s32, s8, setting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pts = [p for p in points if p["setting"] == "ideal"]
    if not pts:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)
    for ax, c in zip(axes, CPRS):
        for p in pts:
            if p["cpr"] not in (0, c):
                continue
            col = FAMILY[p["family"]][0]
            m = MARK.get(p["arch"], "o")
            ax.scatter(p["params"] * 32, p["s32"] * 100, s=60, marker=m, facecolors="none",
                       edgecolors=col, linewidths=1.6, zorder=3)
            ax.scatter(p["params"] * 8, p["s8"] * 100, s=60, marker=m, color=col,
                       edgecolors="white", linewidths=1.2, zorder=4)
        ax.axhline(90, color="#b0afa8", lw=1, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_title(f"{c} counts/rev  (Δv = {2 * np.pi / c / 0.01:.2g} rad/s)", fontsize=10)
        ax.set_xlabel("parameters × bits per parameter")
        ax.grid(True, color="#e6e5e0", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("test success (%)")
    axes[0].set_ylim(-3, 103)
    h = [Line2D([], [], marker="o", ls="", color=v[0], label=v[1]) for k, v in FAMILY.items()]
    h += [Line2D([], [], marker=m, ls="", color="#52514e", mfc="none", label=a) for a, m in MARK.items()]
    h += [Line2D([], [], marker="o", ls="", color="#52514e", mfc="none", label="float32 (hollow)"),
          Line2D([], [], marker="o", ls="", color="#52514e", label="int8 (filled)")]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", default=str(R / "stage2"))
    a = ap.parse_args()
    s1 = load_stage1()
    s2 = load_stage2(a.stage2)
    md = ["# Encoder study: summary tables (generated by experiments/encoder_report.py)\n",
          stage1_tables(s1), classic_table(), stage2_table(s2)]
    (R / "summary.md").write_text("\n".join(md) + "\n")
    pts = [{"family": family(r["mode"]), "arch": r["arch"], "cpr": r["cpr"], "params": r["params"],
            "s32": r["test_20s"], "s8": r["int8_20s"], "setting": r["setting"]}
           for r in s1 if r["mode"] in ("full", "fd", "hist3", "hist4", "pos") and not (r["mode"] == "pos" and r["arch"].startswith("mlp"))]
    figure(pts, R / "fig_success_vs_bits_stage1.png",
           "Stage 1 (balance only, no delay): test success vs controller size in bits, per encoder resolution")
    if s2:
        by = defaultdict(list)
        for r in s2:
            by[(r["sensor"]["mode"] + (str(r["sensor"]["k"]) if r["sensor"]["mode"] == "hist" else ""),
                r["arch"], r["sensor"]["cpr"], r["sensor"]["delay"])].append(r)
        pts2 = []
        for (mode, arch, cpr, delay), rr in by.items():
            b = max(rr, key=lambda r: r["test_20s"])
            pts2.append({"family": family(mode), "arch": arch, "cpr": cpr, "params": b["params"],
                         "s32": b["test_20s"], "s8": b["int8_20s"], "setting": "real" if delay else "ideal"})
        figure(pts2, R / "fig_success_vs_bits_stage2.png",
               "Stage 2 (swing-up + balance, no delay): best-of-3-seeds test success vs size in bits")
    print((R / "summary.md").read_text())


if __name__ == "__main__":
    main()
