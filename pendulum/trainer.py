"""
Generic training loop: any `Method` + physics curriculum + logging + checkpoints
+ periodic validation on the *target* physics.

Run directory layout (runs/<name>_s<seed>/):
  config.yaml          resolved config
  log.jsonl            one GEN record per generation, EVAL records every eval.every gens
  checkpoint.pkl       full state (method, curriculum, RNGs, counters) for exact resume
  best_val.json        best policy so far on the target-physics validation set
  champion_latest.json latest top champion
  report.json          final test-set report of best_val (written at the end)
"""

import json
import os
import pickle
import signal
import time
from pathlib import Path
from typing import Optional

import numba
import numpy as np
import yaml

from .curriculum import Curriculum, CurriculumConfig
from .evaluation import evaluate_policies, full_report, selection_key, val_ics
from .fitness import policy_scores, upright_ratio
from .methods import make_method
from .policy import GraphPolicy, pack
from .rollout import S_BEST_STREAK, rollout
from .task import DEFAULT_TASK, OBS_DIM, TARGET_PHYSICS, sample_initial_states


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path, resume: bool = False):
        self.cfg = cfg
        self.dir = Path(run_dir)
        self.task = DEFAULT_TASK
        self.ckpt_path = self.dir / "checkpoint.pkl"
        numba.set_num_threads(max(1, min(int(cfg["threads"]), numba.config.NUMBA_NUM_THREADS)))
        if resume and self.ckpt_path.exists():
            self._load()
            # allow extending the budget on resume
            self.state["cfg"]["budget"] = cfg["budget"]
            print(f"[resume] {self.dir} at generation {self.state['gen']}")
        else:
            if self.dir.exists() and (self.dir / "log.jsonl").exists() and not resume:
                raise FileExistsError(f"{self.dir} already has a run; use --resume or another name/seed")
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "config.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            rng = np.random.default_rng(cfg["seed"])
            np.random.seed(cfg["seed"])  # pycma draws from the global numpy RNG
            self.state = {
                "cfg": cfg,
                "gen": 0,
                "env_steps": 0,
                "episodes": 0,
                "eval_episodes": 0,
                "wall": 0.0,
                "rng": rng,
                "method": make_method(cfg["method"], cfg["method_cfg"], OBS_DIM, rng),
                "curriculum": Curriculum(CurriculumConfig(**cfg["curriculum"])),
                "best_val": None,           # (key, summary, policy dict, gen)
                "milestones": {},
                "target_reached_gen": None,
            }
            if self.state["curriculum"].at_target:
                self.state["target_reached_gen"] = 0
        self.val_ics = val_ics(cfg["eval"]["n_val"], self.task)

    # ------------------------------------------------------------ persistence
    def _save(self):
        st = dict(self.state)
        st["np_global_rng"] = np.random.get_state()
        tmp = self.ckpt_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.ckpt_path)

    def _load(self):
        with open(self.ckpt_path, "rb") as f:
            st = pickle.load(f)
        np.random.set_state(st.pop("np_global_rng"))
        self.state = st

    def _log(self, rec: dict):
        with open(self.dir / "log.jsonl", "a") as f:
            f.write(json.dumps(_jsonable(rec)) + "\n")

    # --------------------------------------------------------------- budget
    def _budget_left(self, seg_start: float) -> bool:
        b = self.state["cfg"]["budget"]
        st = self.state
        if b.get("generations") is not None and st["gen"] >= b["generations"]:
            return False
        if b.get("env_steps") is not None and st["env_steps"] >= b["env_steps"]:
            return False
        if b.get("wall_hours") is not None and st["wall"] + time.time() - seg_start >= 3600 * b["wall_hours"]:
            return False
        return True

    # ------------------------------------------------------------ main loop
    def run(self) -> Optional[dict]:
        st, cfg, task = self.state, self.state["cfg"], self.task
        method, curr, rng = st["method"], st["curriculum"], st["rng"]
        tcfg = cfg["train"]
        seg_start = time.time()
        stop = {"flag": False}

        def _sigterm(*_):
            stop["flag"] = True
        old = signal.signal(signal.SIGTERM, _sigterm)
        try:
            while self._budget_left(seg_start) and not stop["flag"]:
                t0 = time.time()
                phys = curr.params()
                pols = method.ask()
                ics = sample_initial_states(tcfg["n_trials"], rng, task.ic_noise)
                stats, steps = rollout(pack(pols), ics, phys, task)
                t_sim = time.time() - t0
                scores = policy_scores(stats, tcfg["fitness"], task)
                sizes = np.array([p.n_params for p in pols])
                fitness = scores / (1.0 + tcfg["size_penalty"] * sizes)
                method.tell(fitness, scores, stats)
                ratios = upright_ratio(stats, task)
                advanced = curr.update(ratios)
                if advanced:
                    method.on_physics_change()
                if curr.at_target and st["target_reached_gen"] is None:
                    st["target_reached_gen"] = st["gen"] + 1
                st["gen"] += 1
                st["env_steps"] += steps
                st["episodes"] += stats.shape[0] * stats.shape[1]
                wall = st["wall"] + time.time() - seg_start
                ib = int(np.argmax(fitness))
                rec = {
                    "type": "GEN", "gen": st["gen"], "wall": round(wall, 2),
                    "env_steps": st["env_steps"], "episodes": st["episodes"],
                    "phys": {k: round(v, 5) for k, v in curr.state().items()},
                    "advanced": advanced,
                    "score_max": round(float(scores.max()), 4),
                    "score_mean": round(float(scores.mean()), 4),
                    "score_p90": round(float(np.percentile(scores, 90)), 4),
                    "fit_max": round(float(fitness.max()), 4),
                    "ratio_max": round(float(ratios.max()), 4),
                    "best_params": int(sizes[ib]),
                    "best_streak_s": round(float(stats[ib, :, S_BEST_STREAK].mean() * task.dt), 3),
                    "mean_params": round(float(sizes.mean()), 2),
                    "pop": len(pols),
                    "t_gen": round(time.time() - t0, 3), "t_sim": round(t_sim, 3),
                    "info": method.info(),
                }
                self._log(rec)
                if cfg["print_every"] and st["gen"] % cfg["print_every"] == 0:
                    self._print(rec)
                last = not self._budget_left(seg_start)
                if st["gen"] % cfg["eval"]["every"] == 0 or last:
                    self.validate()
                if st["gen"] % cfg["checkpoint_every"] == 0 or last:
                    st["wall"] = st["wall"] + time.time() - seg_start
                    seg_start = time.time()
                    self._save()
        except KeyboardInterrupt:
            stop["flag"] = True
        finally:
            signal.signal(signal.SIGTERM, old)
            st["wall"] = st["wall"] + time.time() - seg_start
            self._save()
        if stop["flag"]:
            print(f"[stopped] checkpoint saved at generation {st['gen']}")
            return None
        return self.finish()

    def _print(self, r):
        p = r["phys"]
        info = " ".join(f"{k}={v}" for k, v in r["info"].items())
        bv = self.state["best_val"]
        bvs = f"val {bv[1]['success_rate']:.2f}@{bv[2]['n_params']}p" if bv else "val -"
        print(f"g{r['gen']:05d} d={p['difficulty']:.3f} (g={p['g']:.2f} drag={p['drag']:.2f} F={p['max_force']:.0f})"
              f"{'^' if r['advanced'] else ' '} score max {r['score_max']:.2f} mean {r['score_mean']:.3f} "
              f"ratio {r['ratio_max']:.2f} best {r['best_params']}p | {bvs} | {info} | "
              f"{r['env_steps'] / 1e6:.0f}M steps {r['wall'] / 60:.1f}min", flush=True)

    # ------------------------------------------------------------ validation
    def validate(self):
        st = self.state
        champs = [p.compacted() for p in st["method"].champions()]
        if not champs:
            return
        res = evaluate_policies(champs, self.val_ics, TARGET_PHYSICS, self.task)
        st["eval_episodes"] = st.get("eval_episodes", 0) + len(champs) * len(self.val_ics)
        keys = [selection_key(r, p.n_params) for r, p in zip(res, champs)]
        i = max(range(len(keys)), key=lambda j: keys[j])
        best = (keys[i], res[i], champs[i].to_dict(), st["gen"])
        improved = st["best_val"] is None or keys[i] > tuple(st["best_val"][0])
        if improved:
            st["best_val"] = best
            champs[i].save(self.dir / "best_val.json")
        champs[0].save(self.dir / "champion_latest.json")
        sr = res[i]["success_rate"]
        for thr in (0.01, 0.5, 0.9, 1.0):
            k = f"val_success>={thr}"
            if sr >= thr and k not in st["milestones"]:
                st["milestones"][k] = {"gen": st["gen"], "env_steps": st["env_steps"],
                                       "wall": st["wall"]}
        self._log({"type": "EVAL", "gen": st["gen"], "env_steps": st["env_steps"],
                   "champions": [{"params": p.n_params, **r} for p, r in zip(champs, res)],
                   "best_val": {"gen": st["best_val"][3], "params": st["best_val"][2]["n_params"],
                                **st["best_val"][1]},
                   "improved": improved})

    # ---------------------------------------------------------------- final
    def finish(self) -> dict:
        st = self.state
        summary = {
            "name": st["cfg"]["name"], "method": st["cfg"]["method"], "seed": st["cfg"]["seed"],
            "generations": st["gen"], "env_steps": st["env_steps"], "episodes": st["episodes"],
            "wall_s": round(st["wall"], 1), "final_difficulty": st["curriculum"].difficulty,
            "target_reached_gen": st["target_reached_gen"], "milestones": st["milestones"],
        }
        if st["best_val"] is not None:
            pol = GraphPolicy.from_dict(st["best_val"][2])
            rep = full_report(pol)
            pr = rep.pop("_pruned_policy", None)
            if pr is not None:
                pr.save(self.dir / "best_val_pruned.json")
            summary["best_val_gen"] = st["best_val"][3]
            summary["val"] = st["best_val"][1]
            summary["report"] = rep
        with open(self.dir / "report.json", "w") as f:
            json.dump(_jsonable(summary), f, indent=1)
        return summary
