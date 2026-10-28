import json
import time
import torch
import numpy as np
import random
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import Config
from neural_net import CartNN, Species

class ExperimentLogger:
    """
    Manages experiment tracking, artifact storage, and training state persistence.
    Saves full object states for Checkpoints (resuming) and lightweight snapshots for History.
    """

    def __init__(self, run_index: Optional[int] = None):
        self.models_root = Path("models")
        self.models_root.mkdir(parents=True, exist_ok=True)

        # Determine Run ID
        if run_index is not None:
            self.run_idx = run_index
            print(f"[Logger] Attaching to Run Index: {self.run_idx:03d}")
        else:
            self.run_idx = self._get_next_run_index()
            print(f"[Logger] Starting New Run Index: {self.run_idx:03d}")

        self.run_id = f"{Config.RUN_NAME}_{self.run_idx:03d}"
        self.root_dir = self.models_root / self.run_id
        
        self.paths = {
            "history": self.root_dir / "history",
            "logs": self.root_dir / "logs.jsonl",
            "checkpoint": self.root_dir / "latest_checkpoint.pth",
            "best_model": self.root_dir / "best_model.pth"
        }

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.paths["history"].mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.best_ever_score = -float('inf')

        if run_index is None:
            self._log_static_config()
        
        print(f"[Logger] Artifacts:       {self.root_dir}")

    def _get_next_run_index(self) -> int:
        """Finds the next available run index"""
        pattern = re.compile(rf"^{re.escape(Config.RUN_NAME)}_(\d+)$")
        max_idx = 0
        for p in self.models_root.iterdir():
            if p.is_dir():
                match = pattern.match(p.name)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
        return max_idx + 1

    def _log_static_config(self):
        """Logs configuration at start of new run"""
        config_data = {k: v for k, v in Config.__dict__.items() 
                       if k.isupper() and not k.startswith("_")}
        self._write_log_entry({
            "type": "CONFIG",
            "timestamp": time.time(),
            "data": config_data
        })

    def _write_log_entry(self, entry: Dict[str, Any]):
        with open(self.paths["logs"], 'a') as f:
            f.write(json.dumps(entry) + "\n")

    # =========================================================================
    # METRICS & ARCHIVING
    # =========================================================================

    def log_generation(self, 
                       gen_idx: int, 
                       physics: Dict[str, float], 
                       scores: List[float], 
                       population: List[CartNN],
                       species: List[Species]):
        """Logs metrics and archives the best model of the generation"""
        scores_np = np.array(scores)
        best_idx = np.argmax(scores_np)
        current_best_score = float(scores_np[best_idx])
        
        # Metrics
        stats_perf = {
            "max": round(float(np.max(scores_np)), 2),
            "avg": round(float(np.mean(scores_np)), 2),
            "std": round(float(np.std(scores_np)), 2),
        }

        stats_topo = {
            "avg_nodes": round(float(np.mean([len(p.genome.nodes) for p in population])), 2),
            "avg_conns": round(float(np.mean([len(p.genome.connections) for p in population])), 2),
        }

        stats_species = {
            "count": len(species),
            "distribution": [len(s.members) for s in species]
        }

        self._write_log_entry({
            "type": "GEN",
            "gen": gen_idx,
            "wall_time": round(time.time() - self.start_time, 2),
            "physics": physics,
            "performance": stats_perf,
            "topology": stats_topo,
            "species": stats_species
        })

        # Archive Best Model (Lightweight Snapshot)
        best_agent = population[best_idx]
        self._archive_model(gen_idx, best_agent, current_best_score, physics)

    def _archive_model(self, gen: int, model: CartNN, score: float, physics: Dict):
        """Saves a lightweight snapshot (state_dict) for history/viz"""
        snapshot = {
            'epoch': gen,
            'model_state_dict': model.state_dict(),
            'genome': model.genome,
            'score': score,
            'physics': physics,
            'type': 'snapshot'
        }

        # Save to history
        history_path = self.paths["history"] / f"gen_{gen:05d}_score_{int(score)}.pth"
        torch.save(snapshot, history_path)

        # Update best model
        if score > self.best_ever_score:
            self.best_ever_score = score
            torch.save(snapshot, self.paths["best_model"])

    # =========================================================================
    # STATE PERSISTENCE (CHECKPOINTING)
    # =========================================================================

    def save_checkpoint(self, 
                        population: List[CartNN], 
                        species: List[Species], 
                        gen_idx: int, 
                        physics: Dict[str, float], 
                        extra_state: Dict = None):
        """
        Saves the evolutionary state (objects included) to allow resuming
        """
        checkpoint_data = {
            'epoch': gen_idx,
            'physics': physics,  # Save physics params
            'population': population, # Save objects to preserve references
            'species': species,       # Save objects to preserve age/stagnation
            'rng_state': self._get_rng_states(),
            'extra_state': extra_state or {},
            'type': 'checkpoint'
        }

        # Atomic Save
        tmp_path = self.paths["checkpoint"].with_suffix(".tmp")
        torch.save(checkpoint_data, tmp_path)
        tmp_path.rename(self.paths["checkpoint"])

    @staticmethod
    def load_checkpoint(path_str: str) -> Optional[Dict[str, Any]]:
        """Loads a checkpoint, restoring RNG state"""
        path = Path(path_str)
        if not path.exists():
            return None

        print(f"[Logger] Resuming from: {path.name}")
        try:
            # weights_only=False required for custom classes
            data = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"[Logger] CRITICAL: Failed to load checkpoint. {e}")
            return None

        # Restore RNG
        ExperimentLogger._set_rng_states(data.get('rng_state', {}))
        
        return data

    # =========================================================================
    # RNG HELPERS
    # =========================================================================

    def _get_rng_states(self) -> Dict[str, Any]:
        states = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate()
        }
        return states

    @staticmethod
    def _set_rng_states(states: Dict[str, Any]):
        if not states: return
        if 'torch' in states: torch.set_rng_state(states['torch'])
        if 'numpy' in states: np.random.set_state(states['numpy'])
        if 'python' in states: random.setstate(states['python'])