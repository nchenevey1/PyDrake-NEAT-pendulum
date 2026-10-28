import argparse
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from config import Config
from neural_net import CartNN
from simulation import DoublePendulumEnv
from curriculum import CurriculumManager

# =============================================================================
# UTILITIES & IO
# =============================================================================

class ModelLoader:
    """Handles path resolution and loading of both snapshots and full checkpoints"""

    @staticmethod
    def resolve_path(input_val: str) -> Path:
        """Resolves run index or raw path to valid checkpoint file"""
        if input_val.isdigit():
            run_idx = int(input_val)
            run_dir = Path("models") / f"{Config.RUN_NAME}_{run_idx:03d}"
            
            # Priority: Best Model -> Latest Checkpoint
            if (best := run_dir / "best_model.pth").exists():
                return best
            if (latest := run_dir / "latest_checkpoint.pth").exists():
                print(f"[Info] 'best_model.pth' missing. Using latest checkpoint.")
                return latest
            
            raise FileNotFoundError(f"No models found in {run_dir}")

        path = Path(input_val)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    @staticmethod
    def load_data(path: Path, agent_idx: int = 0) -> Tuple[Dict[str, Any], CartNN]:
        """Loads checkpoint and reconstructs agent model"""
        print(f"[*] Loading: {path}")
        
        try:
            # weights_only=False needed for Genome/CartNN objects
            data = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Checkpoint load failed: {e}")

        meta = {'epoch': data.get('epoch', 0)}
        
        # 1. Physics Context
        # Prefer explicit physics params if available, else calc from difficulty
        if 'physics' in data and isinstance(data['physics'], dict):
            meta['physics'] = data['physics']
        else:
            diff = data.get('difficulty', 0.0)
            meta['physics'] = CurriculumManager(diff).get_params()

        # 2. Agent Reconstruction
        if 'population' in data:
            # Full Checkpoint (List of Objects or Dicts)
            pop = data['population']
            if not (0 <= agent_idx < len(pop)):
                raise IndexError(f"Agent index {agent_idx} out of bounds (Size: {len(pop)})")
            
            target = pop[agent_idx]
            
            # Check if target is Object (new format) or Dict (old format)
            if isinstance(target, CartNN):
                agent = target
            elif isinstance(target, dict):
                agent = CartNN(target['genome'])
                agent.load_state_dict(target['state_dict'])
            else:
                raise ValueError("Unknown population format.")
                
            meta['type'] = 'Population Snapshot'

        elif 'model_state_dict' in data:
            # Single Best Model Archive
            genome = data.get('genome')
            if not genome:
                raise ValueError("Checkpoint missing genome data.")
                
            agent = CartNN(genome)
            agent.load_state_dict(data['model_state_dict'])
            meta['type'] = 'Single Model Archive'
            
        else:
            raise ValueError("Unknown checkpoint format.")

        agent.eval()
        return meta, agent

# =============================================================================
# SIMULATION SESSION
# =============================================================================

class PlaybackSession:
    """Manages environment instantiation and visualization loop"""

    def __init__(self, meta: Dict[str, Any], agent: CartNN):
        self.meta = meta
        self.agent = agent
        self.params = meta['physics']
        
        self._print_manifest()

    def _print_manifest(self):
        g_len = len(self.agent.genome.nodes)
        c_len = len(self.agent.genome.connections)
        
        print("\n" + "="*60)
        print(f"  Source:      {self.meta['type']}")
        print(f"  Generation:  {self.meta['epoch']}")
        print(f"  Topology:    {g_len} Nodes | {c_len} Connections")
        print("-" * 60)
        print(f"  Difficulty:  {self.params['difficulty']:.2f}")
        print(f"  Gravity:     {self.params['g']:.2f} m/s²")
        print(f"  Drag:        {self.params['drag']:.4f}")
        print(f"  Max Force:   {self.params['max_force']:.1f} N")
        print("="*60 + "\n")

    def run(self):
        env = DoublePendulumEnv(visualizer=True, g=self.params['g'])
        env.reset(
            g=self.params['g'],
            d=self.params['drag'],
            max_force=self.params['max_force'],
            seed=Config.SEED
        )

        print(f"[*] Simulation started. Press Ctrl+C to stop.")

        obs, done, _, _, _, _ = env.step(0.0)
        step_count = 0
        prev_action = 0.0
        streak = {'curr': 0, 'max': 0}

        try:
            while not done:
                start_time = time.time()

                with torch.no_grad():
                    inp = torch.as_tensor(obs, dtype=torch.float32)
                    action = self.agent(inp).item()

                obs, done, x, is_upright, th1, th2 = env.step(action)
                step_count += 1

                # Update Stats
                if is_upright: streak['curr'] += 1
                else: streak['curr'] = 0
                streak['max'] = max(streak['max'], streak['curr'])

                if step_count % 10 == 0:
                    self._log_step(step_count, x, th1, th2, streak['max'], is_upright)

                # Pacing
                compute_time = time.time() - start_time
                time.sleep(max(0.0, Config.DT - compute_time))

        except KeyboardInterrupt:
            print("\n[!] Interrupted.")
        except Exception as e:
            print(f"\n[!] Error: {e}")

        print(f"\n\n    Best Upright Time: {streak['max'] * Config.DT:.2f}s")
        print("[Info] Done.")

    def _log_step(self, step: int, x: float, th1: float, th2: float, max_st: int, upright: bool):
        status = "UP" if upright else "---"
        sys.stdout.write(
            f"\r    Step: {step:04d} | Cart: {x:5.2f}m | "
            f"Err: {th1:5.2f}/{th2:5.2f} | Best: {max_st*Config.DT:5.2f}s | {status}"
        )
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Visualize Agent")
    parser.add_argument("id", type=str, help="Run Index (e.g. '1') OR Path to .pth")
    parser.add_argument("--agent", type=int, default=0, help="Agent Index (Population only)")
    args = parser.parse_args()

    try:
        path = ModelLoader.resolve_path(args.id)
        meta, agent = ModelLoader.load_data(path, args.agent)
        session = PlaybackSession(meta, agent)
        session.run()
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()