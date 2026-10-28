import sys
import copy
import multiprocessing as mp
import os
import re
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import random

from config import Config
from neural_net import CartNN, mutate_neat, crossover_neat, compatibility_distance, Genome, Species
from simulation import DoublePendulumEnv
from logger import ExperimentLogger
from curriculum import CurriculumManager

# =============================================================================
# WORKER PROCESS
# =============================================================================

_WORKER_ENV: Optional[DoublePendulumEnv] = None

def worker_init() -> None:
    global _WORKER_ENV
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    _WORKER_ENV = DoublePendulumEnv(visualizer=False)

def evaluate_agent(genome: Genome, physics: Dict[str, float], seed: int) -> Tuple[float, float, int]:
    """
    Evaluates agent performance based on the longest consecutive upright duration
    penalized by control instability (jitter) during that specific duration.
    
    Formula: Score = (Time Over Threshold) / (1 + Sum(abs(Action_delta)))
    """
    global _WORKER_ENV
    
    model = CartNN(genome)
    model.eval()
    
    total_score = 0.0
    total_ratio = 0.0
    total_max_steps = 0
    max_episode_steps = int(Config.DURATION / Config.DT)

    for i in range(Config.NUM_TRIALS):
        current_seed = seed + i
        _WORKER_ENV.reset(
            g=physics['g'],
            d=physics['drag'],
            max_force=physics['max_force'],
            seed=current_seed
        )
        model.reset_state()

        obs, done, _, _, _, _ = _WORKER_ENV.step(0.0)
        prev_action = 0.0
        
        # Streak tracking
        current_streak = 0
        current_jitter = 0.0
        best_streak = 0
        best_jitter = 0.0
        
        trial_upright_count = 0

        with torch.inference_mode():
            for _ in range(max_episode_steps):
                # Network inference
                inp = torch.as_tensor(obs, dtype=torch.float32)
                action = model(inp).item()
                
                # Execute step
                obs, done, _, is_upright, _, _ = _WORKER_ENV.step(action)
                
                # Calculate control effort (jitter)
                action_diff = abs(action - prev_action)
                prev_action = action
                
                if is_upright:
                    current_streak += 1
                    current_jitter += action_diff
                    trial_upright_count += 1
                else:
                    # Streak broken: save if it was the best so far
                    if current_streak > best_streak:
                        best_streak = current_streak
                        best_jitter = current_jitter
                    
                    # Reset counters
                    current_streak = 0
                    current_jitter = 0.0
                
                if done: 
                    break
        
        # Final check in case episode ended during a streak
        if current_streak > best_streak:
            best_streak = current_streak
            best_jitter = current_jitter

        # Apply Scoring Formula
        time_over_threshold = best_streak * Config.DT
        score = time_over_threshold / (1.0 + best_jitter)

        total_score += score
        total_ratio += (trial_upright_count / max_episode_steps)
        total_max_steps = max(total_max_steps, best_streak)

    return (
        total_score / Config.NUM_TRIALS, 
        total_ratio / Config.NUM_TRIALS, 
        total_max_steps
    )

# =============================================================================
# TRAINER
# =============================================================================

class EvolutionTrainer:
    def __init__(self):
        resume_idx = self._find_latest_run()
        self.logger = ExperimentLogger(run_index=resume_idx)
        self.curriculum = CurriculumManager()
        
        self.population: List[CartNN] = []
        self.species: List[Species] = []
        self.gen_idx = 1
        
        self.best_ever_score = -9999.0
        self.stagnation_counter = 0
        self.current_sigma = Config.MUT_SIGMA
        self.sigma_limit = Config.MUT_SIGMA_STAGNATION
        self.species_counter = 0
        self.speciation_threshold = 3.0

        self._recover_or_initialize()

    def _find_latest_run(self) -> Optional[int]:
        models_dir = Path("models")
        if not models_dir.exists(): return None
        pattern = re.compile(rf"^{re.escape(Config.RUN_NAME)}_(\d+)$")
        max_idx = -1
        found = False
        for p in models_dir.iterdir():
            if p.is_dir() and (p / "latest_checkpoint.pth").exists():
                match = pattern.match(p.name)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                        found = True
        return max_idx if found else None

    def _recover_or_initialize(self) -> None:
        chk_path = self.logger.paths["checkpoint"]
        if chk_path.exists():
            data = ExperimentLogger.load_checkpoint(str(chk_path))
            if data:
                # Restore Key Components
                self.population = data['population']
                self.species = data.get('species', [])
                self.gen_idx = data['epoch'] + 1
                
                # Restore Physics/Curriculum
                physics = data.get('physics', {})
                self.curriculum.difficulty = physics.get('difficulty', 0.0)
                
                # Restore Meta State
                extra = data.get('extra_state', {})
                self.stagnation_counter = extra.get('stagnation', 0)
                self.current_sigma = extra.get('sigma', 0.1)
                self.best_ever_score = extra.get('best_score', -9999.0)
                self.speciation_threshold = extra.get('spec_threshold', 3.0)
                
                # Update species counter based on max existing ID
                max_sid = max([s.id for s in self.species]) if self.species else 0
                self.species_counter = max_sid + 1

                print(f"    -> Resumed Gen: {self.gen_idx} | Diff: {self.curriculum.difficulty:.2f}")
                print(f"    -> Species: {len(self.species)} | Pop: {len(self.population)}")
                return

        print("[*] Starting Fresh Training Run.")
        proto_genome = Genome(Config.INPUT_DIM, 1)
        self.population = [CartNN(proto_genome.copy()) for _ in range(Config.POPULATION_SIZE)]
        self.curriculum.difficulty = 0.0

    def evaluate_population(self, pool) -> Tuple[List[float], List[float], List[int], Dict]:
        physics = self.curriculum.get_params()
        seed = Config.SEED
        tasks = [(agent.genome, physics, seed) for agent in self.population]
        results = pool.starmap(evaluate_agent, tasks)
        return ([r[0] for r in results], [r[1] for r in results], [r[2] for r in results], physics)

    def calculate_fitness(self, scores: List[float]) -> List[float]:
        fitnesses = []
        for agent, score in zip(self.population, scores):
            base_performance = max(1e-6, score)
            complexity_factor = 1.0 + (agent.param_count * Config.BASE_COMPLEXITY_PENALTY)
            fitnesses.append(base_performance / complexity_factor)
        return fitnesses

    def update_heuristics(self, best_score: float, worst_elite_performance: float) -> Tuple[str, str]:
        if best_score > self.best_ever_score + 1.0:
            self.best_ever_score = best_score
            self.stagnation_counter = 0
            self.current_sigma = Config.MUT_SIGMA
            record_flag = "(*)"
        else:
            self.stagnation_counter += 1
            record_flag = ""

        if self.stagnation_counter > self.sigma_limit:
            stuck_time = self.stagnation_counter - self.sigma_limit - 1
            if stuck_time % 10 == 0:
                self.current_sigma = Config.MUT_SIGMA * min(1 + self.stagnation_counter/10, 5)
            else:
                self.current_sigma = max(Config.MUT_SIGMA, self.current_sigma * 0.9)

        curr_flag = self.curriculum.update(worst_elite_performance)
        return record_flag, curr_flag

    def speciate_population(self, target_species: int = Config.SPECIES_TARGET, step_size: float = Config.SPECIES_THRESH):
        for s in self.species:
            s.reset() # Clears members, keeps representative/stats

        for agent in self.population:
            found = False
            for s in self.species:
                if compatibility_distance(agent.genome, s.representative.genome) < self.speciation_threshold:
                    s.add_member(agent)
                    found = True
                    break
            if not found:
                self.species_counter += 1
                self.species.append(Species(agent, self.species_counter))

        self.species = [s for s in self.species if s.members]

        if len(self.species) < target_species:
            self.speciation_threshold = max(0.5, self.speciation_threshold - step_size)
        elif len(self.species) > target_species:
            self.speciation_threshold += step_size

    def evolve_step(self, fitnesses: List[float]) -> None:
        for agent, fit in zip(self.population, fitnesses):
            agent.fitness = max(0.001, fit)

        self.speciate_population()

        total_adj_fitness = 0.0
        surviving_species = []
        best_species_id = -1
        global_max = -1.0
        
        # Identify Elite Species
        for s in self.species:
            s.sort_members()
            if s.members[0].fitness > global_max:
                global_max = s.members[0].fitness
                best_species_id = s.id

        # Calculate Adjusted Fitness
        for s in self.species:
            s.update_stagnation()
            is_elite = (s.id == best_species_id)
            
            if (s.stagnation < 25 or is_elite or len(self.species) == 1) and len(s.members) > 0:
                s.adjust_fitness()
                s_sum = sum(m.fitness_adj for m in s.members)
                s.avg_fitness_adj = s_sum / len(s.members)
                total_adj_fitness += s_sum
                surviving_species.append(s)

        if not surviving_species:
            # Emergency: if all die, keep the best one
            surviving_species = self.species[:1]
            total_adj_fitness = 1.0

        # Allocation
        target_size = Config.POPULATION_SIZE
        species_counts = {}
        remainders = {}
        total_allocated = 0
        
        for s in surviving_species:
            share = (sum(m.fitness_adj for m in s.members) / total_adj_fitness) * target_size
            count = int(share)
            if count == 0: count = 1
            species_counts[s.id] = count
            remainders[s.id] = share - count
            total_allocated += count

        if total_allocated != target_size:
            diff = target_size - total_allocated
            if diff > 0:
                sorted_rem = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
                for i in range(diff):
                    species_counts[sorted_rem[i % len(sorted_rem)][0]] += 1
            elif diff < 0:
                sorted_counts = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
                for i in range(abs(diff)):
                    sid = sorted_counts[i % len(sorted_counts)][0]
                    if species_counts[sid] > 1: species_counts[sid] -= 1

        # Reproduction
        next_gen_pop = []
        for s in surviving_species:
            count = species_counts.get(s.id, 0)
            if count == 0: continue
            
            # Elitism
            next_gen_pop.append(copy.deepcopy(s.members[0]))
            curr_idx = 1
            
            while curr_idx < count:
                p1 = self._tournament(s)
                p2 = self._tournament(s)
                child_genome = crossover_neat(p1.genome, p2.genome, p1.fitness, p2.fitness)
                child_genome = mutate_neat(child_genome, Config, sigma=self.current_sigma)
                next_gen_pop.append(CartNN(child_genome))
                curr_idx += 1

        self.population = next_gen_pop
        self.species = surviving_species # Keep only survivors for next gen history

    def _tournament(self, species) -> 'CartNN':
        pool = species.members[:max(1, int(len(species.members) * 0.75))]
        return max(random.sample(pool, min(3, len(pool))), key=lambda x: x.fitness)

    def train(self) -> None:
        print(f"Config: {Config.NUM_WORKERS} Workers | {Config.POPULATION_SIZE} Agents")
        print("=" * 80)

        with mp.Pool(processes=Config.NUM_WORKERS, initializer=worker_init, maxtasksperchild=50) as pool:
            try:
                for gen in range(self.gen_idx, Config.NUM_GENERATIONS + 1):
                    self.gen_idx = gen
                    
                    scores, ratios, steps, physics = self.evaluate_population(pool)
                    fitnesses = self.calculate_fitness(scores)
                    
                    # Metrics Calculation
                    avg_score = np.mean(scores)
                    best_score = np.max(scores)
                    best_time = np.max(steps) * Config.DT

                    # metrics for top performers
                    sorted_ratios = np.sort(ratios)[::-1]
                    n = Config.PASS_COUNT
                    if len(sorted_ratios) < n:
                        return "ERR  "
                    worst_elite_performance = sorted_ratios[n - 1]
                    avg_elite_performance = np.mean(sorted_ratios[:n])  # Mean of top N performers

                    # elite_ratio = np.percentile(ratios, Config.PASS_PERCENTILE) * 100

                    avg_nodes = np.mean([len(a.genome.nodes) for a in self.population])
                    avg_conns = np.mean([len(a.genome.connections) for a in self.population])
                    
                    rec_flag, curr_flag = self.update_heuristics(best_score, worst_elite_performance)
                    
                    self.logger.log_generation(gen, physics, scores, self.population, self.species)
                    self.save_state(physics)
                    
                    # Sp: Species Count | Sig: Mutation Rate | Cplx: Nodes/Connections | 
                    # AE: Average elite performance | WE: Worst elite performance
                    print(f"Gen {gen:04d} | Diff {physics['difficulty']:.3f} {curr_flag} | "
                          f"Best {best_score:5.1f}{rec_flag:3} | "
                          f"Sp {len(self.species):02d} | Sig {self.current_sigma:.2f} | "
                          f"Cplx {avg_nodes:.1f}N/{avg_conns:.1f}C | "
                          f"AE {avg_elite_performance:4.3f} | "
                          f"WE {worst_elite_performance:4.3f} | Best Time {best_time:4.3f}s")
                    self.evolve_step(fitnesses)

            except KeyboardInterrupt:
                print("\n[!] Training paused by user.")
            except Exception as e:
                print(f"\n[!] CRITICAL FAILURE: {e}")
                raise
            finally:
                self.save_state(physics)
                print("[*] State saved. Exiting.")

    def save_state(self, physics: Dict) -> None:
        extra_state = {
            'stagnation': self.stagnation_counter,
            'sigma': self.current_sigma,
            'best_score': self.best_ever_score,
            'spec_threshold': self.speciation_threshold
        }
        self.logger.save_checkpoint(
            self.population, 
            self.species, 
            self.gen_idx, 
            physics, 
            extra_state
        )

if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass 
    EvolutionTrainer().train()