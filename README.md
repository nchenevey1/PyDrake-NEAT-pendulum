# Compact Curriculum Learning: Cart and Double Pendulum

Project inspired by: https://github.com/johnBuffer/Pendulum-NEAT/tree/main

This project implements a custom **NeuroEvolution of Augmenting Topologies (NEAT)** strategy coupled with a dynamic **Physics Curriculum**. It utilizes **PyDrake** for simulation and **PyTorch** for graph compilation.
The goal is to observe if a small neural topology can solve complex control tasks (like a cart and double pendulum) if trained via an evolutionary strategy that gradually increases in difficulty.

#### Initial learned swing up under low gravity and high drag:
https://github.com/user-attachments/assets/5bb90e0b-ec8d-48d6-ac88-f8fd59f891be

## Table of Contents

* <a href="#goal">The Goal</a>
* <a href="#approach">The Approach</a>
* <a href="#architecture">System Architecture</a>
* <a href="#installation">Installation & Usage</a>

## <a id="goal" href="#toc">The Goal</a>

The objective is to train a neural network to swing up and balance using a cart and double pendulum system.

* **Initial State:** Hanging downwards (stable equilibrium)
* **Target State:** Vertically upright (unstable equilibrium)
* **Constraint:** The agent must minimize network complexity (nodes/connections) while maximizing stability

## <a id="approach" href="#toc">The Approach</a>

### Evolutionary Strategy (Custom NEAT)

Unlike standard Reinforcement Learning which updates fixed weights via gradients, this project evolves both the **weights** and the **topology** of the network.

* **Genotype:** A sparse list of `NodeGenes` and `ConnectionGenes`
* **Phenotype:** A compiled PyTorch `nn.Module` derived from the genome
* **Operators:**
  * **Structural Mutation:** Adding nodes or adding new connections
  * **Parameter Mutation:** Gaussian perturbation of synapse weights and node biases.
  * **Crossover:** Historical marker based recombination (aligns matching genes between parents)
  * **Speciation:** Genetic diversity is maintained via tournament selection and fitness sharing logic

### The Physics Curriculum

The system is chaotic and difficult to explore randomly. To aid learning, the physics engine (PyDrake) is manipulated to create an easier learning task. The curriculum advances autonomously based on population success rates (90th percentile performance).

**Curriculum Phases:**

1. **Gravity Ramp (1.00 - 9.81):** Start with low gravity, slowing down the system's dynamics to make the "swing" easier to discover
2. **Drag Reduction (3.0 - 0.0):** High aerodynamic drag is initially applied to dampen velocity and prevent "spinning out" (As the agent learns, the air becomes "thinner")

### Observation Space
The agents operate on a state vector containing cart position, pole angles, angular velocities, and relative pole alignment:

$$
S = [x, \dot{x}, sin(\theta_1), cos(\theta_1), \omega_1, sin(\theta_2), cos(\theta_2), \omega_2, \mathcal{A}]
$$

**Where:**
* x - Cart position
* $\theta_1$ - Inner pole angle 
* $\omega_1$ - Inner pole angular velocity
* $\theta_2$ - Outer pole angle
* $\omega_2$ - Outer pole angular velocity.
* $\mathcal{A}$ - Pole alignment, cosine distance between the two poles:

$$
\mathcal{A} = (\sin\theta_1 \cdot \sin\theta_2) + (\cos\theta_1 \cdot \cos\theta_2)
$$


The output is a continuous scalar representing force applied to the cart.

## <a id="architecture" href="#toc">System Architecture</a>


| Module | Description |
| --- | --- |
| **`neural_net.py`** | Implements the Genome data structure and the `CartNN` PyTorch module. Handles the translation from sparse genes to dense tensor operations. |
| **`curriculum.py`** | Manages the difficulty index (0.0 - 1.0). Calculates gravity and drag based on current progress. |
| **`simulation.py`** | A wrapper around **PyDrake**. Builds the `MultibodyPlant`, applies `AerodynamicActuator` forces, and handles observation normalization. |
| **`train.py`** | Orchestrates the evolutionary process using `multiprocessing`. Manages evaluation, selection, mutation, and checkpoints. |
| **`config.py`** | Configuration for physics limits, mutation rates, and population size. |

## <a id="installation" href="#toc">Installation and Usage</a>

### Prerequisites

* Python 3.8+
* PyTorch
* PyDrake
* NumPy

```bash
pip install torch numpy drake
```

### Training

To start a fresh evolutionary run:

```bash
python train.py
```

* The system creates a new run ID (e.g., `compact_curriculum_001`) in the `models/` directory
* Checkpoints are saved every generation
* **Note:** Training uses `multiprocessing`

### Visualization

To watch an agent (renders via planar visualizer):

```bash
# Visualize the best model from run 001
python visualize.py 1

# Or visualize a specific checkpoint file
python visualize.py models/compact_curriculum_001/history/gen_00100_score_250.pth
```

### Monitoring

The logger tracks:

1. **Fitness:** Max, Average, and Standard Deviation
2. **Topology:** Growth of nodes and connections over time
3. **Curriculum:** Current difficulty level and physics parameters

Logs are stored in `models/<run_id>/logs.jsonl`.

### Configuration

Key parameters in `config.py` for tuning:

* `NUM_WORKERS`: Physical core count, or less (e.g., 6 or 8)
* `POPULATION_SIZE`: 96; should be increased if stagnation occurs early
* `DT`: 0.01s (Physics timestep)
* `MUT_*_RATE`: Controls how aggressively the network topology changes
