"""
Neural Network Architecture & Evolutionary Operators
Implements a NEAT-based (NeuroEvolution of Augmenting Topologies) strategy
"""

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from config import Config

# =============================================================================
# GENOTYPE (Data Structures)
# =============================================================================

@dataclass
class NodeGene:
    """Represents a single neuron definition"""
    id: int
    type: str               # 'input', 'hidden', 'output'
    bias: float = 0.0

@dataclass
class ConnectionGene:
    """Represents a synapse between two neurons"""
    innovation: int         # Global historical marker
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True

class Genome:
    """
    Genetic encoding of the network topology
    Manages historical markers (innovation numbers) for crossover alignment
    """
    _global_innovation: int = 0

    def __init__(self, input_size: int, output_size: int, init_topology: bool = True):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = {}
        self.connections = {}
        
        if init_topology:
            self._init_minimal_topology()

    @classmethod
    def get_new_innovation(cls) -> int:
        cls._global_innovation += 1
        return cls._global_innovation

    @property
    def next_node_id(self) -> int:
        return max(self.nodes.keys()) + 1 if self.nodes else 0

    def copy(self) -> 'Genome':
        new_genome = Genome(self.input_size, self.output_size)
        new_genome.nodes = {k: copy.copy(v) for k, v in self.nodes.items()}
        new_genome.connections = {k: copy.copy(v) for k, v in self.connections.items()}
        return new_genome

    def _init_minimal_topology(self):
        # Create Inputs
        for i in range(self.input_size):
            self.nodes[i] = NodeGene(i, 'input')
            
        # Create Output
        out_id = self.input_size
        initial_bias = np.random.normal(0, 1.0) 
        self.nodes[out_id] = NodeGene(out_id, 'output', bias=initial_bias)
        
        # Dense Connect
        for i in range(self.input_size):
            innov = self.get_new_innovation()
            w = np.random.normal(0, 1.0)
            self.connections[innov] = ConnectionGene(innov, i, out_id, w)

# =============================================================================
# PHENOTYPE
# =============================================================================

class CartNN(nn.Module):
    """
    Compiles Genome into PyTorch computation graph
    Optimized for execution by flattening graph into index arrays
    """
    def __init__(self, genome: Genome):
        super().__init__()
        self.genome = genome
        self.state_buffer: Optional[torch.Tensor] = None
        
        # Metadata needed for fitness calculation
        self.fitness = 0.0
        self.fitness_adj = 0.0

        # Compiled execution plan
        self.node_eval_order: List[int] = []
        self.input_indices: List[int] = []
        self.output_index: int = 0
        
        # Adjacency maps for fast forward pass
        # Mapping: node_idx -> (list_of_src_indices, list_of_weight_indices)
        self.adj_src: Dict[int, List[int]] = {}
        self.adj_w_idx: Dict[int, List[int]] = {}

        self._compile()

    @property
    def param_count(self) -> int:
        return len(self.genome.connections) + len(self.genome.nodes)

    def reset_state(self):
        """Clears recurrent activation memory"""
        self.state_buffer = None

    def _compile(self):
        """Builds tensor-ready adjacency lists and topological order"""
        all_nodes = sorted(list(self.genome.nodes.keys()))
        self.num_nodes = len(all_nodes)
        node_to_idx = {node_id: i for i, node_id in enumerate(all_nodes)}

        # Identify IO Indices
        self.input_indices = [node_to_idx[i] for i in range(self.genome.input_size)]
        self.output_index = node_to_idx[self.genome.input_size]

        # Filter active connections
        active_conns = sorted(
            [c for c in self.genome.connections.values() if c.enabled], 
            key=lambda x: x.innovation
        )

        # Parameter Tensors
        weights = [c.weight for c in active_conns]
        biases = [self.genome.nodes[nid].bias for nid in all_nodes]
        
        self.w_tensor = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.b_tensor = nn.Parameter(torch.tensor(biases, dtype=torch.float32))

        # Build Adjacency
        self.adj_src = {i: [] for i in range(self.num_nodes)}
        self.adj_w_idx = {i: [] for i in range(self.num_nodes)}

        for w_idx, c in enumerate(active_conns):
            if c.in_node not in node_to_idx or c.out_node not in node_to_idx:
                continue
            
            dst = node_to_idx[c.out_node]
            src = node_to_idx[c.in_node]
            
            self.adj_src[dst].append(src)
            self.adj_w_idx[dst].append(w_idx)

        # Execution Order 
        self.node_eval_order = self._get_topological_sort(all_nodes, active_conns, node_to_idx)

    def _get_topological_sort(self, nodes: List[int], connections: List[ConnectionGene], map_idx: Dict) -> List[int]:
        graph = {n: [] for n in nodes}
        in_degree = {n: 0 for n in nodes}
        
        for c in connections:
            if c.in_node in graph and c.out_node in graph:
                graph[c.in_node].append(c.out_node)
                in_degree[c.out_node] += 1
        
        queue = [n for n in nodes if in_degree[n] == 0]
        sorted_indices = []
        
        while queue:
            node_id = queue.pop(0)
            sorted_indices.append(map_idx[node_id])
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Handle cycles: Append any remaining nodes
        if len(sorted_indices) < len(nodes):
            processed = set(sorted_indices)
            remaining = [map_idx[n] for n in nodes if map_idx[n] not in processed]
            sorted_indices.extend(remaining)
            
        return sorted_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0) if x.dim() > 1 else 1
        device = x.device
        
        # Manage State Buffer
        if self.state_buffer is None or self.state_buffer.size(0) != batch_size:
            self.state_buffer = torch.zeros(batch_size, self.num_nodes, device=device)
        else:
            # Detach to prevent graph growth
            self.state_buffer = self.state_buffer.detach()

        # Load Inputs
        if batch_size > 1:
            for i, idx in enumerate(self.input_indices):
                self.state_buffer[:, idx] = x[:, i]
        else:
            for i, idx in enumerate(self.input_indices):
                self.state_buffer[0, idx] = x[i]

        # Propagate
        for node_idx in self.node_eval_order:
            if node_idx in self.input_indices: continue
            
            # Start with bias
            node_sum = self.b_tensor[node_idx].expand(batch_size)
            
            # Weighted Sum: Iterate only existing connections
            src_indices = self.adj_src[node_idx]
            w_indices = self.adj_w_idx[node_idx]
            
            if src_indices:
                # Vectorized gather for inputs to this node
                inputs = self.state_buffer[:, src_indices]  # [B, n_inputs]
                weights = self.w_tensor[w_indices]          # [n_inputs]
                node_sum = node_sum + (inputs @ weights)
            
            # Activation (Tanh)
            self.state_buffer[:, node_idx] = torch.tanh(node_sum)

        out = self.state_buffer[:, self.output_index]
        
        # Shape adjustment
        if batch_size == 1:
            return torch.tanh(out)
        return torch.tanh(out).unsqueeze(-1)

# =============================================================================
# EVOLUTIONARY OPERATORS
# =============================================================================

def mutate_neat(parent: Genome, config: Config, sigma: float = 0.1) -> Genome:
    child = parent.copy()
    
    # --- Connection Weight Mutation ---
    for c in child.connections.values():
        if random.random() < config.MUT_WEIGHT_RATE:
            if random.random() < 0.1:
                # 10% Chance: Hard Reset
                c.weight = np.random.normal(0, 1.0)
            else:
                # 90% Chance: Perturb
                c.weight += np.random.normal(0, sigma)
            
            # Keep weights within responsive range of tanh
            c.weight = np.clip(c.weight, -8.0, 8.0)

    # --- Node Bias Mutation ---
    for n in child.nodes.values():
        if n.type == 'input': 
            continue
            
        if random.random() < config.MUT_BIAS_RATE:
            if random.random() < 0.1:
                n.bias = np.random.normal(0, 1.0)
            else:
                n.bias += np.random.normal(0, sigma)
            
            # Clip biases to prevent dead neurons
            n.bias = np.clip(n.bias, -4.0, 4.0)
    
    # --- Structural Mutation ---
    if random.random() < config.MUT_STRUCT_RATE: 
        _mutate_add_connection(child)
        
    if random.random() < (config.MUT_STRUCT_RATE * 0.5): 
        _mutate_add_node(child)
        
    return child

def _mutate_add_connection(genome: Genome):
    node_ids = list(genome.nodes.keys())
    if len(node_ids) < 2: return
    random.shuffle(node_ids)
    
    for i in node_ids:
        for j in node_ids:
            if i == j: continue
            if genome.nodes[j].type == 'input': continue

            # Check if connection exists
            if any(c.in_node == i and c.out_node == j for c in genome.connections.values()):
                continue

            innov = Genome.get_new_innovation()
            w = np.random.normal(0, 1.0)
            genome.connections[innov] = ConnectionGene(innov, i, j, w)
            return

def _mutate_add_node(genome: Genome):
    if not genome.connections: return
    active_conns = [c for c in genome.connections.values() if c.enabled]
    if not active_conns: return
        
    conn = random.choice(active_conns)
    conn.enabled = False
    
    new_id = genome.next_node_id
    # Explicit bias init for clarity
    genome.nodes[new_id] = NodeGene(new_id, 'hidden', bias=0.0)
    
    innov1 = Genome.get_new_innovation()
    genome.connections[innov1] = ConnectionGene(innov1, conn.in_node, new_id, 1.0)
    
    innov2 = Genome.get_new_innovation()
    genome.connections[innov2] = ConnectionGene(innov2, new_id, conn.out_node, conn.weight)

def crossover_neat(p1: Genome, p2: Genome, f1: float, f2: float) -> Genome:
    best, other = (p1, p2) if f1 > f2 else (p2, p1)
    child = Genome(best.input_size, best.output_size, init_topology=False)
    # Copy nodes from best
    child.nodes = {k: copy.copy(v) for k, v in best.nodes.items()}
    
    all_innovs = set(best.connections.keys()) | set(other.connections.keys())
    
    for innov in all_innovs:
        in_best = innov in best.connections
        in_other = innov in other.connections
        
        if in_best and in_other:
            src = best if random.random() < 0.5 else other
            child.connections[innov] = copy.copy(src.connections[innov])
        elif in_best:
            child.connections[innov] = copy.copy(best.connections[innov])
            
    return child

def compatibility_distance(g1: Genome, g2: Genome, c1=1.0, c2=1.0, c3=0.4) -> float:
    k1 = sorted(g1.connections.keys())
    k2 = sorted(g2.connections.keys())
    
    if not k1 and not k2: return 0.0
    
    matching = 0
    disjoint = 0
    weight_diff = 0.0
    
    i, j = 0, 0
    while i < len(k1) and j < len(k2):
        id1, id2 = k1[i], k2[j]
        if id1 == id2:
            matching += 1
            weight_diff += abs(g1.connections[id1].weight - g2.connections[id2].weight)
            i += 1; j += 1
        elif id1 < id2:
            disjoint += 1
            i += 1
        else:
            disjoint += 1
            j += 1
            
    excess = (len(k1) - i) + (len(k2) - j)
    N = max(len(k1), len(k2), 1.0)
    if N < 20: N = 1.0 
    
    avg_w = (weight_diff / matching) if matching > 0 else 0.0
    return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_w)

class Species:
    def __init__(self, representative: CartNN, species_id: int):
        self.id = species_id
        self.representative = representative 
        self.members: List[CartNN] = []
        self.avg_fitness_adj = 0.0
        self.best_fitness = -float('inf')
        self.stagnation = 0
        self.age = 0
        self.add_member(representative)

    def add_member(self, agent: CartNN):
        self.members.append(agent)

    def adjust_fitness(self):
        if not self.members: return
        n = len(self.members)
        for agent in self.members:
            agent.fitness_adj = agent.fitness / n

    def sort_members(self):
        self.members.sort(key=lambda x: getattr(x, 'fitness', -9999), reverse=True)

    def update_stagnation(self):
        if not self.members: return
        current_best = self.members[0].fitness
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.age += 1

    def reset(self):
        # Keep champion as representative
        if self.members:
            self.representative = self.members[0]
        self.members = []
        self.avg_fitness_adj = 0.0