# Experiment configs

Most configs use the original **physics curriculum**, which is now a baseline. The `*_revcurr`
configs use the start-state curriculum (S2, `curriculum.mode: start_state`, see
[docs/CURRICULUM.md](../docs/CURRICULUM.md)) with the adaptive back-off step.

Every file is deep-merged over `pendulum/config.py:DEFAULTS`; anything can be overridden
from the command line (`--set budget.generations=500`). Method hyper-parameters not set
here use the defaults in `pendulum/methods/*.py` (`*_DEFAULTS`).

| config | method | policy | curriculum | purpose |
|---|---|---|---|---|
| neat | custom NEAT | evolved topology | legacy | main NEAT run |
| neat_prune | custom NEAT + connection deletion | evolved | legacy | stronger pressure toward small topologies |
| neat_fitlegacy | custom NEAT | evolved | legacy | ablation: original fitness formula |
| neat_nocurr | custom NEAT | evolved | none | ablation: does the curriculum help? |
| neat_cma | NEAT 300 gens -> CMA-ES | evolved -> fixed | legacy | hybrid architecture search + weight tuning |
| cmaes_linear | CMA-ES | 9->1 (10 params) | legacy | smallest possible baseline |
| cmaes_mlp4 / mlp8 | CMA-ES | 9-4-1 (45) / 9-8-1 (89) | legacy | fixed-topology ES |
| cmaes_rnn4 | CMA-ES | Elman RNN, 4 hidden (61) | legacy | recurrent fixed topology |
| cmaes_mlp4_nocurr | CMA-ES | 9-4-1 | none | ablation: curriculum |
| openai_es_mlp8 | OpenAI-ES | 9-8-1 | legacy | gradient-estimate ES |
| map_elites_mlp4 | MAP-Elites | 9-4-1 | legacy | quality-diversity |
| cmaes_mlp4_revcurr / mlp8_revcurr | CMA-ES | 9-4-1 (45) / 9-8-1 (89) | start_state | reliability of S2 (4B steps, stop at 99 % validation) |

Shared settings (DEFAULTS): 5 random initial conditions per candidate per generation
(common random numbers), fitness `streak_smooth`, size penalty 0.005/param,
validation on the target physics every 25 generations (256 held-out starts), velocity
observation scale 12.
