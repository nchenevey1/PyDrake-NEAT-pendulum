import numpy as np

class Config:
    """
    Configuration for Compact Curriculum Learning (Double Pendulum Swing-up).
    Controls physics simulation, NEAT evolution, and curriculum progression.
    """

    # -------------------------------------------------------------------------
    # System & Experiment
    # -------------------------------------------------------------------------
    RUN_NAME        = "compact_curriculum_02"
    SEED            = 42                    # Global seed
    NUM_WORKERS     = 6                     # Parallel rollout workers

    # -------------------------------------------------------------------------
    # Physics Simulation (PyDrake)
    # -------------------------------------------------------------------------
    DT              = 0.01                  # Integration timestep [s]
    DURATION        = 20.0                  # Max episode duration [s]
    LIMIT_X         = 2.8                   # Track half-length [m]
    MAX_V_CART      = 50.0                  # m/s
    MAX_W_POLE      = 90.0                  # rad/s

    # -------------------------------------------------------------------------
    # Curriculum Physics Ranges
    # -------------------------------------------------------------------------
    # Interpolated based on difficulty index (0.0 -> 1.0)
    G_MIN, G_MAX        = 1.0, 9.81         # Gravity [m/s^2]
    DRAG_MAX, DRAG_MIN  = 3.0, 0.0          # Linear damping coeff
    FORCE_MAX_START     = 1000.0            # Actuation limits [N]
    FORCE_MAX_END       = 100.0

    # -------------------------------------------------------------------------
    # Curriculum Progression Logic
    # -------------------------------------------------------------------------
    DIFFICULTY_STEP = 0.005                 # Difficulty step size on pass
    PASS_TIME_RATIO = 0.20                  # Required survival duration (%)
    PASS_COUNT      = 1

    # -------------------------------------------------------------------------
    # Task Success Criteria
    # -------------------------------------------------------------------------
    SWING_UP_TIME     = 2.5                 # Max time to reach upright state [s]
    UPRIGHT_TOLERANCE = 0.2                 # Success threshold [rad]

    # -------------------------------------------------------------------------
    # Evolutionary Strategy (NEAT)
    # -------------------------------------------------------------------------
    NUM_GENERATIONS = 50000                 # Max training epochs
    POPULATION_SIZE = 1200                  # Agent count per generation
    SPECIES_TARGET  = 100                   # Target quantity of species
    SPECIES_THRESH  = 0.05                  # Species differentiation threshold
    NUM_TRIALS      = 1                     # Robustness checks per agent

    # Mutation Hyperparameters
    MUT_WEIGHT_RATE = 0.60                  # Prob. of weight perturbation
    MUT_BIAS_RATE   = 0.20                  # Prob. of bias perturbation
    MUT_STRUCT_RATE = 0.25                  # Prob. of topology change (node/conn)
    MUT_SIGMA       = 0.5                   # Gaussian std dev for weight mutation
    MUT_SIGMA_STAGNATION  = 10              # Number of stagnant generations before "shock"
    MUT_DEPTH_WIDTH_RATIO = 0.5             # Heuristic for network growth shape

    # Regularization
    BASE_COMPLEXITY_PENALTY = 0.005         # Fitness penalty per parameter

    # -------------------------------------------------------------------------
    # Neural Architecture
    # -------------------------------------------------------------------------
    # Inputs: [x, x_dot, sin1, cos1, th1_dot, sin2, cos2, th2_dot, alignment]
    INPUT_DIM = 9

    # -------------------------------------------------------------------------
    # Reward Function Coefficients
    # -------------------------------------------------------------------------
    # Survival & Dynamics
    W_ALIVE         = 0.1                   # Survival bonus per step
    W_ANGLE         = 0.5                   # Upright orientation incentive
    W_VELOCITY      = 0.1                   # Velocity alignment incentive
    W_CENTER        = 0.5                   # Centering bonus (track middle)
    W_STABLE        = 0.5                   # Stability bonus (low angular velocity)

    # Sparse / Shaping
    W_SHAPING       = 0.05                  # Dense shaping reward
    W_UPRIGHT       = 100.0                 # Sparse success reward