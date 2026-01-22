import threading
import numpy as np
import pydrake.all as drake
from pydrake.systems.framework import LeafSystem, BasicVector, Context
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from config import Config

def angle_normalize(theta: float) -> float:
    """Maps angle to range [-pi, pi]"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

# =============================================================================
# PHYSICS SYSTEM
# =============================================================================

class AerodynamicActuator(LeafSystem):
    """
    Drake LeafSystem
    Applies actuation force to the cart and aerodynamic drag to all joints
    Dynamics: F_net = F_actuation - (C_drag * v)
    """

    def __init__(self, plant: drake.MultibodyPlant):
        super().__init__()
        self.plant = plant
        self._drag_coeff = 0.0

        # Input Port: Control signal (1D)
        self.u_port = self.DeclareVectorInputPort(
            "u", BasicVector(1)
        )
        
        # Input Port: Full system state [q, v]
        self.state_port = self.DeclareVectorInputPort(
            "state", BasicVector(plant.num_positions() + plant.num_velocities())
        )
        
        # Output Port: Generalized forces
        self.DeclareVectorOutputPort(
            "force", 
            BasicVector(plant.num_velocities()), 
            self._calc_force_output
        )

    def set_drag(self, drag: float):
        """Update internal drag coefficient"""
        self._drag_coeff = drag

    def _calc_force_output(self, context: Context, output: BasicVector):
        # Read inputs
        u = self.u_port.Eval(context)[0]
        state = self.state_port.Eval(context)
        
        # Extract velocities (v is last N elements of state)
        # indices: 0=x_dot, 1=th1_dot, 2=th2_dot
        n_v = self.plant.num_velocities()
        v = state[-n_v:]

        # Apply actuation (Cart only, index 0)
        forces = np.zeros(n_v)
        forces[0] = u
        
        # Apply linear drag
        drag_force = -self._drag_coeff * v
        
        output.SetFromVector(forces + drag_force)

# =============================================================================
# RL ENVIRONMENT
# =============================================================================

class DoublePendulumEnv:
    """
    Reinforcement Learning wrapper around PyDrake Double Pendulum simulation
    Handles curriculum updates, observation normalization, and termination logic
    """
    
    URDF_PATH = Path(__file__).parent / "double_pendulum_free.urdf"

    # Normalization Constants (Estimated Upper Bounds)
    MAX_V_CART = Config.MAX_V_CART
    MAX_W_POLE = Config.MAX_W_POLE

    def __init__(self, visualizer: bool = False, g: float = 9.81):
        self.visualizer = visualizer
        
        # Drake objects
        self.diagram: Optional[drake.Diagram] = None
        self.simulator: Optional[drake.Simulator] = None
        self.context: Optional[drake.Context] = None
        
        # Sub contexts
        self._plant: Optional[drake.MultibodyPlant] = None
        self._actuator: Optional[AerodynamicActuator] = None
        self._plant_context = None
        self._actuator_context = None
        
        # Episode State
        self.current_g = -1.0
        self.current_max_force = 0.0
        self.prev_action = 0.0
        
        # Initialize
        self._obs_buffer = np.zeros(9, dtype=np.float32)
        self._build_simulation(g)

    def _build_simulation(self, g: float):
        """Constructs the Drake Diagram. Called when gravity changes"""
        builder = drake.DiagramBuilder()
        
        # Add Plant
        self._plant, self._scene_graph = drake.AddMultibodyPlantSceneGraph(
            builder, time_step=Config.DT
        )
        
        if not self.URDF_PATH.exists():
            raise FileNotFoundError(f"URDF not found: {self.URDF_PATH}")
            
        drake.Parser(self._plant).AddModels(str(self.URDF_PATH))
        self._plant.mutable_gravity_field().set_gravity_vector([0, 0, -g])
        self._plant.Finalize()
        
        # Add Actuator
        self._actuator = builder.AddSystem(AerodynamicActuator(self._plant))
        
        # Connect Systems
        # Actuator Force -> Plant Input
        builder.Connect(
            self._actuator.GetOutputPort("force"), 
            self._plant.GetInputPort("applied_generalized_force")
        )
        # Plant State -> Actuator Input
        builder.Connect(
            self._plant.get_state_output_port(),
            self._actuator.GetInputPort("state")
        )
        
        # Visualization
        if self.visualizer:
            drake.ConnectPlanarSceneGraphVisualizer(
                builder, self._scene_graph, xlim=[-5, 5], ylim=[-5, 5]
            )
            
        # Build & Context
        self.diagram = builder.Build()
        self.simulator = drake.Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()
        
        # Cache sub contexts for fast access during step
        self._plant_context = self.diagram.GetMutableSubsystemContext(
            self._plant, self.context
        )
        self._actuator_context = self.diagram.GetMutableSubsystemContext(
            self._actuator, self.context
        )
        
        self.current_g = g

    def reset(self, g: float, d: float, max_force: float, seed: Optional[int] = None):
        """
        Resets environment for new episode
        Rebuilds physics engine only if gravity 'g' changes
        """
        # Rebuild if gravity changed
        if not np.isclose(self.current_g, g):
            self._build_simulation(g)

        # Update dynamic parameters
        self._actuator.set_drag(d)
        self.current_max_force = max_force
        
        # Reset State
        self.prev_action = 0.0
        self.context.SetTime(0.0)

        # Randomize Initialization (Perturb at start)
        # rng = np.random.default_rng(seed)
        # init_state = rng.uniform(-0.05, 0.05, size=6) # [x, th1, th2, v_x, w1, w2]
        
        init_state = [0, 0, 0, 0, 0, 0]

        self._plant.SetPositionsAndVelocities(self._plant_context, init_state)
        self.simulator.Initialize()

    def step(self, action: float) -> Tuple[np.ndarray, bool, float, bool, float, float]:
        """
        Advances simulation by one timestep (Config.DT)
        
        Returns:
            obs (np.ndarray): Normalized observation vector
            done (bool): Termination flag
            x (float): Cart position
            is_upright (bool): Success flag
            th1_err (float): Pole 1 angle error
            th2_err (float): Pole 2 angle error
        """
        # Apply Control
        self.prev_action = action
        force_cmd = action * self.current_max_force
        self._actuator.u_port.FixValue(self._actuator_context, [force_cmd])

        # Physics Step
        self.simulator.AdvanceTo(self.context.get_time() + Config.DT)

        # Extract Raw State
        # q = [x, th1, th2], v = [x_dot, th1_dot, th2_dot]
        state_vec = self._plant.GetPositionsAndVelocities(self._plant_context)
        x, th1_raw, th2_raw = state_vec[:3]
        x_dot, th1_dot, th2_dot = state_vec[3:]

        # Normalize Angles
        # 0.0 = Upright, +/- pi = Down
        th1_err = angle_normalize(th1_raw - np.pi) 
        th2_err = angle_normalize(th2_raw)         

        # Check Objectives
        is_upright = (
            abs(th1_err) < Config.UPRIGHT_TOLERANCE and 
            abs(th2_err) < Config.UPRIGHT_TOLERANCE
        )
        
        done = self._check_termination(x, x_dot, is_upright)

        # Construct Observation
        obs = self._make_observation(
            x, x_dot, th1_raw, th2_raw, th1_dot, th2_dot
        )

        return obs, done, x, is_upright, th1_err, th2_err

    def _make_observation(self, x, x_dot, th1, th2, w1, w2) -> np.ndarray:
        
        """Constructs and normalizes the 9D observation vector"""
        sin1, cos1 = np.sin(th1), np.cos(th1)
        sin2, cos2 = np.sin(th2), np.cos(th2)
        
        # Alignment metric: cos(theta1 - theta2)
        # 1.0 = poles parallel, -1.0 = poles opposite
        alignment = (sin1 * sin2) + (cos1 * cos2)
        
        self._obs_buffer[0] = x / Config.LIMIT_X                        # Cart Pos [-1, 1]
        self._obs_buffer[1] = np.clip(x_dot / self.MAX_V_CART, -1, 1)   # Cart Vel [-1, 1]
        self._obs_buffer[2] = sin1                                      # Pole 1 Angle
        self._obs_buffer[3] = cos1                                      # Pole 1 Angle
        self._obs_buffer[4] = np.clip(w1 / self.MAX_W_POLE, -1, 1)      # Pole 1 Vel [-1, 1]
        self._obs_buffer[5] = sin2                                      # Pole 2 Angle
        self._obs_buffer[6] = cos2                                      # Pole 2 Angle
        self._obs_buffer[7] = np.clip(w2 / self.MAX_W_POLE, -1, 1)      # Pole 2 Vel [-1, 1]
        self._obs_buffer[8] = alignment                                 # Alignment [-1, 1]
        return self._obs_buffer

    def _check_termination(self, x: float, x_dot: float, is_upright: bool) -> bool:
        """Determines if episode should end early"""
        current_time = self.context.get_time()

        # Hard Time Limit
        if current_time > Config.DURATION:
            return True

        # Track Limits
        if abs(x) > Config.LIMIT_X:
            return True

        # Failure to Swing Up
        if (current_time > Config.SWING_UP_TIME) and not is_upright:
            return True

        return False