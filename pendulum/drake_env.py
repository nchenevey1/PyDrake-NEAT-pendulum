"""
Reference PyDrake simulation (validation + visualization only; training uses `rollout.py`).

Same diagram as `legacy/simulation.py`, with one fix: the legacy env advanced with
`AdvanceTo(t + DT)`, accumulating floating-point time, so Drake occasionally ran
two discrete updates in one control step and none in the next (e.g. control
steps 5/6). Here the simulator advances to the *midpoint* of each control period,
which always contains exactly one periodic discrete update.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pydrake.all as drake
from pydrake.systems.framework import BasicVector, LeafSystem

from .task import DEFAULT_TASK, PhysicsParams, TaskConfig

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "double_pendulum_free.urdf"


class _Actuator(LeafSystem):
    """Cart force input plus linear drag on all joints."""

    def __init__(self, plant):
        super().__init__()
        self.drag = 0.0
        self.n_v = plant.num_velocities()
        self.u_port = self.DeclareVectorInputPort("u", BasicVector(1))
        self.state_port = self.DeclareVectorInputPort("state", BasicVector(plant.num_positions() + self.n_v))
        self.DeclareVectorOutputPort("force", BasicVector(self.n_v), self._calc)

    def _calc(self, context, output):
        v = self.state_port.Eval(context)[-self.n_v:]
        f = -self.drag * v
        f[0] += self.u_port.Eval(context)[0]
        output.SetFromVector(f)


class DrakeEnv:
    def __init__(self, physics: PhysicsParams, task: TaskConfig = DEFAULT_TASK, visualize: bool = False):
        self.task = task
        self.physics = physics
        builder = drake.DiagramBuilder()
        self.plant, scene_graph = drake.AddMultibodyPlantSceneGraph(builder, time_step=task.dt)
        drake.Parser(self.plant).AddModels(str(URDF_PATH))
        self.plant.mutable_gravity_field().set_gravity_vector([0, 0, -physics.g])
        self.plant.Finalize()
        self.act = builder.AddSystem(_Actuator(self.plant))
        self.act.drag = physics.drag
        builder.Connect(self.act.GetOutputPort("force"), self.plant.GetInputPort("applied_generalized_force"))
        builder.Connect(self.plant.get_state_output_port(), self.act.GetInputPort("state"))
        self.viz = None
        if visualize:
            self.viz = drake.ConnectPlanarSceneGraphVisualizer(builder, scene_graph, xlim=[-4, 4], ylim=[-2.5, 2.5])
        self.diagram = builder.Build()
        self.sim = drake.Simulator(self.diagram)
        self.ctx = self.sim.get_mutable_context()
        self.plant_ctx = self.diagram.GetMutableSubsystemContext(self.plant, self.ctx)
        self.act_ctx = self.diagram.GetMutableSubsystemContext(self.act, self.ctx)
        self.k = 0

    def reset(self, ic: Optional[np.ndarray] = None) -> np.ndarray:
        self.k = 0
        self.ctx.SetTime(0.0)
        self.plant.SetPositionsAndVelocities(self.plant_ctx, np.zeros(6) if ic is None else np.asarray(ic, float))
        self.act.u_port.FixValue(self.act_ctx, [0.0])
        self.sim.Initialize()
        return self.state()

    def state(self) -> np.ndarray:
        return self.plant.GetPositionsAndVelocities(self.plant_ctx).copy()

    def step(self, action: float) -> np.ndarray:
        self.act.u_port.FixValue(self.act_ctx, [float(action) * self.physics.max_force])
        self.k += 1
        self.sim.AdvanceTo((self.k - 0.5) * self.task.dt)
        return self.state()
