from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from leap_c.examples.vehicle_steering.env import VehicleSteeringEnv
from leap_c.examples.vehicle_steering.mpc import VehicleSteeringMPC
from leap_c.mpc import MpcInput, MpcParameter
from leap_c.nn.modules import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("vehicle_steering")
class VehicleSteeringTask(Task):
    def __init__(self):
        mpc = VehicleSteeringMPC(
            learnable_params=[
                # "cf",
                # "cr",
                # "m",
                # "vx",
                # "lf",
                # "lr",
                # "iz",
                "q_diag",
                # "r_diag",
                # "q_diag_e",
                # "xref",
                # "uref",
                # "xref_e",
            ]
        )
        mpc_layer = MpcSolutionModule(mpc)

        super().__init__(mpc_layer)

        self.param_low = 0.5 * mpc.ocp.p_global_values
        self.param_high = 1.5 * mpc.ocp.p_global_values

        # TODO: Handle params that are nominally zero
        for i, p in enumerate(mpc.ocp.p_global_values):
            if p == 0:
                self.param_low[i] = -10.0
                self.param_high[i] = 10.0

    @property
    def param_space(self) -> spaces.Box:
        # low = np.array([0.5, 0.0])
        # high = np.array([2.5, 0.5])
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def create_env(self, train: bool) -> gym.Env:
        if train:
            init_state_dist = {
                "low": np.array([-0.5, 0.0, -0.1, 0.0]),
                "high": np.array([0.5, 0.0,  0.1, 0.0]),
            }
        else:
            init_state_dist = {
                "low": np.array([-0.1, 0.0, -0.1, 0.0]),
                "high": np.array([0.1, 0.0,  0.1, 0.0]),
            }

        return VehicleSteeringEnv(max_time=10.0, init_state_dist=init_state_dist)

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore

        return MpcInput(x0=obs, parameters=mpc_param)


@register_task("vehicle_steering_stabilization")
class VehicleSteeringTaskStabilization(VehicleSteeringTask):
    def create_env(self, train: bool) -> gym.Env:
        if train:
            init_state_dist = {
                "low": np.array([-0.5, 0.0, -0.1, 0.0]),
                "high": np.array([0.5, 0.0,  0.1, 0.0]),
            }
        else:
            init_state_dist = {
                "low": np.array([-0.1, 0.0, -0.1, 0.0]),
                "high": np.array([0.1, 0.0,  0.1, 0.0]),
            }

        return VehicleSteeringEnv(
            init_state_dist=init_state_dist,
            reference_type="straight",
            # observation_space=spaces.Box(
            #     low=np.array([0.0, -5.0, -50.0, -50.0]),
            #     high=np.array([8.0, +5.0, 50.0, 50.0]),
            #     dtype=np.float64,
            # ),
            # max_time=10.0,
        )
    

@register_task("vehicle_steering_double_lane_change")
class VehicleSteeringTaskDoubleLaneChange(VehicleSteeringTask):
    def create_env(self, train: bool) -> gym.Env:
        if train:
            init_state_dist = {
                "low": np.array([-0.5, 0.0, -0.1, 0.0]),
                "high": np.array([0.5, 0.0,  0.1, 0.0]),
            }
        else:
            init_state_dist = {
                "low": np.array([-0.1, 0.0, -0.1, 0.0]),
                "high": np.array([0.1, 0.0,  0.1, 0.0]),
            }

        return VehicleSteeringEnv(
            init_state_dist=init_state_dist,
            reference_type="double_lane_change",
            # observation_space=spaces.Box(
            #     low=np.array([0.0, -5.0, -50.0, -50.0]),
            #     high=np.array([8.0, +5.0, 50.0, 50.0]),
            #     dtype=np.float64,
            # ),
            # max_time=10.0,
        )
    
    
if __name__ == '__main__':
    # NOTE: must comment the "@register_task..." lines above before running this
    # TODO: inherit directly from Task
    VehicleSteeringTaskStabilization()
    VehicleSteeringTaskDoubleLaneChange()
    print('Task setup ok')

