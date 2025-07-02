from collections import OrderedDict
from typing import Any, Optional

import gymnasium as gym
import torch
from leap_c.examples.cartpole_dimensionless.env import (
    CartpoleBalanceEnvDimensionless,
    CartpoleSwingupEnvDimensionless,
)
from leap_c.examples.cartpole_dimensionless.mpc import CartpoleMpcDimensionless
from leap_c.ocp.acados.layer import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task

from leap_c.ocp.acados.mpc import MpcInput, MpcParameter
from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params

cartpole_params = get_default_cartpole_params()
PARAMS_SWINGUP = OrderedDict(
    [
        ("M", cartpole_params.M),     # mass of the cart [kg]
        ("m", cartpole_params.m),     # mass of the ball [kg]
        ("g", cartpole_params.g),     # gravity constant [m/s^2]
        ("l", cartpole_params.l),     # length of the rod [m]
        # The quadratic cost matrix is calculated according to L@L.T
        ("L11", cartpole_params.L11),
        ("L22", cartpole_params.L22),
        ("L33", cartpole_params.L33),
        ("L44", cartpole_params.L44),
        ("L55", cartpole_params.L55),
        ("Lloweroffdiag", cartpole_params.Lloweroffdiag),
        ("c1", cartpole_params.c1),    # position linear cost, only used for EXTERNAL cost
        ("c2", cartpole_params.c2),    # theta linear cost, only used for EXTERNAL cost
        ("c3", cartpole_params.c3),    # v linear cost, only used for EXTERNAL cost
        ("c4", cartpole_params.c4),    # thetadot linear cost, only used for EXTERNAL cost
        ("c5", cartpole_params.c5),    # u linear cost, only used for EXTERNAL cost
        ("xref1", cartpole_params.xref1), # reference position, only used for NONLINEAR_LS cost
        ("xref2", cartpole_params.xref2), # reference theta, only used for NONLINEAR_LS cost
        ("xref3", cartpole_params.xref3), # reference v, only used for NONLINEAR_LS cost
        ("xref4", cartpole_params.xref4), # reference thetadot, only used for NONLINEAR_LS cost
        ("uref", cartpole_params.uref),
        ("dt", cartpole_params.dt), # time step [s]
        ("Fmax", cartpole_params.Fmax),  # maximum force applied to the cart [N]
        ("gamma", cartpole_params.gamma),  # discount factor for the cost function
    ]
)


@register_task("cartpole_swingup_dimensionless")
class CartpoleSwingupDimensionless(Task):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def __init__(self):
        params = PARAMS_SWINGUP
        learnable_params = ["xref2"]

        mpc = CartpoleMpcDimensionless(
            N_horizon=5,
            learnable_params=learnable_params,
            params=params,  # type: ignore
        )
        mpc_layer = MpcSolutionModule(mpc)
        super().__init__(mpc_layer)

    def create_env(self, train: bool) -> gym.Env:
        return CartpoleSwingupEnvDimensionless()

    @property
    def param_space(self) -> gym.spaces.Box | None:
        return gym.spaces.Box(low=-2.0 * torch.pi, high=2.0 * torch.pi, shape=(1,))

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        if param_nn is None:
            raise ValueError("Parameter tensor is required for MPC task.")

        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore

        return MpcInput(x0=obs, parameters=mpc_param)


@register_task("cartpole_balance_dimensionless")
class CartpoleBalanceDimensionless(CartpoleSwingupDimensionless):
    """The same as CartpoleSwingup, but the starting position of the pendulum is upright, making the task a balancing task."""

    def create_env(self, train: bool) -> gym.Env:
        return CartpoleBalanceEnvDimensionless()


@register_task("cartpole_swingup_long_horizon_dimensionless")
class CartpoleSwingupLongDimensionless(CartpoleSwingupDimensionless):
    """Swing-up task for the pendulum on a cart system,
    like CartpoleSwingup, but with a much longer horizon.
    """

    def __init__(self):
        params = PARAMS_SWINGUP
        learnable_params = ["xref2"]

        mpc = CartpoleMpcDimensionless(
            N_horizon=20,
            T_horizon=1,
            learnable_params=learnable_params,
            params=params,  # type: ignore
        )
        mpc_layer = MpcSolutionModule(mpc)
        Task.__init__(self, mpc_layer)        
