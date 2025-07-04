from typing import Any, Optional
import gymnasium as gym
import torch
from leap_c.registry import register_task
from leap_c.task import Task
from leap_c.ocp.acados.mpc import MpcInput, MpcParameter
from leap_c.ocp.acados.layer import MpcSolutionModule
from leap_c.examples.cartpole_dimensionless.env import CartpoleSwingupEnvDimensionless
from leap_c.examples.cartpole_dimensionless.mpc import CartpoleMpcDimensionless
from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params, test_similar
from leap_c.examples.cartpole_dimensionless.utils import convert_dataclass_to_dict, get_similar_cartpole_params


@register_task("cartpole_swingup_dimensionless")
class CartpoleSwingupDimensionless(Task):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def __init__(self):
        if test_similar:
            reference_params = get_default_cartpole_params()
            cart_mass = 0.5  # [kg]
            rod_length = 0.1  # [m]
            self.params_dataclass = get_similar_cartpole_params(
                reference_params=reference_params, cart_mass=cart_mass, rod_length=rod_length
            )
        else:
            self.params_dataclass = get_default_cartpole_params()

        params_dict = convert_dataclass_to_dict(self.params_dataclass)
        
        learnable_params = ["xref2"]
        N_horizon = 5  # Number of steps in the MPC horizon

        mpc = CartpoleMpcDimensionless(
            N_horizon=N_horizon,
            learnable_params=learnable_params,
            params=params_dict,  # type: ignore
        )
        mpc_layer = MpcSolutionModule(mpc)
        super().__init__(mpc_layer)

    def create_env(self, train: bool) -> gym.Env:
        return CartpoleSwingupEnvDimensionless(cartpole_params=self.params_dataclass)

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


# @register_task("cartpole_balance_dimensionless")
# class CartpoleBalanceDimensionless(CartpoleSwingupDimensionless):
#     """The same as CartpoleSwingup, but the starting position of the pendulum is upright, making the task a balancing task."""

#     def create_env(self, train: bool) -> gym.Env:
#         return CartpoleBalanceEnvDimensionless()
