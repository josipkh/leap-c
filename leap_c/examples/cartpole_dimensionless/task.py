from typing import Any, Optional
import gymnasium as gym
import torch
from leap_c.registry import register_task
from leap_c.task import Task
from leap_c.ocp.acados.mpc import MpcInput, MpcParameter
from leap_c.ocp.acados.layer import MpcSolutionModule
from leap_c.examples.cartpole_dimensionless.env import CartpoleSwingupEnvDimensionless
from leap_c.examples.cartpole_dimensionless.mpc import CartpoleMpcDimensionless
from leap_c.examples.cartpole_dimensionless.config import CartPoleParams


@register_task("cartpole_swingup_dimensionless")
class CartpoleSwingupDimensionless(Task):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def __init__(self, mpc_params: CartPoleParams, env_params: CartPoleParams, dimensionless: bool):
        self.env_params = env_params
        self.mpc_params = mpc_params
        self.dimensionless = dimensionless
        learnable_params = ["xref2"]
        N_horizon = 5  # Number of steps in the MPC horizon

        mpc = CartpoleMpcDimensionless(
            N_horizon=N_horizon,
            learnable_params=learnable_params,
            cartpole_params=mpc_params,  # type: ignore
            dimensionless=dimensionless,
        )
        mpc_layer = MpcSolutionModule(mpc)
        super().__init__(mpc_layer)

    def create_env(self, train: bool) -> gym.Env:
        return CartpoleSwingupEnvDimensionless(
            cartpole_params=self.env_params,
            mpc_cartpole_params=self.mpc_params,
            dimensionless=self.dimensionless
        )

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


if __name__ == "__main__":
    # comment out the @register_task decorator to run this file directly
    from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params
    params = get_default_cartpole_params()
    dimensionless = True
    task = CartpoleSwingupDimensionless(mpc_params=params, env_params=params, dimensionless=dimensionless)
    print("ok")

