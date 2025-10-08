"""Defines the environment for simulating the race car system."""

import gymnasium as gym
import numpy as np
from leap_c.examples.race_car.config import CarParams, get_default_car_params
from leap_c.examples.race_car.scaling import get_transformation_matrices
from leap_c.examples.race_car.track import get_track
from leap_c.examples.race_car.model import export_acados_integrator


class RaceCarEnvDimensionless(gym.Env):
    """TODO: write me."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        car_params: CarParams | None = None,
        dimensionless: bool = True,
        mpc_car_params: CarParams | None = None,
    ):
        if car_params is None or mpc_car_params is None:
            mpc_car_params = car_params = get_default_car_params()
            # raise ValueError("Car parameters not provided in the env.")

        self.car_params = car_params
        self.dimensionless = dimensionless  # whether to use the dimensionless formulation
        self.nx = 4  # number of states
        self.nu = 2  # number of inputs

        if dimensionless:
            # use agent instead of env parameters
            self.Ms, self.Ma, self.Mt = get_transformation_matrices(
                mpc_car_params
            )  # s(physical) = Ms * s(dimensionless)

            # remove the control inputs from the state transformation
            self.Ma = self.Ms[self.nx:, self.nx:]  # only inputs
            self.Ms = self.Ms[:self.nx, :self.nx]  # only states            

            self.Ms_inv = np.linalg.inv(self.Ms)
            self.Ma_inv = np.linalg.inv(self.Ma)
            self.Mt_inv = np.linalg.inv(self.Mt)

        # integrator for simulating the dynamics (always in physical coordinates)
        self.integrator = export_acados_integrator(car_params=car_params, dimensionless=False)

        # state and action bounds
        large_number = 1e3  # states unbounded for now
        obs_ub = large_number * np.ones((self.nx,), dtype=np.float32)
        obs_lb = -obs_ub
        act_ub = np.array([car_params.D_max.item(), car_params.delta_max.item()], dtype=np.float32)
        act_lb = np.array([car_params.D_min.item(), car_params.delta_min.item()], dtype=np.float32)

        if dimensionless:
            # obs_ub = self.Ms_inv @ obs_ub
            # obs_lb = self.Ms_inv @ obs_lb
            act_ub = self.Ma_inv @ act_ub
            act_lb = self.Ma_inv @ act_lb

        self.action_space = gym.spaces.Box(np.float32(act_lb), np.float32(act_ub))
        self.observation_space = gym.spaces.Box(np.float32(obs_lb), np.float32(obs_ub))

        self.reset_needed = True
        self.s = None  # physical state
        
        s_ref = get_track(car_params)[0]  # [m] progress along the centerline
        self.s_max = s_ref[-1]  # [m] total length of the track

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.window = None
        self.clock = None

        # helper functions for scaling
        self.dim2nondim_s = lambda s: self.Ms_inv @ s if dimensionless else s
        self.nondim2dim_s = lambda s: self.Ms @ s if dimensionless else s
        self.dim2nondim_a = lambda a: self.Ma_inv @ a if dimensionless else a
        self.nondim2dim_a = lambda a: self.Ma @ a if dimensionless else a


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute the dynamics of the race car."""
        if self.reset_needed:
            raise RuntimeError("Call reset before using the step method.")

        # scale the MPC output back if dimensionless
        if self.dimensionless:
            action = self.nondim2dim_a(action)

        # simulate one time step
        s_prev = self.s
        self.s = self.integrator.simulate(x=self.s, u=action)
        self.s_trajectory.append(self.s)  # type: ignore

        # calculate the reward
        # progress along the track (relative to car length)
        r = (self.s[0] - s_prev[0]) / self.car_params.l.item()

        # check for termination
        term = False
        trunc = False
        info = {}
        if self.s[0] > self.s_max:
            term = True  # reached the end of the track
            r += 100.0  # bonus for reaching the goal
        elif not self.observation_space.contains(np.array(self.dim2nondim_s(self.s), dtype=np.float32)):
            trunc = True  # went out of bounds
            r -= 100.0  # penalty for leaving the track
        self.reset_needed = trunc or term

        # make the observation dimensionless if needed
        obs = self.dim2nondim_s(self.s)

        return obs, r, term, trunc, info


    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        self.s = self.init_state()
        self.reset_needed = False

        self.s_trajectory = []

        obs = self.dim2nondim_s(self.s)
        return obs, {}


    def init_state(self) -> np.ndarray:
        """The race car starts behind the start line (warm start)."""
        default_car_params = get_default_car_params()
        default_s0 = 0.0  # [m] start at the beginning of the track
        s0 = default_s0 * self.car_params.l.item() / default_car_params.l.item()
        return np.array([s0, 0.0, 0.0, 0.0])


    def render(self):
        pass


    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    from leap_c.examples.race_car.scaling import get_large_car_params
    import matplotlib.pyplot as plt
    dimensionless = True

    # create envs for the default and similar parameters
    params_ref = get_default_car_params()
    env_ref = RaceCarEnvDimensionless(
        car_params=params_ref,
        dimensionless=dimensionless,
        mpc_car_params=params_ref
    )

    params_sim = get_large_car_params()
    env_sim = RaceCarEnvDimensionless(
        car_params=params_sim,
        dimensionless=dimensionless,
        mpc_car_params=params_sim
    )

    seed = 0
    obs_ref = env_ref.reset(seed=seed)[0]
    obs_sim = env_sim.reset(seed=seed)[0]

    if dimensionless:
        assert env_ref.action_space == env_sim.action_space
        assert env_ref.observation_space == env_sim.observation_space
        assert np.allclose(obs_ref, obs_sim)

    # compare the two envs with random actions
    diffs = []
    obs_ref_log = []
    obs_sim_log = []
    obs_ref_log.append(obs_ref)
    obs_sim_log.append(obs_sim)
    act_log = []
    for _ in range(10):
        action = env_ref.action_space.sample()
        obs_ref, reward_ref, done_ref, truncated_ref, info_ref = env_ref.step(action)
        obs_sim, reward_sim, done_sim, truncated_sim, info_sim = env_sim.step(action)

        if done_ref or done_sim or truncated_ref or truncated_sim:
            seed += 1
            env_ref.reset(seed=seed)
            env_sim.reset(seed=seed)
            print("reset")

        obs_ref_log.append(obs_ref)
        obs_sim_log.append(obs_sim)
        diffs.append(np.max(np.abs(obs_ref - obs_sim)))
        act_log.append(action)
    print(f"max diff: {np.max(diffs)}")

    # plot the results
    obs_ref_log = np.array(obs_ref_log)
    obs_sim_log = np.array(obs_sim_log)
    act_log = np.array(act_log)
    nx = 4
    nu = 2
    fig, ax = plt.subplots(nx + nu, 1, sharex=True)
    labels = ["ref", "sim"]
    for i in range(nx):
        ax[i].plot(obs_ref_log[:, i], color="b", label=labels[0])
        ax[i].plot(obs_sim_log[:, i], color="r", linestyle="--", label=labels[1])
        ax[i].grid()
        ax[i].set_ylabel(env_ref.integrator.acados_sim.model.x_labels[i])
    ax[0].legend()
    for i in range(nu):
        ax[nx+i].step(
            list(range(obs_ref_log.shape[0])),
            np.append([act_log[0, i]], act_log[:, i]),
            where="post",
            color="b",
        )
        ax[nx+i].set_ylabel(env_ref.integrator.acados_sim.model.u_labels[i])
        ax[nx+i].grid()
    ax[-1].set_xlabel("$k$")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels(ax)
    plt.show(block=False)
    print("ok")
    print("Press ENTER to close the plot")
    input()
    plt.close()

    env_ref.close()
    env_sim.close()

