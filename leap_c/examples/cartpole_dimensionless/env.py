import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control import utils as gym_utils
from typing import Optional
from leap_c.examples.cartpole_dimensionless.config import CartPoleParams, get_default_cartpole_params, dimensionless
from leap_c.examples.cartpole_dimensionless.model import export_acados_integrator
from leap_c.examples.cartpole_dimensionless.utils import get_transformation_matrices
from scipy.integrate import solve_ivp


class CartpoleSwingupEnvDimensionless(gym.Env):
    """
    An environment of a pendulum on a cart meant for swinging
    the pole into an upright position and holding it there.

    Observation Space:
    ------------------

    The observation is a `ndarray` with shape `(4,)` and dtype `np.float32`
    representing the state of the system.

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Pole Angle (theta)    | -2pi                | 2pi               |
    | 2   | Cart Velocity         | -Inf                | Inf               |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    NOTE: Like in the original CartPole environment, the range above for the cart position denotes
    the possible range of the cart's center of mass in the observation space,
    but the episode terminates if it leaves the interval (-2.4, 2.4) already.
    NOTE: The pole angle is actually bounded between -2pi and 2pi by always adding/subtracting
    (in the negative / in the positive case) the highest multiple of 2pi
    until the pole angle is within the bounds again.
    NOTE: Contrary to the original CartPoleEnv, the state space here is arranged like
    [x, theta, dx, dtheta] instead of [x, dx, theta, dtheta].
    NOTE: A positive angle theta is interpreted as counterclockwise rotation.


    Action Space:
    -------------

    The action is a `ndarray` with shape `(1,)` which can take values in the range (-Fmax, Fmax) indicating the direction
    of the fixed force the cart is pushed with (action > 0 -> push right).


    Reward:
    -------
    Since this is an environment for the swingup task, the agent achieves maximum reward when the pole
    is upright (theta = 0) and minimum reward when the pole is hanging down (theta = pi or theta = -pi).
    More precisely, the reward in a step is bounded between 0 and 0.1, given by
    r = abs(pi - (abs(theta))) / (10 * pi)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        cartpole_params: CartPoleParams | None = None,
        use_acados_integrator: bool = True,
    ):
        if cartpole_params is None:
            input("Warning: No parameters provided in the env, using default parameters. Press Enter to continue...")
            cartpole_params = get_default_cartpole_params()

        if dimensionless:
            self.Mx, self.Mu, self.Mt = get_transformation_matrices(cartpole_params)  # x(physical) = Mx * x(dimensionless)
            self.Mx_inv = np.linalg.inv(self.Mx)
            self.Mu_inv = np.linalg.inv(self.Mu)
            self.Mt_inv = np.linalg.inv(self.Mt)

        self.length = cartpole_params.l.item()
        self.Fmax = cartpole_params.Fmax.item()
        self.dt = cartpole_params.dt.item()
        self.max_time = 10.0
        self.x_threshold = 3 * self.length  # this should be physical

        self.use_acados_integrator = use_acados_integrator
        if use_acados_integrator:
            self.integrator = export_acados_integrator(cartpole_params=cartpole_params)
        else:
            def f_explicit(
                x,
                u,
                g=cartpole_params.g.item(),
                M=cartpole_params.M.item(),
                m=cartpole_params.m.item(),
                l=self.length,  # noqa E741
            ):
                _, theta, dx, dtheta = x
                F = u.item()
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                denominator = M + m - m * cos_theta * cos_theta
                return np.array(
                    [
                        dx,
                        dtheta,
                        (
                            -m * l * sin_theta * dtheta * dtheta
                            + m * g * cos_theta * sin_theta
                            + F
                        )
                        / denominator,
                        (
                            -m * l * cos_theta * sin_theta * dtheta * dtheta
                            + F * cos_theta
                            + (M + m) * g * sin_theta
                        )
                        / (l * denominator),
                    ]
                )
            
            def scipy_step(f, x, u, h):
                t_span = (0, h)
                fun = lambda t, y: np.hstack(( f(y[:4], y[4], h), 0.0))
                sol = solve_ivp(fun, t_span, np.hstack((x,u)), method="RK45")
                return sol.y[:4,-1]
            self.integrator = lambda x, u: scipy_step(f_explicit, x, u, self.dt)

        # state and action bounds
        obs_ub = np.array(
            [
                self.x_threshold * 2,
                2 * np.pi,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        act_ub = self.Fmax

        if dimensionless:
            obs_ub = self.Mx_inv @ obs_ub
            obs_ub[2:] = np.finfo(np.float32).max
            act_ub = (self.Mu_inv * self.Fmax).item()

        self.action_space = spaces.Box(-np.float32(act_ub), np.float32(act_ub))
        self.observation_space = spaces.Box(-np.float32(obs_ub), np.float32(obs_ub))

        self.reset_needed = True
        self.t = 0
        self.x = None

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.pos_trajectory = None
        self.pole_end_trajectory = None
        self.screen_width = 600
        self.screen_height = 400
        self.window = None
        self.clock = None

        # helper functions for scaling
        self.dim2nondim_x = lambda x: self.Mx_inv @ x if dimensionless else x
        self.nondim2dim_x = lambda x: self.Mx @ x if dimensionless else x
        self.dim2nondim_u = lambda u: self.Mu_inv @ u if dimensionless else u
        self.nondim2dim_u = lambda u: self.Mu @ u if dimensionless else u


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute the dynamics of the pendulum on cart."""
        if self.reset_needed:
            raise RuntimeError("Call reset before using the step method.")
        
        # scale the MPC output back if dimensionless
        if dimensionless:
            action = self.nondim2dim_u(action)

        if self.use_acados_integrator:
            self.x = self.integrator.simulate(x=self.x, u=action)
        else:
            self.x = self.integrator(x=self.x, u=action)
        self.x_trajectory.append(self.x)  # type: ignore
        self.t += self.dt
        theta = self.x[1]
        if theta > 2 * np.pi:
            theta = theta % 2 * np.pi
        elif theta < -2 * np.pi:
            theta = -(-theta % 2 * np.pi)  # "Symmetric" modulo
        self.x[1] = theta

        r = abs(np.pi - (abs(theta))) / (10 * np.pi)  # Reward for swingup; Max: 0.1

        term = False
        trunc = False
        info = {}
        if self.x[0] > self.x_threshold or self.x[0] < -self.x_threshold:
            term = True  # Just terminating should be enough punishment when reward is positive
            info = {"task": {"violation": True, "success": False}}
        if self.t > self.max_time:
            # check if the pole is upright in the last 10 steps
            if len(self.x_trajectory) >= 10:
                success = all(np.abs(self.x_trajectory[i][1]) < 0.1 for i in range(-10, 0))  # TODO: check if 0.1 is a good limit
            else:
                success = False  # Not enough data to determine success

            info = {"task": {"violation": False, "success": success}}
            trunc = True
        self.reset_needed = trunc or term

        # make the observation (x,theta,dx,dtheta) dimensionless
        obs = self.dim2nondim_x(self.x) if dimensionless else self.x

        return obs, r, term, trunc, info


    def reset(
            self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        self.t = 0
        self.x = self.init_state()
        self.reset_needed = False

        self.x_trajectory = []
        self.pos_trajectory = None
        self.pole_end_trajectory = None

        obs = self.dim2nondim_x(self.x) if dimensionless else self.x
        return obs, {}


    def init_state(self) -> np.ndarray:
        """The pendulum is hanging down at the start."""
        return np.array([0.0, np.pi, 0.0, 0.0])

    def include_this_state_trajectory_to_rendering(self, state_trajectory: np.ndarray):
        """Meant for setting a state trajectory for rendering.
        If a state trajectory is not set before the next call of render,
        the rendering will not render a state trajectory.

        NOTE: The record_video wrapper of gymnasium will call render() AFTER every step.
        This means if you use the wrapper,
        make a step,
        calculate action and state trajectory from the observations,
        and input the state trajectory with this function before taking the next step,
        the picture being rendered after this next step will be showing the trajectory planned BEFORE DOING the step.
        """
        self.pos_trajectory = []
        self.pole_end_trajectory = []
        for x in state_trajectory:
            self.pos_trajectory.append(x[0])
            self.pole_end_trajectory.append(self.calc_pole_end(x[0], x[1], self.length))

    def calc_pole_end(
        self, x_coord: float, theta: float, length: float
    ) -> tuple[float, float]:
        # NOTE: The minus is necessary because theta is seen as counterclockwise
        pole_x = x_coord - length * np.sin(theta)
        pole_y = length * np.cos(theta)
        return pole_x, pole_y

    def render(self):
        import pygame
        from pygame import gfxdraw

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        
        # initialize the font for displaying text
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if not hasattr(self, "font"):
            self.font = pygame.font.SysFont(None, 24)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.x is None:
            return None

        world_width = 2 * self.x_threshold
        center = (int(self.screen_width / 2), int(self.screen_height / 2))
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * self.length
        cartwidth = 50.0
        cartheight = 30.0
        axleoffset = cartheight / 4.0
        ground_height = 180

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((255, 255, 255))

        # ground
        gfxdraw.hline(canvas, 0, self.screen_width, ground_height, (0, 0, 0))

        # cart
        left, right, top, bot = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )

        pos = self.x[0]  # type:ignore
        theta = self.x[1]  # type:ignore
        cartx = pos * scale + center[0]
        cart_coords = [(left, bot), (left, top), (right, top), (right, bot)]
        cart_coords = [(c[0] + cartx, c[1] + ground_height) for c in cart_coords]
        gfxdraw.aapolygon(canvas, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(canvas, cart_coords, (0, 0, 0))

        # pole
        left, right, top, bot = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(left, bot), (left, top), (right, top), (right, bot)]:
            coord = pygame.math.Vector2(coord).rotate_rad(theta)
            coord = (coord[0] + cartx, coord[1] + ground_height + axleoffset)
            pole_coords.append(coord)
        pole_color = (202, 152, 101)
        gfxdraw.aapolygon(canvas, pole_coords, pole_color)
        gfxdraw.filled_polygon(canvas, pole_coords, pole_color)

        # Axle of pole
        gfxdraw.aacircle(
            canvas,
            int(cartx),
            int(ground_height + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            canvas,
            int(cartx),
            int(ground_height + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        # Draw the planned trajectory if it exists
        if self.pos_trajectory is not None:
            if self.pole_end_trajectory is None:
                raise AttributeError(
                    "Why is pole_end_trajectory None, but pos_trajectory isn't?"
                )
            planxs = [int(x * scale + center[0]) for x in self.pos_trajectory]
            plan_pole_end = [
                (
                    int(x * scale + center[0]),
                    int(ground_height + axleoffset + y * scale - polewidth / 2),
                )
                for x, y in self.pole_end_trajectory
            ]

            # Draw the positions offset in the y direction for better visibility
            for i, planx in enumerate(planxs):
                if abs(planx) > self.screen_width:
                    # Dont render out of bounds
                    continue
                gfxdraw.pixel(canvas, int(planx), int(ground_height + i), (255, 5, 5))
            for i, plan_pole_end in enumerate(plan_pole_end):
                if abs(plan_pole_end[0]) > self.screen_width:
                    # Dont render out of bounds
                    continue
                gfxdraw.pixel(
                    canvas, int(plan_pole_end[0]), int(plan_pole_end[1]), (5, 255, 5)
                )

        canvas = pygame.transform.flip(canvas, False, True)

        # add the length of the pole to the canvas
        length_text = f"L = {self.length:.2f} m"
        text_surface = self.font.render(length_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(self.screen_width // 2, 20))
        canvas.blit(text_surface, text_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))  # type:ignore
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])  # type:ignore

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


class CartpoleBalanceEnvDimensionless(CartpoleSwingupEnvDimensionless):
    def init_state(self, options: Optional[dict] = None) -> np.ndarray:
        low, high = gym_utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        return self.np_random.uniform(low=low, high=high, size=(4,))
    

if __name__ == "__main__":
    from leap_c.examples.cartpole_dimensionless.utils import get_similar_cartpole_params
    # from leap_c.examples.cartpole_dimensionless.utils import acados_cleanup
    # acados_cleanup()

    # create envs for the default and similar parameters
    params_ref = get_default_cartpole_params()
    env_ref = CartpoleSwingupEnvDimensionless(cartpole_params=params_ref)

    cart_mass = 5.0  # [kg] 0.5
    rod_length = 5.0  # [m] 0.1
    params_sim = get_similar_cartpole_params(reference_params=params_ref, cart_mass=cart_mass, rod_length=rod_length)
    # env_sim = CartpoleSwingupEnvDimensionless(cartpole_params=params_sim)
    env_sim = CartpoleSwingupEnvDimensionless(cartpole_params=params_ref, use_acados_integrator=True)
    
    assert env_ref.action_space == env_sim.action_space
    assert env_ref.observation_space == env_sim.observation_space

    obs_ref = env_ref.reset(seed=0)[0]
    obs_sim = env_sim.reset(seed=0)[0]
    assert np.allclose(obs_ref, obs_sim)

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

        obs_ref_log.append(obs_ref)
        obs_sim_log.append(obs_sim)
        diffs.append(np.max(np.abs(obs_ref-obs_sim)))
        act_log.append(action)
        # assert np.allclose(obs_ref, obs_sim, atol=1e-04)
    print(f'max diff: {np.max(diffs)}')
    
    import matplotlib.pyplot as plt
    obs_ref_log = np.array(obs_ref_log)
    obs_sim_log = np.array(obs_sim_log)
    act_log = np.array(act_log)
    nx = 4
    fig, ax = plt.subplots(nx+1, 1, sharex=True)
    for i in range(nx):
        ax[i].plot(obs_ref_log[:, i], color="b", label='scipy')
        ax[i].plot(obs_sim_log[:, i], color="r", linestyle="--", label='acados')
        ax[i].grid()
        ax[i].set_ylabel(f'$x_{i}$')
    ax[0].legend()
    ax[-1].step(list(range(obs_ref_log.shape[0])), np.append([act_log[0]], act_log), where="post", color="b")
    ax[-1].set_ylabel('$u$')
    ax[-1].grid()    

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels(ax)
    plt.show(block=False)
    print("Press ENTER to close the plot")
    input()
    plt.close()

    env_ref.close()
    env_sim.close()

    print("ok")
