"""
This module contains the environment for the vehicle steering problem.
The system model is based on section 2.5 (eq. 2.45) in "Rajamani - Vehicle Dynamics and Control".
States are the lateral and heading errors and their derivatives (x = [ey ey_dot epsi epsi_dot]).
The action is the steering angle (u = delta).
The reference yaw rate is modelled as an additional control input (will be fixed at runtime).
"""

import numpy as np
import gymnasium as gym
import casadi as ca
from scipy.signal import cont2discrete
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
from leap_c.examples.vehicle_steering.utils import contains_symbolics, cont2discrete_symbolic, lqr
from scipy.integrate import solve_ivp
from leap_c.examples.vehicle_steering.utils import get_double_lane_change_data


@dataclass(kw_only=True)
class VehicleParams:
    cf: float | ca.SX = 72705.0     # [N/rad]   front cornering stiffness (for one tire)
    cr: float | ca.SX = 72705.0     # [N/rad]   rear cornering stiffness (for one tire)
    m:  float | ca.SX = 1600.0      # [kg]      vehicle mass
    vx: float | ca.SX = 60 / 3.6    # [m/s]     longitudinal vehicle speed
    lf: float | ca.SX = 1.311       # [m]       distance from the center of gravity to the front axle
    lr: float | ca.SX = 1.311       # [m]       distance from the center of gravity to the rear axle
    iz: float | ca.SX = 2394.0      # [kg*m^2]  vehicle moment of inertia
    isw:float | ca.SX = 13          # [-]       steering ratio


def get_A_cont(
    vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()
) -> np.ndarray | ca.SX:
    cf = vehicle_params.cf
    cr = vehicle_params.cr
    m = vehicle_params.m
    vx = vehicle_params.vx
    lf = vehicle_params.lf
    lr = vehicle_params.lr
    iz = vehicle_params.iz

    row_1 = (0, 1, 0, 0)
    row_2 = (0,-(2*cf+2*cr)/(m*vx), (2*cf+2*cr)/m, -(2*cf*lf-2*cr*lr)/(m*vx))
    row_3 = (0, 0, 0, 1)
    row_4 = (0, -(2*cf*lf-2*cr*lr)/(iz*vx), (2*cf*lf-2*cr*lr)/iz, -(2*cf*lf**2+2*cr*lr**2)/(iz*vx))
    
    if contains_symbolics(vehicle_params):
        return ca.vertcat(
            ca.horzcat(*row_1),
            ca.horzcat(*row_2),
            ca.horzcat(*row_3),
            ca.horzcat(*row_4)
        )
    else:
        return np.array([row_1, row_2, row_3, row_4])
    

def get_B_steer_cont(
    vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()
) -> np.ndarray | ca.SX:
    cf = vehicle_params.cf
    m = vehicle_params.m
    lf = vehicle_params.lf
    iz = vehicle_params.iz

    if any(isinstance(i, ca.SX) for i in [cf, m, lf, iz]):
        return ca.vertcat(
            0,
            2*cf/m,
            0,
            2*cf*lf/iz
        )
    else:
        return np.array([[0, 2*cf/m, 0, 2*cf*lf/iz]]).T
    
    
def get_B_ref_cont(
    vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()
) -> np.ndarray | ca.SX:
    cf = vehicle_params.cf
    cr = vehicle_params.cr
    m = vehicle_params.m
    vx = vehicle_params.vx
    lf = vehicle_params.lf
    lr = vehicle_params.lr
    iz = vehicle_params.iz

    row_2 = -(2*cf*lf-2*cr*lr)/(m*vx)-vx
    row_4 = -(2*cf*lf**2+2*cr*lr**2)/(iz*vx)

    if contains_symbolics(vehicle_params):
        return ca.vertcat(
            0,
            row_2,
            0,
            row_4
        )
    else:
        return np.array([[0, row_2, 0, row_4]]).T
    
    
def get_continuous_system(vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()) -> tuple[np.ndarray | ca.SX, np.ndarray | ca.SX]:
    A_cont = get_A_cont(vehicle_params=vehicle_params)
    B_steer_cont = get_B_steer_cont(vehicle_params=vehicle_params)
    B_ref_cont = get_B_ref_cont(vehicle_params=vehicle_params)
    if contains_symbolics(B_steer_cont) or contains_symbolics(B_ref_cont):
        B_cont = ca.horzcat(B_steer_cont, B_ref_cont)
    else:
        B_cont = np.hstack((B_steer_cont, B_ref_cont))  # reference becomes part of the input (fixed at runtime)
    return A_cont, B_cont


def get_discrete_system(vehicle_params: dict[str, float | ca.SX] | None = VehicleParams(), dt: float = 0.05) -> tuple[np.ndarray | ca.SX, np.ndarray | ca.SX]:
    A_cont, B_cont = get_continuous_system(vehicle_params=vehicle_params)
    if contains_symbolics(A_cont) or contains_symbolics(B_cont):
        sysd = cont2discrete_symbolic(A=A_cont, B=B_cont, dt=dt, method="bilinear")
    else:
        sysd = cont2discrete(system=(A_cont, B_cont, np.eye(4), np.zeros(B_cont.shape)), dt=dt, method="bilinear")
    A_disc = sysd[0]
    B_disc = sysd[1]
    return A_disc, B_disc


class VehicleSteeringEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # TODO: remove unnecessary inputs
    def __init__(
        self,
        dt: float = 0.05,
        max_time: float = 10.0,  # TODO: check max_time value
        vehicle_params: VehicleParams = VehicleParams(),
        steer_max: float = np.deg2rad(90),  # [rad]
        curvature_max: float = 1,  # [1/m]
        road_bank_angle: float = 0.0,  # [rad]
        reference_type: str = "straight",
        # TODO: check obs_space limits for derivatives
        observation_space: gym.spaces.Box = gym.spaces.Box(
            low=np.array([-1.5, -100, -90*np.pi/180, -100]),
            high=np.array([1.5,  100,  90*np.pi/180,  100]),
            dtype=np.float64,
        ),
        init_state_dist: dict[str, np.ndarray] = {
            "low": np.array([-0.1, 0.0, -0.1, 0.0]),
            "high": np.array([0.1, 0.0,  0.1, 0.0]),
        },
    ):
        super().__init__()

        self.init_state_dist = init_state_dist
        self.observation_space = observation_space
        self.action_space = gym.spaces.Box(
            low=np.array([-steer_max/vehicle_params.isw, -vehicle_params.vx*curvature_max]),
            high=np.array([steer_max/vehicle_params.isw,  vehicle_params.vx*curvature_max]),
            dtype=np.float64,
        )
        self.dt = dt
        self.max_time = max_time
        self.A, self.B = get_continuous_system()
        self.terminal_radius = 0.05  # [m], acceptable distance to centerline
        self.trajectory = []
        self.road_bank_angle = road_bank_angle
        self.reference_type = reference_type
        self.X = 0  # for tracking the global position of the vehicle
        self.vx = vehicle_params.vx

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        self.action_to_take = action
        self.u = action  # TODO: disturb action?

        road_bank_effect = np.sin(self.road_bank_angle) * np.array([0, 9.81, 0, 0])

        t_span = (0, self.dt)
        # if self.u.shape[0] == 1:  # only steering, no curvature reference
        #     system_dynamics = lambda t, x: self.A @ x + self.B[:,0] * self.u + road_bank_effect
        # else:  # steering and curvature reference
        if self.reference_type == "straight":
            curvature = 0.0
        elif self.reference_type == "double_lane_change":
            X_preview = self.X + self.vx * self.dt * np.arange(3)
            curvature = get_double_lane_change_data(X_preview)[0][0]  # get the current road curvature
        else:
            raise ValueError("Invalid reference type")
        u_aug = np.vstack((self.u, curvature*self.vx)).ravel()  # update the reference yaw rate
        self.X = self.X + self.vx * self.dt
        system_dynamics = lambda t, x: self.A @ x + self.B @ u_aug + road_bank_effect
        sol = solve_ivp(system_dynamics, t_span, self.state)
        self.state = sol.y[:, -1]

        observation = self.state
        self.trajectory.append(observation)
        reward = self._calculate_reward()

        if self.state not in self.observation_space:
            reward -= 50

        is_terminated = self._is_done()

        self.time += self.dt
        is_truncated = self.time > self.max_time
        
        info = {}

        return observation, reward, is_terminated, is_truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        self._np_random = np.random.RandomState(seed)
        self.state_trajectory = None
        self.action_to_take = None
        self.state = np.random.uniform(
            low=self.init_state_dist["low"], high=self.init_state_dist["high"]
        )
        self.time = 0.0

        self.trajectory = []
        plt.close("all")

        return self.state, {}

    def _calculate_reward(self):
        # quadratic cost on lateral and heading errors, and their derivatives

        distance_from_centerline = self.state[0]
        heading_error = self.state[1]
        ey_weight = 1.0
        epsi_weight = 10.0
        reward = -self.dt * (ey_weight * distance_from_centerline**2 + epsi_weight * heading_error**2)

        max_derivative = np.max(np.abs(self.state[[1,3]]))
        close_to_zero = distance_from_centerline < self.terminal_radius and max_derivative < 0.1
        if close_to_zero:
            reward += 50
        
        return reward

    def _is_done(self):
        # episode is done if the states (errors and derivatives) are close to the origin
        distance_from_centerline = self.state[0]
        max_derivative = np.max(np.abs(self.state[[1,3]]))
        close_to_zero = distance_from_centerline < self.terminal_radius and max_derivative < 0.1
        close_to_zero = close_to_zero and False  # TODO: remove when alles gut

        # also terminate if the states are outside the observation space
        outside_bounds = self.state not in self.observation_space

        done = close_to_zero or outside_bounds

        if done:
            print(
                f"Close to zero: {close_to_zero}, Outside bounds: {outside_bounds}"
            )

        return done
    

if __name__ == '__main__':
    # environment setup
    env = VehicleSteeringEnv(reference_type="double_lane_change")
    s, _ = env.reset(seed=0)

    # controller setup
    A, B = get_continuous_system()
    B = B[:, 0].reshape(-1, 1)  # only steering input
    Q = np.diag([1, 1e-3, 1, 1e-3])
    R = np.diag([1])
    K, _ = lqr(A, B, Q, R)
    ctrl = lambda state: np.dot(-K, state)
    # ctrl = lambda state: -1.0 * state[0] - 0.1 * state[2]  # hand tuned P controller

    # logging setup
    n_iter = 150
    t = np.arange(n_iter) * env.dt
    S = np.nan * np.ones(shape=(4,n_iter))
    A = np.nan * np.ones(shape=(1,n_iter-1))

    # closed-loop simulation
    S[:, 0] = s    
    for i in range(n_iter - 1):
        lqr_input = ctrl(S[:, i])
        assert lqr_input < env.action_space.high[0], "Control input exceeds action space"
        assert lqr_input > env.action_space.low[0], "Control input exceeds action space"

        A[:, i] = lqr_input
        S[:, i+1], r, term, trunc, _ = env.step(A[:, i])

        assert not trunc, "Simulation was truncated"
        if term:
            print("Terminated at iteration", i)
            break
    
    # ignore the iterations where the simulation was terminated
    t = t[:i+2]
    S = S[:, :i+2]
    A = A[:, :i+1]

    # plot the results
    fig_traj, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    axs[0].plot(t, S[0, :])
    axs[1].plot(t, np.rad2deg(S[2, :]))
    axs[2].plot(t[:-1], np.rad2deg(A.ravel()))
    axs[2].axhline(np.rad2deg(env.action_space.high[0]), color="r", linestyle='--')
    axs[2].axhline(np.rad2deg(env.action_space.low[0]), color="r", linestyle='--')
    axs[0].set_ylabel("$e_y$ [m]")
    axs[1].set_ylabel("$e_\psi$ [rad]")
    axs[2].set_ylabel("$\delta$ [rad]")
    axs[2].set_xlabel("$t$ [s]")
    
    plt.show(block=False)
    print("Press ENTER to close the plot")
    input()
    plt.close()
