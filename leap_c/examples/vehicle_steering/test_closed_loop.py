import numpy as np

from leap_c.examples.vehicle_steering.env import VehicleSteeringEnv
from leap_c.examples.vehicle_steering.mpc import VehicleSteeringMPC
from leap_c.examples.vehicle_steering.utils import get_double_lane_change_data, frenet2inertial

MAX_FINAL_DIST = 0.1    # maximum tolerated error
MAX_FINAL_VEL = 0.1     # maximum tolerated error derivative


def run_closed_loop(
    mpc: VehicleSteeringMPC,
    env: VehicleSteeringEnv,
    n_iter: int = 50,
) -> np.ndarray:

    nx = 4
    nu = 1

    s, _ = env.reset()

    states = np.zeros((n_iter, nx))
    states[0, :] = s
    actions = np.zeros((n_iter, nu))
    for i in range(n_iter - 1):
        actions[i, :] = mpc.policy(state=states[i, :], p_global=None)[0]
        states[i + 1, :], _, _, _, _ = env.step(actions[i, :])

    dx = env.vx*env.dt
    x = np.arange(0, n_iter*dx, dx)  # global X position of the vehicle
    x = x[:,np.newaxis]  # to enable stacking
    return np.hstack([states, actions, x])


def visualize_closed_loop(results: np.ndarray, env: VehicleSteeringEnv) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams['axes.xmargin'] = 0  # tight x range
    vehicle_params = env.vehicle_params
    isw = vehicle_params.isw

    e1 = results[:, 0]
    e2 = results[:, 2]
    delta_sw = isw * np.rad2deg(results[:, 4])
    x = results[:,5]
    steer_max = isw * np.rad2deg(env.action_space.high[0])

    # results in the error frame
    fig, ax = plt.subplots(5, sharex=True, constrained_layout=True)
    fig.suptitle('Vehicle steering in closed loop (error frame)')
    ax[0].plot(x, e1)
    ax[0].set_ylabel('$e_y$ [m]')
    ax[1].plot(x, np.rad2deg(e2))
    ax[1].set_ylabel('$e_\psi$ [deg]')
    ax[2].plot(x, results[:, 1])
    ax[2].set_ylabel('$\dot{e}_y$ [m/s]')
    ax[3].plot(x, np.rad2deg(results[:, 3]))
    ax[3].set_ylabel('$\dot{e}_\psi$ [deg/s]')
    ax[4].plot(x, delta_sw)
    ax[4].axhline( steer_max, color="r", linestyle='--')
    ax[4].axhline(-steer_max, color="r", linestyle='--')
    ax[4].set_ylabel('$\delta_\mathrm{sw}$ [deg]')
    ax[4].set_xlabel('$x$ [m]')

    # results in the inertial frame
    _, y_ref, psi_ref = get_double_lane_change_data(x)
    _, y = frenet2inertial(e1=e1, e2=e2, psi_ref=psi_ref, vx=env.vx, dt=env.dt)
    psi = psi_ref + e2

    fig, ax = plt.subplots(3, sharex=True, constrained_layout=True)
    fig.suptitle('Vehicle steering in closed loop (inertial frame)')

    ax[0].plot(x, y)
    ax[0].plot(x, y_ref, 'r--')    
    ax[0].set_ylabel('$y$ [m]')

    ax[1].plot(x, np.rad2deg(psi))
    ax[1].plot(x, np.rad2deg(psi_ref), 'r--')
    ax[1].set_ylabel('$\psi$ [deg]')

    ax[2].plot(x, delta_sw)
    ax[2].axhline( steer_max, color="r", linestyle='--')
    ax[2].axhline(-steer_max, color="r", linestyle='--')
    ax[2].set_ylabel('$\delta_\mathrm{sw}$ [deg]')
    ax[2].set_xlabel('$x$ [m]')

    plt.show(block=False)
    print("Press ENTER to close the plot")
    input()
    plt.close()


def test_run_closed_loop(
    mpc: VehicleSteeringMPC, env: VehicleSteeringEnv, n_iter: int = 50
) -> None:
    sim_data = run_closed_loop(mpc=mpc, env=env, n_iter=n_iter)

    delta = np.rad2deg(sim_data[:, 4])
    steer_max = np.rad2deg(env.action_space.high[0])
    assert np.all(np.abs(delta) < steer_max), "steering angle out of bounds"

    errors_small_enough = np.max(np.abs(sim_data[-1,[0,2]])) < MAX_FINAL_DIST
    error_derivatives_small_enough = np.max(np.abs(sim_data[-1,[1,3]])) < MAX_FINAL_VEL

    if not errors_small_enough or not error_derivatives_small_enough:
        print("system not stabilized")
    else:
        print("closed-loop test passed")
    visualize_closed_loop(sim_data, env)


if __name__ == "__main__":
    run_dlc = True
    if run_dlc:
        reference_type = "double_lane_change"
        n_iter = 125
    else:
        reference_type = "straight"
        n_iter = 50

    test_run_closed_loop(mpc=VehicleSteeringMPC(learnable_params=["q_diag"]), 
                         env=VehicleSteeringEnv(reference_type=reference_type),
                         n_iter=n_iter)
