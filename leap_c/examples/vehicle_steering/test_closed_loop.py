import numpy as np

from leap_c.examples.vehicle_steering.env import VehicleSteeringEnv
from leap_c.examples.vehicle_steering.mpc import VehicleSteeringMPC

MAX_FINAL_DIST = 0.1    # maximum tolerated error
MAX_FINAL_VEL = 0.1     # maximum tolerated error derivative

def run_closed_loop(
    mpc: VehicleSteeringMPC,
    env: VehicleSteeringEnv,
    n_iter: int = int(50),
) -> np.ndarray:
    """Run a closed-loop simulation of the system using a given model predictive controller (MPC) and environment.

    Args:
        mpc (VehicleSteeringMPC): The model predictive controller to use for generating actions.
        env (VehicleSteeringEnv): The environment representing the system.
        n_iter (int, optional): The number of iterations to run the simulation. Defaults to 200.

    Returns:
        np.ndarray: A numpy array containing the states and actions over the simulation. The array has shape (n_iter, nx+nu).

    """
    nx = 4
    nu = 1

    s, _ = env.reset()

    states = np.zeros((n_iter, nx))
    states[0, :] = s
    actions = np.zeros((n_iter, nu))
    for i in range(n_iter - 1):
        actions[i, :] = mpc.policy(state=states[i, :], p_global=None)[0]
        states[i + 1, :], _, _, _, _ = env.step(actions[i, :])

    return np.hstack([states, actions])


def visualize_closed_loop(results: np.ndarray) -> None:
    """Visualize the results of a closed-loop simulation.

    Args:
        results (np.ndarray): A numpy array containing the states and actions over the simulation. The array has shape (n_iter, nx+nu).

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, sharex=True)
    fig.suptitle('Vehicle steering stabilization')
    ax[0].plot(results[:, 0])
    ax[0].set_ylabel('$e_y$')
    ax[1].plot(results[:, 2])
    ax[1].set_ylabel('$e_\psi$')
    ax[2].plot(results[:, 1])
    ax[2].set_ylabel('$\dot{e}_y$')
    ax[3].plot(results[:, 3])
    ax[3].set_ylabel('$\dot{e}_\psi$')
    ax[4].plot(results[:, 4])
    ax[4].set_ylabel('$\delta$')

    plt.show(block=False)
    print("Press ENTER to close the plot")
    input()
    plt.close()


def test_run_closed_loop(
    mpc: VehicleSteeringMPC, env: VehicleSteeringEnv, n_iter: int = int(50)
) -> None:
    sim_data = run_closed_loop(mpc=mpc, env=env, n_iter=n_iter)
    visualize_closed_loop(sim_data)
    assert np.max(np.abs(sim_data[-1,[0,2]])) < MAX_FINAL_DIST  # Check that the final position is close to the origin
    assert np.max(np.abs(sim_data[-1,[1,3]])) < MAX_FINAL_VEL   # Check that the final velocity is close to zero
    print("Closed-loop test passed!")



if __name__ == "__main__":
    test_run_closed_loop(mpc=VehicleSteeringMPC(learnable_params=["q_diag"]), env=VehicleSteeringEnv())
