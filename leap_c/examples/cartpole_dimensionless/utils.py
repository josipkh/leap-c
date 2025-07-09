import numpy as np
from copy import deepcopy
from leap_c.examples.cartpole_dimensionless.config import CartPoleParams
import os
import shutil


def get_transformation_matrices(cartpole_params: CartPoleParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    l = cartpole_params.l.item()  # length of the rod
    m = cartpole_params.M.item()  # mass of the cart
    g = cartpole_params.g.item()  # gravity constant

    Mx = np.diag([l, 1.0, np.sqrt(g*l), np.sqrt(g/l)])
    Mu = np.diag([m*g])
    Mt = np.diag([np.sqrt(l/g)])
    
    return Mx, Mu, Mt


def get_similar_cartpole_params(reference_params: CartPoleParams, cart_mass: float, rod_length: float) -> CartPoleParams:
    """Returns the parameters of a cartpole system (MDP) dynamically similar to the reference one."""
    Mx, Mu, _ = get_transformation_matrices(reference_params)

    new_params = deepcopy(reference_params)
    new_params.M = np.array([cart_mass])
    new_params.l = np.array([rod_length])

    # match the Pi-group(s)
    new_params.m = reference_params.m * (new_params.M / reference_params.M)

    # match the cost matrices (just Q and R for now)
    Q, R = get_cost_matrices(reference_params)
    mx, mu, _ = get_transformation_matrices(new_params)
    M = Mx @ np.linalg.inv(mx)
    q_diag = (M.T @ Q @ M).diagonal()
    M = Mu @ np.linalg.inv(mu)
    r_diag = (M.T @ R @ M).diagonal()

    for k in range(5):
        new_params.__setattr__(f"L{k+1}{k+1}", np.array([np.sqrt(q_diag[k] if k < 4 else r_diag[k-4])]))

    # check the matrices
    q, r = get_cost_matrices(new_params)
    assert np.allclose(Mx @ Q @ Mx, mx @ q @ mx)
    assert np.allclose(Mu @ R @ Mu, mu @ r @ mu)
    
    # match the input constraint
    new_params.Fmax = reference_params.Fmax * (new_params.M / reference_params.M)

    # match the sampling time
    new_params.dt = reference_params.dt * np.sqrt(new_params.l / reference_params.l)

    # match the discount factor (through the continuous discount rate r = -log(gamma)/dt)
    new_params.gamma = np.power(reference_params.gamma, new_params.dt / reference_params.dt)

    return new_params


def get_cost_matrices(cartpole_params: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns the cost matrices Q and R for the given cartpole system."""
    Q = np.diag([
        cartpole_params.L11.item()**2,
        cartpole_params.L22.item()**2,
        cartpole_params.L33.item()**2,
        cartpole_params.L44.item()**2,
    ])
    R = np.diag([cartpole_params.L55.item()**2])
    return Q, R


def acados_cleanup(path='.'):
    files_to_delete = ['acados_sim.json', 'acados_ocp.json']
    folder_to_delete = 'c_generated_code'

    items_to_delete = []

    # Check files
    for filename in files_to_delete:
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            items_to_delete.append(file_path)

    # Check folder
    folder_path = os.path.join(path, folder_to_delete)
    if os.path.isdir(folder_path):
        items_to_delete.append(folder_path)

    # If nothing to delete
    if not items_to_delete:
        print("No matching files or folders found to delete.")
        return

    # List items to be deleted
    print("The following items will be deleted:")
    for item in items_to_delete:
        print(f"  - {item}")

    # Ask for confirmation
    confirm = input("Do you want to proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # Perform deletion
    for item in items_to_delete:
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"Deleted folder: {item}")
            else:
                os.remove(item)
                print(f"Deleted file: {item}")
        except Exception as e:
            print(f"Error deleting {item}: {e}")


if __name__ == "__main__":
    from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params
    params = get_default_cartpole_params()    
    Mx, Mu, Mt = get_transformation_matrices(params)
    similar_params = get_similar_cartpole_params(reference_params=params, cart_mass=0.5, rod_length=0.1)
    assert similar_params.M.item()/similar_params.m.item() == params.M.item()/params.m.item(), "Pi-group mismatch"
    print("ok")
