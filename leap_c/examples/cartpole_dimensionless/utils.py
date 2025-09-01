import numpy as np
from copy import deepcopy
from leap_c.examples.cartpole_dimensionless.config import CartPoleParams
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def get_transformation_matrices(
    cartpole_params: CartPoleParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    l = cartpole_params.l.item()  # length of the pole
    m = cartpole_params.M.item()  # mass of the cart
    g = cartpole_params.g.item()  # gravity constant

    Mx = np.diag([l, 1.0, np.sqrt(g * l), np.sqrt(g / l)])
    Mu = np.diag([m * g])
    Mt = np.diag([np.sqrt(l / g)])

    return Mx, Mu, Mt


def get_similar_cartpole_params(
    reference_params: CartPoleParams, pole_length: float
) -> CartPoleParams:
    """Returns the parameters of a cartpole system (MDP) dynamically similar to the reference one.

    Dynamic matching is based on eq. (28) in this paper: https://ieeexplore.ieee.org/document/10178119
    which contains five parameters: pole length, cart mass, pole mass, cart friction and gravity.

    It is assumed that the friction and gravity cannot be changed.
    """
    Mx, Mu, _ = get_transformation_matrices(reference_params)

    new_params = deepcopy(reference_params)
    new_params.l = np.array([pole_length])

    # match the pi-groups
    new_params.M = np.array(
        [
            reference_params.M.item()
            * np.sqrt(new_params.l.item() / reference_params.l.item())
        ]
    )  # keep relative friction
    new_params.m = reference_params.m * (
        new_params.M / reference_params.M
    )  # mass ratio

    # check the pi-groups
    pi_1_ref, pi_2_ref = get_pi_groups(reference_params)
    pi_1_sim, pi_2_sim = get_pi_groups(new_params)
    assert np.allclose(pi_1_ref, pi_1_sim), "Pi-group 1 mismatch"
    assert np.allclose(pi_2_ref, pi_2_sim), "Pi-group 2 mismatch"    

    # match the cost matrices (just Q and R for now)
    Q, R = get_cost_matrices(reference_params)
    mx, mu, _ = get_transformation_matrices(new_params)
    M = Mx @ np.linalg.inv(mx)
    q_diag = (M.T @ Q @ M).diagonal()
    M = Mu @ np.linalg.inv(mu)
    r_diag = (M.T @ R @ M).diagonal()

    for k in range(5):
        new_params.__setattr__(
            f"L{k + 1}{k + 1}",
            np.array([np.sqrt(q_diag[k] if k < 4 else r_diag[k - 4])]),
        )

    # check the matrices
    q, r = get_cost_matrices(new_params)
    assert np.allclose(Mx @ Q @ Mx, mx @ q @ mx)
    assert np.allclose(Mu @ R @ Mu, mu @ r @ mu)

    # match the input constraint
    new_params.Fmax = reference_params.Fmax * (new_params.M / reference_params.M)

    # match the sampling time
    new_params.dt = reference_params.dt * np.sqrt(new_params.l / reference_params.l)

    # TODO: match the discount factor (through the continuous discount rate r = -log(gamma)/dt)
    new_params.gamma = np.power(
        reference_params.gamma, new_params.dt / reference_params.dt
    )

    return new_params


def get_cost_matrices(cartpole_params: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns the cost matrices Q and R for the given cartpole system."""
    Q = np.diag(
        [
            cartpole_params.L11.item() ** 2,
            cartpole_params.L22.item() ** 2,
            cartpole_params.L33.item() ** 2,
            cartpole_params.L44.item() ** 2,
        ]
    )
    R = np.diag([cartpole_params.L55.item() ** 2])
    return Q, R


def get_pi_groups(cartpole_params: CartPoleParams) -> tuple[float, float]:
    """Returns the pi-groups for the given cartpole system."""
    M = cartpole_params.M.item()
    m = cartpole_params.m.item()
    l = cartpole_params.l.item()
    mu_f = cartpole_params.mu_f.item()
    g = cartpole_params.g.item()

    pi_1 = m / M
    pi_2 = mu_f / M * np.sqrt(l / g)

    return pi_1, pi_2


def acados_cleanup(path="."):
    files_to_delete = ["acados_sim.json", "acados_ocp.json"]
    folder_to_delete = "c_generated_code"

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
    if confirm != "y":
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


def plot_results(main_folder, plot_std=False):
    """Plots the experiment results averaged over the seeds."""

    experiments = ['default', 'small', 'large', 'transfer_small', 'transfer_large']
    seeds = ['0', '1', '2', '3', '4']
    metric = 'score'
    output_file = os.path.join(main_folder, 'results.pdf')

    # Collect data
    experiment_results = {}

    for exp in experiments:
        seed_dfs = []
        for seed in seeds:
            file_path = os.path.join(main_folder, exp, seed, 'val_log.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                # keep only the scores created after transfer
                if df.shape[0] > len(seeds) + 1:
                    df = df.tail(len(seeds) + 1)
                seed_dfs.append(df)
            else:
                print(f"Warning: Missing file {file_path}")
        if seed_dfs:
            # Stack into 3D array for mean/std
            combined = pd.concat(seed_dfs, axis=0, keys=range(len(seed_dfs)))
            mean_df = combined.groupby(level=1).mean()
            std_df = combined.groupby(level=1).std()
            experiment_results[exp] = {'mean': mean_df, 'std': std_df}
        else:
            print(f"Warning: No valid seed logs found for {exp}")

    # Plotting
    try:
        plt.figure(figsize=(10, 6))
    except Exception as e:
    # switch to a headless backend (https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

    for exp_name, data in experiment_results.items():
        if metric in data['mean'].columns:
            steps = data['mean'].index
            mean_values = data['mean'][metric]
            plt.plot(steps, mean_values, label=exp_name)
            if plot_std:
                std_values = data['std'][metric]
                plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.2)

    # plt.title(f'Metric: {metric}')
    plt.xlabel("Number of samples")
    plt.ylabel("Validation score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to PDF
    with PdfPages(output_file) as pdf:
        pdf.savefig()
        print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    from leap_c.examples.cartpole_dimensionless.config import (
        get_default_cartpole_params,
    )

    params = get_default_cartpole_params()
    Mx, Mu, Mt = get_transformation_matrices(params)
    similar_params = get_similar_cartpole_params(
        reference_params=params, pole_length=0.1
    )

    # check pi groups
    pi_1_ref, pi_2_ref = get_pi_groups(params)
    pi_1_sim, pi_2_sim = get_pi_groups(similar_params)
    assert np.allclose(pi_1_ref, pi_1_sim), "Pi-group 1 mismatch"
    assert np.allclose(pi_2_ref, pi_2_sim), "Pi-group 2 mismatch"
    print("ok")
