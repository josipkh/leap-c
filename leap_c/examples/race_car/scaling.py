"""Utility functions for dynamic matching and non-dimensionalization."""
import numpy as np
from leap_c.examples.race_car.config import CarParams, get_default_car_params
from copy import deepcopy


def get_cost_matrices(car_params: CarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the cost matrices Q, R and Q_e for the given race car."""
    Q = np.diag(car_params.q_diag_sqrt**2)
    R = np.diag(car_params.r_diag_sqrt**2)
    Q_e = np.diag(car_params.qe_diag_sqrt**2)
    return Q, R, Q_e


def get_transformation_matrices(
    car_params: CarParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    # s(physical) = Ms * s(dimensionless)
    l = car_params.l.item()  # length of the pole
    cr3 = car_params.cr3.item()  # [s/m] speed coefficient

    Mx = np.diag([l, l, 1.0, 1/cr3, 1.0, 1.0])  # states include the control inputs
    Mu = np.diag([1.0 / (l * cr3), 1.0 / (l * cr3)])  # inputs are rate of change of controls
    Mt = np.diag([cr3 * l])

    return Mx, Mu, Mt


def get_similar_car_params(
        reference_params: CarParams,
        new_length: float,
        new_mass: float,
        new_cr3: float
) -> CarParams:
    """Returns the parameters of a car dynamically similar to the reference one."""

    new_params = deepcopy(reference_params)
    new_params.l = np.array([new_length])
    new_params.m = np.array([new_mass])
    new_params.cr3 = np.array([new_cr3])

    # match the pi-groups
    pi_groups_ref = get_pi_groups(reference_params)
    new_params.lr = np.array([pi_groups_ref[0] * new_length])
    new_params.cm1 = np.array([pi_groups_ref[1] * new_mass / (new_length * new_cr3**2)])
    new_params.cm2 = np.array([pi_groups_ref[2] * new_mass / (new_cr3 * new_length)])
    new_params.cr0 = np.array([pi_groups_ref[3] * new_mass / (new_length * new_cr3**2)])
    new_params.cr2 = np.array([pi_groups_ref[4] * new_mass / new_length])

    # match the MPC cost matrices
    Q, R, Q_e = get_cost_matrices(reference_params)
    Mx, Mu, Mt = get_transformation_matrices(reference_params)
    mx, mu, mt = get_transformation_matrices(new_params)

    M = Mx @ np.linalg.inv(mx)
    q_diag = (M.T @ Q @ M).diagonal()
    qe_diag = (M.T @ Q_e @ M).diagonal()

    M = Mu @ np.linalg.inv(mu)
    r_diag = (M.T @ R @ M).diagonal()

    new_params.q_diag_sqrt = np.sqrt(q_diag)
    new_params.r_diag_sqrt = np.sqrt(r_diag)
    new_params.qe_diag_sqrt = np.sqrt(qe_diag)

    # match the constraints
    # D_max, D_min, delta_max, delta_min are not changed (they are dimensionless)
    new_params.dD_max = reference_params.dD_max * Mt / mt
    new_params.dD_min = reference_params.dD_min * Mt / mt
    new_params.ddelta_max = reference_params.ddelta_max * Mt / mt
    new_params.ddelta_min = reference_params.ddelta_min * Mt / mt

    MtCr3 = Mt.item() * reference_params.cr3.item()
    mtcr3 = mt.item() * new_params.cr3.item()
    new_params.a_lat_max = reference_params.a_lat_max * MtCr3 / mtcr3
    new_params.a_lat_min = reference_params.a_lat_min * MtCr3 / mtcr3
    new_params.a_long_max = reference_params.a_long_max * MtCr3 / mtcr3
    new_params.a_long_min = reference_params.a_long_min * MtCr3 / mtcr3
    new_params.n_max = reference_params.n_max * (new_length / reference_params.l.item())
    new_params.n_min = reference_params.n_min * (new_length / reference_params.l.item())

    # match the sampling time
    new_params.dt = reference_params.dt * (mt / Mt).item()

    # the discount factor (already dimensionless) stays the same

    # match the slack cost weights (sorry if it's unclear!)
    new_params.slack_n_linear = reference_params.slack_n_linear * reference_params.l.item() / new_params.l.item()
    new_params.slack_n_quadratic = reference_params.slack_n_quadratic * (reference_params.l.item() / new_params.l.item())**2
    new_params.slack_acc_linear = reference_params.slack_acc_linear * mtcr3 / MtCr3
    new_params.slack_acc_quadratic = reference_params.slack_acc_quadratic * (mtcr3 / MtCr3)**2

    return new_params


def get_pi_groups(car_params: CarParams) -> tuple[float, float]:
    """Returns the pi-groups for the given car."""
    m = car_params.m.item()
    l = car_params.l.item()
    cr3 = car_params.cr3.item()

    pi_1 = car_params.lr.item() / l
    pi_2 = car_params.cm1.item() * l * cr3**2 / m
    pi_3 = car_params.cm2.item() * l * cr3 / m
    pi_4 = car_params.cr0.item() * l * cr3**2 / m
    pi_5 = car_params.cr2.item() * l / m

    return np.array([pi_1, pi_2, pi_3, pi_4, pi_5])


def get_large_car_params() -> CarParams:
    """Returns the parameters of a specific larger car."""
    return get_similar_car_params(
        reference_params=get_default_car_params(),
        new_length=4.0,
        new_mass=1500.0,
        new_cr3=0.4,  # chosen approximately, top speed of small car is 2.5 m/s
    )


def compare_params(car1: CarParams, car2: CarParams):
    """Print the parameters of two cars side by side for comparison."""
    # Define parameter groups by field names
    chassis_fields = ['m', 'l', 'lr']
    force_fields = ['cm1', 'cm2', 'cr0', 'cr2', 'cr3']
    controller_fields = ['dt', 'gamma']

    # Helper to print groups side by side
    def print_group(group_name, group_fields):
        print(f"\n--- {group_name} Parameters ---")
        print(f"{'Parameter':<15} | {'Car 1':<20} | {'Car 2':<20}")
        print("-" * 60)
        for name in group_fields:
            val1 = getattr(car1, name)[0]
            val2 = getattr(car2, name)[0]
            # Convert numpy arrays to string, limit length for readability
            str_val1 = np.array2string(val1, precision=4, suppress_small=True)
            str_val2 = np.array2string(val2, precision=4, suppress_small=True)
            print(f"{name:<15} | {str_val1:<20} | {str_val2:<20}")

    print_group("Chassis", chassis_fields)
    print_group("Longitudinal Force", force_fields)
    print_group("Controller", controller_fields)
    print('\n')
    

if __name__ == "__main__":
    from leap_c.examples.race_car.track import get_track
    
    car_params_ref = get_default_car_params()
    car_params_sim = get_large_car_params()
    
    # check dynamic similarity
    assert np.allclose(get_pi_groups(car_params_ref), get_pi_groups(car_params_sim))

    # check the track scaling
    track_data = get_track(car_params=car_params_ref)
    track_data_sim = get_track(car_params=car_params_sim)
    l_ref = car_params_ref.l.item()
    l_sim = car_params_sim.l.item()
    assert np.allclose(track_data[0] / l_ref, track_data_sim[0] / l_sim)  # s
    assert np.allclose(track_data[1] / l_ref, track_data_sim[1] / l_sim)  # x
    assert np.allclose(track_data[2] / l_ref, track_data_sim[2] / l_sim)  # y
    assert np.allclose(track_data[3], track_data_sim[3])  # psi
    assert np.allclose(track_data[4] * l_ref, track_data_sim[4] * l_sim)  # kappa

    # check the cost matrices
    Q, R, Q_e = get_cost_matrices(car_params_ref)
    q, r, q_e = get_cost_matrices(car_params_sim)
    Mx, Mu, Mt = get_transformation_matrices(car_params_ref)
    mx, mu, mt = get_transformation_matrices(car_params_sim)
    assert np.allclose(Mx @ Q @ Mx, mx @ q @ mx)
    assert np.allclose(Mu @ R @ Mu, mu @ r @ mu)
    assert np.allclose(Mx @ Q_e @ Mx, mx @ q_e @ mx)

    # check the sampling time
    assert np.allclose(car_params_ref.dt / Mt, car_params_sim.dt / mt)

    # check the acceleration limits
    acc_scale_ref = 1 / (car_params_ref.l.item() * car_params_ref.cr3.item()**2)
    acc_scale_sim = 1 / (car_params_sim.l.item() * car_params_sim.cr3.item()**2)
    assert np.allclose(car_params_ref.a_lat_max / acc_scale_ref, car_params_sim.a_lat_max / acc_scale_sim)
    assert np.allclose(car_params_ref.a_lat_min / acc_scale_ref, car_params_sim.a_lat_min / acc_scale_sim)
    assert np.allclose(car_params_ref.a_long_max / acc_scale_ref, car_params_sim.a_long_max / acc_scale_sim)
    assert np.allclose(car_params_ref.a_long_min / acc_scale_ref, car_params_sim.a_long_min / acc_scale_sim)

    compare_params(car_params_ref, car_params_sim)

    print("All checks passed.")
