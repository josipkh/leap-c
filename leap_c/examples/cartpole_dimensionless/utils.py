import numpy as np
from collections import OrderedDict
from copy import deepcopy
from leap_c.examples.cartpole_dimensionless.config import CartPoleParams, get_default_cartpole_params


def get_transformation_matrices(cartpole_params: CartPoleParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    l = cartpole_params.l.item()  # length of the rod
    m = cartpole_params.M.item()  # mass of the cart
    g = cartpole_params.g.item()  # gravity constant

    Mx = np.diag([l, 1.0, np.sqrt(g*l), np.sqrt(g/l)])
    Mu = np.diag([m*g])
    Mt = np.diag([np.sqrt(l/g)])
    
    return Mx, Mu, Mt


def convert_dataclass_to_dict(cartpole_params: CartPoleParams) -> OrderedDict[str, np.ndarray]:
    assert isinstance(cartpole_params, CartPoleParams), "Input must be a CartPoleParams dataclass"
    return OrderedDict(
        [
            ("M", cartpole_params.M),     # mass of the cart [kg]
            ("m", cartpole_params.m),     # mass of the ball [kg]
            ("g", cartpole_params.g),     # gravity constant [m/s^2]
            ("l", cartpole_params.l),     # length of the rod [m]
            # The quadratic cost matrix is calculated according to L@L.T
            ("L11", cartpole_params.L11),
            ("L22", cartpole_params.L22),
            ("L33", cartpole_params.L33),
            ("L44", cartpole_params.L44),
            ("L55", cartpole_params.L55),
            ("Lloweroffdiag", cartpole_params.Lloweroffdiag),
            ("c1", cartpole_params.c1),    # position linear cost, only used for EXTERNAL cost
            ("c2", cartpole_params.c2),    # theta linear cost, only used for EXTERNAL cost
            ("c3", cartpole_params.c3),    # v linear cost, only used for EXTERNAL cost
            ("c4", cartpole_params.c4),    # thetadot linear cost, only used for EXTERNAL cost
            ("c5", cartpole_params.c5),    # u linear cost, only used for EXTERNAL cost
            ("xref1", cartpole_params.xref1), # reference position, only used for NONLINEAR_LS cost
            ("xref2", cartpole_params.xref2), # reference theta, only used for NONLINEAR_LS cost
            ("xref3", cartpole_params.xref3), # reference v, only used for NONLINEAR_LS cost
            ("xref4", cartpole_params.xref4), # reference thetadot, only used for NONLINEAR_LS cost
            ("uref", cartpole_params.uref),
            ("dt", cartpole_params.dt), # time step [s]
            ("Fmax", cartpole_params.Fmax),  # maximum force applied to the cart [N]
            ("gamma", cartpole_params.gamma),  # discount factor for the cost function
        ]
    )


def convert_dict_to_dataclass(params_dict: OrderedDict[str, np.ndarray]) -> CartPoleParams:
    """Converts an OrderedDict of parameters to a CartPoleParams dataclass."""
    assert isinstance(params_dict, OrderedDict), "Input must be an OrderedDict"
    return CartPoleParams(
        M=params_dict["M"],
        m=params_dict["m"],
        g=params_dict["g"],
        l=params_dict["l"],
        L11=params_dict["L11"],
        L22=params_dict["L22"],
        L33=params_dict["L33"],
        L44=params_dict["L44"],
        L55=params_dict["L55"],
        Lloweroffdiag=params_dict["Lloweroffdiag"],
        c1=params_dict["c1"],
        c2=params_dict["c2"],
        c3=params_dict["c3"],
        c4=params_dict["c4"],
        c5=params_dict["c5"],
        xref1=params_dict["xref1"],
        xref2=params_dict["xref2"],
        xref3=params_dict["xref3"],
        xref4=params_dict["xref4"],
        uref=params_dict["uref"],
        dt=params_dict["dt"],
        Fmax=params_dict["Fmax"],
        gamma=params_dict["gamma"]
    )


def get_similar_cartpole_params(reference_params: CartPoleParams, cart_mass: float, rod_length: float) -> CartPoleParams:
    """Returns the parameters of a cartpole system (MDP) dynamically similar to the reference one."""
    Mx, Mu, _ = get_transformation_matrices(reference_params)

    new_params = deepcopy(reference_params)
    new_params.M = np.array([cart_mass])
    new_params.l = np.array([rod_length])

    # match the Pi-group(s)
    new_params.m = reference_params.m * (new_params.M / reference_params.M)

    # match the cost matrices (just Q and R for now)
    Q = np.diag([
        reference_params.L11.item()**2,
        reference_params.L22.item()**2,
        reference_params.L33.item()**2,
        reference_params.L44.item()**2,
    ])
    R = np.diag([reference_params.L55.item()**2])
    mx, mu, _ = get_transformation_matrices(new_params)
    M = Mx @ np.linalg.inv(mx)
    q_diag = (M.T @ Q @ M).diagonal()
    M = Mu @ np.linalg.inv(mu)
    r_diag = (M.T @ R @ M).diagonal()

    for k in range(5):
        new_params.__setattr__(f"L{k+1}{k+1}", np.array([np.sqrt(q_diag[k] if k < 4 else r_diag[k-4])]))
    
    # match the input constraint
    new_params.Fmax = reference_params.Fmax * (new_params.M / reference_params.M)

    # match the sampling time
    new_params.dt = reference_params.dt * np.sqrt(new_params.l / reference_params.l)

    # match the discount factor (through the continuous discount rate r = -log(gamma)/dt)
    new_params.gamma = np.power(reference_params.gamma, new_params.dt / reference_params.dt)

    return new_params


if __name__ == "__main__":
    from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params
    params = get_default_cartpole_params()    
    Mx, Mu, Mt = get_transformation_matrices(params)
    params_dict = convert_dataclass_to_dict(params)
    params_dataclass = convert_dict_to_dataclass(params_dict)
    similar_params = get_similar_cartpole_params(reference_params=params, cart_mass=0.5, rod_length=0.1)
    assert similar_params.M.item()/similar_params.m.item() == params.M.item()/params.m.item(), "Pi-group mismatch"
