import numpy as np
from leap_c.examples.cartpole_dimensionless.config import CartPoleParams
from collections import OrderedDict


def get_transformation_matrices(cartpole_params: CartPoleParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    l = cartpole_params.l.item()  # length of the rod
    m = cartpole_params.M.item()  # mass of the cart
    g = cartpole_params.g.item()  # gravity constant

    Mx = np.diag([l, 1.0, np.sqrt(g*l), np.sqrt(g/l)])
    Mu = np.diag([m*g])
    Mt = np.diag([np.sqrt(l/g)])
    
    return Mx, Mu, Mt


def get_params_as_ordered_dict(cartpole_params: CartPoleParams) -> OrderedDict[str, np.ndarray]:
    params_dict = OrderedDict(
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
    return params_dict


def get_params_as_dataclass(params_dict: OrderedDict[str, np.ndarray]) -> CartPoleParams:
    """Converts an OrderedDict of parameters to a CartPoleParams dataclass."""
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


if __name__ == "__main__":
    from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params
    # Example usage
    params = get_default_cartpole_params()    
    Mx, Mu, Mt = get_transformation_matrices(params)
    params_dict = get_params_as_ordered_dict(params)