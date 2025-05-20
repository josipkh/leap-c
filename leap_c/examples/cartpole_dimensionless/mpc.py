from collections import OrderedDict

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi.tools import struct_symSX
from leap_c.examples.util import (
    assign_lower_triangular,
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.acados.mpc import Mpc

dimensionless = True  # global setting

PARAMS = OrderedDict(
    [
        ("M", np.array([1.0])),     # mass of the cart [kg]
        ("m", np.array([0.1])),     # mass of the ball [kg]
        ("g", np.array([9.81])),    # gravity constant [m/s^2]
        ("l", np.array([0.8])),     # length of the rod [m]
        # The quadratic cost matrix is calculated according to L@L.T
        ("L11", np.array([np.sqrt(2e3)])),
        ("L22", np.array([np.sqrt(2e3)])),
        ("L33", np.array([np.sqrt(1e-2)])),
        ("L44", np.array([np.sqrt(1e-2)])),
        ("L55", np.array([np.sqrt(2e-1)])),
        ("Lloweroffdiag", np.array([0.0] * (4 + 3 + 2 + 1))),
        ("c1", np.array([0.0])),    # position linear cost, only used for EXTERNAL cost
        ("c2", np.array([0.0])),    # theta linear cost, only used for EXTERNAL cost
        ("c3", np.array([0.0])),    # v linear cost, only used for EXTERNAL cost
        ("c4", np.array([0.0])),    # thetadot linear cost, only used for EXTERNAL cost
        ("c5", np.array([0.0])),    # u linear cost, only used for EXTERNAL cost
        ("xref1", np.array([0.0])), # reference position, only used for NONLINEAR_LS cost
        ("xref2", np.array([0.0])), # reference theta, only used for NONLINEAR_LS cost
        ("xref3", np.array([0.0])), # reference v, only used for NONLINEAR_LS cost
        ("xref4", np.array([0.0])), # reference thetadot, only used for NONLINEAR_LS cost
        ("uref", np.array([0.0])),  # reference u, only used for NONLINEAR_LS cost
    ]
)


class PendulumOnCartMpcDimensionless(Mpc):
    """
    Describes an inverted pendulum on a cart.
    The (possibly learnable) parameters of the system are given by
        ---------Dynamics---------
        M: mass of the cart [kg]
        m: mass of the ball [kg]
        g: gravity constant [m/s^2]
        l: length of the rod [m]

        ---------Cost---------
        The parameters of the quadratic cost matrix describe a cholesky factorization of the cost matrix.
        In more detail, the cost matrix W is calculated like this:
        L_diag = np.diag([L11, L22, L33, L44, L55]) # cost matrix factorization diagonal
        L_diag[np.tril_indices_from(L_diag, -1)] = L_lower_offdiag
        W = L@L.T

        If the cost is a least squares cost (see docstring of __init__), the parameters
        c1, c2, c3, c4, c5 are not used.
        Instead, the parameters xref1, xref2, xref3, xref4, uref are used for the reference vector.
        If the cost is not the least squares cost, the parameters
        xref1, xref2, xref3, xref4, uref are not used.
        Instead, the parameters c1, c2, c3, c4, c5 are used for the linear cost vector.

        The possible costs are either a least squares cost or a general quadratic cost.
        The least squares cost takes the form of:
            z_ref = cat(xref, uref)
            cost = 0.5 * (z - z_ref).T @ W @ (z - z_ref), where W is the quadratic cost matrix from above.
        The general quadratic cost takes the form of:
            z = cat(x, u)
            cost = 0.5 * z.T @ W @ z + c.T @ z, where W is the quadratic cost matrix from above

    """

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        T_horizon: float = 1.0,
        Fmax: float = 80.0,
        discount_factor: float = 0.99,
        n_batch: int = 64,
        exact_hess_dyn: bool = True,
        cost_type: str = "NONLINEAR_LS",
    ):
        """
        Args:
            params: A dict with the parameters of the ocp, together with their default values.
                For a description of the parameters, see the docstring of the class.
            learnable_params: A list of the parameters that should be learnable
                (necessary for calculating their gradients).
            N_horizon: The number of steps in the MPC horizon.
                The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal node N).
            T_horizon: The length (meaning time) of the MPC horizon.
                One step in the horizon will equal T_horizon/N_horizon simulation time.
            Fmax: The maximum force that can be applied to the cart.
            discount_factor: The discount factor for the cost.
            n_batch: The batch size the MPC should be able to process
                (currently this is static).
            exact_hess_dyn: If False, the contributions of the dynamics will be left out of the Hessian.
            cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
        """
        params = params if params is not None else PARAMS.copy()  # type:ignore

        # non-dimensionalize the time
        if dimensionless:
            dt = T_horizon / N_horizon
            t_scale = np.sqrt(params["l"] / params["g"])[0]
            dt_hat = dt / t_scale
            T_horizon_hat = dt_hat * N_horizon
        else:
            T_horizon_hat = T_horizon

        # non-dimensionalize the maximum force
        if dimensionless:
            Fmax_hat = Fmax / (params["M"] * params["g"])
        else:
            Fmax_hat = Fmax

        # scale the cost
        if dimensionless:
            M = params["M"]
            g = params["g"]
            l = params["l"]

            t_scale = np.sqrt(l/g)
            x_scale = l
            theta_scale = 1
            F_scale = M*g

            params["L11"] = params["L11"] * (x_scale)**2
            params["L22"] = params["L22"] * (x_scale/t_scale)**2
            params["L33"] = params["L33"] * (theta_scale)**2
            params["L44"] = params["L44"] * (theta_scale/t_scale)**2
            params["L55"] = params["L55"] * (F_scale)**2

        ocp = export_parametric_ocp(
            nominal_param=params,
            cost_type=cost_type,
            exact_hess_dyn=exact_hess_dyn,
            name="cartpole_dimensionless",
            learnable_param=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon_hat,
            Fmax=Fmax_hat,
        )

        super().__init__(
            ocp=ocp,
            discount_factor=discount_factor,
            n_batch_max=n_batch,
        )


def f_expl_expr(model: AcadosModel) -> ca.SX:
    p = find_param_in_p_or_p_global(["M", "m", "g", "l"], model)

    M = p["M"]
    m = p["m"]
    g = p["g"]
    l = p["l"]  # noqa E741

    theta = model.x[1]
    dx = model.x[2]
    dtheta = model.x[3]

    F = model.u[0]

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = ca.vertcat(
        dx,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F)
        / denominator,
        (
            -m * l * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (M + m) * g * sin_theta
        )
        / (l * denominator),
    )

    # non-dimensionalize the dynamics
    if dimensionless:
        s = model.x
        a = model.u

        x_hat       = ca.SX.sym('x_hat')
        theta_hat   = ca.SX.sym('theta_hat')
        dx_hat      = ca.SX.sym('dx_hat')
        dtheta_hat  = ca.SX.sym('dtheta_hat')
        s_hat = ca.vertcat(x_hat, dx_hat, theta_hat, dtheta_hat)

        F_hat = ca.SX.sym('F_hat')
        a_hat = ca.vertcat(F_hat)

        t_scale = np.sqrt(l/g)
        x_scale = l
        theta_scale = 1
        F_scale = M*g

        s_scale = [x_scale, theta_scale, x_scale/t_scale, theta_scale/t_scale]
        a_scale = [F_scale]
        ds_scale = [x_scale/t_scale, theta_scale/t_scale, x_scale/t_scale**2, theta_scale/t_scale**2]

        # RHS states
        for k in range(len(s_scale)):
            f_expl = ca.substitute(f_expl, s[k], s_scale[k]*s_hat[k])

        # RHS actions
        for k in range(len(a_scale)):
            f_expl = ca.substitute(f_expl, a[k], a_scale[k]*a_hat[k])

        # LHS (derivatives)
        for k in range(len(ds_scale)):
            f_expl[k] /= ds_scale[k]

        model.x = s_hat
        model.u = a_hat
        model.name += '_dimensionless'

    return f_expl  # type:ignore


def disc_dyn_expr(model: AcadosModel, dt: float) -> ca.SX:
    f_expl = f_expl_expr(model)

    x = model.x
    u = model.u

    # discrete dynamics via RK4
    p = ca.vertcat(*find_param_in_p_or_p_global(["M", "m", "g", "l"], model).values())

    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)  # type:ignore
    k3 = ode(x + dt / 2 * k2, u, p)  # type:ignore
    k4 = ode(x + dt * k3, u, p)  # type:ignore

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # type:ignore


def cost_matrix_casadi(model: AcadosModel) -> ca.SX:
    L = ca.diag(
        ca.vertcat(
            *find_param_in_p_or_p_global(
                ["L11", "L22", "L33", "L44", "L55"], model
            ).values()
        )
    )
    L_offdiag = find_param_in_p_or_p_global(["Lloweroffdiag"], model)["Lloweroffdiag"]

    assign_lower_triangular(L, L_offdiag)

    return L @ L.T


def cost_matrix_numpy(nominal_params: dict[str, np.ndarray]) -> np.ndarray:
    L = np.diag([nominal_params[f"L{i}{i}"].item() for i in range(1, 6)])
    L[np.tril_indices_from(L, -1)] = nominal_params["Lloweroffdiag"]
    return L @ L.T


def yref_numpy(nominal_params: dict[str, np.ndarray]) -> np.ndarray:
    return np.array(
        [nominal_params[f"xref{i}"] for i in range(1, 5)] + [nominal_params["uref"]]
    ).squeeze()


def yref_casadi(model: AcadosModel) -> ca.SX:
    return ca.vertcat(
        *find_param_in_p_or_p_global(
            [f"xref{i}" for i in range(1, 5)] + ["uref"], model
        ).values()
    )  # type:ignore


def c_casadi(model: AcadosModel) -> ca.SX:
    return ca.vertcat(
        *find_param_in_p_or_p_global([f"c{i}" for i in range(1, 6)], model).values()
    )  # type:ignore


def cost_expr_ext_cost(model: AcadosModel) -> ca.SX:
    x = model.x
    u = model.u

    W = cost_matrix_casadi(model)
    c = c_casadi(model)

    z = ca.vertcat(x, u)

    return 0.5 * z.T @ W @ z + c.T @ z


def cost_expr_ext_cost_e(model: AcadosModel) -> ca.SX:
    x = model.x
    W = cost_matrix_casadi(model)
    c = c_casadi(model)

    Q = W[:4, :4]
    c = c[:4]

    return 0.5 * x.T @ Q @ x + c.T @ x  # type:ignore


def export_parametric_ocp(
    nominal_param: dict[str, np.ndarray],
    cost_type: str = "NONLINEAR_LS",
    exact_hess_dyn: bool = True,
    name: str = "pendulum_on_cart",
    learnable_param: list[str] | None = None,
    Fmax: float = 80.0,
    N_horizon: int = 50,
    tf: float = 2.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = tf
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon  # already dimensionless

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nx = 4
    ocp.dims.nu = 1

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)  # type:ignore
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)  # type:ignore
    ocp.model.x_labels = ["x", "theta", "dx", "dtheta"]
    ocp.model.u_labels = ["F"]

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param,
        learnable_param=learnable_param if learnable_param is not None else [],
        ocp=ocp,
    )

    ocp.model.disc_dyn_expr = disc_dyn_expr(model=ocp.model, dt=dt)  # type:ignore

    ######## Cost ########
    if cost_type == "EXTERNAL":
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model)  # type:ignore

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model)  # type:ignore

        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.exact_hess_cost = True
        ocp.solver_options.exact_hess_constr = True
    elif cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.cost.W = cost_matrix_casadi(ocp.model)
        ocp.cost.yref = yref_casadi(ocp.model)
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)

        ocp.cost.W_e = ocp.cost.W[:4, :4]
        ocp.cost.yref_e = ocp.cost.yref[:4]
        ocp.model.cost_y_expr_e = ocp.model.x

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    ocp.constraints.lbu = np.array([-Fmax])  # already dimensionless
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    # scale the constraint on the cart position
    x_min = -2.4  # [m]
    if dimensionless:
        x_min /= nominal_param["l"][0]
    ocp.constraints.lbx = np.array([x_min])
    ocp.constraints.ubx = -ocp.constraints.lbx
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx_e = np.array([x_min])
    ocp.constraints.ubx_e = -ocp.constraints.lbx_e
    ocp.constraints.idxbx_e = np.array([0])

    # scale the slack penalty on the cart position
    slack_penalty = 1e3
    if dimensionless:
        slack_penalty *= nominal_param["l"][0] ** 2
    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([slack_penalty])
    ocp.cost.zu = ocp.cost.zl = np.array([0.0])

    ocp.constraints.idxsbx_e = np.array([0])
    ocp.cost.Zu_e = ocp.cost.Zl_e = np.array([slack_penalty])
    ocp.cost.zu_e = ocp.cost.zl_e = np.array([0.0])

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"

    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_tol = 1e-7
    ocp.solver_options.with_batch_functionality = True

    #####################################################

    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


if __name__ == "__main__":
    mpc = PendulumOnCartMpcDimensionless(learnable_params = ["xref2"])