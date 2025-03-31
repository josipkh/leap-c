from pathlib import Path
import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from casadi.tools import struct_symSX
from leap_c.examples.vehicle_steering.env import VehicleParams, get_discrete_system
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global
)
from leap_c.mpc import Mpc
from dataclasses import asdict


class VehicleSteeringMPC(Mpc):
    def __init__(
        self,
        nominal_params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 10,
        tf: float = 0.5,
        n_batch: int = 64,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
    ):
        if nominal_params is None:
            vehicle_params_dataclass = VehicleParams()
            vehicle_params_dict = asdict(vehicle_params_dataclass)

            controller_params = {
                "q_diag": np.array([1.0, 1e-3, 1.0, 1e-3]),
                "r_diag": np.array([1]),
                "q_diag_e": np.array([1.0, 1e-3, 1.0, 1e-3]),
                "xref": np.array([0.0, 0.0, 0.0, 0.0]),
                "uref": np.array([0.0, 0.0]),
                "xref_e": np.array([0.0, 0.0, 0.0, 0.0]),
                "dpsi_ref": np.array([0.0]),
            }
            nominal_params = vehicle_params_dict | controller_params

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            nominal_params=nominal_params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=tf,
        )
        configure_ocp_solver(ocp=ocp, exact_hess_dyn=True)

        super().__init__(
            ocp=ocp,
            n_batch=n_batch,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )


def _create_diag_matrix(
    _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    else:
        return np.diag(_q_sqrt)


def discrete_dynamics(
    ocp: AcadosOcp,
) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    A, B = get_discrete_system(dt=dt)
    B_steer = B[:, 0]
    B_ref = B[:, 1]

    return A @ x + B_steer @ u + B_ref @ ocp.model.p["dpsi_ref"]


def cost_expr_ext_cost_0(ocp: AcadosOcp) -> ca.SX:
    u = ocp.model.u

    R_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]
    )

    uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    return 0.5 * ca.mtimes([ca.transpose(u - uref), R_sqrt.T, R_sqrt, u - uref])


def cost_expr_ext_cost(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    Q_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"]
    )
    R_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]
    )

    xref = find_param_in_p_or_p_global(["xref"], ocp.model)["xref"]
    uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    return 0.5 * (
        ca.mtimes([ca.transpose(x - xref), Q_sqrt.T, Q_sqrt, x - xref])
        + ca.mtimes([ca.transpose(u - uref), R_sqrt.T, R_sqrt, u - uref])
    )


def cost_expr_ext_cost_e(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x

    Q_sqrt_e = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag_e"], ocp.model)["q_diag_e"]
    )

    xref_e = find_param_in_p_or_p_global(["xref_e"], ocp.model)["xref_e"]

    return 0.5 * ca.mtimes([ca.transpose(x - xref_e), Q_sqrt_e.T, Q_sqrt_e, x - xref_e])


def export_parametric_ocp(
    nominal_params: dict[str, np.ndarray],
    name: str = "vehicle_steering",
    learnable_params: list[str] | None = None,
    N_horizon: int = 10,
    tf: float = 0.5,
    x0: np.ndarray = np.array([0.1, 0.0, 0.0, 0.0])
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = name

    nx = 4  # [ey, dot_ey, epsi, dot_epsi]
    nu = 1  # delta
    # np = 8  # vehicle parameters + yaw rate reference

    ocp.model.x = ca.SX.sym("x", nx)
    ocp.model.u = ca.SX.sym("u", nu)
    # ocp.model.p = ca.SX.sym("p", ocp.dims.np)

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_params, learnable_param=learnable_params, ocp=ocp
    )

    ocp.model.disc_dyn_expr = discrete_dynamics(ocp=ocp)
    ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp=ocp)
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp=ocp)
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.x0 = x0

    steer_max = np.deg2rad(90)

    # Box constraints on u
    ocp.constraints.lbu = np.array([-steer_max])
    ocp.constraints.ubu = np.array([steer_max])
    ocp.constraints.idxbu = np.array([0])

    # TODO: add slack variables?
    # ocp.constraints.lbx = np.array([-5.0, -5.0, -50.0, -50.0])
    # ocp.constraints.ubx = np.array([5.0, 5.0, 50.0, 50.0])
    # ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    # ocp.constraints.idxsbx = np.array([0, 1, 2, 3])

    # ns = ocp.constraints.idxsbx.size
    # ocp.cost.zl = 100 * np.ones((ns,))
    # ocp.cost.Zl = 0 * np.ones((ns,))
    # ocp.cost.zu = 100 * np.ones((ns,))
    # ocp.cost.Zu = 0 * np.ones((ns,))

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def configure_ocp_solver(ocp: AcadosOcp, exact_hess_dyn: bool):
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True

if __name__ == '__main__':
    learnable_params = ["q_diag", "q_diag_e", "xref", "uref"]

    mpc = VehicleSteeringMPC(learnable_params=learnable_params)

    print('MPC init ok')