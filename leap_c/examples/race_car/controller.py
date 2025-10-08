from leap_c.ocp.acados.controller import AcadosController
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch
from leap_c.examples.race_car.config import CarParams, create_acados_params, get_default_car_params
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.examples.race_car.acados_ocp import export_ocp
from acados_template import AcadosOcp
from leap_c.examples.utils.casadi import integrate_erk4
import torch
import numpy as np
import casadi as ca


class RaceCarController(AcadosController):
    """TODO: write me"""

    def __init__(self, car_params: CarParams = get_default_car_params(), dimensionless: bool = False):
        self.car_params = car_params
        self.dimensionless = dimensionless

        self.sref_N = 46.5 * (car_params.l.item() if not dimensionless else 1.0)  # terminal progress reference
        self.N = car_params.N.item()
        self.s_offset = np.linspace(start=0.0, stop=self.sref_N, num=self.N+1)  # will be added to current s

        param_manager = AcadosParameterManager(
            parameters=create_acados_params(car_params),
            N_horizon=car_params.N.item(),
        )

        self.ocp = export_parametric_ocp(
            car_params=car_params, dimensionless=dimensionless, param_manager=param_manager)

        diff_mpc_kwargs = {}

        diff_mpc = AcadosDiffMpcTorch(
            ocp=self.ocp, **diff_mpc_kwargs, export_directory=self.ocp.code_export_directory
        )

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)


    def forward(
        self, obs: torch.Tensor, param: torch.Tensor, ctx: AcadosDiffMpcCtx | None = None
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor]:
        batch_size = obs.shape[0]

        s_current = np.array([obs[i, 0].cpu().numpy() for i in range(batch_size)])
        sref = s_current + self.s_offset

        # s_current = obs[:, 0:1]  # shape: (batch_size, 1)

        # horizon_steps = torch.linspace(0, 1, self.N + 1, device=obs.device)
        # sref = s_current.unsqueeze(1) + horizon_steps.view(1, -1, 1) * self.sref_N

        # p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
        #     batch_size=obs.shape[0]
        # )

        # NOTE: In case we want to pass the data of exogenous influences to the controller,
        # we can do it here
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            sref=sref.reshape(batch_size, -1, 1),
        )

        # set the numerical values for yref (different for each of the batched solvers)
        # for i in range(batch_size):
        #     for k in range(1, self.N):
        #         yref = np.array([sref[i, k].detach().cpu().numpy().item(), 0, 0, 0, 0, 0, 0, 0])
        #         self.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[i].set(k, "yref", yref)
        #     yref_N = np.array([sref[i, self.N+1].detach().cpu().numpy().item(), 0, 0, 0, 0, 0])
        #     self.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[i].set(self.N, "yref", yref_N)

        ctx, u0, x, u, value = self.diff_mpc(
            obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )
        return ctx, u0


def export_parametric_ocp(car_params: CarParams, dimensionless: bool, param_manager: AcadosParameterManager) -> AcadosOcp:
    """Modifies the OCP for use within leap-c."""
    ocp = export_ocp(car_params=car_params, dimensionless=dimensionless)
    
    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )

    # define discrete dynamics (required for leap-c)
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=ocp.model.f_expl_expr,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=ocp.solver_options.tf / ocp.solver_options.N_horizon,
    )
    ocp.solver_options.integrator_type = "DISCRETE"

    # adapt the cost to use AcadosParameters
    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    qe_diag_sqrt = param_manager.get("qe_diag_sqrt")

    # set the NONLINEAR_LS cost
    W_0_sqrt = ca.diag(r_diag_sqrt)
    ocp.cost.W_0 = W_0_sqrt @ W_0_sqrt.T

    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    ocp.cost.W = W_sqrt @ W_sqrt.T
    
    W_e_sqrt = ca.diag(qe_diag_sqrt)
    ocp.cost.W_e = W_e_sqrt @ W_e_sqrt.T

    sref = param_manager.get("sref")
    yref = ca.vertcat(sref, np.zeros(7))
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref[:-2]  # control inputs are not included

    # translate the cost to EXTERNAL, but keep the GN hessian approximation
    ocp.cost.cost_type_0 = ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = translate_nls_cost_to_external_cost(ocp.model.cost_y_expr_0, ocp.cost.yref_0, ocp.cost.W_0)
    ocp.model.cost_expr_ext_cost = translate_nls_cost_to_external_cost(ocp.model.cost_y_expr, ocp.cost.yref, ocp.cost.W)
    ocp.model.cost_expr_ext_cost_e = translate_nls_cost_to_external_cost(ocp.model.cost_y_expr_e, ocp.cost.yref_e, ocp.cost.W_e)
    ocp.model.cost_expr_ext_cost_custom_hess_0 = get_gn_hessian_expression_from_nls_cost(
        ocp.model.cost_y_expr_0, 
        ocp.cost.yref_0, 
        ocp.cost.W_0,
        ocp.model.x,
        ocp.model.u)
    ocp.model.cost_expr_ext_cost_custom_hess = get_gn_hessian_expression_from_nls_cost(
        ocp.model.cost_y_expr,
        ocp.cost.yref,
        ocp.cost.W,
        ocp.model.x,
        ocp.model.u)
    ocp.model.cost_expr_ext_cost_custom_hess_e = get_gn_hessian_expression_from_nls_cost(
        ocp.model.cost_y_expr_e,
        ocp.cost.yref_e,
        ocp.cost.W_e,
        ocp.model.x,
        [])
    
    # remove the NONLINEAR_LS fields
    ocp.cost.W_0 = ocp.cost.W = ocp.cost.W_e = np.zeros((0,0))
    ocp.model.cost_y_expr_0 = ocp.model.cost_y_expr = ocp.model.cost_y_expr_e = []
    ocp.cost.yref_0 = ocp.cost.yref = ocp.cost.yref_e = np.zeros((0,))

    # set some more options to avoid warnings
    ocp.solver_options.qp_solver_cond_ric_alg = 0
    ocp.solver_options.qp_solver_ric_alg = 1
    # ocp.solver_options.qp_solver_warm_start = 1
    # ocp.solver_options.nlp_solver_warm_start_first_qp = True
    # ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = True

    param_manager.assign_to_ocp(ocp)

    return ocp


def translate_nls_cost_to_external_cost(y_expr, yref, W):
    # https://github.com/acados/acados/blob/4b1eb73f1f84ecf4fb19feaf5a672a58a2c3c35a/interfaces/acados_template/acados_template/acados_ocp.py#L1925
    res = y_expr - yref
    return 0.5 * (res.T @ W @ res)


def get_gn_hessian_expression_from_nls_cost(y_expr, yref, W, x, u):
    # https://github.com/acados/acados/blob/4b1eb73f1f84ecf4fb19feaf5a672a58a2c3c35a/interfaces/acados_template/acados_template/acados_ocp.py#L1930
    res = y_expr - yref
    ux = ca.vertcat(u, x)
    inner_jac = ca.jacobian(res, ux)
    gn_hess = inner_jac.T @ W @ inner_jac
    return gn_hess


if __name__ == "__main__":
    from leap_c.examples.race_car.config import get_default_car_params
    RaceCarController(car_params=get_default_car_params(), dimensionless=False)