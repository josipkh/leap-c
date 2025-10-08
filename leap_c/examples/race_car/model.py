"""Defines the dynamics models for use in the OCP and the simulation."""

import os
import casadi as ca
import numpy as np
from leap_c.examples.race_car.track import get_track
from leap_c.examples.race_car.config import CarParams, get_default_car_params
from leap_c.examples.race_car.scaling import get_transformation_matrices
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver

include_terminal_acc_constraint = True  # not set in the original example

def car_model_ocp(car_params: CarParams, dimensionless: bool) -> AcadosModel:
    """Get the prediction model for use in the OCP (control rate formulation)."""
    model = AcadosModel()

    # load track parameters
    [s0, _, _, _, kapparef] = get_track(car_params=car_params)

    # make dimensionless if needed -> automatically done in nondimensionalize_dynamics()
    # if dimensionless:
    #     s0 /= car_params.l.item()
    #     kapparef *= car_params.l.item()

    # copy loop to beginning and end
    length = len(s0)
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    # compute spline interpolations
    kapparef_s = ca.interpolant("kapparef_s", "bspline", [s0], kapparef)

    # race car parameters
    m = car_params.m[0]
    c1 = car_params.lr[0] / car_params.l[0]
    c2 = 1 / car_params.l[0]
    cm1 = car_params.cm1[0]
    cm2 = car_params.cm2[0]
    cr0 = car_params.cr0[0]
    cr2 = car_params.cr2[0]
    cr3 = car_params.cr3[0]

    # states
    s = ca.SX.sym("s")
    n = ca.SX.sym("n")
    alpha = ca.SX.sym("alpha")
    v = ca.SX.sym("v")
    D = ca.SX.sym("D")
    delta = ca.SX.sym("delta")
    x = ca.vertcat(s, n, alpha, v, D, delta)

    # controls
    derD = ca.SX.sym("derD")
    derDelta = ca.SX.sym("derDelta")
    u = ca.vertcat(derD, derDelta)

    # state derivatives
    sdot = ca.SX.sym("sdot")
    ndot = ca.SX.sym("ndot")
    alphadot = ca.SX.sym("alphadot")
    vdot = ca.SX.sym("vdot")
    Ddot = ca.SX.sym("Ddot")
    deltadot = ca.SX.sym("deltadot")
    xdot = ca.vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    # dynamics
    Fxd = (cm1 - cm2 * v) * D - cr2 * v * v - cr0 * ca.tanh(cr3 * v)
    sdot_rhs = (v * ca.cos(alpha + c1 * delta)) / (1 - kapparef_s(s) * n)
    f_expl = ca.vertcat(
        sdot_rhs,
        v * ca.sin(alpha + c1 * delta),
        v * c2 * delta - kapparef_s(s) * sdot_rhs,
        Fxd / m * ca.cos(c1 * delta),
        derD,
        derDelta,
    )

    # constraints expressions
    a_long = Fxd / m
    a_lat = c2 * v * v * delta + Fxd * ca.sin(c1 * delta) / m    
    con_h_expr = ca.vertcat(a_long, a_lat)
    model.con_h_expr = con_h_expr
    if include_terminal_acc_constraint:
        model.con_h_expr_e = con_h_expr

    # add labels for states, controls and time
    model.x_labels = [
        "$s$ [m]",
        "$n$ [m]",
        r"$\alpha$ [rad]",
        "$v$ [m/s]",
        "$D$ [-]",
        r"$\delta$ [rad]",
    ]
    model.u_labels = [
        r"$\dot{D}$ [1/s]",
        r"$\dot{\delta}$ [rad/s]",
    ]
    model.t_label = "$t$ [s]"

    # assign fields to object
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.xdot = xdot    
    model.name = "car_model_ocp"

    # make dimensionless if needed
    if dimensionless:
        model = nondimensionalize_dynamics(car_params=car_params, model=model)

    return model


def car_model_sim(car_params: CarParams, dimensionless: bool) -> AcadosModel:
    """Get the dynamics model for simulation, with the original states and inputs."""
    # get the OCP model
    model_ocp = car_model_ocp(car_params=car_params, dimensionless=dimensionless)

    # remove the control input rate from the model for simulation
    nx = 4
    nu = 2
    model = AcadosModel()
    model.f_impl_expr = model_ocp.f_impl_expr[0:nx]
    model.f_expl_expr = model_ocp.f_expl_expr[0:nx]
    model.x = model_ocp.x[0:nx]
    model.xdot = model_ocp.xdot[0:nx]
    model.u = model_ocp.x[nx:nx+nu]
    model.name = model_ocp.name.replace("ocp", "sim")
    model.x_labels = model_ocp.x_labels[0:nx]
    model.u_labels = model_ocp.x_labels[nx:nx+nu]
    model.t_label = model_ocp.t_label
    return model


def export_acados_integrator(car_params: CarParams, dimensionless: bool) -> AcadosSimSolver:
    """Create and return an acados integrator for simulating the car model."""

    acados_sim = AcadosSim()
    acados_sim.model = car_model_sim(car_params=car_params, dimensionless=dimensionless)
    acados_sim.solver_options.T = car_params.dt.item()
    if dimensionless:
        acados_sim.solver_options.T /= get_transformation_matrices(car_params)[2].item()
    acados_sim.solver_options.integrator_type = "ERK"
    acados_sim.solver_options.num_stages = 4
    acados_sim.solver_options.num_steps = 1
    acados_sim.code_export_directory = os.path.join("codegen", f"sim_{car_params.l.item():.3g}".replace(".", "_"))  # prevent overwriting
    print("Setting up acados integrator...")

    acados_sim_solver = AcadosSimSolver(
        acados_sim=acados_sim, 
        verbose=False,
        json_file=os.path.join("json", f"sim_{car_params.l.item():.3g}".replace(".", "_") + ".json")  # prevent overwriting
    )

    return acados_sim_solver


def nondimensionalize_dynamics(car_params: CarParams, model: AcadosModel) -> AcadosModel:
    """Convert the dynamics to the dimensionless form (without replacing the CasADi variables)."""
    x = model.x
    u = model.u
    f_expl = model.f_expl_expr
    con_h_expr = model.con_h_expr

    Mx, Mu, Mt = get_transformation_matrices(car_params=car_params)
    x_scale = np.diagonal(Mx)
    u_scale = np.diagonal(Mu)
    dx_scale = x_scale / Mt[0]

    # RHS/constraints states
    for k in range(len(x_scale)):
        f_expl = ca.substitute(f_expl, x[k], x_scale[k] * x[k])
        con_h_expr = ca.substitute(con_h_expr, x[k], x_scale[k] * x[k])

    # RHS/constraints actions
    for k in range(len(u_scale)):
        f_expl = ca.substitute(f_expl, u[k], u_scale[k] * u[k])
        con_h_expr = ca.substitute(con_h_expr, u[k], u_scale[k] * u[k])

    # LHS (derivatives)
    for k in range(len(dx_scale)):
        f_expl[k] /= dx_scale[k]

    model.f_impl_expr = model.xdot - f_expl
    model.f_expl_expr = f_expl
    model.con_h_expr = con_h_expr
    if include_terminal_acc_constraint:
        model.con_h_expr_e = con_h_expr
    model.name += "_dimensionless"

    model.x_labels = [
        "$\hat{s}$ [-]",
        "$\hat{n}$ [-]",
        r"$\hat{\alpha}$ [-]",
        "$\hat{v}$ [-]",
        "$\hat{D}$ [-]",
        r"$\hat{\delta}$ [-]",
    ]
    model.u_labels = [
        r"$\dot{\hat{D}}$ [rad/s]",
        r"$\dot{\hat{\delta}}$ [rad/s]",
    ]
    model.t_label = "$\hat{t}$ [-]"

    return model


def test_integrator(car_params: CarParams, model: AcadosModel):
    """Simulate the symbolic dynamics using CasADi."""
    ode = {'x': model.x, 'ode': model.f_expl_expr, 'p': model.u}
    integrator = ca.integrator('integrator', 'rk', ode, 0, car_params.dt.item())
    x = [0.0, 0.0, 0.0, 0.0]
    for _ in range(10):
        res = integrator(x0=x, p=[1.0, 0.0])  # straight driving, full throttle
        x = res['xf'].full().flatten()
    print("="*20)
    print("car length: ", car_params.l.item())
    print("x final: ", x)
    print("x scaled: ", x / car_params.l.item())
    print("="*20)
    return


if __name__ == "__main__":
    from leap_c.examples.race_car.scaling import get_large_car_params, get_transformation_matrices
    import matplotlib.pyplot as plt

    # create the dimensional and dimensionless integrators
    car_params_ref = get_large_car_params()

    integrator_ref = export_acados_integrator(car_params=car_params_ref, dimensionless=False)
    integrator_sim = export_acados_integrator(car_params=car_params_ref, dimensionless=True)
    print("Integrators successfully created.")

    # compare the integrators
    Mx = get_transformation_matrices(car_params_ref)[0]
    Mx = Mx[:4,:4]
    Mx_inv = np.linalg.inv(Mx)

    x0_ref = np.array([0.0, 0.0, 0.0, 0.0])
    x0_sim = x0_ref
    x_log_ref = []
    x_log_sim = []
    u_log = []

    for i in range(100):
        # choose a random input
        u_rand = np.random.uniform(low=[-1.0, -0.4], high=[+1.0, +0.4])
        u_log.append(u_rand)

        # simulate the reference car
        x_next_ref = integrator_ref.simulate(x=x0_ref, u=u_rand)
        x_log_ref.append(x_next_ref)
        x0_ref = x_next_ref

        # simulate the similar car
        x_next_sim = integrator_sim.simulate(x=Mx_inv @ x0_sim, u=u_rand)
        x_log_sim.append(Mx @ x_next_sim)        
        x0_sim = Mx @ x_next_sim

    print("Simulation successful.")
    print("Final state ref: ", x_log_ref[-1])
    print("Final state sim: ", x_log_sim[-1])

    # plot the results
    obs_ref_log = np.array(x_log_ref)
    obs_sim_log = np.array(x_log_sim)
    act_log = np.array(u_log)
    nx = 4
    nu = 2
    fig, ax = plt.subplots(nx + nu, 1, sharex=True)
    labels = ["dimensional", "dimensionless"]
    for i in range(nx):
        ax[i].plot(obs_ref_log[:, i], color="b", label=labels[0])
        ax[i].plot(obs_sim_log[:, i], color="r", linestyle="--", label=labels[1])
        ax[i].grid()
        ax[i].set_ylabel(integrator_ref.acados_sim.model.x_labels[i])
    ax[0].legend()
    for i in range(nu):
        ax[nx+i].step(
            list(range(obs_ref_log.shape[0]+1)),
            np.append([act_log[0, i]], act_log[:, i]),
            where="post",
            color="b",
        )
        ax[nx+i].set_ylabel(integrator_ref.acados_sim.model.u_labels[i])
        ax[nx+i].grid()
    ax[-1].set_xlabel("$k$")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels(ax)
    plt.show(block=False)

    if np.allclose(x_log_ref, x_log_sim):
        print("Results match.")
    else:
        print("Results do not match.")

    print("Press ENTER to close the plot")
    input()
    plt.close()
