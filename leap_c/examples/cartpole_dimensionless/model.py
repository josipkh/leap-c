from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
import casadi as ca
import numpy as np
from leap_c.examples.cartpole_dimensionless.config import CartPoleParams


def export_cartpole_model(cartpole_params: CartPoleParams) -> AcadosModel:

    # constants
    M = cartpole_params.M.item()
    m = cartpole_params.m.item()
    g = cartpole_params.g.item()
    l = cartpole_params.l.item()
    mu_f = cartpole_params.mu_f.item()

    # set up states & actions
    x       = ca.SX.sym('x')
    theta   = ca.SX.sym('theta')
    dx      = ca.SX.sym('dx')
    dtheta  = ca.SX.sym('dtheta')
    s = ca.vertcat(x, theta, dx, dtheta)

    F = ca.SX.sym('F')
    a = ca.vertcat(F)

    x_dot = ca.SX.sym('x_dot')
    theta_dot = ca.SX.sym('theta_dot')
    v_dot = ca.SX.sym('v_dot')
    dtheta_dot = ca.SX.sym('dtheta_dot')
    s_dot = ca.vertcat(x_dot, theta_dot, v_dot, dtheta_dot)

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    
    # (Akhil)
    # equations from: https://github.com/MPC-Based-Reinforcement-Learning/Safe-RL/blob/experimental/Project/Environments/cartpole.py
    # check also: https://github.com/MPC-Based-Reinforcement-Learning/Safe-RL/blob/akhil/Project/Environments/cartpole.py
    # NOTE: positive angle is counterclockwise, differs from the paper https://ieeexplore.ieee.org/document/10178119
    ddx = (
        - 2 * (m*l) * (dtheta**2) * sin_theta
        + 3 * m * g * sin_theta * cos_theta
        + 4 * F
        - 4 * mu_f * dx
        ) / (4 * (m+M) - 3 * m * cos_theta**2)
    ddtheta = (
        - 3 * (m*l) * (dtheta**2) * sin_theta * cos_theta
        + 6 * (m+M) * g * sin_theta
        + 6 * (F - mu_f * dx) * cos_theta
        ) / (
        + 4 * l * (m+M)
        - 3 * (m*l) * cos_theta**2
        )

    # model from eq. (23)-(24) in https://coneural.org/florian/papers/05_cart_pole.pdf
    # NOTE: positive angle is clockwise
    # ddtheta_num = g * sin_theta + cos_theta * ((-F - m * l * dtheta*dtheta * sin_theta) / (m_cart + m))
    # ddtheta_den = l * (4/3 - (m * cos_theta * cos_theta) / (m_cart + m))
    # ddtheta = ddtheta_num / ddtheta_den
    # ddx_num = F + m * l * (dtheta * dtheta * sin_theta - ddtheta * cos_theta)
    # ddx_den = M + m
    # ddx = ddx_num / ddx_den

    # (acados, leap-c)
    # model from eq. (11) in https://arxiv.org/pdf/1910.13753
    # NOTE: positive angle is counterclockwise
    # denominator = M + m - m*cos_theta*cos_theta
    # ddx = (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F ) / denominator
    # ddtheta = (-m * l * cos_theta * sin_theta * dtheta * dtheta + F * cos_theta + (M + m) * g * sin_theta) / (l * denominator)

    rhs = ca.vertcat(dx, dtheta, ddx, ddtheta)

    model = AcadosModel()
    model.x = s
    model.u = a
    model.f_expl_expr = rhs
    model.f_impl_expr = s_dot - rhs
    model.xdot = s_dot
    model.name = 'cartpole'
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', r'$\dot{x}$ [m/s]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$ [Nm]']
    model.t_label = '$t$ [s]'
    return model


def export_acados_integrator(cartpole_params: CartPoleParams) -> AcadosSimSolver:
    sim = AcadosSim()
    sim.model = export_cartpole_model(cartpole_params=cartpole_params)
    sim.solver_options.T = cartpole_params.dt.item()
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    print("Setting up acados integrator...")
    return AcadosSimSolver(sim, verbose=False)


def export_dimensionless_cartpole_model(cartpole_params: CartPoleParams) -> AcadosModel:

    model = export_cartpole_model(cartpole_params=cartpole_params)
    f_expl_expr = model.f_expl_expr
    s = model.x
    a = model.u

    x_hat       = ca.SX.sym('x_hat')
    theta_hat   = ca.SX.sym('theta_hat')
    dx_hat      = ca.SX.sym('dx_hat')
    dtheta_hat  = ca.SX.sym('dtheta_hat')
    s_hat = ca.vertcat(x_hat, theta_hat, dx_hat, dtheta_hat)

    dx_hat = ca.SX.sym('dx_hat')
    dtheta_hat = ca.SX.sym('dtheta_hat')
    ddx_hat = ca.SX.sym('ddx_hat')
    ddtheta_hat = ca.SX.sym('ddtheta_hat')
    ds_hat = ca.vertcat(dx_hat, dtheta_hat, ddx_hat, ddtheta_hat)

    F_hat = ca.SX.sym('F_hat')
    a_hat = ca.vertcat(F_hat)

    m_c = cartpole_params.M.item()
    g = cartpole_params.g.item()
    l = cartpole_params.l.item()

    t_scale = np.sqrt(l/g)
    x_scale = l
    theta_scale = 1
    F_scale = m_c*g

    s_scale = [x_scale, theta_scale, x_scale/t_scale, theta_scale/t_scale]
    a_scale = [F_scale]
    ds_scale = [x_scale/t_scale, theta_scale/t_scale, x_scale/t_scale**2, theta_scale/t_scale**2]

    for k in range(len(s_scale)):
        f_expl_expr = ca.substitute(f_expl_expr, s[k], s_scale[k]*s_hat[k])

    for k in range(len(a_scale)):
        f_expl_expr = ca.substitute(f_expl_expr, a[k], a_scale[k]*a_hat[k])

    for k in range(len(ds_scale)):
        f_expl_expr[k] /= ds_scale[k]

    # substitution with acados:
    # https://docs.acados.org/python_interface/index.html#acados_template.acados_model.AcadosModel.substitute

    model.x = s_hat
    model.u = a_hat
    model.f_expl_expr = f_expl_expr
    model.f_impl_expr = ds_hat - f_expl_expr
    model.xdot = ds_hat
    model.name += '_dimensionless'

    model.x_labels = [r'$\tilde{x}$ [-]', r'$\tilde{\theta}$ [-]', r'$\dot{\tilde{x}}$ [-]', r'$\dot{\tilde{\theta}}$ [-]']
    model.u_labels = [r'$\tilde{F}$ [-]']
    model.t_label = r'$\tilde{t}$ [-]'

    return model


def simulate_dimensionless_version(cartpole_params: CartPoleParams) -> tuple[AcadosSim, np.array]:

    sim = AcadosSim()
    sim.model = export_dimensionless_cartpole_model(cartpole_params=cartpole_params)

    g = cartpole_params.g.item()
    l = cartpole_params.l.item()
    t_scale = np.sqrt(l/g)
    # x_scale = l

    dt_hat = cartpole_params.dt / t_scale

    ns = sim.model.x.rows()
    nsim = 200

    # set simulation time
    sim.solver_options.T = dt_hat.item()
    # set options
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1

    # create
    acados_integrator = AcadosSimSolver(sim, verbose=False)

    s0 = np.array([0.0, 0.0, np.pi+1, 0.0])
    # a0 = np.array([0.0])
    a0 = 0.1 * cartpole_params.Fmax.item() / (cartpole_params.M.item()*g)  # scaled force

    s_log = np.zeros((nsim+1, ns))
    s_log[0,:] = s0

    for i in range(nsim):
        s_log[i+1,:] = acados_integrator.simulate(x=s_log[i,:], u=a0)

    return sim, s_log


if __name__ == "__main__":
    from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params
    from leap_c.examples.cartpole_dimensionless.utils import get_similar_cartpole_params

    params_ref = get_default_cartpole_params()
    rod_length = 5.0  # [m] 0.1
    params_sim = get_similar_cartpole_params(reference_params=params_ref, rod_length=rod_length)

    # large cartpole
    large_cartpole_sim, large_s_log = simulate_dimensionless_version(cartpole_params=params_ref)
    print("-" * 80)
    print("large cartpole dimensionless model:")
    print(large_cartpole_sim.model.f_expl_expr)    
    # print("-" * 80)
    # print("\n".join(large_cartpole_sim.model.f_expl_expr.str(True).splitlines()))

    # small cartpole
    small_cartpole_sim, small_s_log = simulate_dimensionless_version(cartpole_params=params_sim)
    print("-" * 80)
    print("small cartpole dimensionless model:")
    print(small_cartpole_sim.model.f_expl_expr)
    print("-" * 80)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, sharex=True, constrained_layout=True)
    fig.suptitle('Cartpole states (dimensionless)')
    dt = 0.5 * (small_cartpole_sim.solver_options.T + large_cartpole_sim.solver_options.T)
    t = np.arange(0, small_s_log.shape[0]) * dt
    for k in range(4):
        ax[k].plot(t, small_s_log[:, k], color="b", label='small')
        ax[k].plot(t, large_s_log[:, k], color="r", linestyle="--", label='large')
        ax[k].set_ylabel(small_cartpole_sim.model.x_labels[k])
    ax[-1].set_xlabel(small_cartpole_sim.model.t_label)
    ax[0].legend()

    fig.align_ylabels(ax)
    plt.show(block=False)
    print("Press ENTER to close the plot")
    input()
    plt.close()

    