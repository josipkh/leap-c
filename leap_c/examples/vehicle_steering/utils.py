import numpy as np
from dataclasses import fields, is_dataclass
from scipy.linalg import solve_continuous_are as _solve_continuous_are
# from scipy.linalg import solve_discrete_are as _solve_discrete_are


# TODO: typehint for dataclass
def contains_symbolics(object) -> bool:
    """
    Checks if an object contains CasADi symbolics
    """
    casadi_types = (ca.SX, ca.MX, ca.DM)
    if is_dataclass(object):
        for field in fields(object):
            field_value = getattr(object, field.name)
            if isinstance(field_value, casadi_types):
                return True
        return False
    elif isinstance(object, np.ndarray):
        return False
    elif isinstance(object, ca.SX):
        return True
    else:
        raise TypeError("unknown object type")
    

def cont2discrete_symbolic(A: ca.SX, B: ca.SX, dt: float, method: str) -> tuple[ca.SX, ca.SX]:
    """
    Discretizes a continuous-time LTI system, described by matrices A and B, using the selected method.    
    """

    if method == "bilinear":
        # Identity matrix of appropriate dimensions
        I = ca.SX.eye(A.shape[0])

        # Compute the discretized A and B using the Tustin (Bilinear) method
        A_d = ca.mtimes(ca.inv(I + (A * dt / 2)), I - (A * dt / 2))
        B_d = ca.mtimes(ca.inv(I + (A * dt / 2)), B * dt)
    else:
        raise NotImplementedError(f"Discretization method {method} not implemented")
    return A_d, B_d


# taken from https://github.com/FilippoAiraldi/mpc-reinforcement-learning/blob/main/src/mpcrl/util/control.py
def lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    M: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Computes the solution to the continuous-time LQR problem.

    The LQR problem is to solve the following optimization problem

    .. math::
        \min_{u} \int_{0}^{\infty} x^\top Q x + u^\top R u + 2 x^\top M u

    for the linear time-invariant continuous-time system

    .. math:: \dot{x} = A x + B u.

    The (famous) solution takes the form of a state feedback law

    .. math:: u = -K x

    with a quadratic cost-to-go function

    .. math:: V(x) = x^\top P x.

    The function returns the optimal state feedback matrix :math:`K` and the quadratic
    terminal cost-to-go matrix :math:`P`. If not provided, ``M`` is assumed to be zero.

    Parameters
    ----------
    A : array
        State matrix.
    B : array
        Control input matrix.
    Q : array
        State weighting matrix.
    R : array
        Control input weighting matrix.
    M : array, optional
        Mixed state-input weighting matrix, by default ``None``.

    Returns
    -------
    tuple of two arrays
        Returns the optimal state feedback matrix :math:`K` and the quadratic terminal
        cost-to-go matrix :math:`P`.
    """
    P = _solve_continuous_are(A, B, Q, R, s=M)
    rhs = B.T.dot(P) if M is None else B.T.dot(P) + M.T
    K = np.linalg.solve(R, rhs)
    return K, P


# def dlqr(
#     A: npt.NDArray[np.floating],
#     B: npt.NDArray[np.floating],
#     Q: npt.NDArray[np.floating],
#     R: npt.NDArray[np.floating],
#     M: Optional[npt.NDArray[np.floating]] = None,
# ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
#     r"""Computes the solution to the discrete-time LQR problem.

#     The LQR problem is to solve the following optimization problem

#     .. math::
#         \min_{u} \sum_{t=0}^{\infty} x_t^\top Q x_t + u_t^\top R u_t + 2 x_t^\top M u_t

#     for the linear time-invariant discrete-time system

#     .. math:: x_{t+1} = A x_t + B u_t.

#     The (famous) solution takes the form of a state feedback law

#     .. math:: u_t = -K x_t

#     with a quadratic cost-to-go function

#     .. math:: V(x_t) = x_t^\top P x_t.

#     The function returns the optimal state feedback matrix :math:`K` and the quadratic
#     terminal cost-to-go matrix :math:`P`. If not provided, ``M`` is assumed to be zero.

#     Parameters
#     ----------
#     A : array
#         State matrix.
#     B : array
#         Control input matrix.
#     Q : array
#         State weighting matrix.
#     R : array
#         Control input weighting matrix.
#     M : array, optional
#         Mixed state-input weighting matrix, by default ``None``.

#     Returns
#     -------
#     tuple of two arrays
#         Returns the optimal state feedback matrix :math:`K` and the quadratic terminal
#         cost-to-go matrix :math:`P`.
#     """
#     P = _solve_discrete_are(A, B, Q, R, s=M)
#     rhs = B.T.dot(P).dot(A) if M is None else B.T.dot(P).dot(A) + M.T
#     K = np.linalg.solve(B.T.dot(P).dot(B) + R, rhs)
#     return K, P

def compute_curvature(p_xy: np.ndarray) -> np.ndarray:
    """
    :param p_xy: array of size (n,2) representing Cartesian 2D points
    :return: curvature and path length
    """
    assert p_xy.shape[1] == 2, "Input array must have shape (n,2)"
    assert p_xy.shape[0] > 2, "Input array must have at least 3 points"

    # first derivatives
    dx = np.gradient(p_xy[:, 0])
    dy = np.gradient(p_xy[:, 1])

    # second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # calculation of curvature from the typical formula
    curvature = (dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
    # path_length = np.cumsum(np.sqrt(dx ** 2 + dy ** 2))

    return curvature


def get_double_lane_change_data(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the curvature of a double lane change path at the given x coordinates
    :param X: array of global longitudinal position coordinates
    :return: arrays of road curvature, global lateral position and global heading
    """
    shape = 2.4
    dx1 = 25
    dx2 = 21.95
    dy1 = 4.05
    dy2 = 5.7
    Xs1 = 27.19
    Xs2 = 56.46

    z1 = shape/dx1*(X - Xs1) - shape/2
    z2 = shape/dx2*(X - Xs2) - shape/2

    # from eq. (20) in https://ieeexplore.ieee.org/document/10308482
    # NOTE: expressions for Y and psi are switched in the paper
    Y = dy1/2*(1+np.tanh(z1)) - dy2/2*(1+np.tanh(z2))
    psi = np.arctan(dy1 * (1 / np.cosh(z1))**2 * (1.2 / dx1) - dy2 * (1 / np.cosh(z2))**2 * (1.2 / dx2))

    XY = np.vstack((X,Y)).T
    curvature = compute_curvature(XY)
    return curvature, Y, psi


# def state_log_f2c(
#         state_log: np.ndarray,
#         reference_type: str = "double_lane_change",
#         ts: float = 0.05
#     ) -> np.ndarray:
#     """
#     Convert the state log from the Frenet CF to Cartesian CF
#     :param state_log: array of shape (n, m) where n is the number of states and m is the number of time steps
#     :return: array of shape (n, m)
#     """
#     raise NotImplementedError("This function is not implemented yet")


if __name__ == "__main__":
    vx = 60 / 3.6
    ts = 0.05
    # X = vx * np.arange(0, 10, ts)
    X = np.linspace(0,120,1000)

    curvature, Y, psi = get_double_lane_change_data(X)
    print(f'max: {np.max(curvature)}')  # should be around +0.006797
    print(f'min: {np.min(curvature)}')  # should be around -0.003533

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(curvature)
    plt.xlabel('k')
    plt.ylabel('curvature [1/m]')

    # compare with Fig. 5 in https://www.inderscienceonline.com/doi/abs/10.1504/IJVAS.2005.008237
    plt.figure(2)
    plt.plot(X,Y)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    plt.show(block=False)
    print("Press ENTER to close the plot")
    input()
    plt.close()