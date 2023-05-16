import casadi as ca

from spline_traj_optm.models.trajectory import Trajectory
import spline_traj_optm.models.dynamic_bicycle as dyn


def global_to_frenet(p, p0, yaw):
    cos_theta = ca.cos(-yaw)
    sin_theta = ca.sin(-yaw)
    R = ca.DM(ca.vertcat(ca.horzcat(cos_theta, -sin_theta),
                         ca.horzcat(sin_theta, cos_theta)))
    return R @ (p - p0)


def min_time_cost(T):
    return ca.sum1(T)


def align_yaw(yaw1, yaw2):
    k = ca.fabs(yaw2-yaw1)+ca.pi
    l = k - ca.fmod(ca.fabs(yaw2-yaw1)+ca.pi, 2 * ca.pi)
    return yaw1 + l * ca.sign(yaw2 - yaw1)


def hermite_simpson(model, dynamics, x1, x2, u, dt):
    temp = ca.MX(x2)
    temp[0, 2] = align_yaw(temp[0, 2], x1[0, 2])
    f1 = dynamics(model, x1, u).T
    f2 = dynamics(model, temp, u).T
    xm = 0.5 * (x1 + temp) + (dt / 8.0) * (f1 - f2)
    fm = dynamics(model, xm, u).T
    return x1 + (dt / 6.0) * (f1 + 4 * fm + f2) - temp


def set_up_problem(params):
    N = params["N"]
    traj_d = params["traj_d"]
    nu = params["nu"]
    nx = params["nx"]
    model = params["model"]
    dynamics = params["dynamics"]

    opti = ca.Opti("nlp")
    X = opti.variable(N, nx)
    U = opti.variable(N, nu)
    T = opti.variable(N)

    P0 = ca.DM(traj_d[:, Trajectory.X:Trajectory.Y+1])
    Yaws = ca.DM(traj_d[:, Trajectory.YAW])
    BoundL = ca.DM(
        traj_d[:, Trajectory.LEFT_BOUND_X:Trajectory.LEFT_BOUND_Y+1])
    BoundR = ca.DM(
        traj_d[:, Trajectory.RIGHT_BOUND_X:Trajectory.RIGHT_BOUND_Y+1])

    # cost
    cost_function = min_time_cost(T)
    opti.minimize(cost_function)

    # primal bounds
    x_l = params["x_l"](model)
    x_u = params["x_u"](model)
    u_l = params["u_l"](model)
    u_u = params["u_u"](model)

    for i in range(N):
        # dynamics constraint
        opti.subject_to(hermite_simpson(model, dynamics,
                        X[i-1, :], X[i, :], U[i-1, :], T[i]) == 0)

        # boundary constraint in frenet frame
        p = X[i, 0:2]
        p0 = P0[i, :]
        pf = global_to_frenet(p.T, p0.T, Yaws[i])
        dl = ca.norm_2(BoundL[i, :] - p0)
        dr = ca.norm_2(BoundR[i, :] - p0) * -1.0
        # opti.subject_to(opti.bounded(-5e-2, pf[0], 5e-2))
        opti.subject_to(pf[0] == 0)
        opti.subject_to(opti.bounded(dr, pf[1], dl))

        # traction constraint
        lat_acc = dyn.lat_acc(model, X[i, :], U[i, :])
        lon_acc = dyn.lon_acc(model, X[i, :], U[i, :])
        acc = ca.power(lat_acc, 2) + ca.power(lon_acc, 2)
        opti.subject_to(acc <= model["acc_max"] ** 2)

        # initial condition
        opti.set_initial(X[i, :], ca.DM([p0[0], p0[1], Yaws[i], 0.0, 1.0]))
        opti.set_initial(U[i, :], 0.0)
        opti.set_initial(T[i], 1.0)

        # primal bounds
        opti.subject_to(x_l <= X[i, :])
        opti.subject_to(X[i, :] <= x_u)
        opti.subject_to(u_l <= U[i, :])
        opti.subject_to(U[i, :] <= u_u)
        opti.subject_to(0.0 <= T[i])

    print_lvl = 5 if params["verbose"] else 0
    p_opts = {"expand": True}
    s_opts = {"max_iter": params["max_iter"], "tol": params["tol"],
              "constr_viol_tol": params["constr_viol_tol"], "print_level": print_lvl}
    opti.solver('ipopt', p_opts, s_opts)

    return X, U, T, opti
