import casadi as ca

from spline_traj_optm.models.trajectory import Trajectory
import spline_traj_optm.models.dynamic_bicycle as dyn
import spline_traj_optm.models.double_track as dt_dyn
import spline_traj_optm.utils.utils as utils
import spline_traj_optm.utils.integrator as integrator


def min_time_cost(T):
    return ca.sum1(T)


def set_up_bicycle_problem(params):
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
    X_OFFSET = ca.horzcat(P0, ca.DM.zeros(N, nx-2))
    Yaws = ca.DM(traj_d[:, Trajectory.YAW])
    BoundL = ca.DM(
        traj_d[:, Trajectory.LEFT_BOUND_X:Trajectory.LEFT_BOUND_Y+1])
    BoundR = ca.DM(
        traj_d[:, Trajectory.RIGHT_BOUND_X:Trajectory.RIGHT_BOUND_Y+1])
    scale_x = ca.DM([10.0, 10.0, 3.14, 0.1, 80.0]).T
    scale_u = ca.DM([20.0, 1.0]).T
    scale_t = 1.0

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
        xi = X[i-1, :] * scale_x + X_OFFSET[i-1, :]
        xip1 = X[i, :] * scale_x + X_OFFSET[i, :]
        ui = U[i-1, :] * scale_u
        ti = T[i-1]

        opti.subject_to(integrator.rk4(model, dynamics,
                        xi, xip1, ui, ti) == 0)

        # boundary constraint in frenet frame
        p = X[i-1, 0:2] * scale_x[0:2]
        p0 = P0[i-1, :]
        pf = utils.global_to_frenet(p.T, ca.DM.zeros(2, 1), Yaws[i-1])
        dl = ca.norm_2(BoundL[i-1, :] - p0)
        dr = ca.norm_2(BoundR[i-1, :] - p0) * -1.0
        # opti.subject_to(opti.bounded(-5e-2, pf[0], 5e-2))
        opti.subject_to(pf[0] == 0)
        opti.subject_to(opti.bounded(dr, pf[1], dl))

        # traction constraint
        lat_acc = dyn.lat_acc(model, xi, ui)
        lon_acc = dyn.lon_acc(model, xi, ui)
        acc = ca.power(lat_acc, 2) + ca.power(lon_acc, 2)
        opti.subject_to(acc <= model["acc_max"] ** 2)

        # initial condition
        opti.set_initial(xi, ca.DM([0.0, 0.0, Yaws[i-1], 0.0, 1.0]))
        opti.set_initial(ui, 0.0)
        opti.set_initial(ti, 1.0)

        # primal bounds
        opti.subject_to(opti.bounded(x_l, X[i-1, :] * scale_x, x_u))
        opti.subject_to(opti.bounded(u_l, U[i-1, :] * scale_u, u_u))
        opti.subject_to(0.0 <= ti)

    print_lvl = 5 if params["verbose"] else 0
    p_opts = {"expand": True}
    s_opts = {"max_iter": params["max_iter"], "tol": params["tol"],
              "constr_viol_tol": params["constr_viol_tol"], "print_level": print_lvl,
              "nlp_scaling_method": "none"}
    opti.solver('ipopt', p_opts, s_opts)

    return X, U, T, opti


def set_up_double_track_problem(params):
    N = params["N"]
    model = params["model"]
    race_track = params["race_track"]
    traj_d = params["traj_d"]
    nu = dt_dyn.nu()
    nx = dt_dyn.nx()

    opti = ca.Opti("nlp")
    X = opti.variable(N, nx)
    U = opti.variable(N, nu)
    T = opti.variable(N)

    S0 = ca.DM(race_track.abscissa)
    X_OFFSET = ca.horzcat(S0, ca.DM.zeros(N, nx-1))
    # Yaws = ca.DM(traj_d[:, Trajectory.YAW])
    Velocities = ca.DM(traj_d[:, Trajectory.SPEED])
    Times = ca.DM(traj_d[:, Trajectory.TIME])
    BoundL = race_track.left_intp(S0)
    BoundR = race_track.right_intp(S0)
    scale_x = ca.DM([1.0, params["average_track_width"], 1.0, 1.0, 0.5, params["speed_cap"]]).T
    scale_u = ca.DM([model["Fd_max"], abs(model["Fb_max"]),
                    model["delta_max"], model["mass"] * 50.0]).T
    scale_t = 1.0

    # cost
    cost_function = min_time_cost(T)
    opti.minimize(cost_function)

    for i in range(N):
        xi = X[i-1, :] * scale_x + X_OFFSET[i-1, :]
        xip1 = X[i, :] * scale_x + X_OFFSET[i, :]
        ui = U[i-1, :] * scale_u
        uip1 = U[i, :] * scale_u
        ti = T[i-1] * scale_t

        # boundary constraint in frenet frame
        opti.subject_to(xi[0] == X_OFFSET[i-1, 0])
        dr = BoundR[i-1]
        dl = BoundL[i-1]
        margin = model["vehicle_width"] / 2.0 + model["safety_margin"]
        assert dr + margin < dl - margin, f"Track width must be wider than vehicle width plus 2 * safety margin at point {i}."
        opti.subject_to(opti.bounded(dr + margin, xi[1], dl - margin))

        # model constraints
        dt_dyn.add_constraints(model, opti, xi, ui, ti, xip1, uip1, race_track, race_track.curvature_intp(S0[i-1]))

        # time constraint
        opti.subject_to(0.0 <= ti)

        # initial condition
        opti.set_initial(
            X[i-1, :] * scale_x, ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, Velocities[i-1]]))
        u0 = ca.DM([1.0, -1.0, 0.001, 0.0])
        opti.set_initial(ui, u0)
        opti.set_initial(ti, Times[i-1])

    print_lvl = 5 if params["verbose"] else 0
    p_opts = {"expand": True}
    s_opts = {"max_iter": params["max_iter"], "tol": params["tol"],
              "constr_viol_tol": params["constr_viol_tol"], "print_level": print_lvl}
    opti.solver('ipopt', p_opts, s_opts)

    return (X, U, T), (scale_x, scale_u, scale_t), opti
