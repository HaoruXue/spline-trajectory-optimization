from importlib_resources import files
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from spline_traj_optm.tests.test_trajectory import get_bspline, get_trajectory_array
from spline_traj_optm.models.trajectory import Trajectory
import spline_traj_optm.models.dynamic_bicycle as dyn
import spline_traj_optm.models.double_track as dt_dyn
import spline_traj_optm.examples.race_track.uh_maui
import spline_traj_optm.examples.race_track.monza
from spline_traj_optm.models.race_track import RaceTrack
import spline_traj_optm.min_time_optm.min_time_optimizer as optm
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator


def test_bicycle_optimizer():
    # interval = 0.5
    # traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.uh_maui).joinpath(
    #     "uh_maui_center.csv"), s=3.0)
    # traj_d = traj_spline.sample_along(interval)
    # race_track = RaceTrack("Maui", get_trajectory_array(files(spline_traj_optm.examples.race_track.uh_maui).joinpath(
    #     "uh_maui_left.csv")), get_trajectory_array(files(spline_traj_optm.examples.race_track.uh_maui).joinpath("uh_maui_right.csv")))
    # race_track.fill_trajectory_boundaries(traj_d)

    # params = {
    #     "N": len(traj_d),
    #     "traj_d": traj_d,
    #     "nu": dyn.nu(),
    #     "nx": dyn.nx(),
    #     "model": {
    #         "lr": 0.5,
    #         "L": 1.0,
    #         "delta_max": 0.314158999998341,
    #         "v_max": 20.0,
    #         "a_lon_max": 5.0,
    #         "a_lon_min": -5.0,
    #         "delta_dot_max": 1.0,
    #         "acc_max": 20.0
    #     },
    #     "dynamics": dyn.dynamics,
    #     "x_l": dyn.x_l,
    #     "x_u": dyn.x_u,
    #     "u_l": dyn.u_l,
    #     "u_u": dyn.u_u,
    #     "verbose": True,
    #     "max_iter": 100,
    #     "tol": 1e-2,
    #     "constr_viol_tol": 1e-3,
    # }

    # ****************************
    # Uncomment to use Monza track
    # ****************************

    interval = 5.0
    traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath(
        "MONZA_UNOPTIMIZED_LINE_enu.csv"), s=3.0)
    traj_d = traj_spline.sample_along(interval)
    race_track = RaceTrack("Monza", get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath(
        "MONZA_LEFT_BOUNDARY_enu.csv")), get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv")))
    race_track.fill_trajectory_boundaries(traj_d)

    params = {
        "N": len(traj_d),
        "traj_d": traj_d,
        "nu": dyn.nu(),
        "nx": dyn.nx(),
        "model": {
            "lr": 1.5,
            "L": 3.0,
            "delta_max": 0.314158999998341,
            "v_max": 80.0,
            "a_lon_max": 20.0,
            "a_lon_min": -20.0,
            "delta_dot_max": 1.0,
            "acc_max": 20.0,
        },
        "dynamics": dyn.dynamics,
        "x_l": dyn.x_l,
        "x_u": dyn.x_u,
        "u_l": dyn.u_l,
        "u_u": dyn.u_u,
        "verbose": True,
        "max_iter": 100,
        "tol": 1e-2,
        "constr_viol_tol": 1e-3,
    }

    X, U, T, opti = optm.set_up_bicycle_problem(params)
    try:
        sol = opti.solve()
    except Exception as e:
        print(e)
    scale_x = ca.DM([10.0, 10.0, 3.14, 0.1, 80.0]).T
    scale_u = ca.DM([20.0, 1.0]).T
    x = opti.debug.value(X) * scale_x + np.hstack(
        [traj_d[:, Trajectory.X:Trajectory.Y+1], np.zeros((len(traj_d), 3))])
    u = opti.debug.value(U) * scale_u
    t = opti.debug.value(T)

    plt.figure()
    plt.plot(x[:, 0], x[:, 1], "-o")
    plt.plot(traj_d[:, Trajectory.LEFT_BOUND_X],
             traj_d[:, Trajectory.LEFT_BOUND_Y])
    plt.plot(traj_d[:, Trajectory.RIGHT_BOUND_X],
             traj_d[:, Trajectory.RIGHT_BOUND_Y])
    plt.gca().set_aspect("equal")
    plt.show()

    plt.figure()
    plt.plot(x[:, 2], "-o", label="Yaw")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x[:, 4], label="Velocity")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, label="Time")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(u[:, 0], label="Lon Acc")
    plt.legend()
    plt.show()


def test_double_track_optimizer():
    interval = 10.0
    traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath(
        "MONZA_UNOPTIMIZED_LINE_enu.csv"), s=3.0)
    traj_d = traj_spline.sample_along(interval)
    race_track = RaceTrack("Monza", get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath(
        "MONZA_LEFT_BOUNDARY_enu.csv")), get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv")))
    race_track.fill_trajectory_boundaries(traj_d)

    acc_speed_lookup = np.array([[0.0, 10.0], [50.0, 7.0], [100.0, 0.5]])
    dcc_speed_lookup = np.array([[0.0, -13.0], [50.0, -15.0], [100.0, -20.0]])
    vp = VehicleParams(acc_speed_lookup, dcc_speed_lookup,
                       10.0, -20.0, 15.0, -15.0, 40.0, 30.0)
    v = Vehicle(vp)

    sim = Simulator(v)
    result = sim.run_simulation(traj_d, False)
    traj_d = result.trajectory

    params = {
        "N": len(traj_d),
        "traj_d": traj_d,
        "model": {
            "kd_f": 0.0,
            "kb_f": 0.7,
            "mass": 1200.0,
            "Jzz": 1260,
            "lf": 1.5,
            "lr": 1.4,
            "twf": 1.6,
            "twr": 1.5,
            "delta_max": 0.4,

            "fr": 0.01,
            "hcog": 0.4,
            "kroll_f": 0.5,

            "cl_f": 2.4,
            "cl_r": 3.0,
            "rho": 1.2041,
            "A": 1.0,
            "cd": 1.4,
            "mu": 1.5,

            "Bf": 9.62,
            "Cf": 2.59,
            "Ef": 1.0,
            "Fz0_f": 3000.0,
            "eps_f": -0.0813,
            "Br": 8.62,
            "Cr": 2.65,
            "Er": 1.0,
            "Fz0_r": 3000.0,
            "eps_r": -0.1263,

            "Pmax": 270000.0,
            "Fd_max": 7100.0,
            "Fb_max": -20000.0,
            "Td": 1.0,
            "Tb": 1.0,
            "Tdelta": 1.0,
        },
        "verbose": True,
        "max_iter": 2000,
        "tol": 1e-2,
        "constr_viol_tol": 1e-3,
        "speed_cap": 80.0,
        "average_track_width": 10.0,
    }

    (X, U, T), (scale_x, scale_u,
                scale_t), opti = optm.set_up_double_track_problem(params)
    try:
        sol = opti.solve()
    except Exception as e:
        print(e)

    x = opti.debug.value(X) * scale_x + np.hstack(
        [traj_d[:, Trajectory.X:Trajectory.Y+1], np.zeros((len(traj_d), 4))])
    u = opti.debug.value(U) * scale_u
    t = opti.debug.value(T) * scale_t

    plt.figure()
    plt.plot(x[:, 0], x[:, 1], "-o")
    plt.plot(traj_d[:, Trajectory.LEFT_BOUND_X],
             traj_d[:, Trajectory.LEFT_BOUND_Y])
    plt.plot(traj_d[:, Trajectory.RIGHT_BOUND_X],
             traj_d[:, Trajectory.RIGHT_BOUND_Y])
    plt.gca().set_aspect("equal")
    plt.show()

    plt.figure()
    plt.plot(x[:, 2], "-o", label="Yaw")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x[:, 5], label="Velocity")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, label="Time")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(u[:, 0], label="Drive Force")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(u[:, 1], label="Brake Force")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(u[:, 2], label="Steering Angle")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(u[:, 3], label="Load Transfer")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_bicycle_optimizer()
    test_double_track_optimizer()
