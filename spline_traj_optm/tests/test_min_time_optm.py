from importlib_resources import files
import matplotlib.pyplot as plt

from spline_traj_optm.tests.test_trajectory import get_bspline, get_trajectory_array
from spline_traj_optm.models.trajectory import Trajectory
import spline_traj_optm.models.dynamic_bicycle as dyn
import spline_traj_optm.examples.race_track.uh_maui
import spline_traj_optm.examples.race_track.monza
from spline_traj_optm.models.race_track import RaceTrack
import spline_traj_optm.min_time_optm.min_time_optimizer as optm


def test_optimizer():
    interval = 1.0
    traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.uh_maui).joinpath(
        "uh_maui_center.csv"), s=3.0)
    traj_d = traj_spline.sample_along(interval)
    race_track = RaceTrack("Maui", get_trajectory_array(files(spline_traj_optm.examples.race_track.uh_maui).joinpath(
        "uh_maui_left.csv")), get_trajectory_array(files(spline_traj_optm.examples.race_track.uh_maui).joinpath("uh_maui_right.csv")))
    race_track.fill_trajectory_boundaries(traj_d)

    params = {
        "N": len(traj_d),
        "traj_d": traj_d,
        "nu": dyn.nu(),
        "nx": dyn.nx(),
        "model": {
            "lr": 0.5,
            "L": 1.0,
            "delta_max": 0.314158999998341,
            "v_max": 20.0,
            "a_lon_max": 5.0,
            "a_lon_min": -5.0,
            "delta_dot_max": 1.0,
            "acc_max": 20.0
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

    # ****************************
    # Uncomment to use Monza track
    # ****************************

    # interval = 10.0
    # traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath(
    #     "MONZA_UNOPTIMIZED_LINE_enu.csv"), s=3.0)
    # traj_d = traj_spline.sample_along(interval)
    # race_track = RaceTrack("Monza", get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_LEFT_BOUNDARY_enu.csv")), get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv")))
    # race_track.fill_trajectory_boundaries(traj_d)

    # params = {
    #     "N": len(traj_d),
    #     "traj_d": traj_d,
    #     "nu": dyn.nu(),
    #     "nx": dyn.nx(),
    #     "model": {
    #         "lr": 1.5,
    #         "L": 3.0,
    #         "delta_max": 0.314158999998341,
    #         "v_max": 80.0,
    #         "a_lon_max": 20.0,
    #         "a_lon_min": -20.0,
    #         "delta_dot_max": 1.0,
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

    X, U, T, opti = optm.set_up_problem(params)
    try:
        sol = opti.solve()
    except Exception as e:
        print(e)
    x = opti.debug.value(X)
    u = opti.debug.value(U)
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


if __name__ == "__main__":
    test_optimizer()
