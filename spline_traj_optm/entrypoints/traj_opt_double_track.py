from importlib_resources import files
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import os
import yaml

from spline_traj_optm.tests.test_trajectory import get_bspline, get_trajectory_array, get_trajectory_array_with_bank
from spline_traj_optm.models.trajectory import Trajectory, save_ttl
import spline_traj_optm.models.double_track as dt_dyn
from spline_traj_optm.models.race_track import RaceTrack
import spline_traj_optm.min_time_optm.min_time_optimizer as optm
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator

def main():
    param_file = 'traj_opt_double_track.yaml'
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"{param_file} does not exist.")
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)
    
    interval = params["interval"]
    race_track = RaceTrack(
        "Test track",
        get_trajectory_array(params["left_boundary"]),
        get_trajectory_array(params["right_boundary"]),
        get_trajectory_array_with_bank(params["centerline"]),
        s=1.0,
        interval=interval
    )
    traj_d = race_track.center_d.copy()
    race_track.fill_trajectory_boundaries(traj_d)

    

    if ("x0" not in params):
        estimates = params["estimates"]
        acc_speed_lookup = np.array(estimates["acc_speed_loopup"])
        dcc_speed_lookup = np.array(estimates["dcc_speed_lookup"])
        vp = VehicleParams(acc_speed_lookup, dcc_speed_lookup,
                        estimates["max_lon_acc_mpss"],
                        estimates["max_lon_dcc_mpss"],
                        estimates["max_left_acc_mpss"],
                        estimates["max_right_acc_mpss"],
                        estimates["max_speed_mps"],
                        estimates["max_jerk_mpsc"])
        v = Vehicle(vp)
        traj_d[:, Trajectory.SPEED] = 45
        traj_d[:, Trajectory.TIME] = .01

        # result = sim.run_simulation(traj_d, False)
        # traj_d = result.trajectory
    else:
        params["x0"] = ca.DM.from_file(params["x0"], "txt")
        params["u0"] = ca.DM.from_file(params["u0"], "txt")
        params["t0"] = ca.DM.from_file(params["t0"], "txt")

    params["N"] = len(traj_d)
    params["traj_d"] = traj_d
    params["race_track"] = race_track

    (X, U, T), (scale_x, scale_u,
                scale_t), opti = optm.set_up_double_track_problem(params)
    try:
        sol = opti.solve()
    except Exception as e:
        print(e)

    x = np.array(opti.debug.value(X)) * np.array(scale_x) + np.hstack(
        [race_track.abscissa[:, np.newaxis], np.zeros((len(traj_d), 5))])
    u = np.array(opti.debug.value(U)) * np.array(scale_u)
    t = np.array(opti.debug.value(T)) * np.array(scale_t)

    print(f"[Optimal lap time: {ca.sum1(t) * scale_t}]")

    ca.DM(traj_d.points[:, :Trajectory.TIME+1]).to_file("ttl_input.txt", "txt")

    opt_traj_d = traj_d.copy()
    global_pose = race_track.frenet_to_global(x[:, 0].T, x[:, 1].T, x[:, 2].T)
    opt_traj_d[:, 0:2] = global_pose[:, 0:2]
    opt_traj_d[:, Trajectory.YAW] = global_pose[:, 2].full().squeeze()
    opt_traj_d[:, Trajectory.SPEED] = x[:, 5]
    race_track.fill_trajectory_boundaries(opt_traj_d)
    opt_traj_d.fill_distance()
    save_ttl(params["output"], opt_traj_d)
    ca.DM(x).to_file("x_optm.txt", "txt")
    ca.DM(u).to_file("u_optm.txt", "txt")
    ca.DM(t).to_file("t_optm.txt", "txt")
    ca.DM(opt_traj_d.points[:, :Trajectory.TIME+1]).to_file("ttl_optm.txt", "txt")

    plt.figure()
    plt.plot(opt_traj_d[:, 0], opt_traj_d[:, 1], "-o")
    plt.plot(traj_d[:, Trajectory.LEFT_BOUND_X],
             traj_d[:, Trajectory.LEFT_BOUND_Y])
    plt.plot(traj_d[:, Trajectory.RIGHT_BOUND_X],
             traj_d[:, Trajectory.RIGHT_BOUND_Y])
    plt.gca().set_aspect("equal")
    plt.show()

    plt.figure()
    plt.plot(opt_traj_d[:, Trajectory.BANK], "-o", label="Bank Angle")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(opt_traj_d[:, Trajectory.YAW], "-o", label="Yaw")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(opt_traj_d[:, Trajectory.YAW], "-o", label="Yaw")
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
    main()