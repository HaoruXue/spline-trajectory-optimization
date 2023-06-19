from importlib_resources import files
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import os
import yaml

from spline_traj_optm.tests.test_trajectory import get_bspline, get_trajectory_array
from spline_traj_optm.models.trajectory import Trajectory, save_ttl
import spline_traj_optm.models.double_track as dt_dyn
from spline_traj_optm.models.race_track import RaceTrack
import spline_traj_optm.min_time_optm.min_time_optimizer as optm
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator

def main():
    param_file = "traj_opt_double_track.yaml"
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"{param_file} does not exist.")
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)
    
    interval = params["interval"]
    traj_spline = get_bspline(params["centerline"], s=1.0)
    traj_d = traj_spline.sample_along(interval)
    race_track = RaceTrack("Test track", get_trajectory_array(params["left_boundary"]), get_trajectory_array(params["right_boundary"]), s=1.0, interval=1.0)
    race_track.fill_trajectory_boundaries(traj_d)

    acc_speed_lookup = np.array([[0.0, 10.0], [50.0, 7.0], [100.0, 0.5]])
    dcc_speed_lookup = np.array([[0.0, -13.0], [50.0, -15.0], [100.0, -20.0]])
    vp = VehicleParams(acc_speed_lookup, dcc_speed_lookup,
                       10.0, -20.0, 15.0, -15.0, 40.0, 30.0)
    v = Vehicle(vp)

    sim = Simulator(v)
    result = sim.run_simulation(traj_d, False)
    traj_d = result.trajectory
    params["N"] = len(traj_d)
    params["traj_d"] = traj_d

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

    opt_traj_d = traj_d.copy()
    opt_traj_d[:, 0:2] = x[:, 0:2]
    opt_traj_d[:, Trajectory.YAW] = x[:, 2]
    opt_traj_d[:, Trajectory.SPEED] = x[:, 5]
    race_track.fill_trajectory_boundaries(opt_traj_d)
    save_ttl(params["output"], opt_traj_d)
    ca.DM(x).to_file("x_optm.txt", "txt")
    ca.DM(u).to_file("u_optm.txt", "txt")
    ca.DM(t).to_file("t_optm.txt", "txt")

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