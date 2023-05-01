from importlib_resources import files
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from spline_traj_optm.tests.test_trajectory import get_bspline, get_trajectory_array
from spline_traj_optm.optimization.optimizer import TrajectoryOptimizer
from spline_traj_optm.models.trajectory import Trajectory, BSplineTrajectory
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator
from spline_traj_optm.models.race_track import RaceTrack
import spline_traj_optm.examples.race_track.monza

def test_optimizer():
    traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath(
        "MONZA_UNOPTIMIZED_LINE_enu.csv"), s=100.0)
    traj_discrete = traj_spline.sample_along(3.0)
    left_traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_LEFT_BOUNDARY_enu.csv"))
    left_traj_discrete = left_traj_spline.sample_along(3.0)
    right_traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv"))
    right_traj_discrete = right_traj_spline.sample_along(3.0)
    center_traj_spline = traj_spline.copy()
    BSplineTrajectory.save("test_baseline.pkl", traj_spline)

    race_track = RaceTrack("Monza", get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_LEFT_BOUNDARY_enu.csv")), get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv")))

    acc_speed_lookup = np.array([[0.0, 10.0], [50.0, 7.0], [100.0, 0.5]])
    dcc_speed_lookup = np.array([[0.0, -13.0], [50.0, -15.0], [100.0, -20.0]])
    vp = VehicleParams(acc_speed_lookup, dcc_speed_lookup,
                       10.0, -20.0, 15.0, -15.0, 100.0, 30.0)
    v = Vehicle(vp)

    optm = TrajectoryOptimizer(race_track, center_traj_spline, v)
    if not os.path.exists("/tmp/test_optm_result.pkl"):
        result = optm.sim.run_simulation(traj_discrete, enable_vis=True)
        print(result)
        traj_out = result.trajectory
        with open("/tmp/test_optm_result.pkl", "wb") as f:
            pickle.dump(traj_out, f)
    else:
        with open("/tmp/test_optm_result.pkl", "rb") as f:
            traj_out = pickle.load(f)
    #cost = optm.max_velocity_cost(np.array(traj_spline.get_control_point(10)), 10, traj_spline, result.trajectory)
    #print(cost)
    # constraint = optm.all_boundary_constraint(traj_spline, traj_out)
    # z0 = np.array(traj_spline.get_control_point(10))
    # zx = traj_spline._spl_x.c[2:-3]
    # zy = traj_spline._spl_y.c[2:-3]
    # z0 = np.column_stack([zx, zy]).reshape((-1,))

    # test_constraint_l0 = constraint.A @ z0 > constraint.lb
    # test_constraint_u0 = constraint.A @ z0 < constraint.ub
    # diff = (constraint.A @ z0).reshape((-1, 2)) - traj_discrete[:, :2]
    # z1 = z0 * 1.5
    # test_constraint_l1 = constraint.A @ z1 > constraint.lb
    # test_constraint_u1 = constraint.A @ z1 < constraint.ub

    traj_optm = optm.run_min_curvature_qp(traj_spline, traj_out, visualize=False)
    
    traj_optm_d = traj_optm.sample_along(3.0)
    Trajectory.save("test_optm.csv", traj_optm_d)
    BSplineTrajectory.save("test_optm.pkl", traj_optm)

    # plt.figure()
    # plt.plot(traj_discrete[:, Trajectory.X], traj_discrete[:, Trajectory.Y], '-b', label="unoptimized")
    # plt.plot(traj_spline._spl_x.tck[1], traj_spline._spl_y.tck[1], 'ob', label="unoptimized control pts")
    # plt.plot(traj_optm_d[:, Trajectory.X], traj_optm_d[:, Trajectory.Y], '-g', label="optimized")
    # plt.plot(traj_optm._spl_x.tck[1], traj_optm._spl_y.tck[1], 'og', label="optimized control pts")
    # plt.plot(left_traj_discrete[:, Trajectory.X], left_traj_discrete[:, Trajectory.Y], '-r')
    # plt.plot(right_traj_discrete[:, Trajectory.X], right_traj_discrete[:, Trajectory.Y], '-r')

    # plt.gca().set_aspect('equal')
    # plt.title("Monza Circuit")
    # plt.xlabel('easting (m)')
    # plt.ylabel('northing (m)')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    test_optimizer()
