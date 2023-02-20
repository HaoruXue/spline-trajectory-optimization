from importlib_resources import files
import numpy as np
import pickle
import matplotlib.pyplot as plt

from spline_traj_optm.tests.test_trajectory import get_bspline
from spline_traj_optm.optimization.optimizer import TrajectoryOptimizer
from spline_traj_optm.models.trajectory import Trajectory, BSplineTrajectory
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator
import spline_traj_optm.samples.race_track.monza

def test_optimizer():
    traj_spline = get_bspline(files(spline_traj_optm.samples.race_track.monza).joinpath(
        "MONZA_UNOPTIMIZED_LINE_enu.csv"))
    traj_discrete = traj_spline.sample_along(1.0)
    left_traj_spline = get_bspline(files(spline_traj_optm.samples.race_track.monza).joinpath("MONZA_LEFT_BOUNDARY_enu.csv"))
    left_traj_discrete = left_traj_spline.sample_along(3.0)
    right_traj_spline = get_bspline(files(spline_traj_optm.samples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv"))
    right_traj_discrete = right_traj_spline.sample_along(3.0)
    center_traj_spline = traj_spline.copy()

    acc_speed_lookup = np.array([[0.0, 10.0], [50.0, 7.0], [100.0, 0.5]])
    dcc_speed_lookup = np.array([[0.0, -13.0], [50.0, -15.0], [100.0, -20.0]])
    vp = VehicleParams(acc_speed_lookup, dcc_speed_lookup,
                       10.0, -20.0, 15.0, -15.0, 100.0, 30.0)
    v = Vehicle(vp)

    optm = TrajectoryOptimizer(left_traj_spline, right_traj_spline, center_traj_spline, v)
    # result = optm.sim.run_simulation(traj_discrete, enable_vis=False)
    # traj_out = result.trajectory
    # with open("/tmp/test_optm_result.pkl", "wb") as f:
    #     pickle.dump(traj_out, f)

    with open("/tmp/test_optm_result.pkl", "rb") as f:
        traj_out = pickle.load(f)
    #cost = optm.max_velocity_cost(np.array(traj_spline.get_control_point(10)), 10, traj_spline, result.trajectory)
    #print(cost)
    constraint = optm.boundary_constraint(10, traj_spline, traj_out)
    z0 = np.array(traj_spline.get_control_point(10))
    test_constraint_l0 = constraint.A @ z0 > constraint.lb
    test_constraint_u0 = constraint.A @ z0 < constraint.ub
    z1 = z0 * 1.5
    test_constraint_l1 = constraint.A @ z1 > constraint.lb
    test_constraint_u1 = constraint.A @ z1 < constraint.ub

    traj_optm = optm.run(traj_spline, traj_out)
    
    traj_optm_d = traj_optm.sample_along(3.0)

    plt.figure()
    plt.plot(traj_discrete[:, Trajectory.X], traj_discrete[:, Trajectory.Y], '-b')
    plt.plot(traj_spline._spl_x.tck[1], traj_spline._spl_y.tck[1], 'ob')
    plt.plot(traj_optm_d[:, Trajectory.X], traj_optm_d[:, Trajectory.Y], '-g')
    plt.plot(traj_optm._spl_x.tck[1], traj_optm._spl_y.tck[1], 'og')
    plt.plot(left_traj_discrete[:, Trajectory.X], left_traj_discrete[:, Trajectory.Y], '-r')
    plt.plot(right_traj_discrete[:, Trajectory.X], right_traj_discrete[:, Trajectory.Y], '-r')

    plt.gca().set_aspect('equal')
    plt.title("Monza Circuit")
    plt.xlabel('easting (m)')
    plt.ylabel('northing (m)')
    plt.show()

if __name__ == "__main__":
    test_optimizer()
