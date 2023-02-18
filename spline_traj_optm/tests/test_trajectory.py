from importlib_resources import files, as_file
from matplotlib import pyplot as plt
import numpy as np
from time import time

from spline_traj_optm.models.trajectory import Trajectory, BSplineTrajectory
import spline_traj_optm.samples.race_track.monza

def get_bspline(traj_resource):
    traj_file = traj_resource
    with as_file(traj_file) as f:
        traj_arr = np.loadtxt(f, dtype=np.float64, delimiter=',',skiprows=1)
    return BSplineTrajectory(traj_arr[:, :2], 1.0, 5)

def test_bsplines():
    start = time()
    left_traj_spline = get_bspline(files(spline_traj_optm.samples.race_track.monza).joinpath("MONZA_LEFT_BOUNDARY_enu.csv"))
    left_traj_discrete = left_traj_spline.sample_along(3.0)
    
    right_traj_spline = get_bspline(files(spline_traj_optm.samples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv"))
    right_traj_discrete = right_traj_spline.sample_along(3.0)

    traj_spline = get_bspline(files(spline_traj_optm.samples.race_track.monza).joinpath("MONZA_UNOPTIMIZED_LINE_enu.csv"))
    traj_discrete = traj_spline.sample_along(3.0)
    duration = time() - start
    print(f"Interpolating Monza Circuit: {duration / 3}")

    plt.figure()
    plt.plot(traj_discrete[:, Trajectory.X], traj_discrete[:, Trajectory.Y], '-b')
    plt.plot(left_traj_discrete[:, Trajectory.X], left_traj_discrete[:, Trajectory.Y], '-r')
    plt.plot(right_traj_discrete[:, Trajectory.X], right_traj_discrete[:, Trajectory.Y], '-g')
    plt.gca().set_aspect('equal')
    plt.title("Monza Circuit")
    plt.xlabel('easting (m)')
    plt.ylabel('northing (m)')
    plt.show()

    plt.figure()
    plt.plot(left_traj_discrete[:, Trajectory.YAW])
    plt.title("Monza Circuit (Yaw)")
    plt.ylabel("rad")
    plt.show()

test_bsplines()
    