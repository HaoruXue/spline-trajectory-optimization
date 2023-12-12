from importlib_resources import files, as_file
from matplotlib import pyplot as plt
import numpy as np
from time import time

from spline_traj_optm.models.trajectory import Trajectory, BSplineTrajectory
import spline_traj_optm.examples.race_track.monza

def get_trajectory_array(traj_resource):
    traj_file = traj_resource
    if type(traj_file) is str:
        return np.loadtxt(traj_file, dtype=np.float64, delimiter=',',skiprows=1, usecols=(0, 1))
    else:
        with as_file(traj_file) as f:
            return np.loadtxt(f, dtype=np.float64, delimiter=',',skiprows=1, usecols=(0, 1))
        
    
def get_trajectory_array_with_bank(traj_resource):
    traj_file = traj_resource
    if type(traj_file) is str:
        return np.loadtxt(traj_file, dtype=np.float64, delimiter=',',skiprows=1, usecols=(0, 1,3))
    else:
        with as_file(traj_file) as f:
            return np.loadtxt(f, dtype=np.float64, delimiter=',',skiprows=1, usecols=(0, 1,3))

def get_bspline(traj_resource, s=0.8):
    traj_arr = get_trajectory_array(traj_resource)
    return BSplineTrajectory(traj_arr[:, :2], s, 5)

def test_bsplines():
    start = time()
    left_traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_LEFT_BOUNDARY_enu.csv"))
    left_traj_discrete = left_traj_spline.sample_along(3.0)
    
    right_traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv"))
    right_traj_discrete = right_traj_spline.sample_along(3.0)

    traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_UNOPTIMIZED_LINE_enu.csv"), s=20.0)
    traj_discrete = traj_spline.sample_along(3.0)
    duration = time() - start
    print(f"Interpolation of Monza Circuit took: {duration / 3} sec")

    plt.figure()
    plt.plot(traj_discrete[:, Trajectory.X], traj_discrete[:, Trajectory.Y], '-b', label='center line')
    plt.plot(left_traj_discrete[:, Trajectory.X], left_traj_discrete[:, Trajectory.Y], color='orange', label='left edge')
    plt.plot(right_traj_discrete[:, Trajectory.X], right_traj_discrete[:, Trajectory.Y], '-g', label='right edge')

    # for i in range(2, len(traj_spline._spl_x.c)-3):
    #     t = traj_spline._spl_x.t[i+3]
    #     x_t, y_t = traj_spline.eval(t, 0)
    #     xl_t, yl_t = left_traj_spline.eval(t, 0)
    #     # x = [traj_spline._spl_x.c[i], x_t]
    #     # y = [traj_spline._spl_y.c[i], y_t]
    #     x = [xl_t, x_t]
    #     y = [yl_t, y_t]
    #     plt.plot(x, y, '-r')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('square')
    plt.title("Monza Circuit")
    plt.xlabel('easting (m)')
    plt.ylabel('northing (m)')
    plt.plot(traj_spline._spl_x.tck[1], traj_spline._spl_y.tck[1], '-or', label='control points')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(np.clip(traj_discrete[:, Trajectory.CURVATURE], 0.0, 100.0))
    plt.title("Monza Circuit (Turn Radius)")
    plt.ylabel("m")
    plt.show()
    
if __name__ == "__main__":
    test_bsplines()
