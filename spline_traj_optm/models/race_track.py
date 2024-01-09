import numpy as np
from shapely.geometry import LinearRing
import casadi as ca

from spline_traj_optm.models.trajectory import BSplineTrajectory, Trajectory
from spline_traj_optm.utils.utils import align_yaw

class RaceTrack:
    def __init__(self, name: str, left: np.ndarray, right: np.ndarray, centerline: np.array, s=10.0, interval=2.0) -> None:
        """Create a race track representation from raw boundaries.

        Args:
            name (str): Race track name.
            left (np.ndarray): N * 2 Left boundries in meter. N >= 3.
            right (np.ndarray): N * 2 Right boundries in meter. N >= 3.
            centerline (np.ndarray): N * 2 Centerline in meter. N >= 3.
            s (float, optional): Smoothing value for `scipy.interpolate.splprep`. Defaults to 10.0.
            interval (float, optional): Sampling interval for the discretized version of the spline boundaries. Defaults to 2.0.
        """
        assert left.shape[0] >= 3 and right.shape[0] >= 3
        assert left.shape[1] >= 2 and right.shape[1] >= 2

        self.left_s = BSplineTrajectory(left[:, :2], s, 3, False)
        self.right_s = BSplineTrajectory(right[:, :2], s, 3,False )
        self.center_s = BSplineTrajectory(centerline[:, :3], s, 3, centerline.shape[1]==3)
        

        self.left_d = self.left_s.sample_along(interval)
        self.right_d = self.right_s.sample_along(interval)
        self.center_d = self.center_s.sample_along(interval)

        self.left_r = LinearRing(self.left_d[:, :2])
        self.right_r = LinearRing(self.right_d[:, :2])
        self.center_r = LinearRing(self.center_d[:, :2])

        self.name = name

        self.fill_trajectory_boundaries(self.center_d)

        #add bank angle


        # create interpolants
        interpolants = self.center_d.copy()

        # append the first 4 points
        interpolants = np.vstack([interpolants, interpolants[:4, :]])
        # make sure the distance measure is continuious
        interpolants[-4:, Trajectory.DIST_TO_SF_BWD] += interpolants[-4, Trajectory.DIST_TO_SF_FWD]


        # prepend the last 3 points
        interpolants = np.vstack([interpolants[-7:-4, :], interpolants])
        # make sure the distance measure is continuious
        interpolants[0:3, Trajectory.DIST_TO_SF_BWD] -= interpolants[-4, Trajectory.DIST_TO_SF_FWD]
        
        dist_to_left = np.linalg.norm(interpolants[:, Trajectory.X:Trajectory.Y+1] - interpolants[:, Trajectory.LEFT_BOUND_X:Trajectory.LEFT_BOUND_Y+1], axis=1)
        dist_to_right = np.linalg.norm(interpolants[:, Trajectory.X:Trajectory.Y+1] - interpolants[:, Trajectory.RIGHT_BOUND_X:Trajectory.RIGHT_BOUND_Y+1], axis=1)
        self.interpolants = interpolants
        self.abscissa = self.center_d[:, Trajectory.DIST_TO_SF_BWD]
        abscissa = interpolants[:, Trajectory.DIST_TO_SF_BWD]
        left_intp = ca.interpolant("left_intp_impl", "bspline", [abscissa], dist_to_left.tolist())
        right_intp = ca.interpolant("right_intp_impl", "bspline", [abscissa], (-1.0 * dist_to_right).tolist())
        x_intp = ca.interpolant("x_intp_impl", "bspline", [abscissa], interpolants[:, Trajectory.X].tolist())
        y_intp = ca.interpolant("y_intp_impl", "bspline", [abscissa], interpolants[:, Trajectory.Y].tolist())

        bank_intp = ca.interpolant("bank_intp_impl", "bspline", [abscissa], interpolants[:, Trajectory.BANK].tolist())


        # build the interpolation functions
        s = ca.MX.sym("s", 1, 1)
        s_mod = ca.fmod(s, self.center_s.get_length())
        s_mod_sym = ca.MX.sym("s_mod", 1, 1)
        d2x, dx = ca.hessian(x_intp(s_mod_sym), s_mod_sym)
        d2y, dy = ca.hessian(y_intp(s_mod_sym), s_mod_sym)
        d2x_func = ca.Function("d2x", [s_mod_sym], [d2x])
        dx_func = ca.Function("dx", [s_mod_sym], [dx])
        d2y_func = ca.Function("d2y", [s_mod_sym], [d2y])
        dy_func = ca.Function("dy", [s_mod_sym], [dy])
        yaw = ca.atan2(dy_func(s_mod), dx_func(s_mod))
        curvature = (dx_func(s_mod) * d2y_func(s_mod) - dy_func(s_mod) * d2x_func(s_mod)) / ca.sqrt((dx_func(s_mod) ** 2 + dy_func(s_mod) ** 2) ** 3)
        self.yaw_intp = ca.Function("yaw_intp", [s], [yaw])
        self.curvature_intp = ca.Function("curvature_intp", [s], [curvature])
        self.left_intp = ca.Function("left_intp", [s], [left_intp(s_mod)])
        self.right_intp = ca.Function("right_intp", [s], [right_intp(s_mod)])
        self.x_intp = ca.Function("x_intp", [s], [x_intp(s_mod)])
        self.y_intp = ca.Function("y_intp", [s], [y_intp(s_mod)])
        self.bank_intp = ca.Function("bank_intp", [s], [bank_intp(s_mod)])


        # build fast interpolations
        curvatures = self.curvature_intp(abscissa)
        fast_curvature_intp = ca.interpolant("fast_curvature_intp_impl", "linear", [abscissa], curvatures.full().squeeze().tolist())
        self.fast_curvature_intp = ca.Function("fast_curvature_intp", [s], [fast_curvature_intp(s_mod)])

    def frenet_to_global(self, s, t, xi):
        x0 = self.x_intp(s)
        y0 = self.y_intp(s)
        yaw0 = self.yaw_intp(s)
        cos_theta = ca.cos(yaw0)
        sin_theta = ca.sin(yaw0)
        dx = -sin_theta * t
        dy = cos_theta * t
        phi = align_yaw(yaw0 + xi, 0.0)
        return ca.horzcat(x0 + dx, y0 + dy, phi)

    def fill_trajectory_boundaries(self, traj: Trajectory):
        """Fills the boundary properties of a trajectory.

        Args:
            traj (Trajectory): The trajectory to be modified in-place.
        """
        traj.fill_bounds(self.left_r, self.right_r, max_dist=100.0)
