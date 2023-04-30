import numpy as np
from shapely.geometry import LinearRing

from spline_traj_optm.models.trajectory import BSplineTrajectory, Trajectory


class RaceTrack:
    def __init__(self, name: str, left: np.ndarray, right: np.ndarray, s=10.0, interval=2.0) -> None:
        """Create a race track representation from raw boundaries.

        Args:
            name (str): Race track name.
            left (np.ndarray): N * 2 Left boundries in meter. N >= 3.
            right (np.ndarray): N * 2 Right boundries in meter. N >= 3.
            s (float, optional): Smoothing value for `scipy.interpolate.splprep`. Defaults to 10.0.
            interval (float, optional): Sampling interval for the discretized version of the spline boundaries. Defaults to 2.0.
        """
        assert left.shape[0] >= 3 and right.shape[0] >= 3
        assert left.shape[1] >= 2 and right.shape[1] >= 2

        self.left_s = BSplineTrajectory(left[:, :2], s, 5)
        self.right_s = BSplineTrajectory(right[:, :2], s, 5)

        self.left_d = self.left_s.sample_along(interval)
        self.right_d = self.right_s.sample_along(interval)

        self.left_r = LinearRing(self.left_d[:, :2])
        self.right_r = LinearRing(self.right_d[:, :2])

        self.name = name

    def fill_trajectory_boundaries(self, traj: Trajectory):
        """Fills the boundary properties of a trajectory.

        Args:
            traj (Trajectory): The trajectory to be modified in-place.
        """
        traj.fill_bounds(self.left_r, self.right_r, max_dist=100.0)
