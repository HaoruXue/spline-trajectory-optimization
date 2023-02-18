import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline
from scipy.integrate import quad


class Trajectory:
    X = 0
    Y = 1
    Z = 2
    YAW = 3
    SPEED = 4
    CURVATURE = 5
    DIST_TO_SF_BWD = 6
    DIST_TO_SF_FWD = 7
    REGION = 8
    LEFT_BOUND_X = 9
    LEFT_BOUND_Y = 10
    RIGHT_BOUND_X = 11
    RIGHT_BOUND_Y = 12
    BANK = 13
    LON_ACC = 14
    LAT_ACC = 15
    TIME = 16
    IDX = 17
    ITERATION_FLAG = 18

    def __init__(self, num_point: int) -> None:
        self.points = np.zeros((num_point, 19), dtype=np.float64)
        self.points[:, Trajectory.IDX] = np.arange(0, len(self.points), 1)
        self.points[:, Trajectory.ITERATION_FLAG] = -1

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, val):
        self.points[key] = val

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        for pt in self.points:
            yield pt

    def copy(self):
        new_traj = Trajectory(len(self.points))
        new_traj.points = self.points.copy()
        return new_traj

    def inc(self, idx: int):
        if idx + 1 == len(self.points):
            return 0
        else:
            return idx + 1

    def dec(self, idx: int):
        if idx - 1 < 0:
            return len(self.points) - 1
        else:
            return idx - 1

    def fill_time(self):
        # Check for zero speeds
        for pt in self.points:
            if pt[Trajectory.SPEED] == 0.0 and pt[Trajectory.LON_ACC == 0.0]:
                raise Exception(
                    "Zero speed and lon_acc encoutered. Cannot fill time.")

        self.points[0, Trajectory.TIME] = 0.0
        for i in range(len(self.points)):
            this, next = i, i + 1
            if next == len(self.points):
                next = 0
            # x = 1/2 * (v_0 + v) * t
            x = self.distance(self.points[this], self.points[next])
            self.points[next, Trajectory.TIME] = x / (
                0.5
                * (
                    self.points[this, Trajectory.SPEED]
                    + self.points[next, Trajectory.SPEED]
                )
            )
            self.points[next, Trajectory.TIME] += self.points[this,
                                                              Trajectory.TIME]

    def distance(self, pt1, pt2):
        return np.sqrt(
            (pt1[Trajectory.X] - pt2[Trajectory.X]) ** 2
            + (pt1[Trajectory.Y] - pt2[Trajectory.Y]) ** 2
        )

class BSplineTrajectory:
    def __init__(self, coordinates: np.ndarray, s: float, k: int):
        assert coordinates.shape[0] >= 3 and coordinates.shape[1] == 2 and len(
            coordinates.shape) == 2, "coordinates should be N * 2"
        # close the loop
        coordinates_close_loop = np.vstack([coordinates, coordinates[0, np.newaxis, :]])
        tck, u = interpolate.splprep(
            [coordinates_close_loop[:, 0], coordinates_close_loop[:, 1]], s=s, per=True, k=k)
        self._spl_x = BSpline(tck[0], tck[1][0], tck[2])
        self._spl_y = BSpline(tck[0], tck[1][1], tck[2])
        self._length = self.__get_section_length(0.0, 1.0)

    def __integrate(self, t: float):
        return np.sqrt(interpolate.splev(t, self._spl_x, der=1) ** 2 + interpolate.splev(t, self._spl_y, der=1) ** 2)

    def __get_section_length(self, t_min: float, t_max: float):
        length, err = quad(self.__integrate, t_min, t_max, limit=200)
        return length

    def __get_yaw(self, t):
        return np.arctan2(interpolate.splev(t, self._spl_y, der=1), interpolate.splev(t, self._spl_x, der=1))

    def __get_turn_radius(self, t):
        dx = interpolate.splev(t, self._spl_x, der=1)
        dy = interpolate.splev(t, self._spl_y, der=1)
        d2x = interpolate.splev(t, self._spl_x, der=2)
        d2y = interpolate.splev(t, self._spl_y, der=2)
        curvature = (dx * d2y - dy * d2x) / np.sqrt((dx ** 2 + dy ** 2) ** 3)
        return 1.0 / np.abs(curvature) + 1e-4

    def get_length(self):
        return self._length

    def sample_along(self, interval: float) -> Trajectory:
        total_length = self.get_length()
        num_sample = int(total_length // interval)
        interval = total_length / num_sample
        traj = Trajectory(num_sample)

        ts = np.linspace(0.0, 1.0, num_sample, endpoint=False)

        traj[:, Trajectory.X] = interpolate.splev(ts, self._spl_x)
        traj[:, Trajectory.Y] = interpolate.splev(ts, self._spl_y)
        traj[:, Trajectory.YAW] = self.__get_yaw(ts)
        traj[:, Trajectory.CURVATURE] = self.__get_turn_radius(ts)

        for i in range(len(traj)):
            if i == 0:
                continue
            traj[i, Trajectory.DIST_TO_SF_BWD] = traj[i-1, Trajectory.DIST_TO_SF_BWD] + self.__get_section_length(ts[i], ts[i-1])
        traj[:, Trajectory.DIST_TO_SF_FWD] = self._length - traj[:, Trajectory.DIST_TO_SF_BWD]
   
        return traj