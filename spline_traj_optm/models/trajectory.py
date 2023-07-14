import numpy as np
import copy
from scipy import interpolate
from scipy.interpolate import BSpline
from scipy.integrate import quad
from shapely.geometry import Point, LinearRing, GeometryCollection, LineString, MultiPoint
import pickle

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

    def fill_bounds(self, left_poly, right_poly, max_dist=100.0):
        def find_intersect(wp:np.ndarray, poly: LinearRing, norm, max_dist):
            yaw_tan, x, y = wp[Trajectory.YAW], wp[Trajectory.X], wp[Trajectory.Y]
            traj_pt = Point(x, y)
            yaw_norm = yaw_tan + norm
            max_norm = (x + max_dist * np.cos(yaw_norm), y + max_dist * np.sin(yaw_norm))
            min_norm = (x + max_dist * np.cos(yaw_norm + np.pi), y + max_dist * np.sin(yaw_norm + np.pi))
            line_norm = LineString((min_norm, max_norm))
            some_intersects = line_norm.intersection(poly)
            this_some_distance = max_dist
            this_some_intersection = None
            if type(some_intersects) in (GeometryCollection, MultiPoint):
                distances = []
                intersections = []
                for intersection in list(some_intersects.geoms):
                    if type(intersection) is Point:
                        distances.append(
                            traj_pt.distance(intersection))
                        intersections.append(intersection)
                    else:
                        print(
                            f"Issue with boundary at index {wp[Trajectory.IDX]}: intersection with {poly.name} is not a Point but {type(intersection)}.")
                if len(distances) == 0:
                    print(
                        f"Issue with boundary at index {wp[Trajectory.IDX]}: no Point intersection found with Geometry of name {poly.name}.")
                else:
                    min_dist_idx = np.argmin(np.array(distances))
                    this_some_distance = distances[min_dist_idx]
                    this_some_intersection = intersections[min_dist_idx]
            elif type(some_intersects) is Point:
                this_some_distance = traj_pt.distance(
                    some_intersects)
                this_some_intersection = some_intersects
            else:
                # line_vis = np.array(line_norm.coords)
                # poly_vis = np.array(poly.coords)
                # fig, ax = plt.subplots()
                # ax.set_aspect('equal')
                # ax.plot(line_vis[:, 0], line_vis[:, 1])
                # ax.plot(poly_vis[:, 0], poly_vis[:, 1])
                # fig.show()
                # raise Exception("No intersection with boundary found.")
                return 0.0, traj_pt

            return this_some_distance, this_some_intersection

        def calc_left_right_bounds(wp):
            _, left_bound = find_intersect(wp, left_poly, np.pi / 2.0, max_dist)
            _, right_bound = find_intersect(wp, right_poly, -np.pi / 2.0, max_dist)
            wp[Trajectory.LEFT_BOUND_X] = left_bound.x
            wp[Trajectory.LEFT_BOUND_Y] = left_bound.y
            wp[Trajectory.RIGHT_BOUND_X] = right_bound.x
            wp[Trajectory.RIGHT_BOUND_Y] = right_bound.y

        np.apply_along_axis(calc_left_right_bounds, 1, self.points)

    def fill_distance(self):
        dists = np.zeros(len(self.points), dtype=np.float64)
        for i in range(len(self.points)):
            j = i + 1 if i < len(self.points) - 1 else 0
            dists[i] = self.distance(self.points[i, :], self.points[j, :])

        self.points[0, Trajectory.DIST_TO_SF_BWD] = 0.0
        self.points[1:, Trajectory.DIST_TO_SF_BWD] = dists[:-1]
        self.points[:, Trajectory.DIST_TO_SF_BWD] = np.cumsum(self.points[:, Trajectory.DIST_TO_SF_BWD])

        track_length = np.sum(dists)
        self.points[:, Trajectory.DIST_TO_SF_FWD] = track_length - self.points[:, Trajectory.DIST_TO_SF_BWD]

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
            # self.points[next, Trajectory.TIME] += self.points[this,
            #                                                   Trajectory.TIME]

    def distance(self, pt1, pt2):
        return np.linalg.norm(pt1[Trajectory.X:Trajectory.Y+1] - pt2[Trajectory.X:Trajectory.Y+1])

    def ts(self):
        return np.linspace(0.0, 1.0, self.__len__(), endpoint=False)

    def save(f, traj):
        np.savetxt(f, traj.points, delimiter=',')

    def load(f):
        arr = np.loadtxt(f, np.float64, delimiter=',')
        traj = Trajectory(len(arr))
        traj.points = arr
        return traj

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

    def __integrate_length(self, t: float):
        return np.sqrt(interpolate.splev(t, self._spl_x, der=1) ** 2 + interpolate.splev(t, self._spl_y, der=1) ** 2)

    def __get_section_length(self, t_min: float, t_max: float):
        length, err = quad(self.__integrate_length, t_min, t_max, limit=200)
        return length

    def eval_sectional_length(self, ts):
        return self.__get_section_length(ts[0], ts[1])

    def eval_dx_sectional_length(self, ts):
        def to_integrate(t):
            return interpolate.splev(t, self._spl_x, der=2) / np.sqrt(interpolate.splev(t, self._spl_x, der=1) ** 2 + interpolate.splev(t, self._spl_y, der=1) ** 2)
        length, err = quad(to_integrate, ts[0], ts[1], limit=200)
        return length

    def eval_dy_sectional_length(self, ts):
        def to_integrate(t):
            return interpolate.splev(t, self._spl_y, der=2) / np.sqrt(interpolate.splev(t, self._spl_x, der=1) ** 2 + interpolate.splev(t, self._spl_y, der=1) ** 2)
        length, err = quad(to_integrate, ts[0], ts[1], limit=200)
        return length

    def eval(self, t, der=0):
        return interpolate.splev(t, self._spl_x, der=der), interpolate.splev(t, self._spl_y, der=der)

    def __get_yaw(self, t):
        return np.arctan2(interpolate.splev(t, self._spl_y, der=1), interpolate.splev(t, self._spl_x, der=1))

    def __get_turn_radius(self, t):
        dx = interpolate.splev(t, self._spl_x, der=1)
        dy = interpolate.splev(t, self._spl_y, der=1)
        d2x = interpolate.splev(t, self._spl_x, der=2)
        d2y = interpolate.splev(t, self._spl_y, der=2)
        curvature = (dx * d2y - dy * d2x) / np.sqrt((dx ** 2 + dy ** 2) ** 3)
        return 1.0 / (np.abs(curvature))

    def get_length(self):
        return self._length

    def eval_yaw(self, t):
        return self.__get_yaw(t)

    def sample_along(self, interval: float=None, ts=None) -> Trajectory:
        if interval is not None:
            total_length = self.get_length()
            num_sample = int(total_length // interval)
            interval = total_length / num_sample
            traj = Trajectory(num_sample)
            ts = np.linspace(0.0, 1.0, num_sample, endpoint=False)
        else:
            traj = Trajectory(len(ts))

        traj[:, Trajectory.X] = interpolate.splev(ts, self._spl_x)
        traj[:, Trajectory.Y] = interpolate.splev(ts, self._spl_y)
        traj[:, Trajectory.YAW] = self.__get_yaw(ts)
        traj[:, Trajectory.CURVATURE] = self.__get_turn_radius(ts)

        for i in range(len(traj)):
            if i == 0:
                continue
            traj[i, Trajectory.DIST_TO_SF_BWD] = traj[i-1, Trajectory.DIST_TO_SF_BWD] + self.__get_section_length(ts[i-1], ts[i])
        traj[:, Trajectory.DIST_TO_SF_FWD] = self._length - traj[:, Trajectory.DIST_TO_SF_BWD]
   
        return traj

    def copy(self):
        return copy.deepcopy(self)

    def set_control_point(self, idx, coord):
        self._spl_x.c[idx] = coord[0]
        self._spl_y.c[idx] = coord[1]

    def get_control_point(self, idx):
        return self._spl_x.c[idx], self._spl_y.c[idx]

    def save(f, traj):
        with open(f, "wb") as output_file:
            pickle.dump(traj, output_file)

    def load(f):
        with open(f, "rb") as input_file:
            return pickle.load(input_file)

def save_ttl(ttl_path: str, trajectory: Trajectory):
    with open(ttl_path, "w") as f:
        header = ",".join(
            [
                str(15),
                str(len(trajectory)),
                str(trajectory[0, Trajectory.DIST_TO_SF_FWD]),
            ]
        )
        # if trajectory.origin is not None:
        #     header += "," + ",".join([str(x) for x in trajectory.origin])
        f.write(header)
        f.write("\n")

        def save_row(row: np.ndarray):
            vals = [
                str(row[Trajectory.X]),
                str(row[Trajectory.Y]),
                str(row[Trajectory.Z]),
                str(row[Trajectory.YAW]),
                str(row[Trajectory.SPEED]),
                str(row[Trajectory.CURVATURE]),
                str(row[Trajectory.DIST_TO_SF_BWD]),
                str(row[Trajectory.DIST_TO_SF_FWD]),
                str(int(row[Trajectory.REGION])),
                str(row[Trajectory.LEFT_BOUND_X]),
                str(row[Trajectory.LEFT_BOUND_Y]),
                str(row[Trajectory.RIGHT_BOUND_X]),
                str(row[Trajectory.RIGHT_BOUND_Y]),
                str(row[Trajectory.BANK]),
                str(row[Trajectory.LON_ACC]),
                str(row[Trajectory.LAT_ACC]),
                str(row[Trajectory.TIME]),
            ]
            f.writelines([','.join(vals) + '\n'])
        np.apply_along_axis(save_row, 1, trajectory.points)