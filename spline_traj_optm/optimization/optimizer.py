import numpy as np
from tqdm import tqdm

from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize
from spline_traj_optm.models.trajectory import BSplineTrajectory, Trajectory
from spline_traj_optm.models.vehicle import Vehicle, VehicleParams
from spline_traj_optm.simulator.simulator import Simulator

class TrajectoryOptimizer:
    def __init__(self, left_bound: BSplineTrajectory, right_bound: BSplineTrajectory, center_line: BSplineTrajectory, vehicle:Vehicle) -> None:
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.center_line = center_line
        self.vehicle = vehicle
        self.sim = Simulator(self.vehicle)

    def max_velocity_cost(self, z: np.array, idx:int, traj_s: BSplineTrajectory, traj_d: Trajectory):
        # (ti) get the waypoints that can be influenced by this control point
        ts = traj_d.ts()
        k = traj_s._spl_x.k
        t_start = traj_s._spl_x.t[idx]
        t_end = traj_s._spl_x.t[idx+k+1]
        t_mask = (ts >= t_start) & (ts < t_end)
        ti = ts[t_mask]
        N = len(ti)
        
        # (wi) calculate the centerline direction vector (unit length)
        # yaw_centerline = self.center_line.eval_yaw(ti)
        # wi = np.column_stack([np.cos(yaw_centerline), np.sin(yaw_centerline)])

        # (pi) build a projection matrix out of wi
        # wx = wi[:, 0]
        # wy = wi[:, 1]
        # wxwy = wx * wy
        # pi = np.array([[wx ** 2, wxwy], [wxwy, wy ** 2]])
        # pi = np.moveaxis(pi, 2, 0)
        # pi = pi.reshape(2 * N, 2)

        # (vi) calculate the velocity vector at each discretization point, given the new control point z
        traj_copy = traj_s.copy()
        traj_copy.set_control_point(idx, z)
        # yaw_traj = traj_copy.eval_yaw(ti)
        v = traj_d[t_mask, Trajectory.SPEED]
        # vi = np.column_stack([np.cos(yaw_traj), np.sin(yaw_traj)]) * v[:, np.newaxis]

        # (vi_hat) a diagnal version of vi
        # vi_hat = np.zeros((N, N * 2), vi.dtype)
        # i = np.arange(N)
        # vi_hat[i, 2*i] = vi[:, 0]
        # vi_hat[i, 2*i+1] = vi[:, 1]

        # (bi) calculate the basis function of the control point, used to weigh the cost at each point
        # b_x = BSpline.basis_element(traj_copy._spl_x.t[idx:idx+k+2])
        # b_y = BSpline.basis_element(traj_copy._spl_y.t[idx:idx+k+2])
        # bi = np.column_stack([b_x(ti), b_y(ti)])

        # (di) calculate the distances along the trajectory
        tsi = np.zeros((len(ti), 2), ti.dtype)
        tsi[:, 0] = ti
        tsi[:-1, 1] = ti[:-1]
        tsi[-1, 1] = tsi[-1, 0]
        di = np.apply_along_axis(traj_copy.eval_sectional_length, 1, tsi) 

        # finally calculate the cost
        # project velocities onto the centerline direction, weigh each point by the basis function value,
        # take the sum of squre, and normalize by the number of discretization points
        # return np.sum(np.square(di[:, np.newaxis] / vi_hat @ pi * bi)) / N
        return np.sum(di / v)

    def boundary_constraint(self, idx:int, traj_s: BSplineTrajectory, traj_d: Trajectory):
        # (ti) get the waypoints that can be influenced by this control point
        ts = traj_d.ts()
        k = traj_s._spl_x.k
        t_start = traj_s._spl_x.t[idx]
        t_end = traj_s._spl_x.t[idx+k+1]
        t_mask = (ts >= t_start) & (ts < t_end)
        ti = ts[t_mask]
        N = len(ti)

        b_x = BSpline.basis_element(traj_s._spl_x.t[idx:idx+k+2])
        b_y = BSpline.basis_element(traj_s._spl_y.t[idx:idx+k+2])
        bi = np.column_stack([b_x(ti), b_y(ti)])

        old_z = np.array(traj_s.get_control_point(idx))
        non_z = traj_d[t_mask, :2] - bi * old_z[np.newaxis, :]

        # find the boundries
        left_x = self.left_bound._spl_x(ti)
        left_y = self.left_bound._spl_y(ti)
        right_x = self.right_bound._spl_x(ti)
        right_y = self.right_bound._spl_y(ti)
        min_bound = np.empty(2 * N, dtype=left_x.dtype)
        min_bound[0::2] = np.minimum(left_x, right_x) - non_z[:, 0]
        min_bound[1::2] = np.minimum(left_y, right_y) - non_z[:, 1]
        max_bound = np.empty(2 * N, dtype=left_x.dtype)
        max_bound[0::2] = np.maximum(left_x, right_x) - non_z[:, 0]
        max_bound[1::2] = np.maximum(left_y, right_y) - non_z[:, 1]

        # find the A matrix
        A = np.zeros((2 * N, 2), dtype=left_x.dtype)
        A[0::2, 0] = b_x(ti)
        A[1::2, 1] = b_y(ti)
        return LinearConstraint(A, min_bound, max_bound)

    def run(self, traj_in_s: BSplineTrajectory, traj_in_d: Trajectory):
        num_ctrl_pt = len(traj_in_s._spl_x.c)
        ignore_ctrl_pt = traj_in_s._spl_x.k
        ignore_front = ignore_ctrl_pt // 2
        ignore_rear = ignore_ctrl_pt - ignore_front
        traj_out_s = traj_in_s.copy()

        num_success = 0
        for i in tqdm(range(ignore_front, num_ctrl_pt - ignore_rear)):
            z0 = np.array(traj_out_s.get_control_point(i))
            constraint = self.boundary_constraint(i, traj_out_s, traj_in_d)
            new_z = minimize(self.max_velocity_cost, z0, args=(i, traj_out_s, traj_in_d), constraints=(constraint,))
            if new_z.success:
                traj_out_s.set_control_point(i, new_z.x)
                num_success += 1

        print(f"Number of control points successfully updated: {i}")

        return traj_out_s
