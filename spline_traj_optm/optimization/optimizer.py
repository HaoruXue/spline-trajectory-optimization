import numpy as np
from tqdm import tqdm

from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize
from casadi import *
import matplotlib.pyplot as plt

from spline_traj_optm.models.trajectory import BSplineTrajectory, Trajectory
from spline_traj_optm.models.vehicle import Vehicle, VehicleParams
from spline_traj_optm.simulator.simulator import Simulator
from spline_traj_optm.models.race_track import RaceTrack
from spline_traj_optm.optimization.visualization import OptimizationVisualizer

class TrajectoryOptimizer:
    def __init__(self, race_track:RaceTrack, center_line: BSplineTrajectory, vehicle:Vehicle) -> None:
        self.track = race_track
        self.left_bound = race_track.left_s
        self.right_bound = race_track.right_s
        self.center_line = center_line
        self.vehicle = vehicle
        self.sim = Simulator(self.vehicle)

    def min_curvature_cost(self, z:np.ndarray, idx:int, traj_s: BSplineTrajectory, traj_d: Trajectory):
        ts = traj_d.ts()
        k = traj_s._spl_x.k
        t_start = traj_s._spl_x.t[idx]
        t_end = traj_s._spl_x.t[idx+k+1]
        t_mask = (ts >= t_start) & (ts < t_end)
        ti = ts[t_mask]
        M = len(ti)

        dTx = traj_s._spl_x.derivative()(ti)
        dTy = traj_s._spl_y.derivative()(ti)
        d2Tx= traj_s._spl_x.derivative(2)(ti)
        d2Ty = traj_s._spl_y.derivative(2)(ti)

        # d2Bj = BSpline.basis_element(traj_s._spl_x.t[idx+1:idx+k+2-1])(ti)
        # d2Bj1_mask = (ti <= traj_s._spl_x.t[idx+1]) | (ti > traj_s._spl_x.t[idx+k+2-1])
        # d2Bj[d2Bj1_mask] = 0.0

        # d2Bj1 = BSpline.basis_element(traj_s._spl_x.t[idx+2:idx+k+3-1])(ti)
        # d2Bj1_mask = (ti <= traj_s._spl_x.t[idx+2]) | (ti > traj_s._spl_x.t[idx+k+3-1])
        # d2Bj1[d2Bj1_mask] = 0.0

        # d2Bj2 = BSpline.basis_element(traj_s._spl_x.t[idx+3:idx+k+4-1])(ti)
        # d2Bj2_mask = (ti <= traj_s._spl_x.t[idx+3]) | (ti > traj_s._spl_x.t[idx+k+4-1])
        # d2Bj2[d2Bj2_mask] = 0.0

        # tj = traj_s._spl_x.t[idx]
        # tj1 = traj_s._spl_x.t[idx+1]
        # tj2 = traj_s._spl_x.t[idx+2]
        # tj4 = traj_s._spl_x.t[idx+4]
        # tj5 = traj_s._spl_x.t[idx+5]
        # tj6 = traj_s._spl_x.t[idx+6]
        # beta_j = 5.0 / (tj5 - tj)
        # beta_j1 = -5.0 / (tj6 - tj1)
        # gamma_j = 4.0 / (tj4 - tj) * beta_j
        # gamma_j1 = 4.0 / (tj5 - tj1) * (beta_j1 - beta_j)
        # gamma_j2 = -4.0/ (tj6 - tj2) * beta_j1

        # B = d2Bj * gamma_j + d2Bj1 * gamma_j1 + d2Bj2 * gamma_j2

        B = BSpline.basis_element(traj_s._spl_x.t[idx:idx+k+2])(ti, 2)

        Bx = np.column_stack([B, np.zeros(M, B.dtype)])
        By = np.column_stack([np.zeros(M, B.dtype), B])
        Fx = d2Tx - Bx @ z
        Fy = d2Ty - By @ z

        # v = 1/traj_d[t_mask, Trajectory.SPEED]
        v = np.ones(M, np.float64)
        denom = (dTx ** 2 + dTy ** 2) ** 3
        Pxx = (dTy ** 2 * v) / denom
        Pxy = (-2.0 * dTx * dTy * v) / denom
        Pyy = (dTx ** 2 * v) / denom

        Pxx = np.diag(Pxx)
        Pxy = np.diag(Pxy)
        Pyy = np.diag(Pyy)

        # H = 2.0 * (Bx.T @ Pxx @ Bx + By.T @ Pxy @ Bx + By.T @ Pyy @ By)
        H = 2.0 * (Bx.T @ Pxx @ Bx + By.T @ Pyy @ By)
        # H[0, 1] = H[1, 0]
        g = (Fx.T @ Pxx @ Bx + Fy.T @ Pxy @ By + Fy.T @ Pyy @ By) + (Bx.T @ Pxx @ Fx + By.T @ Pxy @ Fx + By.T @ Pyy @ Fy).T
        return H, g

    def joint_min_curvature_cost(self, traj_s: BSplineTrajectory, traj_d: Trajectory, start_idx=None, span=None):
        num_ctrl_pt = len(traj_s._spl_x.c)
        ignore_ctrl_pt = traj_s._spl_x.k
        ignore_front = ignore_ctrl_pt // 2
        ignore_rear = ignore_ctrl_pt - ignore_front

        if start_idx is None:
            i_min = ignore_front
            i_max = num_ctrl_pt - ignore_rear
        else:
            i_min = start_idx
            i_max = start_idx + span
        N = i_max - i_min
        joint_H = np.zeros((N * 2, N * 2), np.float64)
        joint_g = np.zeros(N * 2, np.float64)
        for i in range(i_min, i_max):
            j = i - i_min
            z0 = np.array(traj_s.get_control_point(i))
            H, g = self.min_curvature_cost(z0, i, traj_s, traj_d)
            joint_H[2*j:2*j+2, 2*j:2*j+2] = H
            joint_g[2*j:2*j+2] = g
            
        return joint_H, joint_g

    def joint_track_constraint(self, traj_s: BSplineTrajectory, traj_d: Trajectory, start_idx=None, span=None):
        num_ctrl_pt = len(traj_s._spl_x.c)
        ignore_ctrl_pt = traj_s._spl_x.k
        ignore_front = ignore_ctrl_pt // 2
        ignore_rear = ignore_ctrl_pt - ignore_front
        ts = traj_d.ts()
        k = traj_s._spl_x.k

        if start_idx is None:
            i_min = ignore_front
            i_max = num_ctrl_pt - ignore_rear
        else:
            i_min = start_idx
            i_max = start_idx + span
        N = i_max - i_min

        M = len(traj_d)
        joint_A = np.zeros((2 * M, 2 * N), dtype=np.float64)
        z = np.zeros(N * 2, np.float64)

        for i in range(i_min, i_max):
            j = i - i_min
            t_start = traj_s._spl_x.t[i]
            t_end = traj_s._spl_x.t[i+k+1]
            t_mask = (ts >= t_start) & (ts < t_end)
            ti = ts[t_mask]
            slice_start = np.argmax(t_mask)
            slice_end = np.argmax(np.invert(t_mask[slice_start:])) + slice_start
            if slice_end == slice_start:
                slice_end = len(t_mask)
            b = BSpline.basis_element(traj_s._spl_x.t[i:i+k+2])
            A = joint_A[slice_start*2:slice_end*2, j*2:j*2+2]
            A[0::2, 0] = b(ti)
            A[1::2, 1] = b(ti)
            z[2*j:2*j+2] = np.array(traj_s.get_control_point(i))
            
        non_z = traj_d[:, :2] - (joint_A @ z).reshape((-1, 2))
        
        left_x = traj_d[:, Trajectory.LEFT_BOUND_X]
        left_y = traj_d[:, Trajectory.LEFT_BOUND_Y]
        right_x = traj_d[:, Trajectory.RIGHT_BOUND_X]
        right_y = traj_d[:, Trajectory.RIGHT_BOUND_Y]
        min_bound = np.empty(2 * M, dtype=left_x.dtype)
        min_bound[0::2] = np.minimum(left_x, right_x) - non_z[:, 0]
        min_bound[1::2] = np.minimum(left_y, right_y) - non_z[:, 1]
        max_bound = np.empty(2 * M, dtype=left_x.dtype)
        max_bound[0::2] = np.maximum(left_x, right_x) - non_z[:, 0]
        max_bound[1::2] = np.maximum(left_y, right_y) - non_z[:, 1]

        return joint_A, min_bound, max_bound

    def run_joint_min_curvature_qp(self, traj_in_s: BSplineTrajectory, traj_in_d: Trajectory, max_iter=3, visualize=False):
        traj_out_s = traj_in_s.copy()
        traj_out_d = traj_in_d.copy()
        self.track.fill_trajectory_boundaries(traj_out_d)
        span=5

        visualizer = OptimizationVisualizer(self.track, traj_in_s, traj_in_d)

        for j in range(max_iter):
            num_ctrl_pt = len(traj_out_s._spl_x.c)
            ignore_ctrl_pt = traj_out_s._spl_x.k
            ignore_front = ignore_ctrl_pt // 2
            ignore_rear = ignore_ctrl_pt - ignore_front
            i_max = num_ctrl_pt - ignore_rear - span
            i_min = ignore_front
            i_start = np.random.randint(i_min, i_max)
            print(f'Starting from {i_start}-th control point.')
            for i in tqdm(range(i_max - i_min)):
                k = i + i_start
                if k >= i_max:
                    k = k - i_max + i_min

                H, g = self.joint_min_curvature_cost(traj_out_s, traj_out_d, k, span)
                A, lba, uba = self.joint_track_constraint(traj_out_s, traj_out_d, k, span)

                DM_H = DM(H)
                DM_A = DM(A)
                qp = {
                    'h': DM_H.sparsity(),
                    'a': DM_A.sparsity(),
                }
                opts = {'printLevel': 'none'}
                qp_solver = conic('solver','qpoases', qp, opts)
                try:
                    r = qp_solver(h=DM_H, g=DM(g), a=DM_A, lba=DM(lba), uba=DM(uba))
                    new_zs = np.array(r['x'])
                    new_zs = new_zs.reshape((-1, 2))
                    for j, new_z in enumerate(new_zs):
                        traj_out_s.set_control_point(k+j, new_z)
                    traj_out_s.set_control_point(0, traj_out_s.get_control_point(-5))
                    traj_out_s.set_control_point(1, traj_out_s.get_control_point(-4))
                    traj_out_s.set_control_point(-3, traj_out_s.get_control_point(2))
                    traj_out_s.set_control_point(-2, traj_out_s.get_control_point(3))
                    traj_out_s.set_control_point(-1, traj_out_s.get_control_point(4))
                    new_traj_out_d = traj_out_s.sample_along(ts = traj_out_d.ts())
                    sim_result = self.sim.run_simulation(new_traj_out_d, enable_vis=False)
                    new_traj_out_d = sim_result.trajectory
                    self.track.fill_trajectory_boundaries(new_traj_out_d)
                    traj_out_d = new_traj_out_d
                except Exception as e:
                    print(e)
                    pass

            if (visualize):
                visualizer.visualize(traj_out_s, traj_out_d)

        visualizer.visualize(traj_out_s, traj_out_d)
        return traj_out_s 
        
    def track_constraint(self, idx:int, traj_s: BSplineTrajectory, traj_d: Trajectory):
        ts = traj_d.ts()
        k = traj_s._spl_x.k
        t_start = traj_s._spl_x.t[idx]
        t_end = traj_s._spl_x.t[idx+k+1]
        t_mask = (ts >= t_start) & (ts < t_end)
        ti = ts[t_mask]
        M = len(ti)

        b_x = BSpline.basis_element(traj_s._spl_x.t[idx:idx+k+2])
        b_y = BSpline.basis_element(traj_s._spl_y.t[idx:idx+k+2])
        bi = np.column_stack([b_x(ti), b_y(ti)])

        old_z = np.array(traj_s.get_control_point(idx))
        non_z = traj_d[t_mask, :2] - bi * old_z[np.newaxis, :]

        # find the boundries
        left_x = traj_d[t_mask, Trajectory.LEFT_BOUND_X]
        left_y = traj_d[t_mask, Trajectory.LEFT_BOUND_Y]
        right_x = traj_d[t_mask, Trajectory.RIGHT_BOUND_X]
        right_y = traj_d[t_mask, Trajectory.RIGHT_BOUND_Y]
        min_bound = np.empty(2 * M, dtype=left_x.dtype)
        min_bound[0::2] = np.minimum(left_x, right_x) - non_z[:, 0]
        min_bound[1::2] = np.minimum(left_y, right_y) - non_z[:, 1]
        max_bound = np.empty(2 * M, dtype=left_x.dtype)
        max_bound[0::2] = np.maximum(left_x, right_x) - non_z[:, 0]
        max_bound[1::2] = np.maximum(left_y, right_y) - non_z[:, 1]

        # find the A matrix
        A = np.zeros((2 * M, 2), dtype=left_x.dtype)
        A[0::2, 0] = b_x(ti)
        A[1::2, 1] = b_y(ti)
        return A, min_bound, max_bound

    def run_min_curvature_qp(self, traj_in_s: BSplineTrajectory, traj_in_d: Trajectory, visualize=False, max_iter=5):
        traj_out_s = traj_in_s.copy()
        traj_out_d = traj_in_d.copy()
        self.track.fill_trajectory_boundaries(traj_out_d)

        visualizer = OptimizationVisualizer(self.track, traj_in_s, traj_in_d)

        def optimize(i, ts, td):
            z0 = np.array(ts.get_control_point(i))
            H, g = self.min_curvature_cost(z0, i, ts, td)
            A, lba, uba = self.track_constraint(i, ts, td)

            DM_H = DM(H)
            DM_A = DM(A)
            qp = {
                'h': DM_H.sparsity(),
                'a': DM_A.sparsity(),
            }
            opts = {'printLevel': 'none'}
            qp_solver = conic('solver','qpoases', qp, opts)
            try:
                r = qp_solver(h=DM_H, g=DM(g), a=DM_A, lba=DM(lba), uba=DM(uba))
                new_z = np.array(r['x'])

                ts.set_control_point(i, new_z)
                ts.set_control_point(0, ts.get_control_point(-5))
                ts.set_control_point(1, ts.get_control_point(-4))
                ts.set_control_point(-3, ts.get_control_point(2))
                ts.set_control_point(-2, ts.get_control_point(3))
                ts.set_control_point(-1, ts.get_control_point(4))
                new_td = ts.sample_along(ts = td.ts())
                new_td[:, Trajectory.SPEED] = td[:, Trajectory.SPEED]
                self.track.fill_trajectory_boundaries(new_td)
                td = new_td
                return ts, td
            except Exception as e:
                # print(e)
                return None, None

        for j in range(max_iter):
            num_ctrl_pt = len(traj_out_s._spl_x.c)
            ignore_ctrl_pt = traj_out_s._spl_x.k
            ignore_front = ignore_ctrl_pt // 2
            ignore_rear = ignore_ctrl_pt - ignore_front
            num_success = 0
            i_max = num_ctrl_pt - ignore_rear
            i_min = ignore_front
            i_start = np.random.randint(i_min, i_max)
            print(f'Starting from {i_start}-th control point.')
            for i in tqdm(range(i_max - i_min)):
                k = i + i_start
                if k >= i_max:
                    k = k - i_max + i_min
                new_traj_out_s, new_traj_out_d = optimize(k, traj_out_s, traj_out_d)
                if new_traj_out_s is not None:
                    num_success += 1
                    traj_out_s = new_traj_out_s
                    traj_out_d = new_traj_out_d
            print(f"Forward pass: number of control points successfully updated: {num_success}")
            num_success = 0
            for i in tqdm(range(i_max - i_min, 0, -1)):
                k = i + i_start
                if k >= i_max:
                    k = k - i_max + i_min
                new_traj_out_s, new_traj_out_d = optimize(k, traj_out_s, traj_out_d)
                if new_traj_out_s is not None:
                    num_success += 1
                    traj_out_s = new_traj_out_s
                    traj_out_d = new_traj_out_d
            print(f"Backward pass: number of control points successfully updated: {num_success}")

            # traj_out_s = BSplineTrajectory(traj_out_d[:, :2], s=50.0, k=5)
            # traj_out_d = traj_out_s.sample_along(3.0)
            # self.track.fill_trajectory_boundaries(traj_out_d)
            
            if (visualize):
                visualizer.visualize(traj_out_s, traj_out_d)
            sim_result = self.sim.run_simulation(traj_out_d, enable_vis=visualize)
            print(f"Iteration {j+1}")
            print(sim_result)
            traj_out_d = sim_result.trajectory

        sim_result = self.sim.run_simulation(traj_out_d, enable_vis=True)
        print(sim_result)
        visualizer.visualize(traj_out_s, traj_out_d)
        return traj_out_s

    # def min_time_cost(self, z: np.array, traj_s: BSplineTrajectory, traj_d: Trajectory):
    #     ti = traj_d.ts()
    #     traj_copy = traj_s.copy()
    #     z_hat = z.reshape((-1, 2))
    #     traj_copy._spl_x.c[2:-3] = z_hat[:, 0]
    #     traj_copy._spl_y.c[2:-3] = z_hat[:, 1]
    #     v = traj_d[:, Trajectory.SPEED]

    #     tsi = np.zeros((len(ti), 2), ti.dtype)
    #     tsi[:, 0] = ti
    #     tsi[:-1, 1] = ti[:-1]
    #     tsi[-1, 1] = tsi[-1, 0]
    #     di = np.apply_along_axis(traj_copy.eval_sectional_length, 1, tsi)

    #     return np.sum(di / v)

    # def jac_min_time_cost(self, z: np.array, traj_s: BSplineTrajectory, traj_d: Trajectory):
    #     ti = traj_d.ts()
    #     traj_copy = traj_s.copy()
    #     z_hat = z.reshape((-1, 2))
    #     traj_copy._spl_x.c[2:-3] = z_hat[:, 0]
    #     traj_copy._spl_y.c[2:-3] = z_hat[:, 1]
    #     tsi = np.zeros((len(ti), 2), ti.dtype)
    #     tsi[:, 0] = ti
    #     tsi[:-1, 1] = ti[:-1]
    #     tsi[-1, 1] = tsi[-1, 0]
    #     d_di = np.apply_along_axis(traj_copy.eval_d_sectional_length, 1, tsi)
    #     v = traj_d[:, Trajectory.SPEED]
    #     return d_di / v


    # def all_boundary_constraint(self, traj_s: BSplineTrajectory, traj_d: Trajectory):
    #     # (ti) get the waypoints that can be influenced by this control point
    #     k = traj_s._spl_x.k
    #     ti = traj_d.ts()
    #     N = len(ti)

    #     # find the boundries
    #     left_x = self.left_bound._spl_x(ti)
    #     left_y = self.left_bound._spl_y(ti)
    #     right_x = self.right_bound._spl_x(ti)
    #     right_y = self.right_bound._spl_y(ti)
    #     min_bound = np.empty(2 * N, dtype=left_x.dtype)
    #     min_bound[0::2] = np.minimum(left_x, right_x)
    #     min_bound[1::2] = np.minimum(left_y, right_y)
    #     max_bound = np.empty(2 * N, dtype=left_x.dtype)
    #     max_bound[0::2] = np.maximum(left_x, right_x)
    #     max_bound[1::2] = np.maximum(left_y, right_y)

    #     # find the A matrix
    #     zx = traj_s._spl_x.c[2:-3]
    #     zy = traj_s._spl_y.c[2:-3]
    #     zs = np.column_stack([zx, zy])

    #     zs_ex = np.column_stack([traj_s._spl_x.c, traj_s._spl_y.c])
    #     A_ex = np.zeros((2 * N, 2 * len(traj_s._spl_x.c)), dtype=left_x.dtype)
    #     # A = np.zeros((2 * N, 2 * len(zs)), dtype=left_x.dtype)
    #     for i, z in enumerate(zs_ex):
    #         t_start = traj_s._spl_x.t[i]
    #         t_end = traj_s._spl_x.t[i+k+1]
    #         b_x = BSpline.basis_element(traj_s._spl_x.t[i:i+k+2])
    #         b_y = BSpline.basis_element(traj_s._spl_y.t[i:i+k+2])
    #         for j in range(0, len(A_ex), 2):
    #             t = ti[j // 2]
    #             if t_start <= t < t_end:
    #                 A_ex[j, 2 * i] = b_x(t)
    #                 A_ex[j + 1, 2 * i + 1] = b_y(t)
    #             elif t >= t_end:
    #                 continue
        
    #     A = A_ex[:, 4:-6]
    #     delta = traj_d[:, :2].reshape((-1,)) - A @ zs.reshape((-1,))
    #     min_bound += delta
    #     max_bound += delta

    #     return LinearConstraint(A, min_bound, max_bound)

    # def max_velocity_cost(self, z: np.array, idx:int, traj_s: BSplineTrajectory, traj_d: Trajectory):
    #     # (ti) get the waypoints that can be influenced by this control point
    #     ts = traj_d.ts()
    #     k = traj_s._spl_x.k
    #     t_start = traj_s._spl_x.t[idx]
    #     t_end = traj_s._spl_x.t[idx+k+1]
    #     t_mask = (ts >= t_start) & (ts < t_end)
    #     ti = ts[t_mask]
    #     N = len(ti)
        
    #     # (wi) calculate the centerline direction vector (unit length)
    #     # yaw_centerline = self.center_line.eval_yaw(ti)
    #     # wi = np.column_stack([np.cos(yaw_centerline), np.sin(yaw_centerline)])

    #     # (pi) build a projection matrix out of wi
    #     # wx = wi[:, 0]
    #     # wy = wi[:, 1]
    #     # wxwy = wx * wy
    #     # pi = np.array([[wx ** 2, wxwy], [wxwy, wy ** 2]])
    #     # pi = np.moveaxis(pi, 2, 0)
    #     # pi = pi.reshape(2 * N, 2)

    #     # (vi) calculate the velocity vector at each discretization point, given the new control point z
    #     traj_copy = traj_s.copy()
    #     traj_copy.set_control_point(idx, z)
    #     # yaw_traj = traj_copy.eval_yaw(ti)
    #     v = traj_d[t_mask, Trajectory.SPEED]
    #     # vi = np.column_stack([np.cos(yaw_traj), np.sin(yaw_traj)]) * v[:, np.newaxis]

    #     # (vi_hat) a diagnal version of vi
    #     # vi_hat = np.zeros((N, N * 2), vi.dtype)
    #     # i = np.arange(N)
    #     # vi_hat[i, 2*i] = vi[:, 0]
    #     # vi_hat[i, 2*i+1] = vi[:, 1]

    #     # (bi) calculate the basis function of the control point, used to weigh the cost at each point
    #     # b_x = BSpline.basis_element(traj_copy._spl_x.t[idx:idx+k+2])
    #     # b_y = BSpline.basis_element(traj_copy._spl_y.t[idx:idx+k+2])
    #     # bi = np.column_stack([b_x(ti), b_y(ti)])

    #     # (di) calculate the distances along the trajectory
    #     tsi = np.zeros((len(ti), 2), ti.dtype)
    #     tsi[:, 0] = ti
    #     tsi[:-1, 1] = ti[1:]
    #     tsi[-1, 1] = tsi[-1, 0]
    #     di = np.apply_along_axis(traj_copy.eval_sectional_length, 1, tsi) 

    #     # finally calculate the cost
    #     # project velocities onto the centerline direction, weigh each point by the basis function value,
    #     # take the sum of squre, and normalize by the number of discretization points
    #     # return np.sum(np.square(di[:, np.newaxis] / vi_hat @ pi * bi)) / N
    #     return np.sum(di / v)

    # def jac_max_velocity_cost(self, z: np.array, idx:int, traj_s: BSplineTrajectory, traj_d: Trajectory):
    #     ts = traj_d.ts()
    #     k = traj_s._spl_x.k
    #     t_start = traj_s._spl_x.t[idx]
    #     t_end = traj_s._spl_x.t[idx+k+1]
    #     t_mask = (ts >= t_start) & (ts < t_end)
    #     ti = ts[t_mask]
    #     N = len(ti)

    #     traj_copy = traj_s.copy()
    #     traj_copy.set_control_point(idx, z)

    #     tsi = np.zeros((len(ti), 2), ti.dtype)
    #     tsi[:, 0] = ti
    #     tsi[:-1, 1] = ti[:-1]
    #     tsi[-1, 1] = tsi[-1, 0]
    #     dx_di = np.apply_along_axis(traj_copy.eval_dx_sectional_length, 1, tsi)
    #     dy_di = np.apply_along_axis(traj_copy.eval_dy_sectional_length, 1, tsi)
    #     v = traj_d[:, Trajectory.SPEED]
    #     return np.array([np.sum(dx_di / v), np.sum(dy_di / v)])

    # def boundary_constraint(self, idx:int, traj_s: BSplineTrajectory, traj_d: Trajectory):
    #     # (ti) get the waypoints that can be influenced by this control point
    #     ts = traj_d.ts()
    #     k = traj_s._spl_x.k
    #     t_start = traj_s._spl_x.t[idx]
    #     t_end = traj_s._spl_x.t[idx+k+1]
    #     t_mask = (ts >= t_start) & (ts < t_end)
    #     ti = ts[t_mask]
    #     N = len(ti)

    #     b_x = BSpline.basis_element(traj_s._spl_x.t[idx:idx+k+2])
    #     b_y = BSpline.basis_element(traj_s._spl_y.t[idx:idx+k+2])
    #     bi = np.column_stack([b_x(ti), b_y(ti)])

    #     old_z = np.array(traj_s.get_control_point(idx))
    #     non_z = traj_d[t_mask, :2] - bi * old_z[np.newaxis, :]

    #     # find the boundries
    #     left_x = self.left_bound._spl_x(ti)
    #     left_y = self.left_bound._spl_y(ti)
    #     right_x = self.right_bound._spl_x(ti)
    #     right_y = self.right_bound._spl_y(ti)
    #     min_bound = np.empty(2 * N, dtype=left_x.dtype)
    #     min_bound[0::2] = np.minimum(left_x, right_x) - non_z[:, 0]
    #     min_bound[1::2] = np.minimum(left_y, right_y) - non_z[:, 1]
    #     max_bound = np.empty(2 * N, dtype=left_x.dtype)
    #     max_bound[0::2] = np.maximum(left_x, right_x) - non_z[:, 0]
    #     max_bound[1::2] = np.maximum(left_y, right_y) - non_z[:, 1]

    #     # find the A matrix
    #     A = np.zeros((2 * N, 2), dtype=left_x.dtype)
    #     A[0::2, 0] = b_x(ti)
    #     A[1::2, 1] = b_y(ti)
    #     return LinearConstraint(A, min_bound, max_bound)

    # def run(self, traj_in_s: BSplineTrajectory, traj_in_d: Trajectory):
    #     num_ctrl_pt = len(traj_in_s._spl_x.c)
    #     ignore_ctrl_pt = traj_in_s._spl_x.k
    #     ignore_front = ignore_ctrl_pt // 2
    #     ignore_rear = ignore_ctrl_pt - ignore_front
    #     traj_out_s = traj_in_s.copy()
    #     traj_out_d = traj_in_d.copy()

    #     num_success = 0
    #     for i in tqdm(range(ignore_front, num_ctrl_pt - ignore_rear)):
    #         z0 = np.array(traj_out_s.get_control_point(i))
    #         constraint = self.boundary_constraint(i, traj_out_s, traj_out_d)
    #         new_z = minimize(self.max_velocity_cost, z0, args=(i, traj_out_s, traj_out_d), constraints=(constraint,))
    #         if new_z.success:
    #             traj_out_s.set_control_point(i, new_z.x)
    #             num_success += 1
    #             new_traj_out_d = traj_out_s.sample_along(ts = traj_out_d.ts())
    #             new_traj_out_d[:, Trajectory.SPEED] = traj_out_d[:, Trajectory.SPEED]
    #             traj_out_d = new_traj_out_d

    #     print(f"Number of control points successfully updated: {i}")

    #     return traj_out_s

    # def run_all(self, traj_in_s: BSplineTrajectory, traj_in_d: Trajectory):
    #     num_ctrl_pt = len(traj_in_s._spl_x.c)
    #     ignore_ctrl_pt = traj_in_s._spl_x.k
    #     ignore_front = ignore_ctrl_pt // 2
    #     ignore_rear = ignore_ctrl_pt - ignore_front
    #     traj_out_s = traj_in_s.copy()

    #     zx = traj_in_s._spl_x.c[2:-3]
    #     zy = traj_in_s._spl_y.c[2:-3]
    #     z0 = np.column_stack([zx, zy]).reshape((-1,))
    #     constraint = self.all_boundary_constraint(traj_out_s, traj_in_d)
    #     result = minimize(self.min_time_cost, z0, args=(traj_out_s, traj_in_d), constraints=(constraint,))
    #     if result.success:
    #         z1 = result.x.reshape((-1, 2))
    #         for i, z in enumerate(z1, 2):
    #             traj_out_s.set_control_point(i, z)

    #     return traj_out_s
