from math import sqrt
from spline_traj_optm.models.trajectory import Trajectory
from spline_traj_optm.models.vehicle import Vehicle, VehicleParams
from spline_traj_optm.simulator.visualization import SimulatorVisualization, SimulatorVelocityVisualization
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
import time

@dataclass
class SimulationResult:
    trajectory: Trajectory
    run_time: float
    total_time: float
    average_speed: float
    max_speed: float
    min_speed: float
    max_lat_acc: float
    max_lon_acc: float
    max_lon_dcc: float

    def __str__(self):
        return str(
            f"+---------------------+\n"
            "|  Simulation Result  |\n"
            "+---------------------+\n"
            "| Run Time  | {:7.2f} |\n"
            "| Lap Time  | {:7.2f} |\n"
            "| Ave Speed | {:7.2f} |\n"
            "| Max Speed | {:7.2f} |\n"
            "| Min Speed | {:7.2f} |\n"
            "| Max Lat G | {:7.2f} |\n"
            "| Max ACC   | {:7.2f} |\n"
            "| MAX DCC   | {:7.2f} |\n"
            "+---------------------+\n".format(
                self.run_time,
                self.total_time,
                self.average_speed,
                self.max_speed,
                self.min_speed,
                self.max_lat_acc,
                self.max_lon_acc,
                self.max_lon_dcc,
            )
        )

class Simulator:
    def __init__(self, vehicle: Vehicle) -> None:
        self.vehicle = vehicle

    def calc_lat_acc(self, v: float, r: float, bank: float):
        return v**2 / r + 9.81 * np.sin(bank)

    def calc_v(self, lat_acc: float, r: float, bank: float):
        return sqrt(np.abs(np.abs(lat_acc) - 9.81 * np.sin(bank)) * r)

    def calc_r(self, lat_acc: float, v: float, bank: float):
        return v**2 / np.abs(np.abs(lat_acc) - 9.81 * np.sin(bank))

    def run_simulation(
        self, trajectory: Trajectory, enable_vis=False
    ) -> SimulationResult:
        start_time = time.time()
        trajectory_out = trajectory.copy()
        # x_data = trajectory_out[:, Trajectory.X]
        # x_data = np.append(x_data, x_data[0])
        # x_data = np.insert(x_data, 0, x_data[-1])
        # y_data = trajectory_out[:, Trajectory.Y]
        # y_data = np.append(y_data, y_data[0])
        # y_data = np.insert(y_data, 0, y_data[-1])
        # trajectory_out[:, Trajectory.X] = np.convolve(
        #     x_data, [0.3, 0.4, 0.3], mode="valid"
        # )
        # trajectory_out[:, Trajectory.Y] = np.convolve(
        #     y_data, [0.3, 0.4, 0.3], mode="valid"
        # )
        # trajectory_out[:, Trajectory.CURVATURE] = gaussian_filter1d(trajectory_out[:, Trajectory.CURVATURE], 1.0, mode='wrap')

        if enable_vis:
            vis = SimulatorVelocityVisualization(trajectory_out)

        # Find points on track with max curvatures
        def find_turns() -> np.ndarray:

            # # We only consider curves whose radius is too small for full speed.
            # accs = self.vehicle.lookup_acc_circle(lon=0.0)
            # full_speed_lat_acc = max(accs[0], accs[1] * -1.0) # right acc is negative. flip it.
            # min_curvature_for_full_speed = self.calc_r(
            #     full_speed_lat_acc, self.vehicle.param.max_speed_mps, 0.0
            # )

            # bottle_necks = []
            # for i in range(len(trajectory_out)):
            #     last_curvature = trajectory_out[trajectory_out.dec(i), Trajectory.CURVATURE]
            #     this_curvature = trajectory_out[i, Trajectory.CURVATURE]
            #     next_curvature = trajectory_out[trajectory_out.inc(i), Trajectory.CURVATURE]
            #     # if last_curvature > this_curvature or next_curvature > this_curvature:
            #     if this_curvature < min_curvature_for_full_speed:
            #         bottle_necks.append(i)
            # return np.array(bottle_necks, dtype=int)
            # return np.random.choice(np.arange(len(trajectory_out)), size=int(len(trajectory_out) * 0.7), replace=False)
            # return np.random.choice(np.arange(len(trajectory_out)), size=len(trajectory_out), replace=False)
            return np.arange(len(trajectory_out))

        # Start by identifying where the turns are
        turns = np.array(find_turns())
        print("turns,", turns)
        # iteration_flags is num_turns * 5 where each column means:
        # the turn entry index, the turn index, the turn exit index,
        # if the turn entry iteration is stopped, and
        # if the turn exit iteration is stopped
        iteration_flags = np.repeat(turns[:, np.newaxis], 5, axis=1)
        print(iteration_flags)
        iteration_flags[:, 3:] = 0

        def calc_distance(pt1, pt2):
            return sqrt(
                (pt1[Trajectory.X] - pt2[Trajectory.X]) ** 2
                + (pt1[Trajectory.Y] - pt2[Trajectory.Y]) ** 2
            )

        def calc_distance_along_trajectory(pt1, pt2):
            i = pt1[Trajectory.IDX]
            dist = 0.0
            while (i != pt2[Trajectory.IDX]):
                next_i = trajectory_out.inc(i)
                if next_i != 0.0:
                    dist += trajectory_out[i, Trajectory.DIST_TO_SF_FWD] - trajectory_out[next_i, trajectory_out.DIST_TO_SF_FWD]
                else:
                    dist += trajectory_out[i, Trajectory.DIST_TO_SF_FWD]
                i = next_i
            return dist

        # For every turn, assume zero lon acc, populate initial conditions
        for turn in turns:
            turn_pt = trajectory_out[turn]
            turn_pt[Trajectory.SPEED] = min(
                self.calc_v(
                    self.vehicle.lookup_acc_circle(lon=0.0)[0],
                    turn_pt[Trajectory.CURVATURE],
                    turn_pt[Trajectory.BANK]
                ),
                self.vehicle.param.max_speed_mps,
            )
            turn_pt[Trajectory.LON_ACC] = 0.0
            turn_pt[Trajectory.LAT_ACC] = self.calc_lat_acc(
                turn_pt[Trajectory.SPEED], turn_pt[Trajectory.CURVATURE], turn_pt[Trajectory.BANK]
            )
            turn_pt[Trajectory.ITERATION_FLAG] = turn
        
        def iterate(iteration_flags):
            itr = 0
            while True:
                # For every turn, enter it as fast as possible
                new_flags = []
                for flags in iteration_flags:
                    stopped = flags[3] == 1
                    if stopped:
                        continue
                    last_enter_pt = trajectory_out[flags[0]]
                    flags[0] = trajectory_out.dec(flags[0])
                    enter_pt = trajectory_out[flags[0]]
                    dd = calc_distance(last_enter_pt, enter_pt)

                    # Get the possible speed ranges from the last state
                    np.seterr(all='raise')
                    dt = dd / last_enter_pt[Trajectory.SPEED]
                    max_dacc = dt * self.vehicle.param.max_jerk
                    max_acc = last_enter_pt[Trajectory.LON_ACC] + max_dacc
                    min_acc = last_enter_pt[Trajectory.LON_ACC] - max_dacc
                    vehicle_max_acc = self.vehicle.lookup_acc_from_speed(
                        last_enter_pt[Trajectory.SPEED]
                    )
                    vehicle_max_dcc = self.vehicle.lookup_dcc_from_speed(
                        last_enter_pt[Trajectory.SPEED]
                    )
                    max_acc = np.clip(max_acc, vehicle_max_dcc, vehicle_max_acc)
                    min_acc = np.clip(min_acc, vehicle_max_dcc, vehicle_max_acc)
                    # v_0^2 = 2ax - v^2
                    min_state_speed = sqrt(
                        max(last_enter_pt[Trajectory.SPEED] ** 2 - 2 * max_acc * dd, 0.0)
                    )
                    max_state_speed = sqrt(
                        max(last_enter_pt[Trajectory.SPEED] ** 2 - 2 * min_acc * dd, 0.0)
                    )
                    # Get the max possible speed from the curvature
                    max_lat_acc = self.vehicle.lookup_acc_circle(
                        lon=last_enter_pt[Trajectory.LON_ACC]
                    )[0]
                    max_curve_speed = self.calc_v(
                        max_lat_acc, enter_pt[Trajectory.CURVATURE],
                        enter_pt[Trajectory.BANK]
                    )
                    min_curve_speed = 0.0
                    # Get the max possible speed from vehicle constraint
                    max_vehicle_speed = self.vehicle.param.max_speed_mps
                    min_vehicle_speed = 0.0
                    # Find the minimum of the three maximum speeds
                    max_greedy_speed = min(
                        max_state_speed, max_curve_speed, max_vehicle_speed
                    )
                    # Check if this speed is valid in all constraints
                    if (
                        min_state_speed <= max_greedy_speed <= max_state_speed
                        and min_curve_speed <= max_greedy_speed <= max_curve_speed
                        and min_vehicle_speed <= max_greedy_speed <= max_vehicle_speed
                    ):

                        # Check if another turn has populated a speed at this point
                        # If their speed is slower, do not overwrite because it will not meet their kinematic constraint
                        # Instead, wait for them to overwrite our speed
                        # If our speed is slower, overwrite becuase their speed will not meet our kinematic constraint
                        if (
                            enter_pt[Trajectory.ITERATION_FLAG] != -1
                            and enter_pt[Trajectory.SPEED] < max_greedy_speed
                        ):
                            # Signal the stop flag
                            flags[3] = 1
                        else:
                            if not (min_acc <= enter_pt[Trajectory.LON_ACC] <= max_acc):
                                # Signal the merge mode flag
                                flags[3] = -1
                            # if valid, make it final
                            enter_pt[Trajectory.SPEED] = max_greedy_speed
                            # a = (v^2 - v_0^2) / (2x)
                            enter_pt[Trajectory.LON_ACC] = (
                                last_enter_pt[Trajectory.SPEED] ** 2
                                - enter_pt[Trajectory.SPEED] ** 2
                            ) / (2 * dd)
                            enter_pt[Trajectory.LAT_ACC] = self.calc_lat_acc(
                                enter_pt[Trajectory.SPEED], enter_pt[Trajectory.CURVATURE], enter_pt[Trajectory.BANK]
                            )
                            enter_pt[Trajectory.ITERATION_FLAG] = flags[1]
                    else:
                        # if constraint failed because the curvature is too high
                        # make this point an additional turn

                        # If not valid (meaning the car will fly off), stop for another constraint to handle it
                        # Signal the stop flag
                        flags[3] = 1
                        if max_greedy_speed > max_curve_speed or max_greedy_speed < min_state_speed:
                            idx = enter_pt[Trajectory.IDX]
                            new_flags.append(np.array([idx, idx, idx, 0, 0]))
                            enter_pt[Trajectory.SPEED] = min(
                                self.calc_v(
                                    self.vehicle.lookup_acc_circle(lon=0.0)[0],
                                    enter_pt[Trajectory.CURVATURE],
                                    enter_pt[Trajectory.BANK]
                                ),
                                self.vehicle.param.max_speed_mps,
                            )
                            enter_pt[Trajectory.LON_ACC] = 0.0
                            enter_pt[Trajectory.LAT_ACC] = self.calc_lat_acc(
                                enter_pt[Trajectory.SPEED], enter_pt[Trajectory.CURVATURE], enter_pt[Trajectory.BANK]
                            )
                            enter_pt[Trajectory.ITERATION_FLAG] = idx

                # For every turn, exit it as fast as possible
                for flags in iteration_flags:
                    stopped = flags[4] == 1
                    if stopped:
                        continue
                    last_exit_pt = trajectory_out[flags[2]]
                    flags[2] = trajectory_out.inc(flags[2])
                    exit_pt = trajectory_out[flags[2]]
                    dd = calc_distance(last_exit_pt, exit_pt)

                    # Get the possible speed ranges from the last state
                    dt = dd / last_exit_pt[Trajectory.SPEED]
                    max_dacc = dt * self.vehicle.param.max_jerk
                    max_acc = last_exit_pt[Trajectory.LON_ACC] + max_dacc
                    min_acc = last_exit_pt[Trajectory.LON_ACC] - max_dacc
                    vehicle_max_acc = self.vehicle.lookup_acc_from_speed(
                        last_exit_pt[Trajectory.SPEED]
                    )
                    vehicle_max_dcc = self.vehicle.lookup_dcc_from_speed(
                        last_exit_pt[Trajectory.SPEED]
                    )
                    max_acc = np.clip(max_acc, vehicle_max_dcc, vehicle_max_acc)
                    min_acc = np.clip(min_acc, vehicle_max_dcc, vehicle_max_acc)
                    # v^2 = 2ax + v_0^2
                    max_state_speed = sqrt(
                        max(2 * max_acc * dd + last_exit_pt[Trajectory.SPEED] ** 2, 0.0)
                    )
                    min_state_speed = sqrt(
                        max(2 * min_acc * dd + last_exit_pt[Trajectory.SPEED] ** 2, 0.0)
                    )
                    # Get the max possible speed from the curvature
                    max_lat_acc = self.vehicle.lookup_acc_circle(
                        lon=last_exit_pt[Trajectory.LON_ACC]
                    )[0]
                    max_curve_speed = self.calc_v(
                        max_lat_acc, exit_pt[Trajectory.CURVATURE],
                        exit_pt[Trajectory.BANK]
                    )
                    min_curve_speed = 0.0
                    # Get the max possible speed from vehicle constraint
                    max_vehicle_speed = self.vehicle.param.max_speed_mps
                    min_vehicle_speed = 0.0
                    # Find the minimum of the three maximum speeds
                    max_greedy_speed = min(
                        max_state_speed, max_curve_speed, max_vehicle_speed
                    )
                    # Check if this speed is valid in all constraints
                    if (
                        min_state_speed <= max_greedy_speed <= max_state_speed
                        and min_curve_speed <= max_greedy_speed <= max_curve_speed
                        and min_vehicle_speed <= max_greedy_speed <= max_vehicle_speed
                    ):

                        # Check if another turn has populated a speed at this point
                        # If their speed is slower, do not overwrite because it will not meet their kinematic constraint
                        # Instead, wait for them to overwrite our speed
                        # If our speed is slower, overwrite becuase their speed will not meet our kinematic constraint
                        if (
                            exit_pt[Trajectory.ITERATION_FLAG] != -1
                            and exit_pt[Trajectory.SPEED] < max_greedy_speed
                        ):
                            # Signal the stop flag
                            flags[4] = 1
                        else:
                            if not (min_acc <= exit_pt[Trajectory.LON_ACC] <= max_acc):
                                # Signal the merge mode flag
                                flags[4] = -1
                            # if valid, make it final
                            exit_pt[Trajectory.SPEED] = max_greedy_speed
                            # a = (v^2 - v_0^2) / (2x)
                            exit_pt[Trajectory.LON_ACC] = (
                                exit_pt[Trajectory.SPEED] ** 2
                                - last_exit_pt[Trajectory.SPEED] ** 2
                            ) / (2 * dd)
                            exit_pt[Trajectory.LAT_ACC] = self.calc_lat_acc(
                                exit_pt[Trajectory.SPEED], exit_pt[Trajectory.CURVATURE], exit_pt[Trajectory.BANK]
                            )
                            exit_pt[Trajectory.ITERATION_FLAG] = flags[1]
                    else:
                        # If not valid (meaning the car will fly off), stop for another constraint to handle it
                        # Signal the stop flag
                        flags[4] = 1
                        # if constraint failed because the curvature is too high
                        # make this point an additional turn
                        if max_greedy_speed > max_curve_speed:
                            idx = exit_pt[Trajectory.IDX]
                            new_flags.append(np.array([idx, idx, idx, 0, 0]))
                            exit_pt[Trajectory.SPEED] = max_curve_speed
                            exit_pt[Trajectory.LON_ACC] = last_exit_pt[Trajectory.LON_ACC]
                            exit_pt[Trajectory.LAT_ACC] = self.calc_lat_acc(
                                exit_pt[Trajectory.SPEED], exit_pt[Trajectory.CURVATURE], exit_pt[Trajectory.BANK]
                            )
                            exit_pt[Trajectory.ITERATION_FLAG] = idx

                # add new flags at end of iteration
                if len(new_flags) > 0:
                    iteration_flags = np.vstack([iteration_flags, np.array(new_flags, dtype=int)])

                # remove done iterations
                mask = (iteration_flags[:, 3] != 1) | (iteration_flags[:, 4] != 1)
                iteration_flags = iteration_flags[mask, :]

                if enable_vis and itr % 100 == 0:
                    vis.update_plot(0.001)
                # Check if all iterations are stopped
                # if np.all(iteration_flags[:, 3:] == 1):
                #     break
                if len(iteration_flags) == 0:
                    break
                itr += 1
                print(itr)

        iterate(iteration_flags)
       

        # Populate the time and distance fields
        trajectory_out.fill_time()

        if enable_vis:
            vis.latch_plot()

        return SimulationResult(
            trajectory=trajectory_out,
            run_time=time.time() - start_time,
            total_time=trajectory_out[0, Trajectory.TIME],
            average_speed=trajectory_out[0, Trajectory.DIST_TO_SF_FWD]
            / trajectory_out[0, Trajectory.TIME],
            max_speed=np.max(trajectory_out[:, Trajectory.SPEED]),
            min_speed=np.min(trajectory_out[:, Trajectory.SPEED]),
            max_lat_acc=np.max(trajectory_out[:, Trajectory.LAT_ACC]),
            max_lon_acc=np.max(trajectory_out[:, Trajectory.LON_ACC]),
            max_lon_dcc=np.min(trajectory_out[:, Trajectory.LON_ACC]),
        )
