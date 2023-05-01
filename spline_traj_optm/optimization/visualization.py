from spline_traj_optm.models.trajectory import BSplineTrajectory, Trajectory
from spline_traj_optm.models.race_track import RaceTrack

import matplotlib.pyplot as plt

class OptimizationVisualizer:
    def __init__(self, race_track: RaceTrack, traj_s: BSplineTrajectory, traj_d: Trajectory) -> None:
        self._track = race_track
        self._traj_s = traj_s
        self._traj_d = traj_d

    def on_xlims_change(self, event_ax):
        # print("updated xlims: ", event_ax.get_xlim())
        plt.gca().set_aspect('equal')

    def on_ylims_change(self, event_ax):
        plt.gca().set_aspect('equal')
        # print("updated ylims: ", event_ax.get_ylim())

    def visualize(self, traj_s: BSplineTrajectory, traj_d: Trajectory):
        plt.figure()
        plt.plot(self._track.left_d[:, Trajectory.X], self._track.left_d[:, Trajectory.Y], '-c')
        plt.plot(self._track.right_d[:, Trajectory.X], self._track.right_d[:, Trajectory.Y], '-c')
        plt.plot(self._traj_d[:, Trajectory.X], self._traj_d[:, Trajectory.Y], '-y', label="unoptimized")
        plt.plot(self._traj_s._spl_x.tck[1], self._traj_s._spl_y.tck[1], 'oy', label="unoptimized control pts")
        plt.plot(traj_d[:, Trajectory.X], traj_d[:, Trajectory.Y], '-m', label="optimized")
        plt.plot(traj_s._spl_x.tck[1], traj_s._spl_y.tck[1], 'om', label="optimized control pts")

        for x1, y1, x2, y2 in zip(self._traj_s._spl_x.tck[1], self._traj_s._spl_y.tck[1], traj_s._spl_x.tck[1], traj_s._spl_y.tck[1]):
            plt.plot([x1, x2], [y1, y2], '-m')

        plt.gca().set_aspect('equal')
        plt.gca().callbacks.connect('xlim_changed', self.on_xlims_change)
        plt.gca().callbacks.connect('ylim_changed', self.on_ylims_change)
        plt.title(self._track.name)
        plt.xlabel('easting (m)')
        plt.ylabel('northing (m)')
        plt.legend()
        plt.show()