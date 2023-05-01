from spline_traj_optm.models.trajectory import Trajectory
import matplotlib.pyplot as plt


class SimulatorVisualization:
    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory
        plt.ion()

        self.figure, self.axs = plt.subplots(2, 3, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1]})
        self.figure.suptitle("Offline Trajectory Optimization")

        def heatmap(fig, axes, title, value, cmap):
            axes.set_title(title)
            scat = axes.scatter(
                self.trajectory[:, Trajectory.X],
                self.trajectory[:, Trajectory.Y],
                c=value,
                cmap=cmap,
            )
            fig.colorbar(scat, ax=axes)
            axes.axis("equal")
            return scat

        def plot(fig, axes, title, value, x, line_option):
            axes.set_ylabel(title)
            axes.set_xlabel("Distance $(m)$")
            p, = axes.plot(x, value, line_option)
            return p

        self.scat_speed = heatmap(self.figure, self.axs[0, 0], "Center Line Velocity Profile $(m/s)$", self.trajectory[:, Trajectory.SPEED], "plasma")
        self.scat_lat_acc = heatmap(self.figure, self.axs[0, 1], "Lateral Acceleration $(m/s^2)$", self.trajectory[:, Trajectory.LAT_ACC], "plasma")
        self.scat_lon_acc = heatmap(self.figure, self.axs[0, 2], "Longitudinal Acceleration $(m/s^2)$", self.trajectory[:, Trajectory.LON_ACC], "bwr")
        
        self.plot_speed = plot(self.figure, self.axs[1, 0], "Velocity $(m/s)$", self.trajectory[:, Trajectory.SPEED], self.trajectory[:, Trajectory.DIST_TO_SF_BWD], "-r")
        self.plot_lat_acc = plot(self.figure, self.axs[1, 1], "Lateral Acc $(m/s^2)$", self.trajectory[:, Trajectory.LAT_ACC], self.trajectory[:, Trajectory.DIST_TO_SF_BWD], "-g")
        self.plot_lon_acc = plot(self.figure, self.axs[1, 2], "Longitudinal Acc $(m/s^2)$", self.trajectory[:, Trajectory.LON_ACC], self.trajectory[:, Trajectory.DIST_TO_SF_BWD], "-b")
    def update_plot(self, sleep_time=0.0):
        # self.scat_speed.set_offsets(self.trajectory[:, 0:2])
        self.scat_speed.set_array(self.trajectory[:, Trajectory.SPEED])
        self.scat_speed.autoscale()

        # self.scat_lat_acc.set_offsets(self.trajectory[:, 0:2])
        self.scat_lat_acc.set_array(self.trajectory[:, Trajectory.LAT_ACC])
        self.scat_lat_acc.autoscale()

        # self.scat_lon_acc.set_offsets(self.trajectory[:, 0:2])
        self.scat_lon_acc.set_array(self.trajectory[:, Trajectory.LON_ACC])
        self.scat_lon_acc.autoscale()

        self.plot_speed.set_ydata(self.trajectory[:, Trajectory.SPEED])
        self.axs[1, 0].relim()
        self.axs[1, 0].autoscale()
        self.plot_lat_acc.set_ydata(self.trajectory[:, Trajectory.LAT_ACC])
        self.axs[1, 1].relim()
        self.axs[1, 1].autoscale()
        self.plot_lon_acc.set_ydata(self.trajectory[:, Trajectory.LON_ACC])
        self.axs[1, 2].relim()
        self.axs[1, 2].autoscale()

        self.figure.canvas.draw_idle()

        plt.pause(sleep_time)

    def latch_plot(self):
        plt.ioff()
        plt.show()
        plt.ion()


class SimulatorVelocityVisualization:
    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory
        plt.ion()

        self.figure, self.axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        self.figure.suptitle("Center Line Trajectory Simulation")

        def heatmap(fig, axes, title, value, cmap):
            axes.set_title(title)
            axes.set_xlabel("Easting $(m)$")
            axes.set_ylabel("Northing $(m)$")
            scat = axes.scatter(
                self.trajectory[:, Trajectory.X],
                self.trajectory[:, Trajectory.Y],
                c=value,
                cmap=cmap,
            )
            fig.colorbar(scat, ax=axes)
            axes.axis("equal")
            return scat

        def plot(fig, axes, title, value, x, line_option):
            axes.set_ylabel(title)
            axes.set_xlabel("Distance $(m)$")
            p, = axes.plot(x, value, line_option)
            return p

        self.scat_speed = heatmap(self.figure, self.axs[0], "Velocity Map $(m/s)$", self.trajectory[:, Trajectory.SPEED], "plasma")
        # self.scat_lat_acc = heatmap(self.figure, self.axs[0, 1], "Lateral Acceleration $(m/s^2)$", self.trajectory[:, Trajectory.LAT_ACC], "plasma")
        # self.scat_lon_acc = heatmap(self.figure, self.axs[0, 2], "Longitudinal Acceleration $(m/s^2)$", self.trajectory[:, Trajectory.LON_ACC], "bwr")
        
        self.plot_speed = plot(self.figure, self.axs[1], "Velocity $(m/s)$", self.trajectory[:, Trajectory.SPEED], self.trajectory[:, Trajectory.DIST_TO_SF_BWD], "-r")
        # self.plot_lat_acc = plot(self.figure, self.axs[1, 1], "Lateral Acc $(m/s^2)$", self.trajectory[:, Trajectory.LAT_ACC], self.trajectory[:, Trajectory.DIST_TO_SF_BWD], "-g")
        # self.plot_lon_acc = plot(self.figure, self.axs[1, 2], "Longitudinal Acc $(m/s^2)$", self.trajectory[:, Trajectory.LON_ACC], self.trajectory[:, Trajectory.DIST_TO_SF_BWD], "-b")
    def update_plot(self, sleep_time=0.0):
        # self.scat_speed.set_offsets(self.trajectory[:, 0:2])
        self.scat_speed.set_array(self.trajectory[:, Trajectory.SPEED])
        self.scat_speed.autoscale()

        # self.scat_lat_acc.set_array(self.trajectory[:, Trajectory.LAT_ACC])
        # self.scat_lat_acc.autoscale()

        # self.scat_lon_acc.set_array(self.trajectory[:, Trajectory.LON_ACC])
        # self.scat_lon_acc.autoscale()

        self.plot_speed.set_ydata(self.trajectory[:, Trajectory.SPEED])
        self.axs[1].relim()
        self.axs[1].autoscale()
        # self.plot_lat_acc.set_ydata(self.trajectory[:, Trajectory.LAT_ACC])
        # self.axs[1, 1].relim()
        # self.axs[1, 1].autoscale()
        # self.plot_lon_acc.set_ydata(self.trajectory[:, Trajectory.LON_ACC])
        # self.axs[1, 2].relim()
        # self.axs[1, 2].autoscale()

        self.figure.canvas.draw_idle()

        plt.pause(sleep_time)

    def latch_plot(self):
        plt.ioff()
        plt.show()
        plt.ion()