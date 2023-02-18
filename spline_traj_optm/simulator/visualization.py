from trajectory_tools.simulator.model.trajectory import Trajectory
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

        def plot(fig, axes, title, value, line_option):
            axes.set_title(title)
            p, = axes.plot(value, line_option)
            return p

        self.scat_speed = heatmap(self.figure, self.axs[0, 0], "Speed (m/s)", self.trajectory[:, Trajectory.SPEED], "plasma")
        self.scat_lat_acc = heatmap(self.figure, self.axs[0, 1], "Lateral Acceleration (m/s^2)", self.trajectory[:, Trajectory.LAT_ACC], "plasma")
        self.scat_lon_acc = heatmap(self.figure, self.axs[0, 2], "Longitudinal Acceleration (m/s^2)", self.trajectory[:, Trajectory.LON_ACC], "bwr")
        
        self.plot_speed = plot(self.figure, self.axs[1, 0], "", self.trajectory[:, Trajectory.SPEED], "-r")
        self.plot_lat_acc = plot(self.figure, self.axs[1, 1], "", self.trajectory[:, Trajectory.LAT_ACC], "-g")
        self.plot_lon_acc = plot(self.figure, self.axs[1, 2], "", self.trajectory[:, Trajectory.LON_ACC], "-b")
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
