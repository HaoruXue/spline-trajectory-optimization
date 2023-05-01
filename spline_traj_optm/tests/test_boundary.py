from importlib_resources import files
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from spline_traj_optm.tests.test_trajectory import get_bspline, get_trajectory_array
from spline_traj_optm.optimization.optimizer import TrajectoryOptimizer
from spline_traj_optm.models.trajectory import Trajectory, BSplineTrajectory
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator
from spline_traj_optm.models.race_track import RaceTrack
import spline_traj_optm.examples.race_track.monza

traj_spline = get_bspline(files(spline_traj_optm.examples.race_track.monza).joinpath(
        "MONZA_UNOPTIMIZED_LINE_enu.csv"), s=30.0)
traj_discrete = traj_spline.sample_along(3.0)
race_track = RaceTrack("Monza", get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_LEFT_BOUNDARY_enu.csv")), get_trajectory_array(files(spline_traj_optm.examples.race_track.monza).joinpath("MONZA_RIGHT_BOUNDARY_enu.csv")))

race_track.fill_trajectory_boundaries(traj_discrete)
np.savetxt("test_traj.csv", traj_discrete.points, delimiter=",")