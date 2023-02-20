from importlib_resources import files
import numpy as np

from spline_traj_optm.tests.test_trajectory import get_bspline
from spline_traj_optm.models.trajectory import Trajectory, BSplineTrajectory
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator
import spline_traj_optm.samples.race_track.monza


def test_simulator():
    traj_spline = get_bspline(files(spline_traj_optm.samples.race_track.monza).joinpath(
        "MONZA_UNOPTIMIZED_LINE_enu.csv"))
    traj_discrete = traj_spline.sample_along(1.0)

    acc_speed_lookup = np.array([[0.0, 10.0], [50.0, 7.0], [100.0, 0.5]])
    dcc_speed_lookup = np.array([[0.0, -13.0], [50.0, -15.0], [100.0, -20.0]])
    vp = VehicleParams(acc_speed_lookup, dcc_speed_lookup,
                       10.0, -20.0, 15.0, -15.0, 100.0, 30.0)
    v = Vehicle(vp)

    sim = Simulator(v)
    result = sim.run_simulation(traj_discrete, True)
    print(result)
    return result


test_simulator()
