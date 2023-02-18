import numpy as np
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle


def test_vehicle():
    vp = VehicleParams(np.zeros((5, 2)), np.zeros((5, 2)),
                       10.0, -20.0, 15.0, -15.0, 80.0, 30.0)
    v = Vehicle(vp)
    print(v.lookup_acc_circle(lat=10.0))
    print(v.lookup_acc_circle(lat=-10.0))
    print(v.lookup_acc_circle(lon=10.0))
    print(v.lookup_acc_circle(lon=-10.0))


test_vehicle()
