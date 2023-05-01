import numpy as np
import matplotlib.pyplot as plt
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle


def test_vehicle():
    # vp = VehicleParams(np.zeros((5, 2)), np.zeros((5, 2)),
    #                    10.0, -20.0, 15.0, -15.0, 80.0, 30.0)
    # v = Vehicle(vp)
    # print(v.lookup_acc_circle(lat=10.0))
    # print(v.lookup_acc_circle(lat=-10.0))
    # print(v.lookup_acc_circle(lon=10.0))
    # print(v.lookup_acc_circle(lon=-10.0))

    x = np.linspace(0.0, 15.0, 1001)
    y = np.sqrt((1 - x**2 / 15.0**2) * 10.0**2)
    plt.plot(x, y, '-b', label='traction ellipse constraint')

    y = -np.sqrt((1 - x**2 / 15.0**2) * 20.0**2)
    plt.plot(x, y, '-b')

    x_minus = np.flip(-1.0 * x)
    y = np.sqrt((1 - x_minus**2 / 15.0**2) * 10.0**2)
    plt.plot(x_minus, y, '-b')

    y = -np.sqrt((1 - x_minus**2 / 15.0**2) * 20.0**2)
    plt.plot(x_minus, y, '-b')

    x = np.linspace(0.0, 25.0, 1001)
    y = np.sqrt((1 - x**2 / 20.0**2) * 15.0**2)
    plt.plot(x, y, '-r', label='tire constraint (actual)')

    y = -np.sqrt((1 - x**2 / 20.0**2) * 25.0**2)
    plt.plot(x, y, '-r')

    x_minus = np.flip(-1.0 * x)
    y = np.sqrt((1 - x_minus**2 / 20.0**2) * 15.0**2)
    plt.plot(x_minus, y, '-r')

    y = -np.sqrt((1 - x_minus**2 / 20.0**2) * 25.0**2)
    plt.plot(x_minus, y, '-r')
    plt.xlabel('Lateral Acceleration $(m/s^2)$')
    plt.ylabel('Longitudinal Acceleration $(m/s^2)$')
    plt.title('Traction Ellipse')
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.legend()
    plt.show()


test_vehicle()
