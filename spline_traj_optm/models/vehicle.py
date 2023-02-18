import numpy as np
from dataclasses import dataclass
from scipy import interpolate


@dataclass
class VehicleParams:
    acc_speed_lookup: np.ndarray
    dcc_speed_lookup: np.ndarray
    max_lon_acc_mpss: float  # signed acceleration (positive)
    max_lon_dcc_mpss: float  # signed deceleration (negative)
    max_left_acc_mpss: float  # signed acceleration (positive)
    max_right_acc_mpss: float  # signed acceleration (negative)
    max_speed_mps: float


class Vehicle:
    def __init__(self, param: VehicleParams):
        self.param = param
        self.acc_intp = interpolate.splrep(
            self.param.acc_speed_lookup[:, 0], self.param.acc_speed_lookup[:, 1], s=1.0, k=3)
        self.dcc_intp = interpolate.splrep(
            self.param.dcc_speed_lookup[:, 0], self.param.dcc_speed_lookup[:, 1], s=1.0, k=3)

    def lookup_acc_from_speed(self, speed_mps: float) -> tuple:
        return interpolate.splev(speed_mps, self.acc_intp)

    def lookup_dcc_from_speed(self, speed_mps: float):
        return interpolate.splev(speed_mps, self.dcc_intp)

    def lookup_acc_circle(self, lat=None, lon=None, model='ellipse'):
        assert (lat is not None) or (lon is not None)
        if model == 'ellipse':
            if lat is not None:
                lat = np.clip(lat, self.param.max_right_acc_mpss,
                              self.param.max_left_acc_mpss)
                max_lat = self.param.max_left_acc_mpss if lat > 0.0 else self.param.max_right_acc_mpss
                return self.__lookup_acc_ellipse(lat, max_lat, self.param.max_lon_acc_mpss, self.param.max_lon_dcc_mpss)
            else:
                lon = np.clip(lon, self.param.max_lon_dcc_mpss,
                              self.param.max_lon_acc_mpss)
                max_lon = self.param.max_lon_acc_mpss if lon > 0.0 else self.param.max_lon_dcc_mpss
                return self.__lookup_acc_ellipse(lon, max_lon, self.param.max_left_acc_mpss, self.param.max_right_acc_mpss)

    def __lookup_acc_ellipse(self, val, x, y_pos, y_neg):
        return y_pos * np.sqrt(1.0 - val ** 2 / x ** 2), y_neg * np.sqrt(1.0 - val ** 2 / x ** 2)
