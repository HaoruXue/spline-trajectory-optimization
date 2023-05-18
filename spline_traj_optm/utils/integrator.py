import casadi as ca
import spline_traj_optm.utils.utils as utils

def hermite_simpson(model, dynamics, x1, x2, u, dt):
    temp = ca.MX(x2)
    temp[0, 2] = utils.align_yaw(temp[0, 2], x1[0, 2])
    f1 = dynamics(model, x1, u).T
    f2 = dynamics(model, temp, u).T
    xm = 0.5 * (x1 + temp) + (dt / 8.0) * (f1 - f2)
    fm = dynamics(model, xm, u).T
    return x1 + (dt / 6.0) * (f1 + 4 * fm + f2) - temp

def rk4(model, dynamics, x1, x2, u, dt):
    temp = ca.MX(x2)
    temp[0, 2] = utils.align_yaw(temp[0, 2], x1[0, 2])
    k1 = dynamics(model, x1, u).T
    k2 = dynamics(model, x1 + dt/2 * k1, u).T
    k3 = dynamics(model, x1 + dt/2 * k2, u).T
    k4 = dynamics(model, x1 + dt * k3, u).T
    return x1 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) - temp