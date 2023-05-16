import casadi as ca


def dynamics(model_dict, x, u):
    theta = x[2]
    delta = x[3]
    v = x[4]
    a = u[0]
    delta_dot = u[1]

    lr = model_dict["lr"]
    L = model_dict["L"]

    beta = ca.atan2(lr * delta, L)
    s = ca.sin(theta + beta)
    c = ca.cos(theta + beta)
    omega = v * ca.cos(beta) * ca.tan(delta) / L

    vx = v * c
    vy = v * s

    x_dot = ca.vertcat(vx, vy, omega, delta_dot, a)
    return x_dot


def nx():
    return 5


def nu():
    return 2


def x_l(model_dict):
    max_steer = model_dict["delta_max"]
    return ca.DM([float('-inf'), float('-inf'), float('-inf'), -1.0 * max_steer, 0.0]).T


def x_u(model_dict):
    max_steer = model_dict["delta_max"]
    max_speed = model_dict["v_max"]
    return ca.DM([float('inf'), float('inf'), float('inf'), max_steer, max_speed]).T


def u_l(model_dict):
    max_lon_dcc = model_dict["a_lon_min"]
    max_delta_dot = model_dict["delta_dot_max"]
    return ca.DM([max_lon_dcc, -1.0 * max_delta_dot]).T


def u_u(model_dict):
    max_lon_acc = model_dict["a_lon_max"]
    max_delta_dot = model_dict["delta_dot_max"]
    return ca.DM([max_lon_acc, max_delta_dot]).T


def test_model():
    opti = ca.Opti()
    model_dict = {"lr": 0.5, "L": 1.0}
    x = opti.variable(5)
    u = opti.variable(2)
    dynamics(model_dict, x, u)


def lat_acc(model_dict, x, u):
    x_dot = dynamics(model_dict, x, u)
    return x_dot[2] * ca.norm_2(x_dot[0:2])


def lon_acc(model_dict, x, u):
    return u[0]
