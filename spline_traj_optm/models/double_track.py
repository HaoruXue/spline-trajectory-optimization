# https://doi.org/10.1080/00423114.2019.1704804

import casadi as ca

import spline_traj_optm.utils.utils as utils

GRAVITY = 9.8


def dynamics(model_dict, x, u,bank, race_track=None, k=None):
    px = x[0]
    py = x[1]
    phi = x[2]  # yaw in frenet frame or global frame
    omega = x[3]  # yaw rate
    beta = x[4]  # slip angle
    v = x[5]  # velocity magnitude
    fd = u[0] * (ca.tanh(u[0]) * 0.5 + 0.5)  # drive force
    fb = u[0] * (ca.tanh(-u[0]) * 0.5 + 0.5)  # brake force
    delta = u[2]  # front wheel angle
    gamma_y = u[3]  # lateral load transfer

    kd_f = model_dict["kd_f"]  # front drive force bias (rear is 1 - kd_f)
    kb_f = model_dict["kb_f"]  # front brake force bias (rear is 1 - kb_f)
    m = model_dict["mass"]  # mass of car
    Jzz = model_dict["Jzz"]  # MOI around z axis
    lf = model_dict["lf"]  # cg to front axle
    lr = model_dict["lr"]  # cg to rear axle
    l = lf + lr  # wheelbase
    twf = model_dict["twf"]  # front track width
    twr = model_dict["twr"]  # rear track width
    fr = model_dict["fr"]  # rolling resistance coefficient
    hcog = model_dict["hcog"]  # center of gravity height
    kroll_f = model_dict["kroll_f"]  # front roll moment distribution
    cl_f = model_dict["cl_f"]  # downforce coefficient at front
    cl_r = model_dict["cl_r"]  # downforce coefficient at rear
    rho = model_dict["rho"]  # air density
    A = model_dict["A"]  # frontal area
    cd = model_dict["cd"]  # drag coefficient
    mu = model_dict["mu"]  # tyre - track friction coefficient

    # magic tyre parameters
    Bf = model_dict["Bf"]  # magic formula B - front
    Cf = model_dict["Cf"]  # magic formula C - front
    Ef = model_dict["Ef"]  # magic formula E - front
    Fz0_f = model_dict["Fz0_f"]  # magic formula Fz0 - front
    eps_f = model_dict["eps_f"]  # extended magic formula epsilon - front
    Br = model_dict["Br"]  # magic formula B - rear
    Cr = model_dict["Cr"]  # magic formula C - rear
    Er = model_dict["Er"]  # magic formula E - rear
    Fz0_r = model_dict["Fz0_r"]  # magic formula Fz0 - rear
    eps_r = model_dict["eps_r"]  # extended magic formula epsilon - rear

    # longitudinal tyre force Fx (eq. 4a, 4b)
    # TODO consider differential

    
    N = m * GRAVITY * ca.cos(bank) + (m * (v **2)/ (1/k)) * abs(ca.sin(bank)) #normal force

    Fx_f = 0.5 * kd_f * fd + 0.5 * kb_f * fb - 0.5 * fr * N * lr / l
    Fx_fl = Fx_f
    Fx_fr = Fx_f
    Fx_r = 0.5 * (1 - kd_f) * fd + 0.5 * (1 - kb_f) * \
        fb - 0.5 * fr *  N * lf / l
    Fx_rl = Fx_r
    Fx_rr = Fx_r

    # longitudinal acceleration (eq. 9)
    ax = (fd + fb - 0.5 * cd * A * v ** 2 - fr * N) / m

    # vertical tyre force Fz (eq. 7a, 7b)
    Fz_f = 0.5 * N * lr / \
        (lf + lr) - 0.5 * hcog / (lf + lr) * \
        m * ax + 0.25 * cl_f * rho * A * v ** 2
    Fz_fl = Fz_f - kroll_f * gamma_y
    Fz_fr = Fz_f + kroll_f * gamma_y
    Fz_r = 0.5 * N* lr / \
        (lf + lr) + 0.5 * hcog / (lf + lr) * \
        m * ax + 0.25 * cl_r * rho * A * v ** 2
    Fz_rl = Fz_r - (1 - kroll_f) * gamma_y
    Fz_rr = Fz_r + (1 - kroll_f) * gamma_y

    # tyre sideslip angles alpha (eq. 6a, 6b)
    a_fl = delta - ca.arctan((lf * omega + v * ca.sin(beta)) /
                             (v * ca.cos(beta) - 0.5 * twf * omega))
    a_fr = delta - ca.arctan((lf * omega + v * ca.sin(beta)) /
                             (v * ca.cos(beta) + 0.5 * twf * omega))
    a_rl = ca.arctan((lr * omega - v * ca.sin(beta)) /
                     (v * ca.cos(beta) - 0.5 * twr * omega))
    a_rr = ca.arctan((lr * omega - v * ca.sin(beta)) /
                     (v * ca.cos(beta) + 0.5 * twr * omega))

    # lateral tyre force Fy (eq. 5)
    # Fy_fl = mu * Fz_fl * (1 + eps_f * Fz_fl / Fz0_f) *  \
    #     ca.sin(Cf * ca.arctan(Bf * a_fl - Ef * (Bf * a_fl - ca.arctan(Bf * a_fl))))
    # Fy_fr = mu * Fz_fr * (1 + eps_f * Fz_fr / Fz0_f) * \
    #     ca.sin(Cf * ca.arctan(Bf * a_fr - Ef * (Bf * a_fr - ca.arctan(Bf * a_fr))))
    # Fy_rl = mu * Fz_rl * (1 + eps_r * Fz_rl / Fz0_r) * \
    #     ca.sin(Cr * ca.arctan(Br * a_rl - Er * (Br * a_rl - ca.arctan(Br * a_rl))))
    # Fy_rr = mu * Fz_rr * (1 + eps_r * Fz_rr / Fz0_r) * \
    #     ca.sin(Cr * ca.arctan(Br * a_rr - Er * (Br * a_rr - ca.arctan(Br * a_rr))))

    # lateral tyre force simplified (eq. 5)
    Fy_fl = mu * Fz_fl * ca.sin(Cf * ca.arctan(Bf * a_fl))
    Fy_fr = mu * Fz_fr * ca.sin(Cf * ca.arctan(Bf * a_fr))
    Fy_rl = mu * Fz_rl * ca.sin(Cr * ca.arctan(Br * a_rl))
    Fy_rr = mu * Fz_rr * ca.sin(Cr * ca.arctan(Br * a_rr))

    # dynamics (eq. 3a, 3b, 3c)
    v_dot = 1 / m * ((Fx_rl + Fx_rr) * ca.cos(beta) + (Fx_fl + Fx_fr) * ca.cos(delta - beta)
                     + (Fy_rl + Fy_rr + m * GRAVITY * abs(ca.sin(bank))) * ca.sin(beta) -
                     (Fy_fl + Fy_fr) * ca.sin(delta - beta)
                     - 0.5 * cd * rho * A * v ** 2 * ca.cos(beta))
    beta_dot = -omega + 1 / (m * v) * (-(Fx_rl + Fx_rr) * ca.sin(beta) + (Fx_fl + Fx_fr) * ca.sin(delta - beta)
                                       + (Fy_rl + Fy_rr) * ca.cos(beta) +
                                       (Fy_fl + Fy_fr) * ca.cos(delta - beta)
                                       + 0.5 * cd * rho * A * v ** 2 * ca.sin(beta))
    omega_dot = 1 / Jzz * ((Fx_rr - Fx_rl) * twr / 2 - (Fy_rl + Fy_rr) * lr
                           + ((Fx_fr - Fx_fl) * ca.cos(delta) +
                              (Fy_fl - Fy_fr) * ca.sin(delta)) * twf / 2
                           + ((Fy_fl + Fy_fr) * ca.cos(delta) + (Fx_fl + Fx_fr) * ca.sin(delta)) * lf)

    # cg position
    x_dot = v * ca.cos(phi + beta)
    y_dot = v * ca.sin(phi + beta)
    phi_dot = omega
    if race_track is not None:
        x_dot /= (1 - py * k)
        phi_dot -= k * x_dot

    X_dot = ca.vertcat(x_dot, y_dot, phi_dot, omega_dot, beta_dot, v_dot)

    Fxij = (Fx_fl, Fx_fr, Fx_rl, Fx_rr)
    Fyij = (Fy_fl, Fy_fr, Fy_rl, Fy_rr)
    Fzij = (Fz_fl, Fz_fr, Fz_rl, Fz_rr)

    return X_dot, (Fxij, Fyij, Fzij)


def nx():
    return 6


def nu():
    return 4


def add_constraints(model_dict, opti, x, u, t, xip1, uip1,bank, race_track=None, k=None):
    # tyre constraints
    v = x[5]  # velocity magnitude
    fd = u[0] * (ca.tanh(u[0]) * 0.5 + 0.5)  # drive force
    fb = u[0] * (ca.tanh(-u[0]) * 0.5 + 0.5)  # brake force
    delta = u[2]  # front wheel angle
    gamma_y = u[3]  # lateral load transfer

    twf = model_dict["twf"]  # front track widthS
    twr = model_dict["twr"]  # rear track width
    delta_max = model_dict["delta_max"]  # max front wheel angle

    hcog = model_dict["hcog"]  # center of gravity height
    mu = model_dict["mu"]  # tyre - track friction coefficient

    # powertrain parameters
    Pmax = model_dict["Pmax"]  # motor max power
    Fd_max = model_dict["Fd_max"]  # max driver force
    Fb_max = model_dict["Fb_max"]  # max brake force
    Td = model_dict["Td"]  # drive time constant
    Tb = model_dict["Tb"]  # brake time constant
    Tdelta = model_dict["Tdelta"]  # steering time constant

    bank = bank

    # dynamics constraint
    temp = ca.MX(xip1)
    temp[0, 2] = utils.align_yaw(temp[0, 2], x[0, 2])
    if race_track is not None:
        temp[0, 0] = utils.align_abscissa(temp[0, 0], x[0, 0], race_track.center_s.get_length())
    f1, tyres = dynamics(model_dict, x, u,bank, race_track, k)
    f2, _ = dynamics(model_dict, temp, u,bank, race_track, k)
    xm = 0.5 * (x + temp) + (t / 8.0) * (f1.T - f2.T)
    fm, _ = dynamics(model_dict, xm, u,bank, race_track, k)
    opti.subject_to(x + (t / 6.0) * (f1.T + 4 * fm.T + f2.T) - temp == 0)

    Fxij, Fyij, Fzij = tyres
    Fx_fl, Fx_fr, Fx_rl, Fx_rr = Fxij
    Fy_fl, Fy_fr, Fy_rl, Fy_rr = Fyij

    for Fx_ij, Fy_ij, Fz_ij in zip(Fxij, Fyij, Fzij):
        opti.subject_to(ca.power(Fx_ij / (mu * Fz_ij), 2) +
                        ca.power(Fy_ij / (mu * Fz_ij), 2) <= 1)

    # load transfer constraint
    opti.subject_to(gamma_y == hcog / (0.5 * (twf + twr)) * (Fy_rl + Fy_rr +
                    (Fx_fl + Fx_fr) * ca.sin(delta) + (Fy_fl + Fy_fr) * ca.cos(delta)))

    # static actuator constraint
    opti.subject_to(v * fd <= Pmax)
    opti.subject_to(v >= 1.0)
    # opti.subject_to(opti.bounded(0.0, fd, Fd_max))
    # opti.subject_to(opti.bounded(Fb_max, fb, 0.0))
    # opti.subject_to(ca.power(fd * fb, 2) <= 1.0)
    # opti.subject_to(fd * fb == 0.0)
    opti.subject_to(opti.bounded(Fb_max, u[0], Fd_max))
    opti.subject_to(opti.bounded(-delta_max, delta, delta_max))

    # dynamic actuator constraint
    # opti.subject_to((uip1[0] - fd) / t <= Fd_max / Td)
    # opti.subject_to((uip1[1] - fb) / t >= Fb_max / Tb)
    opti.subject_to(opti.bounded(Fb_max / Tb, (uip1[0] - u[0]) / t, Fd_max / Td))
    opti.subject_to(opti.bounded(-delta_max / Tdelta,
                    (uip1[2] - delta) / t, delta_max / Tdelta))


def test_model():
    opti = ca.Opti()
    model_dict = {
        "kd_f": 0.0,
        "kb_f": 0.7,
        "mass": 1200.0,
        "Jzz": 1260,
        "lf": 1.5,
        "lr": 1.4,
        "twf": 1.6,
        "twr": 1.5,
        "delta_max": 0.4,

        "fr": 0.01,
        "hcog": 0.4,
        "kroll_f": 0.5,

        "cl_f": 2.4,
        "cl_r": 3.0,
        "rho": 1.2041,
        "A": 1.0,
        "cd": 1.4,
        "mu": 1.5,

        "Bf": 9.62,
        "Cf": 2.59,
        "Ef": 1.0,
        "Fz0_f": 3000.0,
        "eps_f": -0.0813,
        "Br": 8.62,
        "Cr": 2.65,
        "Er": 1.0,
        "Fz0_r": 3000.0,
        "eps_r": -0.1263,

        "Pmax": 270000.0,
        "Fd_max": 7100.0,
        "Fb_max": -20000.0,
        "Td": 1.0,
        "Tb": 1.0,
        "Tdelta": 1.0,
    }
    x = opti.variable(nx())
    u = opti.variable(nu())
    dynamics(model_dict, x, u)
