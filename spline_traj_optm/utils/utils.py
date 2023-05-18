import casadi as ca

def global_to_frenet(p, p0, yaw):
    cos_theta = ca.cos(-yaw)
    sin_theta = ca.sin(-yaw)
    R = ca.DM(ca.vertcat(ca.horzcat(cos_theta, -sin_theta),
                         ca.horzcat(sin_theta, cos_theta)))
    return R @ (p - p0)

def align_yaw(yaw1, yaw2):
    k = ca.fabs(yaw2-yaw1)+ca.pi
    l = k - ca.fmod(ca.fabs(yaw2-yaw1)+ca.pi, 2 * ca.pi)
    return yaw1 + l * ca.sign(yaw2 - yaw1)