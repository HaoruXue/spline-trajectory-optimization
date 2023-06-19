import casadi as ca

def global_to_frenet(p, p0, yaw):
    cos_theta = ca.cos(-yaw)
    sin_theta = ca.sin(-yaw)
    R = ca.DM(ca.vertcat(ca.horzcat(cos_theta, -sin_theta),
                         ca.horzcat(sin_theta, cos_theta)))
    return R @ (p - p0)

def align_yaw(yaw1, yaw2):
    d_yaw = yaw1 - yaw2
    d_yaw = ca.atan2(ca.sin(d_yaw), ca.cos(d_yaw))
    return d_yaw + yaw2