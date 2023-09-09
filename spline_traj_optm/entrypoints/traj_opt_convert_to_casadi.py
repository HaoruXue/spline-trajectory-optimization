from spline_traj_optm.models.trajectory import Trajectory, load_ttl
import casadi as ca
from sys import argv

def main():
    if len(argv) != 3:
        print('Usage: python3 traj_opt_encode_region.py <TTL_in> <casadi_txt_out>')
        return
    regions = []
    ttl_in = argv[1]
    ttl_out = argv[2]

    ttl = load_ttl(ttl_in)
    ttl[:, Trajectory.SPEED] = -100.0
    ca.DM(ttl.points[:, :Trajectory.TIME+1]).to_file(ttl_out, "txt")
