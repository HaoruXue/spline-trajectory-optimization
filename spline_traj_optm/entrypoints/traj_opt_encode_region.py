from spline_traj_optm.models.trajectory import Trajectory, Region, save_ttl, load_ttl
import yaml
from sys import argv
import numpy as np

def main():
    if len(argv) != 4:
        print('Usage: python3 traj_opt_encode_region.py <regions.yaml> <TTL_in> <TTL_out>')
        return
    regions = []
    regions_yaml_path = argv[1]
    ttl_in = argv[2]
    ttl_out = argv[3]

    with open(regions_yaml_path, "r") as f:
        polygon_dict = yaml.load(f, yaml.SafeLoader)
    for _, polygon in polygon_dict.items():
        assert type(polygon) is dict
        vertices = np.loadtxt(
            polygon['file'], dtype=float, delimiter=',', skiprows=1)[:, 0:2]
        regions.append(
            Region(
                polygon["name"],
                polygon["code"],
                vertices
            )
        )

    ttl = load_ttl(ttl_in)
    ttl.fill_region(regions)
    save_ttl(ttl_out, ttl)