# spline-trajectory-optimization

Spline-based Trajectory Optimization tool for Autonomous Racing (Indy Autonomous Challenge)

## Install

1. Install `SciPy`, `matplotlib`, `shapely`.
2. Install Julia 1.8.5+.
3. Clone this repository and install with `pip install -e .`.

## Run

Run `julia/spline_traj_opt.ipynb`.

## Repository Organization

- `spline_traj_optm/examples`: Example inputs for the trajectory optimization (Monza)
- `spline_traj_optm/models`: Data classes for holding optimization information (race track, vehicle, and trajectory information)
- `spline_traj_optm/optimization`: Optimization functions
- `spline_traj_optm/simulator`: Quasi-Steady State (QSS) simulation of a given trajectory for optimal speed
- `spline_traj_optm/tests`: Tests for the package
- `spline_traj_optm/visualization`: Functions for visualization the optimization and simulation results
- `julia/spline_traj_opt.ipynb`: Julia notebook of the optimization notebook
