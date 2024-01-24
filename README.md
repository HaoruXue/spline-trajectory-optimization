# spline-trajectory-optimization

**Please head over to `devel` branch for the code in the paper "Spline-Based Minimum-Curvature Trajectory Optimization for Autonomous Racing".**

Spline-based Trajectory Optimization tool for Autonomous Racing (Indy Autonomous Challenge)

## Install

1. Install `SciPy`, `matplotlib`, `shapely`, `casadi`, `bezier`.
2. For min curvature problem, install Julia 1.8.5+.
3. Clone this repository and install with `pip install -e .`.

## Run

- For min curvature problem, run `julia/spline_traj_opt.ipynb`.
- For min time problem, copy `traj_opt_double_track.yaml` in `spline_traj_optm/min_time_otpm/example` to your workspace, and execute `traj_opt_double_track`.

## Repository Organization

- `spline_traj_optm/examples`: Example inputs for the trajectory optimization (Monza)
- `spline_traj_optm/models`: Data classes for holding optimization information (race track, vehicle, and trajectory information)
- `spline_traj_optm/optimization`: Optimization functions
- `spline_traj_optm/simulator`: Quasi-Steady State (QSS) simulation of a given trajectory for optimal speed
- `spline_traj_optm/tests`: Tests for the package
- `spline_traj_optm/visualization`: Functions for visualization the optimization and simulation results
- `julia/spline_traj_opt.ipynb`: Julia notebook of the optimization notebook

## Input Format

csvs for inside bound , outside bound, and center line
- 3 columns

![image](https://github.com/HaoruXue/spline-trajectory-optimization/assets/32603181/f88bc388-4f4a-4837-9324-ef53f03b6f84)

If there is a bank angle, should be 4 columns with the fourth column as bank angle (if 3 columns bank default to zero)
![image](https://github.com/HaoruXue/spline-trajectory-optimization/assets/32603181/39b34fe9-7436-417e-8409-f67922fb796c)

