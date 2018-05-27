# MiniMapper Simulation on Matplotlib

> A simple point-based simulation carried out using a plotting library to verify the functioning of the control algorithms designed for the robot swarm.

## Usage

The Python files to execute for running the simulation are,
- `simulate.py`: Test the wall-following algorithm.
- `demo_sq.py`: Test the physicomimetic framework in forming square lattices.
- `demo_hex.py`: Test the physicomimetic framework in forming hexagonal lattices.

For testing lattice formations, the following parameters may be passed to the `RobotSwarm` constructor,
- `nbot`: number of robots,
- `mass`: mass of robots,
- `R_dist`: inter-robot distance,
- `F_max`: maximum force,
- `V_max`: maximum velocity,
- `power`: parameter *p*, and
- `friction`: friction coefficient *μ*.
