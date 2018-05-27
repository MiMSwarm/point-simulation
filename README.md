# MiniMapper Simulation on Matplotlib

> A naive point-based simulation carried out using a plotting library to verify the functioning of the control algorithms designed for the robot swarm.

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


## Dependencies

The code was successfully tested on the following versions of the required libraries,
- Python 3.6
- NumPy 1.14.0
- SciPy 1.0.0
- Matplotlib 2.1.2

## Authors

Abhijit J Theophilus, abhijit.theo@gmail.com\
Surya Prakash M, starsurya96@gmail.com\
Vinay Kumar Adiga G, vinayadiga96@gmail.com\
Vijayalaxmi Hullatti, vijayalaxmi.6868@gmail.com

## License

Copyright 2018\
Licensed under the MIT License.
