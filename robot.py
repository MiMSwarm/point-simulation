from constants import SONAR
from exceptions import SimulationError

import numpy as np
import utils as ut


class MiniMapper:

    """Models the MiniMapper in the simulation.

    Parameters
    ----------
    ident : hashable object, optional
        Used to identify the MiniMapper.
    env : environment.Environment
        The environment within which the object resides.
    """

    def __init__(self, env, ident=None, mass=1., R_dist=.2, F_max=.02,
                 V_max=.01, power=3, friction=.1, Gconst=.2):
        self.ident = ident
        self.environ = env

        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.orientation = 0.0

        self.mass = mass
        self.rmax = R_dist
        self.fmax = F_max
        self.vmax = V_max

        self.p = power
        self.u = friction
        self.G = Gconst

    def sense_environment(self):
        """The robot senses the environment and determines locations of walls,
        entrances and other robots.
        """
        sensor_estimates = self.environ.estimate_percepts(self.ident)
        sonar = np.vstack((
            sensor_estimates['sonar'][None], SONAR.ANGLE_RES[None])).T
        tmp = ut.pol2cart(sonar)
        fig, ax = ut.new_plot(tmp[:, 0], tmp[:, 1], 'o', ms=1)
        tmp2 = ut.pol2cart(self.sonar)
        ax.plot(tmp2[:, 0], tmp2[:, 1], 'o', ms=1)
        # infra = sensor_estimates['infra']
        self.detect_robots(sonar)

    def detect_robots(self, sonar):
        """Tries to identify robots from the sensor readings."""
        condn = (self.sonar[:, 0] < 2.0) & (sonar[:, 0] < 2.0)

        vec_a = ut.cart2pol(self.velocity)
        vec_c = self.sonar[condn]
        vec_b = np.empty(vec_c.shape)

        θ = vec_c[:, 1] - vec_a[1]                  # Angle between vectors.
        P = -vec_a[0]                               # Mag of vector a.
        Q = vec_c[:, 0]                             # Mag of vector c.

        # Calculating b = c - a
        vec_b[:, 0] = np.sqrt((P**2) + (2*P*Q*np.cos(θ)) + (Q**2))
        vec_b[:, 1] = np.arctan2(Q*np.sin(θ), Q*np.cos(θ) + P)

        # The angles give the indices into sonar
        ind = ut.degree_ind(vec_b[:, 1])
        print(np.round(np.abs(sonar[ind, 0] - vec_b[:, 0]), 2))
        # print(np.sum((sonar[ind, 0] - vec_b[:, 0]) > .5))

    def update_position(self):
        pol_acc = ut.cart2pol(self.accel)
        self.orientation = pol_acc[1]

        self.velocity *= self.u
        self.velocity += self.accel
        np.clip(self.velocity, -self.vmax, self.vmax, out=self.velocity)
        self.position += self.velocity
        self.environ.update_robot(self.ident)

    def initial_setup(self):
        """Called after the environment construction is complete."""
        sensor_estimates = self.environ.estimate_percepts(self.ident)
        self.sonar = np.vstack((
            sensor_estimates['sonar'][None], SONAR.ANGLE_RES[None])).T
        self.infra = sensor_estimates['infra']

        if not self.infra['front']:
            self.velocity[0] = .10
        elif not self.infra['rear']:
            self.velocity[0] = -.10
        elif not self.infra['right']:
            self.velocity[1] = -.10
        elif not self.infra['left']:
            self.velocity[1] = .10
        else:
            raise SimulationError(self.ident)

        self.position += self.velocity
        self.environ.update_robot(self.ident)

    def recalibrate(self):
        del self.accel
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.orientation = 0.0
