from constants import SONAR, INFRA

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

    def __init__(self, env, ident=None, mass=1., R_dist=.1, F_max=.02,
                 V_max=.01, power=3, friction=.1):
        self.ident = ident
        self.environ = env

        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.orientation = 0.0

        self.mass = 1.0
        self.rmax = 0.4
        self.fmax = 0.02
        self.vmax = 0.01

        self.p = 3
        self.u = 0.1
        self.G = 0.2

    def sense_environment(self):
        """The robot senses the environment and determines locations of walls,
        entrances and other robots.
        """
        sensor_estimates = self.environ.estimate_sensor_readings(self.ident)
        sonar = np.vstack((
            sensor_estimates['sonar'][None], SONAR.ANGLE_RES[None])).T
        infra = sensor_estimates['infra']

        # Detect other robots
        estimated_robots = []
        non_robot_detect = []

        res = sonar.shape[0]
        for i in range(sonar.shape[0]):
            if sonar[i-1, 0] == 4.0:
                continue

            d1 = sonar[i-2, 0] - sonar[i-1, 0]
            d2 = sonar[i-1, 0] - sonar[i, 0]
            d3 = sonar[i, 0] - sonar[(i+1) % res, 0]
            d4 = sonar[(i+1) % res, 0] - sonar[(i+2) % res, 0]

            check1 = (d2 * d3) < 0
            check2 = np.abs(d2-d1) > 0.05 and np.abs(d3-d4) > 0.05
            if check1 and check2:
                estimated_robots.append(sonar[i])
            else:
                non_robot_detect.append(sonar[i])

        # diff = [sonar[i, 0] - sonar[i-1, 0] for i in range(sonar.shape[0])]
        # new_plot(np.arange(len(diff)), np.array(diff))
        # new_plot(np.arange(sonar.shape[0]), sonar[:, 0])

        force = np.array([0.0, 0.0])
        for bot in estimated_robots:
            if bot[0] > (2.0 * self.rmax):
                continue

            fmag = self.G * (self.mass**2) / (bot[0]**self.p)
            if bot[0] > self.rmax:
                fmag *= -1
            force += ut.pol2cart((fmag, bot[1]))

        for pnt in non_robot_detect:
            if pnt[0] > (2.0 * self.rmax):
                continue

            fmag = self.G * (self.mass**2) / (pnt[0]**self.p)
            if pnt[0] > (1.1 * self.rmax):
                fmag *= -1
            force += ut.pol2cart((fmag, pnt[1]))

        self.accel = force / self.mass

        # Detecting entrances.
        # estimated_entrance = []

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
        sensor_estimates = self.environ.estimate_sensor_readings(self.ident)
        self.sonar = np.vstack((
            sensor_estimates['sonar'][None], SONAR.ANGLE_RES[None])).T
        self.infra = sensor_estimates['infra']

        if self.infra['front'] > .15:
            self.position[1] += .1
        elif self.infra['rear'] > .15:
            self.position[1] -= .1
        elif self.infra['right'] > .15:
            self.orientation = -np.pi / 2
            self.position[0] += 1
        elif self.infra['left'] > .15:
            self.orientation = np.pi / 2
            self.position[0] -= 1
        else:
            raise RuntimeError(
                "Unable to move robot at ({0}, {1}).".format(*self.position))

        self.environ.update_robot(self.ident)

    def recalibrate(self):
        del self.accel
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.orientation = 0.0
