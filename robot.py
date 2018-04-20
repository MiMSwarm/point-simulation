from constants import SONAR
from exceptions import SimulationError

import numpy as np
import utils as ut
import sys


class MiniMapper:

    """Models the MiniMapper in the simulation.

    Parameters
    ----------
    ident : hashable object, optional
        Used to identify the MiniMapper.
    env : environment.Environment
        The environment within which the object resides.
    """

    def __init__(self, env, ident=None, mass=1., R_dist=.3, F_max=.02,
                 V_max=.05, power=3, friction=.1, Gconst=.2):
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

        self.setup = False
        self.sonar = None
        self.infra = None
        self.reset = False

        self.check1 = False
        self.check2 = False

        self.next_position = None

    def update(self, current=None):
        """Update the state of the robot."""

        # First iteration, every robot senses the environment.
        if self.sonar is None:
            percepts = self.environ.estimate_percepts(self.ident)
            self.sonar = np.vstack((percepts['sonar'], SONAR.ANGLE_RES)).T
            self.infra = percepts['infra']
            if current is None:
                current = self.ident

        # Second iteration onwards, two possibilities:
        #   Lattice formation or Area Sweep.

        # Lattice formation.
        elif not self.setup and current == self.ident:
            self.sense_environment()
            if self.next_position is None:
                self.identify_wall()
                print('\tIdentified wall. Moving to', self.next_position)
            self.update_position()
            if np.all(self.position == self.next_position) or \
                    np.any(self.infra):
                self.setup = True
        else:
            if not self.reset:
                self.sense_environment()
                orient = np.arctan2(self.p2[1]-self.p1[1], self.p2[0]-self.p1[0])
                self.orientation = ut.shift_angles(orient)
                self.environ.update_robot(self.ident)
                self.recalibrate()
                print('\n\tRecalibrated to', ut.rounded_int(np.degrees(
                    self.environ.robots[self.ident]['initial_ang'])))

                self.check1 = True
                self.check2 = True
                self.reset = True
                self.velocity = np.array([0.0, 0.0])

                if not self.infra[0]:
                    self.velocity[0] = -.05
                elif not self.infra[1]:
                    self.velocity[1] = -.05
                elif not self.infra[2]:
                    self.velocity[1] = +.05
                else:
                    self.velocity[0] = +.05

                self.position += self.velocity
                self.infra_prev = np.copy(self.infra)

            else:
                self.sense_environment()

                if np.all(~self.infra):
                    if self.check1:
                        self.velocity = np.array(
                            [[0, -1], [1, 0]]) @ self.velocity
                        self.orientation += np.pi / 2
                        self.orientation = ut.shift_angles(self.orientation)
                    self.check1 = False

                elif np.any(self.infra_prev != self.infra):
                    if self.check2:
                        self.velocity[:] = 0.0
                        if not self.infra[0]:
                            self.velocity[0] = -.05
                        elif not self.infra[1]:
                            self.velocity[1] = -.05
                        elif not self.infra[2]:
                            self.velocity[1] = +.05
                        else:
                            self.velocity[0] = +.05
                    self.check2 = False
                    self.infra_prev[:] = self.infra[:]
                else:
                    self.check1 = True
                    self.check2 = True

                self.position += self.velocity
                self.environ.update_robot(self.ident)

        return current

    def sense_environment(self):
        """The robot senses the environment and determines locations of walls,
        entrances and other robots.
        """
        sensor_estimates = self.environ.estimate_percepts(self.ident)
        self.sonar = np.vstack((
            sensor_estimates['sonar'][None], SONAR.ANGLE_RES[None])).T
        infra = sensor_estimates['infra']
        self.infra = np.array([infra['front'], infra['left'],
                               infra['right'], infra['rear']])

    def identify_wall(self):
        """Identifies the closest wall."""
        window = 3
        extent = 7

        sonar_cart = ut.pol2cart(self.sonar)
        sonar_dist = np.round(
            np.linalg.norm(sonar_cart[1:] - sonar_cart[:-1], axis=1), 2)

        max_val = np.max(sonar_dist)

        condition = True
        while condition and not np.all(sonar_dist == max_val):
            min_val = np.min(sonar_dist)
            nmin = np.sum(sonar_dist == min_val)
            if nmin < extent:
                sonar_dist[sonar_dist == min_val] = max_val
            else:
                ind = np.argsort(sonar_dist)[:nmin]
                ind = np.sort(ind)
                if np.max(ind[1:] - ind[:-1]) <= window:
                    i1 = ind[len(ind)//2]
                    condition = False
                if condition:
                    sonar_dist[sonar_dist == min_val] = max_val

        for w in range(1, window):
            if sonar_dist[i1 - w + 1] == sonar_dist[i1 + w]:
                i0 = i1 - w
                i2 = i1 + w

        self.p1 = sonar_cart[i0]
        pm = sonar_cart[i1]
        self.p2 = sonar_cart[i2]

        w = np.arctan2(self.p1[1] - self.p2[1], self.p2[0] - self.p1[0])
        w -= self.orientation
        r0 = self.rmax
        r1 = -self.rmax

        pt0 = np.round(ut.pol2cart((r0, w)) + pm, 2)
        pt1 = np.round(ut.pol2cart((r1, w)) + pm, 2)

        d1 = np.linalg.norm(self.position - pt0)
        d2 = np.linalg.norm(self.position - pt1)
        self.next_position = pt0 if d1 < d2 else pt1

    def update_position(self):
        self.velocity = self.next_position - self.position
        pol_vel = ut.cart2pol(self.velocity)

        self.orientation = pol_vel[1]
        np.clip(self.velocity, -self.vmax, self.vmax, out=self.velocity)

        self.position += self.velocity
        self.environ.update_robot(self.ident)

    def recalibrate(self):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.orientation = 0.0
        self.environ.robots[self.ident]['initial_pos'] = \
            self.environ.robots[self.ident]['current_pos']
        self.environ.robots[self.ident]['initial_ang'] = \
            self.environ.robots[self.ident]['current_ang']
