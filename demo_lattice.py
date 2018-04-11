#!/usr/bin/python3

import sys
import numpy as np
from scipy.spatial import distance

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Set a few numpy options.
np.seterr(divide='ignore')
np.set_printoptions(precision=2)


class RobotSwarm:

    def __init__(self, nbot, mass=.5, R_dist=1., F_max=.2, V_max=.1,
                 power=2, friction=.05, lattice='hex'):
        self.n = nbot
        self.m = mass
        self.R = R_dist
        self.F = F_max
        self.V = V_max

        self.p = power
        self.u = friction

        self.lattice = lattice
        if lattice == 'hex':
            self.G = 1.8 * self.F * (self.R ** self.p) / (2 * np.sqrt(3))

        elif lattice == 'cube':
            self.G = 1.8 * self.F * (self.R ** self.p) / (2 * np.sqrt(2) + 2)
            h1 = np.arange(0, nbot/2, dtype=int)
            h2 = np.arange(nbot/2, nbot, dtype=int)

            self._likes = np.full((nbot, nbot), False)
            self._likes[np.ix_(h1, h1)] = True
            self._likes[np.ix_(h2, h2)] = True

        else:
            raise ValueError('Lattice must be hex or cube.')

        self.v = np.zeros((self.n, 2))
        self.X = np.random.uniform(-5, 5, (self.n, 2))

    def setup_plot(self, fig, ax):
        self.fig = fig
        self.ax = ax

        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)

        h = self.n // 2
        self.l1, = self.ax.plot(self.X[:h, 0], self.X[:h, 1], 'ro', ms=2)
        self.l2, = self.ax.plot(self.X[h:, 0], self.X[h:, 1], 'bo', ms=2)

    def __call__(self, i):
        r = distance.pdist(self.X)

        # Estimating the magnitude of force.
        f = self.G * (self.m ** 2) / (r ** self.p)
        f = distance.squareform(f)
        r = distance.squareform(r)

        if self.lattice == 'hex':
            f[r > self.R] *= -1
        elif self.lattice == 'cube':
            f[self._likes][r[self._likes] > self.R] *= -1
            f[~self._likes][r[~self._likes] > (self.R * np.sqrt(2))] *= -1

        f[r > (1.5 * self.R)] = 0
        # print(np.max(f), '\t', np.min(f))

        # Estimating the direction of force.
        unit = self.X[None, :] - self.X[:, None]
        umod = np.linalg.norm(unit, axis=2)[:, :, None]
        umod[umod == 0] = 1                 # Just so that NaNs do not occur.
        unit /= umod
        a = (unit * np.clip(f, -self.F, self.F)[:, :, None]) / self.m

        self.v *= self.u
        self.v += np.sum(a, axis=0)
        np.clip(self.v, -self.V, self.V, out=self.v)
        self.X += self.v

        self.l1.set_data(self.X[0::2, 0], self.X[0::2, 1])
        self.l2.set_data(self.X[1::2, 0], self.X[1::2, 1])
        return self.l1, self.l2


if __name__ == '__main__':

    fig, ax = plt.subplots()
    rs = RobotSwarm(64, lattice='hex')
    rs.setup_plot(fig, ax)
    anim = FuncAnimation(fig, rs, interval=50)
    plt.show()
