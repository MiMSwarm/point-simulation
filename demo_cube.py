#!/usr/bin/python3

import sys
import numpy as np
from scipy.spatial import distance

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Set a few numpy options.
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)


class RobotSwarm:

    def __init__(self, nbot, mass=1., R_dist=1., F_max=.2, V_max=.1,
                 power=2, friction=.05):
        self.n = nbot
        self.m = mass
        self.R = R_dist
        self.F = F_max
        self.V = V_max

        self.p = power
        self.u = friction

        self.G = 1.8 * self.F * (self.R ** self.p) / (2 * np.sqrt(2) + 2)

        self.v = np.zeros((self.n, 2))
        self.X = np.random.uniform(-1, 1, (self.n, 2))

        self.b = np.ones(nbot)
        self.b[:(nbot // 2)] = -1
        np.random.shuffle(self.b)
        self._likes = self.b[:, None] @ self.b[:, None].T

        self.ii = 0

    def setup_plot(self, fig, ax):
        self.fig = fig
        self.ax = ax

        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)

        self.l1, = self.ax.plot(
            self.X[self.b < 0, 0], self.X[self.b < 0, 1], 'ro', ms=2)
        self.l2, = self.ax.plot(
            self.X[self.b > 0, 0], self.X[self.b > 0, 1], 'bo', ms=2)

    def __call__(self, i):
        r = distance.pdist(self.X)

        # Estimating the magnitude of force.
        f = self.G * (self.m ** 2) / (r ** self.p)
        f = distance.squareform(f)
        r = distance.squareform(r)

        sel = self._likes > 0
        f[sel][r[sel] > (self.R * np.sqrt(2))] *= -1
        f[~sel][r[~sel] > self.R] *= -1

        f[r > (1.8 * self.R)] = 0

        # Estimating the direction of force.
        unit = self.X[None, :] - self.X[:, None]
        unit = np.nan_to_num(unit / np.linalg.norm(unit, axis=2)[:, :, None])
        a = (unit * np.clip(f, -self.F, self.F)[:, :, None]) / self.m

        self.v *= self.u
        self.v += np.sum(a, axis=0)
        np.clip(self.v, -self.V, self.V, out=self.v)
        self.X += self.v

        if self.ii % 10 == 0:
            self.update_colors(r)
        self.ii += 1

        self.l1.set_data(self.X[self.b < 0, 0], self.X[self.b < 0, 1])
        self.l2.set_data(self.X[self.b > 0, 0], self.X[self.b > 0, 1])
        return self.l1, self.l2

    def update_colors(self, distances):
        thresh = self.R
        nflips = (self.n // 10) or 1

        close = (l[r < thresh] for r, l in zip(distances, self._likes))
        count = np.array([np.sum(p) for p in close])

        random = np.random.uniform(size=distances.shape[0])
        select = np.argsort(random * count)[(self.n - nflips):]
        print(counts)

        for i in select:
            self.b[i] *= -1
            self._likes[i, :] *= -1
            self._likes[:, i] *= -1


if __name__ == '__main__':

    fig, ax = plt.subplots()
    rs = RobotSwarm(20)
    rs.setup_plot(fig, ax)
    anim = FuncAnimation(fig, rs, interval=50)
    plt.show()
