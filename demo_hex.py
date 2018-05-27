import numpy as np
from scipy.spatial import distance

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Set a few numpy options.
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)


class RobotSwarm:

    def __init__(self, nbot, mass=.5, R_dist=1., F_max=.2, V_max=.2,
                 power=3, friction=.01):
        self.n = nbot
        self.m = mass
        self.R = R_dist
        self.F = F_max
        self.V = V_max

        self.p = power
        self.u = friction
        self.G = 0.15

        self.v = np.zeros((self.n, 2))
        self.X = np.random.uniform(-1, 1, (self.n, 2))

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

        f[r > self.R] *= -1                 # Attractive or Repulsive force.
        f[r > (1.1 * self.R)] = 0           # Can't see too far.

        # Estimating the direction of force.
        unit = self.X[None, :] - self.X[:, None]
        unit = np.nan_to_num(unit / np.linalg.norm(unit, axis=2)[:, :, None])
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
    rs = RobotSwarm(20)
    rs.setup_plot(fig, ax)
    anim = FuncAnimation(fig, rs, interval=50)
    plt.show()
