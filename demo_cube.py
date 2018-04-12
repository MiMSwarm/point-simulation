import numpy as np
from scipy.spatial import distance

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Set a few numpy options.
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)


class RobotSwarm:

    def __init__(self, nbot, mass=1., R_dist=.5, F_max=.2, V_max=.1,
                 power=2, friction=.4):
        self.n = nbot
        self.m = mass
        self.R = R_dist
        self.F = F_max
        self.V = V_max

        self.p = power
        self.u = friction
        self.G = 0.18

        self.v = np.zeros((self.n, 2))
        self.X = np.random.uniform(-2, 2, (self.n, 2))

        self.b = np.ones(nbot)
        self.b[:(nbot // 2)] = -1
        np.random.shuffle(self.b)
        self._likes = self.b[:, None] @ self.b[:, None].T

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
        a = self.estimate_accel(r)

        self.v *= self.u
        self.v += np.sum(a, axis=0)
        np.clip(self.v, -self.V, self.V, out=self.v)
        self.X += self.v

        select = np.random.uniform(size=self.n) < .02
        if np.any(select):
            self.b[select] *= -1
            self._likes = self.b[:, None] @ self.b[:, None].T

        self.l1.set_data(self.X[self.b < 0, 0], self.X[self.b < 0, 1])
        self.l2.set_data(self.X[self.b > 0, 0], self.X[self.b > 0, 1])
        return self.l1, self.l2

    def estimate_accel(self, distances):
        # Estimating the magnitude of force.
        f = self.G * (self.m ** 2) / (distances ** self.p)
        f = distance.squareform(f)
        r = distance.squareform(distances)

        sel = self._likes > 0
        f[sel][r[sel] > (self.R * np.sqrt(2))] *= -1
        f[~sel][r[~sel] > self.R] *= -1

        f[r > (2. * self.R)] = 0

        # Estimating the direction of force.
        unit = self.X[None, :] - self.X[:, None]
        unit = np.nan_to_num(unit / np.linalg.norm(unit, axis=2)[:, :, None])
        return (unit * np.clip(f, -self.F, self.F)[:, :, None]) / self.m


if __name__ == '__main__':

    fig, ax = plt.subplots()
    rs = RobotSwarm(256)
    rs.setup_plot(fig, ax)
    anim = FuncAnimation(fig, rs, interval=50)
    plt.show()
