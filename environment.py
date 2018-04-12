import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from robot import MiniMapper
from utils import rounded_int, cart2pol, pol2cart


class ObstacleMap:
    """Map of the environment to simulate in. Modeled as a boolean matrix where
    True indicates an obstacle.

    Parameters
    ----------
    map_fn : callable
        A function that returns all those points that cannot be traversed. The
        resolution of the map should be 0.01.

    """

    def __init__(self, map_fn):
        self._obstacles = map_fn()
        xvalues = self._obstacles[:, 0]
        yvalues = self._obstacles[:, 1]

        self.xlimits = np.min(xvalues), np.max(xvalues)
        self.ylimits = np.min(yvalues), np.max(yvalues)

        self._internal_map = np.ndarray((
            rounded_int((self.xlimits[1] - self.xlimits[0]) * 100) + 1,
            rounded_int((self.ylimits[1] - self.ylimits[0]) * 100) + 1),
            dtype=bool)

        for x, y in self._obstacles:
            self[x, y] = True

    def __getitem__(self, key):
        if len(key) != 2:
            raise ValueError('Must be a 2-tuple (x, y).')
        if not (self.xlimits[0] < key[0] < self.xlimits[1]) or \
                not (self.ylimits[0] < key[1] < self.ylimits[1]):
            return False

        x = rounded_int((key[0] - self.xlimits[0]) * 100)
        y = rounded_int((key[1] - self.ylimits[0]) * 100)
        return self._internal_map[x, y]

    def __setitem__(self, key, value):
        if len(key) != 2:
            raise ValueError('Must be a 2-tuple (x, y).')

        x = rounded_int((key[0] - self.xlimits[0]) * 100)
        y = rounded_int((key[1] - self.ylimits[0]) * 100)
        self._internal_map[x, y] = value

    def __contains__(self, item):
        if len(item) != 2:
            return False
        return self[item]

    def plot(self, fig=None, ax=None, save=None):
        """Plot the map.

        Parameters
        ----------
        fig : matplotlib.figure.Figure or None, optional
            The figure to use while plotting.
        ax : matplotlib.axes.Axes or None, optional
            The axis to use while plotting.
        save : filename or None, optional
            If None, no image is saved. Otherwise, the image is saved to
            filename.
        """

        if fig is None:
            fig, ax = plt.subplots()
        elif ax is None:
            ax = fig.axes

        ax.set_xlim((self.xlimits[0]-2, self.xlimits[1]+2))
        ax.set_ylim((self.ylimits[0]-3, self.ylimits[1]+2))

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))

        ax.plot(self._obstacles[:, 0], self._obstacles[:, 1], 'ko', ms=.5)

        if save:
            plt.savefig(save)


class Environment:

    """The environment to simulate in. Holds any global data required for the
    simulation.

    Provides values for the sensors of the MiniMapper, tracks their actual
    positions and orientation and enables plotting of both the discovered
    environment and the actual environment.

    Parameters
    ----------
    map_fn : callable
        A function that returns all those points that cannot be traversed. The
        resolution of the map should be 0.01.
    nbot : int, optional
        Number of robots.
    person : bool, optional
        Whether to simulate a person through sound and temperature, or not.
    center : array-like, optional
        The Swarm is initialized around this point.
    radius : float, optional
        The max. distance from center a robot can be initialized.
    """

    def __init__(self, map_fn, nbot=20, person=False,
                 center=(0, 0), radius=0.5):
        # Initialize map.
        print('Constructing map ... ', end='', flush=True)
        self.map = ObstacleMap(map_fn)
        print('done.')

        # Initialize robots.
        print('Initializing robots ... ', end='', flush=True)

        self.robots = []
        center = np.array(center)
        for i in range(nbot):

            condition = True
            while condition:
                r = np.random.uniform(0, radius)
                w = np.random.uniform(-np.pi, np.pi)
                pos = np.round(center + pol2cart((r, w)), 2)
                condition = pos in self.map

            orient = np.random.uniform(-np.pi, np.pi)
            self.robots.append({
                'bot': MiniMapper(self, i),
                'initial_pos': np.copy(pos),
                'current_pos': np.copy(pos),
                'initial_ang': orient,
                'current_ang': orient,
            })

        print('done.')

        # Initialize person if required.
        self.person = None
        if person:
            print('Initializing person ... ', end='', flush=True)

            condition = True
            while condition:
                x = np.uniform.random(*self.map.xlimits)
                y = np.uniform.random(*self.map.ylimits)
                condition = (x, y) in self.map

            self.person = np.array((x, y))
            print('done.')

    def update(self):
        """Update the environment."""
        for mim in self.robots:
            mim['bot'].sense_environment()
            mim['bot'].update_position()

    def plot(self, fig=None, ax=None, save=None):
        """Plot the environment.

        Parameters
        ----------
        fig : matplotlib.figure.Figure or None, optional
            The figure to use while plotting.
        ax : matplotlib.axes.Axes or None, optional
            The axis to use while plotting.
        save : filename or None, optional
            If None, no image is saved. Otherwise, the image is saved to
            filename.
        """

        if fig is None:
            fig, ax = plt.subplots()
        elif ax is None:
            ax = fig.axes

        self.map.plot(fig, ax)
        points = np.array([mim['pos'] for mim in self.robots])
        ax.plot(points[:, 0], points[:, 1], 'ko', ms=2)
        plt.grid(which='major')

        if save:
            plt.savefig(save)

    def estimate_sensor_readings(self, mim_id):
        mim = self.robots[mim_id]
        neighbors = [bot['current_pos'] for bot in self.robots if bot != mim]

        pos = mim['current_pos']
        ang = mim['current_ang']

        angles = np.arange(-np.pi, np.pi, np.pi/90)
        radius = np.arange(0.05, 4.0, 0.01)

        # Estimate sonar readings.
        sonar = np.full(angles.shape, 4.0)

        # Check for obstacles first.
        for i, w in enumerate(angles):
            for r in radius:
                p = pos + pol2cart((r, w))
                if self.map[p[0], p[1]]:
                    sonar[i] = r
                    break

        # Check for other objects.
        for bpos in neighbors:
            p = cart2pol(bpos - pos)
            i = rounded_int((p[1] + np.pi) * 90 / np.pi) % 180
            if sonar[i] > p[0]:
                sonar[i] = p[0]

        shift = rounded_int((ang + np.pi) * 90 / np.pi)
        sonar = np.roll(sonar, shift)
        return {'sonar': sonar}

    def update_robot(self, mim_id, pos, ang):
        mim = self.robots[mim_id]
        mim['current_ang'] = mim['initial_ang'] + mim.orientation

        # Undo the rotation of orientation.
        pol_mim = cart2pol(mim.position)
        pol_mim[1] -= mim['initial_ang']

        # Translate the coordinates.
        rec_mim = pol2cart(pol_mim)
        mim['current_pos'] = mim['initial_pos'] + rec_mim
