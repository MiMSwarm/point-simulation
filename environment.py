import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from robot import MiniMapper, ANGLES_RANGE, RADIUS_RANGE
from utils import rounded_int, cart2pol, pol2cart, cartesian_product
from obstacle import ObstacleMap


# Constructing the cartesian circle relative to origin.
# The circle is split into segments each a 2° step more than the previous.
CARTESIAN_CIRCLE = pol2cart(
    cartesian_product((ANGLES_RANGE, RADIUS_RANGE))[:, ::-1].reshape((
        ANGLES_RANGE.shape[0], RADIUS_RANGE.shape[0], 2)))


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

        # Setting up the plot.
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)

        self.axes.set_xlim((self.map.xmin-2, self.map.xmax+2))
        self.axes.set_ylim((self.map.ymin-3, self.map.ymax+2))

        self.axes.xaxis.set_major_locator(MultipleLocator(1.0))
        self.axes.yaxis.set_major_locator(MultipleLocator(1.0))

        self.axes.grid(which='major')
        self.plot()

    def update(self):
        """Update the environment."""
        for mim in self.robots:
            mim['bot'].sense_environment()
            mim['bot'].update_position()

    def plot(self, fig=None, axes=None, save=None):
        """Plot the environment.

        Parameters
        ----------
        fig : matplotlib.figure.Figure or None, optional
            The figure to use while plotting.
        axes : matplotlib.axes.Axes or None, optional
            The axis to use while plotting.
        save : filename or None, optional
            If None, no image is saved. Otherwise, the image is saved to
            filename.
        """

        if fig is None:
            fig = self.fig
            axes = self.axes
        elif axes is None:
            axes = self.axes

        self.map.plot(fig, axes)
        points = np.array([mim['current_pos'] for mim in self.robots])
        self.rsplot, = axes.plot(points[:, 0], points[:, 1], 'ko', ms=2)

        if save:
            plt.savefig(save)

    def estimate_sensor_readings(self, mim_id):
        mim = self.robots[mim_id]
        neighbors = [bot['current_pos'] for bot in self.robots if bot != mim]

        pos = mim['current_pos']
        ang = mim['current_ang']

        # Estimate sonar readings.
        sonar = self.map[pos + CARTESIAN_CIRCLE]
        sonar = np.apply_along_axis(
            lambda w: 4.0 if not np.any(w) else RADIUS_RANGE[np.argmax(w)],
            axis=1, arr=sonar)

        # Check for other objects.
        for bpos in neighbors:
            p = cart2pol(bpos - pos)
            i = rounded_int((p[1] + np.pi) * 90 / np.pi) % 180
            if sonar[i] > p[0]:
                sonar[i] = p[0]

        shift = rounded_int((ang + np.pi) * 90 / np.pi)
        sonar = np.roll(sonar, shift)

        return {'sonar': sonar}

    def update_robot(self, mim_id):
        mim = self.robots[mim_id]
        mim['current_ang'] = mim['initial_ang'] + mim['bot'].orientation

        # Undo the rotation of orientation.
        pol_mim = cart2pol(mim['bot'].position)
        pol_mim[1] -= mim['initial_ang']

        # Translate the coordinates.
        rec_mim = pol2cart(pol_mim)
        mim['current_pos'] = mim['initial_pos'] + rec_mim

    def __call__(self, ii=None):
        if ii is not None:
            print('Iteration {:04}'.format(ii), end='\r')
        self.update()
        points = np.array([mim['current_pos'] for mim in self.robots])
        self.rsplot.set_data(points[:, 0], points[:, 1])
