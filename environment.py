import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from constants import INFRA, SONAR
from robot import MiniMapper
from obstacle import ObstacleMap
import utils as ut


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
                 center=(0, 0), radius=0.5, plot=True):
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
                pos = np.round(center + ut.pol2cart((r, w)), 2)
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
        if plot:
            self.fig = plt.figure()
            self.axes = self.fig.add_subplot(111)

            self.axes.set_xlim((self.map.xmin-2, self.map.xmax+2))
            self.axes.set_ylim((self.map.ymin-3, self.map.ymax+2))

            self.axes.xaxis.set_major_locator(MultipleLocator(1.0))
            self.axes.yaxis.set_major_locator(MultipleLocator(1.0))

            self.axes.grid(which='major')
            self.plot()

    def update(self):
        """Update the environment. Invokes the `MiniMapper.sense_environment`
        and `MiniMapper.update_position` for each robot. However, the order of
        robots for which these methods are invoked is random.
        """
        order = np.arange(len(self.robots))
        np.random.shuffle(order)
        for i in order:
            self.robots[i]['bot'].sense_environment()
            self.robots[i]['bot'].update_position()

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
        """Estimate the SONAR and Infrared readings for the robot identified by
        the argument `mim_id`.

        Parameters
        ----------
        mim_id : int
            The ID of the MiniMapper object given to it in the constructor.

        Returns
        -------
        sensor : dict
            A dictionary of sensor readings,
                sonar : numpy.ndarray
                    The SONAR readings corresponding to `utils.CIRCLE_2DEG`.
                infra : dict
                    The Infrared readings for 'front', 'right', 'rear', 'left'.
        """
        mim = self.robots[mim_id]
        neighbors = [bot['current_pos'] for bot in self.robots if bot != mim]

        pos = mim['current_pos']
        ang = mim['current_ang']

        # Estimate sonar readings.
        srange = SONAR.RANGE_RES
        smaxim = SONAR.MAX_RANGE

        sonar = self.map[pos + SONAR.VIS_PLANE]
        sonar = np.apply_along_axis(
            lambda w: srange[np.argmax(w)] if np.any(w) else smaxim,
            axis=1, arr=sonar)

        # Estimate infrared readings.
        iangle = INFRA.ANGLE_RES
        irange = INFRA.RANGE_RES
        imaxim = INFRA.MAX_RANGE
        angles = np.array([ang + w for w in iangle])
        ut.shift_angles(angles)

        infra_visual = ut.pol2cart(
            ut.cartesian_product((angles, irange))[:, ::-1].reshape((
                iangle.shape[0], irange.shape[0], 2)))

        infra = self.map[pos + infra_visual]
        infra = np.apply_along_axis(
            lambda w: irange[np.argmax(w)] if np.any(w) else imaxim,
            axis=1, arr=infra)

        # Check for other objects.
        for bpos in neighbors:
            pt = ut.cart2pol(bpos - pos)
            ii = ut.rounded_int(np.degrees(pt[1])) // 2 + 90
            if sonar[ii] > pt[0]:       # Update SONAR
                sonar[ii] = pt[0]

            check = np.isclose(pt[1], angles)
            if np.any(check):
                ind = np.argmax(check)
                if infra[ind] > pt[0]:
                    infra[ind] = pt[0]

        return {
            'sonar': np.roll(sonar, ut.rounded_int((ang+np.pi) * 90 / np.pi)),
            'infra': {v: infra[i] for i, v in enumerate(INFRA.ORIENT_ID)}
        }

    def update_robot(self, mim_id):
        mim = self.robots[mim_id]
        mim['current_ang'] = mim['initial_ang'] + mim['bot'].orientation

        # Undo the rotation of orientation.
        pol_mim = ut.cart2pol(mim['bot'].position)
        pol_mim[1] -= mim['initial_ang']

        # Translate the coordinates.
        rec_mim = ut.pol2cart(pol_mim)
        mim['current_pos'] = mim['initial_pos'] + rec_mim

    def __call__(self, ii=None):
        if ii is not None:
            print('Iteration {:04}'.format(ii), end='\r')
        self.update()
        points = np.array([mim['current_pos'] for mim in self.robots])
        self.rsplot.set_data(points[:, 0], points[:, 1])
