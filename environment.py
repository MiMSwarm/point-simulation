import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import sys

from constants import INFRA, SONAR
from exceptions import SimulationError
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
    center : array-like, optional
        The Swarm is initialized around this point.
    radius : float, optional
        The max. distance from center a robot can be initialized.
    plot : bool, optional
        Whether to plot the simulation or not.
    """

    def __init__(self, map_fn, nbot=20, center=(0, 0), radius=0.5, plot=True):
        # Buffer arrays
        self._buf_infpol = np.empty((
            INFRA.ANGLE_RES.shape[0], INFRA.RANGE_RES.shape[0], 2))
        self._buf_infcar = np.empty((self._buf_infpol.shape))

        # Initialize map.
        ut.stat_print('\tConstructing map ...')
        self.map = ObstacleMap(map_fn)
        print('done.')

        # Creating robots.
        ut.stat_print('\tCreating MiniMappers ...')

        center = np.array(center)
        self.robots = []
        positions = np.empty((nbot, 2))
        for i in range(nbot):

            condition = np.full(5, True)
            while np.any(condition):
                r = np.random.uniform(0, radius)
                w = np.random.uniform(-np.pi/2, np.pi/2)
                positions[i] = np.round(center + ut.pol2cart((r, w)), 2)

                # Ensures robots are not initialized within 20cm of the map
                condition[:] = (
                    (positions[i] + np.array([0, .2])) in self.map,
                    (positions[i] - np.array([0, .2])) in self.map,
                    (positions[i] + np.array([.2, 0])) in self.map,
                    (positions[i] - np.array([.2, 0])) in self.map, False)

                # Ensures robots are not initialized within 20cm of themselves.
                if i > 0:
                    condition[4] = np.any(np.linalg.norm(
                        positions[i] - positions[:i], axis=1) < .2)

            orient = np.random.uniform(-np.pi, np.pi)
            self.robots.append({
                'bot': MiniMapper(self, i),
                'initial_pos': np.copy(positions[i]),
                'current_pos': np.copy(positions[i]),
                'initial_ang': orient,
                'current_ang': orient,
            })

        print('done.')

        # Setting up the plot.
        if plot:
            ut.stat_print('\tSetting up the plot ...')
            self.fig = plt.figure()
            self.axes = self.fig.add_subplot(111)

            self.axes.set_xlim((self.map.xmin-2, self.map.xmax+2))
            self.axes.set_ylim((self.map.ymin-3, self.map.ymax+2))

            self.axes.xaxis.set_major_locator(MultipleLocator(1.0))
            self.axes.yaxis.set_major_locator(MultipleLocator(1.0))

            self.axes.grid(which='major')
            self.plot()
            print('done.')

        # For broadcasting current MiM.
        self._current = None

    def update(self):
        """Update the environment. Invokes the `MiniMapper.sense_environment`
        and `MiniMapper.update_position` for each robot. However, the order of
        robots for which these methods are invoked is random.
        """
        for bot in self.robots:
            self._current = bot['bot'].update(self._current)

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

    def estimate_percepts(self, mim_id):
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
                    The SONAR readings corresponding to constants defined in
                    `constants.SONAR`.
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
        angles = ang + iangle
        ut.shift_angles(angles)

        ut.cartesian_product((angles, irange), out=self._buf_infpol)
        ut.pol2cart(self._buf_infpol[:, :, ::-1], out=self._buf_infcar)

        infra = self.map[pos + self._buf_infcar]
        infra = np.apply_along_axis(
            lambda w: irange[np.argmax(w)] if np.any(w) else imaxim,
            axis=1, arr=infra)

        # Check for other objects.
        for bpos in neighbors:
            pt = ut.cart2pol(bpos - pos)
            ii = ut.degree_ind(pt[1])
            if sonar[ii] > pt[0]:                       # Update SONAR
                sonar[ii] = pt[0]

            check = np.isclose(pt[1], angles)
            if np.any(check):
                ind = np.argmax(check)
                if infra[ind] > pt[0]:                  # Update IR
                    infra[ind] = pt[0]

        return {
            'sonar': np.roll(
                sonar, (len(SONAR.ANGLE_RES)//2) - ut.degree_ind(ang)),
            'infra': {v: (infra[i] < .3)
                      for i, v in enumerate(INFRA.ORIENT_ID)}
        }

    def update_robot(self, mim_id, only_ang=False):
        """Called by the robot to register it's updated position with the
        environment. Updates its absolute position and orientation.
        """
        mim = self.robots[mim_id]

        # MiM maintains an orientation relative to initial orientation.
        mim['current_ang'] = mim['initial_ang'] + mim['bot'].orientation

        # Rotate and translate to change frame of reference.
        pos = np.copy(mim['bot'].position)
        cos = pos*np.cos(mim['initial_ang'])
        sin = pos*np.sin(mim['initial_ang'])
        pos[0] = cos[0] - sin[1]
        pos[1] = sin[0] + cos[1]
        mim['current_pos'] = mim['initial_pos'] + pos

    def __call__(self, ii=None):
        """Execute one iteration of the environment."""
        if ii is not None:
            print('\tIteration {:04}'.format(ii), end='\r')
        self.update()
        points = np.array([mim['current_pos'] for mim in self.robots])
        self.rsplot.set_data(points[:, 0], points[:, 1])
