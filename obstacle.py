import numpy as np
import matplotlib.pyplot as plt
from utils import rounded_int


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

        self.xmin, self.xmax = np.min(xvalues), np.max(xvalues)
        self.ymin, self.ymax = np.min(yvalues), np.max(yvalues)

        self._internal_map = np.ndarray((
            rounded_int((self.xmax - self.xmin) * 100) + 1,
            rounded_int((self.ymax - self.ymin) * 100) + 1),
            dtype=bool)

        self[self._obstacles] = True

    def __getitem__(self, key):
        """Allows you to check whether a particular location within the map is
        set or not. The `key` can contain values beyond the range of input.
        Array-like objects are permitted as long as `key.shape[-1] == 2`.
        """
        pts = np.asarray(key)
        res = ((pts[..., 0] >= self.xmin) & (pts[..., 0] <= self.xmax) &
               (pts[..., 1] >= self.ymin) & (pts[..., 1] <= self.ymax))

        x = np.rint((pts[..., 0] - self.xmin) * 100).astype(int)
        y = np.rint((pts[..., 1] - self.ymin) * 100).astype(int)

        return res & self._internal_map[x, y]

    def __setitem__(self, key, value):
        """Allows you to set a particular location within the map as containing
        an obstacle. The `key` cannot contain values beyond the range of input.
        Array-like objects are permitted as long as `key.shape[-1] == 2`.
        """
        pts = np.asarray(key)
        res = np.ndarray(pts.shape[:-1], dtype=bool)

        res = ((pts[..., 0] >= self.xmin) & (pts[..., 0] <= self.xmax) &
               (pts[..., 1] >= self.ymin) & (pts[..., 1] <= self.ymax))

        if np.any(~res):
            raise ValueError('Some values not in range.')

        x = np.rint((pts[..., 0] - self.xmin) * 100).astype(int)
        y = np.rint((pts[..., 1] - self.ymin) * 100).astype(int)
        self._internal_map[x, y] = value

    def plot(self, fig=None, axes=None, save=None):
        """Plot the map.

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
            fig = plt.figure()
            axes = fig.add_subplot(111)
        elif axes is None:
            axes = fig.add_subplot(111)

        axes.plot(self._obstacles[:, 0], self._obstacles[:, 1], 'ko', ms=.5)
        if save:
            plt.savefig(save)
