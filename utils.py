import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
from robot import ANGLES_RANGE


def rounded_int(x):
    """Round the value in x and typecast to int."""
    return int(np.round(x))


def cart2pol(cpoints, out=None):
    """Convert cartesian coordinates to polar form.

    Parameters
    ----------
    cpoints : numpy.ndarray
        Array of point to convert. Size of last dimension should be 2.
    out : numpy.ndarray, optional
        Array to place the output in.

    Returns
    -------
    ppoints : numpy.ndarray
        Converted array of points. Reshaped to `cpoints.shape`.
    """
    cpoints = np.array(cpoints, copy=False)
    if out is None:
        out = np.empty(cpoints.shape)
    out[..., 0] = np.sqrt(cpoints[..., 0]**2 + cpoints[..., 1]**2)
    out[..., 1] = np.arctan2(cpoints[..., 1], cpoints[..., 0])
    return out


def pol2cart(ppoints, out=None):
    """Convert polar coordinates to cartesian form.

    Parameters
    ----------
    ppoints : numpy.ndarray
        Array of point to convert. Size of last dimension should be 2.
    out : numpy.ndarray, optional
        Array to place the output in.

    Returns
    -------
    cpoints : numpy.ndarray
        Converted array of points. Reshaped to `cpoints.shape`.
    """
    ppoints = np.array(ppoints, copy=False)
    if out is None:
        out = np.empty(ppoints.shape)
    out[..., 0] = ppoints[..., 0] * np.cos(ppoints[..., 1])
    out[..., 1] = ppoints[..., 0] * np.sin(ppoints[..., 1])
    return out


def new_plot(*args, **kwargs):
    """Create a new subplot to plot on.

    Parameters
    ----------
    *args and **kwargs are directly passed to `matplotlib.axes.Axes.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The newly created figure.
    ax : matplotlib.axes.Axes
        The axis returned by subplot.
    """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(*args, **kwargs)
    return fig, axes


def cartesian_product(arrays):
    """Function to generate the cartesian product of two or more arrays.

    Parameters
    ----------
    arrays : iterable
        List of arrays to generate the cartesian product of.

    Returns
    -------
    cartesia : numpy.ndarray
        The array containing the cartesian product of `arrays`.
    """
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *itt.accumulate(
        itt.chain((arr,), itt.repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *itt.repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)


def plot_sonar(sonar):
    """Function to plot the sonar visual.

    Parameters
    ----------
    sonar : array-like
        The sonar readings corresponding to `robot.ANGLES_RANGE`.
    """
    points = pol2cart(np.vstack((sonar, ANGLES_RANGE)).T)
    new_plot(points[:, 0], points[:, 1], 'ro', ms=2)
    plt.show()
