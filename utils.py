import matplotlib.pyplot as plt
import numpy as np


def rounded_int(x):
    """Round the value in x and typecase to int."""
    return int(np.round(x))


def cart2pol(p):
    """Convert cartesian coordinates to polar form."""
    r = np.sqrt(p[0]**2 + p[1]**2)
    w = np.arctan2(p[1], p[0])
    return np.array((r, w))


def pol2cart(p):
    """Convert polar coordinates to cartesian form."""
    x = p[0] * np.cos(p[1])
    y = p[0] * np.sin(p[1])
    return np.array((x, y))


def new_plot(*args, **kwargs):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(*args, **kwargs)
    return fig, axes
