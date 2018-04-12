from environment import Environment
import matplotlib.pyplot as plt
import numpy as np


def horizontal_line(xstart, xstop, yval):
    xstop += 0.01
    return np.vstack((
        np.arange(xstart, xstop, 0.01),
        np.full(int(round((xstop - xstart) / 0.01)), yval),
    ))


def vertical_line(xval, ystart, ystop):
    ystop += 0.01
    return np.vstack((
        np.full(int(round((ystop - ystart) / 0.01)), xval),
        np.arange(ystart, ystop, 0.01),
    ))


def simple_map():
    pts = [
        vertical_line(0, 0, 10),
        vertical_line(10, 0, 10),
        horizontal_line(0, 10, 10),
        horizontal_line(0, 4.5, 0),
        horizontal_line(5.5, 10, 0),

        vertical_line(4.5, 0, 2),
        vertical_line(4.5, 3, 5),
        horizontal_line(1.5, 4.5, 5),

        vertical_line(5.5, 0, 3),
        horizontal_line(5.5, 9, 3),
        vertical_line(9, 3, 9),
    ]
    return np.hstack(pts).T


if __name__ == '__main__':
    print('Initializing environment => ')
    env = Environment(simple_map, center=(3, -1), radius=1)
    print('Constructing plot ... ', end='', flush=True)
    env.plot()
    print('done.')
    plt.show()
