from constants import SONAR
from environment import Environment

import matplotlib.pyplot as plt
import numpy as np
import utils as ut


def test_cartesian_circle():
    ppts = ut.cartesian_product((
        SONAR.ANGLE_RES, SONAR.RANGE_RES))[:, ::-1]
    cpts = ut.pol2cart(ppts)

    check = np.full(SONAR.VIS_PLANE.shape[:-1], False)
    for i, w in enumerate(SONAR.ANGLE_RES):
        for j, r in enumerate(SONAR.RANGE_RES):
            pt = ut.pol2cart((r, w))
            check = cpts[i, j] == pt
    print(np.all(check))


def test_cart_pol_conv():
    print(ut.pol2cart((2, np.pi/2)))
    print(ut.cart2pol((5, 5)))
