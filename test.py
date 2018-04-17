import numpy as np
from environment import SONAR_VISUAL
from utils import CIRCLE_2DEG, SONAR_RANGE, pol2cart, cart2pol, \
    cartesian_product


def test_cartesian_circle():
    ppts = cartesian_product((CIRCLE_2DEG, SONAR_RANGE))[:, ::-1].reshape((
        CIRCLE_2DEG.shape[0], SONAR_RANGE.shape[0], 2))
    cpts = pol2cart(ppts)

    check = np.full(SONAR_VISUAL.shape[:-1], False)
    for i, w in enumerate(CIRCLE_2DEG):
        for j, r in enumerate(SONAR_RANGE):
            pt = pol2cart((r, w))
            check = cpts[i, j] == pt
    print(np.all(check))


def test_cart_pol_conv():
    print(pol2cart((2, np.pi/2)))
    print(cart2pol((5, 5)))


if __name__ == '__main__':
    test_cart_pol_conv()
