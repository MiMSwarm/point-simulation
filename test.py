import numpy as np
from robot import ANGLES_RANGE, RADIUS_RANGE
from environment import CARTESIAN_CIRCLE
from utils import pol2cart, cart2pol, cartesian_product


def test_cartesian_circle():
    ppts = cartesian_product((ANGLES_RANGE, RADIUS_RANGE))[:, ::-1].reshape((
        ANGLES_RANGE.shape[0], RADIUS_RANGE.shape[0], 2))
    cpts = pol2cart(ppts)

    check = np.full(CARTESIAN_CIRCLE.shape[:-1], False)
    for i, w in enumerate(ANGLES_RANGE):
        for j, r in enumerate(RADIUS_RANGE):
            pt = pol2cart((r, w))
            check = cpts[i, j] == pt
    print(np.all(check))


def test_cart_pol_conv():
    print(pol2cart((2, np.pi/2)))
    print(cart2pol((5, 5)))


if __name__ == '__main__':
    test_cart_pol_conv()
