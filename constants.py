import numpy as np
import utils as ut


# The SONAR constants.
class SONAR:
    """Contains the constants required for SONAR."""
    ANGLE_RES = np.arange(-np.pi, np.pi, np.pi / 90)
    RANGE_RES = np.arange(.5, 4.0, .005)
    MAX_RANGE = RANGE_RES[1] - RANGE_RES[0] + RANGE_RES[-1]

    # Visual plane in Cartesian Coordinates.
    VIS_PLANE = ut.pol2cart(
        ut.cartesian_product((ANGLE_RES, RANGE_RES))[:, :, ::-1])


# The Infrared constants.
class INFRA:
    ANGLE_RES = np.arange(-np.pi, np.pi, np.pi/2)
    RANGE_RES = np.arange(0.005, 0.3, 0.005)
    MAX_RANGE = RANGE_RES[1] - RANGE_RES[0] + RANGE_RES[-1]
    ORIENT_ID = ['rear', 'right', 'front', 'left']
