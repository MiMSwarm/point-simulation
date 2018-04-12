import numpy as np


class MiniMapper:

    """Models the MiniMapper in the simulation.

    Parameters
    ----------
    ident : hashable object, optional
        Used to identify the MiniMapper.
    env : environment.Environment
        The environment within which the object resides.
    """

    def __init__(self, env, ident=None):
        self.ident = ident
        self.environ = env

        self.position = np.array([0, 0])
        self.orientation = 0.0
        self.sonar_direction = 0.0

    def sense_environment(self):
        sensed = self.environ.estimate_sensor_readings(self.ident)

    def update_position(self):
        pass
