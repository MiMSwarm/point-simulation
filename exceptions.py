class SimulationError(Exception):

    def __init__(self, ident):
        self.ident = ident

    def __str__(self):
        return 'Cannot move robot {0}.'.format(self.ident)
