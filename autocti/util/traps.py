from arcticpy import traps


class TrapInstantCapture(traps.TrapInstantCapture):
    """ For the old C++ style release-then-instant-capture algorithm. """

    def __init__(self, density=0.13, release_timescale=0.25):
        """ The parameters for a single trap species.
        Parameters
        ----------
        density : float
            The density of the trap species in a pixel.
        release_timescale : float
            The release timescale of the trap, in the same units as the time
            spent in each pixel or phase (Clocker sequence).
        surface : bool
            ###
        """
        super().__init__(
            density=density, release_timescale=release_timescale, surface=False
        )
