from arcticpy.src import traps

from autoconf.dictable import Dictable


class TrapInstantCapture(traps.TrapInstantCapture, Dictable):
    def __init__(self, density=1.0, release_timescale=1.0):

        super().__init__(
            density=density,
            release_timescale=release_timescale,
            fractional_volume_none_exposed=0.0,
            fractional_volume_full_exposed=0.0,
        )
