import logging

from autoarray.layout.layout import Layout2D
from autoarray.layout.region import Region2D

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


class Layout2DACS(Layout2D):
    @classmethod
    def from_sizes(cls, roe_corner, serial_prescan_size=24, parallel_overscan_size=20):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.
        """

        parallel_overscan = Region2D(
            (2068 - parallel_overscan_size, 2068, serial_prescan_size, 2072)
        )

        serial_prescan = Region2D((0, 2068, 0, serial_prescan_size))

        return Layout2D.rotated_from_roe_corner(
            roe_corner=roe_corner,
            shape_native=(2068, 2072),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
        )
