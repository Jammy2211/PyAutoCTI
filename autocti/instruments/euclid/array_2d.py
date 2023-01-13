import numpy as np
from typing import Optional

from autoarray.structures.arrays.uniform_2d import Array2D

from autocti.instruments.euclid.header import HeaderEuclid

from autoarray.layout import layout_util


class Array2DEuclid(Array2D):
    """
    In the Euclid FPA, the quadrant id ('E', 'F', 'G', 'H') depends on whether the CCD is located
    on the left side (rows 1-3) or right side (rows 4-6) of the FPA:

    LEFT SIDE ROWS 1-2-3
    --------------------

     <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          |
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx H xxxxxxxxx] [xxxxxxxxx G xxxxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                   | of the ndarrays)
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    P   [xxxxxxxxx E xxxxxxxxx] [xxxxxxxxx F xxxxxxxxx] P          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

    [] [========= 0 =========] [========= 1 =========] []
        <---------S----------   ----------S----------->


    RIGHT SIDE ROWS 4-5-6
    ---------------------

     <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          |
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx F xxxxxxxxx] [xxxxxxxxx E xxxxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                   | of the ndarrays)
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
    P   [xxxxxxxxx G xxxxxxxxx] [xxxxxxxxx H xxxxxxxxx] P          |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]            |

    [] [========= 0 =========] [========= 1 =========] []
        <---------S----------   ----------S----------->

    Therefore, to setup a quadrant image with the correct layout using its CCD id (from which
    we can extract its row number) and quadrant id, we need to first determine if the CCD is on the left / right
    side and then use its quadrant id ('E', 'F', 'G' or 'H') to pick the correct quadrant.
    """

    @classmethod
    def from_fits_header(cls, array, ext_header):
        """
        Use an input array of a Euclid quadrant and its corresponding .fits file header to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        ccd_id = ext_header["CCDID"]
        quadrant_id = ext_header["QUADID"]

        return cls.from_ccd_and_quadrant_id(
            array=array, ccd_id=ccd_id, quadrant_id=quadrant_id
        )

    @classmethod
    def from_ccd_and_quadrant_id(cls, array, ccd_id, quadrant_id):
        """
        Use an input array of a Euclid quadrant, its ccd_id and quadrant_id  to rotate the quadrant to
        the correct orientation for arCTIc clocking.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        row_index = ccd_id[-1]

        if (row_index in "123") and (quadrant_id == "E"):
            return Array2DEuclid.bottom_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "123") and (quadrant_id == "F"):
            return Array2DEuclid.bottom_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "123") and (quadrant_id == "G"):
            return Array2DEuclid.top_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "123") and (quadrant_id == "H"):
            return Array2DEuclid.top_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "E"):
            return Array2DEuclid.top_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "F"):
            return Array2DEuclid.top_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "G"):
            return Array2DEuclid.bottom_left(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )
        elif (row_index in "456") and (quadrant_id == "H"):
            return Array2DEuclid.bottom_right(
                array_electrons=array, ccd_id=ccd_id, quadrant_id=quadrant_id
            )

    @classmethod
    def top_left(
        cls,
        array_electrons: np.ndarray,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ) -> "Array2DEuclid":
        """
        Load a Euclid quadrant where the read out electronics (ROE) are at the top left of the CCD.

        The array is loaded and then rotated to ensure that parallel and serial CTI clocking go in the correct
        direction for arctic clocking.

        ROE's in the `top_left` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id H.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id F.

        The data is rotated by a flip-up-down, such that the ROE electrons are in the bottom left corner after rotation.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        array_electrons
            The array of data (in units of electrons) which has the ROE in the top left and is rotated for arctic
            clocking.
        ccd_id
            The ID of the euclid CCD (1, 2, 3, 4, 5, or 6) indicating which half of the FP the CCD is in.
        quadrant_id
            The ID of the quadrant (E, G, F or H).
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(0, 0)
        )

        header = HeaderEuclid(
            original_roe_corner=(0, 0), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.no_mask(values=array_electrons, pixel_scales=0.1, header=header)

    @classmethod
    def top_right(
        cls,
        array_electrons: np.ndarray,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ) -> "Array2DEuclid":
        """
        Load a Euclid quadrant where the read out electronics (ROE) are at the top right of the CCD.

        The array is loaded and then rotated to ensure that parallel and serial CTI clocking go in the correct
        direction for arctic clocking.

        ROE's in the `top_right` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id G.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id E.

        The data is rotated by a flip-up-down and flip left-right, such that the ROE electrons are in the bottom left
        corner after rotation.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        array_electrons
            The array of data (in units of electrons) which has the ROE in the top right and is rotated for arctic
            clocking.
        ccd_id
            The ID of the euclid CCD (1, 2, 3, 4, 5, or 6) indicating which half of the FP the CCD is in.
        quadrant_id
            The ID of the quadrant (E, G, F or H).
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(0, 1)
        )

        header = HeaderEuclid(
            original_roe_corner=(0, 1), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.no_mask(values=array_electrons, pixel_scales=0.1, header=header)

    @classmethod
    def bottom_left(
        cls,
        array_electrons: np.ndarray,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ) -> "Array2DEuclid":
        """
        Load a Euclid quadrant where the read out electronics (ROE) are at the bottom left of the CCD.

        The array is loaded and then rotated to ensure that parallel and serial CTI clocking go in the correct
        direction for arctic clocking.

        ROE's in the `bottom left` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id E.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id G.

        No rotation is performed for arCTIc clocking because the code assume the ROE are at the bottom left.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        array_electrons
            The array of data (in units of electrons) which has the ROE in the bottom left and is rotated for arctic
            clocking.
        ccd_id
            The ID of the euclid CCD (1, 2, 3, 4, 5, or 6) indicating which half of the FP the CCD is in.
        quadrant_id
            The ID of the quadrant (E, G, F or H).
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 0)
        )

        header = HeaderEuclid(
            original_roe_corner=(1, 0), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.no_mask(values=array_electrons, pixel_scales=0.1, header=header)

    @classmethod
    def bottom_right(
        cls,
        array_electrons: np.ndarray,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ) -> "Array2DEuclid":
        """
        Load a Euclid quadrant where the read out electronics (ROE) are at the bottom right of the CCD.

        The array is loaded and then rotated to ensure that parallel and serial CTI clocking go in the correct
        direction for arctic clocking.

        ROE's in the `bottom right` correspond to 1/4 of the 144 quadrants on a Euclid FPA, specifically:

        - Those in the top half of the FPA (CCDID = 1, 2 or 3) and quadrant id F.
        - Those in the bottom half of the FPA (CCDID = 4, 5 or 6) and quadrant id H.

        The data is rotated by a flip left-right, such that the ROE electrons are in the bottom left corner after
        rotation.

        See the docstring of the `Array2DEuclid` class for a complete description of the Euclid FPA, quadrants and
        rotations.

        Parameters
        ----------
        array_electrons
            The array of data (in units of electrons) which has the ROE in the bottom right and is rotated for arctic
            clocking.
        ccd_id
            The ID of the euclid CCD (1, 2, 3, 4, 5, or 6) indicating which half of the FP the CCD is in.
        quadrant_id
            The ID of the quadrant (E, G, F or H).
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        header = HeaderEuclid(
            original_roe_corner=(1, 1), ccd_id=ccd_id, quadrant_id=quadrant_id
        )

        return cls.no_mask(values=array_electrons, pixel_scales=0.1, header=header)
