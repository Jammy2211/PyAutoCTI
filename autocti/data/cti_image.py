#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: python/VIS_CTI_Image/CTIImage.py

Created on: 02/13/18
Author: James Nightingale
"""

import numpy as np

from autocti import exc
from autocti.model import pyarctic
from autocti.tools import imageio


class ImageFrame(np.ndarray):

    def __new__(cls, frame_geometry, array, **kwargs):
        """The CCD ci_frame of an image, including its geometry and therefore the directions parallel and serial CTI are \
        defined.

        Parameters
        ----------
        frame_geometry : CTIImage.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different regions of the CCD (overscans, prescan, etc.)
        array : ndarray
            The 2D array of the ci_data of this ci_frame.
        """
        quad = np.array(array, dtype='float64').view(cls)
        quad.frame_geometry = frame_geometry
        return quad

    # noinspection PyUnusedLocal
    def __init__(self, frame_geometry, array):
        """
        Params
        ----------
        frame_geometry : CTIImage.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different regions of the CCD (overscans, prescan, etc.)
        array : ndarray
            The 2D array of the ci_data of this ci_frame.
        """
        # noinspection PyArgumentList
        super(ImageFrame, self).__init__()
        self.frame_geometry = frame_geometry

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(ImageFrame, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(ImageFrame, self).__setstate__(state[0:-1])

    @classmethod
    def from_fits(cls, path, filename, hdu, frame_geometry):
        """Load the image ci_data from a fits file.

        Params
        ----------
        path : str
            The path to the ci_data
        filename : str
            The file phase_name of the fits image ci_data.
        hdu : int
            The HDU number in the fits file containing the image ci_data.
        frame_geometry : CTIImage.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different regions of the CCD (overscans, prescan, etc.)
        """
        return cls(frame_geometry=frame_geometry, array=imageio.numpy_array_from_fits(path, filename, hdu))

    def output_as_fits(self, path, filename):
        """Output the image ci_data as a fits file.

        Params
        ----------
        path : str
            The output nlo path of the ci_data
        filename : str
            The file phase_name of the output image.
        """
        imageio.numpy_array_to_fits(self, path, filename)


class CTIImage(ImageFrame):

    def __init__(self, frame_geometry, array):
        """The *CTIImage* class stores an image which cti can be added to or corrected from. It includes the ci_frame
        geometry and therefore direction of parallel / serial clocking.

        Params
        ----------
        frame_geometry : CTIImage.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different regions of the CCD (overscans, prescan, etc.)
        array : ndarray
            The 2D array of the ci_data of this ci_frame.
        """

        super(CTIImage, self).__init__(frame_geometry, array)

    def add_cti_to_image(self, cti_params, cti_settings):
        """Add cti to the image.

        Parameters
        ----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        return self.frame_geometry.add_cti(image=self, cti_params=cti_params, cti_settings=cti_settings)

    def correct_cti_from_image(self, cti_params, cti_settings):
        """Correct cti from the image.

        Parameters
        ----------
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """
        return self.frame_geometry.correct_cti(image=self, cti_params=cti_params, cti_settings=cti_settings)


class FrameGeometry(object):

    def __init__(self, parallel_overscan, serial_prescan, serial_overscan):
        """Abstract class for the geometry of a CTI Image.

        A CTIImage is stored as a 2D NumPy array. When this immage is passed to arctic, clocking goes towards \
        the 'top' of the NumPy array (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the array \
        (e.g. the final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input \
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions \
        defined in this class (and its children). These routines define how an image is rotated before parallel \
        and serial clocking and how to reorient the image back to its original orientation after clocking is performed.

        Currently, only four geometries are available, which are specific to Euclid (and documented in the \
        *QuadGeometryEuclid* class).

        Parameters
        -----------
        parallel_overscan : Region
            The parallel overscan region of the ci_frame.
        serial_prescan : Region
            The serial prescan region of the ci_frame.
        serial_overscan : Region
            The serial overscan region of the ci_frame.
        """

        self.parallel_overscan = parallel_overscan
        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan

    def add_cti(self, image, cti_params, cti_settings):
        """add cti to an image.

        Parameters
        ----------
        image : ndarray
            The image cti is added too.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. the ccd well_depth express option).
        """

        if cti_params.parallel_ccd is not None:
            image_pre_parallel_clocking = self.rotate_before_parallel_cti(image_pre_clocking=image)
            image_post_parallel_clocking = pyarctic.call_arctic(image_pre_parallel_clocking,
                                                                cti_params.parallel_species,
                                                                cti_params.parallel_ccd,
                                                                cti_settings.parallel)
            image = self.rotate_after_parallel_cti(image_post_parallel_clocking)

        if cti_params.serial_ccd is not None:
            image_pre_serial_clocking = self.rotate_before_serial_cti(image_pre_clocking=image)
            image_post_serial_clocking = pyarctic.call_arctic(image_pre_serial_clocking,
                                                              cti_params.serial_species,
                                                              cti_params.serial_ccd,
                                                              cti_settings.serial)
            image = self.rotate_after_serial_cti(image_post_serial_clocking)

        return image

    def correct_cti(self, image, cti_params, cti_settings):
        """Correct cti from an image.

        Parameters
        ----------
        image : ndarray
            The image cti is corrected from.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap lifetimes etc.).
        cti_settings : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        """

        if cti_settings.serial is not None:
            image_pre_serial_clocking = self.rotate_before_serial_cti(image_pre_clocking=image)
            image_post_serial_clocking = pyarctic.call_arctic(image_pre_serial_clocking,
                                                              cti_params.serial_species,
                                                              cti_params.serial_ccd,
                                                              cti_settings.serial,
                                                              correct_cti=True)
            image = self.rotate_after_serial_cti(image_post_serial_clocking)

        if cti_settings.parallel is not None:
            image_pre_parallel_clocking = self.rotate_before_parallel_cti(image_pre_clocking=image)
            image_post_parallel_clocking = pyarctic.call_arctic(image_pre_parallel_clocking,
                                                                cti_params.parallel_species,
                                                                cti_params.parallel_ccd,
                                                                cti_settings.parallel,
                                                                correct_cti=True)
            image = self.rotate_after_parallel_cti(image_post_parallel_clocking)

        return image

    @staticmethod
    def rotate_before_parallel_cti(image_pre_clocking):
        raise AssertionError("rotate_before_parallel_cti should be overridden")

    @staticmethod
    def rotate_before_serial_cti(image_pre_clocking):
        raise AssertionError("rotate_before_serial_cti should be overridden")

    @staticmethod
    def rotate_after_parallel_cti(image_post_clocking):
        raise AssertionError("rotate_after_parallel_cti should be overridden")

    @staticmethod
    def rotate_after_serial_cti(image_post_clocking):
        raise AssertionError("rotate_after_serial_cti should be overridden")


class QuadGeometryEuclid(FrameGeometry):

    def __init__(self, parallel_overscan, serial_prescan, serial_overscan):
        """Abstract class for the ci_frame geometry of Euclid quadrants. CTI uses a bias corrected raw VIS ci_frame, which \
         is  described at http://euclid.esac.esa.int/dm/dpdd/latest/le1dpd/dpcards/le1_visrawframe.html

        A CTIImage is stored as a 2D NumPy array. When an image is passed to arctic, clocking goes towards the 'top' \
        of the NumPy array (e.g. towards row 0). Trails therefore appear towards the 'bottom' of the array (e.g. the \
        final row).

        Arctic has no in-built functionality for changing the direction of clocking depending on the input \
        configuration file. Therefore, image rotations are handled before arctic is called, using the functions \
        defined in this class (and its children). These routines define how an image is rotated before parallel \
        and serial clocking with arctic. They also define how to reorient the image to its original orientation after \
        clocking with arctic is performed.

        A Euclid CCD is defined as below:

        ---KEY---
        ---------

        [] = read-out electronics

        [==========] = read-out register

        [xxxxxxxxxx]
        [xxxxxxxxxx] = CCD panel
        [xxxxxxxxxx]

        P = Parallel Direction
        S = Serial Direction

             <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /\  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /\        |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx 2 xxxxxxxxx] [xxxxxxxxx 3 xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the NumPy array)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx 0 xxxxxxxxx] [xxxxxxxxx 1 xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] \/         |
                                                                      \/
        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->

        Note that the arrow on the right defines the direction of clocking by arctic without any rotation. Therefore, \
        there are 8 circumstances of how arctic requires an image to be rotated before clocking:

        - Quadrant 0 - QuadGeometryEuclidBL  - Parallel Clocking - No rotation.
        - Quadrant 0 - QuadGeometryEuclidBL  - Serial Clocking   - Rotation 90 degrees clockwise.
        - Quadrant 1 - QuadGeometryEuclidBR - Parallel Clocking - No rotation.
        - Quadrant 1 - QuadGeometryEuclidBR - Serial Clocking   - Rotation 270 degrees clockwise.
        - Quadrant 2 - QuadGeometryEuclidTL     - Parallel Clocking - Rotation 180 degrees.
        - Quadrant 2 - QuadGeometryEuclidTL     - Serial Clocking   - Rotation 90 degrees clockwise.
        - Quadrant 3 - QuadGeometryEuclidTR    - Parallel Clocking - Rotation 180 degrees.
        - Quadrant 3 - QuadGeometryEuclidTR    - Serial Clocking   - Rotation 270 degrees clockwise

        After clocking has been performed with arctic (and CTI is added / corrected), it must be re-rotated back to \
        its original orientation. This rotation is the reverse of what is specified above.

        Rotations are performed using flipup / fliplr routines, but will ultimately use the Euclid Image Tools library.

        """
        super(QuadGeometryEuclid, self).__init__(parallel_overscan, serial_prescan, serial_overscan)

    @classmethod
    def from_ccd_and_quadrant_id(cls, ccd_id, quad_id):
        """Before reading this docstring, read the docstring for the __init__function above.

        In the Euclid FPA, the quadrant id ('E', 'F', 'G', 'H') depends on whether the CCD is located \
        on the left side (rows 1-3) or right side (rows 4-6) of the FPA:

        LEFT SIDE ROWS 1-2-3
        --------------------

         <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /\  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /\        |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx H xxxxxxxxx] [xxxxxxxxx G xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the NumPy array)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx E xxxxxxxxx] [xxxxxxxxx F xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] \/         |
                                                                      \/
        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->


        RIGHT SIDE ROWS 4-5-6
        ---------------------

         <--------S-----------   ---------S----------->
        [] [========= 2 =========] [========= 3 =========] []          |
        /\  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /\        |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
        P   [xxxxxxxxx F xxxxxxxxx] [xxxxxxxxx E xxxxxxxxx]  P         | clocks an image
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | (e.g. towards row 0
                                                                       | of the NumPy array)
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        P   [xxxxxxxxx G xxxxxxxxx] [xxxxxxxxx H xxxxxxxxx] P          |
        |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] |          |
        \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx] \/         |
                                                                      \/
        [] [========= 0 =========] [========= 1 =========] []
            <---------S----------   ----------S----------->

        Therefore, to setup a quadrant image with the correct frame_geometry using its CCD id (from which \
        we can extract its row number) and quadrant id, we need to first determine if the CCD is on the left / right \
        side and then use its quadrant id ('E', 'F', 'G' or 'H') to pick the correct quadrant.
        """

        row_index = ccd_id[-1]

        if (row_index in '123') and (quad_id == 'E'):
            return QuadGeometryEuclidBL()
        elif (row_index in '123') and (quad_id == 'F'):
            return QuadGeometryEuclidBR()
        elif (row_index in '123') and (quad_id == 'G'):
            return QuadGeometryEuclidTR()
        elif (row_index in '123') and (quad_id == 'H'):
            return QuadGeometryEuclidTL()
        elif (row_index in '456') and (quad_id == 'E'):
            return QuadGeometryEuclidTR()
        elif (row_index in '456') and (quad_id == 'F'):
            return QuadGeometryEuclidTL()
        elif (row_index in '456') and (quad_id == 'G'):
            return QuadGeometryEuclidBL()
        elif (row_index in '456') and (quad_id == 'H'):
            return QuadGeometryEuclidBR()


class QuadGeometryEuclidBL(QuadGeometryEuclid):

    def __init__(self):
        """This class represents the frame_geometry of a Euclid quadrant in the bottom-left of a CCD (see \
        **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""

        super(QuadGeometryEuclidBL, self).__init__(parallel_overscan=Region((2066, 2086, 51, 2099)),
                                                   serial_prescan=Region((0, 2086, 0, 51)),
                                                   serial_overscan=Region((0, 2086, 2099, 2119)))

    @staticmethod
    def rotate_before_parallel_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the parallel direction.

        For the bottom-left quadrant, no rotation is required for parallel clocking

        Params
        ----------
        image_pre_clocking : ndarray
            The image before parallel clocking, therefore before it has been reoriented for clocking.
        """
        return image_pre_clocking

    @staticmethod
    def rotate_after_parallel_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the parallel direction.

        For the bottom-left quadrant, no re-rotation is required for parallel clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with parallel cti added or corrected.

        """
        return image_post_clocking

    @staticmethod
    def rotate_before_serial_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the serial direction.

        For the bottom-left quadrant, the image is rotated 180 degrees for serial clocking

        NOTE : The NumPy transpose routine does not reorder the array's memory, making it non-contiguous. This is not \
        a useable ci_data-type for C++ (and therefore arctic), so we use .copy() to force a memory re-ordering.

        Params
        ----------
        image_pre_clocking : ndarray
            The image before serial clocking, therefore before it has been reoriented for clocking.
        """

        return image_pre_clocking.T.copy()

    @staticmethod
    def rotate_after_serial_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the serial direction.

        For the bottom-left quadrant, the image is re-rotated 180 degrees after serial clocking.


        NOTE : The NumPy transpose routine does not reorder the array's memory, making it non-contiguous. This is not \
        a useable ci_data-type for C++ (and therefore arctic), so we use .copy() to force a memory re-ordering.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with serial cti added or corrected.

        """
        return image_post_clocking.T.copy()

    @staticmethod
    def parallel_trail_from_y(y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return y, y + dy + 1

    @staticmethod
    def serial_trail_from_x(x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return x, x + dx + 1


class QuadGeometryEuclidBR(QuadGeometryEuclid):

    def __init__(self):
        """This class represents the frame_geometry of a Euclid quadrant in the bottom-right of a CCD (see \
        **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""

        super(QuadGeometryEuclidBR, self).__init__(parallel_overscan=Region((2066, 2086, 20, 2068)),
                                                   serial_prescan=Region((0, 2086, 2068, 2119)),
                                                   serial_overscan=Region((0, 2086, 0, 20)))

    @staticmethod
    def rotate_before_parallel_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the parallel direction.

        For the bottom-right quadrant, no rotation is required for parallel clocking

        Params
        ----------
        image_pre_clocking : ndarray
            The image before parallel clocking, therefore before it has been reoriented for clocking.
            """
        return image_pre_clocking

    @staticmethod
    def rotate_after_parallel_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the parallel direction.

        For the bottom-right quadrant, no re-rotation is required for parallel clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with parallel cti added or corrected.

        """
        return image_post_clocking

    @staticmethod
    def rotate_before_serial_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the serial direction.

        For the bottom-right quadrant, the image is rotated 180 degrees and then flipped across the central x-axis \
        before serial clocking

        NOTE : The NumPy transpose routine does not reorder the array's memory, making it non-contiguous. This is not \
        a useable ci_data-type for C++ (and therefore arctic), so we use .copy() to force a memory re-ordering.


        Params
        ----------
        image_pre_clocking : ndarray
            The image before serial clocking, therefore before it has been reoriented for clocking.
        """
        return image_pre_clocking.T.copy()[::-1, :]

    @staticmethod
    def rotate_after_serial_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the serial direction.

        For the bottom-right quadrant, the image is flipped back across the central-y axis and re-rotated 180 degrees \
        after serial clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with serial cti added or corrected.

        """
        return image_post_clocking[::-1, :].T.copy()

    @staticmethod
    def parallel_trail_from_y(y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return y, y + dy + 1

    @staticmethod
    def serial_trail_from_x(x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return x - dx, x + 1


class QuadGeometryEuclidTL(QuadGeometryEuclid):

    def __init__(self):
        """This class represents the frame_geometry of a Euclid quadrant in the top-left of a CCD (see \
        **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""

        super(QuadGeometryEuclidTL, self).__init__(parallel_overscan=Region((0, 20, 51, 2099)),
                                                   serial_prescan=Region((0, 2086, 0, 51)),
                                                   serial_overscan=Region((0, 2086, 2099, 2119)))

    @staticmethod
    def rotate_before_parallel_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the parallel direction.

        For the top-left quadrant, the image is rotated 180 degrees for parallel clocking


        Params
        ----------
        image_pre_clocking : ndarray
            The image before parallel clocking, therefore before it has been reoriented for clocking.
            """
        return image_pre_clocking[::-1, :]

    @staticmethod
    def rotate_after_parallel_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the parallel direction.

        For the top-left quadrant, the image is rerotated 180 degrees after parallel clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with parallel cti added or corrected.

        """
        return image_post_clocking[::-1, :]

    @staticmethod
    def rotate_before_serial_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the serial direction.

        For the top-left quadrant, the image is rotated 180 degrees before serial clocking

        NOTE : The NumPy transpose routine does not reorder the array's memory, making it non-contiguous. This is not \
        a useable ci_data-type for C++ (and therefore arctic), so we use .copy() to force a memory re-ordering.


        Params
        ----------
        image_pre_clocking : ndarray
            The image before serial clocking, therefore before it has been reoriented for clocking.
        """
        return image_pre_clocking.T.copy()

    @staticmethod
    def rotate_after_serial_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the serial direction.

        For the top-left quadrant, the image is re-rotated 180 degrees after serial clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with serial cti added or corrected.

        """
        return image_post_clocking.T.copy()

    @staticmethod
    def parallel_trail_from_y(y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return y - dy, y + 1

    @staticmethod
    def serial_trail_from_x(x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return x, x + dx + 1


class QuadGeometryEuclidTR(QuadGeometryEuclid):

    def __init__(self):
        """This class represents the frame_geometry of a Euclid quadrant in the top-right of a CCD (see \
        **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""

        super(QuadGeometryEuclidTR, self).__init__(parallel_overscan=Region((0, 20, 20, 2068)),
                                                   serial_prescan=Region((0, 2086, 2068, 2119)),
                                                   serial_overscan=Region((0, 2086, 0, 20)))

    @staticmethod
    def rotate_before_parallel_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the parallel direction.

        For the top-right quadrant, the image is rotated 180 degrees before parallel clocking


        Params
        ----------
        image_pre_clocking : ndarray
            The image before parallel clocking, therefore before it has been reoriented for clocking.
        """
        return image_pre_clocking[::-1, :]

    @staticmethod
    def rotate_after_parallel_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the parallel direction.

        For the top-right quadrant, the image is rerotated 180 degrees after parallel clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with parallel cti added or corrected.

        """
        return image_post_clocking[::-1, :]

    @staticmethod
    def rotate_before_serial_cti(image_pre_clocking):
        """ Rotate the quadrant image ci_data before clocking via arctic in the serial direction.

        For the top-right quadrant, the image is rotated 180 degrees and flipped across the x-axis before \
        serial clocking

        NOTE : The NumPy transpose routine does not reorder the array's memory, making it non-contiguous. This is not \
        a useable ci_data-type for C++ (and therefore arctic), so we use .copy() to force a memory re-ordering.

        Params
        ----------
        image_pre_clocking : ndarray
            The image before serial clocking, therefore before it has been reoriented for clocking.
        """
        return image_pre_clocking.T.copy()[::-1, :]

    @staticmethod
    def rotate_after_serial_cti(image_post_clocking):
        """ Re-rotate the quadrant image ci_data after clocking via arctic in the serial direction.

        For the top-right quadrant, the image is re-rotated 180 degrees and flipped back across the x-axis \
        after serial clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The image after clocking, therefore with serial cti added or corrected.

        """
        return image_post_clocking[::-1, :].T.copy()

    @staticmethod
    def parallel_trail_from_y(y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return y - dy, y + 1

    @staticmethod
    def serial_trail_from_x(x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return x - dx, x + 1


class Region(tuple):

    def __new__(cls, region):
        """Setup a region of an image, which could be where the parallel overscan, serial overscan, etc. are.

        This is defined as a tuple (y0, y1, x0, x1).

        Parameters
        -----------
        region : (int,)
            The coordinates on the image of the region (y0, y1, x0, y1).
        """

        if region[0] < 0 or region[1] < 0 or region[2] < 0 or region[3] < 0:
            raise exc.RegionException('A coordinate of the Region was specified as negative.')

        if region[0] >= region[1]:
            raise exc.RegionException('The first row in the Region was equal to or greater than the second row.')

        if region[2] >= region[3]:
            raise exc.RegionException('The first column in the Region was equal to greater than the second column.')

        region = super(Region, cls).__new__(cls, region)

        region.y0 = region[0]
        region.y1 = region[1]
        region.x0 = region[2]
        region.x1 = region[3]

        region.total_rows = region.y1 - region.y0
        region.total_columns = region.x1 - region.x0

        return region

    def extract_region_from_array(self, array):
        return array[self.y0:self.y1, self.x0:self.x1]

    def add_region_from_image_to_array(self, image, array):
        array[self.y0:self.y1, self.x0:self.x1] += image[self.y0:self.y1, self.x0:self.x1]
        return array

    def set_region_on_array_to_zeros(self, array):
        array[self.y0:self.y1, self.x0:self.x1] = 0.0
        return array


def check_dimensions_are_euclid(dimensions):
    if dimensions[0] != 2048 or dimensions[1] != 2066:
        raise ValueError('The shape of image input to CTIImage are not consistent with a euclid'
                         'quadrant [2048x2066]')
