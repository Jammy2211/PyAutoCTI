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

from autocti.data import util


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
    def from_fits(cls, file_path, hdu, frame_geometry):
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
        return cls(frame_geometry=frame_geometry, array=util.numpy_array_from_fits(file_path=file_path, hdu=hdu))

    def output_as_fits(self, file_path, overwrite=False):
        """Output the image ci_data as a fits file.

        Params
        ----------
        path : str
            The output nlo path of the ci_data
        filename : str
            The file phase_name of the output image.
        """
        util.numpy_array_to_fits(array=self, file_path=file_path, overwrite=overwrite)


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
