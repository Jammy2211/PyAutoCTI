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
File: python/SHE_ArCTIC/ImageIO.py

Created on: 04/23/18
Author: James Nightingale
"""

import os

import numpy as np
from astropy.io import fits


def make_path_if_does_not_exist(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def numpy_array_2d_to_fits(array_2d, file_path, overwrite=False):
    """Write a 2D NumPy array to a .fits file.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is written to fits.
    file_path : str
        The full path of the file that is output, including the file name and '.fits' extension.
    overwrite : bool
        If True and a file already exists with the input file_path the .fits file is overwritten. If False, an error \
        will be raised.

    Returns
    -------
    None

    Examples
    --------
    array_2d = np.ones((5,5))
    numpy_array_to_fits(array=array_2d, file_path='/path/to/file/filename.fits', overwrite=True)
    """
    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array_2d, new_hdr)
    hdu.writeto(file_path)


def numpy_array_2d_from_fits(file_path, hdu):
    """Read a 2D NumPy array to a .fits file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the arrays \
    appear the same orientation as .fits files loaded in DS9.

    Parameters
    ----------
    file_path : str
        The full path of the file that is loaded, including the file name and '.fits' extension.
    hdu : int
        The HDU extension of the array that is loaded from the .fits file.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .fits file.

    Examples
    --------
    array_2d = numpy_array_from_fits(file_path='/path/to/file/filename.fits', hdu=0)
    """
    hdu_list = fits.open(file_path)
    return np.array(hdu_list[hdu].data)

def make_and_return_path(path, folder_names):
    """ For a given path, create a directory structure composed of a set of folders and return the path to the \
    inner-most folder.

    For example, if path='/path/to/folders', and folder_names=['folder1', 'folder2'], the directory created will be
    '/path/to/folders/folder1/folder2/' and the returned path will be '/path/to/folders/folder1/folder2/'.

    If the folders already exist, routine continues as normal.

    Parameters
    ----------
    path : str
        The path where the directories are created.
    folder_names : [str]
        The names of the folders which are created in the path directory.

    Returns
    -------
    path
        A string specifying the path to the inner-most folder created.

    Examples
    --------
    path = '/path/to/folders'
    path = make_and_return_path(path=path, folder_names=['folder1', 'folder2'].
    """
    for folder_name in folder_names:

        path += folder_name + '/'

        if not os.path.exists(path):
            os.makedirs(path)

    return path