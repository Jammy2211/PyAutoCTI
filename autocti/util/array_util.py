import inspect
import os

import numpy as np
from astropy.io import fits
from functools import wraps


class Memoizer:
    def __init__(self):
        """
        Class to store the results of a function given a set of inputs.
        """
        self.results = {}
        self.calls = 0
        self.arg_names = None

    def __call__(self, func):
        """
        Memoize decorator. Any time a function is called that a memoizer has been attached to its results are stored in
        the results dictionary or retrieved from the dictionary if the function has aaready been called with those
        arguments.

        Note that the same memoizer persists over all instances of a class. Any state for a given instance that is not
        given in the representation of that instance will be ignored. That is, it is possible that the memoizer will
        give incorrect results if instance state does not affect __str__ but does affect the value returned by the
        memoized method.

        Parameters
        ----------
        func: function
            A function for which results should be memoized

        Returns
        -------
        decorated : function
            A function that memoizes results
        """
        if self.arg_names is not None:
            raise AssertionError("Instantiate a new Memoizer for each function")
        self.arg_names = inspect.getfullargspec(func).args

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = ", ".join(
                [
                    "('{}', {})".format(arg_name, arg)
                    for arg_name, arg in list(zip(self.arg_names, args))
                    + [(k, v) for k, v in kwargs.items()]
                ]
            )
            if key not in self.results:
                self.calls += 1
            self.results[key] = func(*args, **kwargs)
            return self.results[key]

        return wrapper


def numpy_array_1d_to_fits(array_1d, file_path, overwrite=False):
    """Write a 1D NumPy array to a .fits file.

    Before outputting a NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures \
    appear the same orientation as .fits files loaded in DS9.

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
    numpy_array_to_fits(array_2d=array_2d, file_path='/path/to/file/filename.fits', overwrite=True)
    """
    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array_1d, new_hdr)
    hdu.writeto(file_path)


def numpy_array_1d_from_fits(file_path, hdu):
    """Read a 2D NumPy array to a .fits file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures \
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


def numpy_array_2d_to_fits(array_2d, file_path, overwrite=False):
    """Write a 2D NumPy array to a .fits file.

    Before outputting a NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures \
    appear the same orientation as .fits files loaded in DS9.

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
    numpy_array_to_fits(array_2d=array_2d, file_path='/path/to/file/filename.fits', overwrite=True)
    """
    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(np.flipud(array_2d), new_hdr)
    hdu.writeto(file_path)


def numpy_array_2d_from_fits(file_path, hdu):
    """Read a 2D NumPy array to a .fits file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures \
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
    return np.flipud(np.array(hdu_list[hdu].data)).astype("float64")
