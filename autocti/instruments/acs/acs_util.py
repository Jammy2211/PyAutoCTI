from astropy.io import fits
import copy
import logging
import numpy as np
import os

from autoarray import exc
from autoarray.layout import layout_util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


def fits_hdu_via_quadrant_letter_from(quadrant_letter):

    if quadrant_letter == "D" or quadrant_letter == "C":
        return 1
    elif quadrant_letter == "B" or quadrant_letter == "A":
        return 4
    else:
        raise exc.ArrayException("Quadrant letter for FrameACS must be A, B, C or D.")


def array_eps_to_counts(array_eps, bscale, bzero):

    if bscale is None:
        raise exc.ArrayException(
            "Cannot convert a Frame2D to units COUNTS without a bscale attribute (bscale = None)."
        )

    return (array_eps - bzero) / bscale


def prescan_fitted_bias_column(prescan, n_rows=2048, n_rows_ov=20):
    """
    Generate a bias column to be subtracted from the main image by doing a
    least squares fit to the serial prescan region.

    e.g. image -= prescan_fitted_bias_column(image[18:24])

    See Anton & Rorres (2013), S9.3, p460.

    Parameters
    ----------
    prescan : [[float]]
        The serial prescan part of the image. Should usually cover the full
        number of rows but may skip the first few columns of the prescan to
        avoid trails.

    n_rows
        The number of rows in the image, exculding overscan.

    n_rows_ov, int
        The number of overscan rows in the image.

    Returns
    -------
    bias_column : [float]
        The fitted bias to be subtracted from all image columns.
    """
    n_columns_fit = prescan.shape[1]

    # Flatten the multiple fitting columns to a long 1D array
    # y = [y_1_1, y_2_1, ..., y_nrow_1, y_1_2, y_2_2, ..., y_nrow_ncolfit]
    y = prescan[:-n_rows_ov].T.flatten()
    # x = [1, 2, ..., nrow, 1, ..., nrow, 1, ..., nrow, ...]
    x = np.tile(np.arange(n_rows), n_columns_fit)

    # M = [[1, 1, ..., 1], [x_1, x_2, ..., x_n]].T
    M = np.array([np.ones(n_rows * n_columns_fit), x]).T

    # Best-fit values for y = M v
    v = np.dot(np.linalg.inv(np.dot(M.T, M)), np.dot(M.T, y))

    # Map to full image size for easy subtraction
    bias_column = v[0] + v[1] * np.arange(n_rows + n_rows_ov)

    return np.transpose([bias_column])


def output_quadrants_to_fits(
    file_path: str,
    quadrant_a,
    quadrant_b,
    quadrant_c,
    quadrant_d,
    header_a=None,
    header_b=None,
    header_c=None,
    header_d=None,
    overwrite: bool = False,
):

    file_dir = os.path.split(file_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    array_hdu_1 = np.zeros((2068, 4144))
    array_hdu_4 = np.zeros((2068, 4144))

    def get_header(quadrant):
        try:
            return quadrant.header
        except AttributeError:
            raise (
                "You must pass in the header of the quadrants to output them to an ACS fits file."
            )

    header_a = get_header(quadrant_a) if header_a is None else header_a
    try:
        quadrant_a = copy.copy(np.asarray(quadrant_a.native))
    except AttributeError:
        quadrant_a = copy.copy(np.asarray(quadrant_a))

    quadrant_a = quadrant_convert_to_original(
        quadrant=quadrant_a, roe_corner=(1, 0), header=header_a, use_flipud=True
    )
    array_hdu_4[0:2068, 0:2072] = quadrant_a

    header_b = get_header(quadrant_b) if header_b is None else header_b

    try:
        quadrant_b = copy.copy(np.asarray(quadrant_b.native))
    except AttributeError:
        quadrant_b = copy.copy(np.asarray(quadrant_b))
    quadrant_b = quadrant_convert_to_original(
        quadrant=quadrant_b, roe_corner=(1, 1), header=header_b, use_flipud=True
    )
    array_hdu_4[0:2068, 2072:4144] = quadrant_b

    header_c = get_header(quadrant_c) if header_c is None else header_c
    try:
        quadrant_c = copy.copy(np.asarray(quadrant_c.native))
    except AttributeError:
        quadrant_c = copy.copy(np.asarray(quadrant_c))
    quadrant_c = quadrant_convert_to_original(
        quadrant=quadrant_c, roe_corner=(1, 0), header=header_c, use_flipud=False
    )
    array_hdu_1[0:2068, 0:2072] = quadrant_c

    header_d = get_header(quadrant_d) if header_d is None else header_d
    try:
        quadrant_d = copy.copy(np.asarray(quadrant_d.native))
    except AttributeError:
        quadrant_d = copy.copy(np.asarray(quadrant_d))
    quadrant_d = quadrant_convert_to_original(
        quadrant=quadrant_d, roe_corner=(1, 1), header=header_d, use_flipud=False
    )
    array_hdu_1[0:2068, 2072:4144] = quadrant_d

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_1))
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_4))
    hdu_list.append(fits.ImageHDU())

    def set_header(header):
        header.set("cticor", "ARCTIC", "CTI CORRECTION PERFORMED USING ARCTIC")
        return header

    hdu_list[0].header = set_header(header_a.header_sci_obj)
    hdu_list[1].header = set_header(header_c.header_hdu_obj)
    hdu_list[4].header = set_header(header_a.header_hdu_obj)
    hdu_list.writeto(file_path)


def quadrant_convert_to_original(
    quadrant, roe_corner, header, use_flipud=False, use_calibrated_gain=True
):

    if header.bias is not None:
        quadrant += header.bias.native

    if header.bias_serial_prescan_column is not None:
        quadrant += header.bias_serial_prescan_column

    quadrant = header.array_electrons_to_original(
        array=quadrant, use_calibrated_gain=use_calibrated_gain
    )

    if use_flipud:
        quadrant = np.flipud(quadrant)

    return layout_util.rotate_array_via_roe_corner_from(
        array=quadrant, roe_corner=roe_corner
    )
