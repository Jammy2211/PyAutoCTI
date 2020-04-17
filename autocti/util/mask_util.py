def mask_2d_centres_from_shape_pixel_scale_and_centre(shape, pixel_scales, centre):
    """Determine the (y,x) arc-second central coordinates of a mask from its shape, pixel-scales and centre.

     The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
     ----------
    shape : (int, int)
        The (y,x) shape of the 2D array the arc-second centre is computed for.
    pixel_scales : (float, float)
        The (y,x) arc-second to pixel scales of the 2D array.
    centre : (float, flloat)
        The (y,x) centre of the 2D mask.

    Returns
    --------
    tuple (float, float)
        The (y,x) arc-second central coordinates of the input array.

    Examples
    --------
    centres_arcsec = centres_from_shape_pixel_scales_and_centre(shape=(5,5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    """
    y_centre_arcsec = (float(shape[0] - 1) / 2) - (centre[0] / pixel_scales[0])
    x_centre_arcsec = (float(shape[1] - 1) / 2) + (centre[1] / pixel_scales[1])

    return (y_centre_arcsec, x_centre_arcsec)


def total_pixels_from_mask_2d(mask_2d):
    """Compute the total number of unmasked pixels in a mask.

    Parameters
     ----------
    mask_2d : ndarray
        A 2D array of bools, where *False* values are unmasked and included when counting pixels.

    Returns
    --------
    int
        The total number of pixels that are unmasked.

    Examples
    --------

    mask = np.array([[True, False, True],
                 [False, False, False]
                 [True, False, True]])

    total_regular_pixels = total_regular_pixels_from_mask(mask=mask)
    """

    total_regular_pixels = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                total_regular_pixels += 1

    return total_regular_pixels
