import numpy as np
from scipy.ndimage import uniform_filter
from autocti.warm_pixels.lines import Line


def find_warm_pixels(
    image,
    trail_length=9,
    n_parallel_overscan=0,
    n_serial_prescan=0,
    ignore_bad_columns=True,
    bad_column_factor=3.5,
    bad_column_loops=5,
    smooth_width=3,
    unsharp_masking_factor=4,
    flux_min=0,
    name=None,
    date=None,
):
    """Find warm (and hot) pixels in an image.

    Parameters
    ----------
    image : [[float]]
        The input array of pixel values.

        The first dimension is the "row" index, the second is the "column"
        index. By default (for parallel clocking), charge is transfered "up"
        from row n to row 0 along each independent column. i.e. the readout
        register is above row 0.

    trail_length : int
        The number of pixels following the warm pixel to save as a trail. The
        same number of preceeding pixels are also included.

    n_parallel_overscan : int
        The number of rows in the overscan region of the input image. i.e. the
        final rows, furthest from the readout register, beyond the physical
        image. They should not contain warm pixels so will be ignored.

    n_serial_prescan : int
        The number of rows in the overscan region of the input image. i.e. the
        first columns, closest to the readout register, before the physical
        image. They should not contain warm pixels so will be ignored.

    ignore_bad_columns : bool
        Check for and ignore bad columns wiped out by extremely hot pixels.

    bad_column_factor
        Columns with a mean value more than this number of standard deviations
        above the overall median will be discarded, to avoid columns wiped out
        by extremely hot pixels.

    bad_column_loops : int
        The number of times to check for columns having means close enough to
        the overall median, updating the median each time.

    smooth_width : int
        The width of the window (in pixels) for calculating a smoothed image,
        used to find delta function-like warm pixels.

    unsharp_masking_factor
        Pixels must be this many times brighter than their neighbours in the
        smoothed image to be counted as warm pixels.

    flux_min
        Pixels below this value AFTER background subtraction will be ignored.
        Set None to not ignore any pixels. Defaults to 0 to ignore pixels below
        the background.

    name : str
        The name of the data used to identify it.

    date
        The Julian date for the image, for the Line objects' metadata.

    Returns
    -------
    warm_pixels : [Line]
        A list of the warm pixels and associated data as Line objects.
    """
    n_rows, n_columns = image.shape

    # Pixels flagged with 0 will be ignored
    where_not_ignored = np.ones_like(image)

    # List of not-ignored column indices, initially all
    good_columns = np.arange(n_columns)

    # Mean of each column
    column_means = np.mean(image, axis=0)

    # Identify bad columns to ignore, due to a really bright object or a pixel
    # hot enough to wipe everything above it, so have a high mean value
    if ignore_bad_columns:
        # Initialise to ignore all columns
        where_not_ignored *= 0

        # Remove columns with means far away from the median
        for i in range(1, bad_column_loops):
            median = np.median(column_means[good_columns])
            stddev = np.std(column_means[good_columns])
            # Keep columns with means close to the median
            good_columns = good_columns[
                abs(column_means[good_columns] - median) < bad_column_factor * stddev
            ]

        # Don't ignore the good columns
        where_not_ignored[:, good_columns] = 1

    # Subtract background
    background = 2.5 * np.median(column_means[good_columns]) - 1.5 * np.mean(
        column_means[good_columns]
    )
    image_no_bg = image - background

    # Don't ignore low-flux pixels if requested
    if flux_min is None:
        flux_min = np.nanmin(image_no_bg)

    # Unsharp mask image
    image_smooth = uniform_filter(image_no_bg, size=smooth_width)

    # Ignore the very top of the CCDPhase since we can't get full trails
    where_not_ignored[:trail_length, :] = 0
    # Ignore parallel overscan
    where_not_ignored[-(n_parallel_overscan + trail_length) :, :] = 0
    # Ignore serial prescan
    where_not_ignored[:, :n_serial_prescan] = 0

    # Calculate the maximum of the neighbouring pixels in the same column for
    # each pixel, not including that pixel
    nearby_maxima = np.maximum.reduce(
        [
            np.roll(image_no_bg, 1, axis=0),
            np.roll(image_no_bg, 2, axis=0),
            np.roll(image_no_bg, 3, axis=0),
            np.roll(image_no_bg, 4, axis=0),
            np.roll(image_no_bg, 5, axis=0),
            np.roll(image_no_bg, 6, axis=0),
            np.roll(image_no_bg, 7, axis=0),
            np.roll(image_no_bg, 8, axis=0),
            np.roll(image_no_bg, 9, axis=0),
            np.roll(image_no_bg, -1, axis=0),
            np.roll(image_no_bg, -2, axis=0),
            np.roll(image_no_bg, -3, axis=0),
            np.roll(image_no_bg, -4, axis=0),
            np.roll(image_no_bg, -5, axis=0),
            np.roll(image_no_bg, -6, axis=0),
            np.roll(image_no_bg, -7, axis=0),
            np.roll(image_no_bg, -8, axis=0),
            np.roll(image_no_bg, -9, axis=0),
        ]
    )

    # Identify warm pixels
    warm_pixel_locations = np.argwhere(
        # Not in ignored region_list
        (where_not_ignored.astype(bool))
        # Local maximum
        & (image_no_bg > nearby_maxima)
        & (image_no_bg > np.roll(image_no_bg, 1, axis=0))
        & (image_no_bg > np.roll(image_no_bg, -1, axis=0))
        # Still local maximum after unsharp masking
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, 1, axis=0))
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, -1, axis=0))
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, 1, axis=1))
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, -1, axis=1))
        # Above the minimum flux above background
        & (image_no_bg >= flux_min)
    )
    n_warm_pixels = len(warm_pixel_locations)

    if n_warm_pixels == 0:
        return []

    # Assemble the list of warm pixel data
    warm_pixels = []
    for location in warm_pixel_locations:
        row, column = location
        warm_pixels.append(
            Line(
                line=image[row - trail_length : row + trail_length, column],
                name=name,
                #                noise_map=
                location=[row, column],
                date=date,
                background=background,
            )
        )

    return warm_pixels
