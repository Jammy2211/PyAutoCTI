import numpy as np

import autocti as ac


def test__from_ccd__chooses_correct_array_given_quadrant_letter(acs_ccd):

    array = ac.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="B")

    assert array.shape_native == (2068, 2072)

    array = ac.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="C")

    assert array.shape_native == (2068, 2072)

    array = ac.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="A")

    assert array.shape_native == (2068, 2072)

    array = ac.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="D")

    assert array.shape_native == (2068, 2072)


def test__conversions_to_counts_and_counts_per_second_use_correct_values():

    header_sci_obj = {"EXPTIME": 1.0}
    header_hdu_obj = {"BSCALE": 1.0, "BZERO": 0.0}

    array = ac.Array2D.ones(
        shape_native=(3, 3),
        pixel_scales=1.0,
        header=ac.acs.HeaderACS(
            header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj
        ),
    )

    assert (array.in_counts.native == np.ones(shape=(3, 3))).all()
    assert (array.in_counts_per_second.native == np.ones(shape=(3, 3))).all()

    header_sci_obj = {"EXPTIME": 1.0}
    header_hdu_obj = {"BSCALE": 2.0, "BZERO": 0.0}

    array = ac.Array2D.ones(
        shape_native=(3, 3),
        pixel_scales=1.0,
        header=ac.acs.HeaderACS(
            header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj
        ),
    )

    assert (array.in_counts.native == 0.5 * np.ones(shape=(3, 3))).all()
    assert (array.in_counts_per_second.native == 0.5 * np.ones(shape=(3, 3))).all()

    header_sci_obj = {"EXPTIME": 1.0}
    header_hdu_obj = {"BSCALE": 2.0, "BZERO": 0.1}

    array = ac.Array2D.ones(
        shape_native=(3, 3),
        pixel_scales=1.0,
        header=ac.acs.HeaderACS(
            header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj
        ),
    )

    assert (array.in_counts.native == 0.45 * np.ones(shape=(3, 3))).all()
    assert (array.in_counts_per_second.native == 0.45 * np.ones(shape=(3, 3))).all()

    header_sci_obj = {"EXPTIME": 2.0}
    header_hdu_obj = {"BSCALE": 2.0, "BZERO": 0.1}

    array = ac.Array2D.ones(
        shape_native=(3, 3),
        pixel_scales=1.0,
        header=ac.acs.HeaderACS(
            header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj
        ),
    )

    assert (array.in_counts.native == 0.45 * np.ones(shape=(3, 3))).all()
    assert (array.in_counts_per_second.native == 0.225 * np.ones(shape=(3, 3))).all()
