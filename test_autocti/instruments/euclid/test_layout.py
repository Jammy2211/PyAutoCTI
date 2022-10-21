import autocti as ac


def test__for_four_quadrants__loads_data_and_dimensions(euclid_data):

    layout = ac.euclid.Layout2DEuclid.top_left(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    )

    assert layout.original_roe_corner == (0, 0)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2066, 2086, 51, 2099)
    assert layout.serial_prescan == (0, 2086, 0, 51)
    assert layout.serial_overscan == (20, 2086, 2099, 2128)

    layout = ac.euclid.Layout2DEuclid.top_left(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=41,
        serial_overscan_size=10,
        parallel_overscan_size=15,
    )

    assert layout.original_roe_corner == (0, 0)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2071, 2086, 41, 2118)
    assert layout.serial_prescan == (0, 2086, 0, 41)
    assert layout.serial_overscan == (15, 2086, 2118, 2128)

    layout = ac.euclid.Layout2DEuclid.top_right(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    )

    assert layout.original_roe_corner == (0, 1)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2066, 2086, 51, 2099)
    assert layout.serial_prescan == (0, 2086, 0, 51)
    assert layout.serial_overscan == (20, 2086, 2099, 2128)

    layout = ac.euclid.Layout2DEuclid.top_right(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=41,
        serial_overscan_size=10,
        parallel_overscan_size=15,
    )

    assert layout.original_roe_corner == (0, 1)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2071, 2086, 41, 2118)
    assert layout.serial_prescan == (0, 2086, 0, 41)
    assert layout.serial_overscan == (15, 2086, 2118, 2128)

    layout = ac.euclid.Layout2DEuclid.bottom_left(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    )

    assert layout.original_roe_corner == (1, 0)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2066, 2086, 51, 2099)
    assert layout.serial_prescan == (0, 2086, 0, 51)
    assert layout.serial_overscan == (0, 2066, 2099, 2128)

    layout = ac.euclid.Layout2DEuclid.bottom_left(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=41,
        serial_overscan_size=10,
        parallel_overscan_size=15,
    )

    assert layout.original_roe_corner == (1, 0)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2071, 2086, 41, 2118)
    assert layout.serial_prescan == (0, 2086, 0, 41)
    assert layout.serial_overscan == (0, 2071, 2118, 2128)

    layout = ac.euclid.Layout2DEuclid.bottom_right(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=51,
        serial_overscan_size=29,
        parallel_overscan_size=20,
    )

    assert layout.original_roe_corner == (1, 1)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2066, 2086, 51, 2099)
    assert layout.serial_prescan == (0, 2086, 0, 51)
    assert layout.serial_overscan == (0, 2066, 2099, 2128)

    layout = ac.euclid.Layout2DEuclid.bottom_right(
        parallel_size=2086,
        serial_size=2128,
        serial_prescan_size=41,
        serial_overscan_size=10,
        parallel_overscan_size=15,
    )

    assert layout.original_roe_corner == (1, 1)
    assert layout.shape_2d == (2086, 2128)
    assert layout.parallel_overscan == (2071, 2086, 41, 2118)
    assert layout.serial_prescan == (0, 2086, 0, 41)
    assert layout.serial_overscan == (0, 2071, 2118, 2128)


def test__left_side__chooses_correct_layout_given_input(euclid_data):

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text1", quadrant_id="E"
    )

    assert layout.original_roe_corner == (1, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text2", quadrant_id="E"
    )

    assert layout.original_roe_corner == (1, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text3", quadrant_id="E"
    )

    assert layout.original_roe_corner == (1, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text1", quadrant_id="F"
    )

    assert layout.original_roe_corner == (1, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text2", quadrant_id="F"
    )

    assert layout.original_roe_corner == (1, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text3", quadrant_id="F"
    )

    assert layout.original_roe_corner == (1, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text1", quadrant_id="G"
    )

    assert layout.original_roe_corner == (0, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text2", quadrant_id="G"
    )

    assert layout.original_roe_corner == (0, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text3", quadrant_id="G"
    )

    assert layout.original_roe_corner == (0, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text1", quadrant_id="H"
    )

    assert layout.original_roe_corner == (0, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text2", quadrant_id="H"
    )

    assert layout.original_roe_corner == (0, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text3", quadrant_id="H"
    )

    assert layout.original_roe_corner == (0, 0)


def test__right_side__chooses_correct_layout_given_input(euclid_data):
    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text4", quadrant_id="E"
    )

    assert layout.original_roe_corner == (0, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text5", quadrant_id="E"
    )

    assert layout.original_roe_corner == (0, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text6", quadrant_id="E"
    )

    assert layout.original_roe_corner == (0, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text4", quadrant_id="F"
    )

    assert layout.original_roe_corner == (0, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text5", quadrant_id="F"
    )

    assert layout.original_roe_corner == (0, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text6", quadrant_id="F"
    )

    assert layout.original_roe_corner == (0, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text4", quadrant_id="G"
    )

    assert layout.original_roe_corner == (1, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text5", quadrant_id="G"
    )

    assert layout.original_roe_corner == (1, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text6", quadrant_id="G"
    )

    assert layout.original_roe_corner == (1, 0)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text4", quadrant_id="H"
    )

    assert layout.original_roe_corner == (1, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text5", quadrant_id="H"
    )

    assert layout.original_roe_corner == (1, 1)

    layout = ac.euclid.Layout2DEuclid.from_ccd_and_quadrant_id(
        ccd_id="text6", quadrant_id="H"
    )

    assert layout.original_roe_corner == (1, 1)
