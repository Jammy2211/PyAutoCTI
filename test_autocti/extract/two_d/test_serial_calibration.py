import numpy as np
import autocti as ac


def test__array_2d_list_from():

    extract = ac.Extract2DSerialCalibration(shape_2d=(3, 5), region_list=[(0, 3, 0, 5)])

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 4.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    serial_region = extract.array_2d_list_from(array=array)

    assert (
        serial_region[0]
        == np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 4.0, 4.0],
            ]
        )
    ).all()

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5), region_list=[(0, 1, 1, 4), (2, 3, 1, 4)]
    )

    serial_region = extract.array_2d_list_from(array=array)

    assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
    assert (serial_region[1] == np.array([[0.0, 1.0, 2.0, 4.0, 4.0]])).all()


def test__array_2d_from():

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5), region_list=[(0, 3, 1, 5)], serial_prescan=(0, 3, 0, 1)
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    new_array = extract.array_2d_from(array=array, rows=(0, 3))

    assert (
        new_array.native
        == np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5),
        region_list=[(0, 2, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    new_array = extract.array_2d_from(array=array, rows=(0, 2))

    assert (
        new_array.native
        == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 4.0, 4.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5),
        region_list=[(0, 1, 1, 3), (2, 3, 1, 3)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    new_array = extract.array_2d_from(array=array, rows=(0, 1))

    assert (
        new_array.native
        == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(5, 5),
        region_list=[(0, 2, 1, 4), (3, 5, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 4.0, 4.0, 4.0],
            [0.0, 1.0, 5.0, 5.0, 5.0],
            [0.0, 1.0, 6.0, 6.0, 6.0],
        ],
        pixel_scales=1.0,
    )

    new_array = extract.array_2d_from(array=array, rows=(1, 2))

    assert (
        new_array.native
        == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
    ).all()

    assert new_array.pixel_scales == (1.0, 1.0)


def test__mask_2d_from():

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(5, 5), region_list=[(0, 2, 1, 4), (3, 5, 1, 4)]
    )

    mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 5), pixel_scales=1.0)

    mask[1, 1] = True
    mask[4, 3] = True

    serial_frame = extract.mask_2d_from(mask=mask, rows=(1, 2))

    assert (
        serial_frame
        == np.array(
            [[False, True, False, False, False], [False, False, False, True, False]]
        )
    ).all()


def test__extracted_layout_from():

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5), region_list=[(0, 3, 1, 5)], serial_prescan=(0, 3, 0, 1)
    )

    layout = ac.ci.Layout2DCI(
        shape_2d=extract.shape_2d,
        region_list=extract.region_list,
        serial_prescan=extract.serial_prescan,
    )

    extracted_layout = extract.extracted_layout_from(
        layout=layout, new_shape_2d=(3, 5), rows=(0, 3)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 3, 1, 5)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 3, 0, 1)
    assert extracted_layout.serial_overscan == None

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5),
        region_list=[(0, 2, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    layout = ac.ci.Layout2DCI(
        shape_2d=extract.shape_2d,
        region_list=extract.region_list,
        serial_prescan=extract.serial_prescan,
        serial_overscan=extract.serial_overscan,
    )

    extracted_layout = extract.extracted_layout_from(
        layout=layout, new_shape_2d=(2, 5), rows=(0, 2)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 2, 1, 4)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 2, 0, 1)
    assert extracted_layout.serial_overscan == (0, 2, 3, 4)

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5),
        region_list=[(0, 1, 1, 3), (2, 3, 1, 3)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    layout = ac.ci.Layout2DCI(
        shape_2d=extract.shape_2d,
        region_list=extract.region_list,
        serial_prescan=extract.serial_prescan,
        serial_overscan=extract.serial_overscan,
    )

    extracted_layout = extract.extracted_layout_from(
        layout=layout, new_shape_2d=(2, 5), rows=(0, 1)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 1, 1, 3), (1, 2, 1, 3)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 2, 0, 1)
    assert extracted_layout.serial_overscan == (0, 2, 3, 4)

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(5, 5),
        region_list=[(0, 2, 1, 4), (3, 5, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    extracted_layout = extract.extracted_layout_from(
        layout=layout, new_shape_2d=(2, 5), rows=(1, 2)
    )

    assert extracted_layout.original_roe_corner == (1, 0)
    assert extracted_layout.region_list == [(0, 1, 1, 4), (1, 2, 1, 4)]
    assert extracted_layout.parallel_overscan == None
    assert extracted_layout.serial_prescan == (0, 2, 0, 1)
    assert extracted_layout.serial_overscan == (0, 2, 3, 4)
