import numpy as np
import autocti as ac


def test__array_2d_list_from():
    extract = ac.Extract2DSerialCalibration(shape_2d=(3, 5), region_list=[(0, 3, 0, 5)])

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 4.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    array_2d_list = extract.array_2d_list_from(array=array)

    assert (
        array_2d_list[0]
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

    array_2d_list = extract.array_2d_list_from(array=array)

    assert (array_2d_list[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
    assert (array_2d_list[1] == np.array([[0.0, 1.0, 2.0, 4.0, 4.0]])).all()


def test__mask_2d_from():
    extract = ac.Extract2DSerialCalibration(
        shape_2d=(5, 5), region_list=[(0, 2, 1, 4), (3, 5, 1, 4)]
    )

    mask = ac.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)

    mask[1, 1] = True
    mask[4, 3] = True

    mask_2d = extract.mask_2d_from(mask=mask, rows=(1, 2))

    assert (
        mask_2d
        == np.array(
            [[False, True, False, False, False], [False, False, False, True, False]]
        )
    ).all()


def test__array_2d_from():
    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5), region_list=[(0, 3, 1, 5)], serial_prescan=(0, 3, 0, 1)
    )

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ],
        pixel_scales=1.0,
    )

    array_2d = extract.array_2d_from(array=array, rows=(0, 3))

    assert (
        array_2d.native
        == np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
    ).all()

    assert array_2d.pixel_scales == (1.0, 1.0)

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5),
        region_list=[(0, 2, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    array_2d = extract.array_2d_from(array=array, rows=(0, 2))

    assert (
        array_2d.native
        == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
    ).all()

    assert array_2d.pixel_scales == (1.0, 1.0)

    array = ac.Array2D.no_mask(
        values=[
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

    array_2d = extract.array_2d_from(array=array, rows=(0, 1))

    assert (
        array_2d.native
        == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
    ).all()

    assert array_2d.pixel_scales == (1.0, 1.0)

    extract = ac.Extract2DSerialCalibration(
        shape_2d=(5, 5),
        region_list=[(0, 2, 1, 4), (3, 5, 1, 4)],
        serial_prescan=(0, 3, 0, 1),
        serial_overscan=(0, 3, 3, 4),
    )

    array = ac.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0, 2.0, 2.0],
            [0.0, 1.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 4.0, 4.0, 4.0],
            [0.0, 1.0, 5.0, 5.0, 5.0],
            [0.0, 1.0, 6.0, 6.0, 6.0],
        ],
        pixel_scales=1.0,
    )

    array_2d = extract.array_2d_from(array=array, rows=(1, 2))

    assert (
        array_2d.native
        == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
    ).all()

    assert array_2d.pixel_scales == (1.0, 1.0)


def test__extracted_layout_from():
    extract = ac.Extract2DSerialCalibration(
        shape_2d=(3, 5), region_list=[(0, 3, 1, 5)], serial_prescan=(0, 3, 0, 1)
    )

    layout = ac.Layout2DCI(
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

    layout = ac.Layout2DCI(
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

    layout = ac.Layout2DCI(
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


def test__imaging_ci_from(imaging_ci_7x7):
    # The ci layout spans 2 rows, so two rows are extracted

    extract = ac.Extract2DSerialCalibration(
        shape_2d=imaging_ci_7x7.shape_native,
        region_list=imaging_ci_7x7.layout.region_list,
        serial_prescan=imaging_ci_7x7.layout.serial_prescan,
        serial_overscan=imaging_ci_7x7.layout.serial_overscan,
    )

    dataset = extract.imaging_ci_from(dataset=imaging_ci_7x7, rows=(0, 2))

    assert (dataset.data.native == imaging_ci_7x7.data.native[0:2, :]).all()
    assert (dataset.noise_map.native == imaging_ci_7x7.noise_map.native[0:2, :]).all()
    assert (
        dataset.pre_cti_data.native == imaging_ci_7x7.pre_cti_data.native[0:2, :]
    ).all()
    assert (
        dataset.cosmic_ray_map.native == imaging_ci_7x7.cosmic_ray_map.native[1:3, :]
    ).all()

    assert dataset.layout.region_list == [(0, 2, 1, 5)]

    mask = ac.Mask2D.all_false(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )

    mask[2, 2] = True

    imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask)

    dataset = extract.imaging_ci_from(dataset=imaging_ci_7x7, rows=(0, 6))

    assert dataset.data.mask[2, 1] == False
    assert dataset.data.mask[1, 2] == True

    assert dataset.noise_map.mask[2, 1] == False
    assert dataset.noise_map.mask[1, 2] == True
