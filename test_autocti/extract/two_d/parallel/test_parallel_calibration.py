import numpy as np
import autocti as ac


def test__mask_2d_from():

    extract = ac.Extract2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 5, 0, 3)]
    )

    mask = ac.Mask2D.unmasked(shape_native=(5, 3), pixel_scales=1.0)

    mask[0, 1] = True

    extracted_mask = extract.mask_2d_from(mask=mask, columns=(1, 3))

    assert (
        extracted_mask
        == np.array(
            [
                [True, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
            ]
        )
    ).all()


def test__array_2d_from():

    extract = ac.Extract2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 3, 0, 3)]
    )

    array = ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        pixel_scales=1.0,
    )

    extracted_array = extract.array_2d_from(array=array, columns=(0, 1))

    assert (extracted_array == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])).all()

    extract = ac.Extract2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 5, 0, 3)]
    )

    mask = ac.Mask2D.unmasked(shape_native=(5, 3), pixel_scales=1.0)
    mask[0, 1] = True

    array = array.apply_mask(mask=mask)
    extracted_array = extract.array_2d_from(array=array, columns=(1, 3))
    print(extracted_array.mask)

    assert (
        extracted_array.native
        == np.array([[0.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    ).all()
    assert (
        extracted_array.mask
        == np.array(
            [
                [True, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
            ]
        )
    ).all()


def test__extracted_layout_from():

    extract = ac.Extract2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 3, 0, 3)]
    )

    layout = ac.Layout2DCI(shape_2d=extract.shape_2d, region_list=extract.region_list)

    extracted_extract = extract.extracted_layout_from(layout=layout, columns=(0, 1))

    assert extracted_extract.region_list == [(0, 3, 0, 1)]

    extract = ac.Extract2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 5, 0, 3)]
    )

    layout = ac.Layout2DCI(shape_2d=extract.shape_2d, region_list=extract.region_list)

    extracted_extract = extract.extracted_layout_from(layout=layout, columns=(1, 3))

    assert extracted_extract.region_list == [(0, 5, 0, 2)]


def test__imaging_ci_from(imaging_ci_7x7):

    # The ci layout starts at column 1, so the left most column is removed below

    extract = ac.Extract2DParallelCalibration(
        shape_2d=imaging_ci_7x7.shape_native,
        region_list=imaging_ci_7x7.layout.region_list,
    )

    imaging_ci_parallel_calibration = extract.imaging_ci_from(
        imaging_ci=imaging_ci_7x7, columns=(0, 6)
    )

    assert (
        imaging_ci_parallel_calibration.image.native
        == imaging_ci_7x7.image.native[:, 1:7]
    ).all()
    assert (
        imaging_ci_parallel_calibration.noise_map.native
        == imaging_ci_7x7.noise_map.native[:, 1:7]
    ).all()
    assert (
        imaging_ci_parallel_calibration.pre_cti_data.native
        == imaging_ci_7x7.pre_cti_data.native[:, 1:7]
    ).all()
    assert (
        imaging_ci_parallel_calibration.cosmic_ray_map.native
        == imaging_ci_7x7.cosmic_ray_map.native[:, 1:7]
    ).all()

    assert imaging_ci_parallel_calibration.layout.region_list == [(1, 5, 0, 4)]

    mask = ac.Mask2D.unmasked(
        shape_native=imaging_ci_7x7.shape_native, pixel_scales=1.0
    )

    mask[2, 2] = True

    imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask)

    imaging_ci_parallel_calibration = extract.imaging_ci_from(
        imaging_ci=imaging_ci_7x7, columns=(0, 6)
    )

    assert imaging_ci_parallel_calibration.image.mask[0, 1] == False
    assert imaging_ci_parallel_calibration.image.mask[2, 1] == True

    assert imaging_ci_parallel_calibration.noise_map.mask[0, 1] == False
    assert imaging_ci_parallel_calibration.noise_map.mask[2, 1] == True
