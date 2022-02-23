import numpy as np
import autocti as ac


def test__array_2d_from():

    extractor = ac.Extractor2DParallelCalibration(
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

    extracted_array = extractor.array_2d_from(array=array, columns=(0, 1))

    assert (extracted_array == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])).all()

    extractor = ac.Extractor2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 5, 0, 3)]
    )

    extracted_array = extractor.array_2d_from(array=array, columns=(1, 3))

    assert (
        extracted_array.native
        == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    ).all()


def test__mask_2d_from():

    extractor = ac.Extractor2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 5, 0, 3)]
    )

    mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 3), pixel_scales=1.0)

    mask[0, 1] = True

    extracted_mask = extractor.mask_2d_from(mask=mask, columns=(1, 3))

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


def test__extracted_extractor_2d_from():

    extractor = ac.Extractor2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 3, 0, 3)]
    )

    layout = ac.ci.Layout2DCI(
        shape_2d=extractor.shape_2d, region_list=extractor.region_list
    )

    extracted_extractor = extractor.extracted_layout_from(layout=layout, columns=(0, 1))

    assert extracted_extractor.region_list == [(0, 3, 0, 1)]

    extractor = ac.Extractor2DParallelCalibration(
        shape_2d=(5, 3), region_list=[(0, 5, 0, 3)]
    )

    layout = ac.ci.Layout2DCI(
        shape_2d=extractor.shape_2d, region_list=extractor.region_list
    )

    extracted_extractor = extractor.extracted_layout_from(layout=layout, columns=(1, 3))

    assert extracted_extractor.region_list == [(0, 5, 0, 2)]
